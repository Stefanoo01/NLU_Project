from conll import evaluate
from sklearn.metrics import classification_report
from conll import evaluate
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import copy

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(train_loader, model, optimizer, criterion_slots, criterion_intents, device):
    """
    Single epoch training loop for BERT joint model.
    """
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Move inputs to device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        slot_labels    = batch["slot_labels"].to(device)          # [batch, max_len]
        intent_labels  = batch["intent_label"].view(-1).to(device)  # [batch]

        # Forward pass
        intent_logits, slot_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # intent_logits: [batch, num_intents]
        # slot_logits:   [batch, max_len, num_slots]

        # Compute slot loss (ignore -100)
        num_slots = slot_logits.size(-1)
        slot_loss = criterion_slots(
            slot_logits.view(-1, num_slots),
            slot_labels.view(-1)
        )

        # Compute intent loss
        intent_loss = criterion_intents(intent_logits, intent_labels)

        loss = slot_loss + intent_loss
        total_loss += loss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def eval_loop(data_loader, model, criterion_slots, criterion_intents, device, lang):
    """
    Batch‐based evaluation that returns:
      - eval_loss : average (intent + slot) loss over all batches
      - slot_f1   : token‐level F1 from conll.evaluate
      - intent_acc: accuracy from sklearn classification_report
    """
    model.eval()
    total_loss = 0.0
    nb_batches = 0

    all_intent_preds = []
    all_intent_labels = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            slot_labels    = batch["slot_labels"].to(device)      # [batch, max_len]
            intent_labels  = batch["intent_label"].view(-1).to(device)  # [batch]

            # Forward pass
            intent_logits, slot_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Compute losses
            loss_intent = criterion_intents(intent_logits, intent_labels)
            num_slots = slot_logits.size(-1)
            loss_slot = criterion_slots(
                slot_logits.view(-1, num_slots),
                slot_labels.view(-1)
            )
            loss = loss_intent + loss_slot

            total_loss += loss.item()
            nb_batches += 1

            # Intent predictions
            intent_preds = torch.argmax(intent_logits, dim=1).cpu().tolist()
            intent_labels_cpu = intent_labels.cpu().tolist()
            all_intent_preds.extend(intent_preds)
            all_intent_labels.extend(intent_labels_cpu)

            # Slot predictions
            slot_preds    = torch.argmax(slot_logits, dim=2)  # [batch, max_len]
            slot_pred_seqs = slot_preds.cpu().tolist()
            slot_gold_seqs = slot_labels.cpu().tolist()

            for i in range(len(slot_pred_seqs)):
                gold_seq = slot_gold_seqs[i]
                pred_seq = slot_pred_seqs[i]
                current_ref = []
                current_hyp = []
                for g_id, p_id in zip(gold_seq, pred_seq):
                    if g_id != -100:
                        gold_tag = lang.id2slot[g_id]
                        pred_tag = lang.id2slot[p_id]
                        current_ref.append(("w", gold_tag))
                        current_hyp.append(("w", pred_tag))
                ref_slots.append(current_ref)
                hyp_slots.append(current_hyp)

    # Average loss
    avg_loss = total_loss / nb_batches if nb_batches else 0.0

    # conll.evaluate → token‐level F1
    try:
        slot_results = evaluate(ref_slots, hyp_slots)
        slot_f1 = slot_results["total"]["f"]
    except Exception as ex:
        print("Warning in conll.evaluate:", ex)
        slot_f1 = 0.0

    # Intent accuracy
    report_intent = classification_report(
        [lang.id2intent[l] for l in all_intent_labels],
        [lang.id2intent[p] for p in all_intent_preds],
        zero_division=False,
        output_dict=True
    )
    intent_acc = report_intent.get("accuracy", 0.0)

    return {"eval_loss": avg_loss, "slot_f1": slot_f1, "intent_acc": intent_acc}


def train(model, config, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, optimizer, device, lang):
    """
    Full training routine with early stopping on Dev slot_f1.
    """
    model.train()
    best_dev_f1 = -1.0
    best_model_wts = None
    patience_ctr = config["patience"]

    sampled_epochs = []
    train_loss_hist = []
    dev_loss_hist = []
    dev_f1_hist = []
    dev_acc_hist = []

    epoch_bar = tqdm(range(1, config["n_epochs"] + 1), unit="epoch")
    for epoch in epoch_bar:
        sampled_epochs.append(epoch)
        # 1) Train one epoch
        train_loss = train_loop(train_loader, model, optimizer, criterion_slots, criterion_intents, device)
        train_loss_hist.append(train_loss)

        # 2) Evaluate on Dev
        dev_metrics = eval_loop(dev_loader, model, criterion_slots, criterion_intents, device, lang)
        dev_loss = dev_metrics["eval_loss"]
        dev_f1 = dev_metrics["slot_f1"]
        dev_acc = dev_metrics["intent_acc"]

        dev_loss_hist.append(dev_loss)
        dev_f1_hist.append(dev_f1)
        dev_acc_hist.append(dev_acc)

        epoch_bar.set_description(
            f"Epoch {epoch}/{config['n_epochs']} | Dev_F1: {dev_f1:.4f} | Dev_Acc: {dev_acc:.4f}"
        )

        # 3) Early stopping check based on Dev slot_f1
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model_wts = model.state_dict()
            best_epoch_for_this_run = epoch
            patience_ctr = config["patience"]
        else:
            patience_ctr -= 1

        if patience_ctr <= 0:
            break

    # 4) Load best weights for this run
    model.load_state_dict(best_model_wts)

    # 5) Evaluate on Test set
    test_metrics = eval_loop(test_loader, model, criterion_slots, criterion_intents, device, lang)
    test_f1 = test_metrics["slot_f1"]
    test_acc = test_metrics["intent_acc"]

    # 6) Build record for this single run
    best_run_record = {
        "train_loss": train_loss_hist,
        "dev_loss": dev_loss_hist,
        "dev_f1_per_epoch": dev_f1_hist,
        "dev_acc_per_epoch": dev_acc_hist,
        "best_epoch": best_epoch_for_this_run,
        "slot_f1_score": test_f1,
        "intent_accuracy": test_acc,
        "epochs": sampled_epochs
    }

    return model, best_run_record

def plot_loss_curves(history, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history: Dictionary containing training history with keys 'epochs', 'train_loss', and 'dev_loss'
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["epochs"], history["train_loss"], 'b-', label="Training Loss")
    plt.plot(history["epochs"], history["dev_loss"], 'r-', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")

def extract_report_data(results, output_path):
    """
    Extract essential information needed for writing a report
    
    Args:
        results: Dictionary containing training results and history
    
    Returns:
        report_data: Dictionary with essential information for report writing
    """
    # Extract information from results
    config = results["config"]
    history = results["history"]
    best_epoch = results["best_epoch"]
    
    # Calculate training improvement
    train_loss_reduction = ((history["train_loss"][0] - history["train_loss"][-1]) / 
                           history["train_loss"][0] * 100)
    
    # Compile essential information
    report_data = {
        # Model architecture information
        
        # Training parameters
        "batch_size": config['batch_size'],
        "learning_rate": config['lr'],
        "max_epochs": config['n_epochs'],
        "early_stopping_patience": config['patience'],
        
        # Training results
        "epochs_trained": len(history['epochs']),
        "best_epoch": best_epoch,
        "initial_train_loss": history['train_loss'][0],
        "final_train_loss": history['train_loss'][-1],
        "train_loss_reduction_percent": train_loss_reduction,
        
        # Evaluation metrics
        "slot_f1_score": history['slot_f1_score'],
        "intent_accuracy": history['intent_accuracy'],
        "slot_f1_scores:": history['dev_f1_per_epoch'],
        "intent_accuracies:": history['dev_acc_per_epoch'],

        # Epoch data for plotting
        "epochs": history['epochs'],
        "train_loss_history": history['train_loss'],
        "dev_loss_history": history['dev_loss'],
    }

    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report_data

def create_folder():
    base_dir = "results"
    # 1) Make sure "results" exists
    os.makedirs(base_dir, exist_ok=True)

    # 2) Find the next available "result_i" name
    i = 1
    while True:
        new_folder = os.path.join(base_dir, f"result_{i}")
        if not os.path.exists(new_folder):
            # 3) Create and return it
            os.makedirs(new_folder)
            return new_folder
        i += 1