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
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        slot_labels    = batch["slot_labels"].to(device)         # [batch, max_len]
        intent_labels  = batch["intent_label"].view(-1).to(device)  # [batch]

        intent_logits, slot_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Slot loss
        num_slots = slot_logits.size(-1)
        slot_loss = criterion_slots(
            slot_logits.view(-1, num_slots),
            slot_labels.view(-1)
        )

        # Intent loss
        intent_loss = criterion_intents(intent_logits, intent_labels)

        loss = slot_loss + intent_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def eval_loop(eval_loader, model, criterion_slots, criterion_intents, device):
    """
    Evaluation loop for a single split.
    Returns:
      - slot_f1 (micro over all valid sub-token positions),
      - intent_acc,
      - average loss,
      - a detailed intent classification report (dict).
    """
    model.eval()
    total_loss = 0.0

    all_intent_preds = []
    all_intent_trues = []
    all_slot_preds   = []
    all_slot_trues   = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            slot_labels    = batch["slot_labels"].to(device)       # [batch, max_len]
            intent_labels  = batch["intent_label"].view(-1).to(device)  # [batch]

            intent_logits, slot_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            num_slots = slot_logits.size(-1)
            slot_loss = criterion_slots(
                slot_logits.view(-1, num_slots),
                slot_labels.view(-1)
            )
            intent_loss = criterion_intents(intent_logits, intent_labels)
            loss = slot_loss + intent_loss
            total_loss += loss.item()

            # Intent predictions/trues
            intent_preds = torch.argmax(intent_logits, dim=1).cpu().tolist()
            intent_trues = intent_labels.cpu().tolist()
            all_intent_preds.extend(intent_preds)
            all_intent_trues.extend(intent_trues)

            # Slot predictions/trues
            slot_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy()  # [batch, max_len]
            slot_trues = slot_labels.cpu().numpy()                         # [batch, max_len]

            batch_size, seq_len = slot_trues.shape
            for i in range(batch_size):
                for j in range(seq_len):
                    true_label = slot_trues[i, j]
                    if true_label != -100:  # ignore padded / subtokens
                        all_slot_trues.append(true_label)
                        all_slot_preds.append(int(slot_preds[i, j]))

    intent_acc = accuracy_score(all_intent_trues, all_intent_preds)
    slot_f1    = f1_score(all_slot_trues, all_slot_preds, average="micro")
    avg_loss   = total_loss / len(eval_loader)

    return {
        "slot_f1": slot_f1,
        "intent_acc": intent_acc,
        "eval_loss": avg_loss,
        # we return a full classification report if you need it later
        "intent_report": classification_report(
            all_intent_trues, all_intent_preds, zero_division=False, output_dict=True
        )
    }


def train(model, config, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, optimizer, device):
    """
    Full training routine with early stopping on Dev slot_f1.
    Tracks per-epoch Dev slot_f1 & Dev intent_acc for the *best* run, 
    and final Test slot_f1 & Test intent_acc. Returns:
      - best_model           : model state after early stopping on Dev
      - record_dict          : dictionary containing
          * "dev_f1_per_epoch":  list of Dev slot_f1 per epoch (for best run)
          * "dev_acc_per_epoch": list of Dev intent_acc per epoch (for best run)
          * "test_f1"          : Test slot_f1 (for best run)
          * "test_acc"         : Test intent_acc (for best run)
          * "train_loss_history": list of training loss per epoch (for best run)
          * "dev_loss_history":   list of Dev loss per epoch (for best run)
          * "best_epoch"       : epoch number at which Dev slot_f1 peaked
    """
    # We’ll keep track of “best run” metrics here
    best_run_record = None
    best_overall_dev_f1 = -1.0

    for run in range(config["runs"]):
        model.train()
        best_dev_f1    = -1.0
        best_model_wts = None
        patience_ctr   = config["patience"]

        train_loss_hist = []
        dev_loss_hist   = []
        dev_f1_hist     = []
        dev_acc_hist    = []

        for epoch in range(1, config["n_epochs"] + 1):
            # 1) Train one epoch
            train_loss = train_loop(train_loader, model, optimizer, criterion_slots, criterion_intents, device)
            train_loss_hist.append(train_loss)

            # 2) Evaluate on Dev
            dev_metrics = eval_loop(dev_loader, model, criterion_slots, criterion_intents, device)
            dev_loss     = dev_metrics["eval_loss"]
            dev_f1       = dev_metrics["slot_f1"]
            dev_acc      = dev_metrics["intent_acc"]

            dev_loss_hist.append(dev_loss)
            dev_f1_hist.append(dev_f1)
            dev_acc_hist.append(dev_acc)

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
        test_metrics = eval_loop(test_loader, model, criterion_slots, criterion_intents, device)
        test_f1  = test_metrics["slot_f1"]
        test_acc = test_metrics["intent_acc"]

        # 6) If this run’s Dev-F1 is the highest across all runs, save its record
        if best_dev_f1 > best_overall_dev_f1:
            best_overall_dev_f1 = best_dev_f1
            best_run_record = {
                "train_loss_history": train_loss_hist,
                "dev_loss_history":   dev_loss_hist,
                "dev_f1_per_epoch":   dev_f1_hist,
                "dev_acc_per_epoch":  dev_acc_hist,
                "best_epoch":         best_epoch_for_this_run,
                "test_f1":            test_f1,
                "test_acc":           test_acc
            }

    # 7) Return the model (with best weights) and the record of interest
    # Re-load the best run’s weights (so model is consistent)
    model.load_state_dict(best_model_wts)
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
        "dropout": config['dropout'],
        
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
        "slot_f1_scores:": history['intent_f1_scores'],
        "intent_accuracies:": history['intent_accuracies'],

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