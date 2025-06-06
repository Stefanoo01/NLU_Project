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

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def train(model, config, train_loader, dev_loader, test_loader,
          criterion_slots, criterion_intents, optimizer, lang):
    slot_f1s, intent_accs = [], []

    # Run multiple times 
    for _ in tqdm(range(config["runs"]), desc="Runs"):
        patience = config["patience"]
        best_f1 = 0
        best_model = copy.deepcopy(model).to()
        best_epoch = -1
        losses_train = []
        losses_dev = []
        sampled_epochs = []

        # Create the tqdm bar once—no `desc` here
        pbar = tqdm(range(1, config["n_epochs"] + 1), leave=False)
        f1 = 0.0  # initialize to something

        for epoch in pbar:
            # Update the description whenever you like:
            pbar.set_description(f"Epoch {epoch}/{config['n_epochs']} | Best F1: {best_f1:.4f}")

            # --- Training step ---
            loss = train_loop(train_loader, optimizer,
                              criterion_slots, criterion_intents, model)

            # --- Every 5 epochs, evaluate on dev set ---
            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())

                results_dev, intent_res, loss_dev = eval_loop(
                    dev_loader, criterion_slots, criterion_intents, model, lang
                )
                losses_dev.append(np.asarray(loss_dev).mean())

                f1 = results_dev["total"]["f"]

                # Early stopping logic
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to()
                    best_epoch = epoch
                    patience = config["patience"]
                else:
                    patience -= 1

                if patience <= 0:
                    break

        # Evaluate on test with the best model from this run
        best_model.to()
        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang
        )
        slot_f1s.append(results_test["total"]["f"])
        intent_accs.append(intent_test["accuracy"])

    return best_model, {
        "slot_f1_scores": np.asarray(slot_f1s),
        "intent_accuracies": np.asarray(intent_accs),
        "mean_slot_f1_score": np.asarray(slot_f1s).mean(),
        "std_slot_f1_score": np.asarray(slot_f1s).std(),
        "mean_intent_accuracy": np.asarray(intent_accs).mean(),
        "std_intent_accuracy": np.asarray(intent_accs).std(),
        "best_epoch": best_epoch,
        "train_loss": losses_train,
        "dev_loss": losses_dev,
        "epochs": sampled_epochs,
    }

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

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
        output_path: Path to save the extracted report data
    
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
        "embedding_size": config['emb_size'],
        "hidden_size": config['hid_size'],
        "num_layers": config['n_layers'],
        "dropout": config['dropout'],
        
        # Training parameters
        "learning_rate": config['lr'],
        "gradient_clip": config['clip'],
        "max_epochs": config['n_epochs'],
        "early_stopping_patience": config['patience'],
        
        # Training results
        "epochs_trained": len(history['epochs']),
        "best_epoch": best_epoch,
        "initial_train_loss": history['train_loss'][0],
        "final_train_loss": history['train_loss'][-1],
        "train_loss_reduction_percent": train_loss_reduction,
        
        # Evaluation metrics
        "mean_slot_f1_score": history['mean_slot_f1_score'],
        "std_slot_f1_score": history['std_slot_f1_score'],
        "mean_intent_accuracy": history['mean_intent_accuracy'],
        "std_intent_accuracy": history['std_intent_accuracy'],

        # Epoch data for plotting
        "runs": config['runs'],
        "epochs": history['epochs'],
        "train_loss_history": history['train_loss'],
        "dev_loss_history": history['dev_loss'],
    }

    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report_data

def create_folder():
    base_dir = "results"
    # Make sure "results" exists
    os.makedirs(base_dir, exist_ok=True)

    # Find the next available "result_i" name
    i = 1
    while True:
        new_folder = os.path.join(base_dir, f"result_{i}")
        if not os.path.exists(new_folder):
            # Create and return it
            os.makedirs(new_folder)
            return new_folder
        i += 1