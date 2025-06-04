import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import json
from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from main import DEVICE

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

def get_optimizer(model, type, lr=0.001, weight_decay=0.01):
    if type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def train(model, config, train_loader, dev_loader, n_epochs, criterion_train, criterion_eval, optimizer):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    ppls = []
    best_ppl = math.inf  # Initialize best perplexity as infinity
    best_model = None
    best_epoch = -1
    patience = config["patience"]  # Early stopping patience
    pbar = tqdm(range(1, n_epochs))  # Progress bar for epochs

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, config["clip"])

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            # Evaluate on dev set
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            ppls.append(ppl_dev)

            # Update progress bar
            pbar.set_description("PPL: %f" % ppl_dev)

            # Save model if it's the best so far
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to()  # Deep copy the model
                best_epoch = epoch
                patience = config["patience"]  # Reset patience
            else:
                patience -= 1  # Decrease patience if no improvement

            # Stop if patience is exhausted
            if patience <= 0:
                break

    return best_model, {
        'epochs': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev,
        'dev_ppl': ppls,
        'best_epoch': best_epoch,
    }

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

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
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")

def plot_perplexity(history, save_path=None):
    """
    Plot validation perplexity curve
    
    Args:
        history: Dictionary containing training history with keys 'epochs' and 'dev_ppl'
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["epochs"], history["dev_ppl"], 'g-', marker='o', label="Validation Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line at the best (minimum) perplexity
    best_ppl = min(history["dev_ppl"])
    best_epoch = history["epochs"][history["dev_ppl"].index(best_ppl)]
    plt.axhline(y=best_ppl, color='r', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Perplexity curve saved to {save_path}")

def extract_report_data(results, output_path):
    """
    Extract essential information needed for writing a report and save to a file.
    Args:
        results: Dictionary containing training results and history
        output_path: Path to the output file
    Returns:
        report_data: Dictionary with essential information for report writing
    """
    config = results["config"]
    history = results["history"]
    best_dev_ppl = results["final_ppl"]

    train_loss_reduction = ((history["train_loss"][0] - history["train_loss"][-1]) /
                            history["train_loss"][0] * 100)

    report_data = {
        "embedding_size": config['emb_size'],
        "hidden_size": config['hid_size'],
        "num_layers": config['n_layers'],
        "dropout": config['dropout'],
        "emb_dropout": config['emb_dropout'] if config['dropout'] else 'None',
        "out_dropout": config['out_dropout'] if config['dropout'] else 'None',
        "optimizer": config['optimizer'],
        "learning_rate": config['lr'],
        "weight_decay": config['weight_decay'] if config['optimizer'] == 'AdamW' else 'None', 
        "max_epochs": config['n_epochs'],
        "early_stopping_patience": config['patience'],
        "clip": config['clip'],
        "epochs_trained": len(history['epochs']),
        "best_epoch": history['best_epoch'],
        "initial_train_loss": history['train_loss'][0],
        "final_train_loss": history['train_loss'][-1],
        "train_loss_reduction_percent": train_loss_reduction,
        "best_validation_ppl": best_dev_ppl,
        "epochs": history['epochs'],
        "train_loss_history": history['train_loss'],
        "dev_loss_history": history['dev_loss'],
        "dev_ppl_history": history['dev_ppl']
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