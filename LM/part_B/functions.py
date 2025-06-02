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

def get_optimizer(model, lr=0.1):
    """
    Returns an ASGD optimizer (Averaged SGD) with non-monotonic triggering.

    Args:
        model: the model to optimize
        lr: learning rate
        t0: averaging start trigger
        lambd: decay term
    """
    return optim.SGD(model.parameters(), lr=lr)

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
    class NT_AVSGD_Trigger:
        def __init__(self, non_monotone_n=5):
            self.logs = []
            self.t = 0
            self.T = 0
            self.non_monotone_n = non_monotone_n
            self.triggered = False

        def should_trigger(self, val_metric):
            if self.triggered:
                return False
            if self.t > self.non_monotone_n and val_metric > min(self.logs[-self.non_monotone_n:]):
                self.T = self.t
                self.triggered = True
                return True
            self.logs.append(val_metric)
            self.t += 1
            return False

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    ppls = []
    best_ppl = math.inf
    best_model = None
    best_epoch = -1
    patience = config["patience"]
    improvement_threshold = config["improvement_threshold"]
    trigger = NT_AVSGD_Trigger(non_monotone_n=5)

    pbar = tqdm(range(1, n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, config["clip"])
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            if isinstance(optimizer, optim.ASGD):
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                if hasattr(model, 'lstm') and isinstance(model.lstm, nn.LSTM):
                    model.lstm.flatten_parameters()
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                for prm in model.parameters():
                    prm.data = tmp[prm].clone()
                if hasattr(model, 'lstm') and isinstance(model.lstm, nn.LSTM):
                    model.lstm.flatten_parameters()
            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                if trigger.should_trigger(loss_dev):
                    print(f"NT-AvSGD TRIGGERED at epoch {epoch}")
                    optimizer = optim.ASGD(model.parameters(), lr=config['lr'], t0=0, lambd=0.)
                    if hasattr(model, 'lstm') and isinstance(model.lstm, nn.LSTM):
                        model.lstm.flatten_parameters()

            losses_dev.append(np.asarray(loss_dev).mean())
            ppls.append(ppl_dev)
            pbar.set_description("PPL: %f" % ppl_dev)

            improvement = best_ppl - ppl_dev

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to()
                best_epoch = epoch
                patience = config["patience"]
                if improvement < improvement_threshold:
                    patience -= 1
            else:
                patience -= 1
            if patience <= 0:
                break

    return best_model, {
        'epochs': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev,
        'dev_ppl': ppls,
        'best_epoch': best_epoch,
        'avsgd_trigger_epoch': trigger.T if trigger.triggered else None
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
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")

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
    Extract essential information needed for writing a report
    
    Args:
        results: Dictionary containing training results and history
        model_name: Name of the model (e.g., 'LSTM', 'RNN')
    
    Returns:
        report_data: Dictionary with essential information for report writing
    """
    # Extract information from results
    config = results["config"]
    history = results["history"]
    best_epoch = results["best_epoch"]
    best_dev_ppl = results["final_ppl"]
    
    # Calculate training improvement
    train_loss_reduction = ((history["train_loss"][0] - history["train_loss"][-1]) / 
                           history["train_loss"][0] * 100)
    
    # Compile essential information
    report_data = {
        # Model architecture information
        "embedding_size": config['emb_size'],
        "hidden_size": config['hid_size'],
        "num_layers": config['n_layers'],
        "emb_dropout": config['emb_dropout'],
        "out_dropout": config['out_dropout'],
        
        # Training parameters
        "learning_rate": config['lr'],
        "gamma": config['gamma'],
        "gradient_clip": config['clip'],
        "max_epochs": config['n_epochs'],
        "early_stopping_patience": config['patience'],
        "improvement_threshold": config["improvement_threshold"],
        "avsgd_trigger_epoch": history["avsgd_trigger_epoch"],
        
        # Training results
        "epochs_trained": len(history['epochs']),
        "best_epoch": best_epoch,
        "initial_train_loss": history['train_loss'][0],
        "final_train_loss": history['train_loss'][-1],
        "train_loss_reduction_percent": train_loss_reduction,
        
        # Evaluation metrics
        "best_validation_ppl": best_dev_ppl,
        
        # Epoch data for plotting
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
