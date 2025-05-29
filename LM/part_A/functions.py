import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

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

def get_optimizer(model, lr=0.001, weight_decay=0.01):
    # Replace SGD with AdamW
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
    
    plt.show()

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
    plt.annotate(f'Best PPL: {best_ppl:.2f}', 
                 xy=(best_epoch, best_ppl),
                 xytext=(best_epoch+1, best_ppl+1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Perplexity curve saved to {save_path}")
    
    plt.show()

#create a function that create a new folder
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