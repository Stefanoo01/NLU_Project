from functions import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from functools import partial
import os

TEST_MODEL = True
TEST_MODEL_NAME = "model.pt" # Set the name of the model in the bin folder to test or save
SAVE_MODEL = False
RESULTS = False
DEVICE = "cuda:0"

config = {
    # Optimizer
    'optimizer': 'AdamW',  # "SGD" or "AdamW"
    'lr': 1e-4,
    'weight_decay': 1e-6,  # Used for AdamW

    # Model architecture
    'emb_size': 650,       
    'hid_size': 650,   

    # Dropout
    'dropout': False,
    'emb_dropout': 0.2,
    'out_dropout': 0.3,     

    # Training control
    'clip': 0.25,          
    'n_epochs': 100,       
    'patience': 5,         # Early stop if dev PPL doesn’t improve for 5 epochs
}

if __name__ == "__main__":
    # Set paths
    path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(path, "dataset/PennTreeBank")

    # Load dataset splits
    train_raw = read_file(os.path.join(dataset_path, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(dataset_path, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(dataset_path, "ptb.test.txt"))

    # Build vocabulary
    lang = Lang(train_raw, special_tokens=["<pad>", "<unk>"])

    # Create datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Model setup
    vocab_len = len(lang.word2id)
    model = LM_LSTM(config["emb_size"], config["hid_size"], config["dropout"], vocab_len, pad_index=lang.word2id["<pad>"], emb_dropout=config["emb_dropout"], out_dropout=config["out_dropout"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = get_optimizer(model, config["optimizer"], config["lr"], config["weight_decay"])
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    if TEST_MODEL:
        # Load and evaluate a saved model
        model.load_state_dict(torch.load(os.path.join(path, "bin", TEST_MODEL_NAME)))
        final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
    else:
        # Train and evaluate
        best_model, history = train(model, config, train_loader, dev_loader, config["n_epochs"], criterion_train, criterion_eval, optimizer)
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

        if SAVE_MODEL:
            # Save best model
            torch.save(best_model.state_dict(), os.path.join(path, "bin", TEST_MODEL_NAME))

        if RESULTS:
            # Save results and plots
            result_path = os.path.join(path, create_folder())
            results = {
                'config': config,
                'history': history,
                'best_epoch': history["best_epoch"],
                'final_ppl': final_ppl,
            }
            extract_report_data(results, os.path.join(result_path, "result.json"))
            plot_loss_curves(history, os.path.join(result_path, "loss_curves.png"))
            plot_perplexity(history, os.path.join(result_path, "perplexity.png"))
            torch.save(best_model.state_dict(), os.path.join(result_path, "model.pt"))

    print("Final PPL: %f" % final_ppl)