import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from utils import *
from model import *
from functions import *

TEST_MODEL = False
SAVE_MODEL = False
RESULTS = True
DEVICE = "cuda:0"

config = {
    # Model
    "model": "bert-base-uncased",  # 'bert-base-uncased' or 'bert-large-uncased'

    # Optimizer
    "lr": 2e-5,         

    # Training control
    "n_epochs": 100,     
    "patience": 3,     

    # Data / batching
    "batch_size": 32,
    "max_len": 128,      # max token length for BERT
}

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    # Load raw JSON
    full_train = load_data(os.path.join(path, "dataset", "ATIS", "train.json"))
    full_test  = load_data(os.path.join(path, "dataset", "ATIS", "test.json"))

    # Create a dev split from full_train (stratified by intent)
    train_raw, dev_raw = create_dev_set(full_train, full_test, portion=0.1)

    # Build Lang (for mapping intent/slot strings to IDs)
    words = sum([ex["utterance"].split() for ex in train_raw], [])
    corpus = train_raw + dev_raw + full_test
    slot_tags = sum([ex["slots"].split()   for ex in corpus], [])
    intents   = [ex["intent"]             for ex in corpus]
    lang = Lang(words, set(intents), set(slot_tags), cutoff=0)

    if TEST_MODEL:
        # Load saved mappings and model state
        with open(os.path.join(path, "bin/lang.pkl"), "rb") as f:
            lang = pickle.load(f)
        saved = torch.load(os.path.join(path, "bin/model.pt"))
        lang.word2id   = saved["w2id"]
        lang.slot2id   = saved["slot2id"]
        lang.intent2id = saved["intent2id"]

    # Wrap datasets with ATISDataset (BERT tokenizer + slot alignment)
    train_dataset = ATISDataset(train_raw, lang, max_len=config["max_len"])
    dev_dataset   = ATISDataset(dev_raw,   lang, max_len=config["max_len"])
    test_dataset  = ATISDataset(full_test, lang, max_len=config["max_len"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    # Instantiate model
    num_intents = len(lang.intent2id)
    num_slots   = len(lang.slot2id)
    model = JointBert(
        pretrained_model_name=config["model"],
        num_intent_labels=num_intents,
        num_slot_labels=num_slots,
        dropout_prob=0.1
    ).to(DEVICE)

    # Optimizer + loss functions
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion_slots   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_LABEL)
    criterion_intents = nn.CrossEntropyLoss()

    if TEST_MODEL:
        # If you want to load a saved model and evaluate on test set:
        model.load_state_dict(saved["model"])
        test_metrics = eval_loop(
            test_loader,
            model,
            criterion_slots,
            criterion_intents,
            DEVICE,
            lang
        )
        print(f"Slot F1 on Test: {test_metrics['slot_f1']:.4f}")
        print(f"Intent Accuracy on Test: {test_metrics['intent_acc']:.4f}")
    else:
        # Full training with early stopping
        best_model, history = train(
            model,
            config,
            train_loader,
            dev_loader,
            test_loader,
            criterion_slots,
            criterion_intents,
            optimizer,
            DEVICE,
            lang
        )

        # Print aggregated results
        print(f"Slot F1: {history['slot_f1_score']}")
        print(f"Intent Acc: {history['intent_accuracy']}")

        if SAVE_MODEL:
            os.makedirs(os.path.join(path, "bin"), exist_ok=True)
            # Save model state, optimizer state, and mappings
            to_save = {
                "model": best_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "w2id": lang.word2id,
                "slot2id": lang.slot2id,
                "intent2id": lang.intent2id,
            }
            torch.save(to_save, os.path.join(path, "bin/model.pt"))
            # Save Lang object
            with open(os.path.join(path, "bin/lang.pkl"), "wb") as f:
                pickle.dump(lang, f)
        
        if RESULTS:
            # Save results
            result_path = os.path.join(path, create_folder())
            results = {
                'config': config,
                'history': history,
                'best_epoch': history["best_epoch"],
            }
            extract_report_data(results, os.path.join(result_path, "result.json"))
            plot_loss_curves(history, os.path.join(result_path, "loss_curves.png"))
            saving_object = {"epoch": 0, 
                        "model": model.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "w2id": lang.word2id, 
                        "slot2id": lang.slot2id, 
                        "intent2id": lang.intent2id}
            torch.save(saving_object, os.path.join(result_path, "model.pt"))
            with open(os.path.join(result_path, "lang.pkl"), "wb") as f:
                pickle.dump(lang, f)