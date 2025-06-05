from functions import *
from model import *
from utils import *
from torch.utils.data import DataLoader
import os
import torch.optim as optim

# Experiment flags
TEST_MODEL = False
SAVE_MODEL = False
RESULTS = True
DEVICE = "cuda:0"

# Configuration parameters
config = {
    'lr': 0.001,             # Learning rate for optimizer
    'emb_size': 200,         # Embedding size
    'hid_size': 300,         # Hidden size of LSTM
    'n_layers': 1,           # Number of LSTM layers
    'dropout': 0.5,          # Dropout rate (None to disable)
    'clip': 5,               # Gradient clipping threshold
    'runs': 5,               # Number of experiment runs
    'n_epochs': 100,         # Max number of training epochs
    'patience': 2,           # Early stopping patience
}

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    # Load and split dataset
    tmp_train_raw = load_data(os.path.join(path, 'dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join(path, 'dataset', 'ATIS', 'test.json'))
    train_raw, dev_raw = create_dev_set(tmp_train_raw, test_raw, portion=0.1)
    y_test = [x['intent'] for x in test_raw]  # For evaluation

    # Build vocabulary
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    # Load vocab from saved model if testing
    if TEST_MODEL:
        saving_object = torch.load(os.path.join(path, "bin/model.pt"))
        lang.word2id = saving_object['w2id']
        lang.slot2id = saving_object['slot2id']
        lang.intent2id = saving_object['intent2id']

    # Dataset creation
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Vocabulary sizes
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Model setup
    model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], vocab_len, pad_index=PAD_TOKEN, dropout=config["dropout"]).to(DEVICE)
    model.apply(init_weights)

    # Loss functions and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # Ignore pad tokens in slot loss
    criterion_intents = nn.CrossEntropyLoss()                      # Full intent loss

    if TEST_MODEL:
        # Load and evaluate saved model
        model.load_state_dict(saving_object['model'])
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        print('Slot F1 score:', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])

    else:
        # Train and evaluate
        best_model, history = train(model, config, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, optimizer, lang)
        
        # Print average scores over runs
        print('Slot F1 score', round(history["slot_f1_scores"].mean(), 3), '+-', round(history["slot_f1_scores"].std(), 3))
        print('Intent Accuracy', round(history["intent_accuracies"].mean(), 3), '+-', round(history["intent_accuracies"].std(), 3))

        if SAVE_MODEL:
            # Save model and associated metadata
            saving_object = {
                "epoch": 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "w2id": lang.word2id,
                "slot2id": lang.slot2id,
                "intent2id": lang.intent2id
            }
            torch.save(saving_object, os.path.join(path, "bin/model.pt"))

        if RESULTS:
            # Save results and training curves
            result_path = os.path.join(path, create_folder())
            results = {
                'config': config,
                'history': history,
                'best_epoch': history["best_epoch"],
            }
            extract_report_data(results, os.path.join(result_path, "result.json"))
            plot_loss_curves(history, os.path.join(result_path, "loss_curves.png"))

            # Save model again in results folder
            saving_object = {
                "epoch": 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "w2id": lang.word2id,
                "slot2id": lang.slot2id,
                "intent2id": lang.intent2id
            }
            torch.save(saving_object, os.path.join(result_path, "model.pt"))