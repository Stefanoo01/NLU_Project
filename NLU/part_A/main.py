from functions import *
from model import *
from utils import *
from torch.utils.data import DataLoader
import os
import torch.optim as optim

TEST_MODEL = False
SAVE_MODEL = False
RESULTS = True
DEVICE = "cuda:0"

config = {
    'lr': 2e-5,
    'dropout': 0.1,
    'clip': 1,
    'n_epochs': 30,
    'patience': 5
}

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))

    tmp_train_raw = load_data(os.path.join(path, 'dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join(path, 'dataset','ATIS','test.json'))
    train_raw, dev_raw = create_dev_set(tmp_train_raw, test_raw, portion=0.1)

    corpus = train_raw + dev_raw + test_raw 
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = LangBERT(intents, slots)

    if TEST_MODEL:
        saving_object = torch.load(os.path.join(path, "model.pt"))
        lang.slot2id = saving_object['slot2id']
        lang.intent2id = saving_object['intent2id']
        lang.id2slot = {v: k for k, v in lang.slot2id.items()}
        lang.id2intent = {v: k for k, v in lang.intent2id.items()}

    train_dataset = BERTDataset(train_raw, lang)
    dev_dataset = BERTDataset(dev_raw, lang)
    test_dataset = BERTDataset(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    model = JointBERT(out_slot, out_int, dropout_rate=config["dropout"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    if TEST_MODEL:
        model.load_state_dict(saving_object['model'])
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        print('Slot F1 score:', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])

    else:
        best_model, history = train(model, config, train_loader, dev_loader, test_loader, criterion_slots, criterion_intents, optimizer, lang)

        print('Slot F1 score', round(history["slot_f1_scores"].mean(),3), '+-', round(history["slot_f1_scores"].std(),3))
        print('Intent Accuracy', round(history["intent_accuracies"].mean(), 3), '+-', round(history["intent_accuracies"].std(), 3))

        if SAVE_MODEL:
            saving_object = {
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                "slot2id": lang.slot2id, 
                "intent2id": lang.intent2id
            }
            torch.save(saving_object, os.path.join(path, "model.pt"))

        if RESULTS:
            result_path = os.path.join(path, create_folder())
            results = {
                'config': config,
                'history': history,
                'best_epoch': history["best_epoch"],
            }
            extract_report_data(results, os.path.join(result_path, "result.json"))
            plot_loss_curves(history, os.path.join(result_path, "loss_curves.png"))
            torch.save(saving_object, os.path.join(result_path, "model.pt"))