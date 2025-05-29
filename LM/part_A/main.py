from functions import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from functools import partial
import os

TEST_MODEL = False
SAVE_MODEL = True
RESULTS = False

config = {
    'lr': 0.0001,
    'dropout': 0.5,
    'hid_size': 200,
    'emb_size': 300,
    'clip': 5,
    'n_layers': 1,
    'weight_decay': 0.01,
    'n_epochs': 100,
    'patience': 3
}

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(path, "dataset/PennTreebank")
    train_raw = read_file(os.path.join(dataset_path, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(dataset_path, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(dataset_path, "ptb.test.txt"))

    lang = Lang(train_raw, special_tokens=["<pad>", "<unk>"])
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)

    model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = get_optimizer(model, config["lr"], config["weight_decay"])
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    if TEST_MODEL:
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)

    else:
        best_model, history = train(model, config, train_loader, dev_loader, n_epochs, criterion_train, criterion_eval, optimizer)
        best_model.to(device)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        if SAVE_MODEL:
            torch.save(best_model.state_dict(), os.path.join(path, "model.pt"))

        print("Final PPL: %f" % final_ppl)

    if RESULTS:
        result_path = os.path.join(path, create_folder())
        results = {
            'config': config,
            'history': history,
            'best_epoch': best_epoch,
            'final_ppl': final_ppl,
        }
        extract_report_data(results, os.path.join(result_path, "result.json"))
        plot_loss_curves(history, os.path.join(result_path, "loss_curves.png"))
        plot_perplexity(history, os.path.join(result_path, "perplexity.png"))
        torch.save(best_model.state_dict(), os.path.join(result_path, "model.pt"))