import json
from collections import Counter
from transformers import BertTokenizer
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

DEVICE = 'cuda:0'
PAD_TOKEN = 0

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_data(path):
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class LangBERT():
    def __init__(self, intents, slots):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

class BERTDataset(data.Dataset):
    def __init__(self, dataset, lang):
        self.samples = dataset
        self.lang = lang

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        utterance = item['utterance'].split()
        slots = item['slots']
        intent = item['intent']
        
        bert_tokens = []
        slot_labels = []

        for word, slot in zip(utterance, slots):
            sub_tokens = tokenizer.tokenize(word)
            bert_tokens.extend(sub_tokens)
            # First sub-token gets the label, others get pad label (ignored in loss)
            slot_id = self.lang.slot2id.get(slot, self.lang.slot2id['pad'])
            slot_labels.extend([slot_id] + [PAD_TOKEN] * (len(sub_tokens) - 1))

        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        slot_labels = [PAD_TOKEN] + slot_labels + [PAD_TOKEN]

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)
        slot_labels = torch.tensor(slot_labels)
        intent_id = torch.tensor(self.lang.intent2id[intent])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "slot_labels": slot_labels,
            "intent_id": intent_id
        }

def collate_fn(data):
    def pad_sequence(sequences, pad_value=PAD_TOKEN):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded

    input_ids = pad_sequence([item['input_ids'] for item in data])
    attention_mask = pad_sequence([item['attention_mask'] for item in data])
    token_type_ids = pad_sequence([item['token_type_ids'] for item in data])
    slot_labels = pad_sequence([item['slot_labels'] for item in data])
    intent_ids = torch.stack([item['intent_id'] for item in data])

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "token_type_ids": token_type_ids.to(DEVICE),
        "slot_labels": slot_labels.to(DEVICE),
        "intent_ids": intent_ids.to(DEVICE)
    }

def create_dev_set(tmp_train_raw, test_raw, portion=0.1):
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels
    )
    X_train.extend(mini_train)
    return X_train, X_dev