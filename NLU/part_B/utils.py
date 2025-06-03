import json
from collections import Counter
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

DEVICE = 'cuda:0'
PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.tokenizer = tokenizer
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids, self.slot_ids = self.mapping_seq(self.utterances, self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, slots, mapper):
        res = []
        res_slots = []
        
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        for sentence, slot_seq in zip(data, slots):
            token_ids = []
            slot_ids = []

            words = sentence.split()
            slot_tags = slot_seq.split()

            for word, slot in zip(words, slot_tags):
                word_tokens = self.tokenizer.tokenize(word)
                word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

                token_ids.extend(word_token_ids)
                # First subtoken gets the real slot, others get 'pad'
                slot_ids.extend([mapper[slot]] + [mapper['pad']] * (len(word_token_ids) - 1))

            # Add CLS and SEP
            token_ids = [cls_id] + token_ids + [sep_id]
            slot_ids = [mapper['pad']] + slot_ids + [mapper['pad']]

            res.append(token_ids)
            res_slots.append(slot_ids)

        return res, res_slots

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    attention_mask = torch.LongTensor([[1]*l + [0]*(y_slots.shape[1]-l) for l in y_lengths])
    token_type_ids = torch.zeros_like(y_slots, dtype=torch.long)
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attention_mask"] = attention_mask
    new_item["token_type_ids"] = token_type_ids

    return new_item

def create_dev_set(tmp_train_raw, test_raw, portion=0.1):
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)

    return X_train, X_dev