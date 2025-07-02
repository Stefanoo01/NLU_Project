import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter

DEVICE = "cuda:0"
PAD_TOKEN_LABEL = -100  # for CrossEntropyLoss(ignore_index=PAD_TOKEN_LABEL)

def load_data(path):
    """
    Load a JSON file containing a list of ATIS examples.
    Each example should be a dict with keys:
      - 'utterance' (string of space‐separated words)
      - 'slots'     (string of space‐separated slot tags, same length as utterance)
      - 'intent'    (string intent label)
    Returns the parsed JSON (a Python list of dicts).
    """
    with open(path, 'r') as f:
        return json.load(f)

class Lang:
    """
    Simple utility for mapping slot‐tag strings and intent‐label strings to integer IDs.
    You should pass in:
      - words  : a (flat) list of all words in training utterances (used only for counting)
      - intents: a set of all intent‐strings in train/dev/test
      - slots  : a set of all slot‐tag strings in train/dev/test
      - cutoff : (optional) frequency cutoff for building a vocab of words (not strictly needed for BERT)
    After construction, you can access:
      - lang.intent2id  : dict mapping intent‐string → integer (0,1,2,…)
      - lang.id2intent  : reverse mapping
      - lang.slot2id    : dict mapping slot‐tag string → integer
      - lang.id2slot    : reverse mapping
    """
    def __init__(self, words, intents, slots, cutoff=0):
        # We don’t really need a word2id for BERT, but we build it anyway in case you want it.
        self.word2id = self._build_word2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self._build_label2id(slots, pad_label=True)
        self.intent2id = self._build_label2id(intents, pad_label=False)
        # reverse
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_word2id(self, elements, cutoff=None, unk=True):
        vocab = {"pad": 0}
        if unk:
            vocab["unk"] = len(vocab)
        counter = Counter(elements)
        for w, cnt in counter.items():
            if cnt > cutoff:
                vocab[w] = len(vocab)
        return vocab

    def _build_label2id(self, elements, pad_label=True):
        """
        Build a mapping from slot‐tag/intents → integer. 
        If pad_label=True, we reserve index 0 for a 'pad' label. 
        (For slot tags only; for intent labels we do not pad.)
        """
        lab2id = {}
        if pad_label:
            lab2id["pad"] = 0
        for lab in sorted(elements):
            # sorted ensures reproducibility
            if lab not in lab2id:
                lab2id[lab] = len(lab2id)
        return lab2id


class ATISDataset(Dataset):
    """
    A PyTorch Dataset that:
      - reads a list of dicts (each with 'utterance', 'slots', 'intent')
      - tokenizes with BERT’s tokenizer (sub‐word‐tokenization)
      - aligns word‐level slot tags to the sub‐tokens (marking extra sub‐tokens as PAD_TOKEN_LABEL)
      - returns a dict containing:
          * input_ids       (LongTensor of shape [max_len])
          * attention_mask  (LongTensor of shape [max_len])
          * token_type_ids  (LongTensor of shape [max_len])
          * slot_labels     (LongTensor of shape [max_len])  ← aligned to sub‐tokens
          * intent_label    (LongTensor of shape [1])
    """
    def __init__(self, data_list, lang, 
                 tokenizer_name: str = "bert-base-uncased",
                 max_len: int = 128):
        """
        Args:
          data_list      : a Python list of dicts, each with keys 'utterance','slots','intent'
          lang           : an instance of Lang (so we know slot2id and intent2id)
          tokenizer_name : which BERT model to load
          max_len        : maximum sequence length (pad/truncate to this)
        """
        self.sentences = [example["utterance"].split() for example in data_list]
        self.slot_tags  = [example["slots"].split()    for example in data_list]
        self.intent_ids = [lang.intent2id[example["intent"]] for example in data_list]
        self.slot2id    = lang.slot2id
        self.tokenizer  = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.max_len    = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Returns a dict with:
          - input_ids       : LongTensor [max_len]
          - attention_mask  : LongTensor [max_len]
          - token_type_ids  : LongTensor [max_len]
          - slot_labels     : LongTensor [max_len], where non‐first sub‐tokens = PAD_TOKEN_LABEL
          - intent_label    : LongTensor [1]
        """
        words     = self.sentences[idx]   # e.g. ["book","a","flight","from","new","york"]
        slot_tags = self.slot_tags[idx]   # e.g. ["O","O","O","O","B-fromloc","I-fromloc"]
        intent_id = self.intent_ids[idx]  # e.g. 3

        # Tokenize with BertTokenizer, keeping track of word→subtoken alignment
        #    By passing is_split_into_words=True, tokenizer will know each "words[i]" is one token.
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=False,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].squeeze(0)       # shape [max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # shape [max_len]
        token_type_ids = encoding["token_type_ids"].squeeze(0)  # shape [max_len]

        # Align slot_labels to sub-tokens
        #    encoding.word_ids() gives a list of length max_len, where each entry is:
        #      - None if the token is a special token ([CLS],[SEP], or padding)
        #      - an integer i ∈ [0..len(words)-1] if that sub‐token belongs to words[i]
        word_ids = encoding.word_ids()  # Python list of length `max_len`
        aligned_slot_labels = []

        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # This might be [CLS], [SEP], or padded positions → assign PAD_TOKEN_LABEL
                aligned_slot_labels.append(PAD_TOKEN_LABEL)
            elif word_idx != prev_word_idx:
                # The first sub-token of a given word → assign the true slot label ID
                tag_str = slot_tags[word_idx]  # e.g. "B-fromloc"
                aligned_slot_labels.append(self.slot2id[tag_str])
            else:
                # Subsequent sub-tokens of the same word → mark as PAD_TOKEN_LABEL (ignored in loss)
                aligned_slot_labels.append(PAD_TOKEN_LABEL)
            prev_word_idx = word_idx

        slot_label_ids = torch.LongTensor(aligned_slot_labels)  # shape [max_len]

        # Package everything in a dict
        return {
            "input_ids":      input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE),
            "token_type_ids": token_type_ids.to(DEVICE),
            "slot_labels":    slot_label_ids.to(DEVICE),
            "intent_label":   torch.LongTensor([intent_id]).to(DEVICE)
        }


def create_dev_set(full_train_list, full_test_list, portion=0.1):
    """
    Given raw ATIS JSON lists:
      1) full_train_list (list of dicts, each with 'utterance','slots','intent')
      2) full_test_list  (same format)
    Stratify-split full_train_list into new_train / dev (size = portion), by intent label.
    Returns (new_train_list, dev_list).
    Any example whose intent appears only once will go into new_train (not held out).
    """
    all_intents = [x["intent"] for x in full_train_list]
    from collections import Counter
    intent_counts = Counter(all_intents)

    singleton_examples = []
    multi_examples    = []
    for example in full_train_list:
        if intent_counts[example["intent"]] == 1:
            singleton_examples.append(example)
        else:
            multi_examples.append(example)

    # Perform a stratified split on the "multi_examples" only
    strat_intents = [ex["intent"] for ex in multi_examples]
    train_multi, dev_multi = train_test_split(
        multi_examples,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=strat_intents
    )

    # Put singleton_examples back into the training set
    new_train = train_multi + singleton_examples
    new_dev   = dev_multi

    return new_train, new_dev