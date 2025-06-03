import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

class ModelBERT(nn.Module):

    def __init__(self, config, out_slot, out_int, dropout=0.5):
        super(ModelBERT, self).__init__()
        
        self.num_intents = out_int
        self.num_slots = out_slot

        self.bert = BertModel(config)

        self.slot_out = nn.Linear(config.hidden_size, out_slot)
        self.intent_out = nn.Linear(config.hidden_size, out_int)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, attention_mask, token_type_ids, input_ids):
        bert_output = self.bert(attention_mask=attention_mask, token_type_ids=token_type_ids, input_ids=input_ids)
        last_hidden_states = bert_output.last_hidden_state
        pooler_output = bert_output.pooler_output
        
        dropout_slots = self.dropout(last_hidden_states)
        dropout_intent = self.dropout(pooler_output)
        
        # Compute slot logits
        slots = self.slot_out(dropout_slots)
        # Compute intent logits
        intent = self.intent_out(dropout_intent)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1)
        # Slot size: batch_size, classes, seq_len
        return slots, intent