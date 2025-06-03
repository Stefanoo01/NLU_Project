import torch
import torch.nn as nn
from transformers import BertModel

class JointBERT(nn.Module):
    def __init__(self, slot_size, intent_size, dropout_rate=0.1, pretrained_model_name="bert-base-uncased"):
        super(JointBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # Slot filling head (token-level)
        self.slot_classifier = nn.Linear(hidden_size, slot_size)

        # Intent classification head (sentence-level, [CLS] token)
        self.intent_classifier = nn.Linear(hidden_size, intent_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = self.dropout(outputs.last_hidden_state)  # (B, T, H)
        pooled_output = self.dropout(outputs.pooler_output)        # (B, H)

        # Slot predictions (B, T, slot_size)
        slot_logits = self.slot_classifier(sequence_output)

        # Intent prediction (B, intent_size)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits.permute(0, 2, 1), intent_logits