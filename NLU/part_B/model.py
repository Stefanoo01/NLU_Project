# model.py

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class JointBert(nn.Module):
    """
    Joint BERT model for intent classification + slot filling.

    - Uses a pre-trained BERT encoder.
    - Intent “head” is a linear layer on top of pooled [CLS] output.
    - Slot “head” is a linear layer on top of token-level sequence output.
    - Sub-token alignment (i.e. mapping word-level labels to sub-tokens,
      with padding label = -100) must be handled externally in the data pipeline.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_intent_labels: int,
        num_slot_labels: int,
        dropout_prob: float = 0.1
    ):
        """
        Args:
            pretrained_model_name (str):
                HuggingFace name (or path) of a pre-trained BERT.
            num_intent_labels (int):
                Number of intent classes.
            num_slot_labels (int):
                Number of slot classes (including "O" and any PAD/ignore index if used).
            dropout_prob (float, optional):
                Dropout probability on top of BERT representations.
        """
        super(JointBert, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        # Dropout layer (applied to both pooled_output and sequence_output)
        self.dropout = nn.Dropout(dropout_prob)

        # Intent classification head: from [CLS] → num_intent_labels
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)

        # Slot filling head: from each token’s hidden → num_slot_labels
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)

        # Initialize the classification heads (weights taken from BERT’s initializer)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.constant_(self.intent_classifier.bias, 0)

        nn.init.xavier_uniform_(self.slot_classifier.weight)
        nn.init.constant_(self.slot_classifier.bias, 0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.LongTensor = None
    ):
        """
        Forward pass.

        Args:
            input_ids (torch.LongTensor of shape [batch_size, seq_len]):
                Token IDs from BERT’s tokenizer.
            attention_mask (torch.Tensor of shape [batch_size, seq_len]):
                1 for real tokens, 0 for padding.
            token_type_ids (torch.LongTensor of shape [batch_size, seq_len], optional):
                Segment IDs (only needed if sentences pairs; otherwise all zeros).

        Returns:
            intent_logits: torch.FloatTensor of shape [batch_size, num_intent_labels]
            slot_logits:   torch.FloatTensor of shape [batch_size, seq_len, num_slot_labels]
        """
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        # pooler_output: [batch_size, hidden_size] (i.e. representation for [CLS])
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)             # [batch, hidden_size]
        sequence_output = self.dropout(sequence_output)         # [batch, seq_len, hidden_size]

        # Intent prediction (on [CLS])
        intent_logits = self.intent_classifier(pooled_output)   # [batch, num_intent_labels]

        # Slot prediction (token‐level)
        #    slot_logits will be used with CrossEntropyLoss(ignore_index=-100)
        #    so that any “ignored” subtokens (with label -100) do not contribute.
        slot_logits = self.slot_classifier(sequence_output)     # [batch, seq_len, num_slot_labels]

        return intent_logits, slot_logits