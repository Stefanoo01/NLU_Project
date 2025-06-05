# Natural Language Understanding - BERT-based Intent Classification and Slot Filling (ATIS)

This project implements a joint model using a pre-trained BERT encoder for intent classification and slot filling on the ATIS dataset.

### Running the Script

To train or evaluate the model, run:

```bash
python main.py
```

### Configuration Flags

Modify the following flags at the top of `main.py` to control script behavior:

```python
TEST_MODEL = False
SAVE_MODEL = False
RESULTS = True
```

- `**TEST_MODEL**`:  
  - `True`: Load a saved model from `bin/model.pt` and evaluate it on the test set.  
  - `False`: Train a new model from scratch.

- `**SAVE_MODEL**`:  
  - `True`: Save the best model and associated metadata after training.

- `**RESULTS**`:  
  - `True`: Save training history, evaluation results, and plots to a timestamped folder in `results/`.

> To try different BERT models or training configurations, edit the `config` dictionary (e.g., model, etc.) in `main.py`.

### Output Files

When `RESULTS` is enabled, a folder is created under `results/`, containing:
- `result.json`: training history and config
- `loss_curves.png`: training loss visualization
- `model.pt`: trained model weights and vocab

The saved model in `bin/` includes:
- Weights, optimizer state, and vocab dictionaries (`w2id`, `slot2id`, `intent2id`)

