## Language Modeling - LSTM with Regularization on Penn Treebank

This project implements a language model using an LSTM architecture with some regularization techniques trained on the Penn Treebank dataset.

### Running the Script

To start the program, simply run:

```bash
python main.py
```

### Configuration Flags

You can enable or disable functionality by modifying these flags at the top of `main.py`:

```python
TEST_MODEL = True
TEST_MODEL_NAME = "model.pt"
SAVE_MODEL = True
RESULTS = True
```

- `**TEST_MODEL**`:  
  - `True`: Load a saved model from `bin/model.pt` and evaluate it on the test set.  
  - `False`: Train a new model from scratch.

- `**TEST_MODEL_NAME**`:
  - The name of the model file in the bin folder to load or save.

- `**SAVE_MODEL**`:  
  - `True`: Save the best model after training.

- `**RESULTS**`:  
  - `True`: Save training history, loss and perplexity plots, and model checkpoints in a timestamped folder.

> To test different model configurations (e.g., emb_size, hid_size, dropout, etc.), edit the `config` dictionary in `main.py`.

### Output Files

When `RESULTS` is enabled, a new folder is created under `results/`, containing:
- `result.json`: metrics and configuration
- `loss_curves.png`: training/validation loss curves
- `perplexity.png`: validation perplexity
- `model.pt`: best model weights
