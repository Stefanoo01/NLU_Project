# Natural Language Understanding - ATIS Intent Classification and Slot Filling

This project implements a joint model for intent classification and slot filling on the ATIS dataset using an LSTM-based architecture.

### Running the Script

To execute the training or evaluation pipeline, run:

```bash
python main.py
```

### Configuration Flags

You can control the behavior of the script by setting the following flags at the top of `main.py`:

```python
TEST_MODEL = False
SAVE_MODEL = False
RESULTS = True
```

- `**TEST_MODEL**`:  
  - `True`: Load a previously saved model from `bin/model.pt` and evaluate it on the test set.  
  - `False`: Train a new model on the training set.

- `**SAVE_MODEL**`:  
  - `True`: Save the trained model to disk.

- `**RESULTS**`:  
  - `True`: Save evaluation metrics, loss curves, and the trained model to a timestamped folder inside `results/`.

> To test different model architectures and training settings, modify the `config` dictionary in `main.py`.

### Output Files

If `RESULTS` is enabled, a new folder under `results/` is created with:
- `result.json`: metrics and configuration
- `loss_curves.png`: loss plots for slot and intent tasks for the lasrt run
- `model.pt`: model weights and vocabularies
