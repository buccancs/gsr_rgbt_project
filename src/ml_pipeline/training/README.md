# Model Training

This directory contains scripts and utilities for training machine learning models on the GSR-RGBT dataset.

## Files

### train_model.py

The `train_model.py` script is the main entry point for training machine learning models. It provides a command-line interface for training various types of models (LSTM, Autoencoder, VAE, CNN, CNN-LSTM, Transformer, ResNet) with different configurations.

## Usage

The script can be run from the command line with various options:

```bash
python src/ml_pipeline/training/train_model.py --model-type lstm --config-path configs/models/lstm_config.yaml
```

### Command-line Arguments

- `--model-type`: Type of model to train (lstm, autoencoder, vae, cnn, cnn_lstm, transformer, resnet)
- `--config-path`: Path to model configuration YAML file
- `--data-dir`: Directory containing session recordings
- `--gsr-sampling-rate`: Sampling rate of the GSR signal in Hz
- `--video-fps`: Frame rate of the video in frames per second
- `--cv-method`: Cross-validation method (loso, kfold)
- `--cv-folds`: Number of folds for k-fold cross-validation
- `--validation-split`: Fraction of training data to use for validation
- `--output-dir`: Directory to save models and results
- `--save-metadata`: Whether to save training metadata (true, false)
- `--list-configs`: List available model configurations and exit
- `--create-example-configs`: Create example configuration files for all model types and exit

### Example Configuration File

```yaml
name: lstm
framework: pytorch
model_params:
  input_size: 4
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  bidirectional: false
  fc_layers: [32, 16, 1]
  activations: ["relu", "relu", "linear"]
optimizer_params:
  type: adam
  lr: 0.001
  weight_decay: 1e-5
loss_fn: mse
train_params:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  early_stopping:
    patience: 10
    monitor: val_loss
  checkpoint:
    save_best_only: true
    monitor: val_loss
```

## Cross-validation

The script supports two types of cross-validation:

1. **Leave-One-Subject-Out (LOSO)**: Trains the model on all subjects except one, and tests on the left-out subject. This is repeated for each subject.
2. **K-Fold**: Splits the data into K folds and trains K models, each using K-1 folds for training and the remaining fold for testing.

## Output

The script saves the following files for each fold:

- Trained model (`.pt` for PyTorch, `.keras` for TensorFlow)
- Data scaler (`.joblib`)
- Training metadata (`.json`)
- Cross-validation metrics (`.csv`)

The metadata includes information about the model configuration, preprocessing parameters, training parameters, and evaluation metrics.