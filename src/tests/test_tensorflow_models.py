# src/tests/test_tensorflow_models.py

"""
Test script for TensorFlow model implementations.

This script tests the basic functionality of the TensorFlow models
to ensure they can be instantiated, trained, and used for prediction.
"""

import numpy as np
import tensorflow as tf
from src.ml_models.tensorflow_cnn_models import TensorFlowCNNModel, TensorFlowCNNLSTMModel, TensorFlowDualStreamCNNLSTMModel
from src.ml_models.tensorflow_resnet_models import TensorFlowResNetModel
from src.ml_models.tensorflow_transformer_models import TensorFlowTransformerModel


def test_tensorflow_cnn_model():
    """Test the TensorFlow CNN model."""
    print("Testing TensorFlow CNN Model...")
    
    # Create sample data
    input_shape = (100, 4)  # 100 time steps, 4 features
    X = np.random.randn(50, 100, 4).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    
    # Model configuration
    config = {
        "model_params": {
            "conv_channels": [32, 64],
            "kernel_sizes": [5, 5],
            "strides": [1, 1],
            "pool_sizes": [2, 2],
            "fc_layers": [32, 1],
            "activations": ["relu", "relu", "relu", "linear"],
            "dropout_rate": 0.2
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001
        },
        "loss_fn": "mse",
        "train_params": {
            "batch_size": 16,
            "epochs": 2,
            "validation_split": 0.2,
            "early_stopping": {
                "patience": 5,
                "monitor": "val_loss"
            }
        }
    }
    
    # Create and test model
    model = TensorFlowCNNModel(input_shape, config)
    
    # Test training
    history = model.fit(X, y)
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
    
    # Test prediction
    predictions = model.predict(X[:10])
    print(f"Predictions shape: {predictions.shape}")
    
    # Test evaluation
    metrics = model.evaluate(X[:10], y[:10])
    print(f"Evaluation metrics: {metrics}")
    
    print("TensorFlow CNN Model test passed!\n")


def test_tensorflow_cnn_lstm_model():
    """Test the TensorFlow CNN-LSTM model."""
    print("Testing TensorFlow CNN-LSTM Model...")
    
    # Create sample data
    input_shape = (50, 4)  # 50 time steps, 4 features
    X = np.random.randn(30, 50, 4).astype(np.float32)
    y = np.random.randn(30).astype(np.float32)
    
    # Model configuration
    config = {
        "model_params": {
            "conv_channels": [16, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "pool_sizes": [2, 2],
            "lstm_hidden_size": 32,
            "lstm_num_layers": 1,
            "fc_layers": [16, 1],
            "activations": ["relu", "relu", "tanh", "relu", "linear"],
            "dropout_rate": 0.2,
            "bidirectional": False
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001
        },
        "loss_fn": "mse",
        "train_params": {
            "batch_size": 8,
            "epochs": 2,
            "validation_split": 0.2
        }
    }
    
    # Create and test model
    model = TensorFlowCNNLSTMModel(input_shape, config)
    
    # Test training
    history = model.fit(X, y)
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
    
    # Test prediction
    predictions = model.predict(X[:5])
    print(f"Predictions shape: {predictions.shape}")
    
    print("TensorFlow CNN-LSTM Model test passed!\n")


def test_tensorflow_resnet_model():
    """Test the TensorFlow ResNet model."""
    print("Testing TensorFlow ResNet Model...")
    
    # Create sample data
    input_shape = (64, 4)  # 64 time steps, 4 features
    X = np.random.randn(20, 64, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    
    # Model configuration
    config = {
        "model_params": {
            "layers": [32, 64],
            "blocks_per_layer": [2, 2],
            "fc_layers": [32, 1],
            "activations": ["relu", "relu", "relu", "linear"],
            "dropout_rate": 0.2
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001
        },
        "loss_fn": "mse",
        "train_params": {
            "batch_size": 8,
            "epochs": 2,
            "validation_split": 0.2
        }
    }
    
    # Create and test model
    model = TensorFlowResNetModel(input_shape, config)
    
    # Test training
    history = model.fit(X, y)
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
    
    # Test prediction
    predictions = model.predict(X[:5])
    print(f"Predictions shape: {predictions.shape}")
    
    print("TensorFlow ResNet Model test passed!\n")


def test_tensorflow_transformer_model():
    """Test the TensorFlow Transformer model."""
    print("Testing TensorFlow Transformer Model...")
    
    # Create sample data
    input_shape = (32, 4)  # 32 time steps, 4 features
    X = np.random.randn(20, 32, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    
    # Model configuration
    config = {
        "model_params": {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "fc_layers": [32, 1],
            "activations": ["relu", "linear"]
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001
        },
        "loss_fn": "mse",
        "train_params": {
            "batch_size": 8,
            "epochs": 2,
            "validation_split": 0.2
        }
    }
    
    # Create and test model
    model = TensorFlowTransformerModel(input_shape, config)
    
    # Test training
    history = model.fit(X, y)
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
    
    # Test prediction
    predictions = model.predict(X[:5])
    print(f"Predictions shape: {predictions.shape}")
    
    print("TensorFlow Transformer Model test passed!\n")


def test_model_registry():
    """Test that all models are properly registered."""
    print("Testing Model Registry...")
    
    from src.ml_models.model_interface import ModelRegistry
    
    registered_models = ModelRegistry.get_registered_models()
    print(f"Registered models: {registered_models}")
    
    # Check that our TensorFlow models are registered
    expected_models = [
        "tensorflow_cnn",
        "tensorflow_cnn_lstm", 
        "tensorflow_dual_stream_cnn_lstm",
        "tensorflow_resnet",
        "tensorflow_transformer"
    ]
    
    for model_name in expected_models:
        if model_name in registered_models:
            print(f"✓ {model_name} is registered")
        else:
            print(f"✗ {model_name} is NOT registered")
    
    print("Model Registry test completed!\n")


def main():
    """Run all tests."""
    print("Starting TensorFlow Models Test Suite...")
    print("=" * 50)
    
    try:
        # Test individual models
        test_tensorflow_cnn_model()
        test_tensorflow_cnn_lstm_model()
        test_tensorflow_resnet_model()
        test_tensorflow_transformer_model()
        
        # Test model registry
        test_model_registry()
        
        print("=" * 50)
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()