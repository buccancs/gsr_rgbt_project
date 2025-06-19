# src/tests/test_pytorch_models.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.ml_models.model_interface import ModelRegistry
from src.ml_models.pytorch_cnn_models import (
    PyTorchCNNModel,
    PyTorchCNNLSTMModel,
    PyTorchDualStreamCNNLSTMModel
)


class TestPyTorchModels(unittest.TestCase):
    """
    Test suite for verifying the architecture and functionality of PyTorch models.
    """

    def setUp(self):
        """Set up common input shapes and configurations for all model tests."""
        # Common parameters for 1D models
        self.window_size = 50
        self.num_features = 4
        self.input_shape_1d = (self.window_size, self.num_features)
        
        # Parameters for dual-stream model
        self.seq_len = 30
        self.rgb_channels = 3
        self.thermal_channels = 3
        self.height = 64
        self.width = 64
        
        # Basic configuration for models
        self.config = {
            "model_params": {
                "conv_channels": [32, 64],
                "kernel_sizes": [3, 3],
                "strides": [1, 1],
                "pool_sizes": [2, 2],
                "fc_layers": [32, 1],
                "activations": ["relu", "relu", "relu", "linear"],
                "dropout_rate": 0.2,
                
                # LSTM parameters
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "bidirectional": False,
                
                # Dual-stream parameters
                "rgb_input_shape": (self.rgb_channels, self.height, self.width),
                "thermal_input_shape": (self.thermal_channels, self.height, self.width),
                "cnn_filters": [32, 64, 128],
                "cnn_kernel_sizes": [3, 3, 3],
                "cnn_strides": [1, 1, 1],
                "cnn_pool_sizes": [2, 2, 2]
            },
            "optimizer_params": {
                "type": "adam",
                "lr": 0.001
            },
            "loss_params": {
                "type": "mse"
            }
        }

    def test_pytorch_cnn_model(self):
        """Test that the PyTorch CNN model is created with correct shapes."""
        model = PyTorchCNNModel(self.input_shape_1d, self.config)
        
        # Check if the model is a PyTorch model
        self.assertIsInstance(model.model, torch.nn.Module)
        
        # Create a dummy input tensor
        batch_size = 10
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check output shape (should be a batch of single regression values)
        self.assertEqual(predictions.shape, (batch_size,))

    def test_pytorch_cnn_lstm_model(self):
        """Test that the PyTorch CNN-LSTM model is created with correct shapes."""
        model = PyTorchCNNLSTMModel(self.input_shape_1d, self.config)
        
        # Check if the model is a PyTorch model
        self.assertIsInstance(model.model, torch.nn.Module)
        
        # Create a dummy input tensor
        batch_size = 10
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check output shape (should be a batch of single regression values)
        self.assertEqual(predictions.shape, (batch_size,))

    def test_pytorch_dual_stream_cnn_lstm_model(self):
        """Test that the PyTorch Dual-Stream CNN-LSTM model is created with correct shapes."""
        model = PyTorchDualStreamCNNLSTMModel(self.input_shape_1d, self.config)
        
        # Check if the model is a PyTorch model
        self.assertIsInstance(model.model, torch.nn.Module)
        
        # Create dummy input tensors for RGB and thermal streams
        batch_size = 10
        rgb_frames = np.random.rand(
            batch_size, self.seq_len, self.rgb_channels, self.height, self.width
        ).astype(np.float32)
        thermal_frames = np.random.rand(
            batch_size, self.seq_len, self.thermal_channels, self.height, self.width
        ).astype(np.float32)
        
        # Make predictions
        predictions = model.predict((rgb_frames, thermal_frames))
        
        # Check output shape (should be a batch of single regression values)
        self.assertEqual(predictions.shape, (batch_size,))

    def test_model_registry(self):
        """Test that the models can be created through the ModelRegistry."""
        # Check if our models are registered
        registered_models = ModelRegistry.get_registered_models()
        self.assertIn("pytorch_cnn", registered_models)
        self.assertIn("pytorch_cnn_lstm", registered_models)
        self.assertIn("pytorch_dual_stream_cnn_lstm", registered_models)
        
        # Create models through the registry
        cnn_model = ModelRegistry.create_model("cnn", self.input_shape_1d, self.config)
        self.assertIsInstance(cnn_model, PyTorchCNNModel)
        
        cnn_lstm_model = ModelRegistry.create_model("cnn_lstm", self.input_shape_1d, self.config)
        self.assertIsInstance(cnn_lstm_model, PyTorchCNNLSTMModel)
        
        dual_stream_model = ModelRegistry.create_model("dual_stream_cnn_lstm", self.input_shape_1d, self.config)
        self.assertIsInstance(dual_stream_model, PyTorchDualStreamCNNLSTMModel)


if __name__ == "__main__":
    unittest.main()