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


class TestEarlyStopping(unittest.TestCase):
    """
    Test suite for the EarlyStopping class.
    """

    def setUp(self):
        """Set up common objects for all tests."""
        self.patience = 3
        self.min_delta = 0.01
        self.monitor = "val_loss"
        self.early_stopping = src.ml_models.pytorch_models.EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta,
            monitor=self.monitor
        )

        # Create a simple model for testing
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )

    def test_initialization(self):
        """Test that EarlyStopping is initialized correctly."""
        self.assertEqual(self.early_stopping.patience, self.patience)
        self.assertEqual(self.early_stopping.min_delta, self.min_delta)
        self.assertEqual(self.early_stopping.monitor, self.monitor)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertIsNone(self.early_stopping.best_score)
        self.assertFalse(self.early_stopping.early_stop)
        self.assertIsNone(self.early_stopping.best_state_dict)

    def test_first_call(self):
        """Test the first call to EarlyStopping."""
        # First call should always save the model
        val_metrics = {self.monitor: 0.5}
        should_stop = self.early_stopping(val_metrics, self.model)

        self.assertFalse(should_stop)
        self.assertEqual(self.early_stopping.best_score, 0.5)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertIsNotNone(self.early_stopping.best_state_dict)

    def test_improvement(self):
        """Test when the monitored metric improves."""
        # First call
        val_metrics = {self.monitor: 0.5}
        self.early_stopping(val_metrics, self.model)

        # Second call with improvement
        val_metrics = {self.monitor: 0.4}  # Lower is better for loss
        should_stop = self.early_stopping(val_metrics, self.model)

        self.assertFalse(should_stop)
        self.assertEqual(self.early_stopping.best_score, 0.4)
        self.assertEqual(self.early_stopping.counter, 0)

    def test_no_improvement(self):
        """Test when the monitored metric does not improve."""
        # First call
        val_metrics = {self.monitor: 0.5}
        self.early_stopping(val_metrics, self.model)

        # Second call with no improvement
        val_metrics = {self.monitor: 0.6}  # Higher is worse for loss
        should_stop = self.early_stopping(val_metrics, self.model)

        self.assertFalse(should_stop)
        self.assertEqual(self.early_stopping.best_score, 0.5)
        self.assertEqual(self.early_stopping.counter, 1)

    def test_early_stopping(self):
        """Test that training stops after patience is exceeded."""
        # First call
        val_metrics = {self.monitor: 0.5}
        self.early_stopping(val_metrics, self.model)

        # Calls with no improvement
        for i in range(self.patience):
            val_metrics = {self.monitor: 0.6 + i * 0.1}  # Getting worse
            should_stop = self.early_stopping(val_metrics, self.model)

            if i < self.patience - 1:
                self.assertFalse(should_stop)
            else:
                self.assertTrue(should_stop)
                self.assertTrue(self.early_stopping.early_stop)

    def test_restore_best_weights(self):
        """Test that the best weights are restored."""
        # First call
        val_metrics = {self.monitor: 0.5}
        self.early_stopping(val_metrics, self.model)

        # Save the best state dict
        best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Change the model weights
        for param in self.model.parameters():
            param.data = param.data + 1.0

        # Verify weights changed
        for (k1, v1), (k2, v2) in zip(best_state_dict.items(), self.model.state_dict().items()):
            self.assertFalse(torch.all(torch.eq(v1, v2)))

        # Restore best weights
        self.early_stopping.restore_best_weights(self.model)

        # Verify weights restored
        for (k1, v1), (k2, v2) in zip(best_state_dict.items(), self.model.state_dict().items()):
            self.assertTrue(torch.all(torch.eq(v1, v2)))


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
