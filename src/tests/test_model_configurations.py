# src/tests/test_model_configurations.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.ml_models.model_config import ModelConfig
from src.ml_models.model_interface import ModelRegistry


class TestModelConfigurations(unittest.TestCase):
    """
    Test suite for different ML model configurations.
    These tests verify that different model configurations can be created and used.
    """
    
    def setUp(self):
        """Set up common objects for all tests."""
        # Create synthetic data
        np.random.seed(42)
        self.window_size = 32
        self.num_features = 4
        self.num_samples = 100
        
        # X shape: (num_samples, window_size, num_features)
        self.X = np.random.randn(self.num_samples, self.window_size, self.num_features).astype(np.float32)
        # y shape: (num_samples,)
        self.y = np.random.randn(self.num_samples).astype(np.float32)
        
        # Split data into train and validation sets
        self.X_train, self.X_val = self.X[:80], self.X[80:]
        self.y_train, self.y_val = self.y[:80], self.y[80:]
        
        # Input shape for models
        self.input_shape = (self.window_size, self.num_features)
    
    @patch('src.ml_models.pytorch_models.PyTorchCNNModel')
    def test_cnn_configurations(self, mock_model_class):
        """Test different CNN model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Register the mock model with the ModelRegistry
        ModelRegistry.register_model("cnn", mock_model_class)
        
        # Test different CNN configurations
        cnn_configs = [
            # Basic CNN
            {
                "model_params": {
                    "conv_channels": [32, 64],
                    "kernel_sizes": [3, 3],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "fc_layers": [32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.2
                }
            },
            # Deeper CNN
            {
                "model_params": {
                    "conv_channels": [32, 64, 128],
                    "kernel_sizes": [3, 3, 3],
                    "strides": [1, 1, 1],
                    "pool_sizes": [2, 2, 2],
                    "fc_layers": [64, 32, 1],
                    "activations": ["relu", "relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.3
                }
            },
            # Wider CNN
            {
                "model_params": {
                    "conv_channels": [64, 128],
                    "kernel_sizes": [5, 5],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "fc_layers": [128, 64, 1],
                    "activations": ["relu", "relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.4
                }
            }
        ]
        
        for i, config_dict in enumerate(cnn_configs):
            # Create config
            config = ModelConfig()
            config.config = {
                **config_dict,
                "optimizer_params": {"type": "adam", "lr": 0.001},
                "loss_params": {"type": "mse"},
                "train_params": {"batch_size": 32, "epochs": 5}
            }
            
            # Create model
            model = ModelRegistry.create_model("cnn", self.input_shape, config.get_config())
            
            # Train model
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            
            # Make predictions
            predictions = model.predict(self.X_val)
            
            # Check that the model was called with the right parameters
            mock_model_class.assert_called_with(self.input_shape, config.get_config())
            
            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()
            
            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.X_val),))
            
            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()
    
    @patch('src.ml_models.pytorch_models.PyTorchCNNLSTMModel')
    def test_cnn_lstm_configurations(self, mock_model_class):
        """Test different CNN-LSTM model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Register the mock model with the ModelRegistry
        ModelRegistry.register_model("cnn_lstm", mock_model_class)
        
        # Test different CNN-LSTM configurations
        cnn_lstm_configs = [
            # Basic CNN-LSTM
            {
                "model_params": {
                    "conv_channels": [32, 64],
                    "kernel_sizes": [3, 3],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "fc_layers": [32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.2,
                    "lstm_hidden_size": 64,
                    "lstm_num_layers": 1,
                    "bidirectional": False
                }
            },
            # Bidirectional LSTM
            {
                "model_params": {
                    "conv_channels": [32, 64],
                    "kernel_sizes": [3, 3],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "fc_layers": [32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.2,
                    "lstm_hidden_size": 64,
                    "lstm_num_layers": 1,
                    "bidirectional": True
                }
            },
            # Multi-layer LSTM
            {
                "model_params": {
                    "conv_channels": [32, 64],
                    "kernel_sizes": [3, 3],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "fc_layers": [32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.2,
                    "lstm_hidden_size": 64,
                    "lstm_num_layers": 2,
                    "bidirectional": False
                }
            }
        ]
        
        for i, config_dict in enumerate(cnn_lstm_configs):
            # Create config
            config = ModelConfig()
            config.config = {
                **config_dict,
                "optimizer_params": {"type": "adam", "lr": 0.001},
                "loss_params": {"type": "mse"},
                "train_params": {"batch_size": 32, "epochs": 5}
            }
            
            # Create model
            model = ModelRegistry.create_model("cnn_lstm", self.input_shape, config.get_config())
            
            # Train model
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            
            # Make predictions
            predictions = model.predict(self.X_val)
            
            # Check that the model was called with the right parameters
            mock_model_class.assert_called_with(self.input_shape, config.get_config())
            
            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()
            
            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.X_val),))
            
            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()
    
    @patch('src.ml_models.pytorch_models.PyTorchDualStreamCNNLSTMModel')
    def test_dual_stream_configurations(self, mock_model_class):
        """Test different dual-stream model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Register the mock model with the ModelRegistry
        ModelRegistry.register_model("dual_stream_cnn_lstm", mock_model_class)
        
        # Create dummy RGB and thermal frames
        seq_len = 30
        rgb_channels = 3
        thermal_channels = 3
        height = 64
        width = 64
        
        rgb_frames = np.random.randn(
            self.num_samples, seq_len, rgb_channels, height, width
        ).astype(np.float32)
        thermal_frames = np.random.randn(
            self.num_samples, seq_len, thermal_channels, height, width
        ).astype(np.float32)
        
        # Split data
        rgb_train, rgb_val = rgb_frames[:80], rgb_frames[80:]
        thermal_train, thermal_val = thermal_frames[:80], thermal_frames[80:]
        
        # Test different dual-stream configurations
        dual_stream_configs = [
            # Basic dual-stream
            {
                "model_params": {
                    "rgb_input_shape": (rgb_channels, height, width),
                    "thermal_input_shape": (thermal_channels, height, width),
                    "cnn_filters": [32, 64, 128],
                    "cnn_kernel_sizes": [3, 3, 3],
                    "cnn_strides": [1, 1, 1],
                    "cnn_pool_sizes": [2, 2, 2],
                    "fc_layers": [64, 32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.3,
                    "lstm_hidden_size": 64,
                    "lstm_num_layers": 1,
                    "bidirectional": False
                }
            },
            # Deeper CNN
            {
                "model_params": {
                    "rgb_input_shape": (rgb_channels, height, width),
                    "thermal_input_shape": (thermal_channels, height, width),
                    "cnn_filters": [32, 64, 128, 256],
                    "cnn_kernel_sizes": [3, 3, 3, 3],
                    "cnn_strides": [1, 1, 1, 1],
                    "cnn_pool_sizes": [2, 2, 2, 2],
                    "fc_layers": [128, 64, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.4,
                    "lstm_hidden_size": 128,
                    "lstm_num_layers": 2,
                    "bidirectional": True
                }
            }
        ]
        
        for i, config_dict in enumerate(dual_stream_configs):
            # Create config
            config = ModelConfig()
            config.config = {
                **config_dict,
                "optimizer_params": {"type": "adam", "lr": 0.001},
                "loss_params": {"type": "mse"},
                "train_params": {"batch_size": 16, "epochs": 5}
            }
            
            # Create model
            model = ModelRegistry.create_model("dual_stream_cnn_lstm", self.input_shape, config.get_config())
            
            # Train model (with dual-stream input)
            model.fit((rgb_train, thermal_train), self.y_train, (rgb_val, thermal_val), self.y_val)
            
            # Make predictions
            predictions = model.predict((rgb_val, thermal_val))
            
            # Check that the model was called with the right parameters
            mock_model_class.assert_called_with(self.input_shape, config.get_config())
            
            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()
            
            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.y_val),))
            
            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()


if __name__ == "__main__":
    unittest.main()