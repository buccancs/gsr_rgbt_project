# src/tests/regression/test_model_configurations.py

import sys
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.ml_models.model_config import ModelConfig
from src.ml_models.model_interface import ModelRegistry
from src.ml_models.pytorch_models import PyTorchLSTMModel, PyTorchAutoencoderModel, PyTorchVAEModel


class TestModelConfigurations(unittest.TestCase):
    """
    Tests for different model configurations.
    These tests verify that different model configurations work correctly.
    """

    def setUp(self):
        """Set up test data."""
        # Create test data
        self.num_samples = 100
        self.window_size = 32
        self.num_features = 4

        # X shape: (num_samples, window_size, num_features)
        self.X = np.random.randn(self.num_samples, self.window_size, self.num_features).astype(np.float32)
        # y shape: (num_samples,)
        self.y = np.random.randn(self.num_samples).astype(np.float32)

        # Split data into train and validation sets
        self.X_train, self.X_val = self.X[:80], self.X[80:]
        self.y_train, self.y_val = self.y[:80], self.y[80:]

        # Input shape for models
        self.input_shape = (self.window_size, self.num_features)

    @patch('src.ml_models.pytorch_cnn_models.PyTorchCNNModel')
    def test_cnn_configurations(self, mock_model_class):
        """Test different CNN model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up mock predict method to return array with correct shape
        mock_model.predict.return_value = np.zeros(len(self.X_val))

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("cnn", mock_factory)

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

            # Check that the factory's create_model method was called with the right parameters
            mock_factory.create_model.assert_called_with(self.input_shape, config.get_config())

            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()

            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.X_val),))

            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()

    @patch('src.ml_models.pytorch_cnn_models.PyTorchCNNLSTMModel')
    def test_cnn_lstm_configurations(self, mock_model_class):
        """Test different CNN-LSTM model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up mock predict method to return array with correct shape
        mock_model.predict.return_value = np.zeros(len(self.X_val))

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("cnn_lstm", mock_factory)

        # Test different CNN-LSTM configurations
        cnn_lstm_configs = [
            # Basic CNN-LSTM
            {
                "model_params": {
                    "conv_channels": [32, 64],
                    "kernel_sizes": [3, 3],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "lstm_hidden_size": 64,
                    "lstm_num_layers": 1,
                    "fc_layers": [32, 1],
                    "activations": ["relu", "relu", "linear"],
                    "dropout_rate": 0.2
                }
            },
            # Deeper CNN-LSTM
            {
                "model_params": {
                    "conv_channels": [32, 64, 128],
                    "kernel_sizes": [3, 3, 3],
                    "strides": [1, 1, 1],
                    "pool_sizes": [2, 2, 2],
                    "lstm_hidden_size": 128,
                    "lstm_num_layers": 2,
                    "fc_layers": [64, 32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.3
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

            # Check that the factory's create_model method was called with the right parameters
            mock_factory.create_model.assert_called_with(self.input_shape, config.get_config())

            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()

            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.X_val),))

            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()

    @patch('src.ml_models.pytorch_cnn_models.PyTorchDualStreamCNNLSTMModel')
    def test_dual_stream_configurations(self, mock_model_class):
        """Test different dual-stream model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up mock predict method to return array with correct shape
        mock_model.predict.return_value = np.zeros(len(self.y_val))

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("dual_stream_cnn_lstm", mock_factory)

        # Test different dual-stream configurations
        dual_stream_configs = [
            # Basic dual-stream
            {
                "model_params": {
                    "rgb_conv_channels": [32, 64],
                    "thermal_conv_channels": [32, 64],
                    "kernel_sizes": [3, 3],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "lstm_hidden_size": 64,
                    "lstm_num_layers": 1,
                    "fc_layers": [32, 1],
                    "activations": ["relu", "relu", "linear"],
                    "dropout_rate": 0.2
                }
            },
            # Asymmetric dual-stream
            {
                "model_params": {
                    "rgb_conv_channels": [32, 64, 128],
                    "thermal_conv_channels": [16, 32, 64],
                    "kernel_sizes": [3, 3, 3],
                    "strides": [1, 1, 1],
                    "pool_sizes": [2, 2, 2],
                    "lstm_hidden_size": 128,
                    "lstm_num_layers": 2,
                    "fc_layers": [64, 32, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.3
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
                "train_params": {"batch_size": 32, "epochs": 5}
            }

            # Create model
            model = ModelRegistry.create_model("dual_stream_cnn_lstm", self.input_shape, config.get_config())

            # Create dummy thermal data
            thermal_X_train = np.random.randn(len(self.X_train), self.window_size, self.num_features).astype(np.float32)
            thermal_X_val = np.random.randn(len(self.X_val), self.window_size, self.num_features).astype(np.float32)

            # Train model
            model.fit((self.X_train, thermal_X_train), self.y_train, (self.X_val, thermal_X_val), self.y_val)

            # Make predictions
            predictions = model.predict((self.X_val, thermal_X_val))

            # Check that the factory's create_model method was called with the right parameters
            mock_factory.create_model.assert_called_with(self.input_shape, config.get_config())

            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()

            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.y_val),))

            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()

    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    def test_lstm_configurations(self, mock_model_class):
        """Test different LSTM model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up mock predict method to return array with correct shape
        mock_model.predict.return_value = np.zeros(len(self.X_val))

        # Set up mock fit method to return dict with train_loss
        mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("lstm", mock_factory)

        # Test different LSTM configurations
        lstm_configs = [
            # Basic LSTM
            {
                "model_params": {
                    "input_size": self.num_features,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "dropout": 0.2,
                    "bidirectional": False,
                    "fc_layers": [32, 1],
                    "activations": ["relu", "linear"]
                }
            },
            # Bidirectional LSTM
            {
                "model_params": {
                    "input_size": self.num_features,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "dropout": 0.2,
                    "bidirectional": True,
                    "fc_layers": [32, 1],
                    "activations": ["relu", "linear"]
                }
            },
            # Deep LSTM
            {
                "model_params": {
                    "input_size": self.num_features,
                    "hidden_size": 128,
                    "num_layers": 3,
                    "dropout": 0.3,
                    "bidirectional": False,
                    "fc_layers": [64, 32, 1],
                    "activations": ["relu", "relu", "linear"]
                }
            }
        ]

        for i, config_dict in enumerate(lstm_configs):
            # Create config
            config = ModelConfig()
            config.config = {
                **config_dict,
                "optimizer_params": {"type": "adam", "lr": 0.001},
                "loss_fn": "mse",
                "train_params": {"batch_size": 32, "epochs": 5}
            }

            # Create model through the factory
            model = ModelRegistry.create_model("lstm", self.input_shape, config.get_config())

            # Train model
            model.fit(self.X_train, self.y_train)

            # Make predictions
            predictions = model.predict(self.X_val)

            # Check that the factory's create_model method was called with the right parameters
            mock_factory.create_model.assert_called_with(self.input_shape, config.get_config())

            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()

            # Check that history contains train_loss
            history = mock_model.fit.return_value
            self.assertIn("train_loss", history)

            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, (len(self.X_val),))

            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()

    @patch('src.ml_models.pytorch_models.PyTorchAutoencoderModel')
    def test_autoencoder_configurations(self, mock_model_class):
        """Test different Autoencoder model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up mock predict method to return array with correct shape
        mock_model.predict.return_value = np.zeros(self.X_val.shape)

        # Set up mock fit method to return dict with train_loss
        mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("autoencoder", mock_factory)

        # Test different Autoencoder configurations
        ae_configs = [
            # Basic Autoencoder
            {
                "model_params": {
                    "input_size": self.num_features * self.window_size,
                    "latent_dim": 32,
                    "encoder_layers": [128, "latent_dim"],
                    "decoder_layers": [128, "input_size"],
                    "activations": ["relu", "relu", "relu", "sigmoid"]
                }
            },
            # Deep Autoencoder
            {
                "model_params": {
                    "input_size": self.num_features * self.window_size,
                    "latent_dim": 16,
                    "encoder_layers": [256, 128, 64, "latent_dim"],
                    "decoder_layers": [64, 128, 256, "input_size"],
                    "activations": ["relu", "relu", "relu", "relu", "relu", "relu", "sigmoid"]
                }
            }
        ]

        for i, config_dict in enumerate(ae_configs):
            # Create config
            config = ModelConfig()
            config.config = {
                **config_dict,
                "optimizer_params": {"type": "adam", "lr": 0.001},
                "loss_fn": "mse",
                "train_params": {"batch_size": 32, "epochs": 5}
            }

            # Create model through the factory
            model = ModelRegistry.create_model("autoencoder", self.input_shape, config.get_config())

            # Train model
            model.fit(self.X_train)

            # Make predictions
            predictions = model.predict(self.X_val)

            # Check that the factory's create_model method was called with the right parameters
            mock_factory.create_model.assert_called_with(self.input_shape, config.get_config())

            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()

            # Check that history contains train_loss
            history = mock_model.fit.return_value
            self.assertIn("train_loss", history)

            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, self.X_val.shape)

            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()

    @patch('src.ml_models.pytorch_models.PyTorchVAEModel')
    def test_vae_configurations(self, mock_model_class):
        """Test different VAE model configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up mock predict method to return array with correct shape
        mock_model.predict.return_value = np.zeros(self.X_val.shape)

        # Set up mock fit method to return dict with train_loss
        mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("vae", mock_factory)

        # Test different VAE configurations
        vae_configs = [
            # Basic VAE
            {
                "model_params": {
                    "input_size": self.num_features * self.window_size,
                    "latent_dim": 32,
                    "encoder_layers": [128],
                    "decoder_layers": [128, "input_size"],
                    "activations": ["relu", "relu", "sigmoid"]
                }
            },
            # Deep VAE
            {
                "model_params": {
                    "input_size": self.num_features * self.window_size,
                    "latent_dim": 16,
                    "encoder_layers": [256, 128],
                    "decoder_layers": [128, 256, "input_size"],
                    "activations": ["relu", "relu", "relu", "relu", "sigmoid"]
                }
            }
        ]

        for i, config_dict in enumerate(vae_configs):
            # Create config
            config = ModelConfig()
            config.config = {
                **config_dict,
                "optimizer_params": {"type": "adam", "lr": 0.001},
                "loss_fn": "vae_loss",
                "train_params": {"batch_size": 32, "epochs": 5}
            }

            # Create model through the factory
            model = ModelRegistry.create_model("vae", self.input_shape, config.get_config())

            # Train model
            model.fit(self.X_train)

            # Make predictions
            predictions = model.predict(self.X_val)

            # Check that the factory's create_model method was called with the right parameters
            mock_factory.create_model.assert_called_with(self.input_shape, config.get_config())

            # Check that fit and predict were called
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()

            # Check that history contains train_loss
            history = mock_model.fit.return_value
            self.assertIn("train_loss", history)

            # Check that predictions have the right shape
            self.assertEqual(predictions.shape, self.X_val.shape)

            # Reset mock for next iteration
            mock_model.reset_mock()
            mock_model_class.reset_mock()


if __name__ == "__main__":
    unittest.main()
