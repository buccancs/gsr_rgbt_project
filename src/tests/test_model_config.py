# src/tests/test_model_config.py

import unittest
import tempfile
import yaml
from pathlib import Path
import sys
import os
import shutil

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.ml_models.model_config import (
    ModelConfig,
    list_available_configs,
    create_example_config_files,
    DEFAULT_CONFIGS
)


class TestModelConfig(unittest.TestCase):
    """Test suite for the ModelConfig class and related functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())

        # Create a test configuration
        self.test_config = {
            "name": "test_model",
            "framework": "pytorch",
            "model_params": {
                "hidden_size": 128,
                "num_layers": 3,
                "dropout": 0.3,
                "nested": {
                    "param1": 10,
                    "param2": 20
                }
            },
            "optimizer_params": {
                "type": "adam",
                "lr": 0.002
            }
        }

        # Create a test configuration file
        self.config_path = self.test_dir / "test_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init_with_config_name(self):
        """Test initialization with a config name."""
        # Test with a valid config name
        config = ModelConfig("lstm")
        self.assertEqual(config.get_model_name(), "lstm")
        self.assertEqual(config.get_framework(), "pytorch")

        # Test with an alias that points to a non-existent config
        # In this case, it should fall back to the default PyTorch LSTM config
        config = ModelConfig("cnn")
        self.assertEqual(config.get_model_name(), "lstm")

        # Test with an invalid config name
        config = ModelConfig("invalid_config")
        self.assertEqual(config.get_model_name(), "lstm")  # Should default to lstm

    def test_init_with_config_path(self):
        """Test initialization with a config path."""
        config = ModelConfig(config_path=self.config_path)
        self.assertEqual(config.get_model_name(), "test_model")
        self.assertEqual(config.get_framework(), "pytorch")
        self.assertEqual(config.get_model_params()["hidden_size"], 128)

    def test_load_from_file(self):
        """Test loading configuration from a file."""
        config = ModelConfig()
        config.load_from_file(self.config_path)
        self.assertEqual(config.get_model_name(), "test_model")
        self.assertEqual(config.get_framework(), "pytorch")
        self.assertEqual(config.get_model_params()["hidden_size"], 128)

    def test_save_to_file(self):
        """Test saving configuration to a file."""
        config = ModelConfig("lstm")
        save_path = self.test_dir / "saved_config.yaml"
        config.save_to_file(save_path)

        # Load the saved config and verify
        with open(save_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        self.assertEqual(loaded_config["name"], "lstm")
        self.assertEqual(loaded_config["framework"], "pytorch")

    def test_update_config(self):
        """Test updating the configuration."""
        config = ModelConfig("lstm")

        # Update top-level parameters
        updates = {
            "name": "updated_model",
            "optimizer_params": {
                "lr": 0.005
            }
        }
        config.update_config(updates)

        self.assertEqual(config.get_model_name(), "updated_model")
        self.assertEqual(config.get_optimizer_params()["lr"], 0.005)

        # Test deep update
        updates = {
            "model_params": {
                "hidden_size": 256,
                "nested": {
                    "new_param": 30
                }
            }
        }
        config.update_config(updates)

        self.assertEqual(config.get_model_params()["hidden_size"], 256)
        self.assertEqual(config.get_model_params()["nested"]["new_param"], 30)

    def test_deep_update(self):
        """Test the _deep_update method."""
        config = ModelConfig("lstm")

        # Create nested dictionaries
        d = {
            "a": 1,
            "b": {
                "c": 2,
                "d": 3
            }
        }

        u = {
            "a": 10,
            "b": {
                "c": 20,
                "e": 30
            }
        }

        # Perform deep update
        config._deep_update(d, u)

        # Check results
        self.assertEqual(d["a"], 10)
        self.assertEqual(d["b"]["c"], 20)
        self.assertEqual(d["b"]["d"], 3)
        self.assertEqual(d["b"]["e"], 30)

    def test_get_model_params_pytorch(self):
        """Test getting model parameters for PyTorch models."""
        config = ModelConfig("pytorch_lstm")
        params = config.get_model_params()

        self.assertIn("hidden_size", params)
        self.assertIn("num_layers", params)
        self.assertIn("dropout", params)

    def test_get_model_params_tensorflow_lstm(self):
        """Test getting model parameters for TensorFlow LSTM models."""
        config = ModelConfig("tf_lstm")
        params = config.get_model_params()

        self.assertIn("lstm_units", params)
        self.assertIn("dense_units", params)
        self.assertIn("dropout_rates", params)

    def test_get_model_params_tensorflow_ae(self):
        """Test getting model parameters for TensorFlow Autoencoder models."""
        config = ModelConfig("tf_autoencoder")
        params = config.get_model_params()

        self.assertIn("latent_dim", params)
        self.assertIn("encoder_layers", params)
        self.assertIn("decoder_layers", params)

    def test_get_optimizer_params_pytorch(self):
        """Test getting optimizer parameters for PyTorch models."""
        config = ModelConfig("pytorch_lstm")
        params = config.get_optimizer_params()

        self.assertIn("type", params)
        self.assertIn("lr", params)
        self.assertEqual(params["type"], "adam")

    def test_get_optimizer_params_tensorflow(self):
        """Test getting optimizer parameters for TensorFlow models."""
        config = ModelConfig("tf_lstm")
        params = config.get_optimizer_params()

        self.assertIn("type", params)
        self.assertIn("lr", params)
        self.assertEqual(params["type"], "adam")

    def test_get_loss_fn_pytorch(self):
        """Test getting loss function for PyTorch models."""
        config = ModelConfig("pytorch_lstm")
        loss_fn = config.get_loss_fn()

        self.assertEqual(loss_fn, "mse")

    def test_get_loss_fn_tensorflow(self):
        """Test getting loss function for TensorFlow models."""
        config = ModelConfig("tf_lstm")
        loss_fn = config.get_loss_fn()

        self.assertEqual(loss_fn, "mean_absolute_error")

    def test_get_train_params_pytorch(self):
        """Test getting training parameters for PyTorch models."""
        config = ModelConfig("pytorch_lstm")
        params = config.get_train_params()

        self.assertIn("epochs", params)
        self.assertIn("batch_size", params)
        self.assertIn("validation_split", params)

    def test_get_train_params_tensorflow(self):
        """Test getting training parameters for TensorFlow models."""
        config = ModelConfig("tf_lstm")
        params = config.get_train_params()

        self.assertIn("epochs", params)
        self.assertIn("batch_size", params)
        self.assertIn("validation_split", params)
        self.assertIn("callbacks", params)

    def test_get_fit_params(self):
        """Test getting fit parameters."""
        config = ModelConfig("tf_lstm")
        params = config.get_fit_params()

        self.assertIn("epochs", params)
        self.assertIn("batch_size", params)
        self.assertIn("validation_split", params)

    def test_get_compile_params(self):
        """Test getting compile parameters."""
        config = ModelConfig("tf_lstm")
        params = config.get_compile_params()

        self.assertIn("optimizer", params)
        self.assertIn("loss", params)
        self.assertIn("metrics", params)

    def test_list_available_configs(self):
        """Test listing available configurations."""
        configs = list_available_configs()

        self.assertIsInstance(configs, list)
        self.assertIn("lstm", configs)
        self.assertIn("pytorch_lstm", configs)
        self.assertIn("tf_lstm", configs)

    def test_create_example_config_files(self):
        """Test creating example configuration files."""
        output_dir = self.test_dir / "configs"
        create_example_config_files(output_dir)

        # Check that files were created
        self.assertTrue((output_dir / "pytorch_lstm_config.yaml").exists())
        self.assertTrue((output_dir / "tf_lstm_config.yaml").exists())

        # Check that the files contain valid YAML
        with open(output_dir / "pytorch_lstm_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            self.assertEqual(config["name"], "lstm")
            self.assertEqual(config["framework"], "pytorch")


if __name__ == "__main__":
    unittest.main()
