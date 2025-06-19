# src/tests/unit/test_pytorch_models.py

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.ml_models.model_interface import ModelRegistry
from src.ml_models.pytorch_cnn_models import (
    PyTorchCNNModel,
    PyTorchCNNLSTMModel,
    PyTorchDualStreamCNNLSTMModel
)
from src.ml_models.pytorch_models import (
    EarlyStopping,
    PyTorchLSTMModel,
    PyTorchAutoencoderModel,
    PyTorchVAEModel
)


class TestPyTorchModels(unittest.TestCase):
    """
    Tests for PyTorch models.
    """

    def setUp(self):
        """Set up test data and directories."""
        # Create a temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test data
        self.window_size = 32
        self.num_features = 4
        self.input_shape_1d = (self.window_size, self.num_features)
        self.input_shape_2d = (self.window_size, self.num_features, 1)  # For CNN models

        # Create a basic config for testing
        self.config = {
            "model_params": {
                "input_size": self.num_features,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False,
                "fc_layers": [32, 16, 1],
                "activations": ["relu", "relu", "linear"]
            },
            "optimizer_params": {
                "type": "adam",
                "lr": 0.001,
                "weight_decay": 1e-5
            },
            "loss_fn": "mse",
            "train_params": {
                "epochs": 2,
                "batch_size": 8,
                "validation_split": 0.2,
                "early_stopping": {
                    "patience": 3,
                    "monitor": "val_loss"
                }
            }
        }

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_early_stopping(self):
        """Test the EarlyStopping class."""
        # Create an EarlyStopping instance
        early_stopping = EarlyStopping(patience=2, monitor="val_loss")

        # Create a mock model
        mock_model = MagicMock()

        # Test with improving validation loss
        self.assertFalse(early_stopping({"val_loss": 0.5}, mock_model))  # First epoch
        self.assertFalse(early_stopping({"val_loss": 0.4}, mock_model))  # Improvement
        self.assertFalse(early_stopping({"val_loss": 0.3}, mock_model))  # Improvement

        # Test with worsening validation loss
        self.assertFalse(early_stopping({"val_loss": 0.35}, mock_model))  # Worse than previous, counter = 1
        self.assertTrue(early_stopping({"val_loss": 0.4}, mock_model))   # Worse again, counter = 2, should stop

        # Test reset
        early_stopping.reset()
        self.assertFalse(early_stopping({"val_loss": 0.5}, mock_model))  # First epoch after reset

    def test_pytorch_lstm_model_init(self):
        """Test the initialization of PyTorchLSTMModel."""
        model = PyTorchLSTMModel(self.input_shape_1d, self.config)

        # Check that the model has the expected attributes
        self.assertEqual(model.input_shape, self.input_shape_1d)
        self.assertEqual(model.config, self.config)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.criterion)

    def test_pytorch_lstm_model_fit(self):
        """Test the fit method of PyTorchLSTMModel."""
        model = PyTorchLSTMModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)
        y = np.random.rand(batch_size).astype(np.float32)

        # Train the model
        history = model.fit(X, y)

        # Check that history contains expected keys
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)

    def test_pytorch_lstm_model_predict(self):
        """Test the predict method of PyTorchLSTMModel."""
        model = PyTorchLSTMModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Make predictions
        predictions = model.predict(X)

        # Check output shape
        self.assertEqual(predictions.shape, (batch_size,))

    def test_pytorch_lstm_model_evaluate(self):
        """Test the evaluate method of PyTorchLSTMModel."""
        model = PyTorchLSTMModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)
        y = np.random.rand(batch_size).astype(np.float32)

        # Evaluate the model
        metrics = model.evaluate(X, y)

        # Check that metrics contains expected keys
        self.assertIn("loss", metrics)

    def test_pytorch_lstm_model_save_load(self):
        """Test the save and load methods of PyTorchLSTMModel."""
        model = PyTorchLSTMModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Get predictions before saving
        predictions_before = model.predict(X)

        # Save the model
        save_path = self.temp_dir / "lstm_model.pt"
        model.save(str(save_path))

        # Check that the file exists
        self.assertTrue(save_path.exists())

        # Load the model
        loaded_model = PyTorchLSTMModel.load(str(save_path))

        # Check that the loaded model has the same architecture
        self.assertEqual(model.input_shape, loaded_model.input_shape)

        # Get predictions after loading
        predictions_after = loaded_model.predict(X)

        # Check that predictions are the same
        np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-8)

    def test_pytorch_autoencoder_model_init(self):
        """Test the initialization of PyTorchAutoencoderModel."""
        model = PyTorchAutoencoderModel(self.input_shape_1d, self.config)

        # Check that the model has the expected attributes
        self.assertEqual(model.input_shape, self.input_shape_1d)
        self.assertEqual(model.config, self.config)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.criterion)

    def test_pytorch_autoencoder_model_fit(self):
        """Test the fit method of PyTorchAutoencoderModel."""
        model = PyTorchAutoencoderModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Train the model
        history = model.fit(X, X)

        # Check that history contains expected keys
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)

    def test_pytorch_autoencoder_model_predict(self):
        """Test the predict method of PyTorchAutoencoderModel."""
        model = PyTorchAutoencoderModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Make predictions
        predictions = model.predict(X)

        # Check output shape
        self.assertEqual(predictions.shape, X.shape)

    def test_pytorch_autoencoder_model_evaluate(self):
        """Test the evaluate method of PyTorchAutoencoderModel."""
        model = PyTorchAutoencoderModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Evaluate the model
        metrics = model.evaluate(X)

        # Check that metrics contains expected keys
        self.assertIn("loss", metrics)

    def test_pytorch_autoencoder_model_save_load(self):
        """Test the save and load methods of PyTorchAutoencoderModel."""
        model = PyTorchAutoencoderModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Get predictions before saving
        predictions_before = model.predict(X)

        # Save the model
        save_path = self.temp_dir / "autoencoder_model.pt"
        model.save(str(save_path))

        # Check that the file exists
        self.assertTrue(save_path.exists())

        # Load the model
        loaded_model = PyTorchAutoencoderModel.load(str(save_path))

        # Check that the loaded model has the same architecture
        self.assertEqual(model.input_shape, loaded_model.input_shape)

        # Get predictions after loading
        predictions_after = loaded_model.predict(X)

        # Check that predictions are the same
        np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-8)

    def test_pytorch_vae_model_init(self):
        """Test the initialization of PyTorchVAEModel."""
        model = PyTorchVAEModel(self.input_shape_1d, self.config)

        # Check that the model has the expected attributes
        self.assertEqual(model.input_shape, self.input_shape_1d)
        self.assertEqual(model.config, self.config)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.optimizer)

    def test_pytorch_vae_model_fit(self):
        """Test the fit method of PyTorchVAEModel."""
        model = PyTorchVAEModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Train the model
        history = model.fit(X, X)

        # Check that history contains expected keys
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)

    def test_pytorch_vae_model_predict(self):
        """Test the predict method of PyTorchVAEModel."""
        model = PyTorchVAEModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Make predictions
        predictions = model.predict(X)

        # Check output shape
        self.assertEqual(predictions.shape, X.shape)

    def test_pytorch_vae_model_evaluate(self):
        """Test the evaluate method of PyTorchVAEModel."""
        model = PyTorchVAEModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Evaluate the model
        metrics = model.evaluate(X)

        # Check that metrics contains expected keys
        self.assertIn("loss", metrics)
        self.assertIn("reconstruction_loss", metrics)
        self.assertIn("kl_loss", metrics)

    def test_pytorch_vae_model_save_load(self):
        """Test the save and load methods of PyTorchVAEModel."""
        model = PyTorchVAEModel(self.input_shape_1d, self.config)

        # Create dummy data
        batch_size = 20
        X = np.random.rand(batch_size, self.window_size, self.num_features).astype(np.float32)

        # Get predictions before saving
        predictions_before = model.predict(X)

        # Save the model
        save_path = self.temp_dir / "vae_model.pt"
        model.save(str(save_path))

        # Check that the file exists
        self.assertTrue(save_path.exists())

        # Load the model
        loaded_model = PyTorchVAEModel.load(str(save_path))

        # Check that the loaded model has the same architecture
        self.assertEqual(model.input_shape, loaded_model.input_shape)

        # Get predictions after loading
        predictions_after = loaded_model.predict(X)

        # Check that predictions are the same
        np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-8)

    @patch('src.ml_models.model_interface.ModelRegistry._factories', {})
    def test_model_registry(self):
        """Test that the models can be created through the ModelRegistry."""
        # Reset the ModelRegistry._factories dictionary
        ModelRegistry._factories = {}

        # Register the model factories
        from src.ml_models.pytorch_cnn_models import PyTorchCNNFactory
        ModelRegistry.register("cnn", PyTorchCNNFactory())

        from src.ml_models.pytorch_cnn_models import PyTorchCNNLSTMFactory
        ModelRegistry.register("cnn_lstm", PyTorchCNNLSTMFactory())

        from src.ml_models.pytorch_cnn_models import PyTorchDualStreamCNNLSTMFactory
        ModelRegistry.register("dual_stream_cnn_lstm", PyTorchDualStreamCNNLSTMFactory())

        from src.ml_models.pytorch_models import PyTorchLSTMFactory
        ModelRegistry.register("lstm", PyTorchLSTMFactory())

        from src.ml_models.pytorch_models import PyTorchAutoencoderFactory
        ModelRegistry.register("autoencoder", PyTorchAutoencoderFactory())

        from src.ml_models.pytorch_models import PyTorchVAEFactory
        ModelRegistry.register("vae", PyTorchVAEFactory())

        # Get the updated list of registered models
        registered_models = ModelRegistry.get_registered_models()

        # Check if our models are registered
        self.assertIn("cnn", registered_models)
        self.assertIn("cnn_lstm", registered_models)
        self.assertIn("dual_stream_cnn_lstm", registered_models)
        self.assertIn("lstm", registered_models)
        self.assertIn("autoencoder", registered_models)
        self.assertIn("vae", registered_models)

        # Create models through the registry
        cnn_model = ModelRegistry.create_model("cnn", self.input_shape_1d, self.config)
        self.assertIsInstance(cnn_model, PyTorchCNNModel)

        cnn_lstm_model = ModelRegistry.create_model("cnn_lstm", self.input_shape_1d, self.config)
        self.assertIsInstance(cnn_lstm_model, PyTorchCNNLSTMModel)

        dual_stream_model = ModelRegistry.create_model("dual_stream_cnn_lstm", self.input_shape_1d, self.config)
        self.assertIsInstance(dual_stream_model, PyTorchDualStreamCNNLSTMModel)

        lstm_model = ModelRegistry.create_model("lstm", self.input_shape_1d, self.config)
        self.assertIsInstance(lstm_model, PyTorchLSTMModel)

        autoencoder_model = ModelRegistry.create_model("autoencoder", self.input_shape_1d, self.config)
        self.assertIsInstance(autoencoder_model, PyTorchAutoencoderModel)

        vae_model = ModelRegistry.create_model("vae", self.input_shape_1d, self.config)
        self.assertIsInstance(vae_model, PyTorchVAEModel)


if __name__ == "__main__":
    unittest.main()
