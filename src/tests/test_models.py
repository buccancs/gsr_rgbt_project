# src/tests/test_models.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path
import logging

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow is not available. TensorFlow model tests will be skipped.")

# Try to import the model builders, but don't fail if they depend on TensorFlow
try:
    from src.ml_models.models import build_lstm_model, build_ae_model, build_vae_model
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("TensorFlow model builders are not available. Tests will be skipped.")


class TestModelCreation(unittest.TestCase):
    """
    Test suite for verifying the architecture and compilation of ML models.
    """

    def setUp(self):
        """Set up a common input shape for all model tests."""
        self.window_size = 50
        self.num_features = 4
        self.input_shape = (self.window_size, self.num_features)

    def test_build_lstm_model(self):
        """Test that the LSTM model is created with correct I/O shapes and is compiled."""
        if not TENSORFLOW_AVAILABLE or not MODELS_AVAILABLE:
            self.skipTest("TensorFlow or model builders not available")

        model = build_lstm_model(self.input_shape)

        # Check if the model is a Keras Model instance
        self.assertIsInstance(model, tf.keras.Model)

        # Check input shape
        self.assertEqual(model.input_shape, (None, self.window_size, self.num_features))

        # Check output shape (should be a single regression value)
        self.assertEqual(model.output_shape, (None, 1))

        # Check if the model is compiled with a loss and optimizer
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

    def test_build_ae_model(self):
        """Test that the Autoencoder model has an output shape that matches its input."""
        if not TENSORFLOW_AVAILABLE or not MODELS_AVAILABLE:
            self.skipTest("TensorFlow or model builders not available")

        latent_dim = 16
        model = build_ae_model(self.input_shape, latent_dim=latent_dim)

        self.assertIsInstance(model, tf.keras.Model)

        # AE's output shape must match the input shape for reconstruction
        self.assertEqual(model.input_shape, (None, self.window_size, self.num_features))
        self.assertEqual(
            model.output_shape, (None, self.window_size, self.num_features)
        )

        # Check that the encoder's output shape is the latent dimension
        encoder = model.get_layer("encoder")
        self.assertEqual(encoder.output_shape, (None, latent_dim))

        self.assertIsNotNone(model.optimizer)
        self.assertEqual(model.loss, "mse")

    def test_build_vae_model(self):
        """Test that the VAE model has an output shape matching its input and a custom loss."""
        if not TENSORFLOW_AVAILABLE or not MODELS_AVAILABLE:
            self.skipTest("TensorFlow or model builders not available")

        latent_dim = 16
        model = build_vae_model(self.input_shape, latent_dim=latent_dim)

        self.assertIsInstance(model, tf.keras.Model)

        # VAE's output shape must also match the input shape
        self.assertEqual(model.input_shape, (None, self.window_size, self.num_features))
        self.assertEqual(
            model.output_shape, (None, self.window_size, self.num_features)
        )

        # Check if the model has the custom VAE loss function added
        self.assertTrue(
            len(model.losses) > 0, "VAE model should have a KL divergence loss term."
        )

        self.assertIsNotNone(model.optimizer)


if __name__ == "__main__":
    unittest.main()
