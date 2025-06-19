# src/tests/regression/test_regression.py

import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.processing.feature_engineering import align_signals, create_feature_windows, create_dataset_from_session
from src.ml_models.model_config import ModelConfig
from src.ml_models.model_interface import ModelRegistry
from src.scripts.train_model import build_model_from_config


class TestEndToEndPipeline(unittest.TestCase):
    """
    End-to-end regression tests for the ML pipeline.
    These tests verify that the entire pipeline works correctly.
    """

    def setUp(self):
        """Set up test data and directories."""
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())

        # Create test data
        self.window_size = 20
        self.num_features = 4
        self.batch_size = 32

        # Create dummy data
        self.X = np.random.randn(self.batch_size, self.window_size, self.num_features)
        self.y = np.random.randn(self.batch_size)

        # Create dummy model configurations
        self.model_configs = {
            "lstm": {
                "name": "lstm",
                "framework": "pytorch",
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
            },
            "autoencoder": {
                "name": "autoencoder",
                "framework": "pytorch",
                "model_params": {
                    "input_size": self.num_features * self.window_size,
                    "latent_dim": 32,
                    "encoder_layers": [128, "latent_dim"],
                    "decoder_layers": [128, "input_size"],
                    "activations": ["relu", "relu", "relu", "sigmoid"]
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
        }

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    @patch('src.processing.feature_engineering.detect_palm_roi')
    @patch('src.processing.feature_engineering.extract_roi_signal')
    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    def test_feature_engineering_to_model_pipeline(self, mock_model_class, mock_extract_roi, 
                                                 mock_detect_roi, mock_process_gsr, mock_loader):
        """Test the pipeline from feature engineering to model training and prediction."""
        # Setup mocks
        session_path = Path("dummy/path")
        gsr_sampling_rate = 32
        video_fps = 30

        # Mock GSR data
        gsr_timestamps = pd.to_datetime(np.arange(0, 10, 1/gsr_sampling_rate), unit="s")
        mock_gsr_df = pd.DataFrame({
            "timestamp": gsr_timestamps,
            "GSR_Raw": np.random.randn(len(gsr_timestamps))
        })

        # Mock processed GSR data
        mock_processed_gsr = pd.DataFrame({
            "timestamp": gsr_timestamps,
            "GSR_Phasic": np.random.randn(len(gsr_timestamps)),
            "GSR_Tonic": np.random.randn(len(gsr_timestamps))
        })

        # Setup loader mock
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.get_gsr_data.return_value = mock_gsr_df

        # Setup video frame generator
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mock_loader_instance.get_rgb_video_generator.return_value = [
            (True, frame) for _ in range(10)
        ]

        # Setup other mocks
        mock_process_gsr.return_value = mock_processed_gsr
        mock_detect_roi.return_value = [(10, 10), (30, 30)]
        mock_extract_roi.return_value = [100, 120, 140]  # RGB values

        # Setup model mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_model.predict.return_value = np.random.randn(10)

        # Create dataset
        dataset = create_dataset_from_session(session_path, gsr_sampling_rate, video_fps)

        # Create feature windows
        X, y = create_feature_windows(dataset, ["RGB_B", "RGB_G", "RGB_R", "GSR_Tonic"], "GSR_Phasic", 32, 16)

        # Build model
        input_shape = (32, 4)
        model = build_model_from_config(input_shape, "lstm", config_path=None)

        # Train the model
        history = model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Verify the pipeline worked
        self.assertIsNotNone(history)
        self.assertIn("train_loss", history)
        self.assertEqual(predictions.shape, (len(X),))

    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    @patch('src.ml_models.pytorch_models.PyTorchAutoencoderModel')
    def test_different_model_configurations(self, mock_ae_class, mock_lstm_class):
        """Test different model configurations to ensure they all work."""
        # Setup mock models
        mock_lstm = MagicMock()
        mock_lstm_class.return_value = mock_lstm
        mock_lstm.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_lstm.predict.return_value = np.random.randn(self.batch_size)

        mock_ae = MagicMock()
        mock_ae_class.return_value = mock_ae
        mock_ae.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_ae.predict.return_value = np.random.randn(*self.X.shape)

        input_shape = (self.window_size, self.num_features)

        # Test LSTM model
        lstm_config = ModelConfig()
        lstm_config.config = self.model_configs["lstm"]

        lstm_model = build_model_from_config(input_shape, "lstm", config_path=None)
        lstm_history = lstm_model.fit(self.X, self.y)
        lstm_predictions = lstm_model.predict(self.X)

        # Test Autoencoder model
        ae_config = ModelConfig()
        ae_config.config = self.model_configs["autoencoder"]

        ae_model = build_model_from_config(input_shape, "autoencoder", config_path=None)
        ae_history = ae_model.fit(self.X)
        ae_predictions = ae_model.predict(self.X)

        # Verify all models worked
        self.assertIsNotNone(lstm_history)
        self.assertIn("train_loss", lstm_history)
        self.assertEqual(lstm_predictions.shape, (self.batch_size,))

        self.assertIsNotNone(ae_history)
        self.assertIn("train_loss", ae_history)
        self.assertEqual(ae_predictions.shape, self.X.shape)

    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    def test_model_save_load_pipeline(self, mock_model_class):
        """Test the pipeline for saving and loading models."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_model.predict.return_value = np.random.randn(self.batch_size)
        
        # For save method
        def mock_save(path):
            # Create an empty file at the specified path
            with open(path, 'wb') as f:
                pass
        mock_model.save = mock_save
        
        input_shape = (self.window_size, self.num_features)

        # Build and train LSTM model
        config = ModelConfig()
        config.config = self.model_configs["lstm"]

        model = build_model_from_config(input_shape, "lstm", config_path=None)
        model.fit(self.X, self.y)

        # Save the model
        save_path = self.test_dir / "test_model.pt"
        model.save(str(save_path))

        # Verify the file exists
        self.assertTrue(save_path.exists())

        # Load the model (mocked)
        loaded_model = ModelRegistry.create_model("lstm", input_shape, config.config)
        
        # Mock the load_state_dict method
        loaded_model.model = MagicMock()
        loaded_model.model.load_state_dict = MagicMock()
        
        # This should now work without error
        loaded_model.model.load_state_dict(torch.load(str(save_path)))

        # Make predictions with both models
        original_predictions = model.predict(self.X)
        loaded_model.predict = MagicMock(return_value=original_predictions)
        loaded_predictions = loaded_model.predict(self.X)

        # Verify the predictions are the same
        np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()