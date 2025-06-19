# src/tests/smoke/test_smoke.py

import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.processing.feature_engineering import create_dataset_from_session
from src.scripts.train_model import build_model_from_config


class TestSmokeTests(unittest.TestCase):
    """
    Smoke tests for the main pipelines.
    These tests verify that the main pipelines run without errors.
    """

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    @patch('src.processing.feature_engineering.detect_palm_roi')
    @patch('src.processing.feature_engineering.extract_roi_signal')
    def test_create_dataset_from_session_smoke(self, mock_extract_roi, mock_detect_roi, 
                                              mock_process_gsr, mock_loader):
        """Smoke test for create_dataset_from_session."""
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

        # Call the function
        result = create_dataset_from_session(session_path, gsr_sampling_rate, video_fps)

        # Just check that it runs without errors and returns something
        self.assertIsNotNone(result)

    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    def test_model_training_smoke(self, mock_model_class):
        """Smoke test for model training."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_model.predict.return_value = np.random.randn(10)

        # Create dummy data
        X = np.random.randn(10, 20, 4)
        y = np.random.randn(10)

        # Build model
        input_shape = (20, 4)
        model = build_model_from_config(input_shape, "lstm")

        # Train model
        history = model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Just check that it runs without errors
        self.assertIsNotNone(history)
        self.assertIsNotNone(predictions)

    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    @patch('src.ml_models.pytorch_models.PyTorchAutoencoderModel')
    def test_multiple_models_smoke(self, mock_ae_class, mock_lstm_class):
        """Smoke test for training multiple model types."""
        # Setup mock models
        mock_lstm = MagicMock()
        mock_lstm_class.return_value = mock_lstm
        mock_lstm.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_lstm.predict.return_value = np.random.randn(10)

        mock_ae = MagicMock()
        mock_ae_class.return_value = mock_ae
        mock_ae.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_ae.predict.return_value = np.random.randn(10, 20, 4)

        # Create dummy data
        X = np.random.randn(10, 20, 4)
        y = np.random.randn(10)

        # Build and test LSTM model
        input_shape = (20, 4)
        lstm_model = build_model_from_config(input_shape, "lstm")
        lstm_history = lstm_model.fit(X, y)
        lstm_predictions = lstm_model.predict(X)

        # Build and test Autoencoder model
        ae_model = build_model_from_config(input_shape, "autoencoder")
        ae_history = ae_model.fit(X)
        ae_predictions = ae_model.predict(X)

        # Just check that everything runs without errors
        self.assertIsNotNone(lstm_history)
        self.assertIsNotNone(lstm_predictions)
        self.assertIsNotNone(ae_history)
        self.assertIsNotNone(ae_predictions)


if __name__ == "__main__":
    unittest.main()