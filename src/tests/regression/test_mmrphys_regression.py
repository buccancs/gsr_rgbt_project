import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
from unittest.mock import patch, MagicMock

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.processing.mmrphys_processor import MMRPhysProcessor
from src.ml_pipeline.feature_engineering.feature_engineering import create_dataset_from_session, align_signals, create_feature_windows
from src.ml_models.model_config import ModelConfig
from src.scripts.train_model import build_model_from_config


class TestMMRPhysEndToEndPipeline(unittest.TestCase):
    """
    End-to-end regression tests for the MMRPhysProcessor integration with the ML pipeline.
    These tests verify that the MMRPhysProcessor works correctly in an end-to-end scenario.
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

        # Create a test frame
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create a synthetic pulse signal
        fps = 30
        duration_seconds = 10
        bpm = 72  # 1.2 Hz = 72 BPM
        t = np.linspace(0, duration_seconds, fps * duration_seconds)
        self.test_pulse_signal = np.sin(2 * np.pi * (bpm / 60) * t)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    @patch('src.ml_pipeline.feature_engineering.feature_engineering.SessionDataLoader')
    @patch('src.ml_pipeline.feature_engineering.feature_engineering.process_gsr_signal')
    @patch('src.ml_pipeline.feature_engineering.feature_engineering.detect_palm_roi')
    @patch('src.ml_pipeline.feature_engineering.feature_engineering.extract_roi_signal')
    @patch('src.processing.mmrphys_processor.MMRPhysLEF')
    @patch('torch.load')
    @patch('os.path.exists')
    @patch('sys.path.append')
    @patch('cv2.VideoCapture')
    def test_mmrphys_to_feature_engineering_pipeline(self, mock_video_capture, mock_path_append, 
                                                   mock_exists, mock_torch_load, mock_lef, 
                                                   mock_extract_roi, mock_detect_roi, 
                                                   mock_process_gsr, mock_loader):
        """Test the pipeline from MMRPhysProcessor to feature engineering."""
        # Setup mocks for MMRPhysProcessor
        mock_exists.return_value = True
        mock_torch_load.return_value = {'state_dict': {}}

        mock_lef_instance = MagicMock()
        mock_lef.return_value = mock_lef_instance

        # Mock the model's forward pass
        mock_output = torch.tensor(self.test_pulse_signal)
        mock_lef_instance.forward = MagicMock(return_value=mock_output)

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 90 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

        # Mock read method to return frames then stop
        mock_cap.read.side_effect = [(True, self.test_frame) for _ in range(30)] + [(False, None)]

        # Setup mocks for feature_engineering
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
        mock_loader_instance.get_rgb_video_generator.return_value = [
            (True, self.test_frame) for _ in range(10)
        ]

        # Setup other mocks
        mock_process_gsr.return_value = mock_processed_gsr
        mock_detect_roi.return_value = [(10, 10), (30, 30)]
        mock_extract_roi.return_value = [100, 120, 140]  # RGB values

        # Initialize the processor
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)

        # Process a video
        video_path = str(self.test_dir / "test_video.mp4")
        results = processor.process_video(video_path, output_dir=str(self.test_dir))

        # Check that the processor returned results
        self.assertIsNotNone(results)
        self.assertIn('pulse_signal', results)
        self.assertIn('heart_rate', results)

        # Create dataset using the existing pipeline
        dataset = create_dataset_from_session(session_path, gsr_sampling_rate, video_fps)

        # Add the pulse signal from MMRPhysProcessor to the dataset
        dataset['MMRPhys_Pulse'] = np.nan  # Initialize with NaN
        dataset.loc[:len(results['pulse_signal'])-1, 'MMRPhys_Pulse'] = results['pulse_signal']

        # Create feature windows
        feature_cols = ["RGB_R", "RGB_G", "RGB_B", "GSR_Tonic", "MMRPhys_Pulse"]
        target_col = "GSR_Phasic"
        X, y = create_feature_windows(dataset, feature_cols, target_col, 32, 16)

        # Check that the feature windows were created successfully
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[2], len(feature_cols))  # Number of features

    @patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
    @patch('src.processing.mmrphys_processor.MMRPhysLEF')
    @patch('torch.load')
    @patch('os.path.exists')
    @patch('sys.path.append')
    @patch('cv2.VideoCapture')
    def test_mmrphys_to_model_pipeline(self, mock_video_capture, mock_path_append, 
                                      mock_exists, mock_torch_load, mock_lef, mock_lstm_class):
        """Test the pipeline from MMRPhysProcessor to model training and prediction."""
        # Setup mocks for MMRPhysProcessor
        mock_exists.return_value = True
        mock_torch_load.return_value = {'state_dict': {}}

        mock_lef_instance = MagicMock()
        mock_lef.return_value = mock_lef_instance

        # Mock the model's forward pass
        mock_output = torch.tensor(self.test_pulse_signal)
        mock_lef_instance.forward = MagicMock(return_value=mock_output)

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 90 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

        # Mock read method to return frames then stop
        mock_cap.read.side_effect = [(True, self.test_frame) for _ in range(30)] + [(False, None)]

        # Setup mock for LSTM model
        mock_lstm = MagicMock()
        mock_lstm_class.return_value = mock_lstm
        mock_lstm.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
        mock_lstm.predict.return_value = np.random.randn(self.batch_size)

        # Initialize the processor
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)

        # Process a video
        video_path = str(self.test_dir / "test_video.mp4")
        results = processor.process_video(video_path, output_dir=str(self.test_dir))

        # Create synthetic dataset with the pulse signal from MMRPhysProcessor
        timestamps = pd.to_datetime(np.arange(0, 10, 1/30), unit="s")
        dataset = pd.DataFrame({
            "timestamp": timestamps,
            "RGB_R": np.random.randn(len(timestamps)),
            "RGB_G": np.random.randn(len(timestamps)),
            "RGB_B": np.random.randn(len(timestamps)),
            "GSR_Tonic": np.random.randn(len(timestamps)),
            "GSR_Phasic": np.random.randn(len(timestamps)),
            "MMRPhys_Pulse": np.pad(results['pulse_signal'], 
                                   (0, len(timestamps) - len(results['pulse_signal'])), 
                                   'constant', constant_values=np.nan)
        })

        # Create feature windows
        feature_cols = ["RGB_R", "RGB_G", "RGB_B", "GSR_Tonic", "MMRPhys_Pulse"]
        target_col = "GSR_Phasic"
        X, y = create_feature_windows(dataset, feature_cols, target_col, 32, 16)

        # Build model
        input_shape = (32, len(feature_cols))
        model = build_model_from_config(input_shape, "lstm", config_path=None)

        # Train the model
        history = model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Verify the pipeline worked
        self.assertIsNotNone(history)
        self.assertIn("train_loss", history)
        self.assertEqual(predictions.shape, (len(X),))

    @patch('src.processing.mmrphys_processor.MMRPhysLEF')
    @patch('src.processing.mmrphys_processor.MMRPhysMEF')
    @patch('src.processing.mmrphys_processor.MMRPhysSEF')
    @patch('torch.load')
    @patch('os.path.exists')
    @patch('sys.path.append')
    def test_different_mmrphys_model_types(self, mock_path_append, mock_exists, 
                                          mock_torch_load, mock_sef, mock_mef, mock_lef):
        """Test different MMRPhysProcessor model types."""
        # Setup mocks
        mock_exists.return_value = True
        mock_torch_load.return_value = {'state_dict': {}}

        # Setup mock instances
        mock_lef_instance = MagicMock()
        mock_lef.return_value = mock_lef_instance

        mock_mef_instance = MagicMock()
        mock_mef.return_value = mock_mef_instance

        mock_sef_instance = MagicMock()
        mock_sef.return_value = mock_sef_instance

        # Test with MMRPhysLEF
        processor_lef = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        self.assertEqual(processor_lef.model_type, 'MMRPhysLEF')

        # Test with MMRPhysMEF
        processor_mef = MMRPhysProcessor(model_type='MMRPhysMEF', use_gpu=False)
        self.assertEqual(processor_mef.model_type, 'MMRPhysMEF')

        # Test with MMRPhysSEF
        processor_sef = MMRPhysProcessor(model_type='MMRPhysSEF', use_gpu=False)
        self.assertEqual(processor_sef.model_type, 'MMRPhysSEF')

        # Verify that all model types can preprocess frames
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        frame_tensor_lef = processor_lef.preprocess_frame(test_frame)
        frame_tensor_mef = processor_mef.preprocess_frame(test_frame)
        frame_tensor_sef = processor_sef.preprocess_frame(test_frame)

        # Check that all processors can preprocess frames correctly
        self.assertEqual(frame_tensor_lef.shape, (1, 3, 224, 224))
        self.assertEqual(frame_tensor_mef.shape, (1, 3, 224, 224))
        self.assertEqual(frame_tensor_sef.shape, (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
