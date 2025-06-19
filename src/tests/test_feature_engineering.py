# src/tests/test_feature_engineering.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.processing.feature_engineering import align_signals, create_feature_windows, create_dataset_from_session
from unittest.mock import patch, MagicMock


class TestFeatureEngineering(unittest.TestCase):
    """
    Test suite for the feature engineering functions (alignment and windowing).
    """

    def setUp(self):
        """Set up data that mimics processed GSR and video signals."""
        # Create a high-frequency signal (like GSR at 32Hz)
        gsr_timestamps = pd.to_datetime(np.arange(0, 10, 1 / 32.0), unit="s")
        self.gsr_df = pd.DataFrame(
            {
                "timestamp": gsr_timestamps,
                "GSR_Phasic": np.random.randn(len(gsr_timestamps)),
            }
        )

        # Create a lower-frequency signal (like video features at 30Hz)
        video_timestamps = pd.to_datetime(np.arange(0, 10, 1 / 30.0), unit="s")
        self.video_df = pd.DataFrame(
            {
                "timestamp": video_timestamps,
                "RGB_R": np.random.randn(len(video_timestamps)),
            }
        )

    def test_align_signals(self):
        """Test that signals are correctly aligned and interpolated."""
        aligned_df = align_signals(self.gsr_df, self.video_df)

        self.assertIsInstance(aligned_df, pd.DataFrame)
        self.assertFalse(aligned_df.empty)

        # The aligned DataFrame should have the same number of rows as the target GSR df
        self.assertEqual(len(aligned_df), len(self.gsr_df))

        # Check that there are no NaN values after interpolation
        self.assertFalse(aligned_df.isnull().values.any())

        # The timestamps in the aligned df should match the gsr df's timestamps
        pd.testing.assert_series_equal(
            aligned_df["timestamp"].reset_index(drop=True),
            self.gsr_df["timestamp"].reset_index(drop=True),
            check_names=False,
        )

    def test_create_feature_windows(self):
        """Test that the windowing function produces correctly shaped outputs."""
        # Create a simple, perfectly aligned DataFrame for testing
        timestamps = pd.to_datetime(np.arange(100), unit="s")
        test_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.arange(100),
                "target": np.arange(100) + 1,
            }
        )

        window_size = 10
        step = 5
        feature_cols = ["feature1"]
        target_col = "target"

        X, y = create_feature_windows(
            test_df, feature_cols, target_col, window_size, step
        )

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        # Check shapes
        # Expected number of windows = floor((total_samples - window_size) / step)
        # The loop in create_feature_windows stops before i + window_size exceeds num_rows
        expected_num_windows = (100 - window_size) // step
        self.assertEqual(
            X.shape, (expected_num_windows, window_size, len(feature_cols))
        )
        self.assertEqual(y.shape, (expected_num_windows,))

        # Check content of the first window
        # The first window of X should be features from index 0 to 9
        expected_X0 = test_df[feature_cols].iloc[0:10].values
        np.testing.assert_array_equal(X[0], expected_X0)

        # The first target y should be the target value at the end of the first window (index 9)
        expected_y0 = test_df[target_col].iloc[9]
        self.assertEqual(y[0], expected_y0)

    def test_create_feature_windows_no_overlap(self):
        """Test windowing with a step size equal to the window size."""
        timestamps = pd.to_datetime(np.arange(100), unit="s")
        test_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.arange(100),
                "target": np.arange(100),
            }
        )

        window_size = 10
        step = 10  # No overlap
        X, y = create_feature_windows(
            test_df, ["feature1"], "target", window_size, step
        )

        # Expected number of windows = floor((total_samples - window_size) / step)
        # For non-overlapping windows, this is (100 - 10) // 10 = 9
        expected_num_windows = (100 - window_size) // step
        self.assertEqual(X.shape[0], expected_num_windows)

    def test_align_signals_exception_handling(self):
        """Test that align_signals handles exceptions properly."""
        # Create invalid dataframes (missing timestamp column)
        invalid_gsr_df = pd.DataFrame({"GSR_Phasic": [1, 2, 3]})
        invalid_video_df = pd.DataFrame({"RGB_R": [4, 5, 6]})

        # Test with invalid GSR dataframe
        result = align_signals(invalid_gsr_df, self.video_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        # Test with invalid video dataframe
        result = align_signals(self.gsr_df, invalid_video_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        # Test with both invalid dataframes
        result = align_signals(invalid_gsr_df, invalid_video_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    @patch('src.processing.feature_engineering.detect_palm_roi')
    @patch('src.processing.feature_engineering.extract_roi_signal')
    @patch('src.processing.feature_engineering.align_signals')
    @patch('src.processing.feature_engineering.create_feature_windows')
    def test_create_dataset_from_session(self, mock_create_windows, mock_align, 
                                         mock_extract_roi, mock_detect_roi, 
                                         mock_process_gsr, mock_loader):
        """Test the create_dataset_from_session function."""
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

        # Mock aligned data
        aligned_data = pd.DataFrame({
            "timestamp": gsr_timestamps,
            "GSR_Phasic": np.random.randn(len(gsr_timestamps)),
            "GSR_Tonic": np.random.randn(len(gsr_timestamps)),
            "RGB_B": np.random.randn(len(gsr_timestamps)),
            "RGB_G": np.random.randn(len(gsr_timestamps)),
            "RGB_R": np.random.randn(len(gsr_timestamps))
        })
        mock_align.return_value = aligned_data

        # Mock windowed data
        X = np.random.randn(5, 10, 4)  # 5 windows, 10 timesteps, 4 features
        y = np.random.randn(5)  # 5 target values
        mock_create_windows.return_value = (X, y)

        # Call the function
        result = create_dataset_from_session(session_path, gsr_sampling_rate, video_fps)

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check that the mocks were called with appropriate arguments
        mock_loader.assert_called_once_with(session_path)
        mock_loader_instance.get_gsr_data.assert_called_once()
        mock_process_gsr.assert_called_once_with(mock_gsr_df, sampling_rate=gsr_sampling_rate)
        mock_loader_instance.get_rgb_video_generator.assert_called_once()

        # Check that the result matches the expected output
        X_result, y_result = result
        np.testing.assert_array_equal(X_result, X)
        np.testing.assert_array_equal(y_result, y)

    @patch('src.processing.feature_engineering.SessionDataLoader')
    def test_create_dataset_from_session_no_gsr_data(self, mock_loader):
        """Test create_dataset_from_session when no GSR data is available."""
        # Setup mocks
        session_path = Path("dummy/path")
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.get_gsr_data.return_value = None

        # Call the function
        result = create_dataset_from_session(session_path, 32, 30)

        # Assertions
        self.assertIsNone(result)
        mock_loader.assert_called_once_with(session_path)
        mock_loader_instance.get_gsr_data.assert_called_once()

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    def test_create_dataset_from_session_gsr_processing_failed(self, mock_process_gsr, mock_loader):
        """Test create_dataset_from_session when GSR processing fails."""
        # Setup mocks
        session_path = Path("dummy/path")
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.get_gsr_data.return_value = pd.DataFrame({"GSR_Raw": [1, 2, 3]})
        mock_process_gsr.return_value = None

        # Call the function
        result = create_dataset_from_session(session_path, 32, 30)

        # Assertions
        self.assertIsNone(result)
        mock_loader.assert_called_once_with(session_path)
        mock_loader_instance.get_gsr_data.assert_called_once()
        mock_process_gsr.assert_called_once()

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    @patch('src.processing.feature_engineering.detect_palm_roi')
    def test_create_dataset_from_session_no_video_features(self, mock_detect_roi, mock_process_gsr, mock_loader):
        """Test create_dataset_from_session when no video features can be extracted."""
        # Setup mocks
        session_path = Path("dummy/path")
        gsr_sampling_rate = 32

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
        mock_detect_roi.return_value = None  # No ROI detected

        # Call the function
        result = create_dataset_from_session(session_path, gsr_sampling_rate, 30)

        # Assertions
        self.assertIsNone(result)
        mock_loader.assert_called_once_with(session_path)
        mock_loader_instance.get_gsr_data.assert_called_once()
        mock_process_gsr.assert_called_once()
        mock_loader_instance.get_rgb_video_generator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
