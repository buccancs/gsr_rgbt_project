# src/tests/unit/test_cython_optimizations.py

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Try to import Cython optimizations
try:
    from src.processing.cython_optimizations import cy_align_signals, cy_create_feature_windows
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

from src.processing.feature_engineering import align_signals, create_feature_windows


@unittest.skipIf(not CYTHON_AVAILABLE, "Cython optimizations not available")
class TestCythonOptimizations(unittest.TestCase):
    """
    Test suite for the Cython optimizations.

    These tests verify that the Cython implementations produce the same results
    as the Python implementations.
    """

    def setUp(self):
        """Set up data that mimics processed GSR and video signals."""
        # Create a high-frequency signal (like GSR at 32Hz)
        gsr_timestamps = pd.to_datetime(np.arange(0, 10, 1 / 32.0), unit="s")
        self.gsr_df = pd.DataFrame(
            {
                "timestamp": gsr_timestamps,
                "GSR_Phasic": np.random.randn(len(gsr_timestamps)),
                "GSR_Tonic": np.random.randn(len(gsr_timestamps)),
            }
        )

        # Create a lower-frequency signal (like video features at 30Hz)
        video_timestamps = pd.to_datetime(np.arange(0, 10, 1 / 30.0), unit="s")
        self.video_df = pd.DataFrame(
            {
                "timestamp": video_timestamps,
                "RGB_R": np.random.randn(len(video_timestamps)),
                "RGB_G": np.random.randn(len(video_timestamps)),
                "RGB_B": np.random.randn(len(video_timestamps)),
            }
        )

        # Create a simple, perfectly aligned DataFrame for windowing tests
        timestamps = pd.to_datetime(np.arange(100), unit="s")
        self.test_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.arange(100),
                "feature2": np.arange(100) * 2,
                "target": np.arange(100) + 1,
            }
        )

    def test_cy_align_signals(self):
        """Test that the Cython implementation of align_signals produces the same results as the Python implementation."""
        # Call the Python implementation
        py_aligned_df = align_signals(self.gsr_df, self.video_df)

        # Extract the data needed for the Cython implementation
        gsr_data = self.gsr_df.drop(columns=["timestamp"]).values
        video_data = self.video_df.drop(columns=["timestamp"]).values
        gsr_timestamps = self.gsr_df["timestamp"].astype(np.int64).values
        video_timestamps = self.video_df["timestamp"].astype(np.int64).values

        # Call the Cython implementation directly
        cy_aligned_data = cy_align_signals(gsr_data, video_data, gsr_timestamps, video_timestamps)

        # Convert the Cython result to a DataFrame for comparison
        cy_aligned_df = pd.DataFrame(
            cy_aligned_data,
            columns=list(self.gsr_df.columns[1:]) + list(self.video_df.columns[1:])
        )
        cy_aligned_df["timestamp"] = self.gsr_df["timestamp"].reset_index(drop=True)

        # Check that the results are the same
        pd.testing.assert_frame_equal(
            py_aligned_df.reset_index(drop=True),
            cy_aligned_df.reset_index(drop=True),
            check_dtype=False,  # Allow different dtypes
            check_exact=False,  # Allow small numerical differences
            rtol=1e-5,  # Relative tolerance
            atol=1e-8,  # Absolute tolerance
        )

    def test_cy_create_feature_windows(self):
        """Test that the Cython implementation of create_feature_windows produces the same results as the Python implementation."""
        window_size = 10
        step = 5
        feature_cols = ["feature1", "feature2"]
        target_col = "target"

        # Call the Python implementation
        py_X, py_y = create_feature_windows(
            self.test_df, feature_cols, target_col, window_size, step
        )

        # Extract the data needed for the Cython implementation
        features = self.test_df[feature_cols].values
        targets = self.test_df[target_col].values
        feature_cols_idx = [self.test_df.columns.get_loc(col) - 1 for col in feature_cols]  # Adjust for timestamp column

        # Call the Cython implementation directly
        cy_X, cy_y = cy_create_feature_windows(
            features, targets, feature_cols_idx, window_size, step
        )

        # Check that the results are the same
        np.testing.assert_allclose(py_X, cy_X, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(py_y, cy_y, rtol=1e-5, atol=1e-8)

    def test_cy_align_signals_edge_cases(self):
        """Test the Cython implementation of align_signals with edge cases."""
        # Test with empty DataFrames
        empty_gsr_df = pd.DataFrame(columns=["timestamp", "GSR_Phasic"])
        empty_video_df = pd.DataFrame(columns=["timestamp", "RGB_R"])

        # Call the Python implementation
        py_aligned_df = align_signals(empty_gsr_df, empty_video_df)

        # Create empty arrays for direct testing of Cython function
        empty_gsr_data = np.array([]).reshape(0, 1)
        empty_video_data = np.array([]).reshape(0, 1)
        empty_gsr_timestamps = np.array([], dtype=np.int64)
        empty_video_timestamps = np.array([], dtype=np.int64)

        # Test with both arrays empty
        cy_aligned_data = cy_align_signals(empty_gsr_data, empty_video_data, empty_gsr_timestamps, empty_video_timestamps)
        self.assertEqual(cy_aligned_data.shape[0], 0, "Result should be empty when both inputs are empty")

        # Test with GSR data empty but video data present
        video_data = self.video_df.drop(columns=["timestamp"]).values
        video_timestamps = self.video_df["timestamp"].astype(np.int64).values
        cy_aligned_data = cy_align_signals(empty_gsr_data, video_data, empty_gsr_timestamps, video_timestamps)
        self.assertEqual(cy_aligned_data.shape[0], 0, "Result should be empty when GSR data is empty")

        # Test with video data empty but GSR data present
        gsr_data = self.gsr_df.drop(columns=["timestamp"]).values
        gsr_timestamps = self.gsr_df["timestamp"].astype(np.int64).values
        cy_aligned_data = cy_align_signals(gsr_data, empty_video_data, gsr_timestamps, empty_video_timestamps)
        self.assertEqual(cy_aligned_data.shape[0], gsr_data.shape[0], "Result should have GSR data when video data is empty")

        # Test with GSR timestamps outside the range of video timestamps
        # Create GSR timestamps that are all before the video timestamps
        early_gsr_timestamps = gsr_timestamps - (video_timestamps.max() - video_timestamps.min()) * 2
        cy_aligned_data = cy_align_signals(gsr_data, video_data, early_gsr_timestamps, video_timestamps)
        self.assertEqual(cy_aligned_data.shape[0], gsr_data.shape[0], "Result should handle GSR timestamps before video timestamps")

        # Create GSR timestamps that are all after the video timestamps
        late_gsr_timestamps = gsr_timestamps + (video_timestamps.max() - video_timestamps.min()) * 2
        cy_aligned_data = cy_align_signals(gsr_data, video_data, late_gsr_timestamps, video_timestamps)
        self.assertEqual(cy_aligned_data.shape[0], gsr_data.shape[0], "Result should handle GSR timestamps after video timestamps")

    def test_cy_create_feature_windows_edge_cases(self):
        """Test the Cython implementation of create_feature_windows with edge cases."""
        # Test with window_size > data length
        window_size = 200  # Larger than the test_df length
        step = 10
        feature_cols = ["feature1"]
        target_col = "target"

        # Call the Python implementation
        py_X, py_y = create_feature_windows(
            self.test_df, feature_cols, target_col, window_size, step
        )

        # Extract the data needed for the Cython implementation
        features = self.test_df[feature_cols].values
        targets = self.test_df[target_col].values
        feature_cols_idx = [self.test_df.columns.get_loc(col) - 1 for col in feature_cols]  # Adjust for timestamp column

        # Call the Cython implementation directly
        cy_X, cy_y = cy_create_feature_windows(
            features, targets, feature_cols_idx, window_size, step
        )

        # Check that the results are the same
        np.testing.assert_allclose(py_X, cy_X, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(py_y, cy_y, rtol=1e-5, atol=1e-8)

        # Both should return empty arrays since window_size > data length
        self.assertEqual(py_X.shape[0], 0)
        self.assertEqual(cy_X.shape[0], 0)


if __name__ == "__main__":
    unittest.main()