# src/tests/test_feature_engineering.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.processing.feature_engineering import align_signals, create_feature_windows


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
        # Expected number of windows = floor((total_samples - window_size) / step) + 1
        expected_num_windows = (100 - window_size) // step + 1
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

        expected_num_windows = 100 // window_size
        self.assertEqual(X.shape[0], expected_num_windows)


if __name__ == "__main__":
    unittest.main()
