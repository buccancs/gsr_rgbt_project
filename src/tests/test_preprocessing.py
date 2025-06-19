# src/tests/test_preprocessing.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.processing.preprocessing import (
    process_gsr_signal,
    detect_palm_roi,
    extract_roi_signal,
)


class TestPreprocessing(unittest.TestCase):
    """
    Test suite for the signal and video preprocessing functions.
    """

    def setUp(self):
        """Set up common data for tests."""
        # Create a realistic dummy GSR signal (10 seconds at 32Hz)
        self.sampling_rate = 32
        self.num_samples = 10 * self.sampling_rate
        eda_raw = nk.eda_simulate(
            duration=10, sampling_rate=self.sampling_rate, scr_number=3
        )
        timestamps = pd.to_datetime(
            np.arange(self.num_samples) / self.sampling_rate, unit="s"
        )
        self.gsr_df = pd.DataFrame({"timestamp": timestamps, "gsr_value": eda_raw})

        # Create a dummy video frame
        self.dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_process_gsr_signal_success(self):
        """Test that GSR processing returns a DataFrame with the correct columns."""
        processed_df = process_gsr_signal(self.gsr_df, self.sampling_rate)

        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertFalse(processed_df.empty)

        expected_columns = [
            "timestamp",
            "gsr_value",
            "GSR_Clean",
            "GSR_Tonic",
            "GSR_Phasic",
        ]
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)

        self.assertEqual(len(processed_df), self.num_samples)

    def test_process_gsr_signal_missing_column(self):
        """Test that the function handles a missing 'gsr_value' column gracefully."""
        invalid_df = pd.DataFrame({"timestamp": self.gsr_df["timestamp"]})
        result = process_gsr_signal(invalid_df, self.sampling_rate)
        self.assertIsNone(result)

    def test_detect_palm_roi(self):
        """Test that the palm ROI detection returns a valid bounding box tuple."""
        roi = detect_palm_roi(self.dummy_frame)

        self.assertIsInstance(roi, tuple)
        self.assertEqual(len(roi), 4)

        x, y, w, h = roi
        self.assertTrue(all(isinstance(v, int) for v in [x, y, w, h]))
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_extract_roi_signal(self):
        """Test that ROI signal extraction returns a valid RGB numpy array."""
        # Create a frame with a known region of color
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Make a 10x10 red square at (20, 20)
        test_frame[20:30, 20:30] = [0, 0, 255]  # BGR format
        roi = (20, 20, 10, 10)

        signal = extract_roi_signal(test_frame, roi)

        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.shape, (3,))

        # The mean should be [0, 0, 255] since the ROI is pure red
        np.testing.assert_array_equal(signal, np.array([0, 0, 255]))

    def test_extract_roi_signal_mixed_colors(self):
        """Test ROI signal extraction with mixed colors."""
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        # Top half blue, bottom half red
        test_frame[0:5, :] = [200, 0, 0]  # Blue
        test_frame[5:10, :] = [0, 0, 100]  # Red
        roi = (0, 0, 10, 10)  # Full frame

        signal = extract_roi_signal(test_frame, roi)

        # Expected mean: B=100, G=0, R=50
        expected_signal = np.array([100, 0, 50])
        np.testing.assert_array_almost_equal(signal, expected_signal)


if __name__ == "__main__":
    unittest.main()
