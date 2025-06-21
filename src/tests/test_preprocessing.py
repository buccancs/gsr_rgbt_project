# src/tests/test_preprocessing.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path
import logging

import neurokit2 as nk
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Try to import mediapipe, but don't fail if it's not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("Mediapipe is not available. Some tests will be skipped.")

# Import the functions we want to test
# If mediapipe is not available, some of these imports might fail
try:
    from src.ml_pipeline.preprocessing.preprocessing import (
        process_gsr_signal,
        detect_palm_roi,
        extract_roi_signal,
    )
except ImportError as e:
    if "mediapipe" in str(e):
        # If the error is related to mediapipe, we'll handle it in the tests
        from src.ml_pipeline.preprocessing.preprocessing import process_gsr_signal
        # Define dummy functions for the ones that depend on mediapipe
        def detect_palm_roi(frame):
            return (0, 0, 10, 10)
        def extract_roi_signal(frame, roi):
            return np.array([0, 0, 0])
    else:
        # If it's some other error, re-raise it
        raise


class TestPreprocessing(unittest.TestCase):
    """
    Test suite for the signal and video preprocessing functions.
    """

    def setUp(self):
        """Set up common data for tests."""
        # Create a realistic dummy GSR signal (10 seconds at 128Hz to match Shimmer data)
        self.sampling_rate = 128
        self.num_samples = 10 * self.sampling_rate

        # Generate the EDA signal and ensure it has the correct length
        eda_raw = nk.eda_simulate(
            duration=10, sampling_rate=self.sampling_rate, scr_number=3
        )

        # Ensure eda_raw has exactly self.num_samples elements
        if len(eda_raw) != self.num_samples:
            logging.warning(f"EDA signal length ({len(eda_raw)}) doesn't match expected length ({self.num_samples}). Adjusting...")
            if len(eda_raw) > self.num_samples:
                eda_raw = eda_raw[:self.num_samples]
            else:
                # Pad with the last value if too short
                eda_raw = np.pad(eda_raw, (0, self.num_samples - len(eda_raw)), 'edge')

        # Create timestamps with the same length as eda_raw
        timestamps = pd.to_datetime(
            np.arange(self.num_samples) / self.sampling_rate, unit="s"
        )

        self.gsr_df = pd.DataFrame({"timestamp": timestamps, "gsr_value": eda_raw})

        # Create a dummy PPG signal with the correct length
        ppg_raw = nk.ppg_simulate(
            duration=10, sampling_rate=self.sampling_rate, heart_rate=70
        )

        # Ensure ppg_raw has exactly self.num_samples elements
        if len(ppg_raw) != self.num_samples:
            logging.warning(f"PPG signal length ({len(ppg_raw)}) doesn't match expected length ({self.num_samples}). Adjusting...")
            if len(ppg_raw) > self.num_samples:
                ppg_raw = ppg_raw[:self.num_samples]
            else:
                # Pad with the last value if too short
                ppg_raw = np.pad(ppg_raw, (0, self.num_samples - len(ppg_raw)), 'edge')

        self.gsr_df_with_ppg = self.gsr_df.copy()
        self.gsr_df_with_ppg["ppg_value"] = ppg_raw

        # Create dummy heart rate data (derived from PPG)
        # Start with -1 for the first second (training period) then use realistic values
        hr_values = np.full(self.num_samples, -1.0)
        hr_values[self.sampling_rate:] = 70 + 5 * np.sin(
            np.linspace(0, 2 * np.pi, self.num_samples - self.sampling_rate)
        )
        self.gsr_df_with_ppg_hr = self.gsr_df_with_ppg.copy()
        self.gsr_df_with_ppg_hr["hr_value"] = hr_values

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
        if not MEDIAPIPE_AVAILABLE:
            self.skipTest("Mediapipe is not available, skipping test_detect_palm_roi")

        roi = detect_palm_roi(self.dummy_frame)

        self.assertIsInstance(roi, tuple)
        self.assertEqual(len(roi), 4)

        x, y, w, h = roi
        self.assertTrue(all(isinstance(v, int) for v in [x, y, w, h]))
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_extract_roi_signal(self):
        """Test that ROI signal extraction returns a valid RGB numpy array."""
        if not MEDIAPIPE_AVAILABLE:
            self.skipTest("Mediapipe is not available, skipping test_extract_roi_signal")

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
        if not MEDIAPIPE_AVAILABLE:
            self.skipTest("Mediapipe is not available, skipping test_extract_roi_signal_mixed_colors")

        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        # Top half blue, bottom half red
        test_frame[0:5, :] = [200, 0, 0]  # Blue
        test_frame[5:10, :] = [0, 0, 100]  # Red
        roi = (0, 0, 10, 10)  # Full frame

        signal = extract_roi_signal(test_frame, roi)

        # Expected mean: B=100, G=0, R=50
        expected_signal = np.array([100, 0, 50])
        np.testing.assert_array_almost_equal(signal, expected_signal)

    def test_process_gsr_signal_with_ppg(self):
        """Test that GSR processing with PPG data returns a DataFrame with the correct columns."""
        processed_df = process_gsr_signal(self.gsr_df_with_ppg, self.sampling_rate)

        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertFalse(processed_df.empty)

        expected_columns = [
            "timestamp",
            "gsr_value",
            "ppg_value",
            "GSR_Clean",
            "GSR_Tonic",
            "GSR_Phasic",
            "PPG_Clean",
            "PPG_Rate",
        ]
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)

        self.assertEqual(len(processed_df), self.num_samples)

    def test_process_gsr_signal_with_ppg_hr(self):
        """Test that GSR processing with PPG and HR data returns a DataFrame with the correct columns."""
        processed_df = process_gsr_signal(self.gsr_df_with_ppg_hr, self.sampling_rate)

        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertFalse(processed_df.empty)

        expected_columns = [
            "timestamp",
            "gsr_value",
            "ppg_value",
            "hr_value",
            "GSR_Clean",
            "GSR_Tonic",
            "GSR_Phasic",
            "PPG_Clean",
            "PPG_Rate",
            "HR_Value",
        ]
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)

        self.assertEqual(len(processed_df), self.num_samples)

        # Check that HR_Value matches the input hr_value
        np.testing.assert_array_equal(processed_df["HR_Value"].values, self.gsr_df_with_ppg_hr["hr_value"].values)


if __name__ == "__main__":
    unittest.main()
