# src/tests/test_data_loader.py

import shutil

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.processing.data_loader import SessionDataLoader


class TestSessionDataLoader(unittest.TestCase):
    """
    Test suite for the SessionDataLoader and its associated functions.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a temporary test directory and dummy data once for all tests."""
        cls.test_dir = Path("./temp_test_session_data")
        cls.test_dir.mkdir(exist_ok=True)

        # Create dummy GSR CSV file
        gsr_data = {
            "timestamp": [pd.Timestamp("2025-01-01 12:00:00.000")],
            "gsr_value": [0.5],
        }
        pd.DataFrame(gsr_data).to_csv(cls.test_dir / "gsr_data.csv", index=False)

        # Create dummy video files
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for video_name in ["rgb_video.mp4", "thermal_video.mp4"]:
            out = cv2.VideoWriter(
                str(cls.test_dir / video_name), fourcc, 30.0, (10, 10)
            )
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            out.write(frame)
            out.release()

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests are done."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_loader_initialization(self):
        """Test that the SessionDataLoader initializes correctly with a valid path."""
        try:
            loader = SessionDataLoader(self.test_dir)
            self.assertIsNotNone(loader)
        except FileNotFoundError:
            self.fail("SessionDataLoader raised FileNotFoundError unexpectedly.")

    def test_loader_initialization_invalid_path(self):
        """Test that the loader raises an error for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            SessionDataLoader(Path("./non_existent_path"))

    def test_get_gsr_data_success(self):
        """Test successful loading and parsing of GSR data."""
        loader = SessionDataLoader(self.test_dir)
        gsr_df = loader.get_gsr_data()
        self.assertIsInstance(gsr_df, pd.DataFrame)
        self.assertFalse(gsr_df.empty)
        self.assertIn("timestamp", gsr_df.columns)
        self.assertIn("gsr_value", gsr_df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(gsr_df["timestamp"]))

    def test_get_gsr_data_file_not_found(self):
        """Test that get_gsr_data returns None if the CSV is missing."""
        # Create a temporary empty directory
        empty_dir = self.test_dir / "empty_session"
        empty_dir.mkdir()
        loader = SessionDataLoader(empty_dir)
        self.assertIsNone(loader.get_gsr_data())
        shutil.rmtree(empty_dir)

    def test_get_video_generators(self):
        """Test that video generators are created successfully."""
        loader = SessionDataLoader(self.test_dir)

        rgb_gen = loader.get_rgb_video_generator()
        thermal_gen = loader.get_thermal_video_generator()

        self.assertIsNotNone(rgb_gen)
        self.assertIsNotNone(thermal_gen)

        # Check that we can read at least one frame
        rgb_success, rgb_frame = next(rgb_gen)
        thermal_success, thermal_frame = next(thermal_gen)

        self.assertTrue(rgb_success)
        self.assertIsInstance(rgb_frame, np.ndarray)
        self.assertTrue(thermal_success)
        self.assertIsInstance(thermal_frame, np.ndarray)


if __name__ == "__main__":
    unittest.main()
