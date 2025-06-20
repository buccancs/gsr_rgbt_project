# src/tests/unit/data_collection/utils/test_data_logger.py

import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import tempfile
import os
from pathlib import Path
import csv
import numpy as np
import cv2

from src.data_collection.utils.data_logger import DataLogger

class TestDataLogger(unittest.TestCase):
    """
    Unit tests for the DataLogger class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Set up test parameters
        self.subject_id = "TestSubject01"
        self.fps = 30
        self.video_fourcc = "MJPG"
        
        # Create the DataLogger instance
        self.logger = DataLogger(
            output_dir=self.output_dir,
            subject_id=self.subject_id,
            fps=self.fps,
            video_fourcc=self.video_fourcc
        )
        
    def tearDown(self):
        """
        Clean up after each test method.
        """
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """
        Test that the DataLogger initializes correctly.
        """
        # Check that the parameters were stored correctly
        self.assertEqual(self.logger.output_dir, self.output_dir)
        self.assertEqual(self.logger.subject_id, self.subject_id)
        self.assertEqual(self.logger.fps, self.fps)
        self.assertEqual(self.logger.video_fourcc, self.video_fourcc)
        
        # Check that the video writers and CSV file are initially None
        self.assertIsNone(self.logger.rgb_writer)
        self.assertIsNone(self.logger.thermal_writer)
        self.assertIsNone(self.logger.gsr_csv_file)
        self.assertIsNone(self.logger.timestamps_csv_file)
        
    @patch('src.data_collection.utils.data_logger.cv2.VideoWriter')
    @patch('src.data_collection.utils.data_logger.Path.mkdir')
    @patch('src.data_collection.utils.data_logger.open', new_callable=mock_open)
    @patch('src.data_collection.utils.data_logger.csv.writer')
    def test_start_logging(self, mock_csv_writer, mock_open, mock_mkdir, mock_video_writer):
        """
        Test that start_logging creates the necessary directories and files.
        """
        # Set up mock return values
        mock_video_writer_instance = MagicMock()
        mock_video_writer.return_value = mock_video_writer_instance
        
        mock_csv_writer_instance = MagicMock()
        mock_csv_writer.return_value = mock_csv_writer_instance
        
        # Call start_logging
        frame_size_rgb = (640, 480)
        frame_size_thermal = (320, 240)
        self.logger.start_logging(frame_size_rgb, frame_size_thermal)
        
        # Check that the directory was created
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)
        
        # Check that the video writers were created with the correct parameters
        mock_video_writer.assert_any_call(
            str(self.output_dir / f"{self.subject_id}_rgb.avi"),
            cv2.VideoWriter_fourcc(*self.video_fourcc),
            self.fps,
            frame_size_rgb
        )
        mock_video_writer.assert_any_call(
            str(self.output_dir / f"{self.subject_id}_thermal.avi"),
            cv2.VideoWriter_fourcc(*self.video_fourcc),
            self.fps,
            frame_size_thermal
        )
        
        # Check that the CSV files were created
        mock_open.assert_any_call(self.output_dir / f"{self.subject_id}_gsr.csv", 'w', newline='')
        mock_open.assert_any_call(self.output_dir / f"{self.subject_id}_timestamps.csv", 'w', newline='')
        
        # Check that the CSV writers were created and headers were written
        self.assertEqual(mock_csv_writer.call_count, 2)
        mock_csv_writer_instance.writerow.assert_any_call(['timestamp', 'gsr_value'])
        mock_csv_writer_instance.writerow.assert_any_call(['frame_number', 'rgb_timestamp', 'thermal_timestamp'])
        
    @patch('src.data_collection.utils.data_logger.cv2.VideoWriter')
    @patch('src.data_collection.utils.data_logger.Path.mkdir')
    @patch('src.data_collection.utils.data_logger.open', new_callable=mock_open)
    @patch('src.data_collection.utils.data_logger.csv.writer')
    def test_cleanup_partial_initialization(self, mock_csv_writer, mock_open, mock_mkdir, mock_video_writer):
        """
        Test that _cleanup_partial_initialization properly cleans up resources.
        """
        # Create mock objects
        mock_rgb_writer = MagicMock()
        mock_thermal_writer = MagicMock()
        mock_gsr_csv_file = MagicMock()
        mock_timestamps_csv_file = MagicMock()
        
        # Set the mock objects on the logger
        self.logger.rgb_writer = mock_rgb_writer
        self.logger.thermal_writer = mock_thermal_writer
        self.logger.gsr_csv_file = mock_gsr_csv_file
        self.logger.timestamps_csv_file = mock_timestamps_csv_file
        
        # Call _cleanup_partial_initialization with different combinations
        self.logger._cleanup_partial_initialization(True, False, False)
        mock_rgb_writer.release.assert_called_once()
        mock_thermal_writer.release.assert_not_called()
        mock_gsr_csv_file.close.assert_not_called()
        mock_timestamps_csv_file.close.assert_not_called()
        
        # Reset the mocks
        mock_rgb_writer.reset_mock()
        
        self.logger._cleanup_partial_initialization(True, True, False)
        mock_rgb_writer.release.assert_called_once()
        mock_thermal_writer.release.assert_called_once()
        mock_gsr_csv_file.close.assert_not_called()
        mock_timestamps_csv_file.close.assert_not_called()
        
        # Reset the mocks
        mock_rgb_writer.reset_mock()
        mock_thermal_writer.reset_mock()
        
        self.logger._cleanup_partial_initialization(True, True, True)
        mock_rgb_writer.release.assert_called_once()
        mock_thermal_writer.release.assert_called_once()
        mock_gsr_csv_file.close.assert_called_once()
        mock_timestamps_csv_file.close.assert_called_once()
        
    def test_log_rgb_frame(self):
        """
        Test that log_rgb_frame correctly logs an RGB frame.
        """
        # Create a mock RGB writer
        mock_rgb_writer = MagicMock()
        self.logger.rgb_writer = mock_rgb_writer
        
        # Create a mock timestamps CSV writer
        mock_timestamps_writer = MagicMock()
        self.logger.timestamps_writer = mock_timestamps_writer
        
        # Create a test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8)
        timestamp = 1234567890.123
        frame_number = 42
        
        # Call log_rgb_frame
        self.logger.log_rgb_frame(frame, timestamp, frame_number)
        
        # Check that the frame was written to the video writer
        mock_rgb_writer.write.assert_called_with(frame)
        
        # Check that the timestamp was written to the CSV file
        mock_timestamps_writer.writerow.assert_called_with([frame_number, timestamp, None])
        
    def test_log_thermal_frame(self):
        """
        Test that log_thermal_frame correctly logs a thermal frame.
        """
        # Create a mock thermal writer
        mock_thermal_writer = MagicMock()
        self.logger.thermal_writer = mock_thermal_writer
        
        # Create a mock timestamps CSV writer
        mock_timestamps_writer = MagicMock()
        self.logger.timestamps_writer = mock_timestamps_writer
        
        # Create a test frame
        frame = np.ones((240, 320), dtype=np.uint8)
        timestamp = 1234567890.123
        frame_number = 42
        
        # Call log_thermal_frame
        self.logger.log_thermal_frame(frame, timestamp, frame_number)
        
        # Check that the frame was written to the video writer
        mock_thermal_writer.write.assert_called_with(frame)
        
        # Check that the timestamp was written to the CSV file
        mock_timestamps_writer.writerow.assert_called_with([frame_number, None, timestamp])
        
    def test_log_gsr_data(self):
        """
        Test that log_gsr_data correctly logs GSR data.
        """
        # Create a mock GSR CSV writer
        mock_gsr_writer = MagicMock()
        self.logger.gsr_writer = mock_gsr_writer
        
        # Create test data
        gsr_value = 0.75
        shimmer_timestamp = 1234567890.123
        
        # Call log_gsr_data
        self.logger.log_gsr_data(gsr_value, shimmer_timestamp)
        
        # Check that the data was written to the CSV file
        mock_gsr_writer.writerow.assert_called_with([shimmer_timestamp, gsr_value])
        
    def test_stop_logging(self):
        """
        Test that stop_logging properly releases all resources.
        """
        # Create mock objects
        mock_rgb_writer = MagicMock()
        mock_thermal_writer = MagicMock()
        mock_gsr_csv_file = MagicMock()
        mock_timestamps_csv_file = MagicMock()
        
        # Set the mock objects on the logger
        self.logger.rgb_writer = mock_rgb_writer
        self.logger.thermal_writer = mock_thermal_writer
        self.logger.gsr_csv_file = mock_gsr_csv_file
        self.logger.timestamps_csv_file = mock_timestamps_csv_file
        
        # Call stop_logging
        self.logger.stop_logging()
        
        # Check that all resources were released
        mock_rgb_writer.release.assert_called_once()
        mock_thermal_writer.release.assert_called_once()
        mock_gsr_csv_file.close.assert_called_once()
        mock_timestamps_csv_file.close.assert_called_once()
        
        # Check that all attributes were set to None
        self.assertIsNone(self.logger.rgb_writer)
        self.assertIsNone(self.logger.thermal_writer)
        self.assertIsNone(self.logger.gsr_csv_file)
        self.assertIsNone(self.logger.timestamps_csv_file)
        self.assertIsNone(self.logger.gsr_writer)
        self.assertIsNone(self.logger.timestamps_writer)

if __name__ == '__main__':
    unittest.main()