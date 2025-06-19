import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.system.data_generation.create_mock_data import (
    generate_mock_physiological_data,
    generate_mock_hand_video,
    generate_mock_thermal_video,
    main
)


class TestCreateMockData(unittest.TestCase):
    """Test suite for the create_mock_data.py module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.duration = 5  # seconds
        self.sampling_rate = 32  # Hz
        self.fps = 30
        self.test_dir = Path("test_output")
    
    @patch('src.system.data_generation.create_mock_data.nk')
    def test_generate_mock_physiological_data(self, mock_nk):
        """Test that mock physiological data is generated correctly."""
        # Mock the neurokit2 functions
        mock_nk.ppg_simulate.return_value = np.sin(np.linspace(0, 10, self.duration * self.sampling_rate))
        mock_nk.eda_simulate.return_value = np.cos(np.linspace(0, 10, self.duration * self.sampling_rate))
        
        # Call the function
        result = generate_mock_physiological_data(self.duration, self.sampling_rate)
        
        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), self.duration * self.sampling_rate)
        self.assertIn("timestamp", result.columns)
        self.assertIn("ppg_value", result.columns)
        self.assertIn("gsr_value", result.columns)
        
        # Check that the neurokit2 functions were called with the correct parameters
        mock_nk.ppg_simulate.assert_called_once_with(
            duration=self.duration, 
            sampling_rate=self.sampling_rate, 
            heart_rate=75
        )
        mock_nk.eda_simulate.assert_called_once_with(
            duration=self.duration, 
            sampling_rate=self.sampling_rate, 
            scr_number=5,
            drift=0.1
        )
    
    @patch('src.system.data_generation.create_mock_data.nk')
    def test_generate_mock_physiological_data_exception(self, mock_nk):
        """Test that an empty DataFrame is returned when an exception occurs."""
        # Mock the neurokit2 functions to raise an exception
        mock_nk.ppg_simulate.side_effect = Exception("Test exception")
        
        # Call the function
        result = generate_mock_physiological_data(self.duration, self.sampling_rate)
        
        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        self.assertIn("timestamp", result.columns)
        self.assertIn("ppg_value", result.columns)
        self.assertIn("gsr_value", result.columns)
    
    @patch('src.system.data_generation.create_mock_data.cv2')
    @patch('src.system.data_generation.create_mock_data.config')
    def test_generate_mock_hand_video(self, mock_config, mock_cv2):
        """Test that mock hand video is generated correctly."""
        # Mock the config values
        mock_config.FRAME_WIDTH = 640
        mock_config.FRAME_HEIGHT = 480
        mock_config.VIDEO_FOURCC = 'mp4v'
        
        # Mock the cv2 functions
        mock_video_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_video_writer
        mock_video_writer.isOpened.return_value = True
        mock_cv2.VideoWriter_fourcc.return_value = 828601953  # A random integer
        
        # Create test data
        ppg_signal = np.sin(np.linspace(0, 10, self.duration * self.sampling_rate))
        video_path = self.test_dir / "test_rgb.mp4"
        
        # Call the function
        generate_mock_hand_video(video_path, ppg_signal, self.duration, self.fps)
        
        # Check that the video writer was created with the correct parameters
        mock_cv2.VideoWriter.assert_called_once_with(
            str(video_path), 
            828601953, 
            float(self.fps), 
            (640, 480)
        )
        
        # Check that the video writer's write method was called the correct number of times
        self.assertEqual(mock_video_writer.write.call_count, self.duration * self.fps)
        
        # Check that the video writer's release method was called
        mock_video_writer.release.assert_called_once()
    
    @patch('src.system.data_generation.create_mock_data.cv2')
    @patch('src.system.data_generation.create_mock_data.config')
    def test_generate_mock_hand_video_exception(self, mock_config, mock_cv2):
        """Test that exceptions are handled when generating mock hand video."""
        # Mock the config values
        mock_config.FRAME_WIDTH = 640
        mock_config.FRAME_HEIGHT = 480
        mock_config.VIDEO_FOURCC = 'mp4v'
        
        # Mock the cv2 functions to raise an exception
        mock_cv2.VideoWriter.side_effect = Exception("Test exception")
        
        # Create test data
        ppg_signal = np.sin(np.linspace(0, 10, self.duration * self.sampling_rate))
        video_path = self.test_dir / "test_rgb.mp4"
        
        # Call the function - should not raise an exception
        generate_mock_hand_video(video_path, ppg_signal, self.duration, self.fps)
    
    @patch('src.system.data_generation.create_mock_data.cv2')
    @patch('src.system.data_generation.create_mock_data.config')
    def test_generate_mock_thermal_video(self, mock_config, mock_cv2):
        """Test that mock thermal video is generated correctly."""
        # Mock the config values
        mock_config.FRAME_WIDTH = 640
        mock_config.FRAME_HEIGHT = 480
        mock_config.VIDEO_FOURCC = 'mp4v'
        
        # Mock the cv2 functions
        mock_video_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_video_writer
        mock_video_writer.isOpened.return_value = True
        mock_cv2.VideoWriter_fourcc.return_value = 828601953  # A random integer
        mock_cv2.COLORMAP_INFERNO = 9  # A random integer
        
        # Create test data
        gsr_signal = np.cos(np.linspace(0, 10, self.duration * self.sampling_rate))
        video_path = self.test_dir / "test_thermal.mp4"
        
        # Call the function
        generate_mock_thermal_video(video_path, gsr_signal, self.duration, self.fps)
        
        # Check that the video writer was created with the correct parameters
        mock_cv2.VideoWriter.assert_called_once_with(
            str(video_path), 
            828601953, 
            float(self.fps), 
            (640, 480)
        )
        
        # Check that the video writer's write method was called the correct number of times
        self.assertEqual(mock_video_writer.write.call_count, self.duration * self.fps)
        
        # Check that the video writer's release method was called
        mock_video_writer.release.assert_called_once()
    
    @patch('src.system.data_generation.create_mock_data.cv2')
    @patch('src.system.data_generation.create_mock_data.config')
    def test_generate_mock_thermal_video_exception(self, mock_config, mock_cv2):
        """Test that exceptions are handled when generating mock thermal video."""
        # Mock the config values
        mock_config.FRAME_WIDTH = 640
        mock_config.FRAME_HEIGHT = 480
        mock_config.VIDEO_FOURCC = 'mp4v'
        
        # Mock the cv2 functions to raise an exception
        mock_cv2.VideoWriter.side_effect = Exception("Test exception")
        
        # Create test data
        gsr_signal = np.cos(np.linspace(0, 10, self.duration * self.sampling_rate))
        video_path = self.test_dir / "test_thermal.mp4"
        
        # Call the function - should not raise an exception
        generate_mock_thermal_video(video_path, gsr_signal, self.duration, self.fps)
    
    @patch('src.system.data_generation.create_mock_data.generate_mock_physiological_data')
    @patch('src.system.data_generation.create_mock_data.generate_mock_hand_video')
    @patch('src.system.data_generation.create_mock_data.generate_mock_thermal_video')
    @patch('src.system.data_generation.create_mock_data.config')
    def test_main(self, mock_config, mock_thermal, mock_hand, mock_phys):
        """Test that the main function orchestrates the mock data generation correctly."""
        # Mock the config values
        mock_config.OUTPUT_DIR = Path("data/recordings")
        mock_config.GSR_SAMPLING_RATE = self.sampling_rate
        mock_config.FPS = self.fps
        
        # Mock the physiological data generation
        mock_phys_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=self.duration * self.sampling_rate, freq=f"{1000/self.sampling_rate}ms"),
            "ppg_value": np.sin(np.linspace(0, 10, self.duration * self.sampling_rate)),
            "gsr_value": np.cos(np.linspace(0, 10, self.duration * self.sampling_rate))
        })
        mock_phys.return_value = mock_phys_df
        
        # Call the main function
        with patch('src.system.data_generation.create_mock_data.pd.Timestamp') as mock_timestamp:
            mock_timestamp.now.return_value.strftime.return_value = "20230101_120000"
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                result = main()
        
        # Check that the directory was created
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Check that the physiological data was generated
        mock_phys.assert_called_once_with(30, self.sampling_rate)
        
        # Check that the videos were generated
        mock_hand.assert_called_once()
        mock_thermal.assert_called_once()
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(result, Path("data/recordings/Subject_MockSubject01_20230101_120000"))
    
    @patch('src.system.data_generation.create_mock_data.generate_mock_physiological_data')
    @patch('src.system.data_generation.create_mock_data.config')
    def test_main_exception(self, mock_config, mock_phys):
        """Test that the main function handles exceptions."""
        # Mock the config values
        mock_config.OUTPUT_DIR = Path("data/recordings")
        mock_config.GSR_SAMPLING_RATE = self.sampling_rate
        
        # Mock the physiological data generation to raise an exception
        mock_phys.side_effect = Exception("Test exception")
        
        # Call the main function
        result = main()
        
        # Check the result
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()