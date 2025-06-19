import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Mock cv2 module
mock_cv2 = MagicMock()
mock_cv2.CAP_DSHOW = 700
mock_cv2.CAP_PROP_FPS = 5
sys.modules['cv2'] = mock_cv2

from src.capture.video_capture import VideoCaptureThread


class TestVideoCaptureThread(unittest.TestCase):
    """Test suite for the VideoCaptureThread class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_id = 0
        self.camera_name = "TestCamera"
        self.fps = 30
    
    def test_initialization(self):
        """Test that the thread initializes correctly."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        self.assertEqual(thread.device_name, self.camera_name)
        self.assertEqual(thread.camera_id, self.camera_id)
        self.assertEqual(thread.fps, self.fps)
        self.assertIsNone(thread.cap)
    
    @patch('src.capture.video_capture.sys.platform', 'win32')
    @patch('src.capture.video_capture.cv2')
    def test_run_real_capture_windows(self, mock_cv2):
        """Test that the thread runs correctly on Windows."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        # Mock the emit method to track calls
        thread.frame_captured = MagicMock()
        
        # Mock the get_current_timestamp method to return a fixed value
        thread.get_current_timestamp = MagicMock(return_value=12345)
        
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock the read method to return a valid frame once, then stop the thread
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        # Set is_running to True initially, then False after one iteration
        thread.is_running = True
        
        def read_side_effect():
            thread.is_running = False
            return (True, test_frame)
        
        mock_cap.read.side_effect = read_side_effect
        
        # Run the real capture method directly
        thread._run_real_capture()
        
        # Check that the VideoCapture was created with the correct parameters
        mock_cv2.VideoCapture.assert_called_once_with(self.camera_id, mock_cv2.CAP_DSHOW)
        
        # Check that the FPS was set
        mock_cap.set.assert_called_once_with(mock_cv2.CAP_PROP_FPS, self.fps)
        
        # Check that the emit method was called with the correct values
        thread.frame_captured.emit.assert_called_once_with(test_frame, 12345)
    
    @patch('src.capture.video_capture.sys.platform', 'linux')
    @patch('src.capture.video_capture.cv2')
    def test_run_real_capture_non_windows(self, mock_cv2):
        """Test that the thread runs correctly on non-Windows platforms."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        # Mock the emit method to track calls
        thread.frame_captured = MagicMock()
        
        # Mock the get_current_timestamp method to return a fixed value
        thread.get_current_timestamp = MagicMock(return_value=12345)
        
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock the read method to return a valid frame once, then stop the thread
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        def read_side_effect():
            thread.is_running = False
            return (True, test_frame)
        
        mock_cap.read.side_effect = read_side_effect
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the real capture method directly
        thread._run_real_capture()
        
        # Check that the VideoCapture was created with the correct parameters
        mock_cv2.VideoCapture.assert_called_once_with(self.camera_id)
        
        # Check that the FPS was set
        mock_cap.set.assert_called_once_with(mock_cv2.CAP_PROP_FPS, self.fps)
        
        # Check that the emit method was called with the correct values
        thread.frame_captured.emit.assert_called_once_with(test_frame, 12345)
    
    @patch('src.capture.video_capture.cv2')
    def test_run_real_capture_camera_not_opened(self, mock_cv2):
        """Test that the thread handles the case where the camera cannot be opened."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Run the real capture method directly
        with patch('src.capture.video_capture.logging') as mock_logging:
            thread._run_real_capture()
            
            # Check that an error was logged
            mock_logging.error.assert_called_once_with(
                f"Could not open video source for device {self.camera_id}."
            )
        
        # Check that is_running was set to False
        self.assertFalse(thread.is_running)
    
    @patch('src.capture.video_capture.cv2')
    def test_run_real_capture_dropped_frame(self, mock_cv2):
        """Test that the thread handles dropped frames."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock the read method to return a dropped frame, then a valid frame, then stop the thread
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set up a sequence of return values for read()
        read_returns = [(False, None), (True, test_frame)]
        read_index = 0
        
        def read_side_effect():
            nonlocal read_index
            if read_index == 0:
                read_index += 1
                return read_returns[0]
            else:
                thread.is_running = False
                return read_returns[1]
        
        mock_cap.read.side_effect = read_side_effect
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Mock the sleep method to prevent actual sleeping
        with patch('src.capture.video_capture.time.sleep') as mock_sleep:
            # Run the real capture method directly
            with patch('src.capture.video_capture.logging') as mock_logging:
                thread._run_real_capture()
                
                # Check that a warning was logged for the dropped frame
                mock_logging.warning.assert_called_once_with(f"Dropped frame from {self.camera_name} camera.")
        
        # Check that sleep was called to prevent busy-waiting
        mock_sleep.assert_called_once_with(0.01)
    
    @patch('src.capture.video_capture.cv2')
    def test_run_real_capture_exception(self, mock_cv2):
        """Test that the thread handles exceptions during real capture."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        # Mock the VideoCapture to raise an exception
        mock_cv2.VideoCapture.side_effect = Exception("Test exception")
        
        # Run the real capture method directly
        with patch('src.capture.video_capture.logging') as mock_logging:
            thread._run_real_capture()
            
            # Check that an error was logged
            mock_logging.error.assert_called_once_with(
                f"An exception occurred in {self.camera_name} capture thread: Test exception"
            )
    
    @patch('src.capture.video_capture.cv2')
    def test_cleanup(self, mock_cv2):
        """Test that the cleanup method correctly cleans up resources."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        thread.cap = mock_cap
        
        # Call the cleanup method
        thread._cleanup()
        
        # Check that the VideoCapture was released
        mock_cap.release.assert_called_once()
        
        # Check that the cap was set to None
        self.assertIsNone(thread.cap)
    
    def test_cleanup_no_cap(self):
        """Test that the cleanup method handles the case where there is no cap."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        thread.cap = None
        
        # Call the cleanup method
        thread._cleanup()
        
        # No assertions needed, just checking that it doesn't raise an exception
    
    def test_sleep(self):
        """Test that the _sleep method sleeps for the specified time."""
        thread = VideoCaptureThread(self.camera_id, self.camera_name, self.fps)
        
        with patch('src.capture.video_capture.time.sleep') as mock_sleep:
            thread._sleep(0.1)
            mock_sleep.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()