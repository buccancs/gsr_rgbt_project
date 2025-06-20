# src/tests/unit/data_collection/capture/test_video_capture.py

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

from src.data_collection.capture.video_capture import VideoCaptureThread

class TestVideoCaptureThread(unittest.TestCase):
    """
    Unit tests for the VideoCaptureThread class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.camera_id = 0
        self.camera_name = "TestCamera"
        self.fps = 30
        
    def test_initialization(self):
        """
        Test that the VideoCaptureThread initializes correctly.
        """
        thread = VideoCaptureThread(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            fps=self.fps
        )
        
        self.assertEqual(thread.device_name, self.camera_name)
        self.assertEqual(thread.camera_id, self.camera_id)
        self.assertEqual(thread.fps, self.fps)
        self.assertFalse(thread.is_running)
        self.assertIsNone(thread.cap)
        
    @patch('src.data_collection.capture.video_capture.cv2')
    @patch('src.data_collection.capture.video_capture.logging')
    def test_run_real_capture_success(self, mock_logging, mock_cv2):
        """
        Test that _run_real_capture correctly captures and emits frames when successful.
        """
        # Create a mock VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock the read method to return a valid frame once, then set is_running to False
        mock_frame = np.ones((480, 640, 3), dtype=np.uint8)  # Create a dummy frame
        
        def read_side_effect():
            thread.is_running = False
            return (True, mock_frame)
            
        mock_cap.read.side_effect = read_side_effect
        
        thread = VideoCaptureThread(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            fps=self.fps
        )
        
        # Create a mock to track the frame_captured signal
        mock_callback = MagicMock()
        thread.frame_captured.connect(mock_callback)
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the capture
        thread._run_real_capture()
        
        # Check that the VideoCapture was created with the correct parameters
        if sys.platform == "win32":
            mock_cv2.VideoCapture.assert_called_with(self.camera_id, mock_cv2.CAP_DSHOW)
        else:
            mock_cv2.VideoCapture.assert_called_with(self.camera_id)
            
        # Check that the FPS was set
        mock_cap.set.assert_called_with(mock_cv2.CAP_PROP_FPS, self.fps)
        
        # Check that the frame_captured signal was emitted
        mock_callback.assert_called()
        
        # Check that the arguments are of the correct type
        args, _ = mock_callback.call_args
        frame, timestamp = args
        self.assertIs(frame, mock_frame)
        self.assertIsInstance(timestamp, (int, float))
        
    @patch('src.data_collection.capture.video_capture.cv2')
    @patch('src.data_collection.capture.video_capture.logging')
    def test_run_real_capture_camera_not_opened(self, mock_logging, mock_cv2):
        """
        Test that _run_real_capture handles the case when the camera cannot be opened.
        """
        # Create a mock VideoCapture object that fails to open
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        thread = VideoCaptureThread(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            fps=self.fps
        )
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the capture
        thread._run_real_capture()
        
        # Check that is_running was set to False
        self.assertFalse(thread.is_running)
        
        # Check that an error was logged
        mock_logging.error.assert_called_with(
            f"Could not open video source for device {self.camera_id}."
        )
        
    @patch('src.data_collection.capture.video_capture.cv2')
    @patch('src.data_collection.capture.video_capture.logging')
    def test_run_real_capture_dropped_frame(self, mock_logging, mock_cv2):
        """
        Test that _run_real_capture handles dropped frames correctly.
        """
        # Create a mock VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock the read method to return a dropped frame once, then a valid frame, then set is_running to False
        mock_frame = np.ones((480, 640, 3), dtype=np.uint8)  # Create a dummy frame
        
        # Create a sequence of return values for read()
        read_returns = [(False, None), (True, mock_frame)]
        read_iter = iter(read_returns)
        
        def read_side_effect():
            try:
                result = next(read_iter)
                if result[0]:  # If this is a valid frame
                    thread.is_running = False  # Stop after the valid frame
                return result
            except StopIteration:
                thread.is_running = False
                return (False, None)
                
        mock_cap.read.side_effect = read_side_effect
        
        thread = VideoCaptureThread(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            fps=self.fps
        )
        
        # Mock the _sleep method to avoid actual sleeping
        thread._sleep = MagicMock()
        
        # Create a mock to track the frame_captured signal
        mock_callback = MagicMock()
        thread.frame_captured.connect(mock_callback)
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the capture
        thread._run_real_capture()
        
        # Check that a warning was logged for the dropped frame
        mock_logging.warning.assert_called_with(f"Dropped frame from {self.camera_name} camera.")
        
        # Check that _sleep was called to prevent busy-waiting
        thread._sleep.assert_called_with(0.01)
        
        # Check that the frame_captured signal was emitted for the valid frame
        mock_callback.assert_called_once()
        
    @patch('src.data_collection.capture.video_capture.cv2')
    @patch('src.data_collection.capture.video_capture.logging')
    def test_cleanup(self, mock_logging, mock_cv2):
        """
        Test that the _cleanup method properly releases the camera.
        """
        # Create a mock VideoCapture object
        mock_cap = MagicMock()
        
        thread = VideoCaptureThread(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            fps=self.fps
        )
        thread.cap = mock_cap
        
        # Call cleanup
        thread._cleanup()
        
        # Check that the camera was released
        mock_cap.release.assert_called_once()
        self.assertIsNone(thread.cap)
        
    def test_sleep(self):
        """
        Test that the _sleep method sleeps for the specified time.
        """
        thread = VideoCaptureThread(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            fps=self.fps
        )
        
        # Mock time.sleep to avoid actual sleeping
        with patch('time.sleep') as mock_sleep:
            thread._sleep(0.5)
            mock_sleep.assert_called_with(0.5)

if __name__ == '__main__':
    unittest.main()