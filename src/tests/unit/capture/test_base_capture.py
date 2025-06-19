import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.capture.base_capture import BaseCaptureThread


class MockCaptureThread(BaseCaptureThread):
    """Mock implementation of BaseCaptureThread for testing."""
    
    def __init__(self, device_name, parent=None, simulation_mode=False):
        super().__init__(device_name, parent)
        self.simulation_mode = simulation_mode
        self.cleanup_called = False
        self.simulation_run_called = False
        self.real_capture_run_called = False
    
    def _run_simulation(self):
        self.simulation_run_called = True
        # Simulate some work
        count = 0
        while self.is_running and count < 3:
            time.sleep(0.01)
            count += 1
    
    def _run_real_capture(self):
        self.real_capture_run_called = True
        # Simulate some work
        count = 0
        while self.is_running and count < 3:
            time.sleep(0.01)
            count += 1
    
    def _cleanup(self):
        self.cleanup_called = True


class TestBaseCaptureThread(unittest.TestCase):
    """Test suite for the BaseCaptureThread class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device_name = "TestDevice"
    
    def test_initialization(self):
        """Test that the thread initializes correctly."""
        thread = MockCaptureThread(self.device_name)
        
        self.assertEqual(thread.device_name, self.device_name)
        self.assertFalse(thread.is_running)
        self.assertEqual(thread.objectName(), f"{self.device_name}CaptureThread")
    
    def test_run_simulation_mode(self):
        """Test that the thread runs in simulation mode when configured."""
        thread = MockCaptureThread(self.device_name, simulation_mode=True)
        
        # Connect to the finished signal
        finished_callback = MagicMock()
        thread.finished.connect(finished_callback)
        
        # Start the thread
        thread.start()
        
        # Wait for the thread to finish
        thread.wait(1000)  # Wait up to 1 second
        
        # Check that the simulation method was called
        self.assertTrue(thread.simulation_run_called)
        self.assertFalse(thread.real_capture_run_called)
        self.assertTrue(thread.cleanup_called)
        
        # Check that the finished signal was emitted
        finished_callback.assert_called_once()
    
    def test_run_real_capture_mode(self):
        """Test that the thread runs in real capture mode when configured."""
        thread = MockCaptureThread(self.device_name, simulation_mode=False)
        
        # Connect to the finished signal
        finished_callback = MagicMock()
        thread.finished.connect(finished_callback)
        
        # Start the thread
        thread.start()
        
        # Wait for the thread to finish
        thread.wait(1000)  # Wait up to 1 second
        
        # Check that the real capture method was called
        self.assertFalse(thread.simulation_run_called)
        self.assertTrue(thread.real_capture_run_called)
        self.assertTrue(thread.cleanup_called)
        
        # Check that the finished signal was emitted
        finished_callback.assert_called_once()
    
    def test_stop(self):
        """Test that the stop method correctly stops the thread."""
        thread = MockCaptureThread(self.device_name)
        
        # Start the thread
        thread.start()
        
        # Give it a moment to start
        time.sleep(0.05)
        
        # Stop the thread
        thread.stop()
        
        # Wait for the thread to finish
        thread.wait(1000)  # Wait up to 1 second
        
        # Check that the thread is no longer running
        self.assertFalse(thread.is_running)
        self.assertTrue(thread.cleanup_called)
    
    def test_get_current_timestamp(self):
        """Test that the get_current_timestamp method returns a valid timestamp."""
        thread = MockCaptureThread(self.device_name)
        
        # Get a timestamp
        timestamp = thread.get_current_timestamp()
        
        # Check that it's a valid timestamp (a large positive integer)
        self.assertIsInstance(timestamp, int)
        self.assertGreater(timestamp, 0)
    
    @patch('src.capture.base_capture.logging')
    def test_logging(self, mock_logging):
        """Test that the thread logs appropriate messages."""
        thread = MockCaptureThread(self.device_name)
        
        # Start the thread
        thread.start()
        
        # Wait for the thread to finish
        thread.wait(1000)  # Wait up to 1 second
        
        # Check that appropriate log messages were generated
        mock_logging.info.assert_any_call(f"{self.device_name} capture thread started.")
        mock_logging.info.assert_any_call(f"{self.device_name} capture thread has finished.")
    
    def test_exception_handling(self):
        """Test that exceptions in the run methods are properly handled."""
        # Create a thread that will raise an exception
        thread = MockCaptureThread(self.device_name)
        thread._run_real_capture = MagicMock(side_effect=Exception("Test exception"))
        
        # Connect to the finished signal
        finished_callback = MagicMock()
        thread.finished.connect(finished_callback)
        
        # Start the thread
        thread.start()
        
        # Wait for the thread to finish
        thread.wait(1000)  # Wait up to 1 second
        
        # Check that cleanup was called and the finished signal was emitted
        self.assertTrue(thread.cleanup_called)
        finished_callback.assert_called_once()


if __name__ == "__main__":
    unittest.main()