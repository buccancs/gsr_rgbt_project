import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Mock pyshimmer module
sys.modules['pyshimmer'] = MagicMock()

from src.capture.gsr_capture import GsrCaptureThread


class TestGsrCaptureThread(unittest.TestCase):
    """Test suite for the GsrCaptureThread class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.port = "COM3"
        self.sampling_rate = 32
    
    def test_initialization(self):
        """Test that the thread initializes correctly."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        
        self.assertEqual(thread.device_name, "GSR")
        self.assertEqual(thread.port, self.port)
        self.assertEqual(thread.sampling_rate, self.sampling_rate)
        self.assertFalse(thread.simulation_mode)
        self.assertIsNone(thread.shimmer_device)
    
    def test_initialization_simulation_mode(self):
        """Test that the thread initializes correctly in simulation mode."""
        thread = GsrCaptureThread(self.port, self.sampling_rate, simulation_mode=True)
        
        self.assertEqual(thread.device_name, "GSR")
        self.assertEqual(thread.port, self.port)
        self.assertEqual(thread.sampling_rate, self.sampling_rate)
        self.assertTrue(thread.simulation_mode)
        self.assertIsNone(thread.shimmer_device)
    
    @patch('src.capture.gsr_capture.time.sleep')
    def test_run_simulation(self, mock_sleep):
        """Test that the thread runs correctly in simulation mode."""
        thread = GsrCaptureThread(self.port, self.sampling_rate, simulation_mode=True)
        
        # Mock the emit method to track calls
        thread.gsr_data_point = MagicMock()
        
        # Mock the get_current_timestamp method to return a fixed value
        thread.get_current_timestamp = MagicMock(return_value=12345)
        
        # Set is_running to True initially, then False after a few iterations
        thread.is_running = True
        
        def stop_after_calls(*args, **kwargs):
            thread.is_running = False
            return None
        
        # Make sleep stop the thread
        mock_sleep.side_effect = stop_after_calls
        
        # Run the simulation method directly
        thread._run_simulation()
        
        # Check that the emit method was called at least once
        thread.gsr_data_point.emit.assert_called()
        
        # Check that the first argument is a float (GSR value)
        args, _ = thread.gsr_data_point.emit.call_args
        self.assertIsInstance(args[0], float)
        
        # Check that the second argument is the timestamp
        self.assertEqual(args[1], 12345)
    
    @patch('src.capture.gsr_capture.PYSHIMMER_AVAILABLE', True)
    @patch('src.capture.gsr_capture.pyshimmer')
    def test_run_real_capture(self, mock_pyshimmer):
        """Test that the thread runs correctly with real hardware."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        
        # Mock the emit method to track calls
        thread.gsr_data_point = MagicMock()
        
        # Mock the Shimmer device
        mock_shimmer = MagicMock()
        mock_pyshimmer.Shimmer.return_value = mock_shimmer
        
        # Mock the read_data_packet method to return a valid packet once, then stop the thread
        mock_packet = {'GSR_CAL': 0.75, 'Timestamp_FormattedUnix_CAL': 12345}
        
        def read_packet_side_effect():
            thread.is_running = False
            return mock_packet
        
        mock_shimmer.read_data_packet.side_effect = read_packet_side_effect
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the real capture method directly
        thread._run_real_capture()
        
        # Check that the Shimmer device was configured correctly
        mock_pyshimmer.Shimmer.assert_called_once_with(self.port)
        mock_shimmer.set_sampling_rate.assert_called_once_with(self.sampling_rate)
        mock_shimmer.enable_gsr.assert_called_once()
        mock_shimmer.start_streaming.assert_called_once()
        
        # Check that the emit method was called with the correct values
        thread.gsr_data_point.emit.assert_called_once_with(0.75, 12345)
    
    @patch('src.capture.gsr_capture.PYSHIMMER_AVAILABLE', False)
    @patch('src.capture.gsr_capture.logging')
    def test_run_real_capture_no_pyshimmer(self, mock_logging):
        """Test that the thread handles the case where pyshimmer is not available."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        
        # Run the real capture method directly
        thread._run_real_capture()
        
        # Check that an error was logged
        mock_logging.error.assert_called_once_with(
            "Real GSR sensor mode is selected, but the 'pyshimmer' library is not available."
        )
    
    @patch('src.capture.gsr_capture.PYSHIMMER_AVAILABLE', True)
    @patch('src.capture.gsr_capture.pyshimmer')
    def test_run_real_capture_exception(self, mock_pyshimmer):
        """Test that the thread handles exceptions during real capture."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        
        # Mock the Shimmer device to raise an exception
        mock_pyshimmer.Shimmer.side_effect = Exception("Test exception")
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the real capture method directly
        with patch('src.capture.gsr_capture.logging') as mock_logging:
            thread._run_real_capture()
            
            # Check that an error was logged
            mock_logging.error.assert_called_once_with(
                "Failed to connect or read from Shimmer device: Test exception"
            )
    
    @patch('src.capture.gsr_capture.PYSHIMMER_AVAILABLE', True)
    @patch('src.capture.gsr_capture.pyshimmer')
    def test_cleanup(self, mock_pyshimmer):
        """Test that the cleanup method correctly cleans up resources."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        
        # Mock the Shimmer device
        mock_shimmer = MagicMock()
        mock_shimmer.is_streaming.return_value = True
        thread.shimmer_device = mock_shimmer
        
        # Call the cleanup method
        thread._cleanup()
        
        # Check that the Shimmer device was stopped and closed
        mock_shimmer.stop_streaming.assert_called_once()
        mock_shimmer.close.assert_called_once()
    
    def test_cleanup_no_device(self):
        """Test that the cleanup method handles the case where there is no device."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        thread.shimmer_device = None
        
        # Call the cleanup method
        thread._cleanup()
        
        # No assertions needed, just checking that it doesn't raise an exception
    
    def test_sleep(self):
        """Test that the _sleep method sleeps for the specified time."""
        thread = GsrCaptureThread(self.port, self.sampling_rate)
        
        with patch('src.capture.gsr_capture.time.sleep') as mock_sleep:
            thread._sleep(0.1)
            mock_sleep.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()