import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.system.utils.timestamp_thread import TimestampThread


class TestTimestampThread(unittest.TestCase):
    """Test suite for the TimestampThread class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frequency = 200
    
    def test_initialization(self):
        """Test that the thread initializes correctly."""
        thread = TimestampThread(self.frequency)
        
        self.assertEqual(thread.frequency, self.frequency)
        self.assertEqual(thread.interval, 1.0 / self.frequency)
        self.assertFalse(thread.running)
    
    @patch('src.system.utils.timestamp_thread.time.sleep')
    def test_run(self, mock_sleep):
        """Test that the thread runs correctly."""
        thread = TimestampThread(self.frequency)
        
        # Mock the emit method to track calls
        thread.timestamp_generated = MagicMock()
        
        # Set running to True initially, then False after a few iterations
        thread.running = True
        
        def stop_after_calls(*args, **kwargs):
            thread.running = False
            return None
        
        # Make sleep stop the thread
        mock_sleep.side_effect = stop_after_calls
        
        # Run the thread
        thread.run()
        
        # Check that the emit method was called at least once
        thread.timestamp_generated.emit.assert_called()
        
        # Check that the argument is an integer (timestamp)
        args, _ = thread.timestamp_generated.emit.call_args
        self.assertIsInstance(args[0], int)
    
    def test_stop(self):
        """Test that the stop method correctly stops the thread."""
        thread = TimestampThread(self.frequency)
        thread.running = True
        
        # Call the stop method
        thread.stop()
        
        # Check that running is set to False
        self.assertFalse(thread.running)


if __name__ == "__main__":
    unittest.main()