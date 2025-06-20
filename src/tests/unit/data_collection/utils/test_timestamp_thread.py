# src/tests/unit/data_collection/utils/test_timestamp_thread.py

import unittest
from unittest.mock import MagicMock, patch
import time

from PyQt5.QtCore import QThread

from src.data_collection.utils.timestamp_thread import TimestampThread

class TestTimestampThread(unittest.TestCase):
    """
    Unit tests for the TimestampThread class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a TimestampThread instance with a test frequency
        self.frequency = 10  # Use a lower frequency for testing
        self.thread = TimestampThread(frequency=self.frequency)
        
    def test_initialization(self):
        """
        Test that the TimestampThread initializes correctly.
        """
        # Check that the frequency was stored correctly
        self.assertEqual(self.thread.frequency, self.frequency)
        
        # Check that the interval was calculated correctly
        self.assertEqual(self.thread.interval, 1.0 / self.frequency)
        
        # Check that the running flag is initially False
        self.assertFalse(self.thread.running)
        
    @patch('src.data_collection.utils.timestamp_thread.QThread.setPriority')
    @patch('src.data_collection.utils.timestamp_thread.time.perf_counter_ns')
    @patch('src.data_collection.utils.timestamp_thread.time.sleep')
    def test_run(self, mock_sleep, mock_perf_counter, mock_set_priority):
        """
        Test that the run method emits timestamps at the specified frequency.
        """
        # Set up mock return values
        mock_perf_counter.return_value = 1234567890
        
        # Create a mock to track the timestamp_generated signal
        mock_callback = MagicMock()
        self.thread.timestamp_generated.connect(mock_callback)
        
        # Set up the thread to run for a short time
        def sleep_side_effect(*args, **kwargs):
            # Stop the thread after the first iteration
            self.thread.running = False
            
        mock_sleep.side_effect = sleep_side_effect
        
        # Run the thread
        self.thread.run()
        
        # Check that the thread was set to high priority
        mock_set_priority.assert_called_with(QThread.HighPriority)
        
        # Check that the timestamp was emitted
        mock_callback.assert_called_with(mock_perf_counter.return_value)
        
        # Check that sleep was called with the correct interval
        mock_sleep.assert_called_with(self.thread.interval)
        
    @patch('src.data_collection.utils.timestamp_thread.time.perf_counter_ns')
    @patch('src.data_collection.utils.timestamp_thread.time.sleep')
    def test_run_multiple_iterations(self, mock_sleep, mock_perf_counter):
        """
        Test that the run method emits timestamps for multiple iterations.
        """
        # Set up mock return values for multiple iterations
        timestamps = [1234567890, 1234567891, 1234567892]
        mock_perf_counter.side_effect = timestamps
        
        # Create a mock to track the timestamp_generated signal
        mock_callback = MagicMock()
        self.thread.timestamp_generated.connect(mock_callback)
        
        # Set up the thread to run for a few iterations
        iteration_count = 0
        
        def sleep_side_effect(*args, **kwargs):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= len(timestamps):
                self.thread.running = False
                
        mock_sleep.side_effect = sleep_side_effect
        
        # Run the thread
        self.thread.run()
        
        # Check that the timestamps were emitted
        self.assertEqual(mock_callback.call_count, len(timestamps))
        for i, timestamp in enumerate(timestamps):
            mock_callback.assert_any_call(timestamp)
            
        # Check that sleep was called with the correct interval for each iteration
        self.assertEqual(mock_sleep.call_count, len(timestamps))
        for call in mock_sleep.call_args_list:
            args, kwargs = call
            self.assertEqual(args[0], self.thread.interval)
            
    def test_stop(self):
        """
        Test that the stop method sets the running flag to False.
        """
        # Set the running flag to True
        self.thread.running = True
        
        # Call stop
        self.thread.stop()
        
        # Check that the running flag was set to False
        self.assertFalse(self.thread.running)
        
    @patch('src.data_collection.utils.timestamp_thread.logging')
    def test_logging(self, mock_logging):
        """
        Test that the thread logs appropriate messages.
        """
        # Create a new thread to trigger the initialization log message
        thread = TimestampThread(frequency=self.frequency)
        
        # Check that the initialization was logged
        mock_logging.info.assert_called_with(f"TimestampThread initialized with frequency {self.frequency}Hz")
        
        # Run the thread for a short time
        with patch('src.data_collection.utils.timestamp_thread.time.sleep') as mock_sleep:
            def sleep_side_effect(*args, **kwargs):
                thread.running = False
                
            mock_sleep.side_effect = sleep_side_effect
            thread.run()
            
        # Check that the start was logged
        mock_logging.info.assert_any_call("TimestampThread started")
        
        # Stop the thread
        thread.stop()
        
        # Check that the stop was logged
        mock_logging.info.assert_called_with("TimestampThread stopped")

if __name__ == '__main__':
    unittest.main()