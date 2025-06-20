# src/tests/unit/data_collection/capture/test_base_capture.py

import unittest
from unittest.mock import MagicMock, patch
import time
import logging

from PyQt5.QtCore import QThread, pyqtSignal

from src.data_collection.capture.base_capture import BaseCaptureThread

class TestBaseCaptureThread(unittest.TestCase):
    """
    Unit tests for the BaseCaptureThread class.
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a concrete implementation of BaseCaptureThread for testing
        class ConcreteCaptureThread(BaseCaptureThread):
            def _run_simulation(self):
                # Simple implementation that sets a flag and sleeps briefly
                self.simulation_ran = True
                time.sleep(0.1)
                
            def _run_real_capture(self):
                # Simple implementation that sets a flag and sleeps briefly
                self.real_capture_ran = True
                time.sleep(0.1)
                
        self.thread_class = ConcreteCaptureThread
        
    def test_initialization(self):
        """
        Test that the BaseCaptureThread initializes correctly.
        """
        thread = self.thread_class("TestDevice")
        
        self.assertEqual(thread.device_name, "TestDevice")
        self.assertFalse(thread.is_running)
        self.assertEqual(thread.objectName(), "TestDeviceCaptureThread")
        
    def test_stop_method(self):
        """
        Test that the stop method sets is_running to False.
        """
        thread = self.thread_class("TestDevice")
        thread.is_running = True
        
        thread.stop()
        
        self.assertFalse(thread.is_running)
        
    def test_get_current_timestamp(self):
        """
        Test that get_current_timestamp returns a valid timestamp.
        """
        thread = self.thread_class("TestDevice")
        
        timestamp = thread.get_current_timestamp()
        
        self.assertIsInstance(timestamp, int)
        self.assertGreater(timestamp, 0)
        
    @patch('src.data_collection.capture.base_capture.logging')
    def test_run_simulation_mode(self, mock_logging):
        """
        Test that the run method calls _run_simulation when in simulation mode.
        """
        thread = self.thread_class("TestDevice")
        thread.simulation_mode = True
        thread.simulation_ran = False
        thread.real_capture_ran = False
        
        # Run the thread for a short time
        thread.start()
        time.sleep(0.2)  # Give it time to execute
        thread.stop()
        thread.wait()
        
        self.assertTrue(thread.simulation_ran)
        self.assertFalse(thread.real_capture_ran)
        mock_logging.info.assert_any_call("TestDevice capture thread started.")
        
    @patch('src.data_collection.capture.base_capture.logging')
    def test_run_real_capture_mode(self, mock_logging):
        """
        Test that the run method calls _run_real_capture when not in simulation mode.
        """
        thread = self.thread_class("TestDevice")
        thread.simulation_mode = False
        thread.simulation_ran = False
        thread.real_capture_ran = False
        
        # Run the thread for a short time
        thread.start()
        time.sleep(0.2)  # Give it time to execute
        thread.stop()
        thread.wait()
        
        self.assertFalse(thread.simulation_ran)
        self.assertTrue(thread.real_capture_ran)
        mock_logging.info.assert_any_call("TestDevice capture thread started.")
        
    @patch('src.data_collection.capture.base_capture.logging')
    def test_run_with_exception(self, mock_logging):
        """
        Test that the run method handles exceptions properly.
        """
        class ExceptionThread(BaseCaptureThread):
            def _run_real_capture(self):
                raise Exception("Test exception")
                
        thread = ExceptionThread("TestDevice")
        
        # Run the thread for a short time
        thread.start()
        time.sleep(0.2)  # Give it time to execute
        thread.wait()
        
        mock_logging.error.assert_called_with(
            "An exception occurred in TestDevice capture thread: Test exception"
        )
        
    def test_finished_signal(self):
        """
        Test that the finished signal is emitted when the thread completes.
        """
        thread = self.thread_class("TestDevice")
        
        # Create a mock to track the finished signal
        mock_callback = MagicMock()
        thread.finished.connect(mock_callback)
        
        # Run the thread for a short time
        thread.start()
        time.sleep(0.2)  # Give it time to execute
        thread.stop()
        thread.wait()
        
        # Check that the finished signal was emitted
        mock_callback.assert_called_once()

if __name__ == '__main__':
    unittest.main()