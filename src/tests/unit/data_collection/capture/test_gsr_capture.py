# src/tests/unit/data_collection/capture/test_gsr_capture.py

import unittest
from unittest.mock import MagicMock, patch
import time

from src.data_collection.capture.gsr_capture import GsrCaptureThread, PYSHIMMER_AVAILABLE

class TestGsrCaptureThread(unittest.TestCase):
    """
    Unit tests for the GsrCaptureThread class.
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.port = "COM3"
        self.sampling_rate = 32

    def test_initialization(self):
        """
        Test that the GsrCaptureThread initializes correctly.
        """
        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )

        self.assertEqual(thread.device_name, "GSR")
        self.assertEqual(thread.port, self.port)
        self.assertEqual(thread.sampling_rate, self.sampling_rate)
        self.assertTrue(thread.simulation_mode)
        self.assertFalse(thread.is_running)
        self.assertIsNone(thread.shimmer_device)

    def test_initialization_with_sensors_to_enable(self):
        """
        Test that the GsrCaptureThread initializes correctly with custom sensors_to_enable.
        """
        # Mock pyshimmer for this test
        with patch('src.data_collection.capture.gsr_capture.PYSHIMMER_AVAILABLE', True), \
             patch('src.data_collection.capture.gsr_capture.pyshimmer') as mock_pyshimmer:

            # Create a custom sensor configuration
            mock_pyshimmer.Shimmer.SENSOR_GSR = 0x01
            mock_pyshimmer.Shimmer.SENSOR_PPG = 0x02
            custom_sensors = 0x01  # Only GSR

            thread = GsrCaptureThread(
                port=self.port,
                sampling_rate=self.sampling_rate,
                simulation_mode=False,
                sensors_to_enable=custom_sensors
            )

            self.assertEqual(thread.sensors_to_enable, custom_sensors)
            self.assertEqual(thread.device_name, "GSR")
            self.assertEqual(thread.port, self.port)
            self.assertEqual(thread.sampling_rate, self.sampling_rate)
            self.assertFalse(thread.simulation_mode)
            self.assertFalse(thread.is_running)
            self.assertIsNone(thread.shimmer_device)

    @patch('src.data_collection.capture.gsr_capture.logging')
    def test_run_simulation(self, mock_logging):
        """
        Test that the _run_simulation method generates and emits simulated GSR data.
        """
        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )

        # Create a mock to track the gsr_data_point signal
        mock_callback = MagicMock()
        thread.gsr_data_point.connect(mock_callback)

        # Mock the _sleep method to avoid actual sleeping
        thread._sleep = MagicMock()

        # Set is_running to True initially, then False after a few iterations
        thread.is_running = True

        def side_effect(*args, **kwargs):
            thread.is_running = False

        thread._sleep.side_effect = side_effect

        # Run the simulation
        thread._run_simulation()

        # Check that the gsr_data_point signal was emitted
        mock_callback.assert_called()

        # Check that the first argument (gsr_value) is a float
        args, _ = mock_callback.call_args
        gsr_value, timestamp = args
        self.assertIsInstance(gsr_value, float)
        self.assertIsInstance(timestamp, (int, float))

        # Check that logging was called
        mock_logging.info.assert_any_call(
            f"Running in GSR simulation mode. Sampling rate: {self.sampling_rate}Hz."
        )

    @patch('src.data_collection.capture.gsr_capture.PYSHIMMER_AVAILABLE', False)
    @patch('src.data_collection.capture.gsr_capture.logging')
    def test_run_real_capture_without_pyshimmer(self, mock_logging):
        """
        Test that _run_real_capture handles the case when pyshimmer is not available.
        """
        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )

        thread._run_real_capture()

        mock_logging.error.assert_called_with(
            "Real GSR sensor mode is selected, but the 'pyshimmer' library is not available."
        )

    @patch('src.data_collection.capture.gsr_capture.PYSHIMMER_AVAILABLE', True)
    @patch('src.data_collection.capture.gsr_capture.pyshimmer')
    @patch('src.data_collection.capture.gsr_capture.logging')
    def test_run_real_capture_with_pyshimmer(self, mock_logging, mock_pyshimmer):
        """
        Test that _run_real_capture correctly interacts with the pyshimmer library.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_pyshimmer.Shimmer.return_value = mock_shimmer

        # Mock the read_data_packet method to return a valid packet once, then set is_running to False
        packet = {'GSR_CAL': 0.5, 'Timestamp_FormattedUnix_CAL': 1234567890.123}

        def read_data_packet_side_effect():
            thread.is_running = False
            return packet

        mock_shimmer.read_data_packet.side_effect = read_data_packet_side_effect

        # Set up default sensor configuration
        mock_pyshimmer.Shimmer.SENSOR_GSR = 0x01
        mock_pyshimmer.Shimmer.SENSOR_PPG = 0x02
        mock_pyshimmer.Shimmer.SENSOR_ACCEL = 0x04
        default_sensors = 0x01 | 0x02 | 0x04  # GSR, PPG, and ACCEL

        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )

        # Create a mock to track the gsr_data_point signal
        mock_callback = MagicMock()
        thread.gsr_data_point.connect(mock_callback)

        # Set is_running to True initially
        thread.is_running = True

        # Run the real capture
        thread._run_real_capture()

        # Check that the Shimmer device was configured correctly
        mock_pyshimmer.Shimmer.assert_called_with(self.port)
        mock_shimmer.set_sampling_rate.assert_called_with(self.sampling_rate)
        mock_shimmer.set_enabled_sensors.assert_called_with(default_sensors)
        mock_shimmer.start_streaming.assert_called_once()

        # Check that the gsr_data_point signal was emitted with the correct values
        mock_callback.assert_called_with(packet['GSR_CAL'], packet['Timestamp_FormattedUnix_CAL'])

    @patch('src.data_collection.capture.gsr_capture.PYSHIMMER_AVAILABLE', True)
    @patch('src.data_collection.capture.gsr_capture.pyshimmer')
    @patch('src.data_collection.capture.gsr_capture.logging')
    def test_run_real_capture_with_custom_sensors(self, mock_logging, mock_pyshimmer):
        """
        Test that _run_real_capture correctly uses custom sensor configuration.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_pyshimmer.Shimmer.return_value = mock_shimmer

        # Mock the read_data_packet method to return a valid packet once, then set is_running to False
        packet = {'GSR_CAL': 0.5, 'Timestamp_FormattedUnix_CAL': 1234567890.123}

        def read_data_packet_side_effect():
            thread.is_running = False
            return packet

        mock_shimmer.read_data_packet.side_effect = read_data_packet_side_effect

        # Set up custom sensor configuration
        mock_pyshimmer.Shimmer.SENSOR_GSR = 0x01
        mock_pyshimmer.Shimmer.SENSOR_PPG = 0x02
        custom_sensors = 0x01  # Only GSR

        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False,
            sensors_to_enable=custom_sensors
        )

        # Create a mock to track the gsr_data_point signal
        mock_callback = MagicMock()
        thread.gsr_data_point.connect(mock_callback)

        # Set is_running to True initially
        thread.is_running = True

        # Run the real capture
        thread._run_real_capture()

        # Check that the Shimmer device was configured correctly
        mock_pyshimmer.Shimmer.assert_called_with(self.port)
        mock_shimmer.set_sampling_rate.assert_called_with(self.sampling_rate)
        mock_shimmer.set_enabled_sensors.assert_called_with(custom_sensors)
        mock_shimmer.start_streaming.assert_called_once()

        # Check that the gsr_data_point signal was emitted with the correct values
        mock_callback.assert_called_with(packet['GSR_CAL'], packet['Timestamp_FormattedUnix_CAL'])

    @patch('src.data_collection.capture.gsr_capture.PYSHIMMER_AVAILABLE', True)
    @patch('src.data_collection.capture.gsr_capture.pyshimmer')
    @patch('src.data_collection.capture.gsr_capture.logging')
    def test_cleanup(self, mock_logging, mock_pyshimmer):
        """
        Test that the _cleanup method properly cleans up resources.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_shimmer.is_streaming.return_value = True

        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        thread.shimmer_device = mock_shimmer

        # Call cleanup
        thread._cleanup()

        # Check that the Shimmer device was properly closed
        mock_shimmer.stop_streaming.assert_called_once()
        mock_shimmer.close.assert_called_once()
        mock_logging.info.assert_called_with("Shimmer device connection closed.")

    def test_sleep(self):
        """
        Test that the _sleep method sleeps for the specified time.
        """
        thread = GsrCaptureThread(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )

        # Mock time.sleep to avoid actual sleeping
        with patch('time.sleep') as mock_sleep:
            thread._sleep(0.5)
            mock_sleep.assert_called_with(0.5)

if __name__ == '__main__':
    unittest.main()
