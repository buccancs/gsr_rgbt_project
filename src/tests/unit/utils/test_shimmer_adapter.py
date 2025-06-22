# src/tests/unit/utils/test_shimmer_adapter.py

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from src.utils.shimmer_adapter import (
    ShimmerAdapter,
    ShimmerAdapterError,
    PYSHIMMER_AVAILABLE,
    SHIMMER_C_API_PATH,
    SHIMMER_JAVA_API_PATH,
    SHIMMER_ANDROID_API_PATH
)

class TestShimmerAdapter(unittest.TestCase):
    """
    Unit tests for the ShimmerAdapter class.
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.port = "COM3"
        self.sampling_rate = 32
        
        # Create patch for Path.exists() to control API availability
        self.path_exists_patcher = patch('pathlib.Path.exists')
        self.mock_path_exists = self.path_exists_patcher.start()
        
        # By default, make all APIs available
        self.mock_path_exists.return_value = True
        
    def tearDown(self):
        """
        Clean up after each test method.
        """
        self.path_exists_patcher.stop()

    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_initialization(self, mock_pyshimmer):
        """
        Test that the ShimmerAdapter initializes correctly.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        
        self.assertEqual(adapter.port, self.port)
        self.assertEqual(adapter.sampling_rate, self.sampling_rate)
        self.assertFalse(adapter.simulation_mode)
        self.assertIsNone(adapter.shimmer_device)
        self.assertTrue(adapter.c_api_available)
        self.assertTrue(adapter.java_api_available)
        self.assertTrue(adapter.android_api_available)
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_connect_success(self, mock_pyshimmer):
        """
        Test that connect() successfully connects to a Shimmer device.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_pyshimmer.Shimmer.return_value = mock_shimmer
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        
        result = adapter.connect()
        
        self.assertTrue(result)
        mock_pyshimmer.Shimmer.assert_called_with(self.port)
        mock_shimmer.set_sampling_rate.assert_called_with(self.sampling_rate)
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_connect_exception(self, mock_pyshimmer):
        """
        Test that connect() handles exceptions gracefully.
        """
        # Make the Shimmer constructor raise an exception
        mock_pyshimmer.Shimmer.side_effect = Exception("Connection failed")
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        
        result = adapter.connect()
        
        self.assertFalse(result)
        mock_pyshimmer.Shimmer.assert_called_with(self.port)
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_connect_simulation_mode(self, mock_pyshimmer):
        """
        Test that connect() doesn't actually connect in simulation mode.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )
        
        result = adapter.connect()
        
        self.assertTrue(result)
        mock_pyshimmer.Shimmer.assert_not_called()
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', False)
    def test_connect_no_pyshimmer(self):
        """
        Test that connect() fails when pyshimmer is not available.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        
        result = adapter.connect()
        
        self.assertFalse(result)
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_disconnect_success(self, mock_pyshimmer):
        """
        Test that disconnect() successfully disconnects from a Shimmer device.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_shimmer.is_streaming.return_value = True
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        adapter.shimmer_device = mock_shimmer
        
        result = adapter.disconnect()
        
        self.assertTrue(result)
        mock_shimmer.stop_streaming.assert_called_once()
        mock_shimmer.close.assert_called_once()
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_set_enabled_sensors(self, mock_pyshimmer):
        """
        Test that set_enabled_sensors() correctly sets the enabled sensors.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        adapter.shimmer_device = mock_shimmer
        
        sensors_bitmask = 0x07  # Example bitmask
        result = adapter.set_enabled_sensors(sensors_bitmask)
        
        self.assertTrue(result)
        mock_shimmer.set_enabled_sensors.assert_called_with(sensors_bitmask)
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_start_streaming(self, mock_pyshimmer):
        """
        Test that start_streaming() correctly starts streaming.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        adapter.shimmer_device = mock_shimmer
        
        result = adapter.start_streaming()
        
        self.assertTrue(result)
        mock_shimmer.start_streaming.assert_called_once()
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_stop_streaming(self, mock_pyshimmer):
        """
        Test that stop_streaming() correctly stops streaming.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_shimmer.is_streaming.return_value = True
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        adapter.shimmer_device = mock_shimmer
        
        result = adapter.stop_streaming()
        
        self.assertTrue(result)
        mock_shimmer.stop_streaming.assert_called_once()
    
    @patch('src.utils.shimmer_adapter.PYSHIMMER_AVAILABLE', True)
    @patch('src.utils.shimmer_adapter.pyshimmer')
    def test_read_data_packet(self, mock_pyshimmer):
        """
        Test that read_data_packet() correctly reads a data packet.
        """
        # Create a mock Shimmer device
        mock_shimmer = MagicMock()
        mock_packet = {'GSR_CAL': 0.5, 'Timestamp_FormattedUnix_CAL': 1234567890.123}
        mock_shimmer.read_data_packet.return_value = mock_packet
        
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=False
        )
        adapter.shimmer_device = mock_shimmer
        
        result = adapter.read_data_packet()
        
        self.assertEqual(result, mock_packet)
        mock_shimmer.read_data_packet.assert_called_once()
    
    def test_get_advanced_processing_capabilities(self):
        """
        Test that get_advanced_processing_capabilities() returns the correct capabilities.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )
        
        # Test with all APIs available
        self.mock_path_exists.return_value = True
        capabilities = adapter.get_advanced_processing_capabilities()
        
        self.assertIn("ECG to Heart Rate/IBI", capabilities)
        self.assertIn("PPG to Heart Rate/IBI", capabilities)
        self.assertIn("Advanced filtering", capabilities)
        self.assertIn("GSR calibration", capabilities)
        self.assertIn("3D orientation", capabilities)
        self.assertIn("Battery voltage monitoring", capabilities)
        
        # Test with only C API available
        def side_effect(path):
            return path == SHIMMER_C_API_PATH
        
        self.mock_path_exists.side_effect = side_effect
        capabilities = adapter.get_advanced_processing_capabilities()
        
        self.assertIn("ECG to Heart Rate/IBI", capabilities)
        self.assertIn("PPG to Heart Rate/IBI", capabilities)
        self.assertIn("Advanced filtering", capabilities)
        self.assertNotIn("GSR calibration", capabilities)
        self.assertNotIn("3D orientation", capabilities)
        self.assertNotIn("Battery voltage monitoring", capabilities)
    
    def test_process_ecg_to_hr(self):
        """
        Test that process_ecg_to_hr() returns the correct values.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )
        
        # Test with C API available
        self.mock_path_exists.return_value = True
        hr, ibi = adapter.process_ecg_to_hr([1.0, 2.0, 3.0])
        
        self.assertEqual(hr, 70.0)
        self.assertEqual(ibi, [800.0, 810.0, 790.0])
        
        # Test with C API not available
        self.mock_path_exists.return_value = False
        hr, ibi = adapter.process_ecg_to_hr([1.0, 2.0, 3.0])
        
        self.assertEqual(hr, 0.0)
        self.assertEqual(ibi, [])
    
    def test_process_ppg_to_hr(self):
        """
        Test that process_ppg_to_hr() returns the correct values.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )
        
        # Test with C API available
        self.mock_path_exists.return_value = True
        hr, ibi = adapter.process_ppg_to_hr([1.0, 2.0, 3.0])
        
        self.assertEqual(hr, 72.0)
        self.assertEqual(ibi, [820.0, 830.0, 810.0])
        
        # Test with C API not available
        self.mock_path_exists.return_value = False
        hr, ibi = adapter.process_ppg_to_hr([1.0, 2.0, 3.0])
        
        self.assertEqual(hr, 0.0)
        self.assertEqual(ibi, [])
    
    def test_apply_filter(self):
        """
        Test that apply_filter() returns the correct values.
        """
        adapter = ShimmerAdapter(
            port=self.port,
            sampling_rate=self.sampling_rate,
            simulation_mode=True
        )
        
        data = [1.0, 2.0, 3.0]
        filter_type = "lowpass"
        params = {"cutoff": 10.0}
        
        # Test with C API available
        self.mock_path_exists.return_value = True
        result = adapter.apply_filter(data, filter_type, params)
        
        self.assertEqual(result, data)  # In the placeholder implementation, the data is returned unchanged
        
        # Test with C API not available
        self.mock_path_exists.return_value = False
        result = adapter.apply_filter(data, filter_type, params)
        
        self.assertEqual(result, data)

if __name__ == '__main__':
    unittest.main()