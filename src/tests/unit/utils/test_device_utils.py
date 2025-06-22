# src/tests/unit/utils/test_device_utils.py

import unittest
from unittest.mock import patch, MagicMock

from src.utils.device_utils import find_shimmer_com_port, DeviceNotFoundError


class TestDeviceUtils(unittest.TestCase):
    """
    Unit tests for the device_utils module.
    """

    @patch('src.utils.device_utils.serial.tools.list_ports.comports')
    def test_find_shimmer_com_port_success(self, mock_comports):
        """
        Test that find_shimmer_com_port correctly identifies a Shimmer device.
        """
        # Create a mock port with a description containing "Shimmer"
        mock_port = MagicMock()
        mock_port.description = "Shimmer3 GSR+ Device"
        mock_port.device = "COM3"
        
        # Return a list containing our mock port
        mock_comports.return_value = [mock_port]
        
        # Call the function
        result = find_shimmer_com_port()
        
        # Check that it returned the correct port
        self.assertEqual(result, "COM3")
        
    @patch('src.utils.device_utils.serial.tools.list_ports.comports')
    def test_find_shimmer_com_port_multiple_devices(self, mock_comports):
        """
        Test that find_shimmer_com_port correctly identifies a Shimmer device
        when multiple devices are connected.
        """
        # Create mock ports
        mock_port1 = MagicMock()
        mock_port1.description = "Some other Bluetooth device"
        mock_port1.device = "COM1"
        
        mock_port2 = MagicMock()
        mock_port2.description = "Shimmer3 GSR+ Device"
        mock_port2.device = "COM3"
        
        mock_port3 = MagicMock()
        mock_port3.description = "Another device"
        mock_port3.device = "COM4"
        
        # Return a list containing our mock ports
        mock_comports.return_value = [mock_port1, mock_port2, mock_port3]
        
        # Call the function
        result = find_shimmer_com_port()
        
        # Check that it returned the correct port
        self.assertEqual(result, "COM3")
        
    @patch('src.utils.device_utils.serial.tools.list_ports.comports')
    def test_find_shimmer_com_port_no_device(self, mock_comports):
        """
        Test that find_shimmer_com_port raises DeviceNotFoundError when no Shimmer device is found.
        """
        # Create mock ports without any Shimmer device
        mock_port1 = MagicMock()
        mock_port1.description = "Some other Bluetooth device"
        mock_port1.device = "COM1"
        
        mock_port2 = MagicMock()
        mock_port2.description = "Another device"
        mock_port2.device = "COM4"
        
        # Return a list containing our mock ports
        mock_comports.return_value = [mock_port1, mock_port2]
        
        # Call the function and check that it raises the expected exception
        with self.assertRaises(DeviceNotFoundError):
            find_shimmer_com_port()
            
    @patch('src.utils.device_utils.serial.tools.list_ports.comports')
    def test_find_shimmer_com_port_no_devices(self, mock_comports):
        """
        Test that find_shimmer_com_port raises DeviceNotFoundError when no devices are connected.
        """
        # Return an empty list
        mock_comports.return_value = []
        
        # Call the function and check that it raises the expected exception
        with self.assertRaises(DeviceNotFoundError):
            find_shimmer_com_port()
            
    @patch('src.utils.device_utils.serial.tools.list_ports.comports')
    def test_find_shimmer_com_port_none_description(self, mock_comports):
        """
        Test that find_shimmer_com_port handles ports with None description.
        """
        # Create a mock port with None description
        mock_port1 = MagicMock()
        mock_port1.description = None
        mock_port1.device = "COM1"
        
        # Create a mock port with a description containing "Shimmer"
        mock_port2 = MagicMock()
        mock_port2.description = "Shimmer3 GSR+ Device"
        mock_port2.device = "COM3"
        
        # Return a list containing our mock ports
        mock_comports.return_value = [mock_port1, mock_port2]
        
        # Call the function
        result = find_shimmer_com_port()
        
        # Check that it returned the correct port
        self.assertEqual(result, "COM3")


if __name__ == '__main__':
    unittest.main()