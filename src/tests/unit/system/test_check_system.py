import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.system.validation.check_system import (
    check_camera,
    check_directories,
    check_dependencies,
    list_serial_ports,
    check_gsr_sensor,
    check_flir_cameras,
    main
)


class TestCheckSystem(unittest.TestCase):
    """Test suite for the check_system.py module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_id = 0
        self.camera_name = "Test Camera"
    
    @patch('src.system.validation.check_system.cv2')
    def test_check_camera_success(self, mock_cv2):
        """Test that check_camera returns True when the camera is working."""
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        
        # Call the function
        result = check_camera(self.camera_id, self.camera_name)
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the VideoCapture was created with the correct parameters
        if sys.platform == "win32":
            mock_cv2.VideoCapture.assert_called_once_with(self.camera_id, mock_cv2.CAP_DSHOW)
        else:
            mock_cv2.VideoCapture.assert_called_once_with(self.camera_id)
        
        # Check that the VideoCapture was released
        mock_cap.release.assert_called_once()
    
    @patch('src.system.validation.check_system.cv2')
    def test_check_camera_not_opened(self, mock_cv2):
        """Test that check_camera returns False when the camera cannot be opened."""
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Call the function
        result = check_camera(self.camera_id, self.camera_name)
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.cv2')
    def test_check_camera_read_failure(self, mock_cv2):
        """Test that check_camera returns False when a frame cannot be read."""
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        
        # Call the function
        result = check_camera(self.camera_id, self.camera_name)
        
        # Check the result
        self.assertFalse(result)
        
        # Check that the VideoCapture was released
        mock_cap.release.assert_called_once()
    
    @patch('src.system.validation.check_system.config')
    def test_check_directories_success(self, mock_config):
        """Test that check_directories returns True when directories can be created."""
        # Mock the config values
        mock_config.OUTPUT_DIR = MagicMock()
        
        # Call the function
        result = check_directories()
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the directory was created
        mock_config.OUTPUT_DIR.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('src.system.validation.check_system.config')
    def test_check_directories_failure(self, mock_config):
        """Test that check_directories returns False when directories cannot be created."""
        # Mock the config values
        mock_config.OUTPUT_DIR = MagicMock()
        mock_config.OUTPUT_DIR.mkdir.side_effect = OSError("Test error")
        
        # Call the function
        result = check_directories()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.__import__')
    def test_check_dependencies_success(self, mock_import):
        """Test that check_dependencies returns True when all dependencies are found."""
        # Mock the import function to always succeed
        mock_import.return_value = MagicMock()
        
        # Call the function
        result = check_dependencies()
        
        # Check the result
        self.assertTrue(result)
    
    @patch('src.system.validation.check_system.__import__')
    def test_check_dependencies_failure(self, mock_import):
        """Test that check_dependencies returns False when dependencies are missing."""
        # Mock the import function to fail for one package
        def import_side_effect(name, *args, **kwargs):
            if name == "tensorflow":
                raise ImportError("Test error")
            return MagicMock()
        
        mock_import.side_effect = import_side_effect
        
        # Call the function
        result = check_dependencies()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.SERIAL_TOOLS_AVAILABLE', True)
    @patch('src.system.validation.check_system.serial.tools.list_ports')
    def test_list_serial_ports_with_serial_tools(self, mock_list_ports):
        """Test that list_serial_ports returns a list of ports when serial.tools is available."""
        # Mock the list_ports.comports function
        mock_port1 = MagicMock()
        mock_port1.device = "/dev/ttyUSB0"
        mock_port1.description = "Test Port 1"
        
        mock_port2 = MagicMock()
        mock_port2.device = "/dev/ttyUSB1"
        mock_port2.description = "Test Port 2"
        
        mock_list_ports.comports.return_value = [mock_port1, mock_port2]
        
        # Call the function
        result = list_serial_ports()
        
        # Check the result
        self.assertEqual(result, ["/dev/ttyUSB0", "/dev/ttyUSB1"])
    
    @patch('src.system.validation.check_system.SERIAL_TOOLS_AVAILABLE', False)
    @patch('src.system.validation.check_system.sys.platform', 'linux')
    @patch('src.system.validation.check_system.glob.glob')
    def test_list_serial_ports_linux_fallback(self, mock_glob):
        """Test that list_serial_ports uses the fallback method on Linux."""
        # Mock the glob.glob function
        def glob_side_effect(pattern):
            if pattern == '/dev/ttyUSB*':
                return ['/dev/ttyUSB0', '/dev/ttyUSB1']
            elif pattern == '/dev/ttyACM*':
                return ['/dev/ttyACM0']
            else:
                return []
        
        mock_glob.side_effect = glob_side_effect
        
        # Call the function
        result = list_serial_ports()
        
        # Check the result
        self.assertEqual(result, ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0'])
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.PYSHIMMER_AVAILABLE', True)
    @patch('src.system.validation.check_system.pyshimmer')
    def test_check_gsr_sensor_success(self, mock_pyshimmer, mock_config):
        """Test that check_gsr_sensor returns True when the GSR sensor is working."""
        # Mock the config values
        mock_config.GSR_SIMULATION_MODE = False
        mock_config.GSR_SENSOR_PORT = "/dev/ttyUSB0"
        mock_config.GSR_SAMPLING_RATE = 32
        
        # Mock the Shimmer device
        mock_shimmer = MagicMock()
        mock_pyshimmer.Shimmer.return_value = mock_shimmer
        mock_shimmer.read_data_packet.return_value = {"GSR_CAL": 0.5}
        
        # Call the function
        result = check_gsr_sensor()
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the Shimmer device was configured correctly
        mock_pyshimmer.Shimmer.assert_called_once_with("/dev/ttyUSB0")
        mock_shimmer.set_sampling_rate.assert_called_once_with(32)
        mock_shimmer.enable_gsr.assert_called_once()
        mock_shimmer.start_streaming.assert_called_once()
        mock_shimmer.read_data_packet.assert_called_once()
        mock_shimmer.stop_streaming.assert_called_once()
        mock_shimmer.close.assert_called_once()
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.PYSHIMMER_AVAILABLE', True)
    @patch('src.system.validation.check_system.pyshimmer')
    def test_check_gsr_sensor_no_data(self, mock_pyshimmer, mock_config):
        """Test that check_gsr_sensor returns False when no data is received."""
        # Mock the config values
        mock_config.GSR_SIMULATION_MODE = False
        mock_config.GSR_SENSOR_PORT = "/dev/ttyUSB0"
        mock_config.GSR_SAMPLING_RATE = 32
        
        # Mock the Shimmer device
        mock_shimmer = MagicMock()
        mock_pyshimmer.Shimmer.return_value = mock_shimmer
        mock_shimmer.read_data_packet.return_value = None
        
        # Call the function
        result = check_gsr_sensor()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.PYSHIMMER_AVAILABLE', True)
    @patch('src.system.validation.check_system.pyshimmer')
    def test_check_gsr_sensor_exception(self, mock_pyshimmer, mock_config):
        """Test that check_gsr_sensor returns False when an exception occurs."""
        # Mock the config values
        mock_config.GSR_SIMULATION_MODE = False
        mock_config.GSR_SENSOR_PORT = "/dev/ttyUSB0"
        mock_config.GSR_SAMPLING_RATE = 32
        
        # Mock the Shimmer device to raise an exception
        mock_pyshimmer.Shimmer.side_effect = Exception("Test exception")
        
        # Call the function
        result = check_gsr_sensor()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.PYSHIMMER_AVAILABLE', False)
    def test_check_gsr_sensor_no_pyshimmer(self, mock_config):
        """Test that check_gsr_sensor returns False when pyshimmer is not available."""
        # Mock the config values
        mock_config.GSR_SIMULATION_MODE = False
        mock_config.GSR_SENSOR_PORT = "/dev/ttyUSB0"
        
        # Call the function
        result = check_gsr_sensor()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.config')
    def test_check_gsr_sensor_simulation_mode(self, mock_config):
        """Test that check_gsr_sensor returns True when in simulation mode."""
        # Mock the config values
        mock_config.GSR_SIMULATION_MODE = True
        
        # Call the function
        result = check_gsr_sensor()
        
        # Check the result
        self.assertTrue(result)
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.FLIR_AVAILABLE', True)
    @patch('src.system.validation.check_system.PySpin')
    def test_check_flir_cameras_success(self, mock_pyspin, mock_config):
        """Test that check_flir_cameras returns True when FLIR cameras are detected."""
        # Mock the config values
        mock_config.THERMAL_SIMULATION_MODE = False
        mock_config.THERMAL_CAMERA_ID = 0
        
        # Mock the PySpin System
        mock_system = MagicMock()
        mock_pyspin.System.GetInstance.return_value = mock_system
        
        # Mock the camera list
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        mock_cam_list.GetSize.return_value = 1
        
        # Mock the camera
        mock_camera = MagicMock()
        mock_cam_list.GetByIndex.return_value = mock_camera
        
        # Mock the node map
        mock_nodemap = MagicMock()
        mock_camera.GetTLDeviceNodeMap.return_value = mock_nodemap
        
        # Mock the device name and ID nodes
        mock_name_node = MagicMock()
        mock_id_node = MagicMock()
        mock_nodemap.GetNode.side_effect = lambda x: mock_name_node if x == 'DeviceModelName' else mock_id_node
        
        # Mock the IsAvailable and IsReadable functions
        mock_pyspin.IsAvailable.return_value = True
        mock_pyspin.IsReadable.return_value = True
        
        # Mock the GetValue functions
        mock_name_node.GetValue.return_value = "FLIR A65"
        mock_id_node.GetValue.return_value = "12345"
        
        # Call the function
        result = check_flir_cameras()
        
        # Check the result
        self.assertTrue(result)
        
        # Check that the camera list was cleared and the system was released
        mock_cam_list.Clear.assert_called_once()
        mock_system.ReleaseInstance.assert_called_once()
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.FLIR_AVAILABLE', True)
    @patch('src.system.validation.check_system.PySpin')
    def test_check_flir_cameras_no_cameras(self, mock_pyspin, mock_config):
        """Test that check_flir_cameras returns False when no FLIR cameras are detected."""
        # Mock the config values
        mock_config.THERMAL_SIMULATION_MODE = False
        
        # Mock the PySpin System
        mock_system = MagicMock()
        mock_pyspin.System.GetInstance.return_value = mock_system
        
        # Mock the camera list with no cameras
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        mock_cam_list.GetSize.return_value = 0
        
        # Call the function
        result = check_flir_cameras()
        
        # Check the result
        self.assertFalse(result)
        
        # Check that the camera list was cleared and the system was released
        mock_cam_list.Clear.assert_called_once()
        mock_system.ReleaseInstance.assert_called_once()
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.FLIR_AVAILABLE', True)
    @patch('src.system.validation.check_system.PySpin')
    def test_check_flir_cameras_invalid_index(self, mock_pyspin, mock_config):
        """Test that check_flir_cameras returns False when the camera index is invalid."""
        # Mock the config values
        mock_config.THERMAL_SIMULATION_MODE = False
        mock_config.THERMAL_CAMERA_ID = 1  # Invalid index (only 1 camera available)
        
        # Mock the PySpin System
        mock_system = MagicMock()
        mock_pyspin.System.GetInstance.return_value = mock_system
        
        # Mock the camera list with one camera
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        mock_cam_list.GetSize.return_value = 1
        
        # Call the function
        result = check_flir_cameras()
        
        # Check the result
        self.assertFalse(result)
        
        # Check that the camera list was cleared and the system was released
        mock_cam_list.Clear.assert_called_once()
        mock_system.ReleaseInstance.assert_called_once()
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.FLIR_AVAILABLE', True)
    @patch('src.system.validation.check_system.PySpin')
    def test_check_flir_cameras_exception(self, mock_pyspin, mock_config):
        """Test that check_flir_cameras returns False when an exception occurs."""
        # Mock the config values
        mock_config.THERMAL_SIMULATION_MODE = False
        
        # Mock the PySpin System to raise an exception
        mock_pyspin.System.GetInstance.side_effect = Exception("Test exception")
        
        # Call the function
        result = check_flir_cameras()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.FLIR_AVAILABLE', False)
    def test_check_flir_cameras_no_pyspin(self, mock_config):
        """Test that check_flir_cameras returns False when PySpin is not available."""
        # Mock the config values
        mock_config.THERMAL_SIMULATION_MODE = False
        
        # Call the function
        result = check_flir_cameras()
        
        # Check the result
        self.assertFalse(result)
    
    @patch('src.system.validation.check_system.config')
    def test_check_flir_cameras_simulation_mode(self, mock_config):
        """Test that check_flir_cameras returns True when in simulation mode."""
        # Mock the config values
        mock_config.THERMAL_SIMULATION_MODE = True
        
        # Call the function
        result = check_flir_cameras()
        
        # Check the result
        self.assertTrue(result)
    
    @patch('src.system.validation.check_system.check_dependencies')
    @patch('src.system.validation.check_system.check_directories')
    @patch('src.system.validation.check_system.check_camera')
    @patch('src.system.validation.check_system.list_serial_ports')
    @patch('src.system.validation.check_system.check_flir_cameras')
    @patch('src.system.validation.check_system.check_gsr_sensor')
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.sys.exit')
    def test_main_success(self, mock_exit, mock_config, mock_gsr, mock_flir, mock_ports, mock_camera, mock_dirs, mock_deps):
        """Test that main exits with code 0 when all checks pass."""
        # Mock the config values
        mock_config.RGB_CAMERA_ID = 0
        mock_config.THERMAL_SIMULATION_MODE = False
        mock_config.GSR_SIMULATION_MODE = False
        
        # Mock the check functions to return success
        mock_deps.return_value = True
        mock_dirs.return_value = True
        mock_camera.return_value = True
        mock_ports.return_value = ["/dev/ttyUSB0"]
        mock_flir.return_value = True
        mock_gsr.return_value = True
        
        # Call the function
        main()
        
        # Check that the exit code is 0 (success)
        mock_exit.assert_called_once_with(0)
    
    @patch('src.system.validation.check_system.check_dependencies')
    @patch('src.system.validation.check_system.check_directories')
    @patch('src.system.validation.check_system.check_camera')
    @patch('src.system.validation.check_system.list_serial_ports')
    @patch('src.system.validation.check_system.check_flir_cameras')
    @patch('src.system.validation.check_system.check_gsr_sensor')
    @patch('src.system.validation.check_system.config')
    @patch('src.system.validation.check_system.sys.exit')
    def test_main_failure(self, mock_exit, mock_config, mock_gsr, mock_flir, mock_ports, mock_camera, mock_dirs, mock_deps):
        """Test that main exits with code 1 when a check fails."""
        # Mock the config values
        mock_config.RGB_CAMERA_ID = 0
        mock_config.THERMAL_SIMULATION_MODE = False
        mock_config.GSR_SIMULATION_MODE = False
        
        # Mock the check functions to return failure for one check
        mock_deps.return_value = True
        mock_dirs.return_value = True
        mock_camera.return_value = False  # RGB camera check fails
        mock_ports.return_value = ["/dev/ttyUSB0"]
        mock_flir.return_value = True
        mock_gsr.return_value = True
        
        # Call the function
        main()
        
        # Check that the exit code is 1 (failure)
        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()