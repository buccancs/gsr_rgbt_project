import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import time
import numpy as np

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Mock PySpin module
mock_pyspin = MagicMock()
mock_pyspin.AcquisitionMode_Continuous = 1
mock_pyspin.RW = 2
sys.modules['PySpin'] = mock_pyspin

from src.capture.thermal_capture import ThermalCaptureThread


class TestThermalCaptureThread(unittest.TestCase):
    """Test suite for the ThermalCaptureThread class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_index = 0
        self.fps = 30
    
    def test_initialization(self):
        """Test that the thread initializes correctly."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        self.assertEqual(thread.device_name, "Thermal")
        self.assertEqual(thread.camera_index, self.camera_index)
        self.assertEqual(thread.fps, self.fps)
        self.assertFalse(thread.simulation_mode)
        self.assertIsNone(thread.camera)
        self.assertIsNone(thread.system)
    
    def test_initialization_simulation_mode(self):
        """Test that the thread initializes correctly in simulation mode."""
        thread = ThermalCaptureThread(self.camera_index, self.fps, simulation_mode=True)
        
        self.assertEqual(thread.device_name, "Thermal")
        self.assertEqual(thread.camera_index, self.camera_index)
        self.assertEqual(thread.fps, self.fps)
        self.assertTrue(thread.simulation_mode)
        self.assertIsNone(thread.camera)
        self.assertIsNone(thread.system)
    
    @patch('src.capture.thermal_capture.time.sleep')
    def test_run_simulation(self, mock_sleep):
        """Test that the thread runs correctly in simulation mode."""
        thread = ThermalCaptureThread(self.camera_index, self.fps, simulation_mode=True)
        
        # Mock the emit method to track calls
        thread.frame_captured = MagicMock()
        
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
        thread.frame_captured.emit.assert_called()
        
        # Check that the first argument is a numpy array (frame)
        args, _ = thread.frame_captured.emit.call_args
        self.assertIsInstance(args[0], np.ndarray)
        
        # Check that the frame has the expected shape and type
        self.assertEqual(args[0].shape, (480, 640, 3))
        self.assertEqual(args[0].dtype, np.uint8)
        
        # Check that the second argument is the timestamp
        self.assertEqual(args[1], 12345)
    
    @patch('src.capture.thermal_capture.FLIR_AVAILABLE', True)
    @patch('src.capture.thermal_capture.PySpin')
    def test_run_real_capture(self, mock_pyspin):
        """Test that the thread runs correctly with real hardware."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        # Mock the emit method to track calls
        thread.frame_captured = MagicMock()
        
        # Mock the System instance
        mock_system = MagicMock()
        mock_pyspin.System.GetInstance.return_value = mock_system
        
        # Mock the camera list
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        mock_cam_list.GetSize.return_value = 1
        
        # Mock the camera
        mock_camera = MagicMock()
        mock_cam_list.GetByIndex.return_value = mock_camera
        
        # Mock the camera's acquisition mode
        mock_camera.AcquisitionMode = MagicMock()
        
        # Mock the camera's frame rate settings
        mock_camera.AcquisitionFrameRate = MagicMock()
        mock_camera.AcquisitionFrameRate.GetAccessMode.return_value = mock_pyspin.RW
        mock_camera.AcquisitionFrameRateEnable = MagicMock()
        
        # Mock the image acquisition
        mock_image = MagicMock()
        mock_camera.GetNextImage.return_value = mock_image
        mock_image.IsIncomplete.return_value = False
        
        # Create a simple test image
        test_image_data = np.zeros((480, 640), dtype=np.uint16)
        # Add some patterns to the test image
        for i in range(480):
            for j in range(640):
                test_image_data[i, j] = (i + j) % 256
        
        # Mock the image data
        mock_image.GetData.return_value = test_image_data.flatten()
        mock_image.GetHeight.return_value = 480
        mock_image.GetWidth.return_value = 640
        
        # Set is_running to True initially, then False after one iteration
        thread.is_running = True
        
        def get_next_image_side_effect(*args, **kwargs):
            thread.is_running = False
            return mock_image
        
        mock_camera.GetNextImage.side_effect = get_next_image_side_effect
        
        # Run the real capture method directly
        thread._run_real_capture()
        
        # Check that the camera was initialized correctly
        mock_pyspin.System.GetInstance.assert_called_once()
        mock_system.GetCameras.assert_called_once()
        mock_cam_list.GetByIndex.assert_called_once_with(self.camera_index)
        mock_camera.Init.assert_called_once()
        mock_camera.AcquisitionMode.SetValue.assert_called_once_with(mock_pyspin.AcquisitionMode_Continuous)
        mock_camera.AcquisitionFrameRateEnable.SetValue.assert_called_once_with(True)
        mock_camera.AcquisitionFrameRate.SetValue.assert_called_once_with(float(self.fps))
        mock_camera.BeginAcquisition.assert_called_once()
        
        # Check that the emit method was called at least once
        thread.frame_captured.emit.assert_called()
        
        # Check that the first argument is a numpy array (frame)
        args, _ = thread.frame_captured.emit.call_args
        self.assertIsInstance(args[0], np.ndarray)
        
        # Check that the frame has the expected shape and type
        self.assertEqual(args[0].shape, (480, 640, 3))
        self.assertEqual(args[0].dtype, np.uint8)
    
    @patch('src.capture.thermal_capture.FLIR_AVAILABLE', False)
    @patch('src.capture.thermal_capture.logging')
    def test_run_real_capture_no_pyspin(self, mock_logging):
        """Test that the thread handles the case where PySpin is not available."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        # Run the real capture method directly
        thread._run_real_capture()
        
        # Check that an error was logged
        mock_logging.error.assert_called_once_with(
            "FLIR camera mode is selected, but the 'PySpin' library is not available."
        )
    
    @patch('src.capture.thermal_capture.FLIR_AVAILABLE', True)
    @patch('src.capture.thermal_capture.PySpin')
    def test_run_real_capture_no_cameras(self, mock_pyspin):
        """Test that the thread handles the case where no cameras are detected."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        # Mock the System instance
        mock_system = MagicMock()
        mock_pyspin.System.GetInstance.return_value = mock_system
        
        # Mock the camera list with no cameras
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        mock_cam_list.GetSize.return_value = 0
        
        # Run the real capture method directly
        with patch('src.capture.thermal_capture.logging') as mock_logging:
            thread._run_real_capture()
            
            # Check that an error was logged
            mock_logging.error.assert_called_once_with("No FLIR cameras detected.")
    
    @patch('src.capture.thermal_capture.FLIR_AVAILABLE', True)
    @patch('src.capture.thermal_capture.PySpin')
    def test_run_real_capture_camera_index_out_of_range(self, mock_pyspin):
        """Test that the thread handles the case where the camera index is out of range."""
        thread = ThermalCaptureThread(1, self.fps)  # Camera index 1
        
        # Mock the System instance
        mock_system = MagicMock()
        mock_pyspin.System.GetInstance.return_value = mock_system
        
        # Mock the camera list with only one camera
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        mock_cam_list.GetSize.return_value = 1  # Only one camera available
        
        # Run the real capture method directly
        with patch('src.capture.thermal_capture.logging') as mock_logging:
            thread._run_real_capture()
            
            # Check that an error was logged
            mock_logging.error.assert_called_once_with(
                "Camera index 1 is out of range. Only 1 cameras detected."
            )
    
    @patch('src.capture.thermal_capture.FLIR_AVAILABLE', True)
    @patch('src.capture.thermal_capture.PySpin')
    def test_run_real_capture_exception(self, mock_pyspin):
        """Test that the thread handles exceptions during real capture."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        # Mock the System instance to raise an exception
        mock_pyspin.System.GetInstance.side_effect = Exception("Test exception")
        
        # Run the real capture method directly
        with patch('src.capture.thermal_capture.logging') as mock_logging:
            thread._run_real_capture()
            
            # Check that an error was logged
            mock_logging.error.assert_called_once_with(
                "Error initializing FLIR camera: Test exception"
            )
    
    @patch('src.capture.thermal_capture.FLIR_AVAILABLE', True)
    @patch('src.capture.thermal_capture.PySpin')
    def test_cleanup(self, mock_pyspin):
        """Test that the cleanup method correctly cleans up resources."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        # Mock the camera and system
        mock_camera = MagicMock()
        mock_camera.IsStreaming.return_value = True
        thread.camera = mock_camera
        
        mock_system = MagicMock()
        mock_cam_list = MagicMock()
        mock_system.GetCameras.return_value = mock_cam_list
        thread.system = mock_system
        
        # Call the cleanup method
        thread._cleanup()
        
        # Check that the camera was stopped and deinitialized
        mock_camera.EndAcquisition.assert_called_once()
        mock_camera.DeInit.assert_called_once()
        
        # Check that the system was released
        mock_cam_list.Clear.assert_called_once()
        mock_system.ReleaseInstance.assert_called_once()
        
        # Check that the camera and system were set to None
        self.assertIsNone(thread.camera)
        self.assertIsNone(thread.system)
    
    def test_cleanup_no_devices(self):
        """Test that the cleanup method handles the case where there are no devices."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        thread.camera = None
        thread.system = None
        
        # Call the cleanup method
        thread._cleanup()
        
        # No assertions needed, just checking that it doesn't raise an exception
    
    def test_sleep(self):
        """Test that the _sleep method sleeps for the specified time."""
        thread = ThermalCaptureThread(self.camera_index, self.fps)
        
        with patch('src.capture.thermal_capture.time.sleep') as mock_sleep:
            thread._sleep(0.1)
            mock_sleep.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()