# src/tests/unit/data_collection/capture/test_thermal_capture.py

import unittest
from unittest.mock import MagicMock, patch
import time
import numpy as np

from src.data_collection.capture.thermal_capture import ThermalCaptureThread

# Mock the PySpin module since it might not be available in all environments
class MockPySpin:
    class CameraList:
        def __init__(self, system):
            self.cameras = [MagicMock() for _ in range(2)]
            
        def __getitem__(self, index):
            return self.cameras[index]
            
        def __len__(self):
            return len(self.cameras)
            
        def Clear(self):
            pass
    
    class System:
        def GetInstance(self):
            return self
            
        def GetCameras(self):
            return MockPySpin.CameraList(self)
            
        def ReleaseInstance(self):
            pass
    
    class CameraModes:
        DEFAULT = 0
    
    class PixelFormat:
        Mono16 = 1
    
    class AcquisitionMode:
        Continuous = 1

# Create the patch for PySpin
pyspin_patch = patch('src.data_collection.capture.thermal_capture.PySpin', MockPySpin())

class TestThermalCaptureThread(unittest.TestCase):
    """
    Unit tests for the ThermalCaptureThread class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.camera_index = 0
        self.fps = 30
        
        # Start the PySpin patch
        self.pyspin_patcher = pyspin_patch
        self.mock_pyspin = self.pyspin_patcher.start()
        
    def tearDown(self):
        """
        Clean up after each test method.
        """
        # Stop the PySpin patch
        self.pyspin_patcher.stop()
        
    def test_initialization(self):
        """
        Test that the ThermalCaptureThread initializes correctly.
        """
        thread = ThermalCaptureThread(
            camera_index=self.camera_index,
            fps=self.fps,
            simulation_mode=True
        )
        
        self.assertEqual(thread.device_name, "Thermal")
        self.assertEqual(thread.camera_index, self.camera_index)
        self.assertEqual(thread.fps, self.fps)
        self.assertTrue(thread.simulation_mode)
        self.assertFalse(thread.is_running)
        self.assertIsNone(thread.camera)
        self.assertIsNone(thread.system)
        
    @patch('src.data_collection.capture.thermal_capture.logging')
    @patch('src.data_collection.capture.thermal_capture.np.random.normal')
    def test_run_simulation(self, mock_random_normal, mock_logging):
        """
        Test that the _run_simulation method generates and emits simulated thermal data.
        """
        # Mock the random normal distribution to return a predictable array
        mock_frame = np.ones((480, 640)) * 30.0  # 30 degrees C
        mock_random_normal.return_value = mock_frame
        
        thread = ThermalCaptureThread(
            camera_index=self.camera_index,
            fps=self.fps,
            simulation_mode=True
        )
        
        # Create a mock to track the thermal_frame signal
        mock_callback = MagicMock()
        thread.thermal_frame.connect(mock_callback)
        
        # Mock the _sleep method to avoid actual sleeping
        thread._sleep = MagicMock()
        
        # Set is_running to True initially, then False after a few iterations
        thread.is_running = True
        
        def side_effect(*args, **kwargs):
            thread.is_running = False
            
        thread._sleep.side_effect = side_effect
        
        # Run the simulation
        thread._run_simulation()
        
        # Check that the thermal_frame signal was emitted
        mock_callback.assert_called()
        
        # Check that the arguments are of the correct type
        args, _ = mock_callback.call_args
        frame, timestamp = args
        self.assertIsInstance(frame, np.ndarray)
        self.assertIsInstance(timestamp, (int, float))
        
        # Check that logging was called
        mock_logging.info.assert_any_call(
            f"Running in thermal camera simulation mode. FPS: {self.fps}"
        )
        
    @patch('src.data_collection.capture.thermal_capture.PySpin', None)
    @patch('src.data_collection.capture.thermal_capture.logging')
    def test_run_real_capture_without_pyspin(self, mock_logging):
        """
        Test that _run_real_capture handles the case when PySpin is not available.
        """
        thread = ThermalCaptureThread(
            camera_index=self.camera_index,
            fps=self.fps,
            simulation_mode=False
        )
        
        thread._run_real_capture()
        
        mock_logging.error.assert_called_with(
            "Real thermal camera mode is selected, but the 'PySpin' library is not available."
        )
        
    @patch('src.data_collection.capture.thermal_capture.logging')
    def test_run_real_capture_with_pyspin(self, mock_logging):
        """
        Test that _run_real_capture correctly interacts with the PySpin library.
        """
        thread = ThermalCaptureThread(
            camera_index=self.camera_index,
            fps=self.fps,
            simulation_mode=False
        )
        
        # Create a mock to track the thermal_frame signal
        mock_callback = MagicMock()
        thread.thermal_frame.connect(mock_callback)
        
        # Mock the camera's GetNextImage method to return a mock image
        mock_image = MagicMock()
        mock_image.GetHeight.return_value = 480
        mock_image.GetWidth.return_value = 640
        mock_image.GetData.return_value = np.ones(480 * 640, dtype=np.uint16)
        
        # Set up the camera mock to return our mock image and then set is_running to False
        mock_camera = self.mock_pyspin.CameraList(None)[self.camera_index]
        
        def get_next_image_side_effect():
            thread.is_running = False
            return mock_image
            
        mock_camera.GetNextImage.side_effect = get_next_image_side_effect
        
        # Set is_running to True initially
        thread.is_running = True
        
        # Run the real capture
        thread._run_real_capture()
        
        # Check that the camera was configured correctly
        mock_camera.Init.assert_called_once()
        
        # Check that the thermal_frame signal was emitted
        mock_callback.assert_called()
        
        # Check that the arguments are of the correct type
        args, _ = mock_callback.call_args
        frame, timestamp = args
        self.assertIsInstance(frame, np.ndarray)
        self.assertIsInstance(timestamp, (int, float))
        
    @patch('src.data_collection.capture.thermal_capture.logging')
    def test_cleanup(self, mock_logging):
        """
        Test that the _cleanup method properly cleans up resources.
        """
        thread = ThermalCaptureThread(
            camera_index=self.camera_index,
            fps=self.fps,
            simulation_mode=False
        )
        
        # Create mock camera and system
        mock_camera = MagicMock()
        mock_system = MagicMock()
        
        thread.camera = mock_camera
        thread.system = mock_system
        
        # Call cleanup
        thread._cleanup()
        
        # Check that the camera and system were properly closed
        mock_camera.DeInit.assert_called_once()
        mock_camera.Release.assert_called_once()
        mock_system.ReleaseInstance.assert_called_once()
        mock_logging.info.assert_called_with("Thermal camera resources released.")
        
    def test_sleep(self):
        """
        Test that the _sleep method sleeps for the specified time.
        """
        thread = ThermalCaptureThread(
            camera_index=self.camera_index,
            fps=self.fps,
            simulation_mode=True
        )
        
        # Mock time.sleep to avoid actual sleeping
        with patch('time.sleep') as mock_sleep:
            thread._sleep(0.5)
            mock_sleep.assert_called_with(0.5)

if __name__ == '__main__':
    unittest.main()