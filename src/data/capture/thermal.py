"""Thermal camera capture implementation.

This module provides a concrete implementation of the BaseCaptureThread
for capturing thermal video data from compatible thermal cameras.
"""

from __future__ import annotations

import logging
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .base import BaseCaptureThread, CaptureState
from ...core.exceptions import DeviceError, CaptureError


class ThermalCapture(BaseCaptureThread):
    """Capture implementation for thermal camera data.
    
    This class handles the capture of thermal video data from compatible
    thermal cameras. It supports both real hardware capture and simulation
    mode for testing.
    
    Attributes:
        device_name: Human-readable name of the capture device
        simulation_mode: Whether to run in simulation mode
        device_id: Camera device ID or path to video file
        frame_width: Width of the captured frames
        frame_height: Height of the captured frames
        fps: Frames per second for capture
    """
    
    # Signal for new frame data
    frame_available = None  # Will be initialized in __init__
    
    def __init__(
        self,
        device_name: str = "Thermal",
        simulation_mode: bool = False,
        device_id: Union[int, str] = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        fps: int = 30,
        **kwargs: Any,
    ) -> None:
        """Initialize the thermal capture.
        
        Args:
            device_name: Human-readable name of the capture device
            simulation_mode: Whether to run in simulation mode
            device_id: Camera device ID (int) or path to video file (str)
            frame_width: Width of the captured frames
            frame_height: Height of the captured frames
            fps: Frames per second for capture
            **kwargs: Additional configuration parameters
        """
        super().__init__(device_name, simulation_mode, **kwargs)
        
        # Initialize the frame_available signal with the correct type
        if hasattr(self, 'pyqtSignal'):
            # Define the signal with timestamp, frame, and metadata
            self.frame_available = self.pyqtSignal(int, object, object)
        
        self.device_id = device_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Internal state
        self._camera = None
        self._running = False
        self._last_frame = None
        self._frame_count = 0
        
        self._logger = logging.getLogger(f"{__name__}.{device_name}")
        self._logger.info(
            f"Thermal capture initialized with resolution {frame_width}x{frame_height} at {fps} FPS"
        )
    
    def initialize(self) -> None:
        """Initialize the thermal camera.
        
        This method sets up the connection to the thermal camera or
        prepares the simulation environment.
        
        Raises:
            DeviceError: If device initialization fails
        """
        try:
            self._logger.info(f"Initializing {self.device_name} capture")
            
            if not self.simulation_mode:
                if not CV2_AVAILABLE:
                    raise DeviceError(
                        "OpenCV library not available",
                        "THERMAL001",
                        {"solution": "Install opencv-python package"}
                    )
                
                # Initialize the camera
                self._logger.info(f"Opening thermal camera with ID {self.device_id}")
                self._camera = cv2.VideoCapture(self.device_id)
                
                if not self._camera.isOpened():
                    raise DeviceError(
                        f"Failed to open thermal camera with ID {self.device_id}",
                        "THERMAL002",
                        {"device_id": self.device_id}
                    )
                
                # Set camera properties
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self._camera.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Verify settings
                actual_width = self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self._camera.get(cv2.CAP_PROP_FPS)
                
                self._logger.info(
                    f"Thermal camera initialized with resolution {actual_width}x{actual_height} "
                    f"at {actual_fps} FPS"
                )
            else:
                self._logger.info("Initializing thermal simulation")
                # For simulation, we don't need to connect to a real device
            
            # Reset frame count
            self._frame_count = 0
            self._last_frame = None
            
        except Exception as e:
            error_msg = f"Failed to initialize thermal camera: {e}"
            self._logger.error(error_msg, exc_info=True)
            raise DeviceError(error_msg, "THERMAL003", {"device_id": self.device_id})
    
    def stop_capture(self) -> None:
        """Stop capturing thermal data.
        
        This method signals the capture thread to stop and cleans up
        the camera connection.
        """
        self._logger.info("Stopping thermal capture")
        self._running = False
    
    def cleanup(self) -> None:
        """Clean up thermal camera resources.
        
        This method releases the camera and any acquired resources.
        """
        self._logger.info("Cleaning up thermal resources")
        
        if not self.simulation_mode and self._camera is not None:
            try:
                self._camera.release()
                self._logger.info("Released thermal camera")
            except Exception as e:
                self._logger.error(f"Error releasing thermal camera: {e}")
            
            self._camera = None
        
        self._last_frame = None
    
    def is_available(self) -> bool:
        """Check if the thermal camera is available.
        
        Returns:
            True if the camera is available or in simulation mode, False otherwise
        """
        if self.simulation_mode:
            return True
        
        if not CV2_AVAILABLE:
            return False
        
        # Check if camera is initialized and opened
        return self._camera is not None and self._camera.isOpened()
    
    def _create_simulated_thermal_frame(self) -> np.ndarray:
        """Create a simulated thermal frame.
        
        Returns:
            Numpy array representing a simulated thermal image
        """
        if not CV2_AVAILABLE or not np:
            # Create a simple placeholder if numpy/cv2 not available
            return "Simulated thermal frame"
        
        # Create a base thermal image (grayscale with temperature gradient)
        frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # Add a temperature gradient (hotter in the center, cooler at edges)
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(self.frame_height):
            for x in range(self.frame_width):
                # Calculate distance from center (normalized to 0-1)
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
                # Invert and scale to 0-255 (center is hotter/brighter)
                temp = int(255 * (1 - dist))
                frame[y, x] = temp
        
        # Add some random noise
        noise = np.random.normal(0, 10, frame.shape).astype(np.int8)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        
        # Add a simulated "hot spot" that moves over time
        t = time.time()
        spot_x = int(center_x + center_x * 0.5 * np.sin(t * 0.5))
        spot_y = int(center_y + center_y * 0.5 * np.cos(t * 0.3))
        
        # Draw the hot spot (a bright circle)
        cv2.circle(frame, (spot_x, spot_y), 30, 255, -1)
        cv2.circle(frame, (spot_x, spot_y), 20, 200, -1)
        
        # Convert to color map for visualization (inferno colormap)
        frame_colored = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
        
        return frame_colored
    
    def _run_simulation(self) -> None:
        """Run thermal capture in simulation mode.
        
        This method generates simulated thermal frames for testing purposes.
        """
        self._logger.info("Starting thermal simulation")
        self._running = True
        
        # Time between frames (in seconds)
        frame_interval = 1.0 / self.fps
        
        try:
            while self._running:
                # Get current timestamp
                timestamp = self._get_timestamp()
                
                # Generate simulated thermal frame
                frame = self._create_simulated_thermal_frame()
                
                # Store the frame
                self._last_frame = frame
                self._frame_count += 1
                
                # Create metadata
                metadata = {
                    "timestamp": timestamp,
                    "frame_number": self._frame_count,
                    "temperature_min": 20.0 + random.uniform(-1.0, 1.0),  # Simulated min temp
                    "temperature_max": 35.0 + random.uniform(-1.0, 1.0),  # Simulated max temp
                }
                
                # Emit the frame if signal is available
                if hasattr(self, 'frame_available') and self.frame_available is not None:
                    self.frame_available.emit(timestamp, frame, metadata)
                
                # Sleep to maintain the frame rate
                time.sleep(frame_interval)
                
        except Exception as e:
            error_msg = f"Error in thermal simulation: {e}"
            self._logger.error(error_msg, exc_info=True)
            if not self._running:
                # If we're stopping, this is expected
                return
            raise CaptureError(error_msg, "THERMAL004")
    
    def _run_real_capture(self) -> None:
        """Run thermal capture with real hardware.
        
        This method captures frames from an actual thermal camera.
        """
        if not CV2_AVAILABLE:
            raise DeviceError(
                "OpenCV library not available",
                "THERMAL001",
                {"solution": "Install opencv-python package"}
            )
        
        if self._camera is None or not self._camera.isOpened():
            raise DeviceError(
                "Thermal camera not initialized or failed to open",
                "THERMAL005",
                {"device_id": self.device_id}
            )
        
        self._logger.info("Starting real thermal capture")
        self._running = True
        
        try:
            while self._running:
                # Get current timestamp
                timestamp = self._get_timestamp()
                
                # Capture frame from camera
                ret, frame = self._camera.read()
                
                if not ret:
                    raise CaptureError(
                        "Failed to read frame from thermal camera",
                        "THERMAL006",
                        {"device_id": self.device_id}
                    )
                
                # Store the frame
                self._last_frame = frame
                self._frame_count += 1
                
                # Create metadata (in a real implementation, this would include actual temperature data)
                metadata = {
                    "timestamp": timestamp,
                    "frame_number": self._frame_count,
                }
                
                # Emit the frame if signal is available
                if hasattr(self, 'frame_available') and self.frame_available is not None:
                    self.frame_available.emit(timestamp, frame, metadata)
                
                # Calculate processing time and sleep if needed to maintain frame rate
                elapsed = (self._get_timestamp() - timestamp) / 1e9  # Convert ns to seconds
                sleep_time = max(0, (1.0 / self.fps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            error_msg = f"Error in thermal capture: {e}"
            self._logger.error(error_msg, exc_info=True)
            if not self._running:
                # If we're stopping, this is expected
                return
            raise CaptureError(error_msg, "THERMAL007")
    
    def get_last_frame(self) -> Tuple[Optional[Any], int]:
        """Get the most recently captured frame.
        
        Returns:
            Tuple of (frame, frame_number) or (None, 0) if no frame is available
        """
        return (self._last_frame, self._frame_count) if self._last_frame is not None else (None, 0)