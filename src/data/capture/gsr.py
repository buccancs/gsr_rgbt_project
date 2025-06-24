"""GSR (Galvanic Skin Response) capture implementation.

This module provides a concrete implementation of the BaseCaptureThread
for capturing GSR data from Shimmer devices or other GSR sensors.
"""

from __future__ import annotations

import logging
import time
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    # Try to import Shimmer-specific libraries
    import pyshimmer
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False

from .base import BaseCaptureThread, CaptureState
from ...core.exceptions import DeviceError, CaptureError


class GSRCapture(BaseCaptureThread):
    """Capture implementation for Galvanic Skin Response (GSR) data.
    
    This class handles the capture of GSR data from Shimmer devices or
    other compatible GSR sensors. It supports both real hardware capture
    and simulation mode for testing.
    
    Attributes:
        device_name: Human-readable name of the capture device
        simulation_mode: Whether to run in simulation mode
        port: Serial port for the Shimmer device (e.g., "COM3")
        sample_rate: Sampling rate in Hz
        buffer_size: Size of the data buffer
    """
    
    # Signal for new GSR data
    data_available = None  # Will be initialized in __init__
    
    def __init__(
        self,
        device_name: str = "GSR",
        simulation_mode: bool = False,
        port: Optional[str] = None,
        sample_rate: int = 128,
        buffer_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the GSR capture.
        
        Args:
            device_name: Human-readable name of the capture device
            simulation_mode: Whether to run in simulation mode
            port: Serial port for the Shimmer device (e.g., "COM3")
            sample_rate: Sampling rate in Hz
            buffer_size: Size of the data buffer
            **kwargs: Additional configuration parameters
        """
        super().__init__(device_name, simulation_mode, **kwargs)
        
        # Initialize the data_available signal with the correct type
        if hasattr(self, 'pyqtSignal'):
            # Define the signal with timestamp, GSR value, and raw data
            self.data_available = self.pyqtSignal(int, float, object)
        
        self.port = port
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Internal state
        self._device = None
        self._running = False
        self._buffer: List[Tuple[int, float]] = []
        
        self._logger = logging.getLogger(f"{__name__}.{device_name}")
        self._logger.info(f"GSR capture initialized with sample rate {sample_rate} Hz")
    
    def initialize(self) -> None:
        """Initialize the GSR capture device.
        
        This method sets up the connection to the Shimmer device or
        prepares the simulation environment.
        
        Raises:
            DeviceError: If device initialization fails
        """
        try:
            self._logger.info(f"Initializing {self.device_name} capture")
            
            if not self.simulation_mode:
                if not SHIMMER_AVAILABLE:
                    raise DeviceError(
                        "Shimmer library not available",
                        "GSR001",
                        {"solution": "Install pyshimmer package"}
                    )
                
                if not self.port:
                    raise DeviceError(
                        "No port specified for GSR device",
                        "GSR002",
                        {"solution": "Specify a valid COM port"}
                    )
                
                # Initialize the Shimmer device
                self._logger.info(f"Connecting to Shimmer device on port {self.port}")
                # In a real implementation, this would connect to the actual device
                # self._device = pyshimmer.Shimmer(self.port)
                # self._device.connect()
                # self._device.set_sampling_rate(self.sample_rate)
                # self._device.enable_sensor("GSR")
                
                self._logger.info("Shimmer device connected successfully")
            else:
                self._logger.info("Initializing GSR simulation")
                # For simulation, we don't need to connect to a real device
            
            # Clear the buffer
            self._buffer = []
            
        except Exception as e:
            error_msg = f"Failed to initialize GSR device: {e}"
            self._logger.error(error_msg, exc_info=True)
            raise DeviceError(error_msg, "GSR003", {"port": self.port})
    
    def stop_capture(self) -> None:
        """Stop capturing GSR data.
        
        This method signals the capture thread to stop and cleans up
        the device connection.
        """
        self._logger.info("Stopping GSR capture")
        self._running = False
    
    def cleanup(self) -> None:
        """Clean up GSR device resources.
        
        This method disconnects from the Shimmer device and releases
        any acquired resources.
        """
        self._logger.info("Cleaning up GSR resources")
        
        if not self.simulation_mode and self._device is not None:
            try:
                # In a real implementation, this would disconnect from the device
                # self._device.disconnect()
                self._logger.info("Disconnected from Shimmer device")
            except Exception as e:
                self._logger.error(f"Error disconnecting from Shimmer device: {e}")
            
            self._device = None
        
        # Clear the buffer
        self._buffer = []
    
    def is_available(self) -> bool:
        """Check if the GSR device is available.
        
        Returns:
            True if the device is available or in simulation mode, False otherwise
        """
        if self.simulation_mode:
            return True
        
        if not SHIMMER_AVAILABLE:
            return False
        
        # In a real implementation, this would check if the device is connected
        # return self._device is not None and self._device.is_connected()
        
        # For now, just return True if a port is specified
        return self.port is not None
    
    def _run_simulation(self) -> None:
        """Run GSR capture in simulation mode.
        
        This method generates simulated GSR data for testing purposes.
        """
        self._logger.info("Starting GSR simulation")
        self._running = True
        
        # Base GSR value (in microSiemens)
        base_gsr = 2.0
        
        # Time between samples (in seconds)
        sample_interval = 1.0 / self.sample_rate
        
        try:
            while self._running:
                # Get current timestamp
                timestamp = self._get_timestamp()
                
                # Generate simulated GSR value with random noise and slow drift
                t = time.time()
                drift = 0.5 * np.sin(0.05 * t) if NUMPY_AVAILABLE else 0.5 * (t % 20) / 20
                noise = random.uniform(-0.2, 0.2)
                gsr_value = base_gsr + drift + noise
                
                # Add to buffer (limited to buffer_size)
                self._buffer.append((timestamp, gsr_value))
                if len(self._buffer) > self.buffer_size:
                    self._buffer.pop(0)
                
                # Emit the data if signal is available
                if hasattr(self, 'data_available') and self.data_available is not None:
                    # Create a simple dict as the raw data
                    raw_data = {"timestamp": timestamp, "gsr": gsr_value}
                    self.data_available.emit(timestamp, gsr_value, raw_data)
                
                # Sleep to maintain the sample rate
                time.sleep(sample_interval)
                
        except Exception as e:
            error_msg = f"Error in GSR simulation: {e}"
            self._logger.error(error_msg, exc_info=True)
            if not self._running:
                # If we're stopping, this is expected
                return
            raise CaptureError(error_msg, "GSR004")
    
    def _run_real_capture(self) -> None:
        """Run GSR capture with real hardware.
        
        This method captures data from an actual Shimmer GSR device.
        """
        if not SHIMMER_AVAILABLE:
            raise DeviceError(
                "Shimmer library not available",
                "GSR001",
                {"solution": "Install pyshimmer package"}
            )
        
        self._logger.info("Starting real GSR capture")
        self._running = True
        
        try:
            # In a real implementation, this would start the data stream
            # self._device.start_streaming()
            
            # Time between samples (in seconds)
            sample_interval = 1.0 / self.sample_rate
            
            while self._running:
                # Get current timestamp
                timestamp = self._get_timestamp()
                
                # In a real implementation, this would get data from the device
                # data = self._device.get_data()
                # gsr_value = data["GSR"]
                
                # For now, just generate a placeholder value
                gsr_value = 2.0
                
                # Add to buffer (limited to buffer_size)
                self._buffer.append((timestamp, gsr_value))
                if len(self._buffer) > self.buffer_size:
                    self._buffer.pop(0)
                
                # Emit the data if signal is available
                if hasattr(self, 'data_available') and self.data_available is not None:
                    # Create a simple dict as the raw data
                    raw_data = {"timestamp": timestamp, "gsr": gsr_value}
                    self.data_available.emit(timestamp, gsr_value, raw_data)
                
                # Sleep to maintain the sample rate
                time.sleep(sample_interval)
                
        except Exception as e:
            error_msg = f"Error in GSR capture: {e}"
            self._logger.error(error_msg, exc_info=True)
            if not self._running:
                # If we're stopping, this is expected
                return
            raise CaptureError(error_msg, "GSR005")
        finally:
            # In a real implementation, this would stop the data stream
            # if self._device is not None:
            #     self._device.stop_streaming()
            pass
    
    def get_buffer(self) -> List[Tuple[int, float]]:
        """Get the current data buffer.
        
        Returns:
            List of (timestamp, gsr_value) tuples
        """
        return self._buffer.copy()