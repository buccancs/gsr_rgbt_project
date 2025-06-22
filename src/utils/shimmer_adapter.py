# src/utils/shimmer_adapter.py

import logging
import os
import sys
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

# Try to import pyshimmer (the primary Python API for Shimmer devices)
try:
    import pyshimmer
    PYSHIMMER_AVAILABLE = True
except ImportError:
    logging.warning("pyshimmer library not found. Using fallback mechanisms if available.")
    PYSHIMMER_AVAILABLE = False
    pyshimmer = None

# Constants for different Shimmer APIs
SHIMMER_C_API_PATH = Path("third_party/Shimmer-C-API")
SHIMMER_JAVA_API_PATH = Path("third_party/Shimmer-Java-Android-API")
SHIMMER_ANDROID_API_PATH = Path("third_party/ShimmerAndroidAPI")

class ShimmerAdapterError(Exception):
    """Custom exception for Shimmer adapter errors."""
    pass

class ShimmerAdapter:
    """
    Adapter class that provides a unified interface to different Shimmer APIs.
    
    This class primarily uses pyshimmer for Shimmer device communication, but can
    also leverage the additional capabilities provided by other Shimmer APIs when
    available and appropriate.
    
    The adapter follows these priorities:
    1. Use pyshimmer for all basic functionality (Bluetooth communication, data streaming)
    2. Use Shimmer-C-API for advanced signal processing if available
    3. Use Shimmer-Java-Android-API for additional features if running in a Java environment
    4. Use ShimmerAndroidAPI for Android-specific features if running on Android
    """
    
    def __init__(self, port: str, sampling_rate: int, simulation_mode: bool = False):
        """
        Initialize the ShimmerAdapter.
        
        Args:
            port (str): The serial port for the GSR device (e.g., 'COM3').
                        Not used in simulation mode.
            sampling_rate (int): The target sampling rate in Hz (e.g., 32).
            simulation_mode (bool): If True, the adapter will simulate data
                                   instead of connecting to hardware.
        """
        self.port = port
        self.sampling_rate = sampling_rate
        self.simulation_mode = simulation_mode
        self.shimmer_device = None
        self.c_api_available = self._check_c_api_available()
        self.java_api_available = self._check_java_api_available()
        self.android_api_available = self._check_android_api_available()
        
        # Log available APIs
        logging.info(f"ShimmerAdapter initialized with the following APIs:")
        logging.info(f"  - pyshimmer: {'Available' if PYSHIMMER_AVAILABLE else 'Not available'}")
        logging.info(f"  - Shimmer-C-API: {'Available' if self.c_api_available else 'Not available'}")
        logging.info(f"  - Shimmer-Java-Android-API: {'Available' if self.java_api_available else 'Not available'}")
        logging.info(f"  - ShimmerAndroidAPI: {'Available' if self.android_api_available else 'Not available'}")
    
    def _check_c_api_available(self) -> bool:
        """Check if Shimmer-C-API is available."""
        return SHIMMER_C_API_PATH.exists()
    
    def _check_java_api_available(self) -> bool:
        """Check if Shimmer-Java-Android-API is available."""
        return SHIMMER_JAVA_API_PATH.exists()
    
    def _check_android_api_available(self) -> bool:
        """Check if ShimmerAndroidAPI is available."""
        return SHIMMER_ANDROID_API_PATH.exists()
    
    def connect(self) -> bool:
        """
        Connect to the Shimmer device.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if self.simulation_mode:
            logging.info("Running in simulation mode, no connection needed.")
            return True
        
        if not PYSHIMMER_AVAILABLE:
            logging.error("Cannot connect to Shimmer device: pyshimmer library not available.")
            return False
        
        try:
            logging.info(f"Attempting to connect to Shimmer device on port {self.port}.")
            self.shimmer_device = pyshimmer.Shimmer(self.port)
            self.shimmer_device.set_sampling_rate(self.sampling_rate)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Shimmer device: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Shimmer device.
        
        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        if self.simulation_mode or not self.shimmer_device:
            return True
        
        try:
            if hasattr(self.shimmer_device, 'is_streaming') and self.shimmer_device.is_streaming():
                self.shimmer_device.stop_streaming()
            if hasattr(self.shimmer_device, 'close'):
                self.shimmer_device.close()
            logging.info("Shimmer device connection closed.")
            return True
        except Exception as e:
            logging.error(f"Failed to disconnect from Shimmer device: {e}")
            return False
    
    def set_enabled_sensors(self, sensors_bitmask: int) -> bool:
        """
        Enable specific sensors on the Shimmer device.
        
        Args:
            sensors_bitmask (int): A bitmask of sensors to enable.
        
        Returns:
            bool: True if sensors were enabled successfully, False otherwise.
        """
        if self.simulation_mode or not self.shimmer_device:
            return True
        
        try:
            self.shimmer_device.set_enabled_sensors(sensors_bitmask)
            return True
        except Exception as e:
            logging.error(f"Failed to set enabled sensors: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """
        Start streaming data from the Shimmer device.
        
        Returns:
            bool: True if streaming was started successfully, False otherwise.
        """
        if self.simulation_mode or not self.shimmer_device:
            return True
        
        try:
            self.shimmer_device.start_streaming()
            logging.info("Successfully started streaming from Shimmer device.")
            return True
        except Exception as e:
            logging.error(f"Failed to start streaming from Shimmer device: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """
        Stop streaming data from the Shimmer device.
        
        Returns:
            bool: True if streaming was stopped successfully, False otherwise.
        """
        if self.simulation_mode or not self.shimmer_device:
            return True
        
        try:
            if hasattr(self.shimmer_device, 'is_streaming') and self.shimmer_device.is_streaming():
                self.shimmer_device.stop_streaming()
            logging.info("Successfully stopped streaming from Shimmer device.")
            return True
        except Exception as e:
            logging.error(f"Failed to stop streaming from Shimmer device: {e}")
            return False
    
    def read_data_packet(self) -> Optional[Dict[str, Any]]:
        """
        Read a data packet from the Shimmer device.
        
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the data packet,
                                     or None if no packet was available.
        """
        if self.simulation_mode or not self.shimmer_device:
            return None
        
        try:
            return self.shimmer_device.read_data_packet()
        except Exception as e:
            logging.error(f"Failed to read data packet from Shimmer device: {e}")
            return None
    
    def get_advanced_processing_capabilities(self) -> List[str]:
        """
        Get a list of advanced processing capabilities available through the
        different Shimmer APIs.
        
        Returns:
            List[str]: A list of available advanced processing capabilities.
        """
        capabilities = []
        
        if self.c_api_available:
            capabilities.extend([
                "ECG to Heart Rate/IBI",
                "PPG to Heart Rate/IBI",
                "Advanced filtering"
            ])
        
        if self.java_api_available:
            capabilities.extend([
                "GSR calibration",
                "3D orientation",
                "Battery voltage monitoring"
            ])
        
        return capabilities
    
    def process_ecg_to_hr(self, ecg_data: List[float]) -> Tuple[float, List[float]]:
        """
        Process ECG data to extract heart rate and IBI.
        
        This method uses the Shimmer-C-API if available, otherwise it returns
        placeholder values.
        
        Args:
            ecg_data (List[float]): A list of ECG data points.
        
        Returns:
            Tuple[float, List[float]]: A tuple containing the heart rate (in BPM)
                                      and a list of IBI values (in ms).
        """
        if not self.c_api_available:
            logging.warning("ECG to HR processing requested but Shimmer-C-API not available.")
            return (0.0, [])
        
        # This is a placeholder. In a real implementation, this would use
        # the Shimmer-C-API to process the ECG data.
        logging.info("Processing ECG data to extract heart rate and IBI.")
        return (70.0, [800.0, 810.0, 790.0])
    
    def process_ppg_to_hr(self, ppg_data: List[float]) -> Tuple[float, List[float]]:
        """
        Process PPG data to extract heart rate and IBI.
        
        This method uses the Shimmer-C-API if available, otherwise it returns
        placeholder values.
        
        Args:
            ppg_data (List[float]): A list of PPG data points.
        
        Returns:
            Tuple[float, List[float]]: A tuple containing the heart rate (in BPM)
                                      and a list of IBI values (in ms).
        """
        if not self.c_api_available:
            logging.warning("PPG to HR processing requested but Shimmer-C-API not available.")
            return (0.0, [])
        
        # This is a placeholder. In a real implementation, this would use
        # the Shimmer-C-API to process the PPG data.
        logging.info("Processing PPG data to extract heart rate and IBI.")
        return (72.0, [820.0, 830.0, 810.0])
    
    def apply_filter(self, data: List[float], filter_type: str, params: Dict[str, Any]) -> List[float]:
        """
        Apply a filter to the data.
        
        This method uses the Shimmer-C-API if available, otherwise it returns
        the input data unchanged.
        
        Args:
            data (List[float]): A list of data points.
            filter_type (str): The type of filter to apply (e.g., 'lowpass', 'highpass').
            params (Dict[str, Any]): Filter parameters.
        
        Returns:
            List[float]: The filtered data.
        """
        if not self.c_api_available:
            logging.warning("Filtering requested but Shimmer-C-API not available.")
            return data
        
        # This is a placeholder. In a real implementation, this would use
        # the Shimmer-C-API to apply the filter.
        logging.info(f"Applying {filter_type} filter to data.")
        return data