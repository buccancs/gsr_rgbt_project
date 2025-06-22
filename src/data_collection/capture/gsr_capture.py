# src/data_collection/capture/gsr_capture.py

import logging
import random
import time

from PyQt5.QtCore import pyqtSignal

from src.data_collection.capture.base_capture import BaseCaptureThread

# --- Setup logging for this module ---
# Follows the same format for consistency across the application.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)

# --- Import Real Hardware Library ---
# This allows the code to run without the pyshimmer library being installed.
try:
    import pyshimmer
    PYSHIMMER_AVAILABLE = True
except ImportError:
    logging.warning("pyshimmer library not found. Real GSR sensor capture will not be available.")
    PYSHIMMER_AVAILABLE = False
    pyshimmer = None


class GsrCaptureThread(BaseCaptureThread):
    """
    A dedicated thread to handle GSR data acquisition.

    This class manages the connection to a GSR sensor (or simulates one) and
    continuously streams data in a background thread. This prevents the sensor's
    blocking read calls from freezing the main application GUI.

    Attributes:
        gsr_data_point (pyqtSignal): A signal that emits a single, calibrated
                                     GSR data point as a float and timestamp.
    """

    gsr_data_point = pyqtSignal(float, float)

    def __init__(
        self, port: str, sampling_rate: int, simulation_mode: bool = False, 
        sensors_to_enable=None, parent=None
    ):
        """
        Initializes the GSR capture thread.

        Args:
            port (str): The serial port for the GSR device (e.g., 'COM3').
                        Not used in simulation mode.
            sampling_rate (int): The target sampling rate in Hz (e.g., 32).
            simulation_mode (bool): If True, the thread will generate simulated
                                    GSR data instead of connecting to hardware.
            sensors_to_enable: A bitmask of sensors to enable on the Shimmer device.
                              If None, defaults to GSR, PPG, and Accelerometer.
            parent (QObject, optional): The parent object in the Qt hierarchy.
        """
        super().__init__(device_name="GSR", parent=parent)
        self.port = port
        self.sampling_rate = sampling_rate
        self.simulation_mode = simulation_mode
        self.shimmer_device = None

        # Allow dynamic sensor configuration
        if sensors_to_enable is None and PYSHIMMER_AVAILABLE:
            # Default to the original configuration if none is provided
            self.sensors_to_enable = (
                pyshimmer.Shimmer.SENSOR_GSR |
                pyshimmer.Shimmer.SENSOR_PPG |
                pyshimmer.Shimmer.SENSOR_ACCEL
            )
        else:
            self.sensors_to_enable = sensors_to_enable

    def _run_simulation(self):
        """
        Generates and emits simulated GSR data at the specified sampling rate.
        This is useful for UI development and testing without hardware.
        """
        logging.info(
            f"Running in GSR simulation mode. Sampling rate: {self.sampling_rate}Hz."
        )
        interval = 1.0 / self.sampling_rate
        base_gsr = 0.8  # A stable baseline in microsiemens (µS)

        while self.is_running:
            # Simulate a noisy baseline GSR signal with occasional small peaks
            noise = (random.random() - 0.5) * 0.05
            peak = max(0, (random.random() - 0.95) * 5)  # Infrequent, larger spikes
            gsr_value = base_gsr + noise + peak

            current_capture_time = self.get_current_timestamp()
            logging.info(f"GsrCaptureThread (simulation): Emitting GSR data with timestamp {current_capture_time}")
            self.gsr_data_point.emit(gsr_value, current_capture_time)
            self._sleep(interval)

    def _run_real_capture(self):
        """
        Connects to and streams data from a physical Shimmer GSR device.
        This method contains the logic for real hardware interaction.
        """
        if not PYSHIMMER_AVAILABLE:
            logging.error(
                "Real GSR sensor mode is selected, but the 'pyshimmer' library is not available."
            )
            return

        logging.info(f"Attempting to connect to Shimmer device on port {self.port}.")
        try:
            self.shimmer_device = pyshimmer.Shimmer(self.port)
            # Configure the shimmer device (set sampling rate and enable sensors)
            self.shimmer_device.set_sampling_rate(self.sampling_rate)

            # Use the dynamic sensor configuration
            if self.sensors_to_enable is not None:
                self.shimmer_device.set_enabled_sensors(self.sensors_to_enable)
            else:
                # Fallback to just enabling GSR if no configuration was provided
                self.shimmer_device.enable_gsr()  # Enable GSR sensor

            self.shimmer_device.start_streaming()
            logging.info("Successfully connected to Shimmer device and started streaming.")

            while self.is_running:
                # The read_data_packet call is typically blocking
                packet = self.shimmer_device.read_data_packet()
                if packet:
                    # The exact key depends on the Shimmer's configuration and sensor type.
                    # 'GSR_CAL' is a common key for calibrated skin conductance.
                    gsr_val = packet.get('GSR_CAL')
                    # Get the Shimmer's own timestamp
                    shimmer_timestamp = packet.get('Timestamp_FormattedUnix_CAL')

                    if gsr_val is not None and shimmer_timestamp is not None:
                        logging.info(f"GsrCaptureThread (real): Emitting GSR data with timestamp {shimmer_timestamp}")
                        self.gsr_data_point.emit(gsr_val, shimmer_timestamp)

        except Exception as e:
            logging.error(f"Failed to connect or read from Shimmer device: {e}")

    def _cleanup(self):
        """
        Clean up resources when the thread is stopping.
        """
        if self.shimmer_device:
            if hasattr(self.shimmer_device, 'is_streaming') and self.shimmer_device.is_streaming():
                self.shimmer_device.stop_streaming()
            if hasattr(self.shimmer_device, 'close'):
                self.shimmer_device.close()
            logging.info("Shimmer device connection closed.")

    def _sleep(self, seconds):
        """
        Sleep for the specified number of seconds.

        This method is used to prevent busy-waiting on error.

        Args:
            seconds (float): The number of seconds to sleep.
        """
        time.sleep(seconds)
