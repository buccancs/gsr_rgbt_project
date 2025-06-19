# src/data_collection/capture/base_capture.py

import logging
import time
from PyQt5.QtCore import QThread, pyqtSignal

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)

class BaseCaptureThread(QThread):
    """
    Base class for all capture threads in the GSR-RGBT project.

    This abstract class provides common functionality for video and sensor capture threads,
    including thread management, logging, and signal handling.

    Attributes:
        finished (pyqtSignal): A signal emitted when the capture loop has finished.
        is_running (bool): Flag indicating whether the capture loop is running.
    """

    finished = pyqtSignal()

    def __init__(self, device_name, parent=None):
        """
        Initialize the base capture thread.

        Args:
            device_name (str): A descriptive name for the device (e.g., 'RGB', 'Thermal', 'GSR').
                               Used for logging and thread naming.
            parent (QObject, optional): The parent object in the Qt hierarchy.
        """
        super().__init__(parent)
        self.device_name = device_name
        self.is_running = False
        self.setObjectName(f"{self.device_name}CaptureThread")

    def run(self):
        """
        The main execution method of the thread.

        This method is called when the thread starts. It sets up the capture loop
        and calls the appropriate method based on the configuration.
        """
        logging.info(f"{self.device_name} capture thread started.")
        self.is_running = True

        try:
            # Call the appropriate run method based on the configuration
            if hasattr(self, 'simulation_mode') and self.simulation_mode:
                self._run_simulation()
            else:
                self._run_real_capture()
        except Exception as e:
            logging.error(f"An exception occurred in {self.device_name} capture thread: {e}")
        finally:
            self._cleanup()
            logging.info(f"{self.device_name} capture thread has finished.")
            self.finished.emit()

    def _run_simulation(self):
        """
        Run the capture thread in simulation mode.

        This method should be implemented by subclasses to generate simulated data.
        """
        raise NotImplementedError("Subclasses must implement _run_simulation method")

    def _run_real_capture(self):
        """
        Run the capture thread with real hardware.

        This method should be implemented by subclasses to capture data from real devices.
        """
        raise NotImplementedError("Subclasses must implement _run_real_capture method")

    def _cleanup(self):
        """
        Clean up resources when the thread is stopping.

        This method should be implemented by subclasses to release any resources
        that were acquired during the capture process.
        """
        pass

    def stop(self):
        """
        Signals the thread to gracefully stop its execution loop.
        """
        logging.info(f"Stopping {self.device_name} capture thread.")
        self.is_running = False

    def get_current_timestamp(self):
        """
        Get the current high-resolution timestamp.

        Returns:
            float: The current timestamp in nanoseconds.
        """
        return time.perf_counter_ns()
