import logging
import subprocess
import os
import time
import threading
import numpy as np
from PyQt5.QtCore import pyqtSignal

from src.capture.base_capture import BaseCaptureThread

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)

class FactorizePhysCaptureThread(BaseCaptureThread):
    """
    A dedicated thread to capture synchronized RGB, thermal, and physiological data
    using the FactorizePhys library.

    This class wraps the C++ implementation of FactorizePhys to provide synchronized
    data capture within the Python framework of the GSR-RGBT project. FactorizePhys
    extends RGBTPhys_CPP with specialized functionality for factorizing (separating)
    physiological signals from video data.

    Attributes:
        rgb_frame_captured (pyqtSignal): A signal that emits the captured RGB frame
                                         as a NumPy array (np.ndarray) and timestamp.
        thermal_frame_captured (pyqtSignal): A signal that emits the captured thermal frame
                                            as a NumPy array (np.ndarray) and timestamp.
        phys_data_captured (pyqtSignal): A signal that emits the captured physiological data
                                         and timestamp.
        factorized_signals (pyqtSignal): A signal that emits the factorized physiological signals
                                        extracted from the video data.
    """

    rgb_frame_captured = pyqtSignal(np.ndarray, float)
    thermal_frame_captured = pyqtSignal(np.ndarray, float)
    phys_data_captured = pyqtSignal(object, float)
    factorized_signals = pyqtSignal(object, float)

    def __init__(self, config_file=None, base_save_path=None, participant_id=None, 
                 simulation_mode=False, parent=None):
        """
        Initializes the FactorizePhys capture thread.

        Args:
            config_file (str): Path to the FactorizePhys configuration file.
            base_save_path (str): Base directory to save captured data.
            participant_id (str): ID of the participant for data organization.
            simulation_mode (bool): If True, the thread will generate simulated data.
            parent (QObject, optional): The parent object in the Qt hierarchy.
        """
        super().__init__(device_name="FactorizePhys", parent=parent)
        self.config_file = config_file or "default_config"
        self.base_save_path = base_save_path or os.path.join(os.getcwd(), "data")
        self.participant_id = participant_id or "test_subject"
        self.simulation_mode = simulation_mode
        self.process = None
        self.monitor_thread = None
        self.is_monitoring = False

    def _run_simulation(self):
        """
        Generates and emits simulated RGB, thermal, and physiological data,
        as well as factorized signals.
        This is useful for UI development and testing without hardware.
        """
        logging.info("Running in FactorizePhys simulation mode.")
        
        # Simulate at 30 fps
        interval = 1.0 / 30
        
        # Frame sizes
        rgb_frame_size = (480, 640, 3)
        thermal_frame_size = (480, 640, 3)
        
        while self.is_running:
            # Create simulated RGB frame (colorful pattern)
            rgb_frame = np.zeros(rgb_frame_size, dtype=np.uint8)
            t = time.time()
            
            for i in range(rgb_frame_size[0]):
                for j in range(rgb_frame_size[1]):
                    rgb_frame[i, j, 0] = int(128 + 127 * np.sin(i / 50.0 + t))
                    rgb_frame[i, j, 1] = int(128 + 127 * np.sin(j / 50.0 + t))
                    rgb_frame[i, j, 2] = int(128 + 127 * np.sin((i + j) / 50.0 + t))
            
            # Create simulated thermal frame (grayscale with hot spots)
            thermal_frame = np.zeros(thermal_frame_size, dtype=np.uint8)
            
            # Add a gradient
            for i in range(thermal_frame_size[0]):
                value = int(255 * i / thermal_frame_size[0])
                thermal_frame[i, :, 0] = value
                thermal_frame[i, :, 1] = value
                thermal_frame[i, :, 2] = value
            
            # Add some "hot spots"
            for i in range(3):
                x = int((0.5 + 0.3 * np.sin(t + i)) * thermal_frame_size[1])
                y = int((0.5 + 0.3 * np.cos(t + i)) * thermal_frame_size[0])
                if 0 <= x < thermal_frame_size[1] and 0 <= y < thermal_frame_size[0]:
                    # Create a hot spot with a radius of 30 pixels
                    for dx in range(-30, 31):
                        for dy in range(-30, 31):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < thermal_frame_size[1] and 0 <= ny < thermal_frame_size[0]:
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < 30:
                                    intensity = int(255 * (1 - dist/30))
                                    # Apply red-hot colormap
                                    thermal_frame[ny, nx, 0] = 0
                                    thermal_frame[ny, nx, 1] = 0
                                    thermal_frame[ny, nx, 2] = min(255, intensity)
            
            # Create simulated physiological data
            phys_data = {
                "gsr": 2.0 + 0.5 * np.sin(t),
                "heart_rate": 70 + 10 * np.sin(t / 10),
                "temperature": 36.5 + 0.2 * np.sin(t / 20)
            }
            
            # Create simulated factorized signals
            factorized_data = {
                "pulse_signal": 0.5 + 0.5 * np.sin(2 * np.pi * 1.2 * t),  # ~72 BPM
                "resp_signal": 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t),  # ~15 breaths per minute
                "gsr_signal": 2.0 + 0.3 * np.sin(2 * np.pi * 0.05 * t),   # Slow GSR changes
                "quality_index": 0.85 + 0.15 * np.sin(t)                  # Signal quality metric
            }
            
            # Get current timestamp
            current_capture_time = self.get_current_timestamp()
            
            # Emit the frames and data
            logging.info(f"FactorizePhysCaptureThread (simulation): Emitting data with timestamp {current_capture_time}")
            self.rgb_frame_captured.emit(rgb_frame, current_capture_time)
            self.thermal_frame_captured.emit(thermal_frame, current_capture_time)
            self.phys_data_captured.emit(phys_data, current_capture_time)
            self.factorized_signals.emit(factorized_data, current_capture_time)
            
            # Sleep to maintain frame rate
            self._sleep(interval)

    def _run_real_capture(self):
        """
        Runs the FactorizePhys executable to capture synchronized data from real hardware
        and extract factorized physiological signals.
        """
        logging.info("Starting FactorizePhys for synchronized data capture and signal factorization.")
        
        # Path to the FactorizePhys executable
        factorize_phys_path = os.path.join(os.getcwd(), "third_party", "FactorizePhys", "RGBTPhys.exe")
        
        # Path to the configuration file
        config_path = os.path.join(os.getcwd(), "third_party", "FactorizePhys", self.config_file)
        
        # Ensure the base save path exists
        os.makedirs(self.base_save_path, exist_ok=True)
        
        # Command to run FactorizePhys
        cmd = [
            factorize_phys_path,
            config_path,
            self.base_save_path,
            self.participant_id
        ]
        
        try:
            # Start the FactorizePhys process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Start a thread to monitor the output
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_process_output,
                daemon=True
            )
            self.monitor_thread.start()
            
            # Wait for the process to complete
            while self.is_running and self.process.poll() is None:
                time.sleep(0.1)
            
            # Check if the process exited with an error
            if self.process.returncode is not None and self.process.returncode != 0:
                logging.error(f"FactorizePhys process exited with error code {self.process.returncode}")
            
        except Exception as e:
            logging.error(f"Error running FactorizePhys: {e}")
        
        # Stop monitoring
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

    def _monitor_process_output(self):
        """
        Monitors the output of the FactorizePhys process and emits signals when new data is available.
        """
        while self.is_monitoring and self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if not line:
                break
                
            logging.info(f"FactorizePhys: {line.strip()}")
            
            # Parse the output to detect when new frames are captured
            # This is a simplified example - actual implementation would depend on
            # the output format of FactorizePhys
            if "RGB frame captured" in line:
                # In a real implementation, we would read the actual frame data
                # For now, we'll just emit a placeholder
                current_capture_time = self.get_current_timestamp()
                rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.rgb_frame_captured.emit(rgb_frame, current_capture_time)
                
            elif "Thermal frame captured" in line:
                current_capture_time = self.get_current_timestamp()
                thermal_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.thermal_frame_captured.emit(thermal_frame, current_capture_time)
                
            elif "Physiological data captured" in line:
                current_capture_time = self.get_current_timestamp()
                phys_data = {"gsr": 0.0, "heart_rate": 0.0, "temperature": 0.0}
                self.phys_data_captured.emit(phys_data, current_capture_time)
                
            elif "Factorized signal extracted" in line:
                # Parse the factorized signal data
                # This would extract the actual signal values from the output
                current_capture_time = self.get_current_timestamp()
                factorized_data = {
                    "pulse_signal": 0.0,
                    "resp_signal": 0.0,
                    "gsr_signal": 0.0,
                    "quality_index": 0.0
                }
                self.factorized_signals.emit(factorized_data, current_capture_time)
        
        logging.info("FactorizePhys output monitoring stopped.")

    def _cleanup(self):
        """
        Clean up resources when the thread is stopping.
        """
        logging.info("Cleaning up FactorizePhys capture resources.")
        
        # Stop monitoring
        self.is_monitoring = False
        
        # Terminate the process if it's still running
        if self.process and self.process.poll() is None:
            logging.info("Terminating FactorizePhys process.")
            try:
                self.process.terminate()
                # Give it some time to terminate gracefully
                time.sleep(1)
                # Force kill if still running
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                logging.error(f"Error terminating FactorizePhys process: {e}")
        
        # Wait for the monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        self.process = None
        self.monitor_thread = None

    def _sleep(self, seconds):
        """
        Sleep for the specified number of seconds.
        
        Args:
            seconds (float): The number of seconds to sleep.
        """
        time.sleep(seconds)