# src/capture/thermal_capture.py

import logging
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)

# Try to import the FLIR camera libraries
try:
    import PySpin
    FLIR_AVAILABLE = True
except ImportError:
    logging.warning("PySpin library not found. FLIR camera support will not be available.")
    FLIR_AVAILABLE = False


class ThermalCaptureThread(QThread):
    """
    A dedicated QThread to capture thermal video from a FLIR A65 camera.

    This class is designed to run in the background, continuously capturing frames
    from a FLIR A65 thermal camera. It emits the captured frames as a signal,
    allowing other parts of the application (like the GUI) to receive them
    without freezing the main event loop.

    Attributes:
        frame_captured (pyqtSignal): A signal that emits the captured thermal frame
                                     as a NumPy array (np.ndarray).
        finished (pyqtSignal): A signal emitted when the capture loop has finished.
    """

    frame_captured = pyqtSignal(np.ndarray, float)
    finished = pyqtSignal()

    def __init__(self, camera_index: int = 0, fps: int = 30, simulation_mode: bool = False, parent=None):
        """
        Initializes the thermal capture thread.

        Args:
            camera_index (int): The index of the FLIR camera to use (default: 0).
            fps (int): The target frames per second for capture.
            simulation_mode (bool): If True, the thread will generate simulated
                                   thermal data instead of connecting to hardware.
            parent (QObject, optional): The parent object in the Qt hierarchy.
        """
        super().__init__(parent)
        self.camera_index = camera_index
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.is_running = False
        self.camera = None
        self.system = None
        self.setObjectName("ThermalCaptureThread")

    def run(self):
        """
        The main execution method of the thread.

        This method is called when the thread starts. It enters a loop to
        continuously read frames from the camera until the `stop()` method is called.
        """
        logging.info(f"Thermal capture thread started for camera index {self.camera_index}.")
        self.is_running = True

        if self.simulation_mode:
            self._run_simulation()
        elif FLIR_AVAILABLE:
            self._run_real_capture()
        else:
            logging.error("FLIR camera mode is selected, but the 'PySpin' library is not available.")

        logging.info("Thermal capture thread has finished.")
        self.finished.emit()

    def _run_simulation(self):
        """
        Generates and emits simulated thermal data at the specified frame rate.
        This is useful for UI development and testing without hardware.
        """
        logging.info(f"Running in thermal simulation mode. Frame rate: {self.fps}Hz.")
        interval = 1.0 / self.fps
        frame_size = (480, 640)  # Common thermal camera resolution

        while self.is_running:
            # Create a simulated thermal image (grayscale with some patterns)
            frame = np.zeros(frame_size, dtype=np.uint8)

            # Add a gradient from top to bottom
            for i in range(frame_size[0]):
                value = int(255 * i / frame_size[0])
                frame[i, :] = value

            # Add some "hot spots" that move over time
            t = time.time()
            for i in range(3):
                x = int((0.5 + 0.3 * np.sin(t + i)) * frame_size[1])
                y = int((0.5 + 0.3 * np.cos(t + i)) * frame_size[0])
                if 0 <= x < frame_size[1] and 0 <= y < frame_size[0]:
                    # Create a hot spot with a radius of 30 pixels
                    for dx in range(-30, 31):
                        for dy in range(-30, 31):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < frame_size[1] and 0 <= ny < frame_size[0]:
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < 30:
                                    intensity = int(255 * (1 - dist/30))
                                    frame[ny, nx] = min(255, frame[ny, nx] + intensity)

            # Convert to 3-channel for display compatibility
            color_frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)

            # Apply a colormap (simple hot colormap: black -> red -> yellow -> white)
            for i in range(frame_size[0]):
                for j in range(frame_size[1]):
                    val = frame[i, j]
                    if val < 85:  # Black to red
                        color_frame[i, j, 0] = 0
                        color_frame[i, j, 1] = 0
                        color_frame[i, j, 2] = 3 * val
                    elif val < 170:  # Red to yellow
                        color_frame[i, j, 0] = 0
                        color_frame[i, j, 1] = 3 * (val - 85)
                        color_frame[i, j, 2] = 255
                    else:  # Yellow to white
                        color_frame[i, j, 0] = 3 * (val - 170)
                        color_frame[i, j, 1] = 255
                        color_frame[i, j, 2] = 255

            current_capture_time = time.perf_counter_ns()  # High-resolution timestamp
            self.frame_captured.emit(color_frame, current_capture_time)
            time.sleep(interval)

    def _run_real_capture(self):
        """
        Connects to and streams data from a physical FLIR A65 thermal camera.
        This method contains the logic for real hardware interaction.
        """
        try:
            # Initialize the PySpin System
            self.system = PySpin.System.GetInstance()
            cam_list = self.system.GetCameras()

            if cam_list.GetSize() == 0:
                logging.error("No FLIR cameras detected.")
                return

            if self.camera_index >= cam_list.GetSize():
                logging.error(f"Camera index {self.camera_index} is out of range. Only {cam_list.GetSize()} cameras detected.")
                return

            # Get the camera
            self.camera = cam_list.GetByIndex(self.camera_index)

            # Initialize the camera
            self.camera.Init()

            # Set acquisition mode to continuous
            self.camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

            # Set frame rate
            if self.camera.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                self.camera.AcquisitionFrameRateEnable.SetValue(True)
                self.camera.AcquisitionFrameRate.SetValue(float(self.fps))
                logging.info(f"Set frame rate to {self.fps} fps")

            # Start acquisition
            self.camera.BeginAcquisition()
            logging.info("Started thermal camera acquisition")

            while self.is_running:
                try:
                    # Get the next image
                    image = self.camera.GetNextImage(1000)  # Timeout in ms

                    if image.IsIncomplete():
                        logging.warning(f"Incomplete thermal image received: {image.GetImageStatus()}")
                        continue

                    # Convert the image to a numpy array
                    image_data = image.GetData().reshape(image.GetHeight(), image.GetWidth())

                    # Normalize the data to 0-255 range for display
                    min_val = np.min(image_data)
                    max_val = np.max(image_data)
                    if max_val > min_val:
                        normalized = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(image_data, dtype=np.uint8)

                    # Apply a colormap (similar to the simulation)
                    color_frame = np.zeros((image.GetHeight(), image.GetWidth(), 3), dtype=np.uint8)

                    for i in range(image.GetHeight()):
                        for j in range(image.GetWidth()):
                            val = normalized[i, j]
                            if val < 85:  # Black to red
                                color_frame[i, j, 0] = 0
                                color_frame[i, j, 1] = 0
                                color_frame[i, j, 2] = 3 * val
                            elif val < 170:  # Red to yellow
                                color_frame[i, j, 0] = 0
                                color_frame[i, j, 1] = 3 * (val - 85)
                                color_frame[i, j, 2] = 255
                            else:  # Yellow to white
                                color_frame[i, j, 0] = 3 * (val - 170)
                                color_frame[i, j, 1] = 255
                                color_frame[i, j, 2] = 255

                    # Emit the frame with timestamp
                    current_capture_time = time.perf_counter_ns()  # High-resolution timestamp
                    self.frame_captured.emit(color_frame, current_capture_time)

                    # Release the image
                    image.Release()

                except PySpin.SpinnakerException as e:
                    logging.error(f"Error capturing thermal frame: {e}")
                    time.sleep(0.1)  # Prevent busy-waiting on error

        except PySpin.SpinnakerException as e:
            logging.error(f"Error initializing FLIR camera: {e}")

        finally:
            # Clean up
            if self.camera is not None:
                if self.camera.IsStreaming():
                    self.camera.EndAcquisition()
                self.camera.DeInit()
                del self.camera
                self.camera = None

            if self.system is not None:
                cam_list.Clear()
                self.system.ReleaseInstance()
                self.system = None

    def stop(self):
        """
        Signals the thread to gracefully stop its execution loop.
        """
        logging.info("Stopping thermal capture thread.")
        self.is_running = False
