# src/capture/video_capture.py

import logging
import sys
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal

from src.capture.base_capture import BaseCaptureThread

# --- Setup logging ---
# This allows for more structured output than print statements.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)


class VideoCaptureThread(BaseCaptureThread):
    """
    A dedicated thread to capture video from a camera source.

    This class is designed to run in the background, continuously capturing frames
    from a specified camera device. It emits the captured frames as a signal,
    allowing other parts of the application (like the GUI) to receive them
    without freezing the main event loop.

    Attributes:
        frame_captured (pyqtSignal): A signal that emits the captured video frame
                                     as a NumPy array (np.ndarray) and timestamp.
    """

    frame_captured = pyqtSignal(np.ndarray, float)

    def __init__(self, camera_id: int, camera_name: str, fps: int, parent=None):
        """
        Initializes the video capture thread.

        Args:
            camera_id (int): The numerical ID of the camera device (e.g., 0, 1).
            camera_name (str): A descriptive name for the camera (e.g., 'RGB', 'Thermal').
                               Used for logging purposes.
            fps (int): The target frames per second for capture.
            parent (QObject, optional): The parent object in the Qt hierarchy.
        """
        super().__init__(device_name=camera_name, parent=parent)
        self.camera_id = camera_id
        self.fps = fps
        self.cap = None

    def _run_real_capture(self):
        """
        Run the capture thread with real hardware.

        This method continuously reads frames from the camera and emits them
        with timestamps until the thread is stopped.
        """
        try:
            # Attempt to open the video capture device
            # Use cv2.CAP_DSHOW on Windows for better camera support
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logging.error(
                    f"Could not open video source for device {self.camera_id}."
                )
                self.is_running = False
                return

            # Set desired camera properties
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            while self.is_running:
                ret, frame = self.cap.read()

                if not ret:
                    logging.warning(f"Dropped frame from {self.device_name} camera.")
                    # Prevent busy-waiting on error
                    self._sleep(0.01)
                    continue

                # Emit the captured frame with timestamp for other components to use
                current_capture_time = self.get_current_timestamp()
                logging.info(f"VideoCaptureThread: Emitting frame with timestamp {current_capture_time}")
                self.frame_captured.emit(frame, current_capture_time)

        except Exception as e:
            logging.error(
                f"An exception occurred in {self.device_name} capture thread: {e}"
            )

    def _cleanup(self):
        """
        Clean up resources when the thread is stopping.
        """
        if self.cap:
            self.cap.release()
            self.cap = None

    def _sleep(self, seconds):
        """
        Sleep for the specified number of seconds.

        This method is used to prevent busy-waiting on error.

        Args:
            seconds (float): The number of seconds to sleep.
        """
        import time
        time.sleep(seconds)
