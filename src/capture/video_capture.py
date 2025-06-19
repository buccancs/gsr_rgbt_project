# src/capture/video_capture.py

import logging
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# --- Setup logging ---
# This allows for more structured output than print statements.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)


class VideoCaptureThread(QThread):
    """
    A dedicated QThread to capture video from a camera source.

    This class is designed to run in the background, continuously capturing frames
    from a specified camera device. It emits the captured frames as a signal,
    allowing other parts of the application (like the GUI) to receive them
    without freezing the main event loop.

    Attributes:
        frame_captured (pyqtSignal): A signal that emits the captured video frame
                                     as a NumPy array (np.ndarray).
        finished (pyqtSignal): A signal emitted when the capture loop has finished.
    """

    frame_captured = pyqtSignal(np.ndarray, float)
    finished = pyqtSignal()

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
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.fps = fps
        self.is_running = False
        self.cap = None
        self.setObjectName(f"{self.camera_name}CaptureThread")

    def run(self):
        """
        The main execution method of the thread.

        This method is called when the thread starts. It enters a loop to
        continuously read frames from the camera until the `stop()` method is called.
        """
        logging.info(
            f"Capture thread started for device {self.camera_id} ({self.camera_name})."
        )
        self.is_running = True

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

            # Set desired camera properties
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            while self.is_running:
                ret, frame = self.cap.read()

                if not ret:
                    logging.warning(f"Dropped frame from {self.camera_name} camera.")
                    time.sleep(0.01)  # Prevent busy-waiting on error
                    continue

                # Emit the captured frame with timestamp for other components to use
                current_capture_time = time.perf_counter_ns()  # High-resolution timestamp
                self.frame_captured.emit(frame, current_capture_time)

        except Exception as e:
            logging.error(
                f"An exception occurred in {self.camera_name} capture thread: {e}"
            )

        finally:
            if self.cap:
                self.cap.release()
            logging.info(f"Capture thread for {self.camera_name} has finished.")
            self.finished.emit()

    def stop(self):
        """
        Signals the thread to gracefully stop its execution loop.
        """
        logging.info(f"Stopping capture for {self.camera_name} camera.")
        self.is_running = False
