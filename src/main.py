# src/main.py

import sys
import logging
from pathlib import Path
import numpy as np
from enum import Enum, auto

from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow
from PyQt5.QtCore import QObject, QThread, pyqtSlot

from src.data_collection.gui.main_window import MainWindow
from src.data_collection.capture.video_capture import VideoCaptureThread
from src.data_collection.capture.thermal_capture import ThermalCaptureThread
from src.data_collection.capture.gsr_capture import GsrCaptureThread
from src.data_collection.utils.data_logger import DataLogger
from src.data_collection.utils.timestamp_thread import TimestampThread
from src.utils.device_utils import find_shimmer_com_port, DeviceNotFoundError
from src import config

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

class AppState(Enum):
    """
    Enum representing the possible states of the application.
    Used for formal state management to prevent invalid UI interactions.
    """
    IDLE = auto()
    RECORDING = auto()


class Application(QMainWindow):
    """
    Main application class that orchestrates the GUI and data capture components.

    This class connects the MainWindow UI with the capture threads and data logger,
    handling the application's core logic and event flow.
    """

    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle(config.APP_NAME)
        self.setGeometry(*config.GEOMETRY)

        # Create the main window as a widget
        self.main_window = MainWindow()
        self.setCentralWidget(self.main_window)

        # Initialize application state
        self.state = AppState.IDLE

        # Initialize the timestamp thread (centralized timestamp authority)
        self.timestamp_thread = TimestampThread(frequency=200)  # 200Hz timestamp generation

        # Store the latest timestamp
        self.latest_timestamp = None

        # Connect timestamp signal to update the latest timestamp
        self.timestamp_thread.timestamp_generated.connect(self.update_latest_timestamp)

        # Initialize capture threads (not started yet)
        self.rgb_capture = VideoCaptureThread(
            camera_id=config.RGB_CAMERA_ID,
            camera_name="RGB",
            fps=config.FPS
        )

        self.thermal_capture = ThermalCaptureThread(
            camera_index=config.THERMAL_CAMERA_ID,
            fps=config.FPS,
            simulation_mode=config.THERMAL_SIMULATION_MODE
        )

        try:
            # --- Automatic Shimmer Port Detection ---
            # Call the discovery function to get the port automatically.
            shimmer_port = find_shimmer_com_port() if not config.GSR_SIMULATION_MODE else config.GSR_SENSOR_PORT

            # Initialize the GSR capture thread with the discovered port.
            self.gsr_capture = GsrCaptureThread(
                port=shimmer_port,
                sampling_rate=config.GSR_SAMPLING_RATE,
                simulation_mode=config.GSR_SIMULATION_MODE
            )
            logging.info(f"Successfully detected Shimmer device on port {shimmer_port}")

        except DeviceNotFoundError as e:
            # If the device isn't found, show an error message and disable GSR functionality.
            logging.error(f"Error: {e}")
            self.show_error_message(
                "Shimmer Device Not Found",
                str(e)
            )
            # Fall back to simulation mode if device not found
            self.gsr_capture = GsrCaptureThread(
                port="SIM",
                sampling_rate=config.GSR_SAMPLING_RATE,
                simulation_mode=True
            )
            logging.info("Falling back to GSR simulation mode")

            # Update the UI status bar to clearly indicate simulation mode
            self.main_window.statusBar().showMessage("GSR: SIMULATION MODE - Shimmer device not found", 10000)  # Display for 10s

        # Data logger will be initialized when recording starts
        self.data_logger = None

        # Connect UI signals to application slots
        self.main_window.start_button.clicked.connect(self.start_recording)
        self.main_window.stop_button.clicked.connect(self.stop_recording)

        # Connect capture thread signals to UI update slots
        self.rgb_capture.frame_captured.connect(
            lambda frame, timestamp: self.main_window.update_video_feed(frame, self.main_window.rgb_video_label)
        )
        self.thermal_capture.frame_captured.connect(
            lambda frame, timestamp: self.main_window.update_video_feed(frame, self.main_window.thermal_video_label)
        )

    @pyqtSlot(int)
    def update_latest_timestamp(self, timestamp):
        """
        Update the latest timestamp from the timestamp thread.

        Args:
            timestamp (int): The latest timestamp in nanoseconds.
        """
        self.latest_timestamp = timestamp

    @pyqtSlot(np.ndarray, float)
    def on_rgb_frame(self, frame, timestamp):
        """
        Slot to log RGB frames with the master timestamp.

        Args:
            frame (np.ndarray): The captured RGB frame.
            timestamp (float): The original capture timestamp (not used).
        """
        if self.data_logger:
            self.data_logger.log_rgb_frame(frame, self.latest_timestamp)

    @pyqtSlot(np.ndarray, float)
    def on_thermal_frame(self, frame, timestamp):
        """
        Slot to log Thermal frames with the master timestamp.

        Args:
            frame (np.ndarray): The captured thermal frame.
            timestamp (float): The original capture timestamp (not used).
        """
        if self.data_logger:
            self.data_logger.log_thermal_frame(frame, self.latest_timestamp)

    def start_recording(self):
        """Start all capture threads and initialize data logging."""
        try:
            # Check if we're already recording
            if self.state != AppState.IDLE:
                logging.warning("Start command ignored: Already recording.")
                return

            subject_id = self.main_window.subject_id_input.text()
            if not subject_id:
                self.main_window.show_error_message(
                    "Input Error", "Please enter a Subject ID before starting."
                )
                return

            # Initialize the data logger
            self.data_logger = DataLogger(
                output_dir=config.OUTPUT_DIR,
                subject_id=subject_id,
                fps=config.FPS,
                video_fourcc=config.VIDEO_FOURCC
            )

            # Start the data logger
            self.data_logger.start_logging(
                frame_size_rgb=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
                frame_size_thermal=(config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )

            # Start the timestamp thread first to ensure it's running before capture threads
            self.timestamp_thread.start()
            logging.info("Timestamp thread started")

            # Connect capture signals to data logger with centralized timestamps
            self.rgb_capture.frame_captured.connect(self.on_rgb_frame)
            self.thermal_capture.frame_captured.connect(self.on_thermal_frame)
            self.gsr_capture.gsr_data_point.connect(self.data_logger.log_gsr_data)

            # Start capture threads
            self.rgb_capture.start()
            self.thermal_capture.start()
            self.gsr_capture.start()

            # Update state and UI
            self.state = AppState.RECORDING
            self._update_ui_for_state()

            logging.info(f"Recording started for subject: {subject_id}")
            logging.info("State changed to RECORDING")

        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            self.main_window.show_error_message(
                "Recording Error", f"Failed to start recording: {str(e)}"
            )

    def stop_recording(self):
        """Stop all capture threads and finalize data logging."""
        try:
            # Check if we're not recording
            if self.state != AppState.RECORDING:
                logging.warning("Stop command ignored: Not recording.")
                return

            # Stop capture threads
            self.rgb_capture.stop()
            self.thermal_capture.stop()
            self.gsr_capture.stop()

            # Stop the timestamp thread
            self.timestamp_thread.stop()

            # Reset the latest timestamp
            self.latest_timestamp = None

            # Wait for threads to finish
            if self.rgb_capture.isRunning():
                self.rgb_capture.wait()
            if self.thermal_capture.isRunning():
                self.thermal_capture.wait()
            if self.gsr_capture.isRunning():
                self.gsr_capture.wait()
            if self.timestamp_thread.isRunning():
                self.timestamp_thread.wait()
                logging.info("Timestamp thread stopped")

            # Disconnect signals from data logger
            if self.data_logger:
                # Disconnect all signals from the data logger
                try:
                    self.rgb_capture.frame_captured.disconnect()
                    self.thermal_capture.frame_captured.disconnect()
                    self.gsr_capture.gsr_data_point.disconnect()
                except TypeError:
                    # This can happen if the signals were never connected
                    logging.warning("Could not disconnect all signals from data logger.")

                # Stop the data logger
                self.data_logger.stop_logging()
                self.data_logger = None

            # Update state and UI
            self.state = AppState.IDLE
            self._update_ui_for_state()

            logging.info("Recording stopped")
            logging.info("State changed to IDLE")

        except Exception as e:
            logging.error(f"Error while stopping recording: {e}")
            self.main_window.show_error_message(
                "Recording Error", f"Error while stopping recording: {str(e)}"
            )

    def show_error_message(self, title, message):
        """
        A helper method to display a modal error dialog.

        Args:
            title (str): The title of the error message.
            message (str): The detailed error message to display.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(title)
        msg_box.setInformativeText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()

    def _update_ui_for_state(self):
        """
        Updates the GUI elements based on the current application state.
        This ensures that the UI reflects the current state and prevents invalid interactions.
        """
        if self.state == AppState.IDLE:
            self.main_window.start_button.setEnabled(True)
            self.main_window.stop_button.setEnabled(False)
            self.main_window.subject_id_input.setEnabled(True)
        elif self.state == AppState.RECORDING:
            self.main_window.start_button.setEnabled(False)
            self.main_window.stop_button.setEnabled(True)
            self.main_window.subject_id_input.setEnabled(False)

        logging.info(f"UI updated for state: {self.state.name}")

    def closeEvent(self, event):
        """
        Handles the event when the user closes the window.
        Ensures all threads and resources are cleaned up properly.
        """
        logging.info("Close event triggered. Shutting down gracefully...")

        # 1. Stop recording if it's active
        if self.state == AppState.RECORDING:
            self.stop_recording()

        # 2. Signal all capture threads to stop if they're still running
        for thread in [self.rgb_capture, self.thermal_capture, self.gsr_capture, self.timestamp_thread]:
            if thread and thread.isRunning():
                thread.stop()

        # 3. Wait for all threads to finish
        for thread in [self.rgb_capture, self.thermal_capture, self.gsr_capture, self.timestamp_thread]:
            if thread and thread.isRunning():
                thread.wait(5000)  # Wait up to 5 seconds for thread to finish

        logging.info("All threads stopped. Application closing.")
        event.accept()  # Accept the close event

    def show_main_window(self):
        """Display the main application window."""
        self.show()
        self._update_ui_for_state()  # Set initial UI state


if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the main application
    main_app = Application()
    main_app.show_main_window()

    # Start the event loop
    sys.exit(app.exec_())
