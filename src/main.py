# src/main.py

import sys
import logging
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, QThread, pyqtSlot

from src.data_collection.gui.main_window import MainWindow
from src.data_collection.capture.video_capture import VideoCaptureThread
from src.data_collection.capture.thermal_capture import ThermalCaptureThread
from src.data_collection.capture.gsr_capture import GsrCaptureThread
from src.data_collection.utils.data_logger import DataLogger
from src.data_collection.utils.timestamp_thread import TimestampThread
from src import config

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class Application(QObject):
    """
    Main application class that orchestrates the GUI and data capture components.

    This class connects the MainWindow UI with the capture threads and data logger,
    handling the application's core logic and event flow.
    """

    def __init__(self):
        super().__init__()

        # Create the main window
        self.main_window = MainWindow()

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

        self.gsr_capture = GsrCaptureThread(
            port=config.GSR_SENSOR_PORT,
            sampling_rate=config.GSR_SAMPLING_RATE,
            simulation_mode=config.GSR_SIMULATION_MODE
        )

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

    def start_recording(self):
        """Start all capture threads and initialize data logging."""
        try:
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
            self.rgb_capture.frame_captured.connect(
                lambda frame, _: self.data_logger.log_rgb_frame(frame, self.latest_timestamp)
            )
            self.thermal_capture.frame_captured.connect(
                lambda frame, _: self.data_logger.log_thermal_frame(frame, self.latest_timestamp)
            )
            self.gsr_capture.gsr_data_point.connect(self.data_logger.log_gsr_data)

            # Start capture threads
            self.rgb_capture.start()
            self.thermal_capture.start()
            self.gsr_capture.start()

            # Update UI
            self.main_window.start_button.setEnabled(False)
            self.main_window.stop_button.setEnabled(True)
            self.main_window.subject_id_input.setEnabled(False)

            logging.info(f"Recording started for subject: {subject_id}")

        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            self.main_window.show_error_message(
                "Recording Error", f"Failed to start recording: {str(e)}"
            )

    def stop_recording(self):
        """Stop all capture threads and finalize data logging."""
        try:
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

            # Update UI
            self.main_window.start_button.setEnabled(True)
            self.main_window.stop_button.setEnabled(False)
            self.main_window.subject_id_input.setEnabled(True)

            logging.info("Recording stopped")

        except Exception as e:
            logging.error(f"Error while stopping recording: {e}")
            self.main_window.show_error_message(
                "Recording Error", f"Error while stopping recording: {str(e)}"
            )

    def show_main_window(self):
        """Display the main application window."""
        self.main_window.show()


if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the main application
    main_app = Application()
    main_app.show_main_window()

    # Start the event loop
    sys.exit(app.exec_())
