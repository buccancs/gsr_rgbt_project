# src/utils/data_logger.py

import csv
import logging
from datetime import datetime
from pathlib import Path

import cv2

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class DataLogger:
    """
    Handles the organized and synchronized writing of multimodal data streams to disk.

    For each recording session, this class creates a unique, timestamped directory.
    It manages separate VideoWriter objects for RGB and thermal camera streams
    and a CSV writer for time-stamped GSR sensor data, ensuring all outputs
    are saved in a structured format suitable for later analysis.
    """

    def __init__(self, output_dir: Path, subject_id: str, fps: int, video_fourcc: str):
        """
        Initializes the DataLogger and creates a new session directory.

        Args:
            output_dir (Path): The base directory where all session folders will be stored.
            subject_id (str): The unique identifier for the participant.
            fps (int): The frames per second for the video writers.
            video_fourcc (str): The four-character code for the video codec (e.g., 'mp4v').
        """
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = (
            output_dir / f"Subject_{subject_id}_{self.session_timestamp}"
        )
        self.fps = fps
        self.video_fourcc = cv2.VideoWriter_fourcc(*video_fourcc)
        self.is_logging = False

        self.rgb_writer = None
        self.thermal_writer = None
        self.gsr_writer = None
        self.gsr_csv_file = None

        try:
            self.session_path.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"DataLogger initialized. Session path created at: {self.session_path}"
            )
        except OSError as e:
            logging.error(
                f"Failed to create session directory {self.session_path}: {e}"
            )
            raise

    def start_logging(self, frame_size_rgb: tuple, frame_size_thermal: tuple):
        """
        Initializes video writers and opens the CSV file for GSR data logging.

        This method must be called before any data can be logged. It sets up the
        file handlers and flags the logger as active.

        Args:
            frame_size_rgb (tuple): The (width, height) of the RGB video frames.
            frame_size_thermal (tuple): The (width, height) of the thermal video frames.
        """
        if self.is_logging:
            logging.warning("start_logging called, but logging is already active.")
            return

        try:
            # Setup video writers for both streams
            rgb_video_path = self.session_path / "rgb_video.mp4"
            thermal_video_path = self.session_path / "thermal_video.mp4"
            self.rgb_writer = cv2.VideoWriter(
                str(rgb_video_path), self.video_fourcc, self.fps, frame_size_rgb
            )
            self.thermal_writer = cv2.VideoWriter(
                str(thermal_video_path), self.video_fourcc, self.fps, frame_size_thermal
            )

            # Setup CSV writer for GSR data
            gsr_csv_path = self.session_path / "gsr_data.csv"
            # Open file with write permissions and ensure it's closed properly
            self.gsr_csv_file = open(gsr_csv_path, "w", newline="", encoding="utf-8")
            self.gsr_writer = csv.writer(self.gsr_csv_file)
            # Write the header row for the CSV file
            self.gsr_writer.writerow(["system_timestamp", "shimmer_timestamp", "gsr_value"])

            self.is_logging = True
            logging.info("Logging has started. Video and CSV writers are ready.")

        except Exception as e:
            logging.error(f"Failed to initialize log files: {e}")
            self.is_logging = False

    def log_rgb_frame(self, frame):
        """Writes a single RGB frame to the video file."""
        if self.rgb_writer and self.is_logging:
            self.rgb_writer.write(frame)

    def log_thermal_frame(self, frame):
        """Writes a single thermal frame to the video file."""
        if self.thermal_writer and self.is_logging:
            self.thermal_writer.write(frame)

    def log_gsr_data(self, gsr_value: float, shimmer_timestamp: float):
        """
        Writes a single GSR data point with timestamps to the CSV file.

        Args:
            gsr_value (float): The GSR value from the sensor
            shimmer_timestamp (float): The timestamp from the Shimmer device
        """
        if self.gsr_writer and self.is_logging:
            # Use ISO 8601 format for precise, standardized system timestamps
            system_timestamp = datetime.now().isoformat()
            self.gsr_writer.writerow([system_timestamp, shimmer_timestamp, gsr_value])

    def stop_logging(self):
        """
        Releases all video writers and closes the CSV file.
        This is a critical cleanup step to ensure data is saved correctly.
        """
        if not self.is_logging:
            return

        logging.info("Stopping logger and saving all files...")
        if self.rgb_writer:
            self.rgb_writer.release()
        if self.thermal_writer:
            self.thermal_writer.release()
        if self.gsr_csv_file:
            self.gsr_csv_file.close()

        self.is_logging = False
        logging.info(f"Logging stopped. All data saved in {self.session_path}")
