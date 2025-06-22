# src/data_collection/utils/data_logger.py

import csv
import json
import logging
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

def get_git_commit_hash() -> str:
    """Gets the current git commit hash of the repository."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"


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
        self.subject_id = subject_id
        self.fps = fps
        self.video_fourcc = cv2.VideoWriter_fourcc(*video_fourcc)
        self.is_logging = False

        # Initialize data queue for buffered writing
        self.data_queue = queue.Queue(maxsize=1000)  # Buffer up to 1000 data items
        self.writer_thread = None

        # Frame counters for video frames
        self.frame_count_rgb = 0
        self.frame_count_thermal = 0

        self.rgb_writer = None
        self.thermal_writer = None
        self.gsr_writer = None
        self.gsr_csv_file = None
        self.rgb_timestamps_writer = None
        self.rgb_timestamps_file = None
        self.thermal_timestamps_writer = None
        self.thermal_timestamps_file = None

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

        # Track initialization success for each component
        video_writers_initialized = False
        gsr_writer_initialized = False
        timestamp_writers_initialized = False
        session_info_created = False

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

            if not self.rgb_writer.isOpened() or not self.thermal_writer.isOpened():
                raise IOError("Failed to open one or both video writers")

            video_writers_initialized = True

            # Setup CSV writer for GSR data
            gsr_csv_path = self.session_path / "gsr_data.csv"
            # Open file with write permissions and ensure it's closed properly
            self.gsr_csv_file = open(gsr_csv_path, "w", newline="", encoding="utf-8")
            self.gsr_writer = csv.writer(self.gsr_csv_file)
            # Write the header row for the CSV file
            self.gsr_writer.writerow(["system_timestamp", "shimmer_timestamp", "gsr_value"])

            gsr_writer_initialized = True

            # Setup CSV writers for frame timestamps
            rgb_timestamps_path = self.session_path / "rgb_timestamps.csv"
            thermal_timestamps_path = self.session_path / "thermal_timestamps.csv"

            self.rgb_timestamps_file = open(rgb_timestamps_path, "w", newline="", encoding="utf-8")
            self.rgb_timestamps_writer = csv.writer(self.rgb_timestamps_file)
            self.rgb_timestamps_writer.writerow(["frame_number", "timestamp"])

            self.thermal_timestamps_file = open(thermal_timestamps_path, "w", newline="", encoding="utf-8")
            self.thermal_timestamps_writer = csv.writer(self.thermal_timestamps_file)
            self.thermal_timestamps_writer.writerow(["frame_number", "timestamp"])

            timestamp_writers_initialized = True

            # Create session_info.json with enhanced metadata
            from src import config
            session_info = {
                "participant_id": self.subject_id,
                "session_id": self.session_timestamp,
                "start_time_utc": time.time(),
                "software_version": get_git_commit_hash(),
                "config_parameters": {
                    "FPS": self.fps,
                    "VIDEO_FOURCC": config.VIDEO_FOURCC,
                    "FRAME_WIDTH": frame_size_rgb[0],
                    "FRAME_HEIGHT": frame_size_rgb[1],
                    "RGB_CAMERA_ID": config.RGB_CAMERA_ID,
                    "THERMAL_CAMERA_ID": config.THERMAL_CAMERA_ID,
                    "THERMAL_SIMULATION_MODE": config.THERMAL_SIMULATION_MODE,
                    "GSR_SAMPLING_RATE": config.GSR_SAMPLING_RATE,
                    "GSR_SIMULATION_MODE": config.GSR_SIMULATION_MODE,
                }
            }

            info_path = self.session_path / "session_info.json"
            with open(info_path, 'w') as f:
                json.dump(session_info, f, indent=4)

            session_info_created = True
            logging.info(f"Session metadata saved to {info_path}")

            # Start the writer thread
            self.frame_count_rgb = 0
            self.frame_count_thermal = 0
            self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.writer_thread.start()
            logging.info("Writer thread started")

            self.is_logging = True
            logging.info("Logging has started. Video and CSV writers are ready.")

        except IOError as e:
            logging.error(f"I/O error initializing log files: {e}")
            self.is_logging = False
            self._cleanup_partial_initialization(video_writers_initialized, 
                                               gsr_writer_initialized,
                                               timestamp_writers_initialized)
        except Exception as e:
            logging.error(f"Failed to initialize log files: {e}")
            self.is_logging = False
            self._cleanup_partial_initialization(video_writers_initialized, 
                                               gsr_writer_initialized,
                                               timestamp_writers_initialized)

    def _cleanup_partial_initialization(self, video_initialized, gsr_initialized, timestamps_initialized):
        """
        Cleans up resources if initialization fails partway through.

        Args:
            video_initialized (bool): Whether video writers were successfully initialized
            gsr_initialized (bool): Whether GSR writer was successfully initialized
            timestamps_initialized (bool): Whether timestamp writers were successfully initialized
        """
        if video_initialized:
            if self.rgb_writer:
                self.rgb_writer.release()
            if self.thermal_writer:
                self.thermal_writer.release()

        if gsr_initialized and self.gsr_csv_file:
            self.gsr_csv_file.close()

        if timestamps_initialized:
            if self.rgb_timestamps_file:
                self.rgb_timestamps_file.close()
            if self.thermal_timestamps_file:
                self.thermal_timestamps_file.close()

    def _write_loop(self):
        """
        This method runs in a separate thread and handles all disk writing.
        It processes items from the data queue and writes them to the appropriate files.
        """
        while True:
            try:
                # Get an item from the queue (blocks until an item is available)
                item = self.data_queue.get()

                # Check for sentinel value indicating thread should stop
                if item is None:
                    logging.info("Writer thread received stop signal")
                    break

                # Process the item based on its type
                data_type, data, timestamp = item

                if data_type == 'rgb':
                    frame, timestamp = data
                    self.rgb_writer.write(frame)
                    self.rgb_timestamps_writer.writerow([self.frame_count_rgb, timestamp])
                    self.frame_count_rgb += 1

                elif data_type == 'thermal':
                    frame, timestamp = data
                    self.thermal_writer.write(frame)
                    self.thermal_timestamps_writer.writerow([self.frame_count_thermal, timestamp])
                    self.frame_count_thermal += 1

                elif data_type == 'gsr':
                    gsr_value, shimmer_timestamp = data
                    system_timestamp = datetime.now().isoformat()
                    self.gsr_writer.writerow([system_timestamp, shimmer_timestamp, gsr_value])

                # Mark the task as done
                self.data_queue.task_done()

            except Exception as e:
                logging.error(f"Error in writer thread: {e}")
                # Continue processing other items even if one fails
                continue

    def log_rgb_frame(self, frame, timestamp=None, frame_number=None):
        """
        Adds a single RGB frame to the queue for writing to disk.

        Args:
            frame: The video frame to write
            timestamp (float, optional): The high-resolution timestamp of the frame capture
            frame_number (int, optional): The sequential number of the frame
        """
        if self.is_logging:
            # Add the frame to the queue instead of writing directly
            try:
                item = ('rgb', (frame, timestamp), timestamp)
                self.data_queue.put_nowait(item)  # Non-blocking put
            except queue.Full:
                logging.warning("Data queue is full! Dropping RGB frame.")
                # In a production system, you might want to implement a more sophisticated
                # strategy for handling queue overflow, such as dropping older frames

    def log_thermal_frame(self, frame, timestamp=None, frame_number=None):
        """
        Adds a single thermal frame to the queue for writing to disk.

        Args:
            frame: The video frame to write
            timestamp (float, optional): The high-resolution timestamp of the frame capture
            frame_number (int, optional): The sequential number of the frame
        """
        if self.is_logging:
            # Add the frame to the queue instead of writing directly
            try:
                item = ('thermal', (frame, timestamp), timestamp)
                self.data_queue.put_nowait(item)  # Non-blocking put
            except queue.Full:
                logging.warning("Data queue is full! Dropping thermal frame.")
                # In a production system, you might want to implement a more sophisticated
                # strategy for handling queue overflow, such as dropping older frames

    def log_gsr_data(self, gsr_value: float, shimmer_timestamp: float):
        """
        Adds a single GSR data point to the queue for writing to disk.

        Args:
            gsr_value (float): The GSR value from the sensor
            shimmer_timestamp (float): The timestamp from the Shimmer device
        """
        if self.is_logging:
            # Add the GSR data to the queue instead of writing directly
            try:
                item = ('gsr', (gsr_value, shimmer_timestamp), shimmer_timestamp)
                self.data_queue.put_nowait(item)  # Non-blocking put
            except queue.Full:
                logging.warning("Data queue is full! Dropping GSR data point.")
                # In a production system, you might want to implement a more sophisticated
                # strategy for handling queue overflow, such as aggregating GSR values

    def stop_logging(self):
        """
        Releases all video writers and closes the CSV files.
        This is a critical cleanup step to ensure data is saved correctly.
        """
        if not self.is_logging:
            return

        logging.info("Stopping logger and saving all files...")

        # Stop the writer thread by sending a sentinel value
        if self.writer_thread and self.writer_thread.is_alive():
            logging.info("Stopping writer thread...")
            self.data_queue.put(None)  # Sentinel value to stop the thread
            self.writer_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
            if self.writer_thread.is_alive():
                logging.warning("Writer thread did not stop within timeout. Some data may be lost.")
            else:
                logging.info("Writer thread stopped successfully")

        # Make sure all queued items are processed
        try:
            self.data_queue.join()  # Wait for all queued items to be processed
            logging.info("All queued data items processed")
        except Exception as e:
            logging.warning(f"Error waiting for queue to empty: {e}")

        # Close all file handles
        if self.rgb_writer:
            self.rgb_writer.release()
        if self.thermal_writer:
            self.thermal_writer.release()
        if self.gsr_csv_file:
            self.gsr_csv_file.close()
        if self.rgb_timestamps_file:
            self.rgb_timestamps_file.close()
        if self.thermal_timestamps_file:
            self.thermal_timestamps_file.close()

        self.is_logging = False
        logging.info(f"Logging stopped. All data saved in {self.session_path}")
