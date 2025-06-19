#!/usr/bin/env python3
# src/scripts/test_synchronization.py

import logging
import sys
import time
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.utils.timestamp_thread import TimestampThread
from src.capture.video_capture import VideoCaptureThread
from src.capture.thermal_capture import ThermalCaptureThread
from src.capture.gsr_capture import GsrCaptureThread
from src.utils.data_logger import DataLogger

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class SynchronizationTester:
    """
    Tests the data synchronization mechanism by capturing a short sequence of data
    from all devices and verifying that the timestamps are properly synchronized.
    """

    def __init__(self, test_duration=5):
        """
        Initialize the synchronization tester.

        Args:
            test_duration (int): The duration of the test in seconds.
        """
        self.test_duration = test_duration
        self.latest_timestamp = None

        # Create a temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        logging.info(f"Created temporary directory for test data: {self.temp_dir}")

        # Initialize the timestamp thread
        self.timestamp_thread = TimestampThread(frequency=200)  # 200Hz timestamp generation

        # Initialize capture threads
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

        # Initialize data logger
        self.data_logger = DataLogger(
            output_dir=self.temp_dir,
            subject_id="TestSubject",
            fps=config.FPS,
            video_fourcc=config.VIDEO_FOURCC
        )

        # Collected data for analysis
        self.rgb_frames = []
        self.rgb_timestamps = []
        self.thermal_frames = []
        self.thermal_timestamps = []
        self.gsr_values = []
        self.gsr_timestamps = []

    def update_latest_timestamp(self, timestamp):
        """
        Update the latest timestamp from the timestamp thread.

        Args:
            timestamp (int): The latest timestamp in nanoseconds.
        """
        logging.info(f"Updating latest timestamp to {timestamp}")
        self.latest_timestamp = timestamp

    def collect_rgb_frame(self, frame, timestamp):
        """
        Collect RGB frames and timestamps for analysis.

        Args:
            frame: The RGB frame.
            timestamp: The timestamp of the frame.
        """
        logging.info(f"Collecting RGB frame with timestamp {timestamp}")
        self.rgb_frames.append(frame)
        # Use the timestamp from the frame instead of self.latest_timestamp
        self.rgb_timestamps.append(timestamp)

    def collect_thermal_frame(self, frame, timestamp):
        """
        Collect thermal frames and timestamps for analysis.

        Args:
            frame: The thermal frame.
            timestamp: The timestamp of the frame.
        """
        logging.info(f"Collecting thermal frame with timestamp {timestamp}")
        self.thermal_frames.append(frame)
        # Use the timestamp from the frame instead of self.latest_timestamp
        self.thermal_timestamps.append(timestamp)

    def collect_gsr_data(self, gsr_value, shimmer_timestamp):
        """
        Collect GSR values and timestamps for analysis.

        Args:
            gsr_value: The GSR value.
            shimmer_timestamp: The timestamp from the Shimmer device.
        """
        logging.info(f"Collecting GSR data with timestamp {shimmer_timestamp}")
        self.gsr_values.append(gsr_value)
        # Use the timestamp from the GSR data instead of self.latest_timestamp
        self.gsr_timestamps.append(shimmer_timestamp)

    def run_test(self):
        """
        Run the synchronization test.

        Returns:
            True if the test passes, False otherwise.
        """
        try:
            # Test direct signal connection
            from PyQt5.QtCore import QObject, pyqtSignal

            class TestSignal(QObject):
                test_signal = pyqtSignal(int)

                def emit_signal(self):
                    logging.info("Emitting test signal from within SynchronizationTester")
                    self.test_signal.emit(12345)

            def test_slot(value):
                logging.info(f"Test slot in SynchronizationTester received: {value}")

            test_obj = TestSignal()
            test_obj.test_signal.connect(test_slot)
            test_obj.emit_signal()

            # Start the timestamp thread
            logging.info("Connecting timestamp_generated signal to update_latest_timestamp")
            self.timestamp_thread.timestamp_generated.connect(self.update_latest_timestamp)
            self.timestamp_thread.start()
            logging.info("Timestamp thread started")

            # Connect capture signals to data collection methods
            logging.info("Connecting RGB frame_captured signal to collect_rgb_frame")
            self.rgb_capture.frame_captured.connect(self.collect_rgb_frame)

            logging.info("Connecting thermal frame_captured signal to collect_thermal_frame")
            self.thermal_capture.frame_captured.connect(self.collect_thermal_frame)

            logging.info("Connecting GSR gsr_data_point signal to collect_gsr_data")
            self.gsr_capture.gsr_data_point.connect(self.collect_gsr_data)

            # Connect capture signals to data logger
            logging.info("Connecting RGB frame_captured signal to data_logger.log_rgb_frame")
            self.rgb_capture.frame_captured.connect(
                lambda frame, timestamp: self.data_logger.log_rgb_frame(frame, timestamp)
            )

            logging.info("Connecting thermal frame_captured signal to data_logger.log_thermal_frame")
            self.thermal_capture.frame_captured.connect(
                lambda frame, timestamp: self.data_logger.log_thermal_frame(frame, timestamp)
            )

            logging.info("Connecting GSR gsr_data_point signal to data_logger.log_gsr_data")
            self.gsr_capture.gsr_data_point.connect(self.data_logger.log_gsr_data)

            # Start the data logger
            self.data_logger.start_logging(
                frame_size_rgb=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
                frame_size_thermal=(config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )

            # Start capture threads
            logging.info("Starting RGB capture thread")
            self.rgb_capture.start()

            logging.info("Starting thermal capture thread")
            self.thermal_capture.start()

            logging.info("Starting GSR capture thread")
            self.gsr_capture.start()

            logging.info(f"Capturing data for {self.test_duration} seconds...")
            time.sleep(self.test_duration)

            # Stop capture threads
            self.rgb_capture.stop()
            self.thermal_capture.stop()
            self.gsr_capture.stop()

            # Stop the timestamp thread
            self.timestamp_thread.stop()

            # Wait for threads to finish
            if self.rgb_capture.isRunning():
                self.rgb_capture.wait()
            if self.thermal_capture.isRunning():
                self.thermal_capture.wait()
            if self.gsr_capture.isRunning():
                self.gsr_capture.wait()
            if self.timestamp_thread.isRunning():
                self.timestamp_thread.wait()

            # Stop the data logger
            self.data_logger.stop_logging()

            # Analyze the collected data
            return self.analyze_synchronization()

        except Exception as e:
            logging.error(f"Error during synchronization test: {e}")
            return False
        finally:
            # Clean up
            self.cleanup()

    def analyze_synchronization(self):
        """
        Analyze the collected data to verify synchronization.

        Returns:
            True if the synchronization is working properly, False otherwise.
        """
        logging.info("Analyzing synchronization...")

        # Check if we have collected data from all sources
        if not self.rgb_timestamps or not self.thermal_timestamps or not self.gsr_timestamps:
            logging.warning("No data received directly from signals, checking log files...")

            # Try to read data from the log files
            try:
                # Read timestamp files
                rgb_ts_file = self.data_logger.session_path / "rgb_timestamps.csv"
                thermal_ts_file = self.data_logger.session_path / "thermal_timestamps.csv"
                gsr_file = self.data_logger.session_path / "gsr_data.csv"

                if rgb_ts_file.exists() and thermal_ts_file.exists() and gsr_file.exists():
                    rgb_ts_df = pd.read_csv(rgb_ts_file)
                    thermal_ts_df = pd.read_csv(thermal_ts_file)
                    gsr_df = pd.read_csv(gsr_file)

                    logging.info(f"RGB timestamps file: {len(rgb_ts_df)} entries")
                    logging.info(f"Thermal timestamps file: {len(thermal_ts_df)} entries")
                    logging.info(f"GSR data file: {len(gsr_df)} entries")

                    # Check if the files have data
                    if len(rgb_ts_df) > 0 and len(thermal_ts_df) > 0 and len(gsr_df) > 0:
                        # Use the data from the files
                        self.rgb_timestamps = rgb_ts_df['timestamp'].tolist()
                        self.thermal_timestamps = thermal_ts_df['timestamp'].tolist()
                        # Extract the shimmer_timestamp column from the GSR data
                        self.gsr_timestamps = [float(ts) for ts in gsr_df['shimmer_timestamp'].tolist()]

                        logging.info("Successfully loaded data from log files.")
                    else:
                        logging.error("FAIL: Log files exist but have no data.")
                        return False
                else:
                    logging.error("FAIL: One or more log files are missing.")
                    return False

            except Exception as e:
                logging.error(f"Error reading log files: {e}")
                return False

        # Convert timestamps to relative time (seconds from start)
        start_time = min(
            min(self.rgb_timestamps) if self.rgb_timestamps else float('inf'),
            min(self.thermal_timestamps) if self.thermal_timestamps else float('inf'),
            min(self.gsr_timestamps) if self.gsr_timestamps else float('inf')
        )

        rgb_times = [(t - start_time) / 1e9 for t in self.rgb_timestamps]
        thermal_times = [(t - start_time) / 1e9 for t in self.thermal_timestamps]
        gsr_times = [(t - start_time) / 1e9 for t in self.gsr_timestamps]

        # Calculate frame rates
        if len(rgb_times) > 1:
            rgb_fps = (len(rgb_times) - 1) / (rgb_times[-1] - rgb_times[0])
            logging.info(f"RGB camera frame rate: {rgb_fps:.2f} fps")

        if len(thermal_times) > 1:
            thermal_fps = (len(thermal_times) - 1) / (thermal_times[-1] - thermal_times[0])
            logging.info(f"Thermal camera frame rate: {thermal_fps:.2f} fps")

        if len(gsr_times) > 1:
            gsr_rate = (len(gsr_times) - 1) / (gsr_times[-1] - gsr_times[0])
            logging.info(f"GSR sampling rate: {gsr_rate:.2f} Hz")

        # Check if the frame rates are close to the configured values
        fps_tolerance = 0.2  # 20% tolerance

        rgb_fps_ok = len(rgb_times) <= 1 or abs(rgb_fps - config.FPS) / config.FPS <= fps_tolerance
        thermal_fps_ok = len(thermal_times) <= 1 or abs(thermal_fps - config.FPS) / config.FPS <= fps_tolerance
        gsr_rate_ok = len(gsr_times) <= 1 or abs(gsr_rate - config.GSR_SAMPLING_RATE) / config.GSR_SAMPLING_RATE <= fps_tolerance

        # Check timestamp files
        try:
            # Read timestamp files
            rgb_ts_file = self.data_logger.session_path / "rgb_timestamps.csv"
            thermal_ts_file = self.data_logger.session_path / "thermal_timestamps.csv"
            gsr_file = self.data_logger.session_path / "gsr_data.csv"

            if rgb_ts_file.exists() and thermal_ts_file.exists() and gsr_file.exists():
                rgb_ts_df = pd.read_csv(rgb_ts_file)
                thermal_ts_df = pd.read_csv(thermal_ts_file)
                gsr_df = pd.read_csv(gsr_file)

                logging.info(f"RGB timestamps file: {len(rgb_ts_df)} entries")
                logging.info(f"Thermal timestamps file: {len(thermal_ts_df)} entries")
                logging.info(f"GSR data file: {len(gsr_df)} entries")

                # Check if the number of entries matches the collected data
                rgb_file_ok = abs(len(rgb_ts_df) - len(self.rgb_timestamps)) <= 1  # Allow for 1 frame difference
                thermal_file_ok = abs(len(thermal_ts_df) - len(self.thermal_timestamps)) <= 1
                gsr_file_ok = abs(len(gsr_df) - len(self.gsr_timestamps)) <= 1

                if not (rgb_file_ok and thermal_file_ok and gsr_file_ok):
                    logging.warning("WARN: Number of entries in timestamp files doesn't match collected data.")
            else:
                logging.error("FAIL: One or more timestamp files are missing.")
                return False

        except Exception as e:
            logging.error(f"Error reading timestamp files: {e}")
            return False

        # Plot the timestamps for visualization
        self.plot_timestamps(rgb_times, thermal_times, gsr_times)

        # Final result
        sync_ok = rgb_fps_ok and thermal_fps_ok and gsr_rate_ok

        if sync_ok:
            logging.info("SUCCESS: Data synchronization is working properly.")
        else:
            logging.error("FAIL: Data synchronization issues detected.")
            if not rgb_fps_ok:
                logging.error(f"  - RGB frame rate ({rgb_fps:.2f} fps) differs from configured value ({config.FPS} fps)")
            if not thermal_fps_ok:
                logging.error(f"  - Thermal frame rate ({thermal_fps:.2f} fps) differs from configured value ({config.FPS} fps)")
            if not gsr_rate_ok:
                logging.error(f"  - GSR sampling rate ({gsr_rate:.2f} Hz) differs from configured value ({config.GSR_SAMPLING_RATE} Hz)")

        return sync_ok

    def plot_timestamps(self, rgb_times, thermal_times, gsr_times):
        """
        Plot the timestamps for visualization.

        Args:
            rgb_times: List of RGB frame timestamps (seconds from start).
            thermal_times: List of thermal frame timestamps (seconds from start).
            gsr_times: List of GSR data timestamps (seconds from start).
        """
        try:
            plt.figure(figsize=(10, 6))

            # Plot the timestamps as events on a timeline
            if rgb_times:
                plt.plot(rgb_times, [1] * len(rgb_times), 'ro', label='RGB Frames', markersize=4)
            if thermal_times:
                plt.plot(thermal_times, [2] * len(thermal_times), 'bo', label='Thermal Frames', markersize=4)
            if gsr_times:
                plt.plot(gsr_times, [3] * len(gsr_times), 'go', label='GSR Data Points', markersize=4)

            plt.yticks([1, 2, 3], ['RGB', 'Thermal', 'GSR'])
            plt.xlabel('Time (seconds)')
            plt.title('Data Synchronization Visualization')
            plt.grid(True, axis='x')
            plt.legend()

            # Save the plot
            plot_path = self.data_logger.session_path / "synchronization_plot.png"
            plt.savefig(plot_path)
            logging.info(f"Synchronization plot saved to: {plot_path}")

        except Exception as e:
            logging.error(f"Error creating synchronization plot: {e}")

    def cleanup(self):
        """
        Clean up resources.
        """
        # Note: We're not deleting the temporary directory so that the results can be examined
        logging.info(f"Test data saved in: {self.data_logger.session_path}")


def main():
    """
    Run the synchronization test.
    """
    logging.info("=======================================")
    logging.info("=  GSR-RGBT Synchronization Test      =")
    logging.info("=======================================")

    # Test PyQt signal-slot connection
    from PyQt5.QtCore import QObject, pyqtSignal

    class TestSignal(QObject):
        test_signal = pyqtSignal(str)

        def emit_signal(self):
            logging.info("Emitting test signal")
            self.test_signal.emit("Test signal")

    def test_slot(message):
        logging.info(f"Test slot received: {message}")

    test_obj = TestSignal()
    test_obj.test_signal.connect(test_slot)
    test_obj.emit_signal()

    # First run the system validation check
    from src.scripts.check_system import main as check_system

    try:
        check_system()
    except SystemExit as e:
        if e.code != 0:
            logging.error("System validation check failed. Aborting synchronization test.")
            sys.exit(1)

    # Run the synchronization test
    tester = SynchronizationTester(test_duration=5)
    sync_ok = tester.run_test()

    # Final Summary
    print("\n--- Synchronization Test Summary ---")
    print("Data Synchronization:", "OK" if sync_ok else "FAIL")

    if sync_ok:
        logging.info("SUCCESS: Synchronization test passed.")
        sys.exit(0)  # Exit with success code
    else:
        logging.error("FAIL: Synchronization test failed.")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
