# src/processing/data_loader.py

import csv
import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def load_gsr_data(session_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads GSR data from the CSV file in a session directory.

    Args:
        session_path (Path): The path to the root directory of a single recording session.

    Returns:
        pd.DataFrame: A pandas DataFrame containing 'timestamp' and 'gsr_value' columns,
                      with the timestamp converted to a datetime object. Returns None if the
                      file is not found or cannot be read.
    """
    gsr_file = session_path / "gsr_data.csv"
    if not gsr_file.exists():
        logging.error(f"GSR data file not found at: {gsr_file}")
        return None

    try:
        df = pd.read_csv(gsr_file)
        # Convert the timestamp string to a proper datetime object for time-series analysis
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logging.info(f"Successfully loaded {len(df)} GSR data points from {gsr_file}")
        return df
    except Exception as e:
        logging.error(f"Failed to load or parse GSR data from {gsr_file}: {e}")
        return None


def video_frame_generator(
    video_path: Path,
) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
    """
    Creates a generator to yield video frames one by one from a video file.

    This approach is memory-efficient as it does not load the entire video into memory.

    Args:
        video_path (Path): The full path to the video file (e.g., 'rgb_video.mp4').

    Yields:
        Iterator[Tuple[bool, Optional[np.ndarray]]]: A tuple containing a success flag (bool)
                                                     and the video frame as a NumPy array.
                                                     Yields (False, None) if the frame cannot be read.
    """
    if not video_path.exists():
        logging.error(f"Video file not found at: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                # End of video has been reached
                break
            yield success, frame
    finally:
        cap.release()
        logging.info(f"Finished processing video file: {video_path}")


class SessionDataLoader:
    """
    A convenience class to load all data associated with a single recording session.
    """

    def __init__(self, session_path: Path):
        """
        Initializes the loader with the path to a session directory.

        Args:
            session_path (Path): The path to the directory containing all data for one session
                                 (e.g., './data/recordings/Subject_01_20250619_011530').
        """
        if not session_path.is_dir():
            raise FileNotFoundError(
                f"The specified session path does not exist or is not a directory: {session_path}"
            )

        self.session_path = session_path
        self.rgb_video_path = session_path / "rgb_video.mp4"
        self.thermal_video_path = session_path / "thermal_video.mp4"

        logging.info(f"Initialized data loader for session: {session_path.name}")

    def get_gsr_data(self) -> Optional[pd.DataFrame]:
        """Loads and returns the GSR data for the session."""
        return load_gsr_data(self.session_path)

    def get_rgb_video_generator(self) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
        """Returns a generator for the RGB video frames."""
        return video_frame_generator(self.rgb_video_path)

    def get_thermal_video_generator(
        self,
    ) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
        """Returns a generator for the thermal video frames."""
        return video_frame_generator(self.thermal_video_path)


# --- Example Usage ---
# This demonstrates how the loader would be used in a processing script.
if __name__ == "__main__":
    # Create a dummy session directory and files for demonstration
    dummy_path = Path("./dummy_session")
    dummy_path.mkdir(exist_ok=True)

    # Dummy GSR file
    with open(dummy_path / "gsr_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gsr_value"])
        writer.writerow(["2025-06-19T01:30:00.000Z", 0.81])
        writer.writerow(["2025-06-19T01:30:00.031Z", 0.82])

    # Dummy video file (just needs to exist for the example)
    (dummy_path / "rgb_video.mp4").touch()

    print("--- Testing SessionDataLoader ---")
    try:
        loader = SessionDataLoader(dummy_path)

        # Load GSR
        gsr_df = loader.get_gsr_data()
        if gsr_df is not None:
            print("\nLoaded GSR Data:")
            print(gsr_df.head())

        # Check video generators
        print("\nChecking video generators...")
        rgb_gen = loader.get_rgb_video_generator()
        print(f"RGB video generator created: {rgb_gen}")

    except FileNotFoundError as e:
        print(e)
