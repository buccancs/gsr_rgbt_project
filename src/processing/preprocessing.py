# src/processing/preprocessing.py

import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import neurokit2 as nk
import numpy as np
import pandas as pd

# --- Import from our project ---
from src.processing.data_loader import SessionDataLoader

# Try to import Cython optimizations
try:
    from src.processing.cython_optimizations import cy_extract_roi_signal
    CYTHON_AVAILABLE = True
    logging.info("Cython optimizations are available and will be used.")
except ImportError:
    CYTHON_AVAILABLE = False
    logging.warning("Cython optimizations are not available. Using pure Python implementations.")

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def process_gsr_signal(
    gsr_df: pd.DataFrame, sampling_rate: int
) -> Optional[pd.DataFrame]:
    """
    Cleans and processes a raw GSR signal DataFrame.

    This function uses the NeuroKit2 library to perform standard psychophysiological
    signal processing. It cleans the raw signal and decomposes it into its tonic
    and phasic components, which are essential for model training.

    Args:
        gsr_df (pd.DataFrame): DataFrame containing 'gsr_value' and 'timestamp' columns.
        sampling_rate (int): The sampling rate of the GSR signal in Hz.

    Returns:
        pd.DataFrame: The original DataFrame augmented with cleaned, tonic, and
                      phasic GSR signal columns. Returns None on error.
    """
    if "gsr_value" not in gsr_df.columns:
        logging.error("GSR DataFrame must contain a 'gsr_value' column.")
        return None

    try:
        # Process the signal using NeuroKit2
        signals, info = nk.eda_process(gsr_df["gsr_value"], sampling_rate=sampling_rate)

        # Rename columns for clarity and merge back into the original DataFrame
        processed_df = signals.rename(
            columns={
                "EDA_Raw": "GSR_Raw",
                "EDA_Clean": "GSR_Clean",
                "EDA_Tonic": "GSR_Tonic",
                "EDA_Phasic": "GSR_Phasic",
            }
        )

        # Keep only the essential processed columns and merge with original timestamps
        result_df = pd.concat(
            [
                gsr_df.reset_index(drop=True),
                processed_df[["GSR_Clean", "GSR_Tonic", "GSR_Phasic"]],
            ],
            axis=1,
        )

        logging.info(
            "GSR signal successfully processed into clean, tonic, and phasic components."
        )
        return result_df

    except Exception as e:
        logging.error(
            f"An error occurred during GSR signal processing with NeuroKit2: {e}"
        )
        return None


def detect_palm_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the palm region in a video frame.

    For this MVP, we use a simple placeholder: a fixed rectangle in the center of
    the frame. In a full implementation, this should be replaced with a robust
    hand/palm detection model (e.g., from MediaPipe or a custom-trained detector).

    Args:
        frame (np.ndarray): The input video frame.

    Returns:
        Optional[Tuple[int, int, int, int]]: A tuple representing the bounding box
                                             (x, y, width, height) of the detected palm.
                                             Returns None if no palm is found.
    """
    h, w, _ = frame.shape

    # Placeholder ROI: a rectangle covering 40% of the width and 60% of the height, centered.
    roi_width = int(w * 0.4)
    roi_height = int(h * 0.6)
    roi_x = (w - roi_width) // 2
    roi_y = (h - roi_height) // 2

    return (roi_x, roi_y, roi_width, roi_height)


def extract_roi_signal(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extracts the mean pixel value from a specified Region of Interest (ROI).

    Args:
        frame (np.ndarray): The video frame from which to extract the signal.
        roi (Tuple[int, int, int, int]): The bounding box (x, y, w, h) of the ROI.

    Returns:
        np.ndarray: An array containing the mean value for each channel (e.g., [R, G, B]).
    """
    if CYTHON_AVAILABLE:
        # Use the Cython implementation for better performance
        # Ensure frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Call the Cython function
        mean_signal = cy_extract_roi_signal(frame, roi)
        return mean_signal
    else:
        # Use the original OpenCV implementation
        x, y, w, h = roi
        palm_region = frame[y : y + h, x : x + w]

        # Calculate the mean value for each channel within the ROI
        mean_signal = cv2.mean(palm_region)[:3]  # Take only the B, G, R components
        return np.array(mean_signal)


# --- Example Usage ---
# This demonstrates how the functions would be used in a processing script.
if __name__ == "__main__":
    # Assume a dummy session exists from the data_loader example
    dummy_session_path = Path("./dummy_session")
    if not dummy_session_path.exists():
        logging.error(
            "Dummy session directory not found. Please run data_loader.py first to create it."
        )
    else:
        print("--- Testing Preprocessing Functions ---")

        # 1. Load data using the SessionDataLoader
        loader = SessionDataLoader(dummy_session_path)
        gsr_df_raw = loader.get_gsr_data()

        # Add more dummy data for realistic processing
        if gsr_df_raw is not None:
            num_points = 32 * 10  # 10 seconds of data at 32Hz
            dummy_gsr = np.random.randn(num_points) * 0.05 + 0.8
            dummy_timestamps = pd.to_datetime(
                np.arange(num_points) / 32, unit="s", origin="2025-06-19T10:00:00"
            )
            gsr_df_raw = pd.DataFrame(
                {"timestamp": dummy_timestamps, "gsr_value": dummy_gsr}
            )

            # 2. Process the GSR signal
            print("\nProcessing GSR signal...")
            processed_gsr_df = process_gsr_signal(gsr_df_raw, sampling_rate=32)
            if processed_gsr_df is not None:
                print("Processed GSR DataFrame head:")
                print(processed_gsr_df.head())

        # 3. Process a dummy video frame
        print("\nProcessing a dummy video frame...")
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Detect palm ROI
        palm_roi = detect_palm_roi(dummy_frame)
        if palm_roi:
            print(f"Detected palm ROI (placeholder): {palm_roi}")

            # Extract signal from ROI
            roi_signal = extract_roi_signal(dummy_frame, palm_roi)
            print(f"Extracted mean ROI signal (B, G, R): {roi_signal}")
