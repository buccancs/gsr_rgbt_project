# src/processing/feature_engineering.py

import logging
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd

# --- Import from our project ---
from src.processing.data_loader import SessionDataLoader
from src.processing.preprocessing import (
    process_gsr_signal,
    detect_palm_roi,
    extract_roi_signal,
)

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def align_signals(gsr_df: pd.DataFrame, video_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns video-derived signals to the GSR signal timestamps.

    This function uses pandas' resampling and interpolation capabilities to create
    a unified DataFrame where each GSR timestamp has a corresponding set of
    video features.

    Args:
        gsr_df (pd.DataFrame): DataFrame with GSR data and a 'timestamp' index.
        video_signals (pd.DataFrame): DataFrame with video features and a 'timestamp' index.

    Returns:
        pd.DataFrame: A merged DataFrame with signals aligned to the GSR timestamps.
    """
    try:
        # Combine the two dataframes. This will create NaNs where timestamps don't align.
        combined_df = pd.concat(
            [gsr_df.set_index("timestamp"), video_signals.set_index("timestamp")],
            axis=1,
        )

        # Interpolate the video signals to fill NaN values at GSR timestamps.
        # 'time' method is suitable for time-series data.
        aligned_df = combined_df.interpolate(method="time").reindex(
            gsr_df.set_index("timestamp").index
        )

        # Reset index to have 'timestamp' as a column again and remove any remaining NaNs
        aligned_df = aligned_df.reset_index().dropna()

        logging.info(
            f"Successfully aligned GSR and video signals. Resulting shape: {aligned_df.shape}"
        )
        return aligned_df

    except Exception as e:
        logging.error(f"An error occurred during signal alignment: {e}")
        return pd.DataFrame()


def create_feature_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window_size: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping windows from time-series data for sequence modeling.

    Args:
        df (pd.DataFrame): The synchronized DataFrame of features and targets.
        feature_cols (List[str]): A list of column names to be used as input features.
        target_col (str): The name of the column to be used as the prediction target.
        window_size (int): The number of time steps in each window (sequence length).
        step (int): The number of time steps to move forward to create the next window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the feature windows (X)
                                       and the corresponding target values (y).
    """
    X, y = [], []
    num_rows = len(df)

    for i in range(0, num_rows - window_size, step):
        window_features = df[feature_cols].iloc[i : i + window_size].values
        # The target is the value at the end of the window
        target_value = df[target_col].iloc[i + window_size - 1]

        X.append(window_features)
        y.append(target_value)

    X = np.array(X)
    y = np.array(y)

    logging.info(f"Created feature windows. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


# --- Example Pipeline ---
# This demonstrates how the functions would be used to create a final dataset from a session.
def create_dataset_from_session(
    session_path: Path, gsr_sampling_rate: int, video_fps: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Full pipeline to load, preprocess, align, and window data from a single session.
    """
    # 1. Load Data
    loader = SessionDataLoader(session_path)
    gsr_df = loader.get_gsr_data()
    if gsr_df is None:
        return None

    # 2. Preprocess GSR
    processed_gsr = process_gsr_signal(gsr_df, sampling_rate=gsr_sampling_rate)
    if processed_gsr is None:
        return None

    # 3. Extract Signals from Video
    # In a real scenario, you'd iterate through both RGB and thermal generators
    video_features = []
    video_timestamps = []

    frame_interval = 1.0 / video_fps
    current_time_offset = 0.0

    # For simplicity, we process only the RGB video here. A full implementation would merge
    # features from both RGB and thermal streams.
    for success, frame in loader.get_rgb_video_generator():
        if not success:
            continue

        roi = detect_palm_roi(frame)
        if roi:
            # Extract mean BGR values
            rgb_signal = extract_roi_signal(frame, roi)
            video_features.append(rgb_signal)

            # Associate a timestamp with this frame
            frame_timestamp = processed_gsr["timestamp"].iloc[0] + pd.to_timedelta(
                current_time_offset, unit="s"
            )
            video_timestamps.append(frame_timestamp)
            current_time_offset += frame_interval

    if not video_features:
        logging.error("Could not extract any features from the video.")
        return None

    # Create a DataFrame for the video features
    video_df = pd.DataFrame(video_features, columns=["RGB_B", "RGB_G", "RGB_R"])
    video_df["timestamp"] = video_timestamps

    # 4. Align Signals
    aligned_df = align_signals(processed_gsr, video_df)
    if aligned_df.empty:
        return None

    # 5. Create Feature Windows
    feature_columns = ["RGB_B", "RGB_G", "RGB_R", "GSR_Tonic"]  # Example features
    target_column = "GSR_Phasic"  # The target we want to predict

    # Window size: e.g., 5 seconds of data (5s * 32Hz = 160 samples)
    window_size_samples = 5 * gsr_sampling_rate
    step_size = gsr_sampling_rate // 2  # 50% overlap

    X, y = create_feature_windows(
        aligned_df, feature_columns, target_column, window_size_samples, step_size
    )

    return X, y


if __name__ == "__main__":
    # --- Example Usage ---
    # This requires a dummy session with a valid (even if short) video file.
    dummy_session_path = Path("./dummy_session")

    if not dummy_session_path.exists():
        logging.error(
            f"Dummy session not found at '{dummy_session_path}'. Please run data_loader.py to create it."
        )
    else:
        # Create a dummy video file for the generator to use
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        dummy_video_path = dummy_session_path / "rgb_video.mp4"
        out = cv2.VideoWriter(str(dummy_video_path), fourcc, 30.0, (640, 480))
        for _ in range(30 * 15):  # 15 seconds of dummy video
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        print("\n--- Testing Full Feature Engineering Pipeline ---")
        dataset = create_dataset_from_session(
            dummy_session_path, gsr_sampling_rate=32, video_fps=30
        )

        if dataset:
            X_train, y_train = dataset
            print(f"\nSuccessfully created training dataset.")
            print(f"Features (X) shape: {X_train.shape}")
            print(f"Targets (y) shape: {y_train.shape}")
