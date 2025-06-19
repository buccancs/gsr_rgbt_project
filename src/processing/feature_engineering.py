# src/processing/feature_engineering.py

import logging
from pathlib import Path
from typing import Tuple, List, Optional

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

# Try to import Cython optimizations
try:
    from src.processing.cython_optimizations import cy_create_feature_windows, cy_align_signals
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


def align_signals(gsr_df: pd.DataFrame, video_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns video-derived signals to the GSR signal timestamps.

    This function uses interpolation to create a unified DataFrame where each GSR 
    timestamp has a corresponding set of video features. It supports both Cython 
    and pure Python implementations for performance optimization.

    The function handles exceptions gracefully and returns an empty DataFrame if 
    any errors occur during the alignment process.

    Args:
        gsr_df (pd.DataFrame): DataFrame with GSR data and a 'timestamp' column.
            The timestamp column must contain datetime objects.
        video_signals (pd.DataFrame): DataFrame with video features and a 'timestamp' column.
            The timestamp column must contain datetime objects.

    Returns:
        pd.DataFrame: A merged DataFrame with signals aligned to the GSR timestamps.
            If an error occurs, an empty DataFrame is returned.

    Examples:
        >>> gsr_data = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01']),
        ...     'GSR_Phasic': [0.1, 0.2]
        ... })
        >>> video_data = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2023-01-01 00:00:00.5', '2023-01-01 00:00:01.5']),
        ...     'RGB_R': [100, 110]
        ... })
        >>> aligned_df = align_signals(gsr_data, video_data)
        >>> print(aligned_df.shape)
        (2, 3)
    """
    try:
        if CYTHON_AVAILABLE:
            # Use the Cython implementation for better performance
            # Convert timestamps to nanoseconds for numerical operations
            gsr_timestamps = pd.to_datetime(gsr_df["timestamp"]).astype(np.int64).values
            video_timestamps = pd.to_datetime(video_signals["timestamp"]).astype(np.int64).values

            # Extract data columns (excluding timestamp)
            gsr_data = gsr_df.drop(columns=["timestamp"]).values
            video_data = video_signals.drop(columns=["timestamp"]).values

            # Call the Cython function
            aligned_data = cy_align_signals(gsr_data, video_data, gsr_timestamps, video_timestamps)

            # Create a new DataFrame with the aligned data
            # First, create column names for the result
            gsr_cols = gsr_df.columns.drop("timestamp").tolist()
            video_cols = video_signals.columns.drop("timestamp").tolist()
            result_cols = gsr_cols + video_cols

            # Create the DataFrame
            aligned_df = pd.DataFrame(aligned_data, columns=result_cols)
            aligned_df.insert(0, "timestamp", pd.to_datetime(gsr_timestamps))

            logging.info(
                f"Successfully aligned GSR and video signals using Cython. Resulting shape: {aligned_df.shape}"
            )
            return aligned_df
        else:
            # Use the original pandas implementation
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
                f"Successfully aligned GSR and video signals using pandas. Resulting shape: {aligned_df.shape}"
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

    This function transforms a DataFrame of time-series data into a format suitable
    for sequence modeling by creating overlapping windows of features and their
    corresponding target values. It supports both Cython and pure Python
    implementations for performance optimization.

    The function creates windows by sliding a window of size `window_size` over the
    data with a step size of `step`. For each window, it extracts the features and
    the target value at the end of the window.

    Args:
        df (pd.DataFrame): The synchronized DataFrame of features and targets.
            Must contain columns specified in feature_cols and target_col.
        feature_cols (List[str]): A list of column names to be used as input features.
            All columns must exist in the DataFrame.
        target_col (str): The name of the column to be used as the prediction target.
            Must exist in the DataFrame.
        window_size (int): The number of time steps in each window (sequence length).
            Must be a positive integer less than the length of the DataFrame.
        step (int): The number of time steps to move forward to create the next window.
            Must be a positive integer. Smaller values create more overlapping windows.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Feature windows with shape (n_windows, window_size, n_features)
            - y (np.ndarray): Target values with shape (n_windows,)
              where n_windows = floor((len(df) - window_size) / step) + 1

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> df = pd.DataFrame({
        ...     'feature1': np.arange(100),
        ...     'feature2': np.arange(100, 200),
        ...     'target': np.arange(200, 300)
        ... })
        >>> # Create windows with size 10 and step 5
        >>> X, y = create_feature_windows(df, ['feature1', 'feature2'], 'target', 10, 5)
        >>> print(X.shape)  # (18, 10, 2)
        >>> print(y.shape)  # (18,)
    """
    if CYTHON_AVAILABLE:
        # Use the Cython implementation for better performance
        # Extract features and target as numpy arrays
        features = df[feature_cols].values
        targets = df[target_col].values

        # Get column indices for the Cython function
        feature_cols_idx = list(range(len(feature_cols)))

        # Call the Cython function
        X, y = cy_create_feature_windows(features, targets, feature_cols_idx, window_size, step)

        logging.info(f"Created feature windows using Cython. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    else:
        # Use the original Python implementation
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

        logging.info(f"Created feature windows using Python. X shape: {X.shape}, y shape: {y.shape}")
        return X, y


# --- Example Pipeline ---
# This demonstrates how the functions would be used to create a final dataset from a session.
def create_dataset_from_session(
    session_path: Path, gsr_sampling_rate: int, video_fps: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Full pipeline to load, preprocess, align, and window data from a single session.

    This function implements the complete data processing pipeline for a single recording
    session. It performs the following steps:
    1. Loads GSR data from the session
    2. Preprocesses the GSR signal to extract tonic and phasic components
    3. Extracts RGB signals from video frames using palm ROI detection
    4. Aligns the GSR and video signals to a common timeline
    5. Creates feature windows for machine learning

    The function handles potential failures at each step and returns None if any
    critical step fails.

    Args:
        session_path (Path): Path to the session directory containing GSR and video data.
            The directory should have the structure expected by SessionDataLoader.
        gsr_sampling_rate (int): Sampling rate of the GSR signal in Hz.
            Used for preprocessing and windowing.
        video_fps (int): Frame rate of the video in frames per second.
            Used to associate timestamps with video frames.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: If successful, returns a tuple containing:
            - X (np.ndarray): Feature windows with shape (n_windows, window_size, n_features)
            - y (np.ndarray): Target values with shape (n_windows,)
          If any step fails, returns None.

    Notes:
        - The function uses the following feature columns: ["RGB_B", "RGB_G", "RGB_R", "GSR_Tonic"]
        - The target column is "GSR_Phasic"
        - The window size is set to 5 seconds of data (5 * gsr_sampling_rate samples)
        - The step size is set to 50% overlap (gsr_sampling_rate // 2)

    Example:
        >>> from pathlib import Path
        >>> session_path = Path("data/recordings/Subject_01_20250101_000000")
        >>> X, y = create_dataset_from_session(session_path, 32, 30)
        >>> if X is not None:
        ...     print(f"Created dataset with {X.shape[0]} windows")
        ... else:
        ...     print("Failed to create dataset from session")
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
