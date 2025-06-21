# src/processing/feature_engineering.py

import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd

# --- Import from our project ---
from src.processing.data_loader import SessionDataLoader
from src.processing.preprocessing import (
    process_gsr_signal,
    process_frame_with_multi_roi,
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
            window_features = df[feature_cols].iloc[i: i + window_size].values
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
        session_path: Path, gsr_sampling_rate: int, video_fps: int,
        feature_columns: List[str] = None, target_column: str = "GSR_Phasic",
        use_thermal: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Full pipeline to load, preprocess, align, and window data from a single session.

    This function implements the complete data processing pipeline for a single recording
    session. It performs the following steps:
    1. Loads GSR data from the session
    2. Preprocesses the GSR signal to extract tonic and phasic components
    3. Extracts RGB and optionally thermal signals from video frames using palm ROI detection
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
            Used as fallback if timestamp files are not available.
        feature_columns (List[str], optional): List of column names to use as features.
            If None, defaults to ["RGB_B", "RGB_G", "RGB_R", "GSR_Tonic"] or includes thermal features.
        target_column (str, optional): Column name to use as the prediction target.
            Defaults to "GSR_Phasic".
        use_thermal (bool, optional): Whether to include thermal video features.
            Defaults to False.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: If successful, returns a tuple containing:
            - X (np.ndarray): Feature windows with shape (n_windows, window_size, n_features)
              or a tuple of arrays if dual-stream model is needed
            - y (np.ndarray): Target values with shape (n_windows,)
          If any step fails, returns None.

    Notes:
        - The window size is set to 5 seconds of data (5 * gsr_sampling_rate samples)
        - The step size is set to 50% overlap (gsr_sampling_rate // 2)

    Example:
        >>> from pathlib import Path
        >>> session_path = Path("data/recordings/Subject_01_20250101_000000")
        >>> X, y = create_dataset_from_session(session_path, 32, 30, use_thermal=True)
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
    rgb_features = []
    rgb_timestamps = []
    thermal_features = []
    thermal_timestamps = []

    # Try to load timestamp files first
    rgb_timestamps_path = session_path / "rgb_timestamps.csv"
    thermal_timestamps_path = session_path / "thermal_timestamps.csv"

    rgb_timestamps_df = None
    thermal_timestamps_df = None

    if rgb_timestamps_path.exists():
        try:
            rgb_timestamps_df = pd.read_csv(rgb_timestamps_path)
            logging.info(f"Loaded {len(rgb_timestamps_df)} RGB frame timestamps from file")
        except Exception as e:
            logging.warning(f"Failed to load RGB timestamps file: {e}")

    if use_thermal and thermal_timestamps_path.exists():
        try:
            thermal_timestamps_df = pd.read_csv(thermal_timestamps_path)
            logging.info(f"Loaded {len(thermal_timestamps_df)} thermal frame timestamps from file")
        except Exception as e:
            logging.warning(f"Failed to load thermal timestamps file: {e}")

    # Process RGB video
    frame_count = 0
    for success, frame in loader.get_rgb_video_generator():
        if not success:
            continue

        # Use the Multi-ROI approach to extract signals from multiple regions
        roi_signals = process_frame_with_multi_roi(frame)
        if roi_signals:
            # Combine signals from all ROIs into a single feature vector
            # This creates a richer feature set than just using a single ROI
            combined_signal = []
            for roi_name, signal in roi_signals.items():
                combined_signal.extend(signal)

            rgb_features.append(combined_signal)

            # Get timestamp for this frame
            if rgb_timestamps_df is not None and frame_count < len(rgb_timestamps_df):
                # Use the logged timestamp
                timestamp_ns = rgb_timestamps_df.iloc[frame_count]["timestamp"]
                # Convert to pandas timestamp
                frame_timestamp = pd.Timestamp(timestamp_ns, unit='ns')
            else:
                # Fallback to synthetic timestamp based on frame rate
                frame_interval = 1.0 / video_fps
                frame_timestamp = processed_gsr["timestamp"].iloc[0] + pd.to_timedelta(
                    frame_count * frame_interval, unit="s"
                )

            rgb_timestamps.append(frame_timestamp)
            frame_count += 1

    if not rgb_features:
        logging.error("Could not extract any features from the RGB video.")
        return None

    # Process thermal video if requested
    if use_thermal:
        frame_count = 0
        for success, frame in loader.get_thermal_video_generator():
            if not success:
                continue

            # Use the Multi-ROI approach to extract signals from multiple regions
            roi_signals = process_frame_with_multi_roi(frame)
            if roi_signals:
                # Combine signals from all ROIs into a single feature vector
                # This creates a richer feature set than just using a single ROI
                combined_signal = []
                for roi_name, signal in roi_signals.items():
                    combined_signal.extend(signal)

                thermal_features.append(combined_signal)

                # Get timestamp for this frame
                if thermal_timestamps_df is not None and frame_count < len(thermal_timestamps_df):
                    # Use the logged timestamp
                    timestamp_ns = thermal_timestamps_df.iloc[frame_count]["timestamp"]
                    # Convert to pandas timestamp
                    frame_timestamp = pd.Timestamp(timestamp_ns, unit='ns')
                else:
                    # Fallback to synthetic timestamp based on frame rate
                    frame_interval = 1.0 / video_fps
                    frame_timestamp = processed_gsr["timestamp"].iloc[0] + pd.to_timedelta(
                        frame_count * frame_interval, unit="s"
                    )

                thermal_timestamps.append(frame_timestamp)
                frame_count += 1

        if use_thermal and not thermal_features:
            logging.error("Could not extract any features from the thermal video.")
            return None

    # Create DataFrames for the video features
    # With Multi-ROI, we have more features per frame (3 channels x number of ROIs)
    # Define column names based on ROI names and channels
    roi_names = ["index_finger_base", "ring_finger_base", "palm_center"]
    channels = ["B", "G", "R"]

    # Create column names for RGB features
    rgb_columns = []
    for roi in roi_names:
        for channel in channels:
            rgb_columns.append(f"RGB_{roi}_{channel}")

    # Create DataFrame for RGB features
    rgb_df = pd.DataFrame(rgb_features, columns=rgb_columns)
    rgb_df["timestamp"] = rgb_timestamps

    if use_thermal:
        # Create column names for thermal features
        thermal_columns = []
        for roi in roi_names:
            for channel in channels:
                thermal_columns.append(f"THERMAL_{roi}_{channel}")

        # Create DataFrame for thermal features
        thermal_df = pd.DataFrame(thermal_features, columns=thermal_columns)
        thermal_df["timestamp"] = thermal_timestamps

    # 4. Align Signals
    aligned_rgb_df = align_signals(processed_gsr, rgb_df)
    if aligned_rgb_df.empty:
        return None

    if use_thermal:
        aligned_thermal_df = align_signals(processed_gsr, thermal_df)
        if aligned_thermal_df.empty:
            return None

        # Merge the aligned dataframes
        # First, ensure they have the same timestamps
        common_timestamps = aligned_rgb_df["timestamp"].intersection(aligned_thermal_df["timestamp"])
        aligned_rgb_df = aligned_rgb_df[aligned_rgb_df["timestamp"].isin(common_timestamps)]
        aligned_thermal_df = aligned_thermal_df[aligned_thermal_df["timestamp"].isin(common_timestamps)]

        # Now merge them
        merged_df = pd.merge(aligned_rgb_df, aligned_thermal_df, on="timestamp")
        aligned_df = merged_df
    else:
        aligned_df = aligned_rgb_df

    # 5. Create Feature Windows
    if feature_columns is None:
        # Start with all RGB columns
        feature_columns = rgb_columns.copy()

        # Add thermal columns if using thermal data
        if use_thermal:
            feature_columns.extend(thermal_columns)

        # Remove GSR_Tonic from features to avoid giving the model a "cheat sheet"
        # The critique pointed out that including GSR_Tonic as a feature gives the model
        # a modified version of the target, leading to artificially high performance
        # Instead, we'll rely solely on the video-derived features

    # Window size: e.g., 5 seconds of data (5s * 32Hz = 160 samples)
    window_size_samples = 5 * gsr_sampling_rate
    step_size = gsr_sampling_rate // 2  # 50% overlap

    X, y = create_feature_windows(
        aligned_df, feature_columns, target_column, window_size_samples, step_size
    )

    # For dual-stream models, we might need to reshape X to separate RGB and thermal features
    if use_thermal:
        # Check if this is being called from a dual-stream model context
        # This is a safer approach than referencing a potentially undefined 'model' variable
        caller_frame = sys._getframe(1)
        caller_locals = caller_frame.f_locals
        caller_globals = caller_frame.f_globals

        # Check if the caller has a 'self' that is a dual-stream model
        is_dual_stream = False
        if 'self' in caller_locals and hasattr(caller_locals['self'], '__class__'):
            class_name = caller_locals['self'].__class__.__name__
            is_dual_stream = 'DualStream' in class_name or 'dual_stream' in class_name.lower()

        if is_dual_stream:
            # With our new Multi-ROI approach, we have more features
            # We need to separate RGB and thermal features based on their column indices

            # Get the indices of RGB and thermal features in the feature_columns list
            rgb_indices = [i for i, col in enumerate(feature_columns) if col.startswith("RGB_")]
            thermal_indices = [i for i, col in enumerate(feature_columns) if col.startswith("THERMAL_")]

            # Extract RGB and thermal features based on their indices
            X_rgb = X[:, :, rgb_indices]  # All windows, all timesteps, RGB features
            X_thermal = X[:, :, thermal_indices]  # All windows, all timesteps, thermal features

            logging.info(
                f"Dual-stream model detected. Reshaping features: X_rgb shape: {X_rgb.shape}, X_thermal shape: {X_thermal.shape}")
            return (X_rgb, X_thermal), y

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
