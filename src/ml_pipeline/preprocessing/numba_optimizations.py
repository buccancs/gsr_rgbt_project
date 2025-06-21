# src/ml_pipeline/preprocessing/numba_optimizations.py

"""
Numba optimizations for performance-critical functions in the processing pipeline.

This module contains Numba implementations of performance-critical functions
from the feature_engineering and preprocessing modules.
"""

import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def nb_create_feature_windows(
    features,
    targets,
    feature_cols_idx,
    window_size,
    step
):
    """
    Numba implementation of create_feature_windows function.

    Creates overlapping windows from time-series data for sequence modeling.

    Args:
        features (np.ndarray): 2D array of features
        targets (np.ndarray): 1D array of target values
        feature_cols_idx (list): List of column indices to use as features
        window_size (int): Number of time steps in each window
        step (int): Number of time steps to move forward for the next window

    Returns:
        tuple: (X, y) where X is a 3D array of feature windows and y is a 1D array of targets
    """
    num_rows = features.shape[0]
    num_features = len(feature_cols_idx)
    
    # Handle edge case where window_size > num_rows
    if window_size > num_rows:
        return np.zeros((0, window_size, num_features), dtype=np.float64), np.zeros(0, dtype=np.float64)
    
    num_windows = (num_rows - window_size) // step + 1

    # Pre-allocate arrays for better performance
    X = np.zeros((num_windows, window_size, num_features), dtype=np.float64)
    y = np.zeros(num_windows, dtype=np.float64)

    # Fill the arrays
    for window_idx in prange(num_windows):
        i = window_idx * step
        # Extract features for this window
        for j in range(window_size):
            for k in range(num_features):
                feature_idx = feature_cols_idx[k]
                X[window_idx, j, k] = features[i + j, feature_idx]

        # The target is the value at the end of the window
        y[window_idx] = targets[i + window_size - 1]

    return X, y

@jit(nopython=True)
def nb_extract_roi_signal(frame, roi):
    """
    Numba implementation of extract_roi_signal function.

    Extracts the mean pixel value from a specified Region of Interest (ROI).

    Args:
        frame (np.ndarray): The video frame from which to extract the signal
        roi (tuple): The bounding box (x, y, w, h) of the ROI

    Returns:
        np.ndarray: An array containing the mean value for each channel (e.g., [B, G, R])
    """
    x, y, w, h = roi
    
    # Initialize mean values array
    mean_values = np.zeros(3, dtype=np.float64)
    pixel_count = w * h
    
    # Calculate sum for each channel
    for c in range(3):  # 3 channels (B, G, R)
        for i in range(y, y + h):
            for j in range(x, x + w):
                mean_values[c] += frame[i, j, c]
    
    # Calculate mean
    for c in range(3):
        mean_values[c] /= pixel_count
    
    return mean_values

def align_signals_py(
    gsr_data,
    video_data,
    gsr_timestamps,
    video_timestamps
):
    """
    Pure Python implementation for aligning video-derived signals to GSR signal timestamps.

    This function is used as a fallback when the Numba version fails.
    """
    # Handle empty input arrays
    if gsr_data.shape[0] == 0 or video_data.shape[0] == 0:
        gsr_features = gsr_data.shape[1] if gsr_data.shape[0] > 0 else 0
        video_features = video_data.shape[1] if video_data.shape[0] > 0 else 0
        total_features = gsr_features + video_features
        return np.zeros((0, total_features), dtype=np.float64)

    # Handle empty video timestamps
    if len(video_timestamps) == 0:
        gsr_samples = gsr_data.shape[0]
        gsr_features = gsr_data.shape[1]
        video_features = video_data.shape[1]
        total_features = gsr_features + video_features

        result = np.zeros((gsr_samples, total_features), dtype=np.float64)

        # Copy GSR data to result array
        for i in range(gsr_samples):
            for j in range(gsr_features):
                result[i, j] = gsr_data[i, j]

        return result

    # Normal case - both inputs have data
    gsr_samples = gsr_data.shape[0]
    gsr_features = gsr_data.shape[1]
    video_features = video_data.shape[1]
    total_features = gsr_features + video_features
    video_samples = video_data.shape[0]

    # Pre-allocate result array
    aligned_data = np.zeros((gsr_samples, total_features), dtype=np.float64)

    # Copy GSR data to result array
    for i in range(gsr_samples):
        for j in range(gsr_features):
            aligned_data[i, j] = gsr_data[i, j]

    # Get timestamp bounds
    min_video_timestamp = video_timestamps[0]
    max_video_timestamp = video_timestamps[video_samples - 1]

    # For each GSR timestamp, interpolate video features
    for i in range(gsr_samples):
        gsr_t = gsr_timestamps[i]

        # Handle case where gsr_t is outside the range of video_timestamps
        if gsr_t < min_video_timestamp:
            # Use the first video data point for all timestamps before the first video timestamp
            for j in range(video_features):
                aligned_data[i, gsr_features + j] = video_data[0, j]
            continue

        if gsr_t > max_video_timestamp:
            # Use the last video data point for all timestamps after the last video timestamp
            for j in range(video_features):
                aligned_data[i, gsr_features + j] = video_data[video_samples - 1, j]
            continue

        # Find the two closest video timestamps
        video_idx_low = 0
        video_idx_high = 0

        # Find the first video timestamp that is >= gsr_t
        for j in range(len(video_timestamps)):
            if video_timestamps[j] >= gsr_t:
                video_idx_high = j
                video_idx_low = max(0, j - 1)
                break

        # If we didn't find a higher timestamp, use the last two
        if video_idx_high == 0 and gsr_t > video_timestamps[0]:
            video_idx_high = len(video_timestamps) - 1
            video_idx_low = max(0, video_idx_high - 1)

        # Get the timestamps
        video_t_low = video_timestamps[video_idx_low]
        video_t_high = video_timestamps[video_idx_high]

        # Calculate interpolation ratio
        if video_t_high == video_t_low:
            t_ratio = 0.0
        else:
            t_ratio = float(gsr_t - video_t_low) / float(video_t_high - video_t_low)

        # Interpolate video features
        for j in range(video_features):
            aligned_data[i, gsr_features + j] = (
                video_data[video_idx_low, j] * (1.0 - t_ratio) +
                video_data[video_idx_high, j] * t_ratio
            )

    return aligned_data

@jit(nopython=True)
def nb_align_signals_core(
    gsr_data,
    video_data,
    gsr_timestamps,
    video_timestamps,
    aligned_data
):
    """
    Core Numba implementation for aligning video-derived signals to GSR signal timestamps.
    
    This function is called by nb_align_signals after initial checks and array allocation.
    """
    gsr_samples = gsr_data.shape[0]
    gsr_features = gsr_data.shape[1]
    video_features = video_data.shape[1]
    video_samples = video_data.shape[0]
    
    # Copy GSR data to result array
    for i in range(gsr_samples):
        for j in range(gsr_features):
            aligned_data[i, j] = gsr_data[i, j]
    
    # Get timestamp bounds
    min_video_timestamp = video_timestamps[0]
    max_video_timestamp = video_timestamps[video_samples - 1]
    
    # For each GSR timestamp, interpolate video features
    for i in range(gsr_samples):
        gsr_t = gsr_timestamps[i]
        
        # Handle case where gsr_t is outside the range of video_timestamps
        if gsr_t < min_video_timestamp:
            # Use the first video data point for all timestamps before the first video timestamp
            for j in range(video_features):
                aligned_data[i, gsr_features + j] = video_data[0, j]
            continue
        
        if gsr_t > max_video_timestamp:
            # Use the last video data point for all timestamps after the last video timestamp
            for j in range(video_features):
                aligned_data[i, gsr_features + j] = video_data[video_samples - 1, j]
            continue
        
        # Find the two closest video timestamps
        video_idx_low = 0
        video_idx_high = 0
        
        # Find the first video timestamp that is >= gsr_t
        for j in range(len(video_timestamps)):
            if video_timestamps[j] >= gsr_t:
                video_idx_high = j
                video_idx_low = max(0, j - 1)
                break
        
        # If we didn't find a higher timestamp, use the last two
        if video_idx_high == 0 and gsr_t > video_timestamps[0]:
            video_idx_high = len(video_timestamps) - 1
            video_idx_low = max(0, video_idx_high - 1)
        
        # Get the timestamps
        video_t_low = video_timestamps[video_idx_low]
        video_t_high = video_timestamps[video_idx_high]
        
        # Calculate interpolation ratio
        if video_t_high == video_t_low:
            t_ratio = 0.0
        else:
            t_ratio = float(gsr_t - video_t_low) / float(video_t_high - video_t_low)
        
        # Interpolate video features
        for j in range(video_features):
            aligned_data[i, gsr_features + j] = (
                video_data[video_idx_low, j] * (1.0 - t_ratio) +
                video_data[video_idx_high, j] * t_ratio
            )
    
    return aligned_data

def nb_align_signals(
    gsr_data,
    video_data,
    gsr_timestamps,
    video_timestamps
):
    """
    Numba implementation for aligning video-derived signals to GSR signal timestamps.

    This is a simplified version that uses linear interpolation.

    Args:
        gsr_data (np.ndarray): 2D array of GSR data (samples x features)
        video_data (np.ndarray): 2D array of video features (samples x features)
        gsr_timestamps (np.ndarray): 1D array of GSR timestamps (in nanoseconds)
        video_timestamps (np.ndarray): 1D array of video timestamps (in nanoseconds)

    Returns:
        np.ndarray: 2D array of aligned data (GSR timestamps x combined features)
    """
    # Special case: empty inputs
    if gsr_data.shape[0] == 0 or video_data.shape[0] == 0:
        return align_signals_py(gsr_data, video_data, gsr_timestamps, video_timestamps)

    # Special case: empty video timestamps
    if len(video_timestamps) == 0:
        return align_signals_py(gsr_data, video_data, gsr_timestamps, video_timestamps)

    # Normal case - both inputs have data
    gsr_samples = gsr_data.shape[0]
    gsr_features = gsr_data.shape[1]
    video_features = video_data.shape[1]
    total_features = gsr_features + video_features

    # Pre-allocate result array
    aligned_data = np.zeros((gsr_samples, total_features), dtype=np.float64)
    
    # Call the Numba-optimized core function
    return nb_align_signals_core(gsr_data, video_data, gsr_timestamps, video_timestamps, aligned_data)