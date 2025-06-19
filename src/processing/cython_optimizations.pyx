# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Cython optimizations for performance-critical functions in the processing pipeline.

This module contains Cython implementations of performance-critical functions
from the feature_engineering and preprocessing modules.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Define C types for numpy arrays
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE_int_t


def cy_create_feature_windows(
    np.ndarray[DTYPE_t, ndim=2] features,
    np.ndarray[DTYPE_t, ndim=1] targets,
    list feature_cols_idx,
    int window_size,
    int step
):
    """
    Cython implementation of create_feature_windows function.

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
    cdef int num_rows = features.shape[0]
    cdef int num_features = len(feature_cols_idx)
    cdef int num_windows = (num_rows - window_size) // step + 1

    # Pre-allocate arrays for better performance
    cdef np.ndarray[DTYPE_t, ndim=3] X = np.zeros((num_windows, window_size, num_features), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(num_windows, dtype=np.float64)

    cdef int i, j, k, window_idx
    cdef int feature_idx

    # Fill the arrays
    window_idx = 0
    for i in range(0, num_rows - window_size, step):
        # Extract features for this window
        for j in range(window_size):
            for k in range(num_features):
                feature_idx = feature_cols_idx[k]
                X[window_idx, j, k] = features[i + j, feature_idx]

        # The target is the value at the end of the window
        y[window_idx] = targets[i + window_size - 1]

        window_idx += 1

    return X, y


def cy_extract_roi_signal(
    np.ndarray[np.uint8_t, ndim=3] frame,
    tuple roi
):
    """
    Cython implementation of extract_roi_signal function.

    Extracts the mean pixel value from a specified Region of Interest (ROI).

    Args:
        frame (np.ndarray): The video frame from which to extract the signal
        roi (tuple): The bounding box (x, y, w, h) of the ROI

    Returns:
        np.ndarray: An array containing the mean value for each channel (e.g., [B, G, R])
    """
    cdef int x = roi[0]
    cdef int y = roi[1]
    cdef int w = roi[2]
    cdef int h = roi[3]

    cdef int i, j, c
    cdef double[:] mean_values = np.zeros(3, dtype=np.float64)
    cdef int pixel_count = w * h

    # Calculate sum for each channel
    for c in range(3):  # 3 channels (B, G, R)
        for i in range(y, y + h):
            for j in range(x, x + w):
                mean_values[c] += frame[i, j, c]

    # Calculate mean
    for c in range(3):
        mean_values[c] /= pixel_count

    return np.array([mean_values[0], mean_values[1], mean_values[2]])


def cy_align_signals(
    np.ndarray[DTYPE_t, ndim=2] gsr_data,
    np.ndarray[DTYPE_t, ndim=2] video_data,
    np.ndarray[np.int64_t, ndim=1] gsr_timestamps,
    np.ndarray[np.int64_t, ndim=1] video_timestamps
):
    """
    Cython implementation for aligning video-derived signals to GSR signal timestamps.

    This is a simplified version that uses linear interpolation.

    Args:
        gsr_data (np.ndarray): 2D array of GSR data (samples x features)
        video_data (np.ndarray): 2D array of video features (samples x features)
        gsr_timestamps (np.ndarray): 1D array of GSR timestamps (in nanoseconds)
        video_timestamps (np.ndarray): 1D array of video timestamps (in nanoseconds)

    Returns:
        np.ndarray: 2D array of aligned data (GSR timestamps x combined features)
    """
    # Check for empty input arrays
    if gsr_data.shape[0] == 0 or video_data.shape[0] == 0:
        # Return an empty array with the correct shape
        cdef int gsr_features = gsr_data.shape[1] if gsr_data.shape[0] > 0 else 0
        cdef int video_features = video_data.shape[1] if video_data.shape[0] > 0 else 0
        cdef int total_features = gsr_features + video_features
        return np.zeros((0, total_features), dtype=np.float64)

    # Check if video_timestamps is empty
    if len(video_timestamps) == 0:
        # Return GSR data only, with zeros for video features
        cdef int gsr_samples = gsr_data.shape[0]
        cdef int gsr_features = gsr_data.shape[1]
        cdef int video_features = video_data.shape[1]
        cdef int total_features = gsr_features + video_features
        cdef np.ndarray[DTYPE_t, ndim=2] aligned_data = np.zeros((gsr_samples, total_features), dtype=np.float64)

        # Copy GSR data to result array
        for i in range(gsr_samples):
            for j in range(gsr_features):
                aligned_data[i, j] = gsr_data[i, j]

        return aligned_data

    cdef int gsr_samples = gsr_data.shape[0]
    cdef int gsr_features = gsr_data.shape[1]
    cdef int video_features = video_data.shape[1]
    cdef int total_features = gsr_features + video_features
    cdef int video_samples = video_data.shape[0]

    # Pre-allocate result array
    cdef np.ndarray[DTYPE_t, ndim=2] aligned_data = np.zeros((gsr_samples, total_features), dtype=np.float64)

    # Copy GSR data to result array
    for i in range(gsr_samples):
        for j in range(gsr_features):
            aligned_data[i, j] = gsr_data[i, j]

    cdef int i, j, k
    cdef double t, t_ratio
    cdef np.int64_t gsr_t, video_t_low, video_t_high
    cdef int video_idx_low, video_idx_high
    cdef np.int64_t min_video_timestamp = video_timestamps[0]
    cdef np.int64_t max_video_timestamp = video_timestamps[video_samples - 1]

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
