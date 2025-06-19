# Implementation Notes: Data Synchronization and Model Comparison Improvements

## Overview

This document outlines the changes made to improve data synchronization, feature engineering, and model comparison in the GSR-RGBT project. These changes address several key issues identified in the codebase:

1. Lack of proper timestamp logging for video frames
2. Synthetic timestamp generation in feature engineering
3. Limited support for dual-stream models with thermal data
4. Basic model_run_id extraction in experiment comparison
5. Broad exception handling in DataLogger
6. Missing dependencies in requirements.txt

## Changes Made

### 1. Enhanced DataLogger for Frame Timestamp Logging

The `DataLogger` class in `src/utils/data_logger.py` has been updated to log per-frame timestamps:

- Added new fields for timestamp writers and files:
  ```python
  self.rgb_timestamps_writer = None
  self.rgb_timestamps_file = None
  self.thermal_timestamps_writer = None
  self.thermal_timestamps_file = None
  ```

- Modified `log_rgb_frame` and `log_thermal_frame` to log timestamps:
  ```python
  def log_rgb_frame(self, frame, timestamp=None, frame_number=None):
      # ... existing code ...
      if timestamp is not None and self.rgb_timestamps_writer:
          frame_num = frame_number if frame_number is not None else self.rgb_writer.get(cv2.CAP_PROP_POS_FRAMES)
          self.rgb_timestamps_writer.writerow([frame_num, timestamp])
  ```

- Enhanced error handling in `start_logging` with specific exception types and partial initialization cleanup:
  ```python
  try:
      # ... initialization code ...
  except IOError as e:
      logging.error(f"I/O error initializing log files: {e}")
      self._cleanup_partial_initialization(video_writers_initialized, 
                                         gsr_writer_initialized,
                                         timestamp_writers_initialized)
  ```

- Added a helper method `_cleanup_partial_initialization` to ensure resources are properly released if initialization fails.

### 2. Improved Feature Engineering with Actual Timestamps

The `create_dataset_from_session` function in `src/processing/feature_engineering.py` has been updated to:

- Check for and use actual logged timestamps from CSV files:
  ```python
  rgb_timestamps_path = session_path / "rgb_timestamps.csv"
  thermal_timestamps_path = session_path / "thermal_timestamps.csv"
  
  if rgb_timestamps_path.exists():
      try:
          rgb_timestamps_df = pd.read_csv(rgb_timestamps_path)
          logging.info(f"Loaded {len(rgb_timestamps_df)} RGB frame timestamps from file")
      except Exception as e:
          logging.warning(f"Failed to load RGB timestamps file: {e}")
  ```

- Fall back to synthetic timestamps only if actual timestamps are not available:
  ```python
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
  ```

### 3. Added Support for Thermal Video in Dual-Stream Models

The `create_dataset_from_session` function now supports thermal video data:

- Added a `use_thermal` parameter to enable thermal data processing:
  ```python
  def create_dataset_from_session(
      session_path: Path, gsr_sampling_rate: int, video_fps: int,
      feature_columns: List[str] = None, target_column: str = "GSR_Phasic",
      use_thermal: bool = False
  )
  ```

- Added code to process thermal video frames and extract features:
  ```python
  if use_thermal:
      frame_count = 0
      for success, frame in loader.get_thermal_video_generator():
          # ... process thermal frames ...
  ```

- Added support for merging RGB and thermal features:
  ```python
  if use_thermal:
      thermal_df = pd.DataFrame(thermal_features, columns=["THERMAL_B", "THERMAL_G", "THERMAL_R"])
      thermal_df["timestamp"] = thermal_timestamps
      
      # ... align and merge dataframes ...
  ```

- Added logic to reshape features for dual-stream models:
  ```python
  if use_thermal:
      # Check if this is being called from a dual-stream model context
      # ... check caller frame ...
      
      if is_dual_stream:
          # Assuming the first 3 features are RGB and the next 3 are thermal
          X_rgb = X[:, :, :3]  # All windows, all timesteps, first 3 features (RGB)
          X_thermal = X[:, :, 3:6]  # All windows, all timesteps, next 3 features (thermal)
          return (X_rgb, X_thermal), y
  ```

### 4. Improved Model Run ID Extraction in Experiment Comparison

The `load_all_cv_results` function in `src/scripts/compare_experiments.py` has been updated to:

- Better handle structured experiment directories:
  ```python
  if "experiments" in cv_file.parts:
      # Find the index of "experiments" in the path
      exp_idx = cv_file.parts.index("experiments")
      if exp_idx + 1 < len(cv_file.parts):  # Make sure there's a directory after "experiments"
          experiment_set_name = cv_file.parts[exp_idx + 1]
          model_name_from_file = cv_file.stem.replace("cross_validation_results_", "")
          model_run_id = f"{experiment_set_name}_{model_name_from_file}"
  ```

### 5. Updated Dependencies

The `requirements.txt` file has been updated to include:

- `PyQt5`: Required for the GUI components
- `seaborn`: Used for visualization in compare_experiments.py

## Impact of Changes

These changes significantly improve the project in several ways:

1. **More Accurate Data Synchronization**: By logging actual frame timestamps, the system can now accurately align video frames with GSR data, eliminating the reliance on synthetic timestamps based on assumed frame rates.

2. **Better Support for Dual-Stream Models**: The feature engineering pipeline now properly handles thermal video data, enabling the use of dual-stream models that can leverage both RGB and thermal features.

3. **More Robust Experiment Comparison**: The improved model_run_id extraction makes it easier to organize and compare experiments, especially when using structured output directories.

4. **Enhanced Error Handling**: The DataLogger now has more specific exception handling and proper resource cleanup, reducing the risk of silent failures or resource leaks.

5. **Complete Dependencies**: The requirements.txt file now includes all necessary dependencies, making it easier to set up the development environment.

## Future Considerations

While these changes address the immediate issues, there are some additional improvements that could be considered in the future:

1. **Experiment Tracking Integration**: Integrating a dedicated experiment tracking tool like MLflow or Weights & Biases would provide a more comprehensive solution for tracking and comparing experiments.

2. **Further Modularization of Feature Engineering**: The feature extraction and processing could be further modularized to make it easier to add new feature types or processing methods.

3. **Automated Testing**: Adding automated tests for the new functionality would help ensure that the changes continue to work correctly as the codebase evolves.