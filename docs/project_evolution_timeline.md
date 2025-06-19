# GSR-RGBT Project Evolution Timeline

## Introduction

This document provides a comprehensive timeline of the GSR-RGBT project's evolution, detailing the step-by-step iterations, improvements, and modifications made throughout its development. It captures both the user requests and the corresponding changes implemented in response, offering a clear chronological view of how the project has progressed and matured over time.

## Initial Project State

The GSR-RGBT (Galvanic Skin Response - RGB-Thermal) project began as a research initiative aimed at predicting Galvanic Skin Response from synchronized RGB and thermal video streams. The initial implementation included:

- A data acquisition application with a PyQt5-based GUI
- Capture threads for RGB video, thermal video, and GSR data
- Basic preprocessing and feature extraction from video frames
- Machine learning models for GSR prediction
- A simple evaluation pipeline

While functional, the initial implementation had several limitations and areas for improvement:

- Inconsistent documentation with mismatched repository names
- Basic ROI detection that relied on fixed regions
- Suboptimal data synchronization between different streams
- Limited utilization of thermal data
- Potential data leakage in the feature engineering pipeline

## Iteration 1: Documentation Consistency (May 2023)

### User Request
The first improvement request focused on ensuring consistency across all documentation files, particularly regarding repository names and directory commands.

### Issue Identified
Inconsistencies were found in how the repository was referenced across different documentation files:
- Some files used `gsr-rgbt-project` (with hyphens)
- Others used `gsr_rgbt_project` (with underscores)

This inconsistency could lead to confusion for users trying to follow the setup instructions.

### Changes Implemented
1. **README.md**: Updated the `cd` command to use `gsr_rgbt_project` consistently:
   ```diff
   git clone https://github.com/your-username/gsr-rgbt-project.git
   - cd gsr-rgbt-project
   + cd gsr_rgbt_project
   ```

2. **equipment_setup.md**: Made similar updates to ensure consistency:
   ```diff
   git clone https://github.com/your-username/gsr-rgbt-project.git
   - cd gsr-rgbt-project
   + cd gsr_rgbt_project
   ```

3. **appendix.tex**: Verified that the repository URL format was consistent with other documentation files.

### Impact
These changes ensured that all documentation files consistently referred to the repository with the same naming convention, making it easier for users to follow the setup instructions without encountering errors due to mismatched directory names.

## Iteration 2: Major Code Improvements (June 2023)

### User Request
Following a comprehensive code review, several critical issues were identified that needed to be addressed to improve the robustness and effectiveness of the project.

### Issues Identified
1. **ROI Detection**: The original implementation used a simple, fixed-region approach that was brittle and could fail if the hand moved.
2. **Data Synchronization**: The system relied on "optimistic" software-based synchronization, with each component generating its own timestamps.
3. **Feature Engineering**: The pipeline didn't fully utilize the Multi-ROI approach and included GSR_Tonic as a feature, which could lead to data leakage.

### Changes Implemented

#### 1. Enhanced ROI Detection with MediaPipe

**Before**: The original implementation relied on a simple `detect_palm_roi` function that often fell back to a fixed, centered rectangle:

```python
def detect_palm_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    # Simple placeholder method
    h, w, _ = frame.shape
    roi_width = int(w * 0.4)
    roi_height = int(h * 0.6)
    roi_x = (w - roi_width) // 2
    roi_y = (h - roi_height) // 2
    return (roi_x, roi_y, roi_width, roi_height)
```

**After**: Implemented a robust Multi-ROI approach using MediaPipe hand landmarks:

```python
def process_frame_with_multi_roi(frame: np.ndarray) -> Dict[str, np.ndarray]:
    # Detect hand landmarks
    hand_landmarks_list = detect_hand_landmarks(frame)

    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        logging.warning("No hands detected in the frame. Cannot extract multi-ROI signals.")
        return {}

    # Define multiple ROIs based on hand landmarks
    rois = define_multi_roi(frame, hand_landmarks_list[0])

    # Extract signals from each ROI
    signals = extract_multi_roi_signals(frame, rois)

    return signals
```

The feature engineering pipeline was updated to use this new approach:

```python
# Use the Multi-ROI approach to extract signals from multiple regions
roi_signals = process_frame_with_multi_roi(frame)
if roi_signals:
    # Combine signals from all ROIs into a single feature vector
    combined_signal = []
    for roi_name, signal in roi_signals.items():
        combined_signal.extend(signal)

    rgb_features.append(combined_signal)
```

#### 2. Improved Data Synchronization

**Before**: Each component generated its own timestamps, leading to potential drift:

```python
# In VideoCaptureThread
current_capture_time = time.time()
self.frame_captured.emit(frame, current_capture_time)

# In DataLogger
def log_rgb_frame(self, frame):
    self.rgb_writer.write(frame)
```

**After**: Implemented a centralized timestamp authority:

1. Created a new `TimestampThread` class:

```python
class TimestampThread(QThread):
    # Signal emitted with each new timestamp
    timestamp_generated = pyqtSignal(int)  # Timestamp in nanoseconds

    def __init__(self, frequency=200):
        super().__init__()
        self.frequency = frequency
        self.interval = 1.0 / frequency
        self.running = False
        self.setPriority(QThread.HighPriority)

    def run(self):
        self.running = True
        while self.running:
            # Get current high-resolution timestamp
            current_time = time.perf_counter_ns()
            # Emit the timestamp
            self.timestamp_generated.emit(current_time)
            # Sleep for the interval
            time.sleep(self.interval)
```

2. Updated the `Application` class to use the centralized timestamps:

```python
# Initialize the timestamp thread
self.timestamp_thread = TimestampThread(frequency=200)
self.latest_timestamp = None
self.timestamp_thread.timestamp_generated.connect(self.update_latest_timestamp)

# In start_recording method
self.timestamp_thread.start()
self.rgb_capture.frame_captured.connect(
    lambda frame, _: self.data_logger.log_rgb_frame(frame, self.latest_timestamp)
)
```

3. Enhanced the `DataLogger` to log timestamps:

```python
def log_rgb_frame(self, frame, timestamp=None, frame_number=None):
    if self.rgb_writer and self.is_logging:
        self.rgb_writer.write(frame)

        # Log the timestamp if provided
        if timestamp is not None and self.rgb_timestamps_writer:
            frame_num = frame_number if frame_number is not None else self.rgb_writer.get(cv2.CAP_PROP_POS_FRAMES)
            self.rgb_timestamps_writer.writerow([frame_num, timestamp])
```

#### 3. Refined Feature Engineering

**Before**: The feature engineering pipeline had limited utilization of thermal data and potential data leakage:

```python
if feature_columns is None:
    if use_thermal:
        feature_columns = ["RGB_B", "RGB_G", "RGB_R", "THERMAL_B", "THERMAL_G", "THERMAL_R", "GSR_Tonic"]
    else:
        feature_columns = ["RGB_B", "RGB_G", "RGB_R", "GSR_Tonic"]
```

**After**: Updated to better utilize the Multi-ROI approach and avoid data leakage:

```python
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
```

Also updated the dual-stream feature reshaping:

```python
if is_dual_stream:
    # With our new Multi-ROI approach, we have more features
    # We need to separate RGB and thermal features based on their column indices

    # Get the indices of RGB and thermal features in the feature_columns list
    rgb_indices = [i for i, col in enumerate(feature_columns) if col.startswith("RGB_")]
    thermal_indices = [i for i, col in enumerate(feature_columns) if col.startswith("THERMAL_")]

    # Extract RGB and thermal features based on their indices
    X_rgb = X[:, :, rgb_indices]  # All windows, all timesteps, RGB features
    X_thermal = X[:, :, thermal_indices]  # All windows, all timesteps, thermal features

    logging.info(f"Dual-stream model detected. Reshaping features: X_rgb shape: {X_rgb.shape}, X_thermal shape: {X_thermal.shape}")
    return (X_rgb, X_thermal), y
```

### Impact
These improvements significantly enhanced the robustness and effectiveness of the GSR-RGBT project:

1. **Enhanced ROI Detection**: The Multi-ROI approach provides more stable and physiologically relevant features, even when the hand moves.
2. **Improved Data Synchronization**: The centralized timestamp authority ensures precise temporal alignment between different data streams.
3. **Refined Feature Engineering**: The updated pipeline better utilizes thermal data and avoids data leakage, leading to more reliable model performance.

## Iteration 3: Documentation of Improvements (July 2023)

### User Request
After implementing the major code improvements, there was a need to document these changes comprehensively to help users understand the enhancements and their impact.

### Changes Implemented
Created a detailed documentation file (`implementation_improvements.md`) that explained:

1. The issues identified in the original implementation
2. The solutions implemented to address these issues
3. The technical details of each improvement
4. Future considerations for further enhancements

### Impact
This documentation provides a clear explanation of the improvements made to the project, helping users understand the changes and their benefits. It also serves as a reference for future development, highlighting areas that could be further enhanced.

## Iteration 4: Project Evolution Timeline (August 2023)

### User Request
The final request was to consolidate all the improvement descriptions into a timeline-like document, showing the step-by-step iterations of the project.

### Changes Implemented
Created this comprehensive timeline document (`project_evolution_timeline.md`) that:

1. Provides a chronological view of the project's evolution
2. Details each iteration of improvements
3. Includes code snippets to illustrate the changes
4. Explains the impact of each improvement

### Impact
This timeline document offers a complete historical perspective on the project's development, making it easier for new contributors to understand how the project has evolved and the rationale behind key design decisions.

## Conclusion

The GSR-RGBT project has undergone significant evolution from its initial implementation to its current state. Through multiple iterations of improvements, the project has addressed critical issues in ROI detection, data synchronization, and feature engineering, resulting in a more robust and effective system for contactless GSR prediction.

Key milestones in this evolution include:
1. Ensuring documentation consistency for better user experience
2. Implementing a Multi-ROI approach using MediaPipe for more reliable feature extraction
3. Creating a centralized timestamp authority for precise data synchronization
4. Refining the feature engineering pipeline to better utilize thermal data and avoid data leakage
5. Documenting all improvements for future reference and development

This evolutionary process demonstrates the project's commitment to continuous improvement and technical excellence, setting a strong foundation for future research in contactless physiological monitoring.
