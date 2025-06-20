# GSR-RGBT Project Implementation Overview

## Introduction

This document provides a comprehensive overview of the implementation details and improvements made to the GSR-RGBT (Galvanic Skin Response - RGB-Thermal) project. It consolidates information from various implementation-related documents to provide a single reference point for understanding the technical aspects of the project.

## Key Implementation Areas

The GSR-RGBT project implementation spans several key areas, each with significant improvements and optimizations:

1. **Data Acquisition and Synchronization**: Capturing and synchronizing data from multiple sources
2. **Feature Engineering and Processing**: Extracting and processing features from video and sensor data
3. **Machine Learning Models**: Implementing and training models for GSR prediction
4. **GUI and Visualization**: Providing user interfaces and data visualization
5. **Testing and Quality Assurance**: Ensuring code quality and functionality

## 1. Data Acquisition and Synchronization

### 1.1 Enhanced Data Logger

The `DataLogger` class in `src/data_collection/utils/data_logger.py` has been updated to log per-frame timestamps:

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

- Enhanced error handling with specific exception types and partial initialization cleanup:
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

### 1.2 Centralized Timestamp Authority

To address the issue of "optimistic" software-based synchronization, a centralized timestamp authority was implemented:

- Created a new `TimestampThread` class:
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

- Updated the `Application` class to use the centralized timestamps:
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

### 1.3 Hardware Synchronization Research

Comprehensive research was conducted on hardware synchronization methods:

- **External Trigger Signal**: Using an Arduino to generate a common trigger signal for all devices
- **LED Flash Method**: A visible LED flash that can be detected in both RGB and thermal video frames
- **Post-Processing Alignment**: Using visual cues to align data streams during post-processing

## 2. Feature Engineering and Processing

### 2.1 Enhanced ROI Detection with MediaPipe

The original implementation relied on a simple `detect_palm_roi` function that often fell back to a fixed, centered rectangle. This approach was brittle and could lead to extracting signals from irrelevant parts of the frame if the hand moved.

A robust Multi-ROI approach using MediaPipe hand landmarks was implemented:

- **Multiple Regions of Interest**: Instead of a single ROI, signals are now extracted from three physiologically significant regions:
  - Index finger base (high concentration of sweat glands)
  - Ring finger base (strong vascular patterns)
  - Center of the palm (stable reference point)

- **Robust Hand Tracking**: MediaPipe provides accurate hand landmark detection that is resilient to hand movements and lighting changes.

- **Richer Feature Set**: By combining signals from multiple ROIs, a more comprehensive feature vector is created that captures more information about the hand's physiological state.

Implementation details:
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

### 2.2 Refined Feature Engineering

The feature engineering pipeline had two significant issues:
1. It didn't fully utilize the Multi-ROI approach that was already implemented in the codebase
2. It included GSR_Tonic as a feature, which gave the model a "cheat sheet" by providing a modified version of the target

The pipeline was refined to:

- **Full Multi-ROI Integration**: The pipeline now properly extracts and processes signals from multiple ROIs
- **Removed Data Leakage**: GSR_Tonic has been removed from the feature set to ensure the model learns from video features only
- **Improved Dual-Stream Support**: Updated the code that reshapes features for dual-stream models to handle the new feature structure

Before:
```python
if feature_columns is None:
    if use_thermal:
        feature_columns = ["RGB_B", "RGB_G", "RGB_R", "THERMAL_B", "THERMAL_G", "THERMAL_R", "GSR_Tonic"]
    else:
        feature_columns = ["RGB_B", "RGB_G", "RGB_R", "GSR_Tonic"]
```

After:
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

### 2.3 Improved Thermal Data Support

The `create_dataset_from_session` function was updated to support thermal video data:

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

## 3. Machine Learning Models

### 3.1 Model Interface Standardization

A standardized interface for all models was implemented to ensure consistent usage regardless of the underlying framework (PyTorch or TensorFlow/Keras):

```python
class ModelInterface(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for the given input data."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model to the specified path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load a model from the specified path."""
        pass
```

### 3.2 Advanced Model Architectures

Several advanced model architectures were implemented:

- **LSTM**: Long Short-Term Memory networks for sequence modeling
- **Autoencoder**: For unsupervised feature learning and anomaly detection
- **VAE**: Variational Autoencoders for generative modeling
- **CNN**: Convolutional Neural Networks for feature extraction
- **CNN-LSTM**: Hybrid model combining CNN and LSTM layers
- **Transformer**: Self-attention based models for sequence data
- **ResNet**: Residual Networks with skip connections for deep architectures

### 3.3 Flexible Configuration System

A flexible configuration system was implemented to allow easy experimentation with different model architectures and hyperparameters:

```python
class ModelConfig:
    def __init__(self, model_type=None, config_path=None):
        """
        Initialize a model configuration.
        
        Args:
            model_type (str, optional): Type of model to create a default config for.
            config_path (str or Path, optional): Path to a YAML config file.
        """
        if config_path is not None:
            self.load_from_file(config_path)
        elif model_type is not None:
            self.load_default_config(model_type)
        else:
            self.config = {}
            
    def load_default_config(self, model_type):
        """Load the default configuration for the specified model type."""
        model_type = model_type.lower()
        if model_type == "lstm":
            self.config = self._get_default_lstm_config()
        elif model_type == "cnn":
            self.config = self._get_default_cnn_config()
        # ... other model types ...
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

## 4. GUI and Visualization

### 4.1 Real-time Visualization Integration

The GUI was enhanced to include real-time visualization of GSR data, providing immediate feedback during data collection sessions:

- Added a split-panel interface with video feeds at the top and GSR visualization at the bottom
- Implemented group boxes for better visual organization of different data streams
- Added real-time GSR signal plotting using Matplotlib integration with PyQt5
- Created methods to connect GSR data signals to the visualization components

### 4.2 User Interface Organization

The main window layout was restructured to improve usability:

- Used QSplitter to allow users to adjust the relative sizes of video and GSR visualization panels
- Organized video feeds into labeled group boxes
- Improved the control panel layout for better user interaction
- Added reset functionality for visualization components

### 4.3 Visualization Components

A new real-time visualization module was implemented to display GSR data as it's being collected:

- Created `GSRPlotCanvas` class that extends `FigureCanvasQTAgg` for seamless integration with PyQt5
- Implemented dynamic plot updating with automatic axis scaling
- Added buffer management to handle continuous data streams efficiently
- Implemented time-based x-axis to show temporal relationships

## 5. Testing and Quality Assurance

### 5.1 Comprehensive Testing Framework

A comprehensive testing framework was implemented to ensure code quality and functionality:

- **Unit Tests**: Tests for individual components and functions
- **Regression Tests**: Tests to prevent reintroduction of fixed bugs
- **Smoke Tests**: Basic tests to verify that the system works as expected
- **Integration Tests**: Tests for the interaction between different components

### 5.2 Improved Error Handling

Error handling was improved throughout the codebase to provide more specific information about errors and ensure proper resource cleanup:

- Distinguishes between different types of errors (e.g., connection failures, data processing errors)
- Adds specific exception types and error messages
- Implements proper resource cleanup in case of errors
- Adds logging for debugging and troubleshooting

### 5.3 Code Quality Improvements

Several code quality improvements were made:

- Added type hints to all functions and methods for better code readability and IDE support
- Refactored duplicate code in preprocessing and feature engineering modules
- Implemented consistent naming conventions across the codebase
- Added comprehensive docstrings to all classes and functions

## Future Considerations

While the improvements described in this document have significantly enhanced the GSR-RGBT project, there are additional enhancements that could further improve the project:

1. **Experiment Tracking Integration**: Integrating a dedicated experiment tracking tool like MLflow or Weights & Biases would provide a more comprehensive solution for tracking and comparing experiments.

2. **Further Modularization of Feature Engineering**: The feature extraction and processing could be further modularized to make it easier to add new feature types or processing methods.

3. **Automated Testing**: Adding more automated tests for the new functionality would help ensure that the changes continue to work correctly as the codebase evolves.

4. **Experimental Protocol Control**: Adding a Protocol Control panel to the GUI with task name display and countdown timer to better guide the data collection process.

5. **Event Marking**: Implementing a `log_event` function in the DataLogger to mark experimental events, which would be useful for data analysis.

6. **Advanced Fusion Techniques**: Exploring more sophisticated methods for fusing RGB and thermal features to better leverage the complementary information.

7. **Thermal-Specific Features**: Extracting temperature-based features from thermal data instead of treating it like RGB data, which would better utilize the unique information provided by thermal imaging.

8. **Hardware Synchronization**: Implementing a hardware-based synchronization system (e.g., Arduino-based LED flash) for even more precise temporal alignment of data streams.

## Conclusion

The improvements described in this document have significantly enhanced the GSR-RGBT project in several key areas:

1. **Data Quality**: By implementing proper timestamp logging, a centralized timestamp authority, and the Multi-ROI approach, we've improved the quality and reliability of the data collected and processed by the system.

2. **Feature Richness**: The Multi-ROI approach and improved thermal data support have enhanced the feature set available for machine learning models, potentially leading to better prediction accuracy.

3. **Model Training**: Removing data leakage and improving the experiment comparison functionality have made the model training process more robust and the results more reliable.

4. **Code Quality**: Enhanced error handling, better resource management, and more specific exception handling have improved the overall robustness of the codebase.

5. **Documentation**: Comprehensive documentation of the improvements and future considerations provides a clear roadmap for ongoing development.

These enhancements build on the solid foundation of the original implementation and address the most critical issues identified through code reviews and testing. The project is now better positioned to achieve its research goals of contactless GSR prediction from RGB and thermal video streams.