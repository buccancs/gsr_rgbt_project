# FactorizePhys Repository Overview

## Purpose and Overview

FactorizePhys is a C++ library designed for synchronized capture of RGB video, thermal video, and physiological data. It is an extension of the RGBTPhys_CPP library, with specialized functionality for factorizing (separating) physiological signals from video data.

### Relationship with RGBTPhys_CPP

FactorizePhys builds upon the foundation of RGBTPhys_CPP, extending it with the following enhancements:

1. **Advanced Factorization Algorithms**: Implements specialized algorithms for separating physiological signals from video data
2. **Improved Synchronization**: Enhanced timestamp precision and synchronization mechanisms
3. **Extended Sensor Support**: Additional support for a wider range of physiological sensors
4. **Optimized Performance**: Performance improvements for real-time processing of high-resolution video streams

While RGBTPhys_CPP focuses primarily on synchronized data capture, FactorizePhys adds the capability to process and analyze the captured data in real-time, extracting meaningful physiological signals.

The library provides a multi-threaded framework for capturing data from different sources simultaneously, ensuring that the data streams are properly synchronized. This is crucial for applications that require precise temporal alignment between visual data (RGB and thermal) and physiological measurements.

## Key Components and Functionality

### Core Components

1. **ConfigReader**: Handles reading and parsing configuration files that specify capture parameters.
2. **CaptureRGB**: Manages RGB camera capture, including initialization, frame acquisition, and cleanup.
3. **CaptureThermal**: Manages thermal camera capture, specifically designed for FLIR cameras.
4. **SerialCom**: Handles serial communication for physiological sensors, reading data from devices like Arduino or Shimmer.
5. **Utils**: Provides utility functions for file operations, directory creation, and other common tasks.

### Main Functionality

- **Synchronized Multi-modal Data Capture**: Captures RGB video, thermal video, and physiological data with precise timing synchronization.
- **Configurable Capture Parameters**: Allows customization of frame rates, resolutions, acquisition durations, and other parameters through configuration files.
- **Experimental Condition Support**: Organizes data by participant ID and experimental condition for structured data collection.
- **Cross-platform Support**: Works on both Windows and Linux systems.
- **Real-time Factorization**: Implements algorithms to separate physiological signals from video data in real-time.

### Factorization Algorithms

FactorizePhys implements several advanced algorithms for extracting physiological signals from video data:

1. **Blind Source Separation (BSS)**: Separates mixed signals into their constituent source signals without prior knowledge of the mixing process.
   - **Independent Component Analysis (ICA)**: Assumes statistical independence between source signals.
   - **Principal Component Analysis (PCA)**: Identifies orthogonal components that explain the maximum variance in the data.

2. **Motion-Robust Factorization**: Algorithms designed to be resilient to subject movement.
   - **Motion Compensation**: Tracks and compensates for subject movement before signal extraction.
   - **Adaptive Region Selection**: Dynamically adjusts regions of interest based on motion detection.

3. **Multi-modal Fusion**: Combines information from RGB and thermal video streams for more robust signal extraction.
   - **Weighted Fusion**: Assigns weights to different modalities based on signal quality.
   - **Feature-level Fusion**: Combines features extracted from different modalities.

4. **Temporal Filtering**: Applies various filters to enhance physiological signals and reduce noise.
   - **Bandpass Filtering**: Isolates frequency bands associated with physiological processes.
   - **Wavelet Denoising**: Uses wavelet transforms to separate signal from noise.

## Integration with the Main Project

FactorizePhys integrates with the GSR-RGBT project as a submodule, providing the low-level data capture functionality needed for synchronized multi-modal data collection. The main project uses this library to:

1. Capture synchronized data during experiments
2. Ensure temporal alignment between different data streams
3. Organize the captured data in a structured format for subsequent analysis
4. Extract physiological signals from video data in real-time

### Python Integration via factorize_phys_capture.py

The main project includes a Python wrapper class `FactorizePhysCaptureThread` in `src/capture/factorize_phys_capture.py` that provides a convenient interface to the FactorizePhys C++ library. This class:

1. Extends the `BaseCaptureThread` class to maintain consistency with the project's architecture
2. Provides both real capture and simulation modes for development and testing
3. Emits PyQt signals with captured frames and data
4. Handles configuration and data organization

Example usage of the Python wrapper:

```python
from src.capture.factorize_phys_capture import FactorizePhysCaptureThread

# Create a capture thread instance
capture_thread = FactorizePhysCaptureThread(
    config_file="third_party/FactorizePhys/default_config",
    base_save_path="data/recordings",
    participant_id="Subject_01",
    simulation_mode=False  # Set to True for testing without hardware
)

# Connect signals to handlers
capture_thread.rgb_frame_captured.connect(handle_rgb_frame)
capture_thread.thermal_frame_captured.connect(handle_thermal_frame)
capture_thread.phys_data_captured.connect(handle_phys_data)

# Start capturing
capture_thread.start()

# ... your application code ...

# Stop capturing
capture_thread.stop()
```

For simulation mode, you can use:

```python
# Create a simulation capture thread
simulation_thread = FactorizePhysCaptureThread(
    config_file="third_party/FactorizePhys/default_config",
    base_save_path="data/simulations",
    participant_id="SimSubject_01",
    simulation_mode=True
)

# The simulation will generate synthetic data that mimics real capture
simulation_thread.start()
```

## Usage Examples

### Basic Usage

1. **Prepare a Configuration File**: Create a configuration file specifying capture parameters (e.g., frame rates, resolutions, experimental condition).

2. **Run the Executable**: Execute the RGBTPhys executable with the configuration file, base save path, and participant ID:

   ```bash
   ./RGBTPhys.exe config_file.txt /path/to/save/data participant_id
   ```

3. **Collect Data**: The program will create the necessary directories and start capturing data from all configured sources.

### Configuration Options

The library supports various configuration options, including:

- `thread_sleep_interval_acquisition`: Sleep interval between acquisition cycles (microseconds)
- `acquisition_duration`: Duration of data capture (seconds)
- `exp_condition`: Experimental condition identifier
- `thermal_fps`, `rgb_fps`: Frame rates for thermal and RGB cameras
- `thermal_im_width`, `thermal_im_height`, `rgb_im_width`, `rgb_im_height`: Image dimensions
- `capture_phys`: Enable/disable physiological data capture
- `com_port`, `baud_rate`: Serial port settings for physiological sensors
- `phys_channels`: Channels to capture from physiological sensors

### Example Configuration Files

The repository includes several pre-configured files for different experimental conditions:

- `default_config`: Basic configuration for standard capture
- `config_baseline`: Configuration for baseline recordings
- `config_math_difficult`: Configuration for difficult math task recordings
- `config_math_easy`: Configuration for easy math task recordings
- `config_movement`: Configuration for movement recordings

## Future Improvements

Potential improvements for the FactorizePhys library could include:

1. Better documentation of the factorization algorithms and methods
2. Integration with more types of physiological sensors
3. Real-time visualization of captured data
4. Enhanced error handling and recovery mechanisms
5. Support for network-based data streaming
