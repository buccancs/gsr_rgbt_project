# RGBTPhys_CPP Integration

## Overview

This document describes the integration of the RGBTPhys_CPP library into the GSR-RGBT project. The RGBTPhys_CPP library provides synchronized data capture from RGB cameras, thermal cameras, and physiological sensors, which is essential for accurate multi-modal data analysis.

## Purpose

The integration of RGBTPhys_CPP enables:

1. **Synchronized Data Capture**: Ensures that all data streams (RGB video, thermal video, and physiological data) are captured with precise timing synchronization.
2. **Improved Performance**: The C++ implementation offers better performance for real-time data capture compared to pure Python implementations.
3. **Consistent Data Format**: Standardizes the data format across different capture modalities, making subsequent analysis more straightforward.

## Implementation Details

### Integration Approach

The integration is implemented through a Python wrapper class (`RGBTPhysCaptureThread`) that:

1. Extends the existing `BaseCaptureThread` class to maintain consistency with the project's architecture
2. Launches the RGBTPhys_CPP executable as a subprocess
3. Monitors the output of the subprocess to detect when new data is captured
4. Emits PyQt signals with the captured data and timestamps

### Key Components

- **RGBTPhysCaptureThread**: The main Python wrapper class that interfaces with RGBTPhys_CPP
- **RGBTPhys_CPP**: The C++ library that handles the actual data capture and synchronization
- **test_rgbt_phys_integration.py**: A test script to verify the integration

### Internal Synchronization

RGBTPhys_CPP implements a sophisticated synchronization system to ensure that all data streams (RGB video, thermal video, and physiological data) are properly time-aligned:

#### Timestamp-based Synchronization

1. **High-precision Timestamps**: Each frame and physiological data point is tagged with a high-precision timestamp (nanosecond resolution) using `std::chrono::high_resolution_clock`.

2. **Central Clock**: A central clock thread generates timestamps at a fixed frequency (typically 1000 Hz) to provide a common time reference for all data streams.

3. **Timestamp Interpolation**: For data streams with different sampling rates, timestamps are interpolated to align with the central clock.

4. **Timestamp Logging**: All timestamps are logged to files for post-processing alignment if needed.

#### Hardware-level Synchronization

1. **Thread Priority Management**: Capture threads are assigned high priority to minimize scheduling jitter.

2. **Buffer Management**: Carefully designed buffer systems prevent data loss during high-load situations.

3. **Frame Dropping Prevention**: The system is designed to prevent frame dropping, with monitoring and warning systems in place.

4. **Synchronization Verification**: The system continuously monitors the synchronization quality and logs any discrepancies.

### Shimmer GSR and PPG Integration

RGBTPhys_CPP includes built-in support for Shimmer physiological sensors, particularly the Shimmer GSR+ device which provides Galvanic Skin Response (GSR) and Photoplethysmography (PPG) measurements:

1. **Serial Communication**: Implements serial communication with Shimmer devices using configurable COM ports and baud rates.

2. **Channel Configuration**: Supports flexible channel configuration to capture specific physiological signals (EDA, PPG, respiration, etc.).

3. **Data Processing**: Performs initial processing of raw sensor data, including calibration and unit conversion.

4. **Synchronized Storage**: Stores the physiological data with precise timestamps aligned with the video streams.

For detailed information about the Shimmer integration, please refer to the [Shimmer Integration](shimmer_integration.md) document.

## Usage

### Basic Usage

```python
from src.capture.rgbt_phys_capture import RGBTPhysCaptureThread

# Create an instance of the RGBTPhys capture thread
rgbt_phys = RGBTPhysCaptureThread(
    config_file="default_config",
    base_save_path="/path/to/save/data",
    participant_id="subject_01",
    simulation_mode=False  # Set to True for simulation mode
)

# Connect signals to your data processing methods
rgbt_phys.rgb_frame_captured.connect(your_rgb_processing_method)
rgbt_phys.thermal_frame_captured.connect(your_thermal_processing_method)
rgbt_phys.phys_data_captured.connect(your_phys_processing_method)

# Start the capture
rgbt_phys.start()

# ... your application code ...

# Stop the capture when done
rgbt_phys.stop()
```

### Configuration

The RGBTPhys_CPP library uses configuration files to specify capture parameters. Several default configuration files are provided in the RGBTPhys_CPP directory:

- `default_config`: Basic configuration for standard capture
- `config_baseline`: Configuration for baseline recordings
- `config_math_difficult`: Configuration for difficult math task recordings
- `config_math_easy`: Configuration for easy math task recordings
- `config_movement`: Configuration for movement recordings

You can also create custom configuration files as needed.

### Simulation Mode

For development and testing without hardware, the `RGBTPhysCaptureThread` class provides a simulation mode that generates synthetic data:

```python
rgbt_phys = RGBTPhysCaptureThread(
    config_file="default_config",
    base_save_path="/path/to/save/data",
    participant_id="subject_01",
    simulation_mode=True  # Enable simulation mode
)
```

## Testing

To test the integration, run the provided test script:

```bash
python src/scripts/test_rgbt_phys_integration.py
```

This script will:
1. Create an instance of `RGBTPhysCaptureThread` in simulation mode
2. Capture data for a short period
3. Verify that data is being received from all sources
4. Save sample frames to disk for visual inspection

To test with real hardware, modify the `simulation_mode` parameter in the script.

## Troubleshooting

### Common Issues

1. **RGBTPhys_CPP executable not found**:
   - Ensure that the RGBTPhys_CPP submodule is properly initialized and built
   - Check that the path to the executable is correct in `RGBTPhysCaptureThread._run_real_capture()`

2. **Configuration file not found**:
   - Ensure that the specified configuration file exists in the RGBTPhys_CPP directory
   - Use an absolute path to the configuration file if necessary

3. **No data received**:
   - Check that the hardware is properly connected and powered
   - Verify that the configuration file has the correct settings for your hardware
   - Check the logs for any error messages from RGBTPhys_CPP

### Debugging

The `RGBTPhysCaptureThread` class logs detailed information about its operation. To enable more verbose logging, set the logging level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Improvements

1. **Direct Memory Access**: Implement a more efficient data transfer mechanism between RGBTPhys_CPP and Python, possibly using shared memory or memory-mapped files.
2. **Python Bindings**: Create proper Python bindings for RGBTPhys_CPP using tools like pybind11 or SWIG for tighter integration.
3. **Real-time Visualization**: Add real-time visualization of the synchronized data streams.
4. **Extended Configuration**: Provide more configuration options through the Python interface.
