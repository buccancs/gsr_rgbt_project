# Shimmer Integration Guide

## Overview

This document describes how to use the Shimmer integration in the GSR-RGBT project. The integration allows the application to capture and process physiological data from Shimmer devices, specifically Galvanic Skin Response (GSR) and Photoplethysmography (PPG) signals.

The project includes several components that work together to provide a seamless experience when working with Shimmer devices:

1. **Automatic COM Port Detection**: The application automatically detects the COM port of connected Shimmer devices.
2. **Dynamic Sensor Configuration**: Users can specify which sensors to enable on the Shimmer device.
3. **Unified API Adapter**: The ShimmerAdapter class provides a unified interface to multiple Shimmer APIs.
4. **Integration with RGBTPhys_CPP**: The Shimmer devices can be used with RGBTPhys_CPP for advanced processing.

## Configuration

To use a Shimmer device with RGBTPhys_CPP, you need to configure the following parameters in your configuration file:

```
capture_phys = true
com_port = COM4  # Change to your Shimmer device's COM port
baud_rate = 2000000
phys_channels = EDA,Resp,PPG Finger,PPG Ear,arduino_ts,EventCode
is_shimmer_device = true
shimmer_device_type = GSR+  # Change to your Shimmer device type
```

### Configuration Parameters

- **capture_phys**: Set to `true` to enable physiological data capture.
- **com_port**: The serial port where your Shimmer device is connected (e.g., `COM4` on Windows).
- **baud_rate**: The baud rate for serial communication with the Shimmer device.
- **phys_channels**: A comma-separated list of physiological channels to capture. For Shimmer GSR+ devices, this typically includes:
  - `EDA`: Electrodermal Activity (another term for GSR)
  - `Resp`: Respiration
  - `PPG Finger`: PPG signal from a finger sensor
  - `PPG Ear`: PPG signal from an ear sensor
  - `arduino_ts`: Timestamp from the device
  - `EventCode`: Event marker
- **is_shimmer_device**: Set to `true` to enable Shimmer-specific processing.
- **shimmer_device_type**: The type of Shimmer device you're using (e.g., `GSR+`).

## Supported Shimmer Devices

The current implementation supports the following Shimmer devices:

- **Shimmer GSR+**: Captures GSR (EDA) and PPG signals.

## Data Processing

When using a Shimmer device, RGBTPhys_CPP performs the following processing:

### GSR (EDA) Processing

- Converts raw GSR values (typically in kOhms) to calibrated values in microSiemens (µS).
- The conversion formula is: `microSiemens = 1000.0 / gsr_value_in_kOhms`
- This conversion makes the values more intuitive for analysis, as higher conductance (lower resistance) corresponds to higher arousal.

### PPG Processing

- Captures raw PPG signals from finger and/or ear sensors.
- Logs the values for further processing.
- Future enhancements may include real-time peak detection and heart rate calculation.

## Data Output

The physiological data is saved to a CSV file in the specified output directory. The file includes:

1. A header row with the channel names (from the `phys_channels` parameter).
2. Data rows with comma-separated values for each channel.

The filename includes a timestamp to ensure uniqueness.

## Troubleshooting

### Common Issues

1. **No data received**:
   - Ensure the Shimmer device is properly connected and powered on.
   - Verify that the correct COM port is specified in the configuration file.
   - Check that the baud rate matches the device's configuration.

2. **Incorrect data format**:
   - Ensure the `phys_channels` parameter matches the actual channels being sent by the Shimmer device.
   - Check the Shimmer device's configuration to ensure it's sending the expected data.

3. **GSR values out of range**:
   - Very high or low GSR values may indicate poor electrode contact or sensor issues.
   - Ensure the GSR electrodes are properly attached and have good skin contact.

## Integration with Other Components

The Shimmer GSR and PPG data can be used in conjunction with other components of the GSR-RGBT project:

- **Synchronized with RGB and thermal video**: The physiological data is time-synchronized with the video data, allowing for multimodal analysis.
- **Analysis with neurokit2 and physiokit**: The captured data can be processed and analyzed using the neurokit2 and physiokit libraries.
- **Ground truth for remote sensing**: The contact-based measurements can serve as ground truth for remote physiological sensing algorithms.

## New Features

### Automatic COM Port Detection

The `find_shimmer_com_port` function in `src/utils/device_utils.py` automatically scans all available COM ports and identifies the one connected to a Shimmer device. This eliminates the need for users to manually configure the COM port.

```python
from src.utils.device_utils import find_shimmer_com_port, DeviceNotFoundError

try:
    shimmer_port = find_shimmer_com_port()
    print(f"Found Shimmer device on port {shimmer_port}")
except DeviceNotFoundError as e:
    print(f"Error: {e}")
    # Fall back to simulation mode or another alternative
```

If no Shimmer device is found, the function raises a `DeviceNotFoundError`. The application handles this by falling back to simulation mode.

### Dynamic Sensor Configuration

The `GsrCaptureThread` class in `src/data_collection/capture/gsr_capture.py` supports dynamic configuration of which sensors to enable on the Shimmer device. By default, it enables the GSR, PPG, and Accelerometer sensors, but you can specify a custom configuration:

```python
from src.data_collection.capture.gsr_capture import GsrCaptureThread
import pyshimmer

# Enable only GSR sensor
custom_sensors = pyshimmer.Shimmer.SENSOR_GSR

# Create a GSR capture thread with custom sensor configuration
gsr_thread = GsrCaptureThread(
    port="COM3",
    sampling_rate=32,
    simulation_mode=False,
    sensors_to_enable=custom_sensors
)
```

### Unified API Adapter

The `ShimmerAdapter` class in `src/utils/shimmer_adapter.py` provides a unified interface to multiple Shimmer APIs:

1. **pyshimmer**: Used for basic functionality (Bluetooth communication, data streaming)
2. **Shimmer-C-API**: Used for advanced signal processing (ECG/PPG to Heart Rate/IBI, filtering)
3. **Shimmer-Java-Android-API**: Used for additional features (GSR calibration, 3D orientation)
4. **ShimmerAndroidAPI**: Used for Android-specific features

The adapter automatically detects which APIs are available and provides a consistent interface regardless of the underlying implementation.

#### Basic Usage

```python
from src.utils.shimmer_adapter import ShimmerAdapter

# Create an adapter
adapter = ShimmerAdapter(
    port="COM3",
    sampling_rate=32,
    simulation_mode=False
)

# Connect to the device
if adapter.connect():
    print("Connected to Shimmer device")

    # Configure sensors
    adapter.set_enabled_sensors(0x07)  # Example bitmask

    # Start streaming
    adapter.start_streaming()

    # Read data
    packet = adapter.read_data_packet()
    if packet:
        print(f"GSR value: {packet.get('GSR_CAL')}")

    # Stop streaming and disconnect
    adapter.stop_streaming()
    adapter.disconnect()
else:
    print("Failed to connect to Shimmer device")
```

#### Advanced Processing

The adapter provides methods for advanced signal processing when the Shimmer-C-API is available:

```python
# Process ECG data to extract heart rate and IBI
ecg_data = [1.0, 2.0, 3.0]  # Example data
heart_rate, ibi_values = adapter.process_ecg_to_hr(ecg_data)
print(f"Heart rate: {heart_rate} BPM")
print(f"IBI values: {ibi_values} ms")

# Process PPG data to extract heart rate and IBI
ppg_data = [1.0, 2.0, 3.0]  # Example data
heart_rate, ibi_values = adapter.process_ppg_to_hr(ppg_data)
print(f"Heart rate: {heart_rate} BPM")
print(f"IBI values: {ibi_values} ms")

# Apply a filter to data
data = [1.0, 2.0, 3.0]  # Example data
filter_type = "lowpass"
params = {"cutoff": 10.0}
filtered_data = adapter.apply_filter(data, filter_type, params)
```

#### Checking Available Capabilities

You can check which advanced processing capabilities are available:

```python
capabilities = adapter.get_advanced_processing_capabilities()
print(f"Available capabilities: {capabilities}")
```

## Testing

The project includes comprehensive test suites for all Shimmer-related components:

- `src/tests/unit/utils/test_device_utils.py`: Tests for the automatic COM port detection.
- `src/tests/unit/data_collection/capture/test_gsr_capture.py`: Tests for the GSR capture thread, including dynamic sensor configuration.
- `src/tests/unit/utils/test_shimmer_adapter.py`: Tests for the ShimmerAdapter class.

To run the tests:

```bash
python -m unittest discover -s src/tests
```

## Future Improvements

Potential future improvements to the Shimmer integration include:

1. **Real-time signal quality assessment**: Automatically detect and flag poor-quality signals.
2. **Advanced signal processing**: Implement more sophisticated processing algorithms for GSR and PPG signals.
3. **Support for additional Shimmer sensors**: Add support for other sensors available on Shimmer devices, such as ECG, EMG, etc.
4. **Wireless connectivity**: Add support for Bluetooth connection to Shimmer devices.
