# Device Integration Guide

This document provides information on how to integrate the FLIR A65 thermal camera with Shimmer 3 GSR sensor and Logitech Kyro webcam for synchronized data collection.

## Hardware Requirements

### FLIR A65 Thermal Camera
- **Resolution**: 640 × 480 pixels
- **Thermal Sensitivity**: < 50 mK
- **Spectral Range**: 7.5–13 μm
- **Frame Rate**: Up to 30 Hz
- **Interface**: Gigabit Ethernet (GigE Vision compatible)
- **Power**: Power over Ethernet (PoE) or 12/24 V DC

### Shimmer3 GSR+ Unit
- **Sensors**: Galvanic Skin Response (GSR), Photoplethysmogram (PPG), and 3-axis accelerometer
- **Sampling Rate**: Configurable, typically 32 Hz for GSR
- **Interface**: Bluetooth or USB (via dock)
- **Power**: Rechargeable Li-ion battery

### Logitech Kyro Webcam
- **Resolution**: 1080p/30fps or 720p/60fps
- **Interface**: USB 2.0 or higher
- **Features**: Autofocus, light correction

## Connection Setup

### FLIR A65 Setup
1. **Network Configuration**:
   - Connect the FLIR A65 to a PoE-enabled network switch or use a PoE injector
   - Configure a static IP address for the camera (e.g., 192.168.1.100)
   - Set your computer's network adapter to a compatible IP address (e.g., 192.168.1.10)

2. **Software Installation**:
   - Install FLIR Atlas SDK for development
   - Install PySpin library for Python integration
   - Verify connection using FLIR tools or a simple Python script

### Shimmer3 GSR+ Setup
1. **Hardware Preparation**:
   - Charge the Shimmer3 unit fully before use
   - Attach GSR electrodes to the unit
   - Place the unit in the dock if using USB connection

2. **Software Installation**:
   - Install Shimmer Consensys software for configuration
   - Install PyShimmer library for Python integration
   - Configure the sampling rate to 32 Hz for GSR data

### Logitech Kyro Setup
1. **Hardware Connection**:
   - Connect the webcam to a USB 3.0 port for optimal performance
   - Position the camera to capture the desired field of view

2. **Software Configuration**:
   - Install Logitech G HUB software for advanced settings
   - Configure resolution and frame rate (recommend 30 fps to match thermal camera)
   - Use OpenCV for capturing frames in Python

## Integration with Python

### Sample Code for Device Initialization

```python
# Initialize FLIR A65 camera
def initialize_thermal_camera():
    import PySpin
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        raise Exception("No FLIR cameras detected")
    camera = cam_list.GetByIndex(0)
    camera.Init()
    # Configure camera settings
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(30.0)  # 30 fps
    return camera, system

# Initialize Shimmer3 GSR+
def initialize_gsr_sensor(com_port):
    from pyshimmer import Shimmer
    shimmer = Shimmer(com_port)
    shimmer.connect()
    shimmer.set_sampling_rate(32.0)  # 32 Hz
    shimmer.enable_sensor('GSR')
    shimmer.start_streaming()
    return shimmer

# Initialize Logitech Kyro webcam
def initialize_webcam(device_id=0):
    import cv2
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 30 fps
    if not cap.isOpened():
        raise Exception(f"Could not open webcam (device {device_id})")
    return cap
```

## Timestamp Synchronization

Synchronizing timestamps between different devices is crucial for accurate data analysis. Here are several approaches:

### 1. Network Time Protocol (NTP)

Ensure all devices and the host computer are synchronized to the same NTP server. This provides millisecond-level synchronization.

```python
# Check NTP synchronization status on Linux/macOS
# Run this before starting data collection
import subprocess
subprocess.run(["ntpstat"])  # or "w32tm /query /status" on Windows
```

### 2. Common Time Reference

Use the host computer's time as a reference and record timestamps when data is received:

```python
import time

def record_with_timestamp(data, device_name):
    timestamp = time.time()  # Unix timestamp
    return {
        "timestamp": timestamp,
        "device": device_name,
        "data": data
    }
```

### 3. Hardware Synchronization

For the most precise synchronization, use hardware triggers:

1. **Trigger Signal**: Generate a trigger signal (e.g., using Arduino) that is sent to all devices simultaneously
2. **Event Markers**: Record the trigger events in each data stream
3. **Post-processing Alignment**: Align the data streams based on these markers

### 4. Cross-Correlation Method

For post-processing synchronization:

1. Record a distinctive event that affects all sensors (e.g., a sharp temperature change)
2. Use cross-correlation to find the time offset between signals
3. Apply the offset to align the data streams

```python
import numpy as np
from scipy import signal

def find_time_offset(signal1, signal2):
    correlation = signal.correlate(signal1, signal2, mode='full')
    lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
    lag = lags[np.argmax(correlation)]
    return lag / sampling_rate  # Convert lag to seconds
```

## Data Collection Workflow

1. **Setup Phase**:
   - Position all devices appropriately
   - Connect and initialize all devices
   - Verify connections and data streams

2. **Synchronization Phase**:
   - Generate a synchronization event (e.g., temperature change, light flash)
   - Record this event in all data streams

3. **Recording Phase**:
   - Start recording from all devices
   - Store data with timestamps
   - Monitor for any data loss or connection issues

4. **Post-processing Phase**:
   - Align data streams using synchronization markers
   - Apply any necessary calibration
   - Analyze the synchronized data

## Troubleshooting

### FLIR A65 Issues
- **No Connection**: Verify IP address settings and network connectivity
- **Low Frame Rate**: Check network bandwidth and reduce resolution if necessary
- **Image Quality**: Adjust focus and thermal span/level settings

### Shimmer3 GSR+ Issues
- **Connection Drops**: Ensure battery is charged and within Bluetooth range
- **Noisy Data**: Check electrode placement and skin preparation
- **Missing Data**: Verify sampling rate and buffer settings

### Logitech Kyro Issues
- **Poor Image Quality**: Adjust lighting conditions and camera settings
- **Frame Drops**: Reduce resolution or frame rate if CPU usage is high
- **Latency**: Use a dedicated USB controller if sharing bandwidth with other devices

## References

1. FLIR A65 User Manual: [FLIR Systems Documentation](https://www.flir.com/products/a65/)
2. Shimmer3 GSR+ User Guide: [Shimmer Documentation](https://shimmersensing.com/product/shimmer3-gsr-unit/)
3. Logitech Kyro Documentation: [Logitech Support](https://www.logitech.com/en-us/products/webcams/)
4. PySpin Documentation: [FLIR Systems SDK](https://www.flir.com/products/spinnaker-sdk/)
5. PyShimmer GitHub Repository: [PyShimmer](https://github.com/ShimmerResearch/pyshimmer)