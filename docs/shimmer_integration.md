# Shimmer GSR and PPG Integration with RGBTPhys_CPP

## Overview

This document describes how to use the Shimmer GSR and PPG integration with RGBTPhys_CPP. The integration allows RGBTPhys_CPP to capture and process physiological data from Shimmer devices, specifically Galvanic Skin Response (GSR) and Photoplethysmography (PPG) signals.

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

## Future Improvements

Potential future improvements to the Shimmer integration include:

1. **Real-time signal quality assessment**: Automatically detect and flag poor-quality signals.
2. **Advanced signal processing**: Implement more sophisticated processing algorithms for GSR and PPG signals.
3. **Support for additional Shimmer sensors**: Add support for other sensors available on Shimmer devices, such as ECG, EMG, etc.
4. **Wireless connectivity**: Add support for Bluetooth connection to Shimmer devices.