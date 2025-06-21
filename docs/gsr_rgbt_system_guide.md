# GSR-RGBT Automated Data Acquisition System

This guide provides instructions for setting up and using the GSR-RGBT automated data acquisition system, which captures synchronized RGB video, thermal video, and physiological data during experiments.

## System Overview

The GSR-RGBT system consists of three main components:

1. **PC-Side Data Acquisition** (pc_data_acq.py): Captures RGB video from a Logitech Brio camera and physiological data (GSR, PPG, Accelerometer) from a Shimmer3 GSR+ unit.

2. **Android Thermal App**: Captures thermal video from a Topdon/InfiRay P2 Pro thermal dongle connected to an Android device.

3. **Experiment Orchestration** (run_experiment.py): Controls the entire experimental protocol, presents stimuli, and sends precise LSL markers for events.

All data streams are synchronized using Lab Streaming Layer (LSL), which provides precise timing across devices.

## Prerequisites

### Hardware Requirements

- **PC**: Windows 10/11 with sufficient processing power
- **Logitech Brio 4K RGB Camera**: Connected via USB 3.0
- **Shimmer3 GSR+ Unit**: With electrodes and PPG probe
- **Android Device**: Samsung Galaxy S21/S22 or similar with USB-C port
- **Topdon/InfiRay P2 Pro Thermal Dongle**: Connected to the Android device
- **Display**: For stimulus presentation

### Software Requirements

- **Python 3.11** or later
- **Android Studio** (for building the Android thermal app)
- **LSL LabRecorder**: For recording all data streams
- **ADB** (Android Debug Bridge): For controlling the Android device

## Installation

### 1. PC Environment Setup

1. Install required software:
   ```
   winget install --id Git.Git -e --source winget
   winget install -e --id Python.Python.3.11 --scope machine
   ```

2. Download and install Android SDK Platform-Tools (for ADB):
   - Download from [Android Developers website](https://developer.android.com/studio/releases/platform-tools)
   - Add ADB to your system's PATH

3. Create and set up a Python virtual environment:
   ```
   mkdir C:\Users\YourUser\Documents\GSR_RGBT_Project
   cd C:\Users\YourUser\Documents\GSR_RGBT_Project
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install pyshimmer pylsl psychopy numpy pandas opencv-python
   ```

4. Download LSL LabRecorder from the [LSL GitHub repository](https://github.com/labstreaminglayer/App-LabRecorder/releases)

### 2. Android Environment Setup

1. Install Android Studio from the [official website](https://developer.android.com/studio)

2. Build and install the Android thermal app:
   - Follow the instructions in [android_thermal_app_guide.md](android_thermal_app_guide.md)

3. Enable Developer Options on the Android device:
   - Go to Settings > About phone > Software information
   - Tap Build number 7 times
   - Enable USB debugging in Developer Options

4. Connect the Android device to the PC via USB and verify the connection:
   ```
   adb devices
   ```

### 3. Project Files Setup

1. Clone or download this project repository

2. Create the necessary directories:
   ```
   mkdir -p data/stimuli
   mkdir -p data/recordings
   ```

3. Place your stimulus videos in the `data/stimuli` directory:
   - baseline_video.mp4
   - stress_video.mp4
   - recovery_video.mp4

## Usage

### 1. Prepare the Hardware

1. Connect the Logitech Brio camera to a USB 3.0 port on the PC

2. Turn on the Shimmer3 GSR+ unit and connect it to the PC via Bluetooth
   - Note the COM port assigned to the Shimmer device (e.g., COM3)

3. Connect the Topdon/InfiRay P2 Pro thermal dongle to the Android device

4. Connect the Android device to the same network as the PC (for LSL synchronization)

### 2. Prepare the Participant

1. Explain the study and obtain consent

2. Attach Shimmer3 electrodes:
   - GSR: Non-dominant index/middle fingers
   - PPG: Earlobe or fingertip

3. Position the participant comfortably in front of the display

4. Ensure both cameras (RGB and thermal) have a clear view of the participant

### 3. Run the Experiment

1. Start the Android thermal app on the phone
   - The app will wait for commands from the PC

2. Open a command prompt and activate your Python virtual environment:
   ```
   .\.venv\Scripts\activate
   ```

3. Run the experiment orchestration script:
   ```
   python src\scripts\run_experiment.py --subject-id SUBJECT001 --fullscreen
   ```

4. When prompted, start recording in LSL LabRecorder
   - Ensure all streams are visible: RGB_Markers, Shimmer_Data, Thermal_Video, and ExperimentMarkers

5. The experiment will run automatically through all conditions
   - The script will display the appropriate videos and send markers for each condition
   - The participant should follow the instructions on the screen

6. At the end of the experiment, the script will automatically stop all data acquisition

### 4. Command-Line Options

The `run_experiment.py` script supports several command-line options:

```
python src\scripts\run_experiment.py --help
```

Key options:
- `--subject-id`: Required. Subject ID for this recording session
- `--fullscreen`: Run in fullscreen mode
- `--skip-thermal`: Skip thermal camera (Android) acquisition
- `--skip-shimmer`: Skip Shimmer GSR+ acquisition
- `--labrecorder`: Path to LabRecorder executable
- `--pc-acq`: Path to PC data acquisition script
- `--android-pkg`: Android package name for thermal app
- `--android-svc`: Android service name for thermal app
- `--stimuli-dir`: Directory containing stimulus videos
- `--output-dir`: Directory for saving output data

### 5. Data Collection

The system collects the following data:

1. **RGB Video**: Saved as an AVI file in the current directory
2. **Thermal Video**: Saved on the Android device in the app's storage
3. **Physiological Data**: GSR, PPG, and Accelerometer data streamed to LSL
4. **LSL Recording**: All streams are recorded to an XDF file by LabRecorder

### 6. After the Experiment

1. Transfer the thermal video files from the Android device to the PC:
   ```
   adb pull /sdcard/Android/data/com.yourcompany.thermalapp/files/thermal_recordings/ data/recordings/
   ```

2. Organize all data files in the session directory:
   - Move the RGB video file (brio_output.avi)
   - Move the LSL recording (*.xdf)
   - Include the thermal video and timestamp files

## Troubleshooting

### PC-Side Acquisition Issues

1. **Shimmer Connection Problems**:
   - Ensure the Shimmer is charged and turned on
   - Verify the COM port in Device Manager
   - Try restarting the Shimmer device

2. **Camera Issues**:
   - Check if the camera is recognized in Device Manager
   - Try a different USB port
   - Verify no other applications are using the camera

### Android Thermal App Issues

1. **USB Connection Issues**:
   - Ensure USB debugging is enabled
   - Try disconnecting and reconnecting the device
   - Check the USB cable

2. **Thermal Dongle Not Detected**:
   - Verify the dongle is properly connected
   - Restart the Android app
   - Check if the device has the necessary permissions

### LSL Synchronization Issues

1. **Streams Not Visible**:
   - Ensure all devices are on the same network
   - Check firewall settings
   - Restart the LSL LabRecorder

2. **Clock Synchronization Problems**:
   - Ensure the PC and Android device have accurate time settings
   - Run the experiment with all devices connected to the same network

## Data Analysis

After collecting data, you can use the following steps for analysis:

1. Load the XDF file using the `pyxdf` Python library:
   ```python
   import pyxdf
   data, header = pyxdf.load_xdf('path/to/recording.xdf')
   ```

2. Extract the different data streams:
   ```python
   # Find the streams by name
   shimmer_stream = next(stream for stream in data if stream['info']['name'][0] == 'Shimmer_Data')
   rgb_stream = next(stream for stream in data if stream['info']['name'][0] == 'RGB_Markers')
   thermal_stream = next(stream for stream in data if stream['info']['name'][0] == 'Thermal_Video')
   marker_stream = next(stream for stream in data if stream['info']['name'][0] == 'ExperimentMarkers')
   
   # Extract data and timestamps
   shimmer_data = shimmer_stream['time_series']
   shimmer_timestamps = shimmer_stream['time_stamps']
   
   rgb_markers = rgb_stream['time_series']
   rgb_timestamps = rgb_stream['time_stamps']
   
   thermal_data = thermal_stream['time_series']
   thermal_timestamps = thermal_stream['time_stamps']
   
   markers = marker_stream['time_series']
   marker_timestamps = marker_stream['time_stamps']
   ```

3. Analyze the data using your preferred methods and libraries (e.g., NumPy, Pandas, scikit-learn)

## References

- [Lab Streaming Layer (LSL) Documentation](https://labstreaminglayer.readthedocs.io/)
- [PsychoPy Documentation](https://www.psychopy.org/documentation.html)
- [Shimmer3 GSR+ User Guide](https://shimmersensing.com/support/wireless-sensor-documentation/)
- [OpenCV Documentation](https://docs.opencv.org/)