# GSR-RGBT Project Comprehensive Guide

This guide provides detailed instructions for setting up, configuring, and using the GSR-RGBT Project, which combines RGB video, thermal imaging, and physiological sensing for contactless GSR prediction research.

## Table of Contents
1. [Introduction](#introduction)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Installation](#software-installation)
4. [Device Configuration](#device-configuration)
5. [Physical Setup](#physical-setup)
6. [Data Synchronization](#data-synchronization)
7. [Using the Unified Tool Script](#using-the-unified-tool-script)
8. [Data Collection Protocol](#data-collection-protocol)
9. [Troubleshooting](#troubleshooting)
10. [Post-Recording Checks](#post-recording-checks)
11. [Next Steps](#next-steps)

## Introduction

The GSR-RGBT Project is a research platform for estimating Galvanic Skin Response (GSR) from synchronized RGB and thermal video streams. The system captures:

- RGB video of the participant's hand
- Thermal video of the same hand
- Ground-truth GSR measurements from the opposite hand

This multimodal data is used to train machine learning models that can predict GSR from video data alone, potentially enabling contactless stress and emotional state monitoring.

## Hardware Requirements

### Cameras
1. **RGB Camera**:
   - Recommended: High-quality USB webcam (e.g., Logitech C920 or better)
   - Minimum resolution: 1280×720 at 30fps
   - Connection: USB 3.0 port

2. **FLIR Thermal Camera**:
   - Recommended: FLIR A65 (or similar GigE Vision thermal camera)
   - Resolution: 640×512 pixels
   - Spectral range: 7.5-13 μm
   - Frame rate: 30 Hz
   - Connection: Ethernet port

### Physiological Sensors
1. **Shimmer3 GSR+ Sensor**:
   - Components: Shimmer3 GSR+ unit, electrode leads, disposable electrodes
   - Connection: Bluetooth or serial via dock

### Computer Requirements
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 16GB minimum, 32GB recommended
- Storage: SSD with at least 500GB free space
- GPU: NVIDIA GTX 1660 or better (for real-time processing)
- Ports: Multiple USB 3.0, Ethernet port (Gigabit)
- OS: Windows, macOS, or Linux (some device drivers may be OS-specific)

## Software Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/gsr-rgbt-project.git
cd gsr-rgbt-project
```

### 2. Using the Unified Tool Script

The project includes a unified tool script (`gsr_rgbt_tools.sh`) that automates the setup process:

```bash
# Make the script executable
chmod +x gsr_rgbt_tools.sh

# Run the setup command
./gsr_rgbt_tools.sh setup
```

This script will:
- Check for required system dependencies
- Create and activate a Python virtual environment
- Install required Python packages
- Build Cython extensions
- Check for and help install the FLIR Spinnaker SDK
- Check for and install pyshimmer and other dependencies
- Run system validation checks

### 3. FLIR A65 Thermal Camera Setup

The FLIR thermal camera **requires** the Spinnaker SDK:

1. **Download the FLIR Spinnaker SDK**:
   - Visit the [FLIR website](https://www.flir.com/products/spinnaker-sdk/) and download the SDK for your OS
   - Create a free FLIR account if you don't already have one

2. **Install the Spinnaker SDK**:
   - During installation, make sure to select the option to install the Python bindings (PySpin)
   - You may need administrator privileges to complete the installation

3. **Network Configuration**:
   - Connect the FLIR A65 to a PoE-enabled network switch or use a PoE injector
   - Configure a static IP address for the camera (e.g., 192.168.1.100)
   - Set your computer's network adapter to a compatible IP address (e.g., 192.168.1.10)

### 4. Shimmer3 GSR+ Sensor Setup

1. **Install Shimmer Connect or ConsensysPRO**:
   - Download from [Shimmer website](https://shimmersensing.com/support/wireless-sensor-networks-download/)
   - Use this software to update firmware and configure the sensor

2. **Identify the Serial Port**:
   - When connecting the Shimmer via its USB dock, it will appear as a serial port
   - On macOS, the port will typically be named like `/dev/tty.usbmodem*` or `/dev/cu.*`
   - On Windows, it will be a COM port (e.g., COM3)
   - To find the port, run this command before and after connecting the device:
     ```bash
     # On macOS/Linux
     ls /dev/tty.* /dev/cu.*

     # On Windows (PowerShell)
     Get-WmiObject Win32_SerialPort | Select-Object Name, DeviceID
     ```

## Device Configuration

### Update the Configuration File

Edit the `src/config.py` file to configure the devices properly:

1. **For the FLIR thermal camera**:
   - Set `THERMAL_CAMERA_ID` to the correct camera ID (usually 0 or 1)
   - Set `THERMAL_SIMULATION_MODE = False` to use the real camera

2. **For the Shimmer GSR+ sensor**:
   - Set `GSR_SENSOR_PORT` to the correct serial port (e.g., "/dev/tty.usbmodem1234" or "COM3")
   - Set `GSR_SIMULATION_MODE = False` to use the real sensor

Example configuration:

```python
# Camera settings
RGB_CAMERA_ID = 0
THERMAL_CAMERA_ID = 1
THERMAL_SIMULATION_MODE = False  # Set to False for real camera

# GSR sensor settings
GSR_SENSOR_PORT = "/dev/tty.usbmodem1234"  # Update with your actual port
GSR_SAMPLING_RATE = 128  # Hz
GSR_SIMULATION_MODE = False  # Set to False for real sensor
```

### Verify the System Setup

Run the system validation script to verify that all components are properly configured:

```bash
./gsr_rgbt_tools.sh test
```

This will run both the system validation checks and synchronization tests.

## Physical Setup

### Camera Positioning

1. **Setup Location**:
   - Choose a room with controlled lighting (no direct sunlight)
   - Ensure stable temperature (20-24°C) for consistent thermal readings
   - Minimize air movement (turn off fans/AC during recording)

2. **Camera Mounting**:
   - Position both cameras on tripods approximately 50-70cm from where the participant's hand will be placed
   - Align the RGB and thermal cameras as close together as possible (side by side, ~10cm apart)
   - Angle both cameras to capture the same field of view
   - Ensure the cameras have an unobstructed view of the participant's hand

3. **Camera Settings**:
   - **RGB Camera**:
     - Set resolution to 1280×720 or 1920×1080
     - Frame rate: 30fps
     - Focus: Manual, focused on the hand position
     - Exposure: Manual, to prevent auto-adjustments during recording

   - **FLIR Thermal Camera**:
     - Configure using SpinView:
       - Set acquisition mode to continuous
       - Frame rate: 30Hz
       - Enable timestamp
       - Set emissivity to 0.98 (appropriate for human skin)
       - Temperature range: Typically 20-40°C for human subjects

### Shimmer GSR+ Sensor Setup

1. **Electrode Placement**:
   - Clean the participant's fingers with alcohol wipes
   - Attach electrodes to the palmar surface of:
     - Middle phalanx of index finger
     - Middle phalanx of middle finger
   - Both electrodes should be on the same hand (typically the non-dominant/left hand)
   - Ensure good contact with no air bubbles or gaps

## Data Synchronization

The GSR-RGBT project uses a custom synchronization approach based on a centralized timestamp authority:

### Components

1. **TimestampThread**: A high-priority thread that emits timestamps at a fast, consistent rate (200Hz by default).

2. **Capture Threads**: Separate threads for RGB video, thermal video, and GSR data capture. Each thread captures data at its own rate and associates each data point with the latest timestamp.

3. **DataLogger**: Writes the captured data to disk, along with the associated timestamps.

### Testing Synchronization

To run the synchronization test:

```bash
./gsr_rgbt_tools.sh test
```

This will run both the system validation checks and synchronization tests.

## Using the Unified Tool Script

The `gsr_rgbt_tools.sh` script provides a unified interface for various tasks:

```bash
# Show help
./gsr_rgbt_tools.sh help

# Setup the environment
./gsr_rgbt_tools.sh setup

# Run the data collection application
./gsr_rgbt_tools.sh run --component=app

# Run the full ML pipeline
./gsr_rgbt_tools.sh run --component=pipeline

# Generate mock data
./gsr_rgbt_tools.sh run --component=mock_data

# Run everything (validation, tests, pipeline, and optionally the app)
./gsr_rgbt_tools.sh run

# Run system validation and synchronization tests
./gsr_rgbt_tools.sh test

# Clean up temporary files and build artifacts
./gsr_rgbt_tools.sh clean
```

Options:
- `--no-sdk`: Skip FLIR Spinnaker SDK installation check
- `--force`: Force reinstallation of dependencies
- `--verbose`: Display detailed output
- `--component=COMP`: Run specific component (with 'run' command)
  - Valid components: app, pipeline, mock_data

## Data Collection Protocol

### 1. Pre-Session Setup

1. **Environment Preparation**:
   - Control room temperature (20-24°C)
   - Set up diffuse, consistent lighting
   - Prepare a comfortable chair and hand rest for the participant

2. **System Initialization**:
   - Power on all equipment
   - Start the data acquisition PC
   - Run the system validation checks:
     ```bash
     ./gsr_rgbt_tools.sh test
     ```

### 2. Participant Preparation

1. **Consent and Instructions**:
   - Explain the procedure to the participant
   - Obtain informed consent
   - Provide instructions for the experimental tasks

2. **Sensor Attachment**:
   - Clean the participant's fingers with alcohol wipes
   - Attach GSR electrodes to the non-dominant hand (typically left)
   - Ensure secure contact and verify signal quality

3. **Positioning**:
   - Seat the participant comfortably
   - Position their dominant hand (for video recording) on the hand rest
   - Ensure both cameras have a clear view of the hand
   - Position the hand so the palm is facing up and fully visible

### 3. Recording Session

1. **Launch the Application**:
   ```bash
   ./gsr_rgbt_tools.sh run --component=app
   ```

2. **Start a New Session**:
   - Enter a unique Subject ID in the application
   - Click "Start Recording"
   - Verify that video feeds appear and GSR data is being plotted

3. **Conduct Experimental Protocol**:
   - Guide the participant through the experimental tasks
   - The standard protocol includes:
     - 5-minute baseline rest
     - 3-minute math stressor task
     - 1-minute inter-task rest
     - 5-minute guided relaxation
     - 1-minute inter-task rest
     - 3-minute emotional video stimulus
     - 2-minute final rest

4. **Stop Recording**:
   - Click "Stop Recording" in the application
   - Wait for confirmation that all data has been saved

## Troubleshooting

### Camera Issues

1. **RGB Camera Not Detected**:
   - Check USB connection
   - Try a different USB port
   - Verify camera ID in `config.py`
   - Check if another application is using the camera

2. **FLIR Camera Connection Failure**:
   - Verify Ethernet connection
   - Check IP configuration
   - Ensure GigE Vision drivers are installed
   - Use SpinView to test camera connection

3. **Low Frame Rate or Dropped Frames**:
   - Check CPU usage (Task Manager)
   - Reduce camera resolution or frame rate
   - Close unnecessary applications

### Shimmer GSR Sensor Issues

1. **Connection Failures**:
   - Check battery level
   - Verify COM port in `config.py`
   - Try reconnecting the device

2. **Poor Signal Quality**:
   - Check electrode placement and contact
   - Replace electrodes if necessary
   - Ensure participant's hands are clean and dry

3. **Bluetooth Disconnections**:
   - Move the PC closer to the Shimmer device
   - Check for interference from other devices
   - Use wired connection via dock instead of Bluetooth

### Synchronization Problems

1. **Timestamp Misalignment**:
   - Check system clock stability
   - Verify all devices are connected to the same PC
   - Run the synchronization test:
     ```bash
     ./gsr_rgbt_tools.sh test
     ```

### Software Issues

1. **Application Crashes**:
   - Check console for error messages
   - Verify all dependencies are installed
   - Try reinstalling the environment:
     ```bash
     ./gsr_rgbt_tools.sh setup --force
     ```

2. **Data Not Saving**:
   - Check disk space
   - Verify write permissions to the data directory

## Post-Recording Checks

After each recording session, perform these checks to ensure data quality:

### 1. Data File Verification

1. **Check Session Directory**:
   - Navigate to `data/recordings/Subject_[ID]_[TIMESTAMP]/`
   - Verify the following files exist:
     - `rgb_video.mp4`
     - `thermal_video.mp4`
     - `gsr_data.csv`
     - `rgb_timestamps.csv`
     - `thermal_timestamps.csv`

2. **File Size Check**:
   - Video files should be several MB or larger
   - CSV files should contain multiple rows

### 2. Data Quality Assessment

1. **Video Playback**:
   - Play both video files to ensure they captured correctly
   - Check for dropped frames or corruption

2. **GSR Data Inspection**:
   - Open `gsr_data.csv` in a spreadsheet program
   - Check for continuous data without large gaps
   - Verify that values are within expected ranges (typically 0.1-20 μS)

### 3. Quick Visualization

Run a basic visualization to confirm data quality:

```bash
python src/scripts/visualize_results.py --visualize-multi-roi --subject-id [SUBJECT_ID]
```

## Next Steps

After successfully setting up the equipment and collecting data:

1. **Data Processing**:
   - Process the raw data using the feature engineering pipeline
   - Extract features from the RGB and thermal videos
   - Align with GSR ground truth

2. **Model Training**:
   - Train machine learning models on the processed data
   - Use the provided scripts for model training and evaluation:
   ```bash
   ./gsr_rgbt_tools.sh run --component=pipeline
   ```

3. **Evaluation and Visualization**:
   - Evaluate model performance
   - Generate visualizations and reports

For more detailed information on the machine learning pipeline, refer to the project README.md and documentation.
