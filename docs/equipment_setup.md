# Equipment Setup Guide for GSR-RGBT Data Acquisition

## Table of Contents
1. [Introduction](#introduction)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Prerequisites](#software-prerequisites)
4. [Physical Setup and Connections](#physical-setup-and-connections)
5. [Synchronization Strategy](#synchronization-strategy)
6. [Data Acquisition Protocol](#data-acquisition-protocol)
7. [Troubleshooting](#troubleshooting)
8. [Post-Recording Checks](#post-recording-checks)
9. [Next Steps](#next-steps)

## Introduction

This guide provides detailed instructions for setting up a multimodal data acquisition system that combines RGB video, thermal imaging, and physiological sensing for contactless GSR prediction research. The setup is designed to capture synchronized data from multiple sources:

- RGB video of the participant's hand
- Thermal video of the same hand
- Ground-truth GSR measurements from the opposite hand
- Optional synchronization signals via Arduino

The system is designed for high-precision temporal alignment between all data streams, which is crucial for training machine learning models to predict GSR from video data.

## Hardware Requirements

### Cameras
1. **RGB Camera**:
   - Recommended: High-quality USB webcam (e.g., Logitech C920 or better) or machine vision camera
   - Minimum resolution: 1280×720 at 30fps
   - Mount: Adjustable tripod or camera mount
   - Connection: USB 3.0 port on the data acquisition PC

2. **FLIR Thermal Camera**:
   - Recommended: FLIR A65 (or similar GigE Vision thermal camera)
   - Resolution: 640×512 pixels
   - Spectral range: 7.5-13 μm
   - Frame rate: 30 Hz
   - Mount: Adjustable tripod or camera mount
   - Connection: Dedicated Ethernet port on the data acquisition PC

### Physiological Sensors
1. **Shimmer3 GSR+ Sensor**:
   - Components:
     - Shimmer3 GSR+ unit
     - GSR electrode leads
     - Disposable GSR electrodes (or reusable electrodes with conductive gel)
     - USB dock for charging and configuration
   - Connection: Bluetooth or serial via dock

2. **Optional Secondary GSR Sensor** (for validation):
   - PhysioKit GSR sensor or similar
   - Connection: As specified by manufacturer

### Synchronization Hardware
1. **Arduino Board** (for hardware synchronization):
   - Recommended: Arduino Uno or Nano
   - Components:
     - Bright white LED
     - 220-330 Ohm resistor
     - Breadboard and jumper wires
   - Connection: USB port on the data acquisition PC

### Computer and Accessories
1. **Data Acquisition PC**:
   - Recommended specs:
     - CPU: Intel i7/i9 or AMD Ryzen 7/9
     - RAM: 16GB minimum, 32GB recommended
     - Storage: SSD with at least 500GB free space (high-speed video capture generates large files)
     - GPU: NVIDIA GTX 1660 or better (for real-time processing)
     - Ports: Multiple USB 3.0, Ethernet port (Gigabit)
     - OS: Windows 10/11 (recommended for driver compatibility)

2. **Network Equipment** (for FLIR camera):
   - Gigabit Ethernet switch or direct connection
   - Cat6 Ethernet cable

3. **Mounting and Positioning Equipment**:
   - 2 adjustable camera tripods
   - Table and comfortable chair for participant
   - Adjustable lighting (diffuse, non-flickering)
   - Hand rest or support for stable positioning

## Software Prerequisites

### 1. Operating System
- Windows 10/11 is recommended for maximum compatibility with camera drivers and SDKs

### 2. Python Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gsr-rgbt-project.git
   cd gsr_rgbt_project
   ```

2. Set up the Python environment:
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Build Cython extensions
   python setup.py build_ext --inplace
   ```

### 3. Camera Software
1. **FLIR Camera Software**:
   - Download and install FLIR Spinnaker SDK from the [FLIR website](https://www.flir.com/products/spinnaker-sdk/)
   - During installation, select:
     - GigE Vision filter driver
     - USB3 Vision driver (if applicable)
     - SpinView (for camera testing and configuration)
     - Python bindings (PySpin)

2. **RGB Camera Drivers**:
   - Most USB webcams use standard UVC drivers included with Windows
   - For specialized cameras, install manufacturer-provided drivers

### 4. Shimmer Software
1. **Shimmer Connect/ConsensysPRO**:
   - Download from [Shimmer website](https://www.shimmersensing.com/products/consensys)
   - Install and use for:
     - Firmware updates
     - Initial sensor configuration
     - Testing sensor functionality

2. **pyshimmer Library**:
   - Already included in requirements.txt
   - Verify installation with:
     ```python
     python -c "import pyshimmer; print(pyshimmer.__version__)"
     ```

### 5. Arduino IDE (for synchronization)
1. Download and install from [Arduino website](https://www.arduino.cc/en/software)
2. Install any required libraries through the Library Manager

## Physical Setup and Connections

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

### Network Configuration for FLIR Camera

1. **Ethernet Connection**:
   - Connect the FLIR A65 to the dedicated Ethernet port on your PC
   - For best performance, use a direct connection rather than through a switch

2. **IP Configuration**:
   - Configure your PC's Ethernet adapter with a static IP address:
     - IP: 169.254.0.1
     - Subnet mask: 255.255.0.0
   - The FLIR camera typically uses an IP in the 169.254.x.x range
   - Use FLIR's IP Configuration Tool (included with Spinnaker SDK) to verify or change the camera's IP

3. **Firewall Settings**:
   - Add exceptions in Windows Firewall for:
     - Spinnaker SDK applications
     - Your Python application
     - GigE Vision streams (typically UDP)

### Shimmer GSR+ Sensor Setup

1. **Charging and Preparation**:
   - Fully charge the Shimmer device before use
   - Verify battery status using Shimmer Connect software

2. **Configuration**:
   - Connect Shimmer to PC via dock
   - Using Shimmer Connect:
     - Enable GSR sensor
     - Set sampling rate to 128Hz
     - Enable timestamp
     - Configure Bluetooth if using wireless connection
     - Set range: Auto-range or 56-220 kOhm (typical for GSR)

3. **Electrode Placement**:
   - Clean the participant's fingers with alcohol wipes
   - Attach electrodes to the palmar surface of:
     - Middle phalanx of index finger
     - Middle phalanx of middle finger
   - Both electrodes should be on the same hand (typically the non-dominant/left hand)
   - Ensure good contact with no air bubbles or gaps

4. **Connection Method**:
   - **Option A - Bluetooth**:
     - Pair the Shimmer device with your PC
     - Note the COM port assigned by Windows
     - Update `GSR_SENSOR_PORT` in `src/config.py` with this COM port

   - **Option B - Serial via Dock**:
     - Keep Shimmer connected to dock
     - Note the COM port
     - Update `GSR_SENSOR_PORT` in `src/config.py`

### Arduino Synchronization Setup (Optional but Recommended)

1. **Circuit Assembly**:
   - Connect the LED to digital pin 13 and ground through a 220-330 Ohm resistor
   - The circuit diagram is simple:
     ```
     Arduino Pin 13 ---> 220-330 Ohm Resistor ---> LED ---> GND
     ```

2. **Arduino Code**:
   - Upload the following sketch to the Arduino:
   ```cpp
   // Simple Arduino Sketch for Sync LED
   const int ledPin = 13; // LED connected to digital pin 13

   void setup() {
     pinMode(ledPin, OUTPUT);
     Serial.begin(9600); // For sending a signal to PC
   }

   void loop() {
     // Wait for a signal from PC to trigger, or trigger periodically
     // For manual trigger via Serial Monitor:
     if (Serial.available() > 0) {
       char command = Serial.read();
       if (command == 'T') { // 'T' for Trigger
         triggerLEDPulse();
       }
     }
     delay(100); // Check for serial command periodically
   }

   void triggerLEDPulse() {
     digitalWrite(ledPin, HIGH); // Turn LED on
     Serial.println("SYNC_PULSE_ON"); // Optional: send signal to PC
     delay(200);                  // LED on for 200ms (adjust as needed for camera capture)
     digitalWrite(ledPin, LOW);  // Turn LED off
     Serial.println("SYNC_PULSE_OFF"); // Optional
   }
   ```

3. **LED Positioning**:
   - Place the LED so it's visible in both the RGB and thermal camera views
   - Typically positioned at the edge of the frame where it won't interfere with the hand ROI
   - Secure it to prevent movement during the session

4. **Connection**:
   - Connect the Arduino to the PC via USB
   - Note the COM port assigned by Windows
   - Test the trigger by opening the Arduino Serial Monitor (set to 9600 baud) and sending 'T'

## Synchronization Strategy

Proper synchronization between data streams is critical for this research. The system employs multiple synchronization methods:

### 1. Software Timestamp Synchronization

1. **Host PC Timestamps**:
   - All capture threads use `time.perf_counter_ns()` for high-resolution timestamps
   - The DataLogger logs these timestamps alongside each frame and GSR sample
   - This provides a common time reference based on the host PC's clock

2. **Configuration in Code**:
   - Ensure the following code is properly connected in `src/main.py`:
   ```python
   # Connect signals with timestamps
   self.rgb_capture.frame_captured.connect(self.data_logger.log_rgb_frame)
   self.thermal_capture.frame_captured.connect(self.data_logger.log_thermal_frame)
   self.gsr_capture.gsr_data_point.connect(self.data_logger.log_gsr_data)
   ```

### 2. Hardware Synchronization via Arduino LED

1. **LED Flash Method**:
   - At the start of recording, trigger the Arduino LED with a 'T' command
   - The LED flash will be visible in both RGB and thermal video frames
   - This creates a common visual event in both video streams

2. **Manual Synchronization Event**:
   - In addition to the LED, perform a manual synchronization event:
     - A sharp hand clap in view of both cameras
     - A sudden hand movement
     - This provides a backup synchronization point

3. **Periodic Synchronization**:
   - For longer recordings, trigger the LED periodically (e.g., every 5 minutes)
   - This helps detect and correct for any clock drift

### 3. Post-Processing Alignment

1. **Timestamp File Usage**:
   - The system logs timestamps to CSV files:
     - `rgb_timestamps.csv`
     - `thermal_timestamps.csv`
     - `gsr_data.csv` (includes both system and Shimmer timestamps)

   - These files are used by `src/processing/feature_engineering.py` to align signals:
   ```python
   # In create_dataset_from_session function
   if rgb_timestamps_path.exists():
       rgb_timestamps_df = pd.read_csv(rgb_timestamps_path)
   ```

2. **Visual Verification**:
   - After recording, visually verify synchronization by:
     - Checking the LED flash appears in the same frame number in both videos
     - Confirming that hand movements align between RGB and thermal videos

## Data Acquisition Protocol

### 1. Pre-Session Setup

1. **Environment Preparation**:
   - Control room temperature (20-24°C)
   - Set up diffuse, consistent lighting
   - Prepare a comfortable chair and hand rest for the participant

2. **System Initialization**:
   - Power on all equipment
   - Start the data acquisition PC
   - Activate the Python virtual environment:
   ```bash
   .venv\Scripts\activate
   ```

3. **Camera Check**:
   - Verify RGB camera connection:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); print(f'Camera opened: {cap.isOpened()}'); cap.release()"
   ```
   - Check FLIR camera using SpinView application
   - Adjust camera positions and focus as needed

4. **Shimmer Sensor Check**:
   - Ensure the Shimmer device is charged
   - Verify connection via Shimmer Connect or:
   ```bash
   python -c "import pyshimmer; print('Shimmer library available')"
   ```

5. **Arduino Check** (if using):
   - Open Arduino Serial Monitor
   - Send 'T' command and verify LED flashes

6. **Configuration Update**:
   - Update `src/config.py` with the correct device IDs:
   ```python
   # Example configuration
   RGB_CAMERA_ID = 0  # Adjust based on your system
   THERMAL_CAMERA_ID = 1  # Or IP address if using GigE
   GSR_SENSOR_PORT = "COM3"  # Update with actual COM port
   RGB_SIMULATION_MODE = False
   THERMAL_SIMULATION_MODE = False
   GSR_SIMULATION_MODE = False
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
   python src/main.py
   ```

2. **Start a New Session**:
   - Enter a unique Subject ID in the application
   - Click "Start Recording"
   - Verify that video feeds appear and GSR data is being plotted

3. **Perform Synchronization**:
   - Immediately after starting recording:
     - Trigger the Arduino LED by sending 'T' via Serial Monitor
     - Ask the participant to perform a distinct hand movement (e.g., opening and closing the hand)
     - Note the exact time this occurs

4. **Conduct Experimental Protocol**:
   - Guide the participant through the experimental tasks
   - The standard protocol includes:
     - 5-minute baseline rest
     - 3-minute math stressor task
     - 1-minute inter-task rest
     - 5-minute guided relaxation
     - 1-minute inter-task rest
     - 3-minute emotional video stimulus
     - 2-minute final rest

5. **End-of-Session Synchronization**:
   - Before stopping recording:
     - Trigger the Arduino LED again
     - Ask for another distinct hand movement
     - This helps check for any clock drift

6. **Stop Recording**:
   - Click "Stop Recording" in the application
   - Wait for confirmation that all data has been saved

7. **Participant Debriefing**:
   - Remove the GSR electrodes
   - Thank the participant and answer any questions
   - Provide any post-session information

## Troubleshooting

### Camera Issues

1. **RGB Camera Not Detected**:
   - Check USB connection
   - Try a different USB port
   - Verify camera ID in `config.py`
   - Check if another application is using the camera
   - Solution: Close other applications or restart the PC

2. **FLIR Camera Connection Failure**:
   - Verify Ethernet connection
   - Check IP configuration
   - Ensure GigE Vision drivers are installed
   - Use SpinView to test camera connection
   - Solution: Reset camera power, reconfigure network settings

3. **Low Frame Rate or Dropped Frames**:
   - Check CPU usage (Task Manager)
   - Reduce camera resolution or frame rate
   - Close unnecessary applications
   - Solution: Optimize application settings or upgrade hardware

### Shimmer GSR Sensor Issues

1. **Connection Failures**:
   - Check battery level
   - Verify COM port in `config.py`
   - Try reconnecting the device
   - Solution: Restart Shimmer device, update firmware

2. **Poor Signal Quality**:
   - Check electrode placement and contact
   - Replace electrodes if necessary
   - Ensure participant's hands are clean and dry
   - Solution: Reapply electrodes with fresh conductive gel

3. **Bluetooth Disconnections**:
   - Move the PC closer to the Shimmer device
   - Check for interference from other devices
   - Solution: Use wired connection via dock instead of Bluetooth

### Synchronization Problems

1. **LED Not Visible in Videos**:
   - Check LED circuit connections
   - Verify Arduino code is running
   - Reposition LED to be in camera view
   - Solution: Test LED independently, then reintegrate

2. **Timestamp Misalignment**:
   - Check system clock stability
   - Verify all devices are connected to the same PC
   - Solution: Use post-processing alignment based on visual cues

### Software Issues

1. **Application Crashes**:
   - Check console for error messages
   - Verify all dependencies are installed
   - Solution: Debug based on error message or restart application

2. **Data Not Saving**:
   - Check disk space
   - Verify write permissions to the data directory
   - Solution: Free up disk space or run as administrator

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
   - Example check:
   ```bash
   dir data\recordings\Subject_*\*.*
   ```

### 2. Data Quality Assessment

1. **Video Playback**:
   - Play both video files to ensure they captured correctly
   - Check for dropped frames or corruption
   - Verify the synchronization LED is visible

2. **GSR Data Inspection**:
   - Open `gsr_data.csv` in a spreadsheet program
   - Check for continuous data without large gaps
   - Verify that values are within expected ranges (typically 0.1-20 μS)

3. **Timestamp Verification**:
   - Check that timestamp files contain entries for each frame
   - Verify the timestamps are monotonically increasing
   - Example Python check:
   ```python
   import pandas as pd
   df = pd.read_csv("data/recordings/Subject_ID_TIMESTAMP/rgb_timestamps.csv")
   print(f"Number of frames: {len(df)}")
   print(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()) / 1e9} seconds")
   ```

### 3. Quick Visualization

Run a basic visualization to confirm data quality:

```bash
python src/scripts/visualize_results.py --visualize-multi-roi --subject-id [SUBJECT_ID]
```

This will generate visualizations of the Multi-ROI detection on sample frames and basic plots of the GSR data.

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
   python src/scripts/train_model.py --model-type dual_stream_cnn_lstm --config-path configs/models/dual_stream_cnn_lstm.yaml
   ```

3. **Evaluation and Visualization**:
   - Evaluate model performance
   - Generate visualizations and reports:
   ```bash
   python src/scripts/visualize_results.py --plot-predictions --model-comparison
   ```

4. **Iterative Improvement**:
   - Refine the experimental protocol based on initial results
   - Optimize camera positions and settings
   - Improve synchronization methods if needed

For more detailed information on the machine learning pipeline, refer to the project README.md and documentation.
