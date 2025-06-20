# GSR-RGBT Project Technical Details

## Introduction

This document provides a comprehensive technical guide for the GSR-RGBT (Galvanic Skin Response - RGB-Thermal) project, covering hardware setup, device integration, data synchronization, and system validation. It consolidates information from various technical documents to provide a single reference point for understanding the technical aspects of the project.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Physical Setup and Connections](#physical-setup-and-connections)
3. [Device Integration](#device-integration)
4. [Data Synchronization](#data-synchronization)
5. [Data Acquisition Protocol](#data-acquisition-protocol)
6. [Troubleshooting](#troubleshooting)
7. [System Validation](#system-validation)
8. [References](#references)

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

## Device Integration

### FLIR A65 Integration

1. **Software Installation**:
   - Install FLIR Atlas SDK for development
   - Install PySpin library for Python integration
   - Verify connection using FLIR tools or a simple Python script

2. **Python Integration**:
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
   ```

### Shimmer3 GSR+ Integration

1. **Specifications**:
   - **GSR Channel**:
     - Sampling Rate: 128 Hz
     - Format: 16 bits, signed
     - Units: kOhms
   - **PPG Channel** (if used):
     - Sampling Rate: 128 Hz
     - Format: 16 bits, signed
     - Units: mV

2. **Python Integration**:
   ```python
   # Initialize Shimmer3 GSR+
   def initialize_gsr_sensor(com_port):
       from pyshimmer import Shimmer
       shimmer = Shimmer(com_port)
       shimmer.connect()
       shimmer.set_sampling_rate(128.0)  # 128 Hz
       shimmer.enable_sensor('GSR')
       shimmer.start_streaming()
       return shimmer
   ```

3. **Data Format**:
   The Shimmer3 GSR+ unit outputs data in a tab-separated CSV file with the following columns:
   - Timestamp (yyyy/mm/dd hh:mm:ss.000)
   - Accelerometer X, Y, Z (m/s^2)
   - GSR (kOhms)
   - PPG-to-HR (BPM)
   - PPG (mV)

4. **Integration Changes**:
   - Updated `data_loader.py` to handle the Shimmer data format
   - Enhanced `preprocessing.py` to process both GSR and PPG signals
   - Updated `config.py` to set the GSR sampling rate to 128 Hz

### RGB Camera Integration

1. **Python Integration**:
   ```python
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

## Data Synchronization

### Synchronization Challenges

Each device in our setup has different characteristics that make synchronization challenging:

| Device | Sampling Rate | Clock Source | Timestamp Precision | Typical Latency |
|--------|---------------|--------------|---------------------|-----------------|
| FLIR A65 | 30 Hz | Internal | Millisecond | 10-50 ms |
| Shimmer3 GSR+ | 128 Hz | Internal | Microsecond | 5-20 ms |
| Logitech Kyro | 30 Hz | Internal | Millisecond | 30-100 ms |

Additional challenges include:
- **Clock Drift**: Device internal clocks may drift relative to each other over time
- **Variable Latency**: Network and processing delays can vary
- **Different Sampling Rates**: Devices sample at different frequencies
- **Jitter**: Inconsistent intervals between samples

### Synchronization Approach

The GSR-RGBT project uses a custom synchronization approach based on a centralized timestamp authority. This approach ensures that all data streams (RGB video, thermal video, and GSR data) are properly synchronized, which is critical for accurate analysis.

#### Components

1. **TimestampThread**: A high-priority thread that emits timestamps at a fast, consistent rate (200Hz by default). This thread serves as a centralized timestamp authority for all data capture components.

2. **Capture Threads**: Separate threads for RGB video, thermal video, and GSR data capture. Each thread captures data at its own rate and associates each data point with the latest timestamp from the TimestampThread.

3. **DataLogger**: Writes the captured data to disk, along with the associated timestamps. This allows for precise synchronization during later analysis.

#### How It Works

1. The TimestampThread generates high-resolution timestamps at 200Hz (much faster than any capture rate).
2. When a frame or data point is captured, it's associated with the latest timestamp from the TimestampThread.
3. The DataLogger writes the frames/data points to files along with their timestamps.
4. During analysis, the timestamps can be used to align the different data streams.

This approach ensures that all data streams share a common time reference, even if they are captured at different rates or with different latencies.

### Hardware Synchronization Methods

For the most precise synchronization, hardware-based methods can be used:

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

### Post-Processing Alignment

After data collection, post-processing can be used to further refine the synchronization:

1. **Cross-Correlation Method**:
   ```python
   import numpy as np
   from scipy import signal

   def synchronize_signals(signal1, signal2, sampling_rate1, sampling_rate2):
       """
       Find the time offset between two signals using cross-correlation.
       
       Args:
           signal1: First signal array
           signal2: Second signal array
           sampling_rate1: Sampling rate of first signal (Hz)
           sampling_rate2: Sampling rate of second signal (Hz)
           
       Returns:
           time_offset: Time offset in seconds (positive if signal2 is delayed)
       """
       # Resample signals to the same rate if necessary
       if sampling_rate1 != sampling_rate2:
           # Resample signal2 to match signal1's rate
           new_length = int(len(signal2) * sampling_rate1 / sampling_rate2)
           signal2 = signal.resample(signal2, new_length)
       
       # Compute cross-correlation
       correlation = signal.correlate(signal1, signal2, mode='full')
       
       # Find the lag with maximum correlation
       lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
       lag = lags[np.argmax(correlation)]
       
       # Convert lag to time offset
       time_offset = lag / sampling_rate1
       
       return time_offset
   ```

2. **Event-Based Synchronization**:
   - Identify distinct events (e.g., LED flashes, hand movements) in all data streams
   - Use these events as synchronization points
   - Align the data streams based on these events

### Testing Synchronization

The project includes a test script that verifies the data synchronization mechanism is working properly:

```bash
python src/scripts/test_synchronization.py
```

This script:
1. First runs the system validation check to ensure all devices are working properly.
2. Initializes the TimestampThread, capture threads, and DataLogger.
3. Captures data for a short duration (5 seconds by default).
4. Analyzes the collected data to verify synchronization.
5. Creates a visualization plot of the timestamps.
6. Reports success/failure with detailed information.

## Data Acquisition Protocol

### Pre-Session Setup

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

### Participant Preparation

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

### Recording Session

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

## System Validation

### Running System Validation

To verify that all components of the system are working correctly, run the system validation script:

```bash
python src/scripts/check_system.py
```

Alternatively, if you have Make installed, you can use:

```bash
make test
```

This script checks:
1. Python environment and required packages
2. Camera connections
3. GSR sensor connection
4. File system access
5. GPU availability (if applicable)

### Post-Recording Checks

After each recording session, perform these checks to ensure data quality:

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

3. **Data Quality Assessment**:
   - Play both video files to ensure they captured correctly
   - Check for dropped frames or corruption
   - Verify the synchronization LED is visible
   - Open `gsr_data.csv` in a spreadsheet program
   - Check for continuous data without large gaps
   - Verify that values are within expected ranges (typically 0.1-20 μS)

4. **Timestamp Verification**:
   - Check that timestamp files contain entries for each frame
   - Verify the timestamps are monotonically increasing

5. **Quick Visualization**:
   ```bash
   python src/scripts/visualize_results.py --visualize-multi-roi --subject-id [SUBJECT_ID]
   ```
   This will generate visualizations of the Multi-ROI detection on sample frames and basic plots of the GSR data.

## References

1. FLIR A65 User Manual: [FLIR Systems Documentation](https://www.flir.com/products/a65/)
2. Shimmer3 GSR+ User Guide: [Shimmer Documentation](https://shimmersensing.com/product/shimmer3-gsr-unit/)
3. Logitech Kyro Documentation: [Logitech Support](https://www.logitech.com/en-us/products/webcams/)
4. PySpin Documentation: [FLIR Systems SDK](https://www.flir.com/products/spinnaker-sdk/)
5. PyShimmer GitHub Repository: [PyShimmer](https://github.com/ShimmerResearch/pyshimmer)
6. Mills, D. L. (2006). Network Time Protocol Version 4 Reference and Implementation Guide. University of Delaware.
7. Olson, E. (2011). AprilTag: A robust and flexible visual fiducial system. In 2011 IEEE International Conference on Robotics and Automation (pp. 3400-3407).