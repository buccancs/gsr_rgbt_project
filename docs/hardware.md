# GSR-RGBT Project Hardware Documentation

## Introduction

This document provides detailed information about all hardware components used in the GSR-RGBT project, including specifications, setup instructions, dependencies, advantages and limitations, and the rationale behind hardware selection decisions.

## Hardware Components Overview

The GSR-RGBT project uses the following key hardware components:

1. **RGB Camera**: For capturing visible light video of the participant's hand
2. **FLIR Thermal Camera**: For capturing thermal video of the same hand
3. **Shimmer3 GSR+ Sensor**: For measuring ground-truth Galvanic Skin Response
4. **Arduino Board**: For hardware synchronization (optional but recommended)
5. **Data Acquisition PC**: For running the software and storing the data
6. **Mounting and Positioning Equipment**: For ensuring stable and consistent data capture

## RGB Camera

### Specifications

- **Recommended Model**: Logitech C920 or better
- **Resolution**: 1280×720 at 30fps minimum (1920×1080 preferred)
- **Connection**: USB 3.0
- **Field of View**: 78° diagonal
- **Focus**: Autofocus with manual override capability
- **Exposure Control**: Automatic with manual override capability

### Setup and Configuration

1. **Physical Setup**:
   - Mount on a stable tripod approximately 50-70cm from the participant's hand
   - Position to capture a clear view of the palm and fingers
   - Ensure consistent, diffuse lighting to avoid shadows and reflections

2. **Software Configuration**:
   - Set resolution to 1280×720 or 1920×1080
   - Set frame rate to 30fps
   - Use manual focus, focused on the hand position
   - Use manual exposure to prevent auto-adjustments during recording

### Advantages and Limitations

**Advantages**:
- Widely available and affordable
- Easy to set up and use
- Good image quality in controlled lighting conditions
- USB connection simplifies integration

**Limitations**:
- Performance degrades in poor lighting conditions
- Limited dynamic range compared to professional cameras
- USB bandwidth limitations may affect frame rate at higher resolutions
- No hardware synchronization capabilities

### Selection Rationale

The Logitech C920 (or similar) webcam was selected for its balance of cost, quality, and ease of use. While professional machine vision cameras offer better performance and synchronization capabilities, consumer webcams provide sufficient quality for this application at a fraction of the cost. The USB 3.0 connection ensures adequate bandwidth for 30fps capture at HD resolution.

## FLIR Thermal Camera

### Specifications

- **Recommended Model**: FLIR A65
- **Resolution**: 640×512 pixels
- **Spectral Range**: 7.5-13 μm
- **Frame Rate**: 30 Hz
- **Thermal Sensitivity**: < 50 mK
- **Temperature Range**: -25°C to +135°C
- **Connection**: Gigabit Ethernet (GigE Vision compatible)
- **Power**: Power over Ethernet (PoE) or 12/24 V DC

### Setup and Configuration

1. **Network Configuration**:
   - Connect the FLIR A65 to a dedicated Ethernet port on your PC
   - Configure your PC's Ethernet adapter with a static IP address:
     - IP: 169.254.0.1
     - Subnet mask: 255.255.0.0
   - The FLIR camera typically uses an IP in the 169.254.x.x range
   - Use FLIR's IP Configuration Tool to verify or change the camera's IP

2. **Software Configuration**:
   - Configure using SpinView:
     - Set acquisition mode to continuous
     - Frame rate: 30Hz
     - Enable timestamp
     - Set emissivity to 0.98 (appropriate for human skin)
     - Temperature range: Typically 20-40°C for human subjects

3. **Physical Setup**:
   - Mount on a stable tripod approximately 50-70cm from the participant's hand
   - Position as close as possible to the RGB camera (side by side, ~10cm apart)
   - Ensure both cameras capture the same field of view

### Advantages and Limitations

**Advantages**:
- High thermal sensitivity allows detection of subtle temperature changes
- GigE Vision interface provides reliable, high-bandwidth connection
- Good spatial resolution for hand thermal imaging
- Provides unique physiological information not available in RGB video

**Limitations**:
- Expensive compared to consumer thermal cameras
- Requires dedicated network configuration
- Larger and heavier than RGB camera, making alignment more challenging
- Requires specialized software and drivers

### Selection Rationale

The FLIR A65 was selected for its high thermal sensitivity, good spatial resolution, and industry-standard GigE Vision interface. While consumer thermal cameras (like FLIR ONE) are more affordable, they lack the resolution, sensitivity, and frame rate needed for this research. The A65's ability to detect temperature differences as small as 50 mK is crucial for capturing the subtle thermal changes associated with sympathetic nervous system activation.

## Shimmer3 GSR+ Sensor

### Specifications

- **GSR Channel**:
  - Sampling Rate: 128 Hz
  - Format: 16 bits, signed
  - Units: kOhms
  - Range: 10-4700 kOhms (adjustable)

- **Optional PPG Channel**:
  - Sampling Rate: 128 Hz
  - Format: 16 bits, signed
  - Units: mV

- **Power**: Rechargeable Li-ion battery (450 mAh)
- **Connection**: Bluetooth or USB (via dock)
- **Size**: 51mm × 34mm × 14mm
- **Weight**: 22g

### Setup and Configuration

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

### Advantages and Limitations

**Advantages**:
- Purpose-built for GSR measurement with high accuracy
- Small and lightweight, minimizing participant discomfort
- Wireless operation allows natural hand positioning
- Open-source SDK facilitates integration
- Can measure multiple physiological signals simultaneously

**Limitations**:
- Battery life limited to ~8 hours
- Bluetooth connection can be unreliable in some environments
- Requires proper electrode placement for accurate readings
- More expensive than simple GSR sensors

### Selection Rationale

The Shimmer3 GSR+ was selected for its combination of accuracy, portability, and open SDK. Unlike consumer GSR sensors, it provides research-grade measurements with precise timing, which is crucial for this project. The ability to measure both GSR and PPG (photoplethysmography) allows for potential future expansion to include heart rate data. The open SDK and extensive documentation make it easier to integrate with custom software.

## Arduino Board (for Synchronization)

### Specifications

- **Recommended Model**: Arduino Uno or Nano
- **Components**:
  - Bright white LED
  - 220-330 Ohm resistor
  - Breadboard and jumper wires
- **Connection**: USB port on the data acquisition PC

### Setup and Configuration

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

### Advantages and Limitations

**Advantages**:
- Provides a visual synchronization point visible in both RGB and thermal video
- Simple and inexpensive to implement
- Can be triggered programmatically or manually
- Allows for precise temporal alignment of video streams

**Limitations**:
- Requires manual triggering at the start and end of recording
- LED must be positioned carefully to be visible in both camera views
- Does not directly synchronize with GSR data (requires software timestamp correlation)

### Selection Rationale

The Arduino-based synchronization system was selected for its simplicity, low cost, and effectiveness. While more sophisticated hardware synchronization systems exist (e.g., professional trigger boxes), the Arduino solution provides sufficient accuracy for this application at a fraction of the cost. The ability to trigger the LED both programmatically and manually provides flexibility in the experimental protocol.

## Data Acquisition PC

### Specifications

- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: SSD with at least 500GB free space
- **GPU**: NVIDIA GTX 1660 or better (for real-time processing)
- **Ports**: Multiple USB 3.0, Ethernet port (Gigabit)
- **OS**: Windows 10/11 (recommended for driver compatibility)

### Setup and Configuration

1. **Software Installation**:
   - Install Python and required libraries
   - Install FLIR Spinnaker SDK for thermal camera
   - Install Shimmer Connect for GSR sensor
   - Install Arduino IDE for synchronization hardware

2. **Network Configuration**:
   - Configure Ethernet adapter for thermal camera
   - Ensure firewall allows necessary connections

3. **Power Management**:
   - Disable sleep/hibernation during data collection
   - Set power plan to "High performance"
   - Disable automatic updates during experiments

### Advantages and Limitations

**Advantages**:
- Powerful CPU and GPU enable real-time processing and visualization
- Multiple high-speed ports support simultaneous device connections
- SSD storage provides fast write speeds for high-bandwidth data
- Windows OS provides good compatibility with device drivers

**Limitations**:
- High-end hardware increases cost
- Windows OS may introduce timing variability compared to real-time operating systems
- Power management and background processes can affect performance

### Selection Rationale

A high-performance PC was selected to handle the demanding requirements of simultaneous video capture, processing, and storage. The real-time visualization and processing of multiple data streams require significant computational resources. Windows was chosen as the operating system due to better driver support for the hardware components, particularly the FLIR camera and Shimmer sensor.

## Mounting and Positioning Equipment

### Specifications

- **Camera Tripods**: 2 adjustable tripods with 1/4" mounting screws
- **Hand Rest**: Comfortable, stable surface for participant's hand
- **Lighting**: Diffuse, non-flickering light sources
- **Room Setup**: Temperature-controlled environment (20-24°C)

### Setup and Configuration

1. **Camera Positioning**:
   - Position both cameras on tripods approximately 50-70cm from where the participant's hand will be placed
   - Align the RGB and thermal cameras as close together as possible (side by side, ~10cm apart)
   - Angle both cameras to capture the same field of view
   - Ensure the cameras have an unobstructed view of the participant's hand

2. **Hand Rest Setup**:
   - Place the hand rest at a comfortable height for the participant
   - Position it so the participant's hand is clearly visible to both cameras
   - Ensure the surface is non-reflective to avoid thermal artifacts

3. **Lighting Setup**:
   - Use diffuse lighting to minimize shadows and reflections
   - Position lights to evenly illuminate the hand
   - Avoid direct lighting on the thermal camera to prevent lens heating

### Advantages and Limitations

**Advantages**:
- Stable camera mounting reduces motion artifacts
- Consistent hand positioning improves data quality
- Controlled environment reduces variability in measurements

**Limitations**:
- Fixed setup limits natural hand movement
- Room temperature fluctuations can affect thermal measurements
- Perfect alignment of RGB and thermal cameras is challenging

### Selection Rationale

The mounting and positioning equipment was selected to provide a stable, consistent environment for data collection. Tripods offer flexibility in camera positioning while maintaining stability. A dedicated hand rest ensures that participants can maintain a comfortable hand position throughout the recording session, reducing motion artifacts. Controlled lighting and temperature are essential for consistent thermal measurements.

## Conclusion

The hardware components selected for the GSR-RGBT project represent a balance between research-grade quality, cost, and ease of use. Each component was chosen based on its specific capabilities and how it integrates with the overall system. While there are limitations to each component, the combined system provides the necessary capabilities for contactless GSR prediction research.

The modular nature of the hardware setup allows for future upgrades or replacements of individual components as technology advances or research requirements change. The detailed specifications and setup instructions provided in this document should enable researchers to replicate the hardware setup and understand the rationale behind the hardware selection decisions.