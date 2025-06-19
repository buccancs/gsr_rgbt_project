# Timestamp Synchronization Guide

This document provides detailed information on methods for synchronizing timestamps between different devices in a multi-sensor data collection setup, specifically focusing on the integration of FLIR A65 thermal camera, Shimmer3 GSR+ sensor, and Logitech Kyro webcam.

## The Importance of Timestamp Synchronization

In multi-sensor setups, accurate timestamp synchronization is critical for:

1. **Data Alignment**: Ensuring that measurements from different sensors correspond to the same physical events
2. **Temporal Analysis**: Enabling accurate analysis of time-dependent relationships between signals
3. **Causality Determination**: Establishing cause-and-effect relationships between observed phenomena
4. **Feature Extraction**: Creating meaningful features that combine data from multiple sensors

Without proper synchronization, temporal misalignment can lead to incorrect conclusions about relationships between physiological signals (GSR) and visual data (thermal and RGB video).

## Synchronization Challenges

Each device in our setup has different characteristics that make synchronization challenging:

| Device | Sampling Rate | Clock Source | Timestamp Precision | Typical Latency |
|--------|---------------|--------------|---------------------|-----------------|
| FLIR A65 | 30 Hz | Internal | Millisecond | 10-50 ms |
| Shimmer3 GSR+ | 32 Hz | Internal | Microsecond | 5-20 ms |
| Logitech Kyro | 30 Hz | Internal | Millisecond | 30-100 ms |

Additional challenges include:

- **Clock Drift**: Device internal clocks may drift relative to each other over time
- **Variable Latency**: Network and processing delays can vary
- **Different Sampling Rates**: Devices sample at different frequencies
- **Jitter**: Inconsistent intervals between samples

## Synchronization Methods

### 1. Software-Based Synchronization

#### 1.1 Network Time Protocol (NTP)

NTP synchronizes computer clocks across a network, providing millisecond-level accuracy.

**Implementation Steps:**

1. **Configure NTP on the host computer:**

   ```bash
   # On Windows
   w32tm /config /syncfromflags:manual /manualpeerlist:"pool.ntp.org"
   w32tm /config /update
   
   # On Linux
   sudo apt-get install ntp
   sudo nano /etc/ntp.conf  # Add "server pool.ntp.org" if not present
   sudo systemctl restart ntp
   ```

2. **Verify synchronization:**

   ```bash
   # On Windows
   w32tm /query /status
   
   # On Linux
   ntpq -p
   ```

3. **In your Python code:**

   ```python
   import time
   import datetime
   
   def get_ntp_synchronized_timestamp():
       """Get current timestamp from NTP-synchronized system clock"""
       return time.time()
   
   def timestamp_to_readable(timestamp):
       """Convert Unix timestamp to human-readable format"""
       return datetime.datetime.fromtimestamp(timestamp).strftime(
           '%Y-%m-%d %H:%M:%S.%f'
       )
   ```

**Advantages:**
- Easy to implement
- No additional hardware required
- Sufficient for many applications

**Limitations:**
- Typical accuracy of 1-10 ms, which may not be sufficient for high-precision applications
- Requires network connectivity
- Subject to network jitter

#### 1.2 Common Time Reference

Use the host computer's clock as a common reference and timestamp data as it arrives.

**Implementation:**

```python
import time
import threading
import queue

class TimestampedDataCollector:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.running = False
    
    def start_collection(self):
        self.running = True
        # Start device threads
        threading.Thread(target=self._collect_gsr_data).start()
        threading.Thread(target=self._collect_thermal_data).start()
        threading.Thread(target=self._collect_rgb_data).start()
    
    def _collect_gsr_data(self):
        # Example GSR data collection
        while self.running:
            gsr_data = shimmer_device.read_data()  # Placeholder
            timestamp = time.time()
            self.data_queue.put({
                'type': 'gsr',
                'timestamp': timestamp,
                'data': gsr_data
            })
            time.sleep(1/32)  # Approximate 32 Hz
    
    # Similar methods for thermal and RGB data...
    
    def process_data(self):
        while not self.data_queue.empty():
            data_point = self.data_queue.get()
            # Process timestamped data...
```

**Advantages:**
- Simple implementation
- Works with any device that can be accessed from the host computer

**Limitations:**
- Does not account for variable latency between the device and when data is received
- Accuracy limited by thread scheduling and processing delays

### 2. Hardware-Based Synchronization

#### 2.1 External Trigger Signal

Use an external device (e.g., Arduino) to generate a common trigger signal for all devices.

**Hardware Setup:**

1. Configure an Arduino to generate a periodic pulse on digital pins
2. Connect these pins to the trigger inputs of each device (if available)
3. For devices without hardware trigger inputs, use visual or other sensory markers

**Arduino Code Example:**

```cpp
const int triggerPin1 = 2;  // For FLIR camera
const int triggerPin2 = 3;  // For Shimmer GSR
const int triggerPin3 = 4;  // For visual marker (LED)
const unsigned long triggerInterval = 10000;  // 10 seconds

void setup() {
  pinMode(triggerPin1, OUTPUT);
  pinMode(triggerPin2, OUTPUT);
  pinMode(triggerPin3, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // Generate synchronization pulse
  digitalWrite(triggerPin1, HIGH);
  digitalWrite(triggerPin2, HIGH);
  digitalWrite(triggerPin3, HIGH);
  delay(50);  // 50ms pulse
  digitalWrite(triggerPin1, LOW);
  digitalWrite(triggerPin2, LOW);
  digitalWrite(triggerPin3, LOW);
  
  delay(triggerInterval - 50);  // Wait for next trigger
}
```

**Python Implementation:**

```python
def detect_trigger_in_thermal_frame(frame):
    """Detect LED trigger marker in thermal frame"""
    # Implementation depends on the specific setup
    # Could look for a sudden brightness change in a specific region
    return is_trigger_detected, frame_index

def detect_trigger_in_rgb_frame(frame):
    """Detect LED trigger marker in RGB frame"""
    # Similar implementation for RGB frames
    return is_trigger_detected, frame_index

def align_data_streams(gsr_data, thermal_frames, rgb_frames):
    """Align data streams based on detected trigger points"""
    gsr_triggers = [i for i, d in enumerate(gsr_data) if d['trigger_flag']]
    thermal_triggers = [detect_trigger_in_thermal_frame(f)[1] for f in thermal_frames]
    rgb_triggers = [detect_trigger_in_rgb_frame(f)[1] for f in rgb_frames]
    
    # Use these trigger points to align the data streams
    # ...
```

**Advantages:**
- High precision synchronization (sub-millisecond possible)
- Not affected by clock drift
- Works even with closed systems that don't expose their internal clock

**Limitations:**
- Requires additional hardware
- May require device-specific integration
- More complex setup

#### 2.2 GPS Time Synchronization

For outdoor applications or when extreme precision is needed, GPS receivers can provide a common time reference.

**Implementation:**

1. Connect GPS receivers to each device or to the host computer
2. Use the GPS PPS (Pulse Per Second) signal for precise timing
3. Timestamp data using the GPS time reference

**Advantages:**
- Very high precision (nanosecond level)
- Not affected by local network issues
- Global time reference

**Limitations:**
- Requires GPS hardware
- Only works with line-of-sight to GPS satellites (outdoor use)
- More expensive and complex

### 3. Post-Processing Synchronization

#### 3.1 Cross-Correlation Method

This method finds the time offset between signals by analyzing their correlation.

**Implementation:**

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

# Example usage
gsr_signal = np.array([...])  # GSR signal
thermal_feature = np.array([...])  # Feature extracted from thermal video
time_offset = synchronize_signals(gsr_signal, thermal_feature, 32, 30)
print(f"Thermal video is delayed by {time_offset:.3f} seconds relative to GSR")
```

**Advantages:**
- Can be applied after data collection
- Works with any type of signal that has correlated features
- No special hardware required

**Limitations:**
- Requires identifiable correlated features in the signals
- May not work well with signals that have weak correlation
- Less precise than hardware-based methods

#### 3.2 Event-Based Synchronization

Identify distinct events that are visible in all data streams and use them as synchronization points.

**Implementation:**

```python
def identify_events(data_stream, event_detector_function):
    """
    Identify events in a data stream using the provided detector function.
    
    Args:
        data_stream: Array of data points
        event_detector_function: Function that returns True when an event is detected
        
    Returns:
        event_indices: Indices where events were detected
    """
    event_indices = []
    for i, data_point in enumerate(data_stream):
        if event_detector_function(data_point):
            event_indices.append(i)
    return event_indices

def align_streams_using_events(stream1, stream2, events1, events2, rate1, rate2):
    """
    Align two data streams based on detected events.
    
    Args:
        stream1, stream2: Data streams to align
        events1, events2: Indices of events in each stream
        rate1, rate2: Sampling rates of each stream
        
    Returns:
        aligned_stream2: Stream2 aligned to stream1's timeline
    """
    if len(events1) == 0 or len(events2) == 0:
        raise ValueError("No events detected in one or both streams")
    
    # Calculate time points of events in seconds
    time_events1 = [i / rate1 for i in events1]
    time_events2 = [i / rate2 for i in events2]
    
    # Match events and calculate offsets
    offsets = []
    for t1 in time_events1:
        # Find closest event in stream2
        closest_idx = np.argmin([abs(t1 - t2) for t2 in time_events2])
        offsets.append(t1 - time_events2[closest_idx])
    
    # Use median offset for robustness
    median_offset = np.median(offsets)
    
    # Create timeline for stream2 aligned to stream1
    timeline2 = np.arange(len(stream2)) / rate2 + median_offset
    
    # Interpolate stream2 to stream1's timeline
    timeline1 = np.arange(len(stream1)) / rate1
    aligned_stream2 = np.interp(timeline1, timeline2, stream2)
    
    return aligned_stream2
```

**Advantages:**
- Can be very accurate if events are well-defined
- Works with diverse data types
- Can be automated for batch processing

**Limitations:**
- Requires identifiable events in all data streams
- May introduce errors if events are misidentified
- Accuracy depends on event detection precision

## Practical Implementation for Our Setup

For the FLIR A65, Shimmer3 GSR+, and Logitech Kyro setup, we recommend a hybrid approach:

### Recommended Approach

1. **Initial Synchronization**:
   - Use NTP to synchronize the host computer's clock
   - Configure all devices to use the host computer's time when possible

2. **Hardware Synchronization**:
   - Set up an Arduino to generate a periodic LED flash visible to both cameras
   - Record the LED flash events in the GSR data stream (e.g., by pressing a marker button)

3. **Data Collection**:
   - Timestamp all data with the host computer's time as it arrives
   - Record the LED flash events in all data streams

4. **Post-Processing**:
   - Use the LED flash events to calculate precise offsets between data streams
   - Apply cross-correlation to fine-tune synchronization
   - Resample all signals to a common timeline

### Implementation Example

```python
import time
import numpy as np
import cv2
from scipy import signal

# 1. Data Collection with timestamps
def collect_synchronized_data(duration_seconds=60):
    # Initialize devices
    thermal_camera = initialize_thermal_camera()
    gsr_sensor = initialize_gsr_sensor("COM3")
    rgb_camera = initialize_webcam(0)
    
    # Prepare data structures
    thermal_data = []
    gsr_data = []
    rgb_data = []
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Collect data with timestamps
    while time.time() < end_time:
        # Get current time
        current_time = time.time()
        
        # Collect thermal frame
        thermal_frame = capture_thermal_frame(thermal_camera)
        if thermal_frame is not None:
            thermal_data.append({
                'timestamp': current_time,
                'frame': thermal_frame
            })
        
        # Collect GSR data
        gsr_value = read_gsr_value(gsr_sensor)
        if gsr_value is not None:
            gsr_data.append({
                'timestamp': current_time,
                'value': gsr_value
            })
        
        # Collect RGB frame
        ret, rgb_frame = rgb_camera.read()
        if ret:
            rgb_data.append({
                'timestamp': current_time,
                'frame': rgb_frame
            })
    
    # Clean up
    thermal_camera.release()
    gsr_sensor.disconnect()
    rgb_camera.release()
    
    return thermal_data, gsr_data, rgb_data

# 2. Post-processing synchronization
def synchronize_data_streams(thermal_data, gsr_data, rgb_data):
    # Detect LED flash events in each stream
    thermal_events = detect_events_in_thermal(thermal_data)
    gsr_events = detect_events_in_gsr(gsr_data)
    rgb_events = detect_events_in_rgb(rgb_data)
    
    # Calculate offsets
    thermal_to_gsr_offset = calculate_offset(thermal_events, gsr_events)
    rgb_to_gsr_offset = calculate_offset(rgb_events, gsr_events)
    
    # Apply offsets to timestamps
    for item in thermal_data:
        item['timestamp'] += thermal_to_gsr_offset
    
    for item in rgb_data:
        item['timestamp'] += rgb_to_gsr_offset
    
    # Create a common timeline
    min_time = min(
        min(item['timestamp'] for item in thermal_data),
        min(item['timestamp'] for item in gsr_data),
        min(item['timestamp'] for item in rgb_data)
    )
    
    max_time = max(
        max(item['timestamp'] for item in thermal_data),
        max(item['timestamp'] for item in gsr_data),
        max(item['timestamp'] for item in rgb_data)
    )
    
    # Resample to a common timeline (e.g., 100 Hz)
    sampling_rate = 100  # Hz
    timeline = np.arange(min_time, max_time, 1/sampling_rate)
    
    # Create synchronized dataset
    synchronized_data = {
        'timeline': timeline,
        'thermal': interpolate_data(thermal_data, timeline),
        'gsr': interpolate_data(gsr_data, timeline),
        'rgb': interpolate_data(rgb_data, timeline)
    }
    
    return synchronized_data
```

## Validation and Quality Assessment

To ensure your synchronization is accurate:

1. **Visual Inspection**:
   - Plot synchronized signals and look for expected temporal relationships
   - Check that known events align across all data streams

2. **Quantitative Validation**:
   - Calculate the standard deviation of offsets between multiple synchronization events
   - Aim for standard deviation < 1/sampling_rate for good synchronization

3. **Controlled Experiments**:
   - Create controlled events that should appear simultaneously in all data streams
   - Measure the time difference between event detections after synchronization

4. **Continuous Monitoring**:
   - For long recordings, include periodic synchronization events
   - Check for clock drift by comparing early vs. late synchronization points

## References

1. Bannach, D., Amft, O., & Lukowicz, P. (2009). Automatic event-based synchronization of multimodal data streams from wearable and ambient sensors. In European Conference on Smart Sensing and Context (pp. 135-148).

2. Weiss, A., Hirshberg, D., & Black, M. J. (2011). Home 3D body scans from noisy image and range data. In 2011 International Conference on Computer Vision (pp. 1951-1958).

3. Chadwell, A., Kenney, L., Thies, S., Galpin, A., & Head, J. (2016). The reality of myoelectric prostheses: Understanding what makes these devices difficult for some users to control. Frontiers in Neurorobotics, 10, 7.

4. Mills, D. L. (2006). Network Time Protocol Version 4 Reference and Implementation Guide. University of Delaware.

5. Olson, E. (2011). AprilTag: A robust and flexible visual fiducial system. In 2011 IEEE International Conference on Robotics and Automation (pp. 3400-3407).