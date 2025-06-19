# System Utilities

This directory contains general system utilities that are used across the project.

## Files

### data_logger.py

The `DataLogger` class handles the organized and synchronized writing of multimodal data streams to disk. For each recording session, it creates a unique, timestamped directory and manages separate VideoWriter objects for RGB and thermal camera streams and a CSV writer for time-stamped GSR sensor data.

### timestamp_thread.py

The `TimestampThread` class is a high-priority thread that emits timestamps at a fast, consistent rate. It serves as a centralized timestamp authority for all data capture components, ensuring precise synchronization between different data streams.

## Usage

These utilities are designed to be used by various components of the system, including the data collection application and the machine learning pipeline. They provide common functionality that is needed across the project.

Example usage of the `TimestampThread`:

```python
from src.system.utils.timestamp_thread import TimestampThread

# Create a timestamp thread with a frequency of 200Hz
timestamp_thread = TimestampThread(frequency=200)

# Connect to the timestamp signal
timestamp_thread.timestamp_generated.connect(handle_timestamp)

# Start the thread
timestamp_thread.start()

# Later, stop the thread
timestamp_thread.stop()
```

Example usage of the `DataLogger`:

```python
from pathlib import Path
from src.system.utils.data_logger import DataLogger

# Create a data logger
output_dir = Path("data/recordings")
subject_id = "Subject01"
fps = 30
video_fourcc = "mp4v"
logger = DataLogger(output_dir, subject_id, fps, video_fourcc)

# Start logging
frame_size_rgb = (640, 480)
frame_size_thermal = (640, 480)
logger.start_logging(frame_size_rgb, frame_size_thermal)

# Log data
logger.log_rgb_frame(rgb_frame, timestamp)
logger.log_thermal_frame(thermal_frame, timestamp)
logger.log_gsr_data(gsr_value, shimmer_timestamp)

# Stop logging
logger.stop_logging()
```