# src/config.py

from pathlib import Path

# --- General Application Settings ---
APP_NAME = "GSR-RGBT Data Collection"
GEOMETRY = (100, 100, 1290, 560)  # (x, y, width, height)

# --- Camera Hardware Settings ---
# CRITICAL: These IDs must be verified on the experiment machine.
# Use a camera utility program to find the correct device index for each camera.
RGB_CAMERA_ID = 0
THERMAL_CAMERA_ID = 1

# --- Video Recording Settings ---
# Defines the properties for the output video files.
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VIDEO_FOURCC = (
    "mp4v"  # Codec for creating .mp4 files. 'XVID' for .avi is also an option.
)

# --- Data Output Settings ---
# The root directory where all recorded session data will be saved.
OUTPUT_DIR = Path("./data/recordings")

# --- GSR Sensor Settings ---
# This section configures the ground-truth sensor.
# For a Shimmer3 device on Windows, this would be the COM port. On Linux, it might be '/dev/ttyACM0'.
GSR_SENSOR_PORT = "COM3"
# The sampling rate should match the sensor's capabilities.
GSR_SAMPLING_RATE = 32  # Hz
# Set to False to use a real sensor. Set to True to use simulated data for development.
GSR_SIMULATION_MODE = True
