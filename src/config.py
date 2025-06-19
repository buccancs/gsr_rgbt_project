# src/config.py

from pathlib import Path

# --- General Application Settings ---
APP_NAME = "GSR-RGBT Data Collection"
GEOMETRY = (
    100,
    100,
    1280,
    520,
)  # Default window position and size (x, y, width, height)

# --- Camera Hardware Settings ---
# IMPORTANT: These IDs must be verified on the experiment machine.
# Use a system tool or a simple OpenCV script to list camera devices to find the correct IDs.
RGB_CAMERA_ID = 0
THERMAL_CAMERA_ID = 1

# --- Video Recording Settings ---
# Frame rate should match the capabilities of your cameras.
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VIDEO_FOURCC = "mp4v"  # Codec for MP4 files, widely compatible.

# --- Data Output Settings ---
# The main directory where all session data will be saved.
# This path is added to.gitignore to prevent accidental versioning of sensitive data.
OUTPUT_DIR = Path("./data/recordings")

# --- GSR Sensor Settings ---
# This section will be used by the gsr_capture module.
# For a real Shimmer3 sensor, this would be the COM port (e.g., 'COM3' on Windows).
GSR_SENSOR_PORT = "COM3"
GSR_SAMPLING_RATE = 128  # Hz, matching the Shimmer3 GSR+ unit sampling rate.
GSR_SIMULATION_MODE = True  # Set to False when using a real sensor.

# --- Experimental Protocol Tasks ---
# Defines the sequence and duration (in seconds) of experimental tasks.
# This can be used later to automate the experimental flow.
EXPERIMENTAL_TASKS = {
    "Baseline": 300,  # 5 minutes
    "Math_Stressor": 180,  # 3 minutes
    "Rest_1": 60,  # 1 minute
    "Relaxation": 300,  # 5 minutes
    "Rest_2": 60,  # 1 minute
    "Emotional_Video": 180,  # 3 minutes
    "Final_Rest": 120,  # 2 minutes
}
