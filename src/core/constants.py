"""Constants for the GSR-RGBT project.

This module contains all constant values used throughout the project.
These values should not change during runtime and represent fixed
system parameters, default values, and configuration constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Application Information
APP_NAME: Final[str] = "GSR-RGBT Data Collection"
APP_VERSION: Final[str] = "1.0.0"
APP_DESCRIPTION: Final[str] = "Galvanic Skin Response and RGB-Thermal data collection system"

# Default Window Geometry (x, y, width, height)
DEFAULT_WINDOW_GEOMETRY: Final[tuple[int, int, int, int]] = (100, 100, 1280, 520)

# Camera Hardware Defaults
DEFAULT_RGB_CAMERA_ID: Final[int] = 0
DEFAULT_THERMAL_CAMERA_ID: Final[int] = 1
DEFAULT_THERMAL_SIMULATION_MODE: Final[bool] = True

# Video Recording Defaults
DEFAULT_FPS: Final[int] = 30
DEFAULT_FRAME_WIDTH: Final[int] = 640
DEFAULT_FRAME_HEIGHT: Final[int] = 480
DEFAULT_VIDEO_FOURCC: Final[str] = "mp4v"

# Supported Video Codecs
SUPPORTED_VIDEO_CODECS: Final[tuple[str, ...]] = (
    "mp4v",  # MPEG-4 Part 2
    "XVID",  # Xvid MPEG-4
    "H264",  # H.264/AVC
    "MJPG",  # Motion JPEG
)

# Frame Size Constraints
MIN_FRAME_WIDTH: Final[int] = 320
MAX_FRAME_WIDTH: Final[int] = 1920
MIN_FRAME_HEIGHT: Final[int] = 240
MAX_FRAME_HEIGHT: Final[int] = 1080

# FPS Constraints
MIN_FPS: Final[int] = 1
MAX_FPS: Final[int] = 120

# Default Data Output Directory
DEFAULT_OUTPUT_DIR: Final[Path] = Path("./data/recordings")

# GSR Sensor Defaults
DEFAULT_GSR_SENSOR_PORT: Final[str] = "COM3"
DEFAULT_GSR_SAMPLING_RATE: Final[int] = 128  # Hz
DEFAULT_GSR_SIMULATION_MODE: Final[bool] = True

# GSR Sampling Rate Constraints
MIN_GSR_SAMPLING_RATE: Final[int] = 1
MAX_GSR_SAMPLING_RATE: Final[int] = 1000

# Supported GSR Sampling Rates (common Shimmer3 rates)
SUPPORTED_GSR_SAMPLING_RATES: Final[tuple[int, ...]] = (
    1, 10, 25, 50, 100, 128, 200, 250, 500, 1000
)

# Default Experimental Protocol Tasks (duration in seconds)
DEFAULT_EXPERIMENTAL_TASKS: Final[dict[str, int]] = {
    "Baseline": 300,        # 5 minutes
    "Math_Stressor": 180,   # 3 minutes
    "Rest_1": 60,           # 1 minute
    "Relaxation": 300,      # 5 minutes
    "Rest_2": 60,           # 1 minute
    "Emotional_Video": 180, # 3 minutes
    "Final_Rest": 120,      # 2 minutes
}

# Task Duration Constraints (in seconds)
MIN_TASK_DURATION: Final[int] = 1
MAX_TASK_DURATION: Final[int] = 3600  # 1 hour

# File Extensions
VIDEO_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".mp4", ".avi", ".mov", ".mkv")
DATA_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".csv", ".json", ".pkl", ".h5")
IMAGE_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Logging Configuration
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s - [%(levelname)s] - %(module)s - %(message)s"

# Thread and Process Defaults
DEFAULT_TIMESTAMP_FREQUENCY: Final[int] = 200  # Hz
DEFAULT_THREAD_TIMEOUT: Final[int] = 5  # seconds

# Data Validation Constraints
MAX_SUBJECT_ID_LENGTH: Final[int] = 50
MIN_SUBJECT_ID_LENGTH: Final[int] = 1

# Device Detection Timeouts (in seconds)
DEVICE_DETECTION_TIMEOUT: Final[int] = 10
DEVICE_CONNECTION_TIMEOUT: Final[int] = 5

# Buffer Sizes
DEFAULT_FRAME_BUFFER_SIZE: Final[int] = 100
DEFAULT_GSR_BUFFER_SIZE: Final[int] = 1000

# Error Codes
ERROR_CODES: Final[dict[str, str]] = {
    "DEVICE_NOT_FOUND": "DEV001",
    "DEVICE_CONNECTION_FAILED": "DEV002",
    "CAPTURE_FAILED": "CAP001",
    "PROCESSING_FAILED": "PROC001",
    "MODEL_FAILED": "ML001",
    "CONFIG_INVALID": "CFG001",
    "FILE_OPERATION_FAILED": "FILE001",
    "VALIDATION_FAILED": "VAL001",
    "SYNCHRONIZATION_FAILED": "SYNC001",
}

# Status Messages
STATUS_MESSAGES: Final[dict[str, str]] = {
    "IDLE": "Ready to start recording",
    "RECORDING": "Recording in progress",
    "STOPPING": "Stopping recording",
    "ERROR": "An error occurred",
    "SIMULATION": "Running in simulation mode",
}

# Color Codes for UI (RGB tuples)
UI_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "SUCCESS": (0, 255, 0),      # Green
    "WARNING": (255, 255, 0),    # Yellow
    "ERROR": (255, 0, 0),        # Red
    "INFO": (0, 0, 255),         # Blue
    "NEUTRAL": (128, 128, 128),  # Gray
}

# Model Training Defaults
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_EPOCHS: Final[int] = 100
DEFAULT_LEARNING_RATE: Final[float] = 0.001
DEFAULT_VALIDATION_SPLIT: Final[float] = 0.2

# Model Architecture Defaults
DEFAULT_LSTM_HIDDEN_SIZE: Final[int] = 64
DEFAULT_LSTM_NUM_LAYERS: Final[int] = 2
DEFAULT_DROPOUT_RATE: Final[float] = 0.2

# Performance Monitoring
DEFAULT_MEMORY_THRESHOLD: Final[float] = 0.8  # 80% memory usage
DEFAULT_CPU_THRESHOLD: Final[float] = 0.9     # 90% CPU usage