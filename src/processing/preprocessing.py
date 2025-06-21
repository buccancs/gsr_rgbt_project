# src/processing/preprocessing.py
# This file is a redirection to src/ml_pipeline/preprocessing/preprocessing.py
# to maintain backward compatibility while consolidating duplicate code.

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union

import cv2
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from the consolidated location
from src.ml_pipeline.preprocessing.preprocessing import (
    process_gsr_signal,
    detect_hand_landmarks,
    define_multi_roi,
    detect_palm_roi,
    extract_roi_signal,
    extract_multi_roi_signals,
    process_frame_with_multi_roi,
    visualize_multi_roi,
    CYTHON_AVAILABLE
)

# Try to import MediaPipe (for backward compatibility)
try:
    import mediapipe as mp
except ImportError:
    logging.warning("MediaPipe is not available. Hand landmark detection will not work.")

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

# The function implementations have been moved to src/ml_pipeline/preprocessing/preprocessing.py
# This file now imports them from there to maintain backward compatibility.