# src/processing/data_loader.py
# This file is a redirection to src/ml_pipeline/preprocessing/data_loader.py
# to maintain backward compatibility while consolidating duplicate code.

import csv
import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# Import from the consolidated location
from src.ml_pipeline.preprocessing.data_loader import (
    load_gsr_data,
    video_frame_generator,
    SessionDataLoader
)

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

# The function implementations have been moved to src/ml_pipeline/preprocessing/data_loader.py
# This file now imports them from there to maintain backward compatibility.