# src/processing/feature_engineering.py
# This file is a redirection to src/ml_pipeline/feature_engineering/feature_engineering.py
# to maintain backward compatibility while consolidating duplicate code.

import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd

# Import from the consolidated location
from src.ml_pipeline.feature_engineering.feature_engineering import (
    align_signals,
    create_feature_windows,
    create_dataset_from_session,
    CYTHON_AVAILABLE
)

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

# The function implementations have been moved to src/ml_pipeline/feature_engineering/feature_engineering.py
# This file now imports them from there to maintain backward compatibility.