"""Data module for GSR-RGBT project.

This module contains all data-related functionality including capture,
processing, and logging components.
"""

# Import capture classes
from .capture import *

__all__ = [
    # Base capture classes (always available)
    "BaseCapture",
    "BaseCaptureThread",
    "CaptureState", 
    "CaptureManager",
]

# Import processing classes if they exist
try:
    from .processing import *
    __all__.extend([
        "DataLoader",
        "Preprocessor", 
        "FeatureExtractor",
    ])
except ImportError:
    pass

# Import logger if it exists
try:
    from .logger import DataLogger
    __all__.append("DataLogger")
except ImportError:
    pass
