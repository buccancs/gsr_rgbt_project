"""Core module for GSR-RGBT project.

This module contains core functionality including configuration, constants,
and custom exceptions used throughout the project.
"""

__version__ = "1.0.0"
__author__ = "GSR-RGBT Development Team"

from .config import Config
from .constants import *
from .exceptions import *

__all__ = [
    "Config",
    "GSRRGBTError",
    "DeviceError",
    "CaptureError",
    "ProcessingError",
    "ModelError",
]