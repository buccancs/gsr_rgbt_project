"""Data capture module for GSR-RGBT project.

This module contains classes for capturing data from various sources
including GSR sensors, thermal cameras, and RGB cameras.
"""

from .base import BaseCapture, BaseCaptureThread, CaptureState, CaptureManager

# Import specific capture classes if they exist
__all__ = [
    "BaseCapture",
    "BaseCaptureThread", 
    "CaptureState",
    "CaptureManager",
]

try:
    from .gsr import GSRCapture
    __all__.append("GSRCapture")
except ImportError:
    pass

try:
    from .thermal import ThermalCapture
    __all__.append("ThermalCapture")
except ImportError:
    pass

try:
    from .video import VideoCapture
    __all__.append("VideoCapture")
except ImportError:
    pass
