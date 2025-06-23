"""GUI module for GSR-RGBT project.

This module contains all GUI-related components including the main application,
windows, and widgets.
"""

# Import main application components if they exist
__all__ = []

try:
    from .application import GSRRGBTApplication
    __all__.append("GSRRGBTApplication")
except ImportError:
    pass

try:
    from .main_window import MainWindow
    __all__.append("MainWindow")
except ImportError:
    pass