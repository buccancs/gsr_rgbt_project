"""Main application class for GSR-RGBT project.

This module contains the main application class that orchestrates the GUI
and data capture components using the new core architecture.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Dict, Any
from enum import Enum, auto

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow
    from PyQt5.QtCore import QObject, QThread, pyqtSlot
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    # Create dummy classes for type hints
    class QApplication:
        def __init__(self, *args): pass
        def exec_(self): return 0
    class QMainWindow:
        def __init__(self): pass
        def setWindowTitle(self, title): pass
        def setGeometry(self, *args): pass
        def setCentralWidget(self, widget): pass
        def show(self): pass
    class QMessageBox:
        Critical = 0
        def __init__(self): pass
        def setIcon(self, icon): pass
        def setText(self, text): pass
        def setInformativeText(self, text): pass
        def setWindowTitle(self, title): pass
        def exec_(self): pass
    def pyqtSlot(*args, **kwargs):
        def decorator(func): return func
        return decorator

from ..core.config import Config, get_config
from ..core.constants import APP_NAME, DEFAULT_TIMESTAMP_FREQUENCY
from ..core.exceptions import (
    GSRRGBTError, 
    DeviceNotFoundError, 
    CaptureError,
    ConfigurationError
)
from ..data.capture import CaptureManager, CaptureState


class ApplicationState(Enum):
    """Enumeration of possible application states."""
    INITIALIZING = auto()
    IDLE = auto()
    RECORDING = auto()
    STOPPING = auto()
    ERROR = auto()


class GSRRGBTApplication(QMainWindow):
    """Main application class for GSR-RGBT data collection.
    
    This class orchestrates the GUI and data capture components using
    the new core architecture with proper separation of concerns.
    
    Attributes:
        config: Application configuration
        state: Current application state
        capture_manager: Manager for all capture devices
    """
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the application.
        
        Args:
            config: Optional configuration object. If None, uses global config.
            
        Raises:
            ConfigurationError: If configuration is invalid
            RuntimeError: If PyQt5 is not available
        """
        if not PYQT5_AVAILABLE:
            raise RuntimeError("PyQt5 is required for GUI functionality")
            
        super().__init__()
        
        # Initialize configuration
        self.config = config or get_config()
        self._logger = logging.getLogger(f"{__name__}.GSRRGBTApplication")
        
        # Initialize state
        self.state = ApplicationState.INITIALIZING
        self._logger.info("Initializing GSR-RGBT application")
        
        # Initialize capture manager
        self.capture_manager = CaptureManager()
        
        # Initialize UI
        self._setup_ui()
        
        # Initialize capture devices
        self._setup_capture_devices()
        
        # Set initial state
        self._set_state(ApplicationState.IDLE)
        self._logger.info("Application initialization complete")
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Set window properties from config
            self.setWindowTitle(self.config.app_name)
            self.setGeometry(*self.config.get_window_geometry())
            
            # TODO: Create and set main window widget
            # This would be implemented when the main window is refactored
            self._logger.info("UI setup complete")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to setup UI: {e}") from e
    
    def _setup_capture_devices(self) -> None:
        """Set up capture devices based on configuration."""
        try:
            # TODO: Create specific capture implementations
            # For now, this is a placeholder showing the intended structure
            
            # Example of how capture devices would be added:
            # rgb_capture = VideoCapture(
            #     device_name="RGB",
            #     camera_id=self.config.rgb_camera_id,
            #     fps=self.config.fps,
            #     simulation_mode=False
            # )
            # self.capture_manager.add_capture("rgb", rgb_capture)
            
            self._logger.info("Capture devices setup complete")
            
        except Exception as e:
            self._logger.error(f"Failed to setup capture devices: {e}")
            # Don't raise here - allow app to start in degraded mode
    
    def _set_state(self, new_state: ApplicationState) -> None:
        """Set the application state and update UI accordingly.
        
        Args:
            new_state: The new state to set
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self._logger.info(f"Application state changed: {old_state.name} → {new_state.name}")
            self._update_ui_for_state()
    
    def _update_ui_for_state(self) -> None:
        """Update UI elements based on current application state."""
        # TODO: Implement UI updates based on state
        # This would be implemented when the main window is refactored
        pass
    
    def start_recording(self, subject_id: str) -> None:
        """Start data recording for the specified subject.
        
        Args:
            subject_id: Unique identifier for the subject
            
        Raises:
            CaptureError: If recording cannot be started
            ValidationError: If subject_id is invalid
        """
        try:
            if self.state != ApplicationState.IDLE:
                raise CaptureError(f"Cannot start recording in state: {self.state.name}")
            
            if not subject_id or not subject_id.strip():
                raise ValueError("Subject ID cannot be empty")
            
            self._logger.info(f"Starting recording for subject: {subject_id}")
            self._set_state(ApplicationState.RECORDING)
            
            # Start all capture devices
            self.capture_manager.start_all()
            
            self._logger.info("Recording started successfully")
            
        except Exception as e:
            self._set_state(ApplicationState.ERROR)
            error_msg = f"Failed to start recording: {e}"
            self._logger.error(error_msg)
            self._show_error_message("Recording Error", error_msg)
            raise CaptureError(error_msg) from e
    
    def stop_recording(self) -> None:
        """Stop data recording."""
        try:
            if self.state != ApplicationState.RECORDING:
                self._logger.warning(f"Stop recording called in state: {self.state.name}")
                return
            
            self._logger.info("Stopping recording")
            self._set_state(ApplicationState.STOPPING)
            
            # Stop all capture devices
            self.capture_manager.stop_all()
            
            self._set_state(ApplicationState.IDLE)
            self._logger.info("Recording stopped successfully")
            
        except Exception as e:
            self._set_state(ApplicationState.ERROR)
            error_msg = f"Error stopping recording: {e}"
            self._logger.error(error_msg)
            self._show_error_message("Recording Error", error_msg)
    
    def _show_error_message(self, title: str, message: str) -> None:
        """Display an error message to the user.
        
        Args:
            title: Title of the error dialog
            message: Error message to display
        """
        if PYQT5_AVAILABLE:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(title)
            msg_box.setInformativeText(message)
            msg_box.setWindowTitle("Error")
            msg_box.exec_()
        else:
            # Fallback for when PyQt5 is not available
            print(f"ERROR - {title}: {message}")
    
    def get_capture_states(self) -> Dict[str, str]:
        """Get the current states of all capture devices.
        
        Returns:
            Dictionary mapping device names to their state names
        """
        states = self.capture_manager.get_all_states()
        return {name: state.name for name, state in states.items()}
    
    def closeEvent(self, event) -> None:
        """Handle application close event.
        
        Args:
            event: Close event from Qt
        """
        self._logger.info("Application close requested")
        
        try:
            # Stop recording if active
            if self.state == ApplicationState.RECORDING:
                self.stop_recording()
            
            # Clean up capture devices
            self.capture_manager.cleanup_all()
            
            self._logger.info("Application cleanup complete")
            
        except Exception as e:
            self._logger.error(f"Error during application cleanup: {e}")
        
        finally:
            if hasattr(event, 'accept'):
                event.accept()


def create_application(config_path: Optional[str] = None) -> GSRRGBTApplication:
    """Create and configure the GSR-RGBT application.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured application instance
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
        RuntimeError: If PyQt5 is not available
    """
    try:
        # Load configuration if path provided
        if config_path:
            from ..core.config import load_config_from_file
            config = load_config_from_file(config_path)
        else:
            config = get_config()
        
        # Create application
        app = GSRRGBTApplication(config)
        
        return app
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create application: {e}") from e


def main(config_path: Optional[str] = None) -> int:
    """Main entry point for the GSR-RGBT application.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not PYQT5_AVAILABLE:
        print("Error: PyQt5 is required to run the GUI application")
        return 1
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create Qt application
        qt_app = QApplication(sys.argv)
        
        # Create our application
        app = create_application(config_path)
        
        # Show the application
        app.show()
        
        logger.info("Starting GSR-RGBT application")
        
        # Run the application
        return qt_app.exec_()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())