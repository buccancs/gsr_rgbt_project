"""Base capture classes for the GSR-RGBT project.

This module provides abstract base classes and interfaces for all data capture
implementations, ensuring consistent behavior across different capture types.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable
from enum import Enum, auto

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    PYQT5_AVAILABLE = True
except ImportError:
    # PyQt5 not available - create dummy classes for type hints
    PYQT5_AVAILABLE = False

    class QThread:
        """Dummy QThread class when PyQt5 is not available."""
        def __init__(self, parent=None):
            pass
        def setObjectName(self, name):
            pass
        def wait(self, timeout=None):
            return True
        def terminate(self):
            pass

    def pyqtSignal(*args, **kwargs):
        """Dummy pyqtSignal decorator when PyQt5 is not available."""
        def decorator(func):
            return func
        return decorator

from ...core.exceptions import CaptureError, DeviceError
from ...core.constants import DEFAULT_THREAD_TIMEOUT


class CaptureState(Enum):
    """Enumeration of possible capture states."""
    IDLE = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


@runtime_checkable
class CaptureDevice(Protocol):
    """Protocol defining the interface for capture devices."""

    def initialize(self) -> None:
        """Initialize the capture device."""
        ...

    def start_capture(self) -> None:
        """Start capturing data from the device."""
        ...

    def stop_capture(self) -> None:
        """Stop capturing data from the device."""
        ...

    def cleanup(self) -> None:
        """Clean up device resources."""
        ...

    def is_available(self) -> bool:
        """Check if the device is available."""
        ...


class BaseCapture(ABC):
    """Abstract base class for all data capture implementations.

    This class provides the common interface and functionality for all
    capture implementations, including state management, error handling,
    and basic lifecycle methods.

    Attributes:
        device_name: Human-readable name of the capture device
        simulation_mode: Whether to run in simulation mode
        state: Current state of the capture device
    """

    def __init__(
        self,
        device_name: str,
        simulation_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the base capture.

        Args:
            device_name: Human-readable name of the capture device
            simulation_mode: Whether to run in simulation mode
            **kwargs: Additional configuration parameters
        """
        self.device_name = device_name
        self.simulation_mode = simulation_mode
        self.state = CaptureState.IDLE
        self._logger = logging.getLogger(f"{__name__}.{device_name}")

        # Validate device name
        if not device_name or not isinstance(device_name, str):
            raise ValueError("Device name must be a non-empty string")

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the capture device.

        This method should set up the device for capturing data.
        Must be implemented by subclasses.

        Raises:
            DeviceError: If device initialization fails
        """
        pass

    @abstractmethod
    def start_capture(self) -> None:
        """Start capturing data.

        This method should begin the data capture process.
        Must be implemented by subclasses.

        Raises:
            CaptureError: If capture cannot be started
        """
        pass

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing data.

        This method should gracefully stop the data capture process.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources.

        This method should release any resources acquired during capture.
        Must be implemented by subclasses.
        """
        pass

    def is_available(self) -> bool:
        """Check if the capture device is available.

        Returns:
            True if the device is available, False otherwise
        """
        return True  # Default implementation

    def get_state(self) -> CaptureState:
        """Get the current capture state.

        Returns:
            Current capture state
        """
        return self.state

    def _set_state(self, new_state: CaptureState) -> None:
        """Set the capture state and log the change.

        Args:
            new_state: The new state to set
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self._logger.info(f"State changed from {old_state.name} to {new_state.name}")

    def _get_timestamp(self) -> int:
        """Get a high-resolution timestamp.

        Returns:
            Current timestamp in nanoseconds
        """
        return time.perf_counter_ns()

    def __str__(self) -> str:
        """Return string representation of the capture device."""
        mode = "simulation" if self.simulation_mode else "real"
        return f"{self.device_name}Capture({mode}, {self.state.name})"

    def __repr__(self) -> str:
        """Return detailed representation of the capture device."""
        return (
            f"{self.__class__.__name__}("
            f"device_name='{self.device_name}', "
            f"simulation_mode={self.simulation_mode}, "
            f"state={self.state})"
        )


class BaseCaptureThread(QThread, BaseCapture):
    """Base class for threaded capture implementations.

    This class combines the BaseCapture interface with Qt's QThread
    to provide threaded data capture capabilities.

    Signals:
        capture_started: Emitted when capture starts
        capture_stopped: Emitted when capture stops
        capture_error: Emitted when an error occurs
        state_changed: Emitted when the capture state changes
    """

    # Qt signals
    capture_started = pyqtSignal()
    capture_stopped = pyqtSignal()
    capture_error = pyqtSignal(str)  # Error message
    state_changed = pyqtSignal(str)  # State name

    def __init__(
        self,
        device_name: str,
        simulation_mode: bool = False,
        parent: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the threaded capture.

        Args:
            device_name: Human-readable name of the capture device
            simulation_mode: Whether to run in simulation mode
            parent: Qt parent object
            **kwargs: Additional configuration parameters
        """
        QThread.__init__(self, parent)
        BaseCapture.__init__(self, device_name, simulation_mode, **kwargs)

        self.setObjectName(f"{device_name}CaptureThread")

        # Connect state changes to signal emission
        self._original_set_state = self._set_state
        self._set_state = self._emit_state_change

    def _emit_state_change(self, new_state: CaptureState) -> None:
        """Emit state change signal and update state.

        Args:
            new_state: The new state to set
        """
        self._original_set_state(new_state)
        self.state_changed.emit(new_state.name)

    def run(self) -> None:
        """Main thread execution method.

        This method handles the complete capture lifecycle including
        initialization, capture loop, and cleanup.
        """
        try:
            self._set_state(CaptureState.INITIALIZING)
            self._logger.info(f"Starting {self.device_name} capture thread")

            # Initialize the capture device
            self.initialize()

            # Start capturing
            self._set_state(CaptureState.RUNNING)
            self.capture_started.emit()
            self.start_capture()

        except Exception as e:
            self._set_state(CaptureState.ERROR)
            error_msg = f"Error in {self.device_name} capture: {e}"
            self._logger.error(error_msg, exc_info=True)
            self.capture_error.emit(error_msg)

        finally:
            self._set_state(CaptureState.STOPPING)
            try:
                self.cleanup()
            except Exception as e:
                self._logger.error(f"Error during cleanup: {e}", exc_info=True)

            self._set_state(CaptureState.IDLE)
            self.capture_stopped.emit()
            self._logger.info(f"{self.device_name} capture thread finished")

    def stop(self) -> None:
        """Request the thread to stop gracefully.

        This method signals the capture to stop and waits for the thread
        to finish within a reasonable timeout.
        """
        if self.state == CaptureState.RUNNING:
            self._logger.info(f"Requesting {self.device_name} capture to stop")
            self.stop_capture()

            # Wait for thread to finish
            if not self.wait(DEFAULT_THREAD_TIMEOUT * 1000):  # Convert to milliseconds
                self._logger.warning(f"{self.device_name} capture thread did not stop gracefully")
                self.terminate()
                self.wait()

    @abstractmethod
    def _run_simulation(self) -> None:
        """Run capture in simulation mode.

        This method should generate simulated data for testing purposes.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _run_real_capture(self) -> None:
        """Run capture with real hardware.

        This method should capture data from actual hardware devices.
        Must be implemented by subclasses.
        """
        pass

    def start_capture(self) -> None:
        """Start the appropriate capture method based on mode."""
        if self.simulation_mode:
            self._run_simulation()
        else:
            self._run_real_capture()


class CaptureManager:
    """Manager class for coordinating multiple capture devices.

    This class provides centralized management of multiple capture devices,
    including synchronized start/stop operations and state monitoring.
    """

    def __init__(self) -> None:
        """Initialize the capture manager."""
        self._captures: dict[str, BaseCapture] = {}
        self._logger = logging.getLogger(f"{__name__}.CaptureManager")

    def add_capture(self, name: str, capture: BaseCapture) -> None:
        """Add a capture device to the manager.

        Args:
            name: Unique name for the capture device
            capture: The capture device instance

        Raises:
            ValueError: If name already exists or capture is invalid
        """
        if name in self._captures:
            raise ValueError(f"Capture with name '{name}' already exists")

        if not isinstance(capture, BaseCapture):
            raise ValueError("Capture must be an instance of BaseCapture")

        self._captures[name] = capture
        self._logger.info(f"Added capture device: {name}")

    def remove_capture(self, name: str) -> None:
        """Remove a capture device from the manager.

        Args:
            name: Name of the capture device to remove

        Raises:
            KeyError: If capture device not found
        """
        if name not in self._captures:
            raise KeyError(f"Capture device '{name}' not found")

        capture = self._captures.pop(name)
        if capture.get_state() == CaptureState.RUNNING:
            capture.stop_capture()

        self._logger.info(f"Removed capture device: {name}")

    def start_all(self) -> None:
        """Start all capture devices."""
        self._logger.info("Starting all capture devices")
        for name, capture in self._captures.items():
            try:
                if capture.get_state() == CaptureState.IDLE:
                    capture.initialize()
                    capture.start_capture()
            except Exception as e:
                self._logger.error(f"Failed to start capture '{name}': {e}")

    def stop_all(self) -> None:
        """Stop all capture devices."""
        self._logger.info("Stopping all capture devices")
        for name, capture in self._captures.items():
            try:
                if capture.get_state() == CaptureState.RUNNING:
                    capture.stop_capture()
            except Exception as e:
                self._logger.error(f"Failed to stop capture '{name}': {e}")

    def get_capture(self, name: str) -> BaseCapture:
        """Get a capture device by name.

        Args:
            name: Name of the capture device

        Returns:
            The capture device instance

        Raises:
            KeyError: If capture device not found
        """
        if name not in self._captures:
            raise KeyError(f"Capture device '{name}' not found")
        return self._captures[name]

    def get_all_states(self) -> dict[str, CaptureState]:
        """Get the states of all capture devices.

        Returns:
            Dictionary mapping device names to their states
        """
        return {name: capture.get_state() for name, capture in self._captures.items()}

    def cleanup_all(self) -> None:
        """Clean up all capture devices."""
        self._logger.info("Cleaning up all capture devices")
        for name, capture in self._captures.items():
            try:
                capture.cleanup()
            except Exception as e:
                self._logger.error(f"Failed to cleanup capture '{name}': {e}")
