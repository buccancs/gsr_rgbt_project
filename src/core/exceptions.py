"""Custom exceptions for the GSR-RGBT project.

This module defines custom exception classes that provide more specific
error handling throughout the application.
"""

from __future__ import annotations

from typing import Any, Optional


class GSRRGBTError(Exception):
    """Base exception class for all GSR-RGBT related errors.
    
    This is the base class for all custom exceptions in the project.
    It provides a consistent interface for error handling.
    
    Args:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        details: Optional additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def __repr__(self) -> str:
        """Return a detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"details={self.details})"
        )


class DeviceError(GSRRGBTError):
    """Exception raised for device-related errors.
    
    This exception is raised when there are issues with hardware devices
    such as cameras, sensors, or communication interfaces.
    """
    pass


class DeviceNotFoundError(DeviceError):
    """Exception raised when a required device is not found.
    
    This is a specific type of DeviceError for cases where a device
    that should be present cannot be detected or accessed.
    """
    pass


class DeviceConnectionError(DeviceError):
    """Exception raised when device connection fails.
    
    This exception is raised when a device is found but cannot be
    properly connected or initialized.
    """
    pass


class CaptureError(GSRRGBTError):
    """Exception raised for data capture related errors.
    
    This exception is raised when there are issues during the data
    capture process, such as frame drops, synchronization issues,
    or storage problems.
    """
    pass


class ProcessingError(GSRRGBTError):
    """Exception raised for data processing related errors.
    
    This exception is raised when there are issues during data
    processing, such as invalid data formats, preprocessing failures,
    or feature extraction problems.
    """
    pass


class ModelError(GSRRGBTError):
    """Exception raised for machine learning model related errors.
    
    This exception is raised when there are issues with ML models,
    such as training failures, prediction errors, or model loading
    problems.
    """
    pass


class ConfigurationError(GSRRGBTError):
    """Exception raised for configuration related errors.
    
    This exception is raised when there are issues with configuration
    files, invalid parameters, or missing required settings.
    """
    pass


class ValidationError(GSRRGBTError):
    """Exception raised for data validation errors.
    
    This exception is raised when input data fails validation checks
    or doesn't meet expected criteria.
    """
    pass


class FileOperationError(GSRRGBTError):
    """Exception raised for file operation errors.
    
    This exception is raised when there are issues with file operations
    such as reading, writing, or accessing files.
    """
    pass


class SynchronizationError(GSRRGBTError):
    """Exception raised for synchronization related errors.
    
    This exception is raised when there are issues with timing
    synchronization between different data streams or processes.
    """
    pass