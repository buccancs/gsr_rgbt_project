"""Configuration management for the GSR-RGBT project.

This module provides a centralized configuration system with validation,
type safety, and support for loading from files and environment variables.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass, field

from .constants import (
    DEFAULT_WINDOW_GEOMETRY,
    DEFAULT_RGB_CAMERA_ID,
    DEFAULT_THERMAL_CAMERA_ID,
    DEFAULT_THERMAL_SIMULATION_MODE,
    DEFAULT_FPS,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_VIDEO_FOURCC,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_GSR_SENSOR_PORT,
    DEFAULT_GSR_SAMPLING_RATE,
    DEFAULT_GSR_SIMULATION_MODE,
    DEFAULT_EXPERIMENTAL_TASKS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TIMESTAMP_FREQUENCY,
    SUPPORTED_VIDEO_CODECS,
    SUPPORTED_GSR_SAMPLING_RATES,
    MIN_FRAME_WIDTH,
    MAX_FRAME_WIDTH,
    MIN_FRAME_HEIGHT,
    MAX_FRAME_HEIGHT,
    MIN_FPS,
    MAX_FPS,
    MIN_GSR_SAMPLING_RATE,
    MAX_GSR_SAMPLING_RATE,
)
from .exceptions import ConfigurationError, ValidationError


@dataclass
class Config:
    """Configuration class for the GSR-RGBT application.
    
    This class holds all configuration parameters with validation
    and provides methods to load from files or environment variables.
    
    Attributes:
        app_name: Name of the application
        geometry: Window geometry (x, y, width, height)
        rgb_camera_id: ID of the RGB camera
        thermal_camera_id: ID of the thermal camera
        thermal_simulation_mode: Whether to use thermal simulation
        fps: Frames per second for recording
        frame_width: Width of recorded frames
        frame_height: Height of recorded frames
        video_fourcc: Video codec for recording
        output_dir: Directory for output files
        gsr_sensor_port: Port for GSR sensor
        gsr_sampling_rate: Sampling rate for GSR sensor
        gsr_simulation_mode: Whether to use GSR simulation
        experimental_tasks: Dictionary of experimental tasks and durations
        log_level: Logging level
        timestamp_frequency: Frequency for timestamp generation
    """
    
    # Application settings
    app_name: str = "GSR-RGBT Data Collection"
    geometry: tuple[int, int, int, int] = field(default_factory=lambda: DEFAULT_WINDOW_GEOMETRY)
    
    # Camera settings
    rgb_camera_id: int = DEFAULT_RGB_CAMERA_ID
    thermal_camera_id: int = DEFAULT_THERMAL_CAMERA_ID
    thermal_simulation_mode: bool = DEFAULT_THERMAL_SIMULATION_MODE
    
    # Video recording settings
    fps: int = DEFAULT_FPS
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    video_fourcc: str = DEFAULT_VIDEO_FOURCC
    
    # Data output settings
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    
    # GSR sensor settings
    gsr_sensor_port: str = DEFAULT_GSR_SENSOR_PORT
    gsr_sampling_rate: int = DEFAULT_GSR_SAMPLING_RATE
    gsr_simulation_mode: bool = DEFAULT_GSR_SIMULATION_MODE
    
    # Experimental protocol
    experimental_tasks: dict[str, int] = field(default_factory=lambda: DEFAULT_EXPERIMENTAL_TASKS.copy())
    
    # System settings
    log_level: str = DEFAULT_LOG_LEVEL
    timestamp_frequency: int = DEFAULT_TIMESTAMP_FREQUENCY
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all configuration parameters.
        
        Raises:
            ValidationError: If any configuration parameter is invalid
        """
        try:
            self._validate_camera_settings()
            self._validate_video_settings()
            self._validate_gsr_settings()
            self._validate_paths()
            self._validate_experimental_tasks()
        except Exception as e:
            raise ValidationError(f"Configuration validation failed: {e}") from e
    
    def _validate_camera_settings(self) -> None:
        """Validate camera-related settings."""
        if not isinstance(self.rgb_camera_id, int) or self.rgb_camera_id < 0:
            raise ValidationError("RGB camera ID must be a non-negative integer")
        
        if not isinstance(self.thermal_camera_id, int) or self.thermal_camera_id < 0:
            raise ValidationError("Thermal camera ID must be a non-negative integer")
        
        if self.rgb_camera_id == self.thermal_camera_id:
            raise ValidationError("RGB and thermal camera IDs must be different")
    
    def _validate_video_settings(self) -> None:
        """Validate video recording settings."""
        if not (MIN_FPS <= self.fps <= MAX_FPS):
            raise ValidationError(f"FPS must be between {MIN_FPS} and {MAX_FPS}")
        
        if not (MIN_FRAME_WIDTH <= self.frame_width <= MAX_FRAME_WIDTH):
            raise ValidationError(f"Frame width must be between {MIN_FRAME_WIDTH} and {MAX_FRAME_WIDTH}")
        
        if not (MIN_FRAME_HEIGHT <= self.frame_height <= MAX_FRAME_HEIGHT):
            raise ValidationError(f"Frame height must be between {MIN_FRAME_HEIGHT} and {MAX_FRAME_HEIGHT}")
        
        if self.video_fourcc not in SUPPORTED_VIDEO_CODECS:
            raise ValidationError(f"Video codec must be one of: {SUPPORTED_VIDEO_CODECS}")
    
    def _validate_gsr_settings(self) -> None:
        """Validate GSR sensor settings."""
        if not (MIN_GSR_SAMPLING_RATE <= self.gsr_sampling_rate <= MAX_GSR_SAMPLING_RATE):
            raise ValidationError(
                f"GSR sampling rate must be between {MIN_GSR_SAMPLING_RATE} and {MAX_GSR_SAMPLING_RATE}"
            )
        
        if self.gsr_sampling_rate not in SUPPORTED_GSR_SAMPLING_RATES:
            raise ValidationError(f"GSR sampling rate must be one of: {SUPPORTED_GSR_SAMPLING_RATES}")
    
    def _validate_paths(self) -> None:
        """Validate file paths."""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        
        # Create output directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory {self.output_dir}: {e}") from e
    
    def _validate_experimental_tasks(self) -> None:
        """Validate experimental tasks configuration."""
        if not isinstance(self.experimental_tasks, dict):
            raise ValidationError("Experimental tasks must be a dictionary")
        
        for task_name, duration in self.experimental_tasks.items():
            if not isinstance(task_name, str) or not task_name.strip():
                raise ValidationError("Task names must be non-empty strings")
            
            if not isinstance(duration, int) or duration <= 0:
                raise ValidationError(f"Task duration for '{task_name}' must be a positive integer")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> Config:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Config instance loaded from file
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert output_dir to Path if it's a string
            if 'output_dir' in data and isinstance(data['output_dir'], str):
                data['output_dir'] = Path(data['output_dir'])
            
            return cls(**data)
        
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file: {e}") from e
    
    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with 'GSRRGBT_'.
        For example: GSRRGBT_FPS=30
        
        Returns:
            Config instance with values from environment variables
        """
        config_data = {}
        
        # Map environment variable names to config attributes
        env_mapping = {
            'GSRRGBT_RGB_CAMERA_ID': ('rgb_camera_id', int),
            'GSRRGBT_THERMAL_CAMERA_ID': ('thermal_camera_id', int),
            'GSRRGBT_THERMAL_SIMULATION_MODE': ('thermal_simulation_mode', lambda x: x.lower() == 'true'),
            'GSRRGBT_FPS': ('fps', int),
            'GSRRGBT_FRAME_WIDTH': ('frame_width', int),
            'GSRRGBT_FRAME_HEIGHT': ('frame_height', int),
            'GSRRGBT_VIDEO_FOURCC': ('video_fourcc', str),
            'GSRRGBT_OUTPUT_DIR': ('output_dir', Path),
            'GSRRGBT_GSR_SENSOR_PORT': ('gsr_sensor_port', str),
            'GSRRGBT_GSR_SAMPLING_RATE': ('gsr_sampling_rate', int),
            'GSRRGBT_GSR_SIMULATION_MODE': ('gsr_simulation_mode', lambda x: x.lower() == 'true'),
            'GSRRGBT_LOG_LEVEL': ('log_level', str),
            'GSRRGBT_TIMESTAMP_FREQUENCY': ('timestamp_frequency', int),
        }
        
        for env_var, (attr_name, converter) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    config_data[attr_name] = converter(value)
                except Exception as e:
                    raise ConfigurationError(f"Invalid value for {env_var}: {value}") from e
        
        return cls(**config_data)
    
    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.
        
        Args:
            config_path: Path where to save the configuration file
            
        Raises:
            ConfigurationError: If file cannot be written
        """
        config_path = Path(config_path)
        
        try:
            # Convert Path objects to strings for JSON serialization
            data = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Path):
                    data[key] = str(value)
                else:
                    data[key] = value
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration file: {e}") from e
    
    def update(self, **kwargs: Any) -> None:
        """Update configuration parameters and re-validate.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            ValidationError: If updated configuration is invalid
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ConfigurationError(f"Unknown configuration parameter: {key}")
        
        self.validate()
    
    def get_frame_size(self) -> tuple[int, int]:
        """Get frame size as a tuple.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.frame_width, self.frame_height)
    
    def get_window_geometry(self) -> tuple[int, int, int, int]:
        """Get window geometry as a tuple.
        
        Returns:
            Tuple of (x, y, width, height)
        """
        return self.geometry


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        The global Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.
    
    Args:
        config: The Config instance to set as global
    """
    global _config
    _config = config


def load_config_from_file(config_path: Union[str, Path]) -> Config:
    """Load and set global configuration from file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        The loaded Config instance
    """
    config = Config.from_file(config_path)
    set_config(config)
    return config


def load_config_from_env() -> Config:
    """Load and set global configuration from environment variables.
    
    Returns:
        The loaded Config instance
    """
    config = Config.from_env()
    set_config(config)
    return config