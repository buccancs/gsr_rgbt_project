# src/ml_models/model_config.py

"""
Configuration module for machine learning models.

This module centralizes all model configurations and hyperparameters,
making it easier to experiment with different model architectures and settings.
"""

import yaml
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union

# Default configurations for different model types
DEFAULT_CONFIGS = {
    "lstm": {
        "name": "lstm",
        "layers": [
            {"type": "lstm", "units": 64, "return_sequences": True},
            {"type": "dropout", "rate": 0.2},
            {"type": "lstm", "units": 32, "return_sequences": False},
            {"type": "dropout", "rate": 0.2},
            {"type": "dense", "units": 16, "activation": "relu"},
            {"type": "dense", "units": 1, "activation": "linear"}
        ],
        "compile_params": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
            "loss": "mean_absolute_error",
            "metrics": ["mean_squared_error"]
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 10,
                    "restore_best_weights": True
                },
                "model_checkpoint": {
                    "save_best_only": True,
                    "monitor": "val_loss"
                },
                "tensorboard": {}
            }
        }
    },
    "autoencoder": {
        "name": "autoencoder",
        "latent_dim": 32,
        "encoder_layers": [
            {"type": "flatten"},
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": "latent_dim", "activation": "relu"}
        ],
        "decoder_layers": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": "input_size", "activation": "sigmoid"},
            {"type": "reshape", "target_shape": "input_shape"}
        ],
        "compile_params": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
            "loss": "mse"
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 10,
                    "restore_best_weights": True
                },
                "model_checkpoint": {
                    "save_best_only": True,
                    "monitor": "val_loss"
                },
                "tensorboard": {}
            }
        }
    },
    "vae": {
        "name": "vae",
        "latent_dim": 32,
        "encoder_layers": [
            {"type": "flatten"},
            {"type": "dense", "units": 128, "activation": "relu"}
        ],
        "decoder_layers": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": "input_size", "activation": "sigmoid"},
            {"type": "reshape", "target_shape": "input_shape"}
        ],
        "compile_params": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            }
        },
        "fit_params": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 10,
                    "restore_best_weights": True
                },
                "model_checkpoint": {
                    "save_best_only": True,
                    "monitor": "val_loss"
                },
                "tensorboard": {}
            }
        }
    }
}


class ModelConfig:
    """
    Class to manage model configurations.
    
    This class provides methods to load, save, and modify model configurations,
    making it easier to experiment with different model architectures and hyperparameters.
    """
    
    def __init__(self, config_name: str = None, config_path: Optional[Path] = None):
        """
        Initialize a model configuration.
        
        Args:
            config_name: Name of the default configuration to use (e.g., 'lstm', 'autoencoder', 'vae')
            config_path: Path to a YAML configuration file
        """
        self.config = {}
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
        elif config_name and config_name in DEFAULT_CONFIGS:
            self.config = DEFAULT_CONFIGS[config_name].copy()
        else:
            # Default to LSTM if no valid config is provided
            self.config = DEFAULT_CONFIGS["lstm"].copy()
            
    def load_from_file(self, config_path: Path) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logging.info(f"Loaded model configuration from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load configuration from {config_path}: {e}")
            # Fall back to default LSTM config
            self.config = DEFAULT_CONFIGS["lstm"].copy()
            
    def save_to_file(self, config_path: Path) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path where the configuration will be saved
        """
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logging.info(f"Saved model configuration to {config_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration to {config_path}: {e}")
            
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            The complete configuration dictionary
        """
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        This method performs a deep update of nested dictionaries.
        
        Args:
            updates: Dictionary containing the updates to apply
        """
        self._deep_update(self.config, updates)
        
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
                
    def get_model_name(self) -> str:
        """
        Get the name of the current model configuration.
        
        Returns:
            The model name as a string
        """
        return self.config.get("name", "unknown")
    
    def get_fit_params(self) -> Dict[str, Any]:
        """
        Get the parameters for model.fit().
        
        Returns:
            Dictionary of fit parameters
        """
        return self.config.get("fit_params", {})
    
    def get_compile_params(self) -> Dict[str, Any]:
        """
        Get the parameters for model.compile().
        
        Returns:
            Dictionary of compile parameters
        """
        return self.config.get("compile_params", {})


def list_available_configs() -> List[str]:
    """
    List all available default configurations.
    
    Returns:
        List of configuration names
    """
    return list(DEFAULT_CONFIGS.keys())


def create_example_config_files(output_dir: Path) -> None:
    """
    Create example configuration files for all default model types.
    
    Args:
        output_dir: Directory where the configuration files will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type, config in DEFAULT_CONFIGS.items():
        config_path = output_dir / f"{model_type}_config.yaml"
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logging.info(f"Created example configuration file: {config_path}")
        except Exception as e:
            logging.error(f"Failed to create example configuration file {config_path}: {e}")