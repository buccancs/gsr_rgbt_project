# src/ml_models/model_config.py

"""
Configuration module for machine learning models.

This module centralizes all model configurations and hyperparameters,
making it easier to experiment with different model architectures and settings.
It supports multiple frameworks (PyTorch, TensorFlow) and provides a unified
interface for model configuration.
"""

import yaml
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Union

# Default configurations for different model types
DEFAULT_CONFIGS = {
    # PyTorch LSTM model
    "pytorch_lstm": {
        "name": "lstm",
        "framework": "pytorch",
        "model_params": {
            "input_size": None,  # Will be set based on input shape
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
            "fc_layers": [32, 16, 1],
            "activations": ["relu", "relu", "linear"]
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5
        },
        "loss_fn": "mse",
        "train_params": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping": {
                "patience": 10,
                "monitor": "val_loss"
            },
            "checkpoint": {
                "save_best_only": True,
                "monitor": "val_loss"
            }
        }
    },

    # PyTorch Autoencoder model
    "pytorch_autoencoder": {
        "name": "autoencoder",
        "framework": "pytorch",
        "model_params": {
            "input_size": None,  # Will be set based on input shape
            "latent_dim": 32,
            "encoder_layers": [128, "latent_dim"],
            "decoder_layers": [128, "input_size"],
            "activations": ["relu", "relu", "relu", "sigmoid"]
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5
        },
        "loss_fn": "mse",
        "train_params": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping": {
                "patience": 10,
                "monitor": "val_loss"
            },
            "checkpoint": {
                "save_best_only": True,
                "monitor": "val_loss"
            }
        }
    },

    # PyTorch VAE model
    "pytorch_vae": {
        "name": "vae",
        "framework": "pytorch",
        "model_params": {
            "input_size": None,  # Will be set based on input shape
            "latent_dim": 32,
            "encoder_layers": [128],
            "decoder_layers": [128, "input_size"],
            "activations": ["relu", "relu", "sigmoid"]
        },
        "optimizer_params": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5
        },
        "loss_fn": "vae_loss",  # Special loss function for VAE
        "train_params": {
            "epochs": 100,
            "batch_size": 32,
            "validation_split": 0.2,
            "early_stopping": {
                "patience": 10,
                "monitor": "val_loss"
            },
            "checkpoint": {
                "save_best_only": True,
                "monitor": "val_loss"
            }
        }
    },

    # Legacy TensorFlow LSTM model (for backward compatibility)
    "tf_lstm": {
        "name": "lstm",
        "framework": "tensorflow",
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

    # Legacy TensorFlow Autoencoder model (for backward compatibility)
    "tf_autoencoder": {
        "name": "autoencoder",
        "framework": "tensorflow",
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

    # Legacy TensorFlow VAE model (for backward compatibility)
    "tf_vae": {
        "name": "vae",
        "framework": "tensorflow",
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
    },

    # Aliases for backward compatibility
    "lstm": "pytorch_lstm",
    "autoencoder": "pytorch_autoencoder",
    "vae": "pytorch_vae"
}


class ModelConfig:
    """
    Class to manage model configurations.

    This class provides methods to load, save, and modify model configurations,
    making it easier to experiment with different model architectures and hyperparameters.
    It supports multiple frameworks (PyTorch, TensorFlow) and handles framework-specific
    configuration parameters.
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
        elif config_name:
            # Resolve aliases if necessary
            resolved_name = self._resolve_config_name(config_name)
            if resolved_name in DEFAULT_CONFIGS:
                self.config = DEFAULT_CONFIGS[resolved_name].copy()
            else:
                # Default to PyTorch LSTM if no valid config is provided
                logging.warning(f"Config name '{config_name}' not found. Using default PyTorch LSTM config.")
                self.config = DEFAULT_CONFIGS["pytorch_lstm"].copy()
        else:
            # Default to PyTorch LSTM if no config is provided
            self.config = DEFAULT_CONFIGS["pytorch_lstm"].copy()

    def _resolve_config_name(self, config_name: str) -> str:
        """
        Resolve config name aliases to their actual configuration names.

        Args:
            config_name: The configuration name or alias

        Returns:
            The resolved configuration name
        """
        # Check if the name is an alias
        if config_name in DEFAULT_CONFIGS and isinstance(DEFAULT_CONFIGS[config_name], str):
            return DEFAULT_CONFIGS[config_name]
        return config_name

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

    def get_framework(self) -> str:
        """
        Get the framework of the current model configuration.

        Returns:
            The framework name as a string (e.g., 'pytorch', 'tensorflow')
        """
        return self.config.get("framework", "pytorch")  # Default to PyTorch

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model architecture parameters.

        Returns:
            Dictionary of model parameters
        """
        # For PyTorch models
        if self.get_framework() == "pytorch":
            return self.config.get("model_params", {})
        # For TensorFlow models
        else:
            # For LSTM models, extract parameters from layers
            if self.get_model_name() == "lstm":
                layers = self.config.get("layers", [])
                lstm_units = [layer.get("units") for layer in layers if layer.get("type") == "lstm"]
                dense_units = [layer.get("units") for layer in layers if layer.get("type") == "dense"]
                dropout_rates = [layer.get("rate") for layer in layers if layer.get("type") == "dropout"]

                return {
                    "lstm_units": lstm_units,
                    "dense_units": dense_units,
                    "dropout_rates": dropout_rates
                }
            # For autoencoder/VAE models
            else:
                return {
                    "latent_dim": self.config.get("latent_dim", 32),
                    "encoder_layers": self.config.get("encoder_layers", []),
                    "decoder_layers": self.config.get("decoder_layers", [])
                }

    def get_optimizer_params(self) -> Dict[str, Any]:
        """
        Get the optimizer parameters.

        Returns:
            Dictionary of optimizer parameters
        """
        # For PyTorch models
        if self.get_framework() == "pytorch":
            return self.config.get("optimizer_params", {})
        # For TensorFlow models
        else:
            compile_params = self.config.get("compile_params", {})
            optimizer_config = compile_params.get("optimizer", {})
            return {
                "type": optimizer_config.get("type", "adam"),
                "lr": optimizer_config.get("learning_rate", 0.001)
            }

    def get_loss_fn(self) -> str:
        """
        Get the loss function name.

        Returns:
            The loss function name as a string
        """
        # For PyTorch models
        if self.get_framework() == "pytorch":
            return self.config.get("loss_fn", "mse")
        # For TensorFlow models
        else:
            compile_params = self.config.get("compile_params", {})
            return compile_params.get("loss", "mean_squared_error")

    def get_train_params(self) -> Dict[str, Any]:
        """
        Get the training parameters.

        Returns:
            Dictionary of training parameters
        """
        # For PyTorch models
        if self.get_framework() == "pytorch":
            return self.config.get("train_params", {})
        # For TensorFlow models
        else:
            return self.config.get("fit_params", {})

    def get_fit_params(self) -> Dict[str, Any]:
        """
        Get the parameters for model.fit() (TensorFlow compatibility).

        Returns:
            Dictionary of fit parameters
        """
        return self.config.get("fit_params", {})

    def get_compile_params(self) -> Dict[str, Any]:
        """
        Get the parameters for model.compile() (TensorFlow compatibility).

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
