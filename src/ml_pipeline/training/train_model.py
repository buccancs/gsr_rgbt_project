#!/usr/bin/env python3
# src/ml_pipeline/training/train_model.py

import logging
import argparse
import json
import datetime
import subprocess
from typing import Dict, Any, Optional, Tuple, List

# --- Import project modules ---
# Add the project root to the Python path to allow for absolute imports
import sys
import os
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src import config
from src.ml_pipeline.feature_engineering.feature_engineering import create_dataset_from_session
from src.ml_models.model_interface import ModelRegistry
from src.ml_models.model_config import ModelConfig, list_available_configs, create_example_config_files

# Import PyTorch models (this will register them with the ModelRegistry)
import src.ml_models.pytorch_models

# For backward compatibility with TensorFlow models
try:
    import tensorflow as tf
    from src.ml_models.models import build_lstm_model, build_ae_model, build_vae_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Only PyTorch models will be supported.")

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def load_all_session_data(data_dir: Path, gsr_sampling_rate: int, video_fps: int):
    """
    Load and process data from all session directories.

    Args:
        data_dir (Path): Path to the directory containing session recordings
        gsr_sampling_rate (int): Sampling rate of the GSR signal in Hz
        video_fps (int): Frame rate of the video in frames per second

    Returns:
        tuple: (X, y, subject_ids) where X is the feature data, y is the target data,
               and subject_ids is a list of subject IDs for each sample
    """
    all_X = []
    all_y = []
    all_subject_ids = []

    # Find all session directories
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Subject_")]

    if not session_dirs:
        logging.error(f"No session directories found in {data_dir}")
        return None, None, None

    logging.info(f"Found {len(session_dirs)} session directories")

    # Process each session
    for session_dir in session_dirs:
        subject_id = session_dir.name.split("_")[1]
        logging.info(f"Processing session for subject {subject_id}")

        # Create dataset from session
        dataset = create_dataset_from_session(session_dir, gsr_sampling_rate, video_fps)

        if dataset is None:
            logging.warning(f"Failed to create dataset for session {session_dir.name}")
            continue

        X, y = dataset

        # Add to the combined dataset
        all_X.append(X)
        all_y.append(y)
        all_subject_ids.extend([subject_id] * len(y))

    if not all_X:
        logging.error("No valid data was loaded from any session")
        return None, None, None

    # Combine all sessions
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y, all_subject_ids


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a model on the GSR-RGBT dataset")

    # Model type and configuration
    parser.add_argument("--model-type", type=str, default="lstm",
                        choices=["lstm", "autoencoder", "vae", "cnn", "cnn_lstm", "transformer", "resnet"],
                        help="Type of model to train")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to model configuration YAML file")

    # Data parameters
    parser.add_argument("--data-dir", type=str, default="data/recordings",
                        help="Directory containing session recordings")
    parser.add_argument("--gsr-sampling-rate", type=int, default=32,
                        help="Sampling rate of the GSR signal in Hz")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Frame rate of the video in frames per second")

    # Cross-validation parameters
    parser.add_argument("--cv-method", type=str, default="loso",
                        choices=["loso", "kfold"],
                        help="Cross-validation method (leave-one-subject-out or k-fold)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of folds for k-fold cross-validation")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Fraction of training data to use for validation")

    # Output parameters
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save models and results")
    parser.add_argument("--save-metadata", type=str, default="true",
                        choices=["true", "false"],
                        help="Whether to save training metadata")

    # Utility options
    parser.add_argument("--list-configs", action="store_true",
                        help="List available model configurations and exit")
    parser.add_argument("--create-example-configs", action="store_true",
                        help="Create example configuration files for all model types and exit")

    args = parser.parse_args()

    # Handle utility options
    if args.list_configs:
        list_available_configs()
        sys.exit(0)

    if args.create_example_configs:
        create_example_config_files()
        sys.exit(0)

    # Convert string boolean to actual boolean
    args.save_metadata = args.save_metadata.lower() == "true"

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = Path("data/recordings/models")
    else:
        args.output_dir = Path(args.output_dir)

    # Convert data directory to Path
    args.data_dir = Path(args.data_dir)

    return args


def build_model_from_config(input_shape: Tuple[int, int], model_type: str, config_path: Optional[str] = None):
    """
    Build a model from a configuration file or default configuration.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, num_features)
        model_type (str): Type of model to build
        config_path (str, optional): Path to model configuration YAML file

    Returns:
        object: Model instance
    """
    # Load configuration
    model_config = ModelConfig(model_type)

    if config_path is not None:
        try:
            model_config.load_from_file(Path(config_path))
            logging.info(f"Loaded model configuration from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load model configuration from {config_path}: {e}")
            logging.info("Using default configuration instead")

    # Get the framework from the configuration
    framework = model_config.config.get("framework", "pytorch")

    if framework == "tensorflow" and not TF_AVAILABLE:
        logging.error("TensorFlow framework specified but TensorFlow is not available")
        logging.info("Falling back to PyTorch framework")
        framework = "pytorch"
        model_config.config["framework"] = "pytorch"

    # Log the configuration
    logging.info(f"Building {framework} {model_type} model with configuration:")
    for key, value in model_config.config.items():
        if key != "model_params":
            logging.info(f"  {key}: {value}")
    logging.info("  model_params:")
    for key, value in model_config.config.get("model_params", {}).items():
        logging.info(f"    {key}: {value}")

    # Build the model
    try:
        model = ModelRegistry.create_model(model_type, input_shape, model_config.config)
        logging.info(f"Successfully built {framework} {model_type} model")
        return model
    except Exception as e:
        logging.error(f"Failed to build model: {e}")
        raise


def get_git_commit_hash():
    """
    Get the current git commit hash.

    Returns:
        str: Git commit hash or "unknown" if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def create_training_metadata(model_type: str, model_config: ModelConfig, fold: int, subject_id: str,
                           input_shape: Tuple[int, int], preprocessing_params: Dict[str, Any],
                           training_params: Dict[str, Any], metrics: Dict[str, float]):
    """
    Create metadata about the training process.

    Args:
        model_type (str): Type of model
        model_config (ModelConfig): Model configuration
        fold (int): Fold number
        subject_id (str): Subject ID for LOSO cross-validation
        input_shape (tuple): Shape of the input data
        preprocessing_params (dict): Preprocessing parameters
        training_params (dict): Training parameters
        metrics (dict): Training and validation metrics

    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "model_type": model_type,
        "model_config": model_config.config,
        "fold": fold,
        "subject_id": subject_id,
        "input_shape": input_shape,
        "preprocessing_params": preprocessing_params,
        "training_params": training_params,
        "metrics": metrics
    }

    # Add system information
    try:
        import platform
        metadata["system_info"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }
    except ImportError:
        pass

    # Add package versions
    try:
        import pkg_resources
        metadata["package_versions"] = {
            pkg.key: pkg.version for pkg in pkg_resources.working_set
        }
    except ImportError:
        pass

    return metadata


def save_training_metadata(metadata: Dict[str, Any], output_dir: Path, model_type: str, fold: int, subject_id: str):
    """
    Save training metadata to a JSON file.

    Args:
        metadata (dict): Metadata dictionary
        output_dir (Path): Output directory
        model_type (str): Type of model
        fold (int): Fold number
        subject_id (str): Subject ID for LOSO cross-validation

    Returns:
        Path: Path to the saved metadata file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata filename
    metadata_filename = f"metadata_{model_type}_fold_{fold}_subject_{subject_id}.json"
    metadata_path = output_dir / metadata_filename

    # Save metadata to JSON file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved training metadata to {metadata_path}")

    return metadata_path


def setup_callbacks(model_config, fold, subject_id, output_dir):
    """
    Set up callbacks for model training.

    Args:
        model_config (ModelConfig): Model configuration
        fold (int): Fold number
        subject_id (str): Subject ID for LOSO cross-validation
        output_dir (Path): Output directory

    Returns:
        dict: Callbacks dictionary
    """
    callbacks = {}

    # Get the framework from the configuration
    framework = model_config.config.get("framework", "pytorch")

    # Get training parameters
    train_params = model_config.config.get("train_params", {})

    # Early stopping
    early_stopping = train_params.get("early_stopping", {})
    if early_stopping:
        patience = early_stopping.get("patience", 10)
        monitor = early_stopping.get("monitor", "val_loss")

        if framework == "tensorflow":
            callbacks["early_stopping"] = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True
            )
        else:
            # For PyTorch, early stopping is handled by the model class
            callbacks["early_stopping"] = {
                "patience": patience,
                "monitor": monitor
            }

    # Model checkpoint
    checkpoint = train_params.get("checkpoint", {})
    if checkpoint:
        save_best_only = checkpoint.get("save_best_only", True)
        monitor = checkpoint.get("monitor", "val_loss")

        if framework == "tensorflow":
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Create checkpoint filename
            checkpoint_filename = f"model_fold_{fold}_subject_{subject_id}.keras"
            checkpoint_path = checkpoint_dir / checkpoint_filename

            callbacks["checkpoint"] = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=save_best_only
            )
        else:
            # For PyTorch, checkpointing is handled by the model class
            callbacks["checkpoint"] = {
                "save_best_only": save_best_only,
                "monitor": monitor,
                "filepath": output_dir / f"model_{model_config.config['name']}_fold_{fold}_subject_{subject_id}.pt"
            }

    return callbacks


def main():
    """
    Main function to train a model on the GSR-RGBT dataset.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    logging.info(f"Loading data from {args.data_dir}")
    X, y, subject_ids = load_all_session_data(args.data_dir, args.gsr_sampling_rate, args.video_fps)

    if X is None or y is None or subject_ids is None:
        logging.error("Failed to load data")
        return

    logging.info(f"Loaded data with shape X: {X.shape}, y: {y.shape}")
    logging.info(f"Unique subjects: {set(subject_ids)}")

    # Preprocessing parameters for metadata
    preprocessing_params = {
        "gsr_sampling_rate": args.gsr_sampling_rate,
        "video_fps": args.video_fps,
        "data_dir": str(args.data_dir)
    }

    # Training parameters for metadata
    training_params = {
        "cv_method": args.cv_method,
        "cv_folds": args.cv_folds,
        "validation_split": args.validation_split
    }

    # Cross-validation
    if args.cv_method == "loso":
        # Leave-One-Subject-Out cross-validation
        cv = LeaveOneGroupOut()
        folds = list(cv.split(X, y, subject_ids))
        logging.info(f"Using Leave-One-Subject-Out cross-validation with {len(folds)} folds")
    else:
        # K-fold cross-validation
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        folds = list(cv.split(X, y))
        logging.info(f"Using {args.cv_folds}-fold cross-validation")

    # Initialize metrics for each fold
    all_metrics = []

    # Train a model for each fold
    for fold, (train_idx, test_idx) in enumerate(folds):
        logging.info(f"Fold {fold + 1}/{len(folds)}")

        # Get training and test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # For LOSO, get the subject ID for this fold
        if args.cv_method == "loso":
            test_subject_id = subject_ids[test_idx[0]]
            logging.info(f"Test subject: {test_subject_id}")
        else:
            test_subject_id = f"fold_{fold}"

        # Split training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.validation_split, random_state=42
        )

        logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

        # Scale the data
        scaler = StandardScaler()

        # Reshape to 2D for scaling
        original_shape = X_train.shape
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_val_2d = X_val.reshape(-1, X_val.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])

        # Fit the scaler on the training data and transform all sets
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_val_2d = scaler.transform(X_val_2d)
        X_test_2d = scaler.transform(X_test_2d)

        # Reshape back to 3D
        X_train = X_train_2d.reshape(original_shape)
        X_val = X_val_2d.reshape(X_val.shape[0], original_shape[1], original_shape[2])
        X_test = X_test_2d.reshape(X_test.shape[0], original_shape[1], original_shape[2])

        # Save the scaler
        scaler_path = args.output_dir / f"scaler_{args.model_type}_fold_{fold}_subject_{test_subject_id}.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler to {scaler_path}")

        # Build the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model_from_config(input_shape, args.model_type, args.config_path)

        # Get model configuration
        model_config = ModelConfig(args.model_type)
        if args.config_path is not None:
            try:
                model_config.load_from_file(Path(args.config_path))
            except Exception:
                pass

        # Set up callbacks
        callbacks = setup_callbacks(model_config, fold, test_subject_id, args.output_dir)

        # Train the model
        logging.info("Training model...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=callbacks)

        # Evaluate the model on the test set
        logging.info("Evaluating model on test set...")
        test_metrics = model.evaluate(X_test, y_test)

        # Save the model
        model_path = args.output_dir / f"{args.model_type}_fold_{fold}_subject_{test_subject_id}"
        if model_config.config.get("framework", "pytorch") == "tensorflow":
            model_path = model_path.with_suffix(".keras")
        else:
            model_path = model_path.with_suffix(".pt")

        model.save(str(model_path))
        logging.info(f"Saved model to {model_path}")

        # Combine metrics
        fold_metrics = {
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
            "test_loss": test_metrics.get("loss", 0.0),
            "test_mae": test_metrics.get("mae", 0.0),
            "test_mse": test_metrics.get("mse", 0.0),
            "test_rmse": test_metrics.get("rmse", 0.0),
            "test_r2": test_metrics.get("r2", 0.0)
        }

        # Save training metadata
        if args.save_metadata:
            metadata = create_training_metadata(
                args.model_type, model_config, fold, test_subject_id,
                input_shape, preprocessing_params, training_params, fold_metrics
            )
            save_training_metadata(metadata, args.output_dir, args.model_type, fold, test_subject_id)

        # Add to all metrics
        all_metrics.append({
            "fold": fold,
            "subject_id": test_subject_id,
            "test_loss": fold_metrics["test_loss"],
            "test_mae": fold_metrics["test_mae"],
            "test_mse": fold_metrics["test_mse"],
            "test_rmse": fold_metrics["test_rmse"],
            "test_r2": fold_metrics["test_r2"]
        })

    # Calculate average metrics across folds
    avg_metrics = {
        "test_loss": np.mean([m["test_loss"] for m in all_metrics]),
        "test_mae": np.mean([m["test_mae"] for m in all_metrics]),
        "test_mse": np.mean([m["test_mse"] for m in all_metrics]),
        "test_rmse": np.mean([m["test_rmse"] for m in all_metrics]),
        "test_r2": np.mean([m["test_r2"] for m in all_metrics])
    }

    logging.info("Cross-validation results:")
    for metric, value in avg_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # Save all metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = args.output_dir / f"{args.model_type}_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved cross-validation metrics to {metrics_path}")

    # Add average metrics as a new row
    avg_row = {"fold": "average", "subject_id": "all"}
    avg_row.update(avg_metrics)
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save updated metrics to CSV
    metrics_df.to_csv(metrics_path, index=False)

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
