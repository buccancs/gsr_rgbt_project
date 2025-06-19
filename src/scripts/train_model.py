# src/scripts/train_model.py

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

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.processing.feature_engineering import create_dataset_from_session
from src.ml_models.model_interface import ModelRegistry
from src.ml_models.model_config import ModelConfig, list_available_configs, create_example_config_files

# Import PyTorch models (this will register them with the ModelRegistry)
import src.ml_models.pytorch_models

# For backward compatibility with TensorFlow models
try:
    import tensorflow as tf
    from src.ml_models.models import build_lstm_model, build_ae_model, build_vae_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Only PyTorch models will be supported.")

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def load_all_session_data(
    data_dir: Path, gsr_sampling_rate: int, video_fps: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads and processes data from all session directories for model training.

    This function iterates through all subject session folders in the data directory,
    processes each session using the create_dataset_from_session function, and combines
    the results into unified arrays for training. It also creates a groups array
    containing subject IDs for each sample, which can be used for leave-one-subject-out
    cross-validation.

    Args:
        data_dir (Path): The root directory containing all subject session folders.
            Each folder should follow the naming convention "Subject_<ID>_<DATE>_<TIME>".
        gsr_sampling_rate (int): The sampling rate of the GSR sensor in Hz.
            Used for preprocessing and feature extraction.
        video_fps (int): The frames per second of the video recordings.
            Used for synchronizing video features with GSR data.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]: 
            A tuple containing:
            - all_X: Feature windows with shape (n_samples, window_size, n_features)
            - all_y: Target values with shape (n_samples,)
            - all_groups: Subject IDs for each sample with shape (n_samples,)

            If no data could be processed, returns (None, None, None).
    """
    all_X, all_y, all_groups = [], [], []

    session_paths = [
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Subject_")
    ]
    if not session_paths:
        logging.error(
            f"No session directories found in {data_dir}. Cannot proceed with training."
        )
        return None, None, None

    for session_path in sorted(session_paths):
        subject_id = session_path.name.split("_")[1]
        logging.info(f"--- Processing session for subject: {subject_id} ---")

        dataset = create_dataset_from_session(
            session_path, gsr_sampling_rate, video_fps
        )

        if dataset:
            X, y = dataset
            all_X.append(X)
            all_y.append(y)
            # Create a group identifier for each sample from this subject
            all_groups.extend([subject_id] * len(y))

    if not all_X:
        logging.error("Failed to process any sessions. No data available for training.")
        return None, None, None

    return np.vstack(all_X), np.concatenate(all_y), np.array(all_groups)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    This function sets up the argument parser with all available options for
    configuring the model training process. It includes options for:

    - Model selection and configuration
    - Output directory specification
    - Cross-validation settings
    - Validation split ratio
    - Metadata saving preferences

    Returns:
        argparse.Namespace: Parsed command line arguments containing all the
            configuration options specified by the user or their default values.
    """
    parser = argparse.ArgumentParser(description="Train ML models for GSR prediction")

    # Model selection and configuration
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="lstm",
        choices=list_available_configs(),
        help=f"Type of model to train. Available options: {', '.join(list_available_configs())}"
    )

    parser.add_argument(
        "--config-path", 
        type=str, 
        help="Path to a YAML configuration file for the model"
    )

    parser.add_argument(
        "--create-example-configs", 
        action="store_true",
        help="Create example configuration files for all model types and exit"
    )

    # Output directory for models and results
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save models and results (defaults to config.OUTPUT_DIR)"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique identifier for this experiment run, used for organizing outputs"
    )

    parser.add_argument(
        "--milestone-epochs",
        type=str,
        default="",
        help="Comma-separated list of epoch numbers at which to save model checkpoints (e.g., '10,20,50')"
    )

    # Cross-validation options
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=0,
        help="Number of cross-validation folds (0 for LOSO cross-validation)"
    )

    # Validation split options
    parser.add_argument(
        "--validation-split", 
        type=float, 
        default=0.2,
        help="Fraction of training data to use for validation (default: 0.2)"
    )

    # Save metadata options
    parser.add_argument(
        "--save-metadata", 
        action="store_true",
        default=True,
        help="Save metadata about the training process"
    )

    return parser.parse_args()


def build_model_from_config(
    input_shape: Tuple[int, int], 
    model_type: str, 
    config_path: Optional[str] = None
) -> 'BaseModel':
    """
    Build a model based on the specified type and configuration.

    This function creates a model instance based on the provided model type and
    configuration. It supports both PyTorch and TensorFlow frameworks, with PyTorch
    being the primary implementation. The function follows these steps:

    1. Create a ModelConfig object from the provided model type and optional config file
    2. Determine the framework (PyTorch or TensorFlow) from the configuration
    3. For PyTorch models, use the ModelRegistry to create the model
    4. For TensorFlow models, use the legacy builder functions if TensorFlow is available

    The function includes error handling to provide clear error messages when model
    creation fails.

    Args:
        input_shape (Tuple[int, int]): Shape of the input data as (window_size, features),
            where window_size is the number of time steps and features is the number of
            input features at each time step.
        model_type (str): Type of model to build (e.g., 'lstm', 'autoencoder', 'vae').
            Must be one of the registered model types in ModelRegistry or a supported
            TensorFlow model type.
        config_path (Optional[str], optional): Path to a YAML configuration file that
            contains custom model parameters. If provided, these parameters will override
            the default configuration for the specified model type. Defaults to None.

    Returns:
        BaseModel: The built model instance that implements the BaseModel interface,
            which provides a consistent API regardless of the underlying framework.

    Raises:
        ValueError: If the model type is not supported or if there's an error in the
            model configuration.
        ImportError: If TensorFlow is requested but not available.
    """
    # Create model configuration
    if config_path:
        model_config = ModelConfig(config_name=model_type, config_path=Path(config_path))
    else:
        model_config = ModelConfig(config_name=model_type)

    # Get the framework from the configuration
    framework = model_config.get_framework()

    # For PyTorch models, use the ModelRegistry
    if framework == "pytorch":
        try:
            # Create the model using the registry
            return ModelRegistry.create_model(
                name=model_type,
                input_shape=input_shape,
                config=model_config.get_config()
            )
        except ValueError as e:
            logging.error(f"Error creating PyTorch model: {e}")
            # Raise the error instead of falling back to TensorFlow
            # This makes it clearer what went wrong and prevents masking of issues
            raise ValueError(f"Failed to create PyTorch model '{model_type}'. Please check your model configuration. Error: {e}")

    # For TensorFlow models (or fallback)
    if framework == "tensorflow":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot build TensorFlow model.")

        # Build the appropriate model type using the legacy builders
        if model_type == "lstm" or model_type == "tf_lstm":
            return build_lstm_model(input_shape=input_shape, config=model_config)
        elif model_type == "autoencoder" or model_type == "tf_autoencoder":
            return build_ae_model(input_shape=input_shape, config=model_config)
        elif model_type == "vae" or model_type == "tf_vae":
            return build_vae_model(input_shape=input_shape, config=model_config)
        else:
            raise ValueError(f"Unsupported TensorFlow model type: {model_type}")

    # If we get here, the framework is not supported
    raise ValueError(f"Unsupported framework: {framework}")


def get_git_commit_hash() -> str:
    """
    Get the current Git commit hash.

    Returns:
        str: The current Git commit hash, or "unknown" if not in a Git repository.
    """
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except Exception:
        return "unknown"


def create_training_metadata(
    model_type: str,
    model_config: ModelConfig,
    fold: int,
    subject_id: str,
    input_shape: Tuple[int, int],
    preprocessing_params: Dict[str, Any],
    training_params: Dict[str, Any],
    metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create metadata about the training process for documentation and reproducibility.

    This function collects all relevant information about a model training run,
    including model configuration, preprocessing parameters, training parameters,
    and evaluation metrics. The metadata is structured as a dictionary that can be
    saved to a JSON file for later analysis or to reproduce the experiment.

    The metadata includes a timestamp to track when the training was performed and
    the complete model configuration to ensure reproducibility.

    Args:
        model_type (str): Type of model being trained (e.g., 'lstm', 'autoencoder')
        model_config (ModelConfig): Model configuration object containing all model
            hyperparameters and settings
        fold (int): Current cross-validation fold number (0-based index)
        subject_id (str): ID of the subject being used for testing in this fold
        input_shape (Tuple[int, int]): Shape of the input data as (window_size, features)
        preprocessing_params (Dict[str, Any]): Parameters used for preprocessing the data,
            such as normalization method, window size, etc.
        training_params (Dict[str, Any]): Parameters used for training the model,
            such as batch size, learning rate, number of epochs, etc.
        metrics (Dict[str, float]): Evaluation metrics from the model's performance,
            such as MSE, MAE, R^2, etc.

    Returns:
        Dict[str, Any]: A comprehensive dictionary containing all metadata about the
            training process, which can be saved to a file for documentation and
            reproducibility purposes.
    """
    # Create metadata dictionary
    metadata = {
        "model_type": model_type,
        "model_name": model_config.get_model_name(),
        "framework": model_config.get_framework(),
        "fold": fold + 1,
        "subject_id": subject_id,
        "input_shape": input_shape,
        "preprocessing": preprocessing_params,
        "training": training_params,
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "config": model_config.get_config()
    }

    return metadata


def save_training_metadata(
    metadata: Dict[str, Any],
    output_dir: Path,
    model_type: str,
    fold: int,
    subject_id: str
) -> Path:
    """
    Save metadata about the training process.

    Args:
        metadata (Dict[str, Any]): Metadata about the training process
        output_dir (Path): Directory to save metadata
        model_type (str): Type of model being trained
        fold (int): Current fold number
        subject_id (str): ID of the subject being tested on

    Returns:
        Path: Path to the saved metadata file
    """
    # Create metadata directory
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)

    # Create metadata file path
    metadata_path = metadata_dir / f"{model_type}_fold_{fold+1}_subject_{subject_id}_metadata.json"

    # Save metadata to file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved training metadata to {metadata_path}")
    return metadata_path


def setup_callbacks(model_config, fold, subject_id, output_dir):
    """
    Set up training callbacks based on the model configuration.

    For TensorFlow models, this returns a list of Keras callbacks.
    For PyTorch models, this function is not used as callbacks are handled internally.

    Args:
        model_config (ModelConfig): Model configuration object
        fold (int): Current fold number
        subject_id (str): ID of the subject being tested on
        output_dir (Path): Directory to save models and logs

    Returns:
        list: List of Keras callbacks (for TensorFlow models only)
    """
    # Check if we're using TensorFlow
    if not TENSORFLOW_AVAILABLE or model_config.get_framework() != "tensorflow":
        logging.warning("setup_callbacks is only used for TensorFlow models")
        return []

    callbacks = []
    fit_params = model_config.get_fit_params()
    callback_configs = fit_params.get("callbacks", {})

    # Model checkpoint callback
    if "model_checkpoint" in callback_configs:
        model_save_path = output_dir / f"models/{model_config.get_model_name()}_fold_{fold+1}_subject_{subject_id}.keras"
        model_save_path.parent.mkdir(exist_ok=True, parents=True)

        checkpoint_config = callback_configs["model_checkpoint"]
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_save_path),
                save_best_only=checkpoint_config.get("save_best_only", True),
                monitor=checkpoint_config.get("monitor", "val_loss")
            )
        )

    # Early stopping callback
    if "early_stopping" in callback_configs:
        early_stopping_config = callback_configs["early_stopping"]
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_loss"),
                patience=early_stopping_config.get("patience", 10),
                restore_best_weights=early_stopping_config.get("restore_best_weights", True)
            )
        )

    # TensorBoard callback
    if "tensorboard" in callback_configs:
        log_dir = output_dir / f"logs/{model_config.get_model_name()}/fold_{fold+1}"
        log_dir.mkdir(exist_ok=True, parents=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))
        )

    return callbacks


def main():
    """
    Main training loop using Leave-One-Subject-Out (LOSO) cross-validation.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Create example configurations if requested
    if args.create_example_configs:
        config_dir = project_root / "configs" / "models"
        create_example_config_files(config_dir)
        logging.info(f"Example configuration files created in {config_dir}")
        return

    # Set output directory
    base_output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    base_output_dir.mkdir(exist_ok=True, parents=True)

    # Create experiment directory structure
    if args.experiment_id:
        # Use the provided experiment ID
        experiment_id = args.experiment_id
    else:
        # Generate a timestamp-based experiment ID
        experiment_id = f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Parse milestone epochs if provided
    milestone_epochs = []
    if args.milestone_epochs:
        try:
            milestone_epochs = [int(epoch) for epoch in args.milestone_epochs.split(',')]
            logging.info(f"Will save checkpoints at epochs: {milestone_epochs}")
        except ValueError:
            logging.warning(f"Invalid milestone epochs format: {args.milestone_epochs}. Expected comma-separated integers.")
            milestone_epochs = []

    # Create experiment directory
    experiment_dir = base_output_dir / "experiments" / experiment_id
    experiment_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories for different artifact types
    models_dir = experiment_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    scalers_dir = experiment_dir / "scalers"
    scalers_dir.mkdir(exist_ok=True, parents=True)

    metadata_dir = experiment_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True, parents=True)

    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    # Use experiment directory as the output directory
    output_dir = experiment_dir

    logging.info(f"Experiment ID: {experiment_id}")
    logging.info(f"Experiment directory: {experiment_dir}")

    logging.info(f"Starting model training pipeline for model type: {args.model_type}")

    # 1. Load and process data from all subjects
    X, y, groups = load_all_session_data(
        data_dir=config.OUTPUT_DIR,
        gsr_sampling_rate=config.GSR_SAMPLING_RATE,
        video_fps=config.FPS,
    )

    if X is None:
        return

    # 2. Setup Cross-Validation
    if args.cv_folds <= 0:
        # Use Leave-One-Subject-Out cross-validation
        cv = LeaveOneGroupOut()
        logging.info(
            f"Using Leave-One-Subject-Out cross-validation for {len(np.unique(groups))} subjects."
        )
    else:
        # Use k-fold cross-validation
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        logging.info(f"Using {args.cv_folds}-fold cross-validation.")

    fold_results = []

    # Create model configuration
    if args.config_path:
        model_config = ModelConfig(config_name=args.model_type, config_path=Path(args.config_path))
    else:
        model_config = ModelConfig(config_name=args.model_type)

    # Save the configuration used for this run
    config_save_path = models_dir / f"{args.model_type}_config_used.yaml"
    model_config.save_to_file(config_save_path)
    logging.info(f"Saved model configuration to {config_save_path}")

    # 3. Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        # For LOSO, get the subject ID being tested on
        if isinstance(cv, LeaveOneGroupOut):
            subject_id = groups[test_idx][0]
            logging.info(f"--- Fold {fold+1}: Testing on subject {subject_id} ---")
        else:
            subject_id = f"fold{fold+1}"
            logging.info(f"--- Fold {fold+1} ---")

        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

        # Split training data into training and validation sets
        if args.validation_split > 0:
            # If using LOSO, we need to ensure that subjects are not split between train and validation
            if isinstance(cv, LeaveOneGroupOut):
                # Get groups for training/validation data
                groups_train_val = groups[train_idx]

                # Get unique subjects in training/validation data
                unique_subjects = np.unique(groups_train_val)

                # Randomly select subjects for validation
                np.random.seed(42)  # For reproducibility
                val_subjects = np.random.choice(
                    unique_subjects, 
                    size=max(1, int(len(unique_subjects) * args.validation_split)), 
                    replace=False
                )

                # Create masks for training and validation
                train_mask = ~np.isin(groups_train_val, val_subjects)
                val_mask = np.isin(groups_train_val, val_subjects)

                # Split data
                X_train, X_val = X_train_val[train_mask], X_train_val[val_mask]
                y_train, y_val = y_train_val[train_mask], y_train_val[val_mask]

                logging.info(f"Split training data into {len(X_train)} training samples and {len(X_val)} validation samples")
                logging.info(f"Training subjects: {np.unique(groups_train_val[train_mask])}")
                logging.info(f"Validation subjects: {np.unique(groups_train_val[val_mask])}")
            else:
                # If not using LOSO, we can use a simple random split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, 
                    test_size=args.validation_split, 
                    random_state=42
                )
                logging.info(f"Split training data into {len(X_train)} training samples and {len(X_val)} validation samples")
        else:
            # If validation split is 0, use all training data for training
            X_train, X_val = X_train_val, X_train_val
            y_train, y_val = y_train_val, y_train_val
            logging.info("Using all training data for training (no validation split)")

        # 4. Scale the features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)

        # Transform all datasets
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # Save the scaler
        scaler_path = scalers_dir / f"scaler_{args.model_type}_fold_{fold+1}_subject_{subject_id}.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler to {scaler_path}")

        # 5. Build and train the model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        model = build_model_from_config(input_shape, args.model_type, args.config_path)

        # Determine if we're using a PyTorch or TensorFlow model
        is_pytorch_model = hasattr(model, 'fit') and callable(getattr(model, 'fit'))

        # Use the models directory created earlier
        model_save_dir = models_dir

        # Create preprocessing parameters dictionary for metadata
        preprocessing_params = {
            "gsr_sampling_rate": config.GSR_SAMPLING_RATE,
            "video_fps": config.FPS,
            "scaler": "StandardScaler",
            "validation_split": args.validation_split,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        }

        # Create training parameters dictionary for metadata
        training_params = {
            "framework": model_config.get_framework(),
            "fit_params": model_config.get_fit_params()
        }

        if is_pytorch_model:
            # For PyTorch models, use the BaseModel interface
            logging.info(f"Training PyTorch model for {args.model_type}...")

            # Train the model
            history = model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                model_save_dir=checkpoints_dir,
                fold_num=fold,
                milestone_epochs=milestone_epochs
            )

            # Save the model
            model_save_path = model_save_dir / f"{args.model_type}_fold_{fold+1}_subject_{subject_id}.pt"
            model.save(str(model_save_path))
            logging.info(f"Saved model to {model_save_path}")

            # Update training parameters with history
            training_params["history"] = history

        else:
            # For TensorFlow models, use the legacy approach
            logging.info(f"Training TensorFlow model for {args.model_type}...")

            # Setup callbacks
            callbacks = setup_callbacks(model_config, fold, subject_id, output_dir)

            # Get training parameters from config
            fit_params = model_config.get_fit_params()

            # Train the model
            history = model.fit(
                X_train_scaled,
                y_train,
                epochs=fit_params.get("epochs", 100),
                batch_size=fit_params.get("batch_size", 32),
                validation_data=(X_val_scaled, y_val),
                callbacks=callbacks,
                verbose=2,
            )

            # Update training parameters with history
            training_params["history"] = {
                "loss": history.history.get("loss", []),
                "val_loss": history.history.get("val_loss", [])
            }

        # 6. Evaluate the model on the held-out data
        logging.info(f"Evaluating model on {'subject ' + subject_id if isinstance(cv, LeaveOneGroupOut) else 'test set'}...")

        if is_pytorch_model:
            # For PyTorch models, use the evaluate method
            metrics = model.evaluate(X_test_scaled, y_test)

            # Log the metrics
            metrics_str = ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
            logging.info(
                f"Test Results for {'subject ' + subject_id if isinstance(cv, LeaveOneGroupOut) else 'fold ' + str(fold+1)}: "
                f"{metrics_str}"
            )

            # Create result dictionary
            result = {"fold": fold+1, "subject": subject_id}
            result.update(metrics)

        else:
            # For TensorFlow models, use the legacy approach
            if args.model_type == "lstm" or args.model_type == "tf_lstm":
                test_metrics = model.evaluate(X_test_scaled, y_test, verbose=0)

                # Handle different return formats from model.evaluate()
                if isinstance(test_metrics, list):
                    test_loss = test_metrics[0]
                    test_mse = test_metrics[1] if len(test_metrics) > 1 else None
                else:
                    test_loss = test_metrics
                    test_mse = None

                metric_names = model.metrics_names

                logging.info(
                    f"Test Results for {'subject ' + subject_id if isinstance(cv, LeaveOneGroupOut) else 'fold ' + str(fold+1)}: "
                    f"{metric_names[0]} = {test_loss:.4f}" + 
                    (f", {metric_names[1]} = {test_mse:.4f}" if test_mse is not None else "")
                )

                result = {"fold": fold+1, "subject": subject_id, metric_names[0]: test_loss}
                if test_mse is not None:
                    result[metric_names[1]] = test_mse

                # Create metrics dictionary for metadata
                metrics = {metric_names[0]: test_loss}
                if test_mse is not None:
                    metrics[metric_names[1]] = test_mse

            else:
                # For autoencoder models, calculate reconstruction error
                predictions = model.predict(X_test_scaled)
                mse = np.mean(np.square(X_test_scaled - predictions))
                mae = np.mean(np.abs(X_test_scaled - predictions))

                logging.info(
                    f"Test Results for {'subject ' + subject_id if isinstance(cv, LeaveOneGroupOut) else 'fold ' + str(fold+1)}: "
                    f"MSE = {mse:.4f}, MAE = {mae:.4f}"
                )

                result = {"fold": fold+1, "subject": subject_id, "mse": mse, "mae": mae}

                # Create metrics dictionary for metadata
                metrics = {"mse": mse, "mae": mae}

        # Save metadata if requested
        if args.save_metadata:
            # Create metadata
            metadata = create_training_metadata(
                model_type=args.model_type,
                model_config=model_config,
                fold=fold,
                subject_id=subject_id,
                input_shape=input_shape,
                preprocessing_params=preprocessing_params,
                training_params=training_params,
                metrics=metrics
            )

            # Save metadata
            save_training_metadata(
                metadata=metadata,
                output_dir=metadata_dir,
                model_type=args.model_type,
                fold=fold,
                subject_id=subject_id
            )

        fold_results.append(result)

    # 7. Report final results
    results_df = pd.DataFrame(fold_results)
    logging.info("\n--- Cross-Validation Summary ---")
    print(results_df)

    # Calculate average metrics
    metric_columns = [col for col in results_df.columns if col not in ["fold", "subject"]]
    for metric in metric_columns:
        mean_metric = results_df[metric].mean()
        std_metric = results_df[metric].std()
        logging.info(f"\nAverage {metric.upper()}: {mean_metric:.4f} (+/- {std_metric:.4f})")

    # Save results to a CSV file
    results_path = output_dir / f"cross_validation_results_{args.model_type}.csv"
    results_df.to_csv(results_path, index=False)
    logging.info(f"Cross-validation results saved to {results_path}")


if __name__ == "__main__":
    main()
