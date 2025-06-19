# src/scripts/train_model.py

import logging
import argparse

# --- Import project modules ---
# Add the project root to the Python path to allow for absolute imports
import sys
import os
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.processing.feature_engineering import create_dataset_from_session
from src.ml_models.models import build_lstm_model, build_ae_model, build_vae_model
from src.ml_models.model_config import ModelConfig, list_available_configs, create_example_config_files

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def load_all_session_data(
    data_dir: Path, gsr_sampling_rate: int, video_fps: int
) -> tuple:
    """
    Loads and processes data from all session directories.

    Args:
        data_dir (Path): The root directory containing all subject session folders.
        gsr_sampling_rate (int): The sampling rate of the GSR sensor.
        video_fps (int): The FPS of the video recordings.

    Returns:
        A tuple containing (all_X, all_y, all_groups), where 'groups' is an
        array of subject IDs for cross-validation.
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


def parse_arguments():
    """
    Parse command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments
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

    # Cross-validation options
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=0,
        help="Number of cross-validation folds (0 for LOSO cross-validation)"
    )

    return parser.parse_args()


def build_model_from_config(input_shape, model_type, config_path=None):
    """
    Build a model based on the specified type and configuration.

    Args:
        input_shape (tuple): Shape of the input data (window_size, features)
        model_type (str): Type of model to build ('lstm', 'autoencoder', 'vae')
        config_path (str, optional): Path to a YAML configuration file

    Returns:
        tf.keras.Model: The built and compiled model
    """
    # Create model configuration
    if config_path:
        model_config = ModelConfig(config_name=model_type, config_path=Path(config_path))
    else:
        model_config = ModelConfig(config_name=model_type)

    # Build the appropriate model type
    if model_type == "lstm":
        return build_lstm_model(input_shape=input_shape, config=model_config)
    elif model_type == "autoencoder":
        return build_ae_model(input_shape=input_shape, config=model_config)
    elif model_type == "vae":
        return build_vae_model(input_shape=input_shape, config=model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def setup_callbacks(model_config, fold, subject_id, output_dir):
    """
    Set up training callbacks based on the model configuration.

    Args:
        model_config (ModelConfig): Model configuration object
        fold (int): Current fold number
        subject_id (str): ID of the subject being tested on
        output_dir (Path): Directory to save models and logs

    Returns:
        list: List of Keras callbacks
    """
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
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

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
    config_save_path = output_dir / f"models/{args.model_type}_config_used.yaml"
    config_save_path.parent.mkdir(exist_ok=True, parents=True)
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

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 4. Scale the features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)

        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # Save the scaler
        scaler_path = output_dir / f"models/scaler_{args.model_type}_fold_{fold+1}_subject_{subject_id}.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler to {scaler_path}")

        # 5. Build and train the model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        model = build_model_from_config(input_shape, args.model_type, args.config_path)

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
            validation_split=fit_params.get("validation_split", 0.2),
            callbacks=callbacks,
            verbose=2,
        )

        # 6. Evaluate the model on the held-out data
        logging.info(f"Evaluating model on {'subject ' + subject_id if isinstance(cv, LeaveOneGroupOut) else 'test set'}...")

        # For LSTM models, evaluate with standard metrics
        if args.model_type == "lstm":
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

        # For autoencoder models, calculate reconstruction error
        else:
            predictions = model.predict(X_test_scaled)
            mse = np.mean(np.square(X_test_scaled - predictions))
            mae = np.mean(np.abs(X_test_scaled - predictions))

            logging.info(
                f"Test Results for {'subject ' + subject_id if isinstance(cv, LeaveOneGroupOut) else 'fold ' + str(fold+1)}: "
                f"MSE = {mse:.4f}, MAE = {mae:.4f}"
            )

            result = {"fold": fold+1, "subject": subject_id, "mse": mse, "mae": mae}

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
