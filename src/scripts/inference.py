# src/scripts/train_model.py

import logging

# --- Import project modules ---
# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path

import joblib  # Added for saving the scaler
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.processing.feature_engineering import create_dataset_from_session
from src.ml_models.models import build_lstm_model  # Or other models

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


def main():
    """
    Main training loop using Leave-One-Subject-Out (LOSO) cross-validation.
    """
    logging.info("Starting model training pipeline...")

    # 1. Load and process data from all subjects
    X, y, groups = load_all_session_data(
        data_dir=config.OUTPUT_DIR,
        gsr_sampling_rate=config.GSR_SAMPLING_RATE,
        video_fps=config.FPS,
    )

    if X is None:
        return

    # 2. Setup LOSO Cross-Validation
    logo = LeaveOneGroupOut()
    fold_results = []

    models_dir = config.OUTPUT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    logging.info(
        f"Starting Leave-One-Subject-Out cross-validation for {len(np.unique(groups))} subjects."
    )

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subject_id = groups[test_idx][0]
        logging.info(f"--- Fold {fold+1}: Testing on subject {test_subject_id} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 3. Scale the features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)

        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
            X_test.shape
        )

        # --- ADDED: Save the scaler for this fold ---
        scaler_path = models_dir / f"scaler_fold_{fold+1}_test_{test_subject_id}.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler for fold {fold+1} to {scaler_path}")

        # 4. Build and train the model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        model = build_lstm_model(input_shape=input_shape)

        model_save_path = (
            models_dir / f"model_fold_{fold+1}_test_{test_subject_id}.keras"
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_save_path), save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(models_dir / f"logs/fold_{fold+1}")
            ),
        ]

        history = model.fit(
            X_train_scaled,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=2,
        )

        # 5. Evaluate the model on the held-out subject
        logging.info(f"Evaluating model on subject {test_subject_id}...")
        test_loss, test_mse = model.evaluate(X_test_scaled, y_test, verbose=0)

        logging.info(
            f"Test Results for subject {test_subject_id}: MAE Loss = {test_loss:.4f}, MSE = {test_mse:.4f}"
        )
        fold_results.append(
            {"subject": test_subject_id, "mae": test_loss, "mse": test_mse}
        )

    # 6. Report final results
    results_df = pd.DataFrame(fold_results)
    logging.info("\n--- Cross-Validation Summary ---")
    print(results_df)

    mean_mae = results_df["mae"].mean()
    std_mae = results_df["mae"].std()
    logging.info(f"\nAverage MAE: {mean_mae:.4f} (+/- {std_mae:.4f})")

    results_path = config.OUTPUT_DIR / "cross_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    logging.info(f"Cross-validation results saved to {results_path}")


if __name__ == "__main__":
    main()
