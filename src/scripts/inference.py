# src/scripts/inference.py

import logging
import argparse

# --- Import project modules ---
# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.processing.feature_engineering import create_dataset_from_session
from src.ml_models.models import build_lstm_model, build_ae_model, build_vae_model
from src.ml_models.model_config import ModelConfig

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def load_session_data(
    session_path: Path, gsr_sampling_rate: int, video_fps: int
) -> tuple:
    """
    Loads and processes data from a single session directory.

    Args:
        session_path (Path): The directory containing the session data.
        gsr_sampling_rate (int): The sampling rate of the GSR sensor.
        video_fps (int): The FPS of the video recordings.

    Returns:
        A tuple containing (X, y) for the session, or (None, None) if processing fails.
    """
    if not session_path.exists() or not session_path.is_dir():
        logging.error(f"Session directory not found: {session_path}")
        return None, None

    subject_id = session_path.name.split("_")[1]
    logging.info(f"--- Processing session for subject: {subject_id} ---")

    dataset = create_dataset_from_session(
        session_path, gsr_sampling_rate, video_fps
    )

    if not dataset:
        logging.error(f"Failed to process session: {session_path}")
        return None, None

    return dataset


def parse_arguments():
    """
    Parse command line arguments for the inference script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run inference with trained ML models")

    # Model selection and configuration
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="lstm",
        choices=["lstm", "autoencoder", "vae"],
        help="Type of model to use for inference"
    )

    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to the trained model file (.keras)"
    )

    parser.add_argument(
        "--scaler-path", 
        type=str, 
        required=True,
        help="Path to the saved scaler file (.joblib)"
    )

    # Data selection
    parser.add_argument(
        "--subject-id", 
        type=str, 
        required=True,
        help="ID of the subject to run inference on (e.g., 'Subject01')"
    )

    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save prediction results (defaults to config.OUTPUT_DIR)"
    )

    return parser.parse_args()


def main():
    """
    Main inference function to run predictions on a specific subject.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True, parents=True)

    # Construct the session path
    subject_session_path = config.OUTPUT_DIR / f"Subject_{args.subject_id}"

    logging.info(f"Starting inference for subject {args.subject_id} using {args.model_type} model")

    # 1. Load the session data
    X, y = load_session_data(
        session_path=subject_session_path,
        gsr_sampling_rate=config.GSR_SAMPLING_RATE,
        video_fps=config.FPS,
    )

    if X is None or y is None:
        logging.error(f"Failed to load data for subject {args.subject_id}")
        return

    logging.info(f"Loaded {len(X)} samples for subject {args.subject_id}")

    # 2. Load the scaler and scale the features
    try:
        scaler = joblib.load(args.scaler_path)
        logging.info(f"Loaded scaler from {args.scaler_path}")
    except Exception as e:
        logging.error(f"Failed to load scaler: {e}")
        return

    # Scale the features
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)

    # 3. Load the trained model
    try:
        model = tf.keras.models.load_model(args.model_path)
        logging.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 4. Run inference
    logging.info("Running inference...")

    if args.model_type == "lstm":
        # For LSTM models, predict the target values
        predictions = model.predict(X_scaled)

        # Create a DataFrame with ground truth and predictions
        results_df = pd.DataFrame({
            "timestamp": range(len(y)),
            "ground_truth": y.flatten(),
            "prediction": predictions.flatten()
        })

        # Calculate metrics
        mae = np.mean(np.abs(results_df["ground_truth"] - results_df["prediction"]))
        mse = np.mean(np.square(results_df["ground_truth"] - results_df["prediction"]))
        rmse = np.sqrt(mse)

        logging.info(f"Inference Results for subject {args.subject_id}:")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  RMSE: {rmse:.4f}")

    else:
        # For autoencoder models, reconstruct the input
        reconstructions = model.predict(X_scaled)

        # Calculate reconstruction error
        mse = np.mean(np.square(X_scaled - reconstructions))
        mae = np.mean(np.abs(X_scaled - reconstructions))

        logging.info(f"Reconstruction Results for subject {args.subject_id}:")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  MAE: {mae:.4f}")

        # Create a simplified results DataFrame for autoencoders
        results_df = pd.DataFrame({
            "timestamp": range(len(X)),
            "reconstruction_mse": np.mean(np.square(X_scaled - reconstructions), axis=(1, 2)),
            "reconstruction_mae": np.mean(np.abs(X_scaled - reconstructions), axis=(1, 2))
        })

    # 5. Save the results
    prediction_file = predictions_dir / f"predictions_{args.subject_id}_{args.model_type}.csv"
    results_df.to_csv(prediction_file, index=False)
    logging.info(f"Saved prediction results to {prediction_file}")

    # 6. Return success
    logging.info(f"Inference completed successfully for subject {args.subject_id}")
    return results_df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
    )
    main()
