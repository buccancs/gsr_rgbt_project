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
import torch
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.ml_pipeline.feature_engineering.feature_engineering import create_dataset_from_session
from src.ml_models.model_interface import ModelRegistry, BaseModel
from src.ml_models.model_config import ModelConfig

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
# Configure logging once at the module level for consistency
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def load_session_data(
    session_path: Path, gsr_sampling_rate: int, video_fps: int
) -> tuple:
    """
    Loads and processes data from a single session directory.

    This function serves as a wrapper around the feature engineering pipeline,
    handling error checking and logging. It extracts the subject ID from the
    session directory name and calls the dataset creation function.

    Args:
        session_path (Path): The directory containing the session data.
        gsr_sampling_rate (int): The sampling rate of the GSR sensor in Hz.
        video_fps (int): The frames per second of the video recordings.

    Returns:
        tuple: A tuple containing (X, y) for the session, where:
              - X is the feature matrix (windows of multimodal features)
              - y is the target vector (GSR values)
              Returns (None, None) if processing fails.
    """
    # Validate input path
    if not session_path.exists() or not session_path.is_dir():
        logging.error(f"Session directory not found: {session_path}")
        return None, None

    try:
        # Extract subject ID from directory name (format: Subject_ID_TIMESTAMP)
        subject_id = session_path.name.split("_")[1]
        logging.info(f"--- Processing session for subject: {subject_id} ---")

        # Process the session data through the feature engineering pipeline
        # This creates aligned, windowed features from GSR and video data
        logging.info(f"Creating dataset from session: {session_path}")
        dataset = create_dataset_from_session(
            session_path, gsr_sampling_rate, video_fps
        )

        # Validate the returned dataset
        if not dataset:
            logging.error(f"Failed to process session: {session_path}")
            return None, None

        X, y = dataset
        logging.info(f"Successfully created dataset with {len(X)} samples and {X.shape[-1]} features")

        return dataset

    except Exception as e:
        logging.error(f"Error processing session data: {e}")
        return None, None


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

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a specific model checkpoint to use instead of the full model"
    )

    return parser.parse_args()


def main():
    """
    Main inference function to run predictions on a specific subject.

    This function orchestrates the entire inference pipeline:
    1. Loads and processes session data
    2. Scales the features using a pre-trained scaler
    3. Loads the trained model
    4. Runs inference based on the model type
    5. Calculates performance metrics
    6. Saves the results to a CSV file

    The function handles different model types (LSTM, autoencoder, VAE) with
    appropriate processing for each.

    Returns:
        pd.DataFrame: DataFrame containing the prediction results, or None if inference fails
    """
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Set output directory
        output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True, parents=True)

        # Construct the session path
        subject_session_path = config.OUTPUT_DIR / f"Subject_{args.subject_id}"

        logging.info(f"Starting inference for subject {args.subject_id} using {args.model_type} model")

        # 1. Load and process the session data
        logging.info("Step 1/5: Loading session data...")
        X, y = load_session_data(
            session_path=subject_session_path,
            gsr_sampling_rate=config.GSR_SAMPLING_RATE,
            video_fps=config.FPS,
        )

        if X is None or y is None:
            logging.error(f"Failed to load data for subject {args.subject_id}")
            return None

        logging.info(f"Loaded {len(X)} samples for subject {args.subject_id}")

        # 2. Load the scaler and scale the features
        logging.info("Step 2/5: Scaling features...")
        try:
            scaler = joblib.load(args.scaler_path)
            logging.info(f"Loaded scaler from {args.scaler_path}")
        except Exception as e:
            logging.error(f"Failed to load scaler: {e}")
            return None

        # Scale the features more efficiently
        # Reshape to 2D for scaling, then back to original shape
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_reshaped).reshape(original_shape)

        # Free memory
        del X_reshaped

        # 3. Load the trained model
        logging.info("Step 3/5: Loading model...")
        try:
            # Check if the model is a PyTorch model (.pt extension) or TensorFlow model (.keras extension)
            model_path = args.model_path
            is_pytorch_model = model_path.endswith('.pt')

            if is_pytorch_model:
                # For PyTorch models, use the appropriate model class based on the model type
                if args.model_type.startswith('pytorch_') or args.model_type in ['lstm', 'autoencoder', 'vae']:
                    # Get the model class from the registry
                    model_type = args.model_type
                    # If it's an alias, resolve it
                    if model_type in ['lstm', 'autoencoder', 'vae']:
                        model_type = f"pytorch_{model_type}"

                    # Get the model class from the registry
                    model_factories = {
                        'pytorch_lstm': src.ml_models.pytorch_models.PyTorchLSTMModel,
                        'pytorch_autoencoder': src.ml_models.pytorch_models.PyTorchAutoencoderModel,
                        'pytorch_vae': src.ml_models.pytorch_models.PyTorchVAEModel
                    }

                    if model_type in model_factories:
                        if args.checkpoint_path:
                            # Load the model structure from the model path
                            model = model_factories[model_type].load(model_path)

                            # Load the state_dict from the checkpoint path
                            checkpoint_path = Path(args.checkpoint_path)
                            if checkpoint_path.exists():
                                logging.info(f"Loading checkpoint from {checkpoint_path}")
                                state_dict = torch.load(str(checkpoint_path))
                                model.model.load_state_dict(state_dict)
                                logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                            else:
                                logging.warning(f"Checkpoint path {checkpoint_path} does not exist. Using the full model instead.")
                        else:
                            # Load the full model
                            model = model_factories[model_type].load(model_path)
                    else:
                        raise ValueError(f"Unknown PyTorch model type: {model_type}")
                else:
                    raise ValueError(f"Unsupported model type for PyTorch: {args.model_type}")
            else:
                # For TensorFlow models, use the legacy approach
                if not TENSORFLOW_AVAILABLE:
                    raise ImportError("TensorFlow is not available. Cannot load TensorFlow model.")

                model = tf.keras.models.load_model(model_path)

            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None

        # 4. Run inference
        logging.info("Step 4/5: Running inference...")

        try:
            if is_pytorch_model:
                # For PyTorch models, use the BaseModel interface
                if args.model_type in ['lstm', 'pytorch_lstm']:
                    # For LSTM models, predict the target values (GSR)
                    predictions = model.predict(X_scaled)

                    # Create a DataFrame with ground truth and predictions
                    results_df = pd.DataFrame({
                        "timestamp": range(len(y)),
                        "ground_truth": y.flatten(),
                        "prediction": predictions
                    })

                    # Calculate regression metrics
                    metrics = model.evaluate(X_scaled, y)

                    logging.info(f"Inference Results for subject {args.subject_id}:")
                    for metric_name, metric_value in metrics.items():
                        logging.info(f"  {metric_name.upper()}: {metric_value:.4f}")

                else:
                    # For autoencoder/VAE models, reconstruct the input
                    reconstructions = model.predict(X_scaled)

                    # Calculate metrics using the evaluate method
                    metrics = model.evaluate(X_scaled)

                    logging.info(f"Reconstruction Results for subject {args.subject_id}:")
                    for metric_name, metric_value in metrics.items():
                        logging.info(f"  {metric_name.upper()}: {metric_value:.4f}")

                    # Create a simplified results DataFrame for autoencoders
                    results_df = pd.DataFrame({
                        "timestamp": range(len(X)),
                        "reconstruction_mse": metrics.get('mse', 0.0),
                        "reconstruction_mae": metrics.get('mae', 0.0)
                    })

            else:
                # For TensorFlow models, use the legacy approach
                if args.model_type == "lstm" or args.model_type == "tf_lstm":
                    # For LSTM models, predict the target values (GSR)
                    predictions = model.predict(X_scaled, verbose=1)

                    # Ensure predictions are the right shape
                    if len(predictions.shape) > 1:
                        predictions = predictions.flatten()

                    # Create a DataFrame with ground truth and predictions
                    results_df = pd.DataFrame({
                        "timestamp": range(len(y)),
                        "ground_truth": y.flatten(),
                        "prediction": predictions
                    })

                    # Calculate regression metrics
                    mae = np.mean(np.abs(results_df["ground_truth"] - results_df["prediction"]))
                    mse = np.mean(np.square(results_df["ground_truth"] - results_df["prediction"]))
                    rmse = np.sqrt(mse)

                    logging.info(f"Inference Results for subject {args.subject_id}:")
                    logging.info(f"  MAE: {mae:.4f}")
                    logging.info(f"  MSE: {mse:.4f}")
                    logging.info(f"  RMSE: {rmse:.4f}")

                else:
                    # For autoencoder/VAE models, reconstruct the input
                    reconstructions = model.predict(X_scaled, verbose=1)

                    # Calculate reconstruction error metrics
                    # Use axis parameter to maintain dimensionality during calculation
                    reconstruction_mse = np.mean(np.square(X_scaled - reconstructions), axis=(1, 2))
                    reconstruction_mae = np.mean(np.abs(X_scaled - reconstructions), axis=(1, 2))

                    # Overall metrics
                    mse = np.mean(reconstruction_mse)
                    mae = np.mean(reconstruction_mae)

                    logging.info(f"Reconstruction Results for subject {args.subject_id}:")
                    logging.info(f"  MSE: {mse:.4f}")
                    logging.info(f"  MAE: {mae:.4f}")

                    # Create a simplified results DataFrame for autoencoders
                    results_df = pd.DataFrame({
                        "timestamp": range(len(X)),
                        "reconstruction_mse": reconstruction_mse,
                        "reconstruction_mae": reconstruction_mae
                    })

                    # Free memory
                    del reconstructions

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return None

        # 5. Save the results
        logging.info("Step 5/5: Saving results...")
        prediction_file = predictions_dir / f"predictions_{args.subject_id}_{args.model_type}.csv"
        results_df.to_csv(prediction_file, index=False)
        logging.info(f"Saved prediction results to {prediction_file}")

        # Return success
        logging.info(f"Inference completed successfully for subject {args.subject_id}")
        return results_df

    except Exception as e:
        logging.error(f"Unexpected error during inference pipeline: {e}")
        return None


if __name__ == "__main__":
    # Logging is already configured at the module level
    main()
