#!/usr/bin/env python3
# src/scripts/analyze_and_train.py

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.processing.data_analysis import DataAnalyzer
from src.ml_pipeline.training.train_model import build_model_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

def prepare_features_for_training(
    features_df: pd.DataFrame,
    target_feature: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features for model training by splitting into train and test sets
    and scaling the features.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        target_feature (str): Name of the target feature column
        test_size (float, optional): Fraction of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    # Ensure target feature exists in the DataFrame
    if target_feature not in features_df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in features DataFrame")
    
    # Separate features and target
    X = features_df.drop(columns=[target_feature])
    y = features_df[target_feature]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logging.info(f"Prepared features for training: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train.values, y_test.values

def reshape_features_for_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape features for sequence models (LSTM, CNN_LSTM, Transformer).
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        model_type (str): Type of model ('lstm', 'cnn_lstm', 'transformer', etc.)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Reshaped X_train and X_test
    """
    # Check if reshaping is needed based on model type
    sequence_models = ['lstm', 'cnn_lstm', 'transformer']
    if model_type.lower() not in sequence_models:
        logging.info(f"No reshaping needed for model type: {model_type}")
        return X_train, X_test
    
    # For sequence models, reshape to (samples, timesteps, features)
    # Here we use a simple approach: each sample is treated as a sequence of length 1
    # This can be adjusted based on the specific requirements of the model
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    logging.info(f"Reshaped features for {model_type} model: X_train shape: {X_train_reshaped.shape}, X_test shape: {X_test_reshaped.shape}")
    
    return X_train_reshaped, X_test_reshaped

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a trained model on the test set.
    
    Args:
        model (Any): Trained model with predict method
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # Log metrics
    logging.info("Model evaluation metrics:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    return metrics

def plot_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_feature: str,
    output_dir: Path
) -> None:
    """
    Plot actual vs predicted values and save the plot.
    
    Args:
        y_test (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted target values
        target_feature (str): Name of the target feature
        output_dir (Path): Directory to save the plot
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    plt.xlabel(f'Actual {target_feature}')
    plt.ylabel(f'Predicted {target_feature}')
    plt.title(f'Actual vs Predicted {target_feature}')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / f"predictions_{target_feature}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved predictions plot to {plot_path}")
    
    # Close the plot to free memory
    plt.close()

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Analyze GSR-RGBT data and train models")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing session recordings")
    parser.add_argument("--output-dir", type=str, default="output/results",
                        help="Directory to save analysis results and trained models")
    parser.add_argument("--gsr-sampling-rate", type=int, default=32,
                        help="Sampling rate of the GSR signal in Hz")
    
    # Analysis parameters
    parser.add_argument("--save-visualizations", action="store_true",
                        help="Save visualizations of the data and features")
    
    # Model parameters
    parser.add_argument("--model-type", type=str, default="lstm",
                        choices=["lstm", "cnn", "cnn_lstm", "transformer", "resnet"],
                        help="Type of model to train")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to model configuration YAML file")
    parser.add_argument("--target-feature", type=str, default="GSR_mean",
                        help="Feature to use as the target for model training")
    
    # Training parameters
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random state for reproducibility")
    
    return parser.parse_args()

def main():
    """
    Main function to analyze GSR-RGBT data and train models.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert string paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find session directories
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Subject_")]
    
    if not session_dirs:
        logging.error(f"No session directories found in {data_dir}")
        return
    
    logging.info(f"Found {len(session_dirs)} session directories")
    
    # Create analyzer
    analyzer = DataAnalyzer(output_dir=output_dir)
    
    # Analyze sessions
    results = analyzer.analyze_multiple_sessions(
        session_paths=session_dirs,
        gsr_sampling_rate=args.gsr_sampling_rate,
        save_visualizations=args.save_visualizations
    )
    
    if not results or 'combined_features' not in results:
        logging.error("Failed to analyze sessions")
        return
    
    # Get combined features
    features_df = results['combined_features']
    
    # Save features to CSV
    features_path = output_dir / "features.csv"
    features_df.to_csv(features_path, index=False)
    logging.info(f"Saved features to {features_path}")
    
    # Prepare features for training
    try:
        X_train, X_test, y_train, y_test = prepare_features_for_training(
            features_df=features_df,
            target_feature=args.target_feature,
            test_size=args.test_size,
            random_state=args.random_state
        )
    except ValueError as e:
        logging.error(f"Error preparing features: {e}")
        return
    
    # Reshape features for sequence models if needed
    X_train_reshaped, X_test_reshaped = reshape_features_for_model(
        X_train, X_test, model_type=args.model_type
    )
    
    # Build model
    try:
        model = build_model_from_config(
            input_shape=X_train_reshaped.shape[1:],
            model_type=args.model_type,
            config_path=args.config_path
        )
    except Exception as e:
        logging.error(f"Error building model: {e}")
        return
    
    # Train model
    logging.info(f"Training {args.model_type} model...")
    history = model.fit(X_train_reshaped, y_train, validation_split=0.2)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_reshaped, y_test)
    
    # Save metrics to JSON
    import json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved metrics to {metrics_path}")
    
    # Plot predictions
    y_pred = model.predict(X_test_reshaped)
    plot_predictions(
        y_test, y_pred,
        target_feature=args.target_feature,
        output_dir=output_dir
    )
    
    # Save model
    model_path = output_dir / f"{args.model_type}_model"
    model.save(str(model_path))
    logging.info(f"Saved model to {model_path}")
    
    logging.info("Analysis and training complete!")

if __name__ == "__main__":
    main()