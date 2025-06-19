# src/ml_pipeline/evaluation/evaluate_model.py

import logging

# --- Import project modules ---
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src import config
from src.ml_pipeline.evaluation.visualization import (
    plot_prediction_vs_ground_truth,
    plot_bland_altman,
)

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculates key performance metrics from a results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame with 'ground_truth' and 'prediction' columns.

    Returns:
        A dictionary containing MAE, RMSE, and Pearson correlation.
    """
    if results_df.empty:
        return {}

    truth = results_df["ground_truth"]
    pred = results_df["prediction"]

    mae = (truth - pred).abs().mean()
    rmse = ((truth - pred) ** 2).mean() ** 0.5
    # Pearson correlation returns (correlation, p-value)
    corr, _ = pearsonr(truth, pred)

    return {"mae": mae, "rmse": rmse, "pearson_r": corr}


def main():
    """
    Main function to load predictions and generate evaluation plots and metrics.
    """
    # --- Configuration for Evaluation ---
    # This specifies which subject's prediction file to evaluate.
    # In a larger study, you would loop through all subjects.
    TEST_SUBJECT_ID = "Subject01"

    logging.info(f"--- Starting Evaluation for Subject: {TEST_SUBJECT_ID} ---")

    # Define paths
    predictions_dir = config.OUTPUT_DIR / "predictions"
    plots_dir = config.OUTPUT_DIR / "evaluation_plots"
    prediction_file = predictions_dir / f"predictions_{TEST_SUBJECT_ID}.csv"

    if not prediction_file.exists():
        logging.error(f"Prediction file not found: {prediction_file}")
        logging.error(
            "Please run 'src/ml_pipeline/evaluation/inference.py' first to generate predictions."
        )
        return

    # 1. Load the prediction data
    try:
        results_df = pd.read_csv(prediction_file)
        logging.info(f"Loaded {len(results_df)} predictions from {prediction_file}")
    except Exception as e:
        logging.error(f"Failed to load prediction data: {e}")
        return

    # 2. Calculate performance metrics
    metrics = calculate_metrics(results_df)
    if metrics:
        logging.info("Calculated Performance Metrics:")
        logging.info(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        logging.info(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.4f}")
        logging.info(f"  Pearson Correlation (r): {metrics['pearson_r']:.4f}")

    # 3. Generate and save visualization plots
    logging.info("Generating visualization plots...")
    plot_prediction_vs_ground_truth(results_df, TEST_SUBJECT_ID, plots_dir)
    plot_bland_altman(results_df, TEST_SUBJECT_ID, plots_dir)

    logging.info(f"Evaluation complete. Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()