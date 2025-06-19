# src/scripts/visualize_results.py

"""
Script for visualization and reporting of ML model results.

This script generates plots and repo_forensic to help analyze the performance of the models.
It includes functions for:
1. Plotting training history (loss curves)
2. Visualizing predictions vs. ground truth
3. Generating performance metrics repo_forensic
4. Creating comparison visualizations between different models
"""

import argparse
import json
import logging
import re
# --- Add project root to path for absolute imports ---
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)

# --- Constants ---
PLOTS_DIR = config.OUTPUT_DIR / "evaluation_plots"
REPORTS_DIR = config.OUTPUT_DIR / "repo_forensic"
PREDICTIONS_DIR = config.OUTPUT_DIR / "predictions"
MODELS_DIR = config.OUTPUT_DIR / "models"


def setup_directories():
    """Create necessary directories for outputs."""
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    logging.info(f"Created output directories: {PLOTS_DIR}, {REPORTS_DIR}")


def plot_training_history(history_file, output_path=None, annotate=True):
    """
    Plot training and validation loss curves from a training history file.

    Args:
        history_file (Path): Path to the JSON file containing training history
        output_path (Path, optional): Path to save the plot. If None, will be auto-generated.
        annotate (bool): Whether to annotate the plot with additional information

    Returns:
        Path: Path to the saved plot
    """
    try:
        # Load history data
        with open(history_file, 'r') as f:
            history = json.load(f)

        # Extract model name from file path
        model_name = history_file.stem.split('_history')[0]

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot training & validation loss
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')

        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')

            # Find the best validation loss for annotation
            if annotate:
                best_epoch = np.argmin(history['val_loss']) + 1  # +1 because epochs start at 1
                best_val_loss = history['val_loss'][best_epoch - 1]
                plt.annotate(f'Best val_loss: {best_val_loss:.4f}',
                             xy=(best_epoch, best_val_loss),
                             xytext=(best_epoch + 5, best_val_loss * 1.1),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                             fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

        plt.title(f'Training and Validation Loss - {model_name.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add model configuration details if annotation is enabled
        if annotate:
            # Try to get model configuration details
            try:
                from src.ml_models.model_config import ModelConfig
                model_config = ModelConfig(model_name.split('_')[0])  # Extract base model name

                # Create annotation text
                annotation_text = f"Model: {model_name.upper()}\n"
                annotation_text += f"Framework: {model_config.get_framework()}\n"

                # Add model-specific parameters based on model type
                model_type = model_name.split('_')[0].lower()
                model_params = model_config.get_model_params()

                if "lstm" in model_type:
                    annotation_text += f"Hidden Size: {model_params.get('hidden_size', 'N/A')}\n"
                    annotation_text += f"Layers: {model_params.get('num_layers', 'N/A')}\n"
                    annotation_text += f"Bidirectional: {model_params.get('bidirectional', False)}\n"
                elif "transformer" in model_type:
                    annotation_text += f"d_model: {model_params.get('d_model', 'N/A')}\n"
                    annotation_text += f"nhead: {model_params.get('nhead', 'N/A')}\n"
                    annotation_text += f"num_layers: {model_params.get('num_layers', 'N/A')}\n"
                elif "resnet" in model_type:
                    annotation_text += f"Blocks: {model_params.get('blocks_per_layer', 'N/A')}\n"
                elif "cnn" in model_type:
                    annotation_text += f"Conv Channels: {model_params.get('conv_channels', 'N/A')}\n"
                    annotation_text += f"Kernel Sizes: {model_params.get('kernel_sizes', 'N/A')}\n"
                elif "vae" in model_type or "autoencoder" in model_type:
                    annotation_text += f"Latent Dim: {model_params.get('latent_dim', 'N/A')}\n"

                # Add timestamp
                from datetime import datetime
                annotation_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

                # Add annotation to plot
                plt.figtext(0.02, 0.02, annotation_text, fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
            except Exception as e:
                logging.warning(f"Could not add model configuration annotation: {e}")

        # Save the plot
        if output_path is None:
            output_path = PLOTS_DIR / f"{model_name}_training_history.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logging.info(f"Training history plot saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error plotting training history: {e}")
        return None


def plot_predictions_vs_ground_truth(predictions_file, output_path=None, annotate=True):
    """
    Plot predictions vs. ground truth from a predictions file.

    Args:
        predictions_file (Path): Path to the CSV file containing predictions
        output_path (Path, optional): Path to save the plot. If None, will be auto-generated.
        annotate (bool): Whether to annotate the plot with additional information

    Returns:
        Path: Path to the saved plot
    """
    try:
        # Load predictions data
        predictions_df = pd.read_csv(predictions_file)

        # Extract model and subject info from filename
        filename = predictions_file.stem
        match = re.search(r'predictions_(.+)_(.+)', filename)
        if match:
            subject_id, model_type = match.groups()
        else:
            subject_id = "unknown"
            model_type = "unknown"

        # Check if this is a regression prediction (has ground_truth and prediction columns)
        if 'ground_truth' in predictions_df.columns and 'prediction' in predictions_df.columns:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Time series plot
            ax1.plot(predictions_df['ground_truth'], 'b-', label='Ground Truth')
            ax1.plot(predictions_df['prediction'], 'r-', label='Prediction')
            ax1.set_title(f'GSR Predictions vs. Ground Truth - {subject_id} ({model_type})')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('GSR Value')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Scatter plot with perfect prediction line
            ax2.scatter(predictions_df['ground_truth'], predictions_df['prediction'], alpha=0.5)

            # Add perfect prediction line
            min_val = min(predictions_df['ground_truth'].min(), predictions_df['prediction'].min())
            max_val = max(predictions_df['ground_truth'].max(), predictions_df['prediction'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

            ax2.set_title('Prediction Scatter Plot')
            ax2.set_xlabel('Ground Truth')
            ax2.set_ylabel('Prediction')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Calculate metrics
            mse = mean_squared_error(predictions_df['ground_truth'], predictions_df['prediction'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(predictions_df['ground_truth'], predictions_df['prediction'])
            r2 = r2_score(predictions_df['ground_truth'], predictions_df['prediction'])

            # Add metrics as text
            metrics_text = f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
            ax2.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                         va='top')

        # For autoencoder/VAE models (reconstruction error)
        elif 'reconstruction_mse' in predictions_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot reconstruction error over time
            ax.plot(predictions_df['reconstruction_mse'], 'r-', label='MSE')
            if 'reconstruction_mae' in predictions_df.columns:
                ax.plot(predictions_df['reconstruction_mae'], 'b-', label='MAE')

            ax.set_title(f'Reconstruction Error - {subject_id} ({model_type})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Error')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            # Calculate average metrics
            avg_mse = predictions_df['reconstruction_mse'].mean()
            if 'reconstruction_mae' in predictions_df.columns:
                avg_mae = predictions_df['reconstruction_mae'].mean()
                metrics_text = f'Avg MSE: {avg_mse:.4f}\nAvg MAE: {avg_mae:.4f}'
            else:
                metrics_text = f'Avg MSE: {avg_mse:.4f}'

            # Add metrics as text
            ax.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                        va='top')

        else:
            logging.warning(f"Unrecognized prediction format in {predictions_file}")
            return None

        # Save the plot
        if output_path is None:
            output_path = PLOTS_DIR / f"{subject_id}_{model_type}_predictions.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logging.info(f"Predictions plot saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error plotting predictions: {e}")
        return None


def generate_model_comparison_report(results_file, output_path=None, annotate=True):
    """
    Generate a comparison report of different models from cross-validation results.

    Args:
        results_file (Path): Path to the CSV file containing cross-validation results
        output_path (Path, optional): Path to save the report. If None, will be auto-generated.
        annotate (bool): Whether to annotate the plot with additional information

    Returns:
        Path: Path to the saved report
    """
    try:
        # Load results data
        results_df = pd.read_csv(results_file)

        # Extract model type from filename
        model_type = results_file.stem.split('_results_')[1]

        # Create figure
        plt.figure(figsize=(12, 8))

        # Get metric columns (excluding fold and subject)
        metric_columns = [col for col in results_df.columns if col not in ['fold', 'subject']]

        # Create subplots for each metric
        fig, axes = plt.subplots(len(metric_columns), 1, figsize=(10, 4 * len(metric_columns)))
        if len(metric_columns) == 1:
            axes = [axes]  # Make it iterable if there's only one metric

        for i, metric in enumerate(metric_columns):
            # Box plot for this metric across subjects
            sns.boxplot(x='subject', y=metric, data=results_df, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} by Subject - {model_type}')
            axes[i].set_xlabel('Subject')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, linestyle='--', alpha=0.7)

            # Rotate x-axis labels if there are many subjects
            if len(results_df['subject'].unique()) > 5:
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

        # Save the plot
        if output_path is None:
            output_path = PLOTS_DIR / f"{model_type}_comparison.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        # Generate a summary table
        summary_df = results_df.groupby('subject')[metric_columns].mean().reset_index()
        summary_df.loc['Average'] = summary_df[metric_columns].mean()
        summary_df.loc['Average', 'subject'] = 'Average'

        # Save the summary table
        summary_path = REPORTS_DIR / f"{model_type}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        logging.info(f"Model comparison plot saved to {output_path}")
        logging.info(f"Summary table saved to {summary_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error generating model comparison report: {e}")
        return None


def save_model_milestone(model_dir, model_type, milestone_name):
    """
    Save a milestone of a model by copying its files to a milestone directory.

    Args:
        model_dir (Path): Path to the directory containing model files
        model_type (str): Type of the model (e.g., 'lstm', 'autoencoder')
        milestone_name (str): Name of the milestone (e.g., 'baseline', 'improved')

    Returns:
        Path: Path to the milestone directory
    """
    try:
        import shutil

        # Create milestone directory
        milestone_dir = config.OUTPUT_DIR / "milestones" / f"{model_type}_{milestone_name}"
        milestone_dir.mkdir(exist_ok=True, parents=True)

        # Find all model files of this type
        model_files = list(model_dir.glob(f"{model_type}*"))

        if not model_files:
            logging.warning(f"No model files found for {model_type} in {model_dir}")
            return None

        # Copy files to milestone directory
        for file_path in model_files:
            dest_path = milestone_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            logging.info(f"Copied {file_path} to {dest_path}")

        # Create a milestone info file
        info_path = milestone_dir / "milestone_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Milestone Name: {milestone_name}\n")
            f.write(f"Created: {pd.Timestamp.now()}\n")
            f.write(f"Files: {[f.name for f in model_files]}\n")

        logging.info(f"Saved milestone {milestone_name} for {model_type} to {milestone_dir}")
        return milestone_dir

    except Exception as e:
        logging.error(f"Error saving model milestone: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize and report ML model results")

    parser.add_argument(
        "--plot-history",
        action="store_true",
        help="Plot training history for all models"
    )

    parser.add_argument(
        "--plot-predictions",
        action="store_true",
        help="Plot predictions vs. ground truth for all prediction files"
    )

    parser.add_argument(
        "--model-comparison",
        action="store_true",
        help="Generate model comparison repo_forensic"
    )

    parser.add_argument(
        "--save-milestone",
        type=str,
        help="Save a milestone of the specified model type"
    )

    parser.add_argument(
        "--milestone-name",
        type=str,
        default="milestone",
        help="Name for the milestone (used with --save-milestone)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all visualization and reporting tasks"
    )

    parser.add_argument(
        "--annotate-graphs",
        action="store_true",
        help="Annotate graphs with additional information (model config, metrics, etc.)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for visualizations"
    )

    return parser.parse_args()


def main():
    """Main function to run visualization and reporting tasks."""
    args = parse_arguments()

    # Create output directories
    setup_directories()

    # Set custom output directory if provided
    global PLOTS_DIR, REPORTS_DIR
    if args.output_dir:
        custom_output_dir = Path(args.output_dir)
        custom_output_dir.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR = custom_output_dir / "plots"
        REPORTS_DIR = custom_output_dir / "repo_forensic"
        PLOTS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        logging.info(f"Using custom output directory: {custom_output_dir}")

    # Track what was processed
    processed_items = {
        "history_plots": 0,
        "prediction_plots": 0,
        "comparison_reports": 0,
        "milestones": 0
    }

    # Plot training history
    if args.plot_history or args.all:
        logging.info("Plotting training history...")
        history_files = list(MODELS_DIR.glob("*_history.json"))

        if not history_files:
            logging.warning("No training history files found.")

        for history_file in history_files:
            if plot_training_history(history_file, annotate=args.annotate_graphs):
                processed_items["history_plots"] += 1

    # Plot predictions
    if args.plot_predictions or args.all:
        logging.info("Plotting predictions...")
        prediction_files = list(PREDICTIONS_DIR.glob("predictions_*.csv"))

        if not prediction_files:
            logging.warning("No prediction files found.")

        for prediction_file in prediction_files:
            if plot_predictions_vs_ground_truth(prediction_file, annotate=args.annotate_graphs):
                processed_items["prediction_plots"] += 1

    # Generate model comparison repo_forensic
    if args.model_comparison or args.all:
        logging.info("Generating model comparison repo_forensic...")
        results_files = list(config.OUTPUT_DIR.glob("cross_validation_results_*.csv"))

        if not results_files:
            logging.warning("No cross-validation results files found.")

        for results_file in results_files:
            if generate_model_comparison_report(results_file, annotate=args.annotate_graphs):
                processed_items["comparison_reports"] += 1

    # Save model milestone
    if args.save_milestone:
        model_type = args.save_milestone
        milestone_name = args.milestone_name

        logging.info(f"Saving milestone '{milestone_name}' for model type '{model_type}'...")
        if save_model_milestone(MODELS_DIR, model_type, milestone_name):
            processed_items["milestones"] += 1

    # Print summary
    logging.info("=== Visualization and Reporting Summary ===")
    logging.info(f"Training history plots: {processed_items['history_plots']}")
    logging.info(f"Prediction plots: {processed_items['prediction_plots']}")
    logging.info(f"Model comparison repo_forensic: {processed_items['comparison_reports']}")
    logging.info(f"Model milestones: {processed_items['milestones']}")
    logging.info("==========================================")


if __name__ == "__main__":
    main()
