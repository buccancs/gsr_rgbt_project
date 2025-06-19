"""
Script for comparing multiple machine learning experiments.

This script provides functionality to load, aggregate, and visualize results
from multiple model training runs, allowing for systematic comparison of
different model architectures, hyperparameters, and configurations.
"""

import argparse
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path for absolute imports
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

# Constants
PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")


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


def load_all_cv_results(base_output_dir: Path, experiment_pattern: str = "cross_validation_results_*.csv") -> Optional[pd.DataFrame]:
    """
    Loads all cross-validation result CSVs from a base directory, potentially
    matching a pattern to distinguish different experiment sets or model types.
    
    Args:
        base_output_dir (Path): The base directory to search for CV result files
        experiment_pattern (str): Glob pattern to match CV result files
        
    Returns:
        Optional[pd.DataFrame]: A DataFrame containing all CV results, or None if no files found
    """
    all_dfs = []
    # Search for results files recursively
    for cv_file in base_output_dir.glob(f"**/{experiment_pattern}"):
        try:
            # Extract model_type or experiment_id from filename or path
            match = re.search(r"cross_validation_results_(.+)\.csv", cv_file.name)
            if match:
                model_run_id = match.group(1)
            else:
                # Fallback or derive from parent directory if structured by experiment
                model_run_id = cv_file.parent.name + "_" + cv_file.stem.replace("cross_validation_results_", "")

            df = pd.read_csv(cv_file)
            df['model_run_id'] = model_run_id
            all_dfs.append(df)
            logging.info(f"Loaded CV results from {cv_file} for run '{model_run_id}'")
        except Exception as e:
            logging.error(f"Error loading CV results from {cv_file}: {e}")
    
    if not all_dfs:
        logging.warning(f"No cross-validation result files found in {base_output_dir} matching {experiment_pattern}")
        return None
    
    return pd.concat(all_dfs, ignore_index=True)


def load_all_history_files(base_output_dir: Path, experiment_pattern: str = "*_history.json") -> Optional[Dict[str, pd.DataFrame]]:
    """
    Loads all training history JSON files from a base directory.
    
    Args:
        base_output_dir (Path): The base directory to search for history files
        experiment_pattern (str): Glob pattern to match history files
        
    Returns:
        Optional[Dict[str, pd.DataFrame]]: A dictionary mapping model IDs to history DataFrames,
                                          or None if no files found
    """
    import json
    
    histories = {}
    # Search for history files recursively
    for history_file in base_output_dir.glob(f"**/{experiment_pattern}"):
        try:
            # Extract model_type or experiment_id from filename
            model_id = history_file.stem.replace("_history", "")
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Convert to DataFrame for easier plotting
            history_df = pd.DataFrame(history_data)
            histories[model_id] = history_df
            logging.info(f"Loaded history from {history_file} for model '{model_id}'")
        except Exception as e:
            logging.error(f"Error loading history from {history_file}: {e}")
    
    if not histories:
        logging.warning(f"No history files found in {base_output_dir} matching {experiment_pattern}")
        return None
    
    return histories


def plot_metric_comparison_bar(df_all_results: pd.DataFrame, metric: str, output_dir: Path):
    """
    Creates a bar plot comparing the average value of a metric across different model runs.
    
    Args:
        df_all_results (pd.DataFrame): DataFrame containing all CV results
        metric (str): The metric to plot
        output_dir (Path): Directory to save the plot
    """
    if df_all_results.empty or metric not in df_all_results.columns:
        logging.warning(f"Cannot plot comparison for metric '{metric}'. Data missing or metric not found.")
        return

    plt.figure(figsize=(12, 7))
    # Group by model_run_id and calculate mean and std for the metric
    summary_stats = df_all_results.groupby('model_run_id')[metric].agg(['mean', 'std']).reset_index()
    
    sns.barplot(x='model_run_id', y='mean', data=summary_stats, palette='viridis', yerr=summary_stats['std'])
    plt.title(f'Comparison of Average {metric.upper()} Across Model Runs (with StdDev)')
    plt.ylabel(f'Average {metric.upper()}')
    plt.xlabel('Model Run / Configuration')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"comparison_avg_bar_{metric}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved average {metric} bar plot comparison to {save_path}")


def plot_metric_comparison_box(df_all_results: pd.DataFrame, metric: str, output_dir: Path):
    """
    Creates a box plot showing the distribution of a metric across different model runs.
    
    Args:
        df_all_results (pd.DataFrame): DataFrame containing all CV results
        metric (str): The metric to plot
        output_dir (Path): Directory to save the plot
    """
    if df_all_results.empty or metric not in df_all_results.columns:
        logging.warning(f"Cannot plot box comparison for metric '{metric}'. Data missing or metric not found.")
        return
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='model_run_id', y=metric, data=df_all_results, palette='viridis')
    plt.title(f'Distribution of {metric.upper()} Across Model Runs')
    plt.ylabel(metric.upper())
    plt.xlabel('Model Run / Configuration')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"comparison_dist_box_{metric}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved {metric} distribution box plot comparison to {save_path}")


def plot_history_comparison(histories: Dict[str, pd.DataFrame], metric: str, output_dir: Path):
    """
    Creates a line plot comparing the training history of a specific metric across different model runs.
    
    Args:
        histories (Dict[str, pd.DataFrame]): Dictionary mapping model IDs to history DataFrames
        metric (str): The metric to plot (e.g., 'train_loss', 'val_loss')
        output_dir (Path): Directory to save the plot
    """
    if not histories:
        logging.warning(f"Cannot plot history comparison for metric '{metric}'. No history data provided.")
        return
    
    plt.figure(figsize=(12, 7))
    
    for model_id, history_df in histories.items():
        if metric in history_df.columns:
            plt.plot(history_df.index, history_df[metric], label=model_id)
        else:
            logging.warning(f"Metric '{metric}' not found in history for model '{model_id}'")
    
    plt.title(f'Training History: {metric.upper()} Comparison')
    plt.ylabel(metric.upper())
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"history_comparison_{metric}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved {metric} history comparison plot to {save_path}")


def plot_metric_scatter(df_all_results: pd.DataFrame, x_metric: str, y_metric: str, output_dir: Path):
    """
    Creates a scatter plot comparing two metrics across different model runs.
    
    Args:
        df_all_results (pd.DataFrame): DataFrame containing all CV results
        x_metric (str): The metric to plot on the x-axis
        y_metric (str): The metric to plot on the y-axis
        output_dir (Path): Directory to save the plot
    """
    if df_all_results.empty or x_metric not in df_all_results.columns or y_metric not in df_all_results.columns:
        logging.warning(f"Cannot plot scatter for metrics '{x_metric}' vs '{y_metric}'. Data missing or metrics not found.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Calculate mean values for each model run
    summary_stats = df_all_results.groupby('model_run_id')[[x_metric, y_metric]].mean().reset_index()
    
    # Create scatter plot
    sns.scatterplot(x=x_metric, y=y_metric, data=summary_stats, s=100, hue='model_run_id')
    
    # Add labels for each point
    for i, row in summary_stats.iterrows():
        plt.annotate(row['model_run_id'], 
                    (row[x_metric], row[y_metric]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.title(f'{y_metric.upper()} vs {x_metric.upper()} Across Model Runs')
    plt.xlabel(x_metric.upper())
    plt.ylabel(y_metric.upper())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"scatter_{y_metric}_vs_{x_metric}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved scatter plot of {y_metric} vs {x_metric} to {save_path}")


def generate_comparison_report(df_all_results: pd.DataFrame, output_dir: Path):
    """
    Generates a comprehensive comparison report in CSV format.
    
    Args:
        df_all_results (pd.DataFrame): DataFrame containing all CV results
        output_dir (Path): Directory to save the report
    """
    if df_all_results.empty:
        logging.warning("Cannot generate comparison report. No data provided.")
        return
    
    # Create a summary of all metrics by model_run_id
    metrics = [col for col in df_all_results.columns 
               if col not in ['fold', 'subject', 'model_run_id'] 
               and pd.api.types.is_numeric_dtype(df_all_results[col])]
    
    # Calculate mean and std for each metric by model_run_id
    summary = df_all_results.groupby('model_run_id')[metrics].agg(['mean', 'std'])
    
    # Flatten the MultiIndex columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Reset index to make model_run_id a column
    summary = summary.reset_index()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_comparison_report.csv"
    summary.to_csv(report_path, index=False)
    logging.info(f"Generated comparison report saved to {report_path}")
    
    return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare multiple ML experiments')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory containing experiment outputs (defaults to config.OUTPUT_DIR)'
    )
    
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='plots',
        help='Directory to save comparison plots'
    )
    
    parser.add_argument(
        '--reports-dir',
        type=str,
        default='reports',
        help='Directory to save comparison reports'
    )
    
    parser.add_argument(
        '--cv-pattern',
        type=str,
        default='cross_validation_results_*.csv',
        help='Glob pattern to match cross-validation result files'
    )
    
    parser.add_argument(
        '--history-pattern',
        type=str,
        default='*_history.json',
        help='Glob pattern to match training history files'
    )
    
    parser.add_argument(
        '--plot-metrics',
        nargs='+',
        default=['mae', 'mse', 'r2_score'],
        help='Metrics to plot (default: mae mse r2_score)'
    )
    
    parser.add_argument(
        '--plot-history',
        action='store_true',
        help='Plot training history comparisons'
    )
    
    parser.add_argument(
        '--plot-scatter',
        action='store_true',
        help='Generate scatter plots comparing pairs of metrics'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the experiment comparison."""
    args = parse_args()
    
    # Set up directories
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    plots_dir = Path(args.plots_dir)
    reports_dir = Path(args.reports_dir)
    
    logging.info(f"Looking for experiment results in: {output_dir}")
    
    # Load all cross-validation results
    df_all_results = load_all_cv_results(output_dir, args.cv_pattern)
    
    if df_all_results is not None:
        # Generate comparison plots for each metric
        for metric in args.plot_metrics:
            if metric in df_all_results.columns:
                plot_metric_comparison_bar(df_all_results, metric, plots_dir)
                plot_metric_comparison_box(df_all_results, metric, plots_dir)
            else:
                logging.warning(f"Metric '{metric}' not found in results data")
        
        # Generate scatter plots if requested
        if args.plot_scatter:
            # Create scatter plots for pairs of metrics
            metrics = [m for m in args.plot_metrics if m in df_all_results.columns]
            for i, x_metric in enumerate(metrics):
                for y_metric in metrics[i+1:]:
                    plot_metric_scatter(df_all_results, x_metric, y_metric, plots_dir)
        
        # Generate comparison report
        summary = generate_comparison_report(df_all_results, reports_dir)
        print("\nModel Comparison Summary:")
        print(summary)
    
    # Load and plot training histories if requested
    if args.plot_history:
        histories = load_all_history_files(output_dir, args.history_pattern)
        if histories:
            # Determine common metrics across all histories
            common_metrics = set()
            for model_id, history_df in histories.items():
                if common_metrics:
                    common_metrics &= set(history_df.columns)
                else:
                    common_metrics = set(history_df.columns)
            
            # Plot each common metric
            for metric in common_metrics:
                plot_history_comparison(histories, metric, plots_dir)
    
    logging.info("Experiment comparison completed")


if __name__ == "__main__":
    main()