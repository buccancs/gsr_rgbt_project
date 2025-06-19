# src/ml_pipeline/evaluation/visualization.py

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def plot_prediction_vs_ground_truth(
    results_df: pd.DataFrame, subject_id: str, output_dir: Path
):
    """
    Creates and saves a time-series plot comparing predicted vs. ground truth GSR.

    Args:
        results_df (pd.DataFrame): DataFrame with 'ground_truth' and 'prediction' columns.
        subject_id (str): The ID of the subject being plotted.
        output_dir (Path): The directory where the plot image will be saved.
    """
    if (
        not isinstance(results_df, pd.DataFrame)
        or "ground_truth" not in results_df
        or "prediction" not in results_df
    ):
        logging.error("Invalid DataFrame passed to plotting function.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        results_df.index,
        results_df["ground_truth"],
        label="Ground Truth (Contact Sensor)",
        color="royalblue",
        linewidth=2,
    )
    ax.plot(
        results_df.index,
        results_df["prediction"],
        label="Predicted (Contactless)",
        color="darkorange",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title(
        f"GSR Prediction vs. Ground Truth for {subject_id}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("GSR Signal (Phasic Component)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{subject_id}_prediction_vs_truth.png"

    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved prediction plot to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
    finally:
        plt.close(fig)


def plot_bland_altman(results_df: pd.DataFrame, subject_id: str, output_dir: Path):
    """
    Creates and saves a Bland-Altman plot to assess agreement between two measurements.

    Args:
        results_df (pd.DataFrame): DataFrame with 'ground_truth' and 'prediction' columns.
        subject_id (str): The ID of the subject being plotted.
        output_dir (Path): The directory where the plot image will be saved.
    """
    if (
        not isinstance(results_df, pd.DataFrame)
        or "ground_truth" not in results_df
        or "prediction" not in results_df
    ):
        logging.error("Invalid DataFrame passed to Bland-Altman plotting function.")
        return

    ground_truth = results_df["ground_truth"]
    prediction = results_df["prediction"]

    mean = np.mean([ground_truth, prediction], axis=0)
    diff = ground_truth - prediction
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    limit_of_agreement_upper = mean_diff + 1.96 * std_diff
    limit_of_agreement_lower = mean_diff - 1.96 * std_diff

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(mean, diff, alpha=0.5, s=20, label="Difference Points")
    ax.axhline(
        mean_diff, color="red", linestyle="--", label=f"Mean Diff ({mean_diff:.3f})"
    )
    ax.axhline(
        limit_of_agreement_upper,
        color="gray",
        linestyle=":",
        label=f"Upper LoA (+1.96 SD)",
    )
    ax.axhline(
        limit_of_agreement_lower,
        color="gray",
        linestyle=":",
        label=f"Lower LoA (-1.96 SD)",
    )

    ax.set_title(f"Bland-Altman Plot for {subject_id}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Average of Ground Truth and Prediction", fontsize=12)
    ax.set_ylabel("Difference (Ground Truth - Prediction)", fontsize=12)
    ax.legend()
    ax.grid(True)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{subject_id}_bland_altman_plot.png"

    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved Bland-Altman plot to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save Bland-Altman plot: {e}")
    finally:
        plt.close(fig)