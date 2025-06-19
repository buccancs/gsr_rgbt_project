# src/scripts/run_ml_pipeline.py

"""
Script to run the full machine learning pipeline.

This script orchestrates the entire ML pipeline:
1. Generates mock data for training
2. Builds Cython extensions for performance optimization
3. Trains models with cross-validation
4. Runs inference on test subjects
5. Visualizes and reports results
6. Saves model milestones

It serves as a one-stop solution for running the complete pipeline.
"""

import logging
import argparse
import subprocess
import time
from pathlib import Path
import sys

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_pipeline.log"),
        logging.StreamHandler()
    ]
)

# --- Constants ---
MODELS_TO_TRAIN = ["lstm", "autoencoder", "vae"]
TEST_SUBJECTS = ["MockSubject01", "MockSubject02"]  # Subjects to use for testing


def run_command(command, description):
    """
    Run a shell command and log the output.
    
    Args:
        command (str): Command to run
        description (str): Description of the command for logging
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    logging.info(f"Running: {description}")
    logging.info(f"Command: {command}")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output to log
        for line in process.stdout:
            logging.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        # Check if command succeeded
        if process.returncode == 0:
            logging.info(f"Successfully completed: {description}")
            return True
        else:
            # Log error output
            error_output = process.stderr.read()
            logging.error(f"Command failed with return code {process.returncode}")
            logging.error(f"Error output: {error_output}")
            return False
            
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False


def generate_mock_data(num_subjects=10, sessions_per_subject=2):
    """
    Generate mock data for training.
    
    Args:
        num_subjects (int): Number of subjects to generate
        sessions_per_subject (int): Number of sessions per subject
        
    Returns:
        bool: True if data generation succeeded, False otherwise
    """
    logging.info(f"Generating mock data for {num_subjects} subjects, {sessions_per_subject} sessions each")
    
    # Run the generate_training_data.py script
    command = f"python {project_root / 'src' / 'scripts' / 'generate_training_data.py'}"
    return run_command(command, "Mock data generation")


def build_cython_extensions():
    """
    Build Cython extensions for performance optimization.
    
    Returns:
        bool: True if build succeeded, False otherwise
    """
    logging.info("Building Cython extensions")
    
    # Run the setup.py build_ext command
    command = f"python {project_root / 'setup.py'} build_ext --inplace"
    return run_command(command, "Cython build")


def train_model(model_type):
    """
    Train a model with cross-validation.
    
    Args:
        model_type (str): Type of model to train (e.g., 'lstm', 'autoencoder', 'vae')
        
    Returns:
        bool: True if training succeeded, False otherwise
    """
    logging.info(f"Training {model_type} model")
    
    # Run the train_model.py script
    command = f"python {project_root / 'src' / 'scripts' / 'train_model.py'} --model-type {model_type}"
    return run_command(command, f"{model_type} model training")


def run_inference(model_type, subject_id, model_path=None, scaler_path=None):
    """
    Run inference on a test subject.
    
    Args:
        model_type (str): Type of model to use for inference
        subject_id (str): ID of the subject to run inference on
        model_path (str, optional): Path to the model file. If None, will be auto-detected.
        scaler_path (str, optional): Path to the scaler file. If None, will be auto-detected.
        
    Returns:
        bool: True if inference succeeded, False otherwise
    """
    logging.info(f"Running inference with {model_type} model on subject {subject_id}")
    
    # If model_path and scaler_path are not provided, try to find them
    if model_path is None or scaler_path is None:
        models_dir = config.OUTPUT_DIR / "models"
        
        # Find the model file
        if model_path is None:
            model_files = list(models_dir.glob(f"{model_type}*fold_1_subject_{subject_id}.*"))
            if model_files:
                model_path = model_files[0]
            else:
                logging.error(f"Could not find model file for {model_type} and subject {subject_id}")
                return False
        
        # Find the scaler file
        if scaler_path is None:
            scaler_files = list(models_dir.glob(f"scaler_{model_type}*fold_1_subject_{subject_id}.*"))
            if scaler_files:
                scaler_path = scaler_files[0]
            else:
                logging.error(f"Could not find scaler file for {model_type} and subject {subject_id}")
                return False
    
    # Run the inference.py script
    command = (f"python {project_root / 'src' / 'scripts' / 'inference.py'} "
               f"--model-type {model_type} "
               f"--model-path {model_path} "
               f"--scaler-path {scaler_path} "
               f"--subject-id {subject_id}")
    
    return run_command(command, f"Inference with {model_type} on {subject_id}")


def visualize_results(plot_history=True, plot_predictions=True, model_comparison=True):
    """
    Visualize and report results.
    
    Args:
        plot_history (bool): Whether to plot training history
        plot_predictions (bool): Whether to plot predictions
        model_comparison (bool): Whether to generate model comparison reports
        
    Returns:
        bool: True if visualization succeeded, False otherwise
    """
    logging.info("Visualizing and reporting results")
    
    # Build command with appropriate flags
    command = f"python {project_root / 'src' / 'scripts' / 'visualize_results.py'}"
    
    if plot_history:
        command += " --plot-history"
    
    if plot_predictions:
        command += " --plot-predictions"
    
    if model_comparison:
        command += " --model-comparison"
    
    return run_command(command, "Results visualization")


def save_model_milestones():
    """
    Save milestones for all models.
    
    Returns:
        bool: True if saving milestones succeeded, False otherwise
    """
    logging.info("Saving model milestones")
    
    success = True
    for model_type in MODELS_TO_TRAIN:
        # Run the visualize_results.py script with --save-milestone flag
        command = (f"python {project_root / 'src' / 'scripts' / 'visualize_results.py'} "
                  f"--save-milestone {model_type} "
                  f"--milestone-name final")
        
        if not run_command(command, f"Saving milestone for {model_type}"):
            success = False
    
    return success


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the full ML pipeline")
    
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip the mock data generation step"
    )
    
    parser.add_argument(
        "--skip-cython-build",
        action="store_true",
        help="Skip building Cython extensions"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS_TO_TRAIN),
        help="Comma-separated list of models to train (default: lstm,autoencoder,vae)"
    )
    
    parser.add_argument(
        "--test-subjects",
        type=str,
        default=",".join(TEST_SUBJECTS),
        help="Comma-separated list of subjects to use for testing"
    )
    
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip the visualization and reporting step"
    )
    
    parser.add_argument(
        "--skip-milestones",
        action="store_true",
        help="Skip saving model milestones"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the full ML pipeline."""
    start_time = time.time()
    logging.info("=== Starting Full ML Pipeline ===")
    
    args = parse_arguments()
    
    # Parse models and test subjects
    models_to_train = args.models.split(",")
    test_subjects = args.test_subjects.split(",")
    
    # Track pipeline steps
    pipeline_steps = {
        "data_generation": not args.skip_data_generation,
        "cython_build": not args.skip_cython_build,
        "model_training": {},
        "inference": {},
        "visualization": not args.skip_visualization,
        "milestones": not args.skip_milestones
    }
    
    # 1. Generate mock data
    if pipeline_steps["data_generation"]:
        pipeline_steps["data_generation"] = generate_mock_data()
    else:
        logging.info("Skipping mock data generation")
    
    # 2. Build Cython extensions
    if pipeline_steps["cython_build"]:
        pipeline_steps["cython_build"] = build_cython_extensions()
    else:
        logging.info("Skipping Cython build")
    
    # 3. Train models
    for model_type in models_to_train:
        logging.info(f"=== Processing {model_type} model ===")
        
        # Train the model
        pipeline_steps["model_training"][model_type] = train_model(model_type)
        
        # Run inference for each test subject
        pipeline_steps["inference"][model_type] = {}
        for subject_id in test_subjects:
            pipeline_steps["inference"][model_type][subject_id] = run_inference(model_type, subject_id)
    
    # 4. Visualize results
    if pipeline_steps["visualization"]:
        pipeline_steps["visualization"] = visualize_results()
    else:
        logging.info("Skipping visualization")
    
    # 5. Save model milestones
    if pipeline_steps["milestones"]:
        pipeline_steps["milestones"] = save_model_milestones()
    else:
        logging.info("Skipping model milestones")
    
    # Calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    hours, remainder = divmod(total_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print pipeline summary
    logging.info("=== ML Pipeline Summary ===")
    logging.info(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logging.info(f"Data generation: {'Success' if pipeline_steps['data_generation'] else 'Skipped/Failed'}")
    logging.info(f"Cython build: {'Success' if pipeline_steps['cython_build'] else 'Skipped/Failed'}")
    
    for model_type in models_to_train:
        logging.info(f"{model_type} model:")
        logging.info(f"  Training: {'Success' if pipeline_steps['model_training'].get(model_type, False) else 'Failed'}")
        
        for subject_id in test_subjects:
            inference_result = pipeline_steps['inference'].get(model_type, {}).get(subject_id, False)
            logging.info(f"  Inference on {subject_id}: {'Success' if inference_result else 'Failed'}")
    
    logging.info(f"Visualization: {'Success' if pipeline_steps['visualization'] else 'Skipped/Failed'}")
    logging.info(f"Model milestones: {'Success' if pipeline_steps['milestones'] else 'Skipped/Failed'}")
    logging.info("===========================")


if __name__ == "__main__":
    main()