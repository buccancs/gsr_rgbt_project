# src/scripts/run_ml_pipeline_from_config.py

"""
Script to run the ML pipeline from a configuration file.

This script orchestrates the entire ML pipeline based on a configuration file:
1. Loads the pipeline configuration from a YAML file
2. Generates mock data for training if needed
3. Builds Cython extensions for performance optimization if needed
4. Trains models specified in the configuration
5. Runs inference on test subjects
6. Visualizes and repo_forensic results with annotations
7. Saves model milestones

It serves as a config-driven solution for running the complete pipeline.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.ml_models.model_config import ModelConfig

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_pipeline.log"),
        logging.StreamHandler()
    ]
)

# --- Default pipeline configuration ---
DEFAULT_PIPELINE_CONFIG = {
    "data_generation": {
        "enabled": True,
        "num_subjects": 10,
        "sessions_per_subject": 2
    },
    "cython_build": {
        "enabled": True
    },
    "models": [
        {
            "name": "lstm",
            "config": {
                "model_params": {
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "bidirectional": False,
                    "fc_layers": [32, 16, 1],
                    "activations": ["relu", "relu", "linear"]
                }
            }
        },
        {
            "name": "transformer",
            "config": {
                "model_params": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 256,
                    "dropout": 0.1,
                    "fc_layers": [32, 16, 1],
                    "activations": ["relu", "relu", "linear"]
                }
            }
        }
    ],
    "test_subjects": ["MockSubject01", "MockSubject02"],
    "visualization": {
        "enabled": True,
        "plot_history": True,
        "plot_predictions": True,
        "model_comparison": True,
        "output_dir": "outputs/visualizations",
        "annotate_graphs": True
    },
    "milestones": {
        "enabled": True,
        "milestone_name": "final"
    }
}


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


def generate_mock_data(config_data):
    """
    Generate mock data for training.
    
    Args:
        config_data (dict): Configuration data for data generation
        
    Returns:
        bool: True if data generation succeeded, False otherwise
    """
    num_subjects = config_data.get("num_subjects", 10)
    sessions_per_subject = config_data.get("sessions_per_subject", 2)

    logging.info(f"Generating mock data for {num_subjects} subjects, {sessions_per_subject} sessions each")

    # Run the generate_training_data.py script
    command = f"python {project_root / 'src' / 'scripts' / 'generate_training_data.py'} --num-subjects {num_subjects} --sessions-per-subject {sessions_per_subject}"
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


def create_model_config_file(model_name, model_config, output_dir):
    """
    Create a model configuration file.
    
    Args:
        model_name (str): Name of the model
        model_config (dict): Model configuration
        output_dir (Path): Directory to save the configuration file
        
    Returns:
        Path: Path to the created configuration file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{model_name}_config.yaml"

    # Create a ModelConfig instance
    model_config_obj = ModelConfig(model_name)

    # Update with custom configuration
    if model_config:
        model_config_obj.update_config(model_config)

    # Save to file
    model_config_obj.save_to_file(config_path)

    logging.info(f"Created model configuration file: {config_path}")
    return config_path


def train_model(model_name, model_config_path):
    """
    Train a model with cross-validation.
    
    Args:
        model_name (str): Type of model to train
        model_config_path (Path): Path to the model configuration file
        
    Returns:
        bool: True if training succeeded, False otherwise
    """
    logging.info(f"Training {model_name} model")

    # Run the train_model.py script
    command = f"python {project_root / 'src' / 'scripts' / 'train_model.py'} --model-type {model_name} --config-path {model_config_path}"
    return run_command(command, f"{model_name} model training")


def run_inference(model_name, subject_id, model_path=None, scaler_path=None):
    """
    Run inference on a test subject.
    
    Args:
        model_name (str): Type of model to use for inference
        subject_id (str): ID of the subject to run inference on
        model_path (str, optional): Path to the model file. If None, will be auto-detected.
        scaler_path (str, optional): Path to the scaler file. If None, will be auto-detected.
        
    Returns:
        bool: True if inference succeeded, False otherwise
    """
    logging.info(f"Running inference with {model_name} model on subject {subject_id}")

    # If model_path and scaler_path are not provided, try to find them
    if model_path is None or scaler_path is None:
        models_dir = config.OUTPUT_DIR / "models"

        # Find the model file
        if model_path is None:
            model_files = list(models_dir.glob(f"{model_name}*fold_1_subject_{subject_id}.*"))
            if model_files:
                model_path = model_files[0]
            else:
                logging.error(f"Could not find model file for {model_name} and subject {subject_id}")
                return False

        # Find the scaler file
        if scaler_path is None:
            scaler_files = list(models_dir.glob(f"scaler_{model_name}*fold_1_subject_{subject_id}.*"))
            if scaler_files:
                scaler_path = scaler_files[0]
            else:
                logging.error(f"Could not find scaler file for {model_name} and subject {subject_id}")
                return False

    # Run the inference.py script
    command = (f"python {project_root / 'src' / 'scripts' / 'inference.py'} "
               f"--model-type {model_name} "
               f"--model-path {model_path} "
               f"--scaler-path {scaler_path} "
               f"--subject-id {subject_id}")

    return run_command(command, f"Inference with {model_name} on {subject_id}")


def visualize_results(config_data):
    """
    Visualize and report results.
    
    Args:
        config_data (dict): Configuration data for visualization
        
    Returns:
        bool: True if visualization succeeded, False otherwise
    """
    logging.info("Visualizing and reporting results")

    plot_history = config_data.get("plot_history", True)
    plot_predictions = config_data.get("plot_predictions", True)
    model_comparison = config_data.get("model_comparison", True)
    output_dir = config_data.get("output_dir", "outputs/visualizations")
    annotate_graphs = config_data.get("annotate_graphs", True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build command with appropriate flags
    command = f"python {project_root / 'src' / 'scripts' / 'visualize_results.py'} --output-dir {output_dir}"

    if plot_history:
        command += " --plot-history"

    if plot_predictions:
        command += " --plot-predictions"

    if model_comparison:
        command += " --model-comparison"

    if annotate_graphs:
        command += " --annotate-graphs"

    return run_command(command, "Results visualization")


def save_model_milestones(models, milestone_name):
    """
    Save milestones for all models.
    
    Args:
        models (list): List of model configurations
        milestone_name (str): Name of the milestone
        
    Returns:
        bool: True if saving milestones succeeded, False otherwise
    """
    logging.info("Saving model milestones")

    success = True
    for model_config in models:
        model_name = model_config["name"]

        # Run the visualize_results.py script with --save-milestone flag
        command = (f"python {project_root / 'src' / 'scripts' / 'visualize_results.py'} "
                   f"--save-milestone {model_name} "
                   f"--milestone-name {milestone_name}")

        if not run_command(command, f"Saving milestone for {model_name}"):
            success = False

    return success


def create_example_pipeline_config(output_path):
    """
    Create an example pipeline configuration file.
    
    Args:
        output_path (Path): Path to save the configuration file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(DEFAULT_PIPELINE_CONFIG, f, default_flow_style=False)

    logging.info(f"Created example pipeline configuration file: {output_path}")


def load_pipeline_config(config_path):
    """
    Load pipeline configuration from a YAML file.
    
    Args:
        config_path (Path): Path to the configuration file
        
    Returns:
        dict: Pipeline configuration
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        logging.info(f"Loaded pipeline configuration from {config_path}")
        return config_data
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        logging.info("Using default pipeline configuration")
        return DEFAULT_PIPELINE_CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the ML pipeline from a configuration file")

    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the pipeline configuration file"
    )

    parser.add_argument(
        "--create-example-config",
        action="store_true",
        help="Create an example pipeline configuration file"
    )

    parser.add_argument(
        "--example-config-path",
        type=str,
        default="configs/pipeline/example_pipeline_config.yaml",
        help="Path to save the example configuration file"
    )

    return parser.parse_args()


def main():
    """Main function to run the ML pipeline from a configuration file."""
    start_time = time.time()
    logging.info("=== Starting Config-Driven ML Pipeline ===")

    args = parse_arguments()

    # Create example configuration if requested
    if args.create_example_config:
        create_example_pipeline_config(Path(args.example_config_path))
        return

    # Load pipeline configuration
    if args.config_path:
        pipeline_config = load_pipeline_config(Path(args.config_path))
    else:
        logging.warning("No configuration file provided. Using default configuration.")
        pipeline_config = DEFAULT_PIPELINE_CONFIG

    # Create configs directory if it doesn't exist
    configs_dir = project_root / "configs" / "models"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Track pipeline steps
    pipeline_steps = {
        "data_generation": pipeline_config.get("data_generation", {}).get("enabled", True),
        "cython_build": pipeline_config.get("cython_build", {}).get("enabled", True),
        "model_training": {},
        "inference": {},
        "visualization": pipeline_config.get("visualization", {}).get("enabled", True),
        "milestones": pipeline_config.get("milestones", {}).get("enabled", True)
    }

    # 1. Generate mock data
    if pipeline_steps["data_generation"]:
        data_gen_config = pipeline_config.get("data_generation", {})
        pipeline_steps["data_generation"] = generate_mock_data(data_gen_config)
    else:
        logging.info("Skipping mock data generation")

    # 2. Build Cython extensions
    if pipeline_steps["cython_build"]:
        pipeline_steps["cython_build"] = build_cython_extensions()
    else:
        logging.info("Skipping Cython build")

    # 3. Train models
    models = pipeline_config.get("models", [])
    test_subjects = pipeline_config.get("test_subjects", ["MockSubject01", "MockSubject02"])

    for model_config in models:
        model_name = model_config["name"]
        model_custom_config = model_config.get("config", {})

        logging.info(f"=== Processing {model_name} model ===")

        # Create model configuration file
        model_config_path = create_model_config_file(model_name, model_custom_config, configs_dir)

        # Train the model
        pipeline_steps["model_training"][model_name] = train_model(model_name, model_config_path)

        # Run inference for each test subject
        pipeline_steps["inference"][model_name] = {}
        for subject_id in test_subjects:
            pipeline_steps["inference"][model_name][subject_id] = run_inference(model_name, subject_id)

    # 4. Visualize results
    if pipeline_steps["visualization"]:
        visualization_config = pipeline_config.get("visualization", {})
        pipeline_steps["visualization"] = visualize_results(visualization_config)
    else:
        logging.info("Skipping visualization")

    # 5. Save model milestones
    if pipeline_steps["milestones"]:
        milestone_config = pipeline_config.get("milestones", {})
        milestone_name = milestone_config.get("milestone_name", "final")
        pipeline_steps["milestones"] = save_model_milestones(models, milestone_name)
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

    for model_name in [model["name"] for model in models]:
        logging.info(f"{model_name} model:")
        logging.info(
            f"  Training: {'Success' if pipeline_steps['model_training'].get(model_name, False) else 'Failed'}")

        for subject_id in test_subjects:
            inference_result = pipeline_steps['inference'].get(model_name, {}).get(subject_id, False)
            logging.info(f"  Inference on {subject_id}: {'Success' if inference_result else 'Failed'}")

    logging.info(f"Visualization: {'Success' if pipeline_steps['visualization'] else 'Skipped/Failed'}")
    logging.info(f"Model milestones: {'Success' if pipeline_steps['milestones'] else 'Skipped/Failed'}")
    logging.info("===========================")


if __name__ == "__main__":
    main()
