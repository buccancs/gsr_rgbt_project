# Makefile for the GSR-RGBT Project
# This file automates the setup, data processing, training, and evaluation pipeline.

# --- Variables ---
# Use the system's Python interpreter by default.
# For virtual environments, this will point to the venv Python.
PYTHON = python

# --- Phony targets do not correspond to actual files ---
.PHONY: all setup clean test run_app train inference evaluate pipeline mock_data build_cython

# --- Main Targets ---

# The default target, run when you just type 'make'
all:
	@echo "Makefile for GSR-RGBT Project"
	@echo "-----------------------------------"
	@echo "Available commands:"
	@echo "  make setup        - Creates a virtual environment and installs dependencies."
	@echo "  make clean        - Removes temporary files and build artifacts."
	@echo "  make build_cython - Builds Cython extensions for performance optimization."
	@echo "  make test         - Runs system validation checks for cameras and dependencies."
	@echo "  make run_app      - Runs the data collection GUI application."
	@echo "  make mock_data    - Generates synthetic data for testing the pipeline."
	@echo "  make train        - Runs the model training and cross-validation script."
	@echo "  make inference    - Runs inference on test data using a trained model."
	@echo "  make evaluate     - Generates evaluation plots from prediction results."
	@echo "  make pipeline     - Runs the full train, inference, and evaluate pipeline in sequence."
	@echo "-----------------------------------"

# Target to set up the project environment
setup:
	@echo ">>> Setting up Python virtual environment..."
	$(PYTHON) -m venv .venv
	@echo ">>> Activating environment and installing dependencies from requirements.txt..."
	@# The following command must be run in a shell that supports this syntax.
	@. .venv/bin/activate && pip install -r requirements.txt || \
		echo "Failed to install dependencies. Please activate the venv manually ('source .venv/bin/activate') and run 'pip install -r requirements.txt'"
	@echo ">>> Building Cython extensions..."
	@. .venv/bin/activate && $(PYTHON) setup.py build_ext --inplace || \
		echo "Failed to build Cython extensions. Please activate the venv manually and run 'python setup.py build_ext --inplace'"
	@echo "\nSetup complete. To activate the environment, run: source .venv/bin/activate"

# Target to build Cython extensions
build_cython:
	@echo ">>> Building Cython extensions..."
	$(PYTHON) setup.py build_ext --inplace

# Target to clean the project directory
clean:
	@echo ">>> Cleaning up project directory..."
	@rm -rf .venv __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
	@rm -f data/recordings/models/*.keras data/recordings/models/*.joblib data/recordings/models/*.pt
	@rm -rf data/recordings/models/logs/
	@rm -f data/recordings/predictions/*.csv
	@rm -f data/recordings/evaluation_plots/*.png
	@rm -f data/recordings/cross_validation_results.csv
	@rm -rf build/ dist/ *.egg-info/
	@rm -f src/processing/*.c src/processing/*.so src/processing/*.pyd
	@echo "Clean complete."

# --- Application and Pipeline Targets ---

# Target to run the system validation script
test:
	@echo ">>> Running system validation checks..."
	@$(PYTHON) src/scripts/check_system.py

# Target to run the data collection application
run_app:
	@echo ">>> Launching Data Collection Application..."
	@$(PYTHON) src/main.py

# Target to run the model training script
train:
	@echo ">>> Starting Model Training (LOSO Cross-Validation)..."
	@$(PYTHON) src/scripts/train_model.py

# Target to run the inference script
inference:
	@echo ">>> Running Inference on Test Data..."
	@$(PYTHON) src/scripts/inference.py

# Target to run the evaluation script
evaluate:
	@echo ">>> Generating Evaluation Plots and Metrics..."
	@$(PYTHON) src/scripts/evaluate_model.py

# Target to generate mock data for testing the pipeline
mock_data:
	@echo ">>> Generating Mock Data for Pipeline Testing..."
	@$(PYTHON) src/scripts/create_mock_data.py

# Target to run the full machine learning pipeline sequentially
pipeline: build_cython train inference evaluate
	@echo "\n>>> Full ML pipeline (build_cython -> train -> inference -> evaluate) complete."
