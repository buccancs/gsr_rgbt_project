Contactless GSR Estimation from RGB-Thermal Video
This repository contains the complete software implementation for the research project focused on estimating Galvanic
Skin Response (GSR) from synchronized RGB and thermal video streams. The project includes a data acquisition application
with a graphical user interface (GUI) and a full machine learning pipeline for data processing, model training, and
evaluation.

Project Architecture
The project is structured into two main parts:

Data Acquisition Application (src/): A PyQt5-based application for collecting synchronized multimodal data.

main.py: The main entry point that runs the GUI application.

gui/: Defines the main window and UI components.

capture/: Contains threaded modules for video and physiological sensor data capture.

utils/: Includes helper classes like the DataLogger.

config.py: A centralized file for all hardware and application settings.

Machine Learning Pipeline (src/processing, src/ml_models, src/scripts): A series of scripts to process the collected
data, train a predictive model, and evaluate its performance.

processing/: Modules for loading, preprocessing, and creating feature windows from the raw data.

ml_models/: Defines the neural network architectures (LSTM, AE, VAE) and model configuration system.
  - models.py: Implements the model architectures with configurable parameters.
  - model_config.py: Provides a flexible configuration system for model hyperparameters.

scripts/: Contains the high-level scripts for training, inference, and evaluation.

Setup and Installation
Follow these steps to set up the project environment.

1. Clone the Repository
   Clone this repository to your local machine.

git clone [https://github.com/your-username/gsr-rgbt-project.git](https://github.com/your-username/gsr-rgbt-project.git)
cd gsr-rgbt-project

2. Create a Python Virtual Environment
   It is strongly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

# Create the environment

python -m venv .venv

# Activate the environment

# On macOS/Linux:

source .venv/bin/activate

# On Windows:

.venv\Scripts\activate

3. Install Dependencies
   Install all required Python packages using the requirements.txt file.

pip install -r requirements.txt

4. Configure Hardware
   Before running the data collection application, you must configure your hardware settings in src/config.py:

Camera IDs: Run a camera utility on your machine to find the correct device indices for your RGB and thermal cameras.
Update RGB_CAMERA_ID and THERMAL_CAMERA_ID accordingly.

GSR Sensor: If you are using a physical Shimmer sensor, set GSR_SIMULATION_MODE = False and update GSR_SENSOR_PORT to
the correct serial port (e.g., 'COM3' on Windows).

How to Use the Pipeline
The project pipeline consists of three main stages, executed in sequence.

Stage 1: Data Collection
Run the GUI application to collect session data from participants.

python src/main.py

Launch the application.

Enter a unique Subject ID in the input field.

Click the Start Recording button to begin capturing video and GSR data.

Follow your experimental protocol (guiding the participant through tasks).

Click the Stop Recording button to end the session.

A new folder containing all recorded data for that session will be created in data/recordings/. Repeat for all
participants.

Stage 2: Model Training
Once you have collected data for all subjects, run the training script. This script supports different model types and configurations through a flexible command-line interface.

### Creating Example Configuration Files
First, create example configuration files for all supported model types:

```bash
python src/scripts/train_model.py --create-example-configs
```

This will create YAML configuration files in the `configs/models/` directory that you can customize for your experiments.

### Training with Default Configuration
To train a model using the default configuration (LSTM model with Leave-One-Subject-Out cross-validation):

```bash
python src/scripts/train_model.py
```

### Training with Custom Configuration
To train a specific model type with a custom configuration:

```bash
python src/scripts/train_model.py --model-type lstm --config-path configs/models/lstm_config.yaml
```

### Using K-Fold Cross-Validation
To use k-fold cross-validation instead of Leave-One-Subject-Out:

```bash
python src/scripts/train_model.py --cv-folds 5
```

### Specifying Output Directory
To save models and results to a custom directory:

```bash
python src/scripts/train_model.py --output-dir path/to/output
```

The training script will:

- Load and process the data for all subjects found in data/recordings/.
- Perform cross-validation based on the specified method (LOSO by default).
- Save the trained model (.keras) and the corresponding data scaler (.joblib) for each fold.
- Generate logs for TensorBoard.
- Save the final cross-validation performance metrics to a CSV file.

Stage 3: Inference and Evaluation
After training, you need to run inference on your test data and then evaluate the results.

### Running Inference
Use the inference script to generate predictions for a specific subject:

```bash
python src/scripts/inference.py --model-type lstm --model-path data/recordings/models/lstm_fold_1_subject_Subject01.keras --scaler-path data/recordings/models/scaler_lstm_fold_1_subject_Subject01.joblib --subject-id Subject01
```

This script will:
- Load the specified trained model and scaler
- Process the data for the specified subject
- Generate predictions
- Save the results to a CSV file in data/recordings/predictions/

### Running Evaluation
After generating predictions, run the evaluation script to visualize the results:

```bash
python src/scripts/evaluate_model.py
```

The script is pre-configured to evaluate the results for Subject01. You can change the TEST_SUBJECT_ID inside the script to evaluate other subjects.

It generates and saves two plots:
- A time-series plot comparing the predicted GSR vs. the ground truth.
- A Bland-Altman plot to assess the agreement between the two measurements.

All output plots are saved to data/recordings/evaluation_plots/.

## Model Configuration System

The project includes a flexible configuration system for machine learning models, allowing you to easily experiment with different architectures and hyperparameters.

### Configuration Files
Model configurations are stored in YAML files with the following structure:

```yaml
# Example LSTM configuration
name: lstm
layers:
  - type: lstm
    units: 64
    return_sequences: true
  - type: dropout
    rate: 0.2
  - type: lstm
    units: 32
    return_sequences: false
  - type: dropout
    rate: 0.2
  - type: dense
    units: 16
    activation: relu
  - type: dense
    units: 1
    activation: linear
compile_params:
  optimizer:
    type: adam
    learning_rate: 0.001
  loss: mean_absolute_error
  metrics:
    - mean_squared_error
fit_params:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  callbacks:
    early_stopping:
      monitor: val_loss
      patience: 10
      restore_best_weights: true
    model_checkpoint:
      save_best_only: true
      monitor: val_loss
    tensorboard: {}
```

### Using Configurations in Code
You can load and modify configurations programmatically:

```python
from src.ml_models.model_config import ModelConfig

# Load a default configuration
config = ModelConfig("lstm")

# Update specific parameters
config.update_config({
    "layers": [
        {"type": "lstm", "units": 128, "return_sequences": True},
        {"type": "dropout", "rate": 0.3},
        # ... other layers
    ],
    "compile_params": {
        "optimizer": {
            "learning_rate": 0.0005
        }
    }
})

# Save the configuration to a file
config.save_to_file(Path("configs/models/custom_lstm.yaml"))
```

This configuration system makes it easy to experiment with different model architectures and hyperparameters without modifying the code.
