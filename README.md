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

ml_models/: Defines the neural network architectures and model configuration system.
  - model_interface.py: Provides a common interface for all models, regardless of framework.
  - models.py: Implements TensorFlow/Keras model architectures (legacy support).
  - pytorch_models.py: Implements PyTorch versions of LSTM, Autoencoder, and VAE models.
  - pytorch_cnn_models.py: Implements CNN and CNN-LSTM hybrid models.
  - pytorch_transformer_models.py: Implements Transformer models for time series data.
  - pytorch_resnet_models.py: Implements ResNet models for time series data.
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
Once you have collected data for all subjects, you can either run the individual training script or use the config-driven pipeline execution script. Both support different model types and configurations through a flexible command-line interface.

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

### Using the Config-Driven Pipeline
For a more streamlined experience, you can use the config-driven pipeline execution script, which handles all stages of the pipeline based on a single configuration file:

```bash
python src/scripts/run_ml_pipeline_from_config.py --config-path configs/pipeline/pipeline_config.yaml
```

To create an example pipeline configuration file:

```bash
python src/scripts/run_ml_pipeline_from_config.py --create-example-config
```

The pipeline configuration file allows you to specify:
- Data generation settings
- Which models to train and their configurations
- Test subjects for inference
- Visualization settings
- Model milestone settings

The training script will:

- Load and process the data for all subjects found in data/recordings/.
- Perform cross-validation based on the specified method (LOSO by default).
- Save the trained model (.keras) and the corresponding data scaler (.joblib) for each fold.
- Generate logs for TensorBoard.
- Save the final cross-validation performance metrics to a CSV file.

Stage 3: Inference and Evaluation
After training, you need to run inference on your test data and then evaluate the results. You can do this using individual scripts or the config-driven pipeline.

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

### Running Evaluation and Visualization
After generating predictions, run the visualization script to create plots and reports:

```bash
python src/scripts/visualize_results.py --plot-history --plot-predictions --model-comparison --annotate-graphs
```

Command-line options:
- `--plot-history`: Plot training history for all models
- `--plot-predictions`: Plot predictions vs. ground truth for all prediction files
- `--model-comparison`: Generate model comparison reports
- `--annotate-graphs`: Annotate graphs with additional information (model config, metrics, etc.)
- `--output-dir`: Custom output directory for visualizations
- `--save-milestone MODEL_TYPE`: Save a milestone of the specified model type
- `--milestone-name NAME`: Name for the milestone (used with --save-milestone)
- `--all`: Run all visualization and reporting tasks

The script generates and saves various plots:
- Training history plots showing loss curves
- Prediction plots comparing the predicted values vs. the ground truth
- Model comparison reports with performance metrics
- All plots can be annotated with model configuration details, metrics, and timestamps

All output plots are saved to data/recordings/evaluation_plots/ by default, or to the specified output directory.

## Model Configuration System

The project includes a flexible configuration system for machine learning models, allowing you to easily experiment with different architectures and hyperparameters.

### Supported Model Architectures
The project supports the following neural network architectures:

- **LSTM**: Long Short-Term Memory networks for sequence modeling
- **Autoencoder**: For unsupervised feature learning and anomaly detection
- **VAE**: Variational Autoencoders for generative modeling
- **CNN**: Convolutional Neural Networks for feature extraction
- **CNN-LSTM**: Hybrid model combining CNN and LSTM layers
- **Transformer**: Self-attention based models for sequence data
- **ResNet**: Residual Networks with skip connections for deep architectures

### Configuration Files
Model configurations are stored in YAML files with the following structure:

```yaml
# Example LSTM configuration
name: lstm
framework: pytorch
model_params:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  bidirectional: false
  fc_layers: [32, 16, 1]
  activations: ["relu", "relu", "linear"]
optimizer_params:
  type: adam
  lr: 0.001
  weight_decay: 1e-5
loss_fn: mse
train_params:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  early_stopping:
    patience: 10
    monitor: val_loss
  checkpoint:
    save_best_only: true
    monitor: val_loss
```

### Model-Specific Configuration Parameters

Different model architectures require different configuration parameters:

#### Transformer Model
```yaml
model_params:
  d_model: 64        # Dimension of the model
  nhead: 4           # Number of attention heads
  num_layers: 2      # Number of transformer layers
  dim_feedforward: 256  # Dimension of feedforward network
  dropout: 0.1
  fc_layers: [32, 16, 1]
  activations: ["relu", "relu", "linear"]
```

#### ResNet Model
```yaml
model_params:
  layers: [64, 128, 256, 512]  # Number of channels for each layer
  blocks_per_layer: [2, 2, 2, 2]  # Number of residual blocks per layer
  fc_layers: [256, 64, 1]
  activations: ["relu", "relu", "linear"]
  dropout_rate: 0.2
```

### Using Configurations in Code
You can load and modify configurations programmatically:

```python
from src.ml_models.model_config import ModelConfig

# Load a default configuration
config = ModelConfig("transformer")

# Update specific parameters
config.update_config({
    "model_params": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 512
    },
    "optimizer_params": {
        "lr": 0.0005
    }
})

# Save the configuration to a file
config.save_to_file(Path("configs/models/custom_transformer.yaml"))
```

### Pipeline Configuration
In addition to model configurations, you can also create pipeline configurations that specify the entire ML workflow:

```yaml
data_generation:
  enabled: true
  num_subjects: 10
  sessions_per_subject: 2
cython_build:
  enabled: true
models:
  - name: lstm
    config:
      model_params:
        hidden_size: 64
        num_layers: 2
  - name: transformer
    config:
      model_params:
        d_model: 64
        nhead: 4
test_subjects: ["MockSubject01", "MockSubject02"]
visualization:
  enabled: true
  plot_history: true
  plot_predictions: true
  model_comparison: true
  output_dir: "outputs/visualizations"
  annotate_graphs: true
milestones:
  enabled: true
  milestone_name: "final"
```

This configuration system makes it easy to experiment with different model architectures and hyperparameters without modifying the code.
