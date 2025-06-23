# Contactless GSR Estimation from RGB-Thermal Video

![CI Status](https://github.com/username/gsr_rgbt_project/workflows/GSR-RGBT%20CI/badge.svg)
[![codecov](https://codecov.io/gh/username/gsr_rgbt_project/branch/main/graph/badge.svg)](https://codecov.io/gh/username/gsr_rgbt_project)

> **Note**: To see the GitHub Actions workflow files, make sure to:
> 1. Commit and push the `.github/workflows/ci.yml` file to your GitHub repository
> 2. View the workflows in GitHub by navigating to the "Actions" tab in your repository
> 3. If you don't see the workflows, check that the file is in the correct location (`.github/workflows/ci.yml`)

This repository contains the complete software implementation for the research project focused on estimating Galvanic
Skin Response (GSR) from synchronized RGB and thermal video streams. The project includes a data acquisition application
with a graphical user interface (GUI) and a full machine learning pipeline for data processing, model training, and
evaluation.

## Features

- **Multi-modal Data Collection**: Synchronized capture from RGB cameras, thermal cameras (FLIR), and GSR sensors (Shimmer3 GSR+)
- **Real-time Processing**: Multi-threaded architecture for responsive data acquisition
- **Machine Learning Pipeline**: Complete ML workflow with multiple model architectures (LSTM, CNN, Transformer, ResNet, VAE)
- **Flexible Configuration**: YAML-based configuration system for models and pipelines
- **Comprehensive Testing**: Full test suite with coverage reporting and CI/CD integration
- **Cross-platform Support**: Works on Windows, macOS, and Linux

## Quick Start

Get up and running in 3 simple steps:

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/your-organization/gsr-rgbt-project.git
   cd gsr-rgbt-project
   chmod +x gsr_rgbt_tools.sh
   ./gsr_rgbt_tools.sh setup
   ```

2. **Run Data Collection**:
   ```bash
   ./gsr_rgbt_tools.sh collect
   ```

3. **Train Models**:
   ```bash
   ./gsr_rgbt_tools.sh train --config configs/pipeline/default.yaml
   ```

For detailed instructions, see the collapsible sections below.

## Detailed Guide

<details>
<summary><strong>🔧 Installation & Setup</strong> (Click to expand)</summary>

### Prerequisites
- Git
- Python 3.9+
- Windows Subsystem for Linux (WSL) or a native Linux/macOS environment

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-organization/gsr-rgbt-project.git
cd gsr-rgbt-project
```

### Step 2: Run the Unified Tool Script
This script will guide you through installing all dependencies, including Python packages and the FLIR Spinnaker SDK.

```bash
# Make the script executable (if needed)
chmod +x gsr_rgbt_tools.sh

# Run the setup command
./gsr_rgbt_tools.sh setup
```

This script will:
- Check for required system dependencies
- Set up a Python virtual environment
- Install required Python packages
- Build Cython extensions
- Run system validation checks
- Provide guidance for hardware-specific setup

### Alternative Manual Setup
If you prefer to set up manually:

```bash
# Create and activate a virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt

# Build Cython extensions
python setup.py build_ext --inplace
```

Or use the Makefile:
```bash
make setup
```

</details>

<details>
<summary><strong>⚙️ Hardware Setup</strong> (Click to expand)</summary>

### Shimmer3 GSR+ Sensor Setup

1. **Pairing**: Use your operating system's Bluetooth settings to pair the device.
2. **Connection**: The application will automatically detect the COM port. Ensure the device is powered on before starting the application.

#### Detailed Shimmer Setup Steps:

1. **Install Shimmer Connect or ConsensysPRO**:
   - Download [Shimmer Connect](https://shimmersensing.com/support/wireless-sensor-networks-download/) or [ConsensysPRO](https://shimmersensing.com/support/consensys-download/)
   - Use this software to:
     - Update the Shimmer's firmware
     - Configure its settings (enable the GSR sensor and set the sampling rate)
     - Pair it via Bluetooth (if using Bluetooth instead of the dock's serial connection)

2. **Identify the Serial Port**:
   - When connecting the Shimmer via its USB dock, it will appear as a serial port
   - On macOS: `/dev/tty.usbmodem*` or `/dev/cu.*`
   - On Windows: `COM*` ports
   - On Linux: `/dev/ttyUSB*` or `/dev/ttyACM*`

3. **USB-to-Serial Driver (if needed)**:
   - Most operating systems have built-in drivers for common USB-to-Serial chips
   - If the device isn't recognized, you might need to install a specific driver for the chip used in the Shimmer dock
   - Look for the chip model on the dock (often FTDI or CP210x) and download the appropriate driver

### FLIR Camera (Spinnaker)

The FLIR thermal camera **requires** the Spinnaker SDK, which provides the PySpin library used by the application.

1. **Download the FLIR Spinnaker SDK**:
   - Visit the [FLIR website](https://www.flir.com/products/spinnaker-sdk/)
   - You will need to create a free FLIR account if you don't already have one
   - Select the appropriate version for your operating system

2. **Install the Spinnaker SDK**:
   - Open the downloaded package and follow the installation wizard
   - **IMPORTANT**: During installation, make sure to select the option to install the Python bindings (PySpin)
   - You may need administrator privileges to complete the installation
   - After installation, you may need to restart your computer

3. **Verify the installation**:
   ```bash
   # Activate your Python environment first if you're using one
   source .venv/bin/activate

   # Then run this command to verify PySpin is installed
   python -c "import PySpin; system = PySpin.System.GetInstance(); version = system.GetLibraryVersion(); print(f'PySpin installed successfully. Version: {version.major}.{version.minor}.{version.type}.{version.build}'); system.ReleaseInstance()"
   ```

4. **Troubleshooting SDK Installation**:
   - If you get an import error for PySpin, ensure that you installed the Python bindings during SDK installation
   - If you're using a virtual environment, you may need to reinstall the SDK or create a symlink to the PySpin module
   - On some systems, you might need to install additional dependencies like libusb
   - Check the FLIR Spinnaker SDK documentation for platform-specific installation issues

</details>

<details>
<summary><strong>🚀 Usage</strong> (Click to expand)</summary>

### Running the Data Collection Application

1. **Start the GUI Application**:
   ```bash
   ./gsr_rgbt_tools.sh collect
   ```

   Or manually:
   ```bash
   python src/data_collection/main.py
   ```

2. **Configure Data Collection**:
   - Set the subject ID and session parameters
   - Configure camera settings (resolution, frame rate)
   - Set GSR sensor parameters
   - Choose output directory

3. **Start Recording**:
   - Click "Start Recording" to begin synchronized data capture
   - Monitor real-time data streams
   - Stop recording when complete

### Running the Machine Learning Pipeline

1. **Process Collected Data**:
   ```bash
   ./gsr_rgbt_tools.sh process --input data/recordings/subject01
   ```

2. **Train Models**:
   ```bash
   ./gsr_rgbt_tools.sh train --config configs/pipeline/default.yaml
   ```

3. **Evaluate Models**:
   ```bash
   ./gsr_rgbt_tools.sh evaluate --model outputs/models/lstm_model.pth
   ```

### Configuration Options

The system uses YAML configuration files for flexible setup:

- **Model Configurations**: `configs/models/`
- **Pipeline Configurations**: `configs/pipeline/`
- **Hardware Configurations**: `src/data_collection/config.py`

</details>

<details>
<summary><strong>🤔 Troubleshooting</strong> (Click to expand)</summary>

### Common Issues and Solutions

**Problem: Shimmer device not found**
- **Solution 1**: Ensure the device is fully charged and powered on
- **Solution 2**: Verify that the device is successfully paired in your system's Bluetooth settings
- **Solution 3**: On Windows, check the Device Manager to see if the virtual COM port has been created
- **Solution 4**: Try reconnecting the USB dock or restarting the Bluetooth connection

**Problem: FLIR camera not detected**
- **Solution 1**: Verify that the Spinnaker SDK is properly installed with Python bindings
- **Solution 2**: Check that the camera is connected via USB 3.0 for optimal performance
- **Solution 3**: Ensure no other applications are using the camera
- **Solution 4**: Try running the camera test utility provided with the Spinnaker SDK

**Problem: Python import errors**
- **Solution 1**: Ensure you're using the correct Python environment (activate your virtual environment)
- **Solution 2**: Reinstall dependencies: `pip install -r requirements.txt`
- **Solution 3**: Rebuild Cython extensions: `python setup.py build_ext --inplace`

**Problem: GUI application crashes**
- **Solution 1**: Check the console output for error messages
- **Solution 2**: Verify all hardware is properly connected and configured
- **Solution 3**: Try running in simulation mode first to isolate hardware issues

**Problem: Low performance or dropped frames**
- **Solution 1**: Close unnecessary applications to free up system resources
- **Solution 2**: Use USB 3.0 ports for cameras
- **Solution 3**: Reduce camera resolution or frame rate if needed
- **Solution 4**: Ensure adequate disk space for data storage

</details>

<details>
<summary><strong>👨‍💻 Developer Guide</strong> (Click to expand)</summary>

### Development Setup

1. **Install Development Dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

3. **Run Tests**:
   ```bash
   ./gsr_rgbt_tools.sh test
   ```

### Code Style and Standards

- **Python**: Follow PEP 8 guidelines
- **Code Formatting**: Use Black for automatic formatting
- **Linting**: Use flake8 for code quality checks
- **Type Hints**: Use type annotations where appropriate

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `make test`
5. Submit a pull request

### Testing

- **Unit Tests**: Located in `src/tests/`
- **Integration Tests**: Test complete workflows
- **Coverage**: Aim for >80% code coverage
- **CI/CD**: Automated testing via GitHub Actions

For detailed developer information, see [docs/developer/DEVELOPER_GUIDE.md](docs/developer/DEVELOPER_GUIDE.md).

</details>

## Project Architecture

<details>
<summary><strong>🏗️ System Architecture</strong> (Click to expand)</summary>

### Overview

This project is designed as a modular, multi-threaded application for synchronized, multi-modal physiological data collection and subsequent machine learning analysis. The architecture is divided into three main components: `data_collection`, `ml_pipeline`, and `utils`.

### Core Components

- **`data_collection/`**: Contains the main PyQt5 application and all modules related to real-time data acquisition.
  - **`main.py`**: The entry point for the GUI application. It initializes all capture threads and manages the application state.
  - **`capture/`**: Holds individual `QThread` classes for each sensor (RGB camera, thermal camera, Shimmer GSR). This ensures the GUI remains responsive during data capture.
  - **`ui/`**: Contains the user interface files generated from Qt Designer (`.ui`).

- **`ml_pipeline/`**: Includes all scripts and modules for offline data processing and machine learning.
  - **`run_ml_pipeline_from_config.py`**: The master script to run the entire ML pipeline based on a YAML configuration file.
  - **`models/`**: Defines the neural network architectures.
  - **`processing/`**: Contains scripts for data loading, preprocessing, and feature extraction.

- **`utils/`**: A collection of shared utility modules used across the project.
  - **`data_logger.py`**: A robust class for handling the writing of all data streams to disk.
  - **`device_utils.py`**: Contains helper functions for hardware, such as the automatic Shimmer COM port detection.

### Data Flow

1. **Data Collection**: Multi-threaded capture from various sensors
2. **Data Storage**: Synchronized writing to disk with timestamps
3. **Data Processing**: Feature extraction and preprocessing
4. **Model Training**: ML pipeline with configurable architectures
5. **Evaluation**: Performance assessment and visualization

For detailed architecture information, see [docs/technical/ARCHITECTURE.md](docs/technical/ARCHITECTURE.md).

</details>

## Documentation

For comprehensive documentation, please refer to:

- **[User Guide](docs/user/USER_GUIDE.md)**: Complete tutorial for using the system
- **[Developer Guide](docs/developer/DEVELOPER_GUIDE.md)**: Contributing guidelines and development workflow
- **[Architecture Guide](docs/technical/ARCHITECTURE.md)**: Detailed system architecture and design
- **[Technical Guide](docs/technical/technical_guide.md)**: Hardware integration and technical details
- **[Testing Guide](docs/developer/testing_guide.md)**: Testing strategy and execution

## Command Line Interface

The project includes a unified tool script that provides easy access to all functionality:

```bash
# Get help and see all available commands
./gsr_rgbt_tools.sh help

# Setup the environment
./gsr_rgbt_tools.sh setup

# Run data collection
./gsr_rgbt_tools.sh collect

# Train models
./gsr_rgbt_tools.sh train --config configs/pipeline/default.yaml

# Run tests
./gsr_rgbt_tools.sh test
```

## Citation & License

### Citation

If you use this software in your research, please cite:

```bibtex
@article{gsr_rgbt_2024,
  title={Contactless GSR Estimation from RGB-Thermal Video},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]}
}
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- FLIR Systems for the Spinnaker SDK
- Shimmer Research for the GSR+ sensor platform
- The open-source community for the various libraries and tools used in this project

---

**Note**: This project is for research purposes. Ensure compliance with all applicable regulations and ethical guidelines when collecting physiological data.


#### Training with Default Configuration

To train a model using the default configuration (LSTM model with Leave-One-Subject-Out cross-validation):

```bash
python src/scripts/train_model.py
```

#### Training with Custom Configuration

To train a specific model type with a custom configuration:

```bash
python src/scripts/train_model.py --model-type lstm --config-path configs/models/lstm_config.yaml
```

#### Using K-Fold Cross-Validation

To use k-fold cross-validation instead of Leave-One-Subject-Out:

```bash
python src/scripts/train_model.py --cv-folds 5
```

#### Specifying Validation Split

To specify the fraction of training data to use for validation:

```bash
python src/scripts/train_model.py --validation-split 0.2
```

#### Saving Training Metadata

By default, the training script saves detailed metadata about the training process. This includes model configuration, preprocessing parameters, training parameters, and evaluation metrics. You can disable this feature if needed:

```bash
python src/scripts/train_model.py --save-metadata false
```

#### Specifying Output Directory

To save models and results to a custom directory:

```bash
python src/scripts/train_model.py --output-dir path/to/output
```

#### Using the Config-Driven Pipeline

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
- Extract signals from multiple ROIs (index finger base, ring finger base, palm center) using MediaPipe hand landmark detection.
- Perform cross-validation based on the specified method (LOSO by default).
- Implement a proper train/validation/test split with subject-aware validation to prevent data leakage.
- Save the trained model (.keras or .pt) and the corresponding data scaler (.joblib) for each fold.
- Save detailed metadata about the training process, including model configuration, preprocessing parameters, training parameters, and evaluation metrics.
- Generate logs for TensorBoard.
- Save the final cross-validation performance metrics to a CSV file.

### Stage 3: Inference and Evaluation

After training, you need to run inference on your test data and then evaluate the results. You can do this using individual scripts or the config-driven pipeline.

#### Running Inference

Use the inference script to generate predictions for a specific subject:

```bash
python src/scripts/inference.py --model-type lstm --model-path data/recordings/models/lstm_fold_1_subject_Subject01.keras --scaler-path data/recordings/models/scaler_lstm_fold_1_subject_Subject01.joblib --subject-id Subject01
```

This script will:
- Load the specified trained model and scaler
- Process the data for the specified subject
- Generate predictions
- Save the results to a CSV file in `data/recordings/predictions/`

#### Running Evaluation and Visualization

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
- `--plot-roi-contributions`: Plot the contribution of each ROI to the prediction accuracy
- `--visualize-multi-roi`: Generate visualizations of the Multi-ROI detection on sample frames
- `--all`: Run all visualization and reporting tasks

The script generates and saves various plots:
- Training history plots showing loss curves
- Prediction plots comparing the predicted values vs. the ground truth
- Model comparison reports with performance metrics
- ROI contribution plots showing the importance of each ROI for prediction
- Multi-ROI visualization on sample frames
- All plots can be annotated with model configuration details, metrics, and timestamps

All output plots are saved to `data/recordings/evaluation_plots/` by default, or to the specified output directory.

## Development Tools

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality and consistency. These hooks run automatically before each commit, checking for issues and fixing them when possible.

To set up pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Set up the hooks
python setup_hooks.py
```

The pre-commit hooks include:
- Code formatting (black, isort)
- Linting (flake8)
- Static type checking (mypy)
- Security checks (bandit)
- Check for large files or secrets
- Basic test execution

For more information about the pre-commit hooks, see [docs/pre_commit_hooks.md](docs/pre_commit_hooks.md).

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
