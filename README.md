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

ml_models/: Defines the neural network architectures (LSTM, AE, VAE).

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
Once you have collected data for all subjects, run the training script. This script will automatically perform
Leave-One-Subject-Out (LOSO) cross-validation.

python src/scripts/train_model.py

This script will:

Load and process the data for all subjects found in data/recordings/.

Iterate through each subject, training a model on all other subjects and testing on the held-out subject.

Save the trained model (.keras) and the corresponding data scaler (.joblib) for each fold into data/recordings/models/.

Generate logs for TensorBoard in data/recordings/models/logs/.

Save the final cross-validation performance metrics to data/recordings/cross_validation_results.csv.

Stage 3: Evaluation and Visualization
After training, run the evaluation script to generate visualizations for a specific subject's performance.

python src/scripts/evaluate_model.py

The script is pre-configured to evaluate the results for Subject01 using the model from Fold 1. You can change the
TEST_SUBJECT_ID inside the script to evaluate other subjects.

It loads the predictions generated during the inference step (src/scripts/inference.py must be run first or its logic
integrated here).

It generates and saves two plots:

A time-series plot comparing the predicted GSR vs. the ground truth.

A Bland-Altman plot to assess the agreement between the two measurements.

All output plots are saved to data/recordings/evaluation_plots/.