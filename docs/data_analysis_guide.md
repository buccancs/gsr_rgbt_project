# GSR-RGBT Data Analysis Guide

This guide provides detailed instructions on how to use the data analysis module and machine learning pipeline to analyze GSR-RGBT data and train predictive models.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Analysis Module](#data-analysis-module)
   - [Feature Extraction](#feature-extraction)
   - [Data Visualization](#data-visualization)
   - [Session Analysis](#session-analysis)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Preparing Data for Training](#preparing-data-for-training)
   - [Building and Training Models](#building-and-training-models)
   - [Evaluating Model Performance](#evaluating-model-performance)
4. [End-to-End Workflow](#end-to-end-workflow)
   - [Using the analyze_and_train.py Script](#using-the-analyze_and_trainpy-script)
   - [Custom Analysis Workflows](#custom-analysis-workflows)
5. [Advanced Topics](#advanced-topics)
   - [Feature Selection](#feature-selection)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Cross-Validation Strategies](#cross-validation-strategies)
6. [Troubleshooting](#troubleshooting)

## Introduction

The GSR-RGBT system collects multimodal data including:

- Galvanic Skin Response (GSR) from a Shimmer3 GSR+ unit
- RGB video from a Logitech Brio camera
- Thermal video from a Topdon/InfiRay P2 Pro thermal dongle

This guide explains how to use the provided tools to:

1. Extract meaningful features from the raw data
2. Visualize the data and extracted features
3. Train machine learning models to predict physiological responses
4. Evaluate model performance

## Data Analysis Module

The data analysis module (`src/processing/data_analysis.py`) provides tools for extracting features from GSR and PPG signals, visualizing the data, and analyzing recording sessions.

### Feature Extraction

#### GSR Feature Extraction

The `GSRFeatureExtractor` class extracts various features from GSR signals:

```python
from src.processing.data_analysis import GSRFeatureExtractor

# Initialize the extractor
gsr_extractor = GSRFeatureExtractor(sampling_rate=32)

# Extract features from a GSR signal
features = gsr_extractor.extract_all_features(gsr_signal)
```

The extracted features include:

1. **Statistical Features**:
   - Mean, standard deviation, min, max, range
   - Median, percentiles, interquartile range
   - Skewness, kurtosis
   - Signal derivatives

2. **Frequency Domain Features**:
   - Total power
   - Power in specific frequency bands (VLF, LF, HF)
   - Relative powers and ratios
   - Peak frequency

3. **Non-linear Features**:
   - Sample entropy
   - Detrended Fluctuation Analysis (DFA) alpha
   - Poincaré plot features (SD1, SD2)

#### PPG Feature Extraction

The `PPGFeatureExtractor` class extracts features from PPG signals:

```python
from src.processing.data_analysis import PPGFeatureExtractor

# Initialize the extractor
ppg_extractor = PPGFeatureExtractor(sampling_rate=32)

# Extract features from a PPG signal
features = ppg_extractor.extract_all_features(ppg_signal)
```

The extracted features include:

1. **Heart Rate Features**:
   - Mean heart rate
   - Heart rate variability

2. **Heart Rate Variability (HRV) Features**:
   - RMSSD (Root Mean Square of Successive Differences)
   - SDNN (Standard Deviation of NN intervals)
   - pNN50 (Percentage of NN intervals differing by more than 50ms)

3. **Pulse Wave Features**:
   - Pulse amplitude
   - Pulse width
   - Pulse rise and fall times

### Data Visualization

The `DataVisualizer` class provides tools for visualizing the data:

```python
from src.processing.data_analysis import DataVisualizer

# Initialize the visualizer with an output directory
visualizer = DataVisualizer(output_dir=Path("output/visualizations"))

# Plot time series data
visualizer.plot_time_series(
    df, columns=["GSR_Phasic", "GSR_Tonic"],
    title="GSR Signal Components",
    save_as="gsr_components.png"
)

# Plot correlation matrix
visualizer.plot_correlation_matrix(
    df, title="Signal Correlations",
    save_as="correlation_matrix.png"
)

# Plot feature distributions
visualizer.plot_feature_distributions(
    features_df, columns=feature_cols,
    title="Feature Distributions",
    save_as="feature_distributions.png"
)

# Plot PCA visualization
visualizer.plot_pca_visualization(
    features_df, columns=feature_cols,
    n_components=2,
    title="PCA of Features",
    save_as="pca_visualization.png"
)
```

### Session Analysis

The `DataAnalyzer` class provides tools for analyzing recording sessions:

```python
from src.processing.data_analysis import DataAnalyzer

# Initialize the analyzer with an output directory
analyzer = DataAnalyzer(output_dir=Path("output/analysis"))

# Analyze a single session
results = analyzer.analyze_session(
    session_path=Path("data/recordings/Subject_001_20250101_000000"),
    gsr_sampling_rate=32,
    save_visualizations=True
)

# Analyze multiple sessions
session_paths = [
    Path("data/recordings/Subject_001_20250101_000000"),
    Path("data/recordings/Subject_002_20250102_000000")
]
results = analyzer.analyze_multiple_sessions(
    session_paths=session_paths,
    gsr_sampling_rate=32,
    save_visualizations=True
)

# Access the combined features
features_df = results['combined_features']
```

## Machine Learning Pipeline

The machine learning pipeline provides tools for training models on the extracted features.

### Preparing Data for Training

Before training a model, you need to prepare the data:

```python
from src.scripts.analyze_and_train import prepare_features_for_training

# Prepare features for training
X_train, X_test, y_train, y_test = prepare_features_for_training(
    features_df=features_df,
    target_feature="GSR_mean",
    test_size=0.2,
    random_state=42
)
```

For sequence models (LSTM, CNN_LSTM, Transformer), you need to reshape the data:

```python
from src.scripts.analyze_and_train import reshape_features_for_model

# Reshape features for a sequence model
X_train_reshaped, X_test_reshaped = reshape_features_for_model(
    X_train, X_test, model_type="lstm"
)
```

### Building and Training Models

You can build and train models using the `build_model_from_config` function:

```python
from src.ml_pipeline.training.train_model import build_model_from_config

# Build a model
model = build_model_from_config(
    input_shape=X_train_reshaped.shape[1:],
    model_type="lstm",
    config_path="configs/lstm_config.yaml"
)

# Train the model
model.fit(X_train_reshaped, y_train, validation_split=0.2)
```

### Evaluating Model Performance

After training, you can evaluate the model's performance:

```python
from src.scripts.analyze_and_train import evaluate_model, plot_predictions

# Evaluate the model
metrics = evaluate_model(model, X_test_reshaped, y_test)

# Plot predictions
y_pred = model.predict(X_test_reshaped)
plot_predictions(
    y_test, y_pred,
    target_feature="GSR_mean",
    output_dir=Path("output/results")
)
```

## End-to-End Workflow

### Using the analyze_and_train.py Script

The `analyze_and_train.py` script provides an end-to-end workflow for analyzing data and training models:

```bash
python src/scripts/analyze_and_train.py \
    --data-dir data/recordings \
    --output-dir output/results \
    --gsr-sampling-rate 32 \
    --save-visualizations \
    --model-type lstm \
    --config-path configs/lstm_config.yaml \
    --target-feature GSR_mean \
    --test-size 0.2 \
    --random-state 42
```

Command-line options:

- `--data-dir`: Directory containing session recordings
- `--output-dir`: Directory to save analysis results and trained models
- `--gsr-sampling-rate`: Sampling rate of the GSR signal in Hz
- `--save-visualizations`: Save visualizations of the data and features
- `--model-type`: Type of model to train (lstm, cnn, cnn_lstm, transformer, resnet)
- `--config-path`: Path to model configuration YAML file
- `--target-feature`: Feature to use as the target for model training
- `--test-size`: Fraction of data to use for testing
- `--random-state`: Random state for reproducibility

### Custom Analysis Workflows

You can also create custom analysis workflows by combining the various components:

```python
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.processing.data_analysis import DataAnalyzer
from src.scripts.analyze_and_train import prepare_features_for_training, reshape_features_for_model
from src.ml_pipeline.training.train_model import build_model_from_config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up paths
data_dir = Path("data/recordings")
output_dir = Path("output/custom_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Find session directories
session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Subject_")]

# Create analyzer
analyzer = DataAnalyzer(output_dir=output_dir)

# Analyze sessions
results = analyzer.analyze_multiple_sessions(
    session_paths=session_dirs,
    gsr_sampling_rate=32,
    save_visualizations=True
)

# Get combined features
features_df = results['combined_features']

# Save features to CSV
features_df.to_csv(output_dir / "features.csv", index=False)

# Prepare features for training
X_train, X_test, y_train, y_test = prepare_features_for_training(
    features_df=features_df,
    target_feature="GSR_mean",
    test_size=0.2,
    random_state=42
)

# Reshape features for LSTM model
X_train_reshaped, X_test_reshaped = reshape_features_for_model(
    X_train, X_test, model_type="lstm"
)

# Build and train model
model = build_model_from_config(
    input_shape=X_train_reshaped.shape[1:],
    model_type="lstm"
)

# Train the model
history = model.fit(X_train_reshaped, y_train, validation_split=0.2)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")

# Save the model
model.save(str(output_dir / "lstm_model.pt"))
```

## Advanced Topics

### Feature Selection

Feature selection can improve model performance by removing irrelevant or redundant features:

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select the k best features based on F-statistic
selector = SelectKBest(f_regression, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the selected feature indices
selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]
print(f"Selected features: {selected_features}")
```

### Hyperparameter Tuning

You can use grid search or random search to find the best hyperparameters for your model:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Create the model
rf = RandomForestRegressor(random_state=42)

# Create the grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Get the best model
best_model = grid_search.best_estimator_
```

### Cross-Validation Strategies

For physiological data, it's often important to use subject-wise cross-validation:

```python
from sklearn.model_selection import LeaveOneGroupOut

# Create a cross-validation iterator
cv = LeaveOneGroupOut()

# Get subject IDs for each sample
subject_ids = features_df['subject_id'].values

# Perform cross-validation
for train_idx, test_idx in cv.split(X, y, groups=subject_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train and evaluate model
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test score: {score:.4f}")
```

## Troubleshooting

### Common Issues

1. **Missing Data**:
   - Ensure that all required data files are present in the session directory.
   - Check that the file formats are correct.

2. **Feature Extraction Errors**:
   - Verify that the GSR and PPG signals have the expected sampling rates.
   - Check for NaN or infinite values in the signals.

3. **Model Training Issues**:
   - Ensure that the input shape matches the model's expected input shape.
   - Check for NaN or infinite values in the features.
   - Try scaling the features before training.

4. **Visualization Errors**:
   - Ensure that the output directory exists and is writable.
   - Check that the column names exist in the DataFrame.

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the unit tests for examples of how to use the various components.
2. Review the docstrings in the source code for detailed information.
3. Consult the project's issue tracker for known issues and solutions.