# GSR-RGBT Project Architecture

This document outlines the architecture of the GSR-RGBT project, including the current structure and a proposed improved structure for future development.

## Current Structure

The current project structure is organized as follows:

```
src/
├── capture/              # Capture threads for video and GSR data
├── evaluation/           # Visualization and evaluation utilities
├── gui/                  # GUI components for the data collection application
├── ml_models/            # Machine learning model definitions
├── processing/           # Data processing and feature engineering
├── scripts/              # Scripts for training, inference, and evaluation
├── utils/                # Utility classes for data logging and timestamps
├── config.py             # Configuration settings
└── main.py               # Main entry point for the data collection application
```

## Proposed Improved Structure

To improve code organization and maintainability, the following structure is proposed for future development:

```
src/
├── data_collection/      # Data collection application
│   ├── capture/          # Capture threads for video and GSR data
│   ├── gui/              # GUI components
│   └── utils/            # Utilities specific to data collection
├── ml_pipeline/          # Machine learning pipeline
│   ├── preprocessing/    # Data preprocessing
│   ├── feature_engineering/ # Feature extraction and engineering
│   ├── training/         # Model training and cross-validation
│   └── evaluation/       # Model evaluation and visualization
├── system/               # System-level utilities
│   ├── validation/       # System validation and testing
│   ├── data_generation/  # Mock data generation
│   └── utils/            # General system utilities
├── config.py             # Configuration settings
├── main.py               # Main entry point for the data collection application
└── train.py              # Main entry point for the machine learning pipeline
```

## Module Boundaries

### Data Collection

The `data_collection` module is responsible for capturing data from the RGB camera, thermal camera, and GSR sensor, and providing a GUI for the user to control the data collection process. It includes:

- **capture**: Threads for capturing data from the RGB camera, thermal camera, and GSR sensor
- **gui**: PyQt5-based GUI components for the data collection application
- **utils**: Utilities specific to data collection, such as data logging and timestamp synchronization

### Machine Learning Pipeline

The `ml_pipeline` module is responsible for processing the collected data, training machine learning models, and evaluating their performance. It includes:

- **preprocessing**: Data loading, cleaning, and preprocessing
- **feature_engineering**: Feature extraction and engineering
- **training**: Model training, cross-validation, and hyperparameter tuning
- **evaluation**: Model evaluation, visualization, and comparison

### System

The `system` module is responsible for system-level utilities, such as validation, testing, and data generation. It includes:

- **validation**: System validation and testing scripts
- **data_generation**: Mock data generation for testing and development
- **utils**: General system utilities, such as file handling and logging

## Naming Conventions

To ensure consistency throughout the codebase, the following naming conventions are proposed:

- **Files**: Use snake_case for file names (e.g., `video_capture.py`, `data_loader.py`)
- **Classes**: Use PascalCase for class names (e.g., `VideoCaptureThread`, `DataLoader`)
- **Functions and Methods**: Use snake_case for function and method names (e.g., `process_frame`, `load_data`)
- **Variables**: Use snake_case for variable names (e.g., `frame_count`, `gsr_value`)
- **Constants**: Use UPPER_CASE for constants (e.g., `FPS`, `OUTPUT_DIR`)

## Documentation

All modules, classes, functions, and methods should be documented using docstrings following the Google style guide. Example:

```python
def process_frame(frame, roi):
    """
    Process a video frame to extract features from a region of interest.
    
    Args:
        frame (np.ndarray): The input video frame.
        roi (tuple): The region of interest as (x, y, width, height).
        
    Returns:
        np.ndarray: The extracted features.
    """
    # Implementation
```

## Implementation Plan

The proposed improved structure can be implemented gradually, following these steps:

1. Create the new directory structure
2. Move files to their appropriate locations, updating imports as needed
3. Update documentation to reflect the new structure
4. Run tests to ensure everything still works correctly
5. Update the build system (Makefile) to reflect the new structure

This should be done in a separate branch and thoroughly tested before merging into the main branch.