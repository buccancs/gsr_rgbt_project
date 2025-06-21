# MMRPhys Repository Overview

## Purpose and Overview

MMRPhys (Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization) is a deep learning framework designed for remote physiological sensing. It focuses on extracting physiological signals such as heart rate and respiration from RGB and thermal videos without requiring physical contact with the subject.

The repository implements a novel approach using multidimensional attention mechanisms and target signal constrained factorization to improve the accuracy and robustness of remote physiological measurements. This is particularly valuable for applications in healthcare monitoring, affective computing, and human-computer interaction where non-contact physiological monitoring is desired.

## Key Components and Functionality

### Core Components

1. **MMRPhys Models**: 
   - **MMRPhysLEF**: Light Efficient Factorization variant
   - **MMRPhysMEF**: Medium Efficient Factorization variant
   - **MMRPhysSEF**: Standard Efficient Factorization variant
   - **TSFM**: Target Signal Constrained Factorization Module

2. **Training Framework**:
   - **MMRPhysTrainer**: Specialized trainer for the MMRPhys models

3. **Dataset Support**:
   - Support for multiple datasets including iBVP, PURE, SCAMPS, UBFC-rPPG, and BP4D+
   - Standardized data loading and preprocessing

4. **Evaluation Tools**:
   - Metrics for physiological signal quality assessment
   - Visualization tools for results analysis

### Main Functionality

- **Remote Photoplethysmography (rPPG)**: Extracts pulse signals from facial videos
- **Multi-modal Sensing**: Works with both RGB and thermal video inputs
- **Cross-dataset Evaluation**: Supports training on one dataset and testing on another
- **Pre-trained Models**: Provides pre-trained models for immediate use

## Integration with the Main Project

MMRPhys integrates with the GSR-RGBT project as a submodule, providing advanced signal processing capabilities for extracting physiological information from the video data captured by the FactorizePhys or RGBTPhys_CPP components. The main project uses this library to:

1. Process the synchronized RGB and thermal videos to extract physiological signals
2. Provide non-contact alternatives to traditional contact-based physiological measurements
3. Enable advanced analysis of physiological states from video data

### Python Integration via mmrphys_processor.py

The main project includes a Python wrapper class `MMRPhysProcessor` in `src/processing/mmrphys_processor.py` that provides a convenient interface to the MMRPhys models. This class:

1. Handles loading and initialization of MMRPhys models
2. Provides methods for processing video files and frame sequences
3. Extracts physiological signals such as heart rate from video data
4. Includes visualization tools for the extracted signals

Example usage of the Python wrapper:

```python
from src.processing.mmrphys_processor import MMRPhysProcessor

# Initialize the processor with a specific model type
processor = MMRPhysProcessor(
    model_type='MMRPhysLEF',
    weights_path='third_party/MMRPhys/final_model_release/MMRPhysLEF_weights.pth',
    device='cuda'  # Use 'cpu' if no GPU is available
)

# Process a video file
results = processor.process_video(
    video_path='data/recordings/Subject_01/rgb_video.avi',
    output_dir='data/results/Subject_01',
    frame_limit=None  # Process the entire video
)

# Extract heart rate from the processed signal
heart_rate = processor.extract_heart_rate(
    pulse_signal=results['pulse_signal'],
    fps=30,
    window_size=300
)

print(f"Average heart rate: {heart_rate['average_hr']} BPM")
print(f"Heart rate variability: {heart_rate['hrv']} ms")
```

For real-time processing of frame sequences:

```python
# Initialize the processor
processor = MMRPhysProcessor(model_type='MMRPhysLEF')

# Process a sequence of frames (e.g., from a live capture)
frames = []  # List of frames from your capture source
results = processor.process_frame_sequence(
    frames=frames,
    frame_type='rgb'  # or 'thermal' for thermal frames
)

# The results contain the extracted physiological signals
pulse_signal = results['pulse_signal']
```

### Installation and Setup

To use MMRPhys in your project:

1. **Add as a Submodule**:
   ```bash
   git submodule add https://github.com/your-organization/MMRPhys.git third_party/MMRPhys
   git submodule update --init --recursive
   ```

2. **Install Dependencies**:
   ```bash
   cd third_party/MMRPhys
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:
   ```bash
   mkdir -p final_model_release
   # Download pre-trained models from the repository release page
   # or use the provided download script if available
   python download_models.py
   ```

4. **Configure Environment**:
   - Set up environment variables if needed:
     ```bash
     export MMRPHYS_MODEL_PATH=/path/to/models
     ```
   - For GPU acceleration, ensure you have the appropriate CUDA version installed

## Usage Examples

### Using Pre-trained Models

```python
# Example of using a pre-trained model on UBFC-rPPG dataset
python main.py --config_file configs/infer_configs/BVP/Cross/RGB/iBVP_UBFC-rPPG_MMRPhys_SFSAM_Label.yaml
```

### Training a New Model

1. **Prepare the Dataset**: Organize the dataset according to the required structure.

2. **Configure Training Parameters**: Modify the appropriate configuration file in `configs/train_configs/`.

3. **Run Training**:
   ```python
   python main.py --config_file configs/train_configs/BVP/Cross/RGB/iBVP_PURE_MMRPhys_SFSAM_Label.yaml
   ```

4. **Evaluate the Model**:
   The training process will automatically evaluate the model on the test set and report metrics.

### Configuration Options

The configuration files in MMRPhys allow customization of various aspects:

- **Dataset**: Specify training, validation, and test datasets
- **Model Architecture**: Select the model variant and parameters
- **Training Parameters**: Learning rate, batch size, number of epochs, etc.
- **Preprocessing**: Enable/disable preprocessing steps
- **Evaluation Metrics**: Configure which metrics to use for evaluation

## Target Signal Constrained Factorization (TSFM)

The core innovation in MMRPhys is the Target Signal Constrained Factorization Module (TSFM), which addresses the challenge of separating physiological signals from noise and motion artifacts in video data.

### TSFM Principles

TSFM is based on the following key principles:

1. **Signal Factorization**: Decomposes the input signal into multiple components, separating the physiological signal from noise and artifacts.
2. **Target Signal Constraint**: Uses a target signal (e.g., ground truth or estimated physiological signal) to guide the factorization process.
3. **Adaptive Weighting**: Dynamically adjusts the importance of different signal components based on their similarity to the target signal.
4. **Multi-dimensional Attention**: Applies attention mechanisms across spatial, temporal, and feature dimensions to focus on the most relevant information.

### TSFM Architecture

The TSFM architecture consists of several key components:

1. **Feature Extraction**: Convolutional layers extract features from the input video frames.
2. **Multidimensional Attention**: Self-attention mechanisms applied across spatial, temporal, and feature dimensions.
3. **Factorization Module**: Decomposes the features into multiple components using matrix factorization techniques.
4. **Target Signal Guidance**: Uses the target signal to guide the factorization process.
5. **Signal Reconstruction**: Reconstructs the physiological signal from the factorized components.

### Mathematical Formulation

The TSFM factorization can be expressed mathematically as:

```
X ≈ WH
```

Where:
- X is the input feature matrix
- W is the basis matrix (representing signal components)
- H is the coefficient matrix (representing the strength of each component)

The factorization is constrained by the target signal Y:

```
minimize ||X - WH||² + λ||Y - f(H)||²
```

Where:
- f(H) is a function that maps the coefficient matrix to the target signal domain
- λ is a regularization parameter that controls the influence of the target signal constraint

## Supported Algorithms

MMRPhys supports several algorithms for remote physiological sensing:

1. **MMRPhys with TSFM**: The proposed method with Target Signal Constrained Factorization
2. **FactorizePhys with FSAM**: From Joshi et al., 2024
3. **PhysNet**: Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
4. **EfficientPhys**: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement
5. **PhysFormer**: Facial Video-based Physiological Measurement with Temporal Difference Transformer

## Future Improvements

Potential improvements for the MMRPhys framework could include:

1. Integration with real-time capture systems for live physiological monitoring
2. Support for additional physiological signals beyond heart rate
3. Optimization for edge devices and mobile platforms
4. Enhanced robustness to challenging conditions (motion, lighting variations)
5. Extension to group-based monitoring for multiple subjects simultaneously
