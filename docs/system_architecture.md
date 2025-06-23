# GSR-RGBT Project System Architecture

## Introduction

This comprehensive system architecture document explains how all the pieces of the GSR-RGBT project fit together to form a complete physiological sensing system. It provides crucial context about the system's holistic design, repository integration, and the purpose and function of each sub-repository.

The GSR-RGBT project combines multiple repositories and technologies to create a comprehensive platform for physiological sensing using RGB and thermal imaging, enhanced with contact-based ground truth measurements and advanced signal processing techniques.

---

# System Overview

## Architecture Overview

The GSR-RGBT project implements a multi-layered architecture that integrates data capture, preprocessing, signal extraction, and analysis components. The system is designed to provide both contact-based and non-contact physiological measurements, enabling comprehensive validation and research in remote physiological sensing.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GSR-RGBT System Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  Data Acquisition → Preprocessing → Signal Extraction → Analysis │
│  [FactorizePhys]    [TC001_SAMCL]    [MMRPhys]         [Analysis]│
│  [RGBTPhys_CPP]                                        [Tools]   │
│  [Shimmer Sensors]                                               │
└─────────────────────────────────────────────────────────────────┘
```

### System Components

The integrated system consists of the following major components:

1. **Data Acquisition Layer**:
   - **FactorizePhys**: Advanced synchronized capture with factorization capabilities
   - **RGBTPhys_CPP**: Original synchronized data stream capture
   - **TC001_SAMCL**: Specialized thermal camera capture and segmentation
   - **Shimmer Integration**: Contact-based physiological sensor data

2. **Processing Layer**:
   - **MMRPhys**: Deep learning framework for remote physiological sensing
   - **Signal Processing Tools**: Filtering, analysis, and validation utilities

3. **Analysis Layer**:
   - **neurokit2**: Neurophysiological signal processing toolbox
   - **physiokit**: Physiological data analysis tools
   - **pyshimmer**: Shimmer sensor interface and analysis

4. **Integration Layer**:
   - **Python Wrappers**: Unified interfaces for all components
   - **Configuration Management**: Centralized configuration system
   - **Data Management**: Structured data organization and storage

## Data Flow Architecture

### Complete Physiological Monitoring Pipeline

1. **Data Acquisition**:
   - **FactorizePhys** and **RGBTPhys_CPP** handle synchronized capture of RGB video, thermal video, and contact-based physiological signals
   - **TC001_SAMCL** provides specialized thermal imaging with advanced segmentation
   - All data streams are properly time-aligned with high-precision timestamps

2. **Preprocessing**:
   - **TC001_SAMCL** provides thermal image processing and segmentation
   - Regions of interest are identified in thermal imagery
   - Data is organized by participant ID and experimental condition

3. **Signal Extraction**:
   - **MMRPhys** applies deep learning techniques to extract physiological signals from video data
   - Works with both RGB and thermal video inputs
   - Provides non-contact alternatives to traditional physiological measurements

4. **Analysis and Validation**:
   - **neurokit2** and **physiokit** provide comprehensive signal analysis tools
   - **pyshimmer** enables comparison with ground truth data from contact sensors
   - Cross-validation between contact and non-contact measurements

### Data Formats and Compatibility

The repositories are designed with compatible data formats:

- **FactorizePhys** and **RGBTPhys_CPP** save data in formats directly usable by **MMRPhys**
- **TC001_SAMCL** produces segmentation masks applicable to thermal data before **MMRPhys** processing
- All extracted signals can be analyzed using **neurokit2** and **physiokit**
- Unified timestamp format ensures temporal alignment across all data streams

---

# Repository Integration

## Integration Architecture

The integration of multiple repositories creates a comprehensive pipeline for physiological sensing, where each component contributes specialized capabilities while maintaining compatibility with the overall system.

### Repository Relationships

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FactorizePhys │    │   RGBTPhys_CPP  │    │   TC001_SAMCL   │
│                 │    │                 │    │                 │
│ • Advanced      │    │ • Original      │    │ • Thermal       │
│   Factorization │    │   Synchronized  │    │   Segmentation  │
│ • Multi-modal   │    │   Capture       │    │ • TOPDON TC001  │
│   Capture       │    │ • Proven        │    │ • SAM + CL      │
│                 │    │   Reliability   │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │        MMRPhys            │
                    │                           │
                    │ • Deep Learning Framework │
                    │ • Remote Physiological    │
                    │   Sensing                 │
                    │ • TSFM Algorithm          │
                    │ • Multi-modal Processing  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     Analysis Tools        │
                    │                           │
                    │ • neurokit2              │
                    │ • physiokit              │
                    │ • pyshimmer              │
                    └───────────────────────────┘
```

### Use Cases and Workflows

#### Complete Physiological Monitoring Workflow

1. Use **FactorizePhys** to capture synchronized RGB video, thermal video, and contact-based physiological data
2. Apply **TC001_SAMCL** to segment regions of interest in the thermal imagery
3. Process the video data with **MMRPhys** to extract non-contact physiological signals
4. Analyze both contact-based and non-contact signals using **neurokit2** and **physiokit**
5. Validate the results by comparing the extracted signals with the ground truth from contact sensors

#### Remote Physiological Sensing Only

1. Use **RGBTPhys_CPP** or **FactorizePhys** to capture RGB and thermal video
2. Apply **MMRPhys** directly to the video data to extract physiological signals
3. Analyze the extracted signals using **neurokit2** and **physiokit**

#### Thermal-Focused Analysis

1. Use **TC001_SAMCL** for specialized thermal imaging with the TOPDON TC001 camera
2. Apply the segmentation capabilities to identify specific regions of interest
3. Extract temporal signals from these regions
4. Process the signals with **MMRPhys** or analyze them directly with **neurokit2**

### Configuration and Setup

Each repository has its own configuration system that integrates with the overall project:

- **FactorizePhys**: Text-based configuration files for capture parameters
- **MMRPhys**: YAML configuration files for model and training parameters
- **TC001_SAMCL**: Command-line arguments and configuration files for camera and segmentation settings
- **Main Project**: Unified configuration management that coordinates all components

When integrating these repositories, configurations are designed to be compatible, particularly for aspects like frame rates, resolutions, and file paths.

---

# FactorizePhys Repository

## Purpose and Overview

FactorizePhys is a C++ library designed for synchronized capture of RGB video, thermal video, and physiological data. It is an extension of the RGBTPhys_CPP library, with specialized functionality for factorizing (separating) physiological signals from video data.

### Relationship with RGBTPhys_CPP

FactorizePhys builds upon the foundation of RGBTPhys_CPP, extending it with the following enhancements:

1. **Advanced Factorization Algorithms**: Implements specialized algorithms for separating physiological signals from video data
2. **Improved Synchronization**: Enhanced timestamp precision and synchronization mechanisms
3. **Extended Sensor Support**: Additional support for a wider range of physiological sensors
4. **Optimized Performance**: Performance improvements for real-time processing of high-resolution video streams

While RGBTPhys_CPP focuses primarily on synchronized data capture, FactorizePhys adds the capability to process and analyze the captured data in real-time, extracting meaningful physiological signals.

## Key Components and Functionality

### Core Components

1. **ConfigReader**: Handles reading and parsing configuration files that specify capture parameters.
2. **CaptureRGB**: Manages RGB camera capture, including initialization, frame acquisition, and cleanup.
3. **CaptureThermal**: Manages thermal camera capture, specifically designed for FLIR cameras.
4. **SerialCom**: Handles serial communication for physiological sensors, reading data from devices like Arduino or Shimmer.
5. **Utils**: Provides utility functions for file operations, directory creation, and other common tasks.

### Main Functionality

- **Synchronized Multi-modal Data Capture**: Captures RGB video, thermal video, and physiological data with precise timing synchronization.
- **Configurable Capture Parameters**: Allows customization of frame rates, resolutions, acquisition durations, and other parameters through configuration files.
- **Experimental Condition Support**: Organizes data by participant ID and experimental condition for structured data collection.
- **Cross-platform Support**: Works on both Windows and Linux systems.
- **Real-time Factorization**: Implements algorithms to separate physiological signals from video data in real-time.

### Factorization Algorithms

FactorizePhys implements several advanced algorithms for extracting physiological signals from video data:

1. **Blind Source Separation (BSS)**: Separates mixed signals into their constituent source signals without prior knowledge of the mixing process.
   - **Independent Component Analysis (ICA)**: Assumes statistical independence between source signals.
   - **Principal Component Analysis (PCA)**: Identifies orthogonal components that explain the maximum variance in the data.

2. **Motion-Robust Factorization**: Algorithms designed to be resilient to subject movement.
   - **Motion Compensation**: Tracks and compensates for subject movement before signal extraction.
   - **Adaptive Region Selection**: Dynamically adjusts regions of interest based on motion detection.

3. **Multi-modal Fusion**: Combines information from RGB and thermal video streams for more robust signal extraction.
   - **Weighted Fusion**: Assigns weights to different modalities based on signal quality.
   - **Feature-level Fusion**: Combines features extracted from different modalities.

4. **Temporal Filtering**: Applies various filters to enhance physiological signals and reduce noise.
   - **Bandpass Filtering**: Isolates frequency bands associated with physiological processes.
   - **Wavelet Denoising**: Uses wavelet transforms to separate signal from noise.

## Integration with the Main Project

FactorizePhys integrates with the GSR-RGBT project as a submodule, providing the low-level data capture functionality needed for synchronized multi-modal data collection. The main project uses this library to:

1. Capture synchronized data during experiments
2. Ensure temporal alignment between different data streams
3. Organize the captured data in a structured format for subsequent analysis
4. Extract physiological signals from video data in real-time

### Python Integration

The main project includes a Python wrapper class `FactorizePhysCaptureThread` in `src/capture/factorize_phys_capture.py` that provides a convenient interface to the FactorizePhys C++ library. This class:

1. Extends the `BaseCaptureThread` class to maintain consistency with the project's architecture
2. Provides both real capture and simulation modes for development and testing
3. Emits PyQt signals with captured frames and data
4. Handles configuration and data organization

Example usage:

```python
from src.capture.factorize_phys_capture import FactorizePhysCaptureThread

# Create a capture thread instance
capture_thread = FactorizePhysCaptureThread(
    config_file="third_party/FactorizePhys/default_config",
    base_save_path="data/recordings",
    participant_id="Subject_01",
    simulation_mode=False  # Set to True for testing without hardware
)

# Connect signals to handlers
capture_thread.rgb_frame_captured.connect(handle_rgb_frame)
capture_thread.thermal_frame_captured.connect(handle_thermal_frame)
capture_thread.phys_data_captured.connect(handle_phys_data)

# Start capturing
capture_thread.start()
```

### Configuration Options

The library supports various configuration options, including:

- `thread_sleep_interval_acquisition`: Sleep interval between acquisition cycles (microseconds)
- `acquisition_duration`: Duration of data capture (seconds)
- `exp_condition`: Experimental condition identifier
- `thermal_fps`, `rgb_fps`: Frame rates for thermal and RGB cameras
- `thermal_im_width`, `thermal_im_height`, `rgb_im_width`, `rgb_im_height`: Image dimensions
- `capture_phys`: Enable/disable physiological data capture
- `com_port`, `baud_rate`: Serial port settings for physiological sensors
- `phys_channels`: Channels to capture from physiological sensors

---

# MMRPhys Repository

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

## Integration with the Main Project

MMRPhys integrates with the GSR-RGBT project as a submodule, providing advanced signal processing capabilities for extracting physiological information from the video data captured by the FactorizePhys or RGBTPhys_CPP components.

### Python Integration

The main project includes a Python wrapper class `MMRPhysProcessor` in `src/processing/mmrphys_processor.py` that provides a convenient interface to the MMRPhys models. This class:

1. Handles loading and initialization of MMRPhys models
2. Provides methods for processing video files and frame sequences
3. Extracts physiological signals such as heart rate from video data
4. Includes visualization tools for the extracted signals

Example usage:

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

### Supported Algorithms

MMRPhys supports several algorithms for remote physiological sensing:

1. **MMRPhys with TSFM**: The proposed method with Target Signal Constrained Factorization
2. **FactorizePhys with FSAM**: From Joshi et al., 2024
3. **PhysNet**: Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
4. **EfficientPhys**: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement
5. **PhysFormer**: Facial Video-based Physiological Measurement with Temporal Difference Transformer

### Configuration Options

The configuration files in MMRPhys allow customization of various aspects:

- **Dataset**: Specify training, validation, and test datasets
- **Model Architecture**: Select the model variant and parameters
- **Training Parameters**: Learning rate, batch size, number of epochs, etc.
- **Preprocessing**: Enable/disable preprocessing steps
- **Evaluation Metrics**: Configure which metrics to use for evaluation

---

# TC001_SAMCL Repository

## Purpose and Overview

TC001_SAMCL is a specialized repository for live capture and segmentation using the TOPDON TC001 thermal camera. It combines thermal imaging with advanced segmentation techniques, likely based on the Segment Anything Model (SAM) with contrastive learning (CL) approaches.

The repository provides tools for real-time thermal image acquisition, processing, and segmentation, which can be valuable for applications requiring precise thermal region identification such as medical imaging, building inspection, or physiological monitoring.

## Key Components and Functionality

### Core Components

1. **Thermal Camera Interface**: 
   - Tools for connecting to and capturing frames from the TOPDON TC001 thermal camera
   - Real-time frame acquisition and processing

2. **Segmentation Models**:
   - Implementation of thermal image segmentation using advanced deep learning techniques
   - ThermSeg class for thermal image segmentation inference

3. **Visualization Tools**:
   - Real-time visualization of thermal images and segmentation results
   - Animated plotting of thermal data and derived signals

4. **Signal Processing**:
   - Utilities for processing signals extracted from thermal regions of interest
   - Filtering and analysis of temporal thermal patterns

### Main Functionality

- **Live Thermal Capture**: Real-time acquisition of thermal imagery from the TC001 camera
- **Automated Segmentation**: Identification and segmentation of regions of interest in thermal images
- **Signal Extraction**: Derivation of temporal signals from segmented thermal regions
- **Real-time Visualization**: Dynamic display of thermal data, segmentation results, and extracted signals

## Segmentation Approach

The segmentation uses a variant of the Segment Anything Model (SAM) adapted for thermal imagery, enhanced with Self-supervised Adversarial Multimodal Contrastive Learning (SAMCL). This approach allows for:

- Robust segmentation despite the lower resolution of thermal imagery
- Adaptation to different thermal conditions and environments
- Identification of specific regions of interest based on thermal patterns

### Contrastive Learning Approach

TC001_SAMCL implements a specialized contrastive learning approach that enhances the segmentation capabilities for thermal imagery:

#### Principles of Contrastive Learning in TC001_SAMCL

1. **Self-supervised Learning**: The model learns useful representations from unlabeled thermal data by creating and solving pretext tasks.

2. **Contrastive Objective**: The model is trained to bring similar samples (positive pairs) closer in the embedding space while pushing dissimilar samples (negative pairs) apart.

3. **Multimodal Fusion**: Combines information from both thermal and RGB modalities when available, learning cross-modal correspondences.

4. **Adversarial Training**: Incorporates adversarial examples during training to improve robustness to variations in thermal imagery.

#### Implementation Details

The contrastive learning implementation in TC001_SAMCL includes:

1. **Data Augmentation Pipeline**:
   - Thermal-specific augmentations (temperature scaling, emissivity simulation)
   - Standard augmentations (rotation, flipping, cropping)
   - Adversarial perturbations

2. **Embedding Network**:
   - Specialized encoder for thermal imagery
   - Projection head that maps representations to the space where contrastive loss is applied

3. **Loss Function**:
   ```
   L_contrastive = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]
   ```
   Where:
   - z_i and z_j are embeddings of positive pairs
   - z_k are embeddings of negative examples
   - sim() is a similarity function (e.g., cosine similarity)
   - τ is a temperature parameter

4. **Memory Bank**: Maintains a bank of negative examples to increase the number of contrastive pairs without requiring large batch sizes.

## Integration with the Main Project

TC001_SAMCL integrates with the GSR-RGBT project as a submodule, providing specialized thermal imaging and segmentation capabilities. The main project uses this library to:

1. Capture thermal data from the TOPDON TC001 camera
2. Automatically segment regions of interest in the thermal imagery
3. Extract physiological signals from specific thermal regions
4. Provide real-time visualization of thermal data and derived signals

### Python Integration

The main project includes a Python wrapper class `TC001ThermalCaptureThread` in `src/capture/tc001_thermal_capture.py` that provides a convenient interface to the TC001_SAMCL library. This class:

1. Extends the `BaseCaptureThread` class to maintain consistency with the project's architecture
2. Handles initialization and configuration of the TOPDON TC001 thermal camera
3. Provides both real capture and simulation modes for development and testing
4. Integrates the segmentation capabilities from TC001_SAMCL
5. Emits PyQt signals with captured thermal frames and segmentation results

Example usage:

```python
from src.capture.tc001_thermal_capture import TC001ThermalCaptureThread

# Create a capture thread instance
thermal_thread = TC001ThermalCaptureThread(
    config_path="third_party/TC001_SAMCL/configs/default_config.yaml",
    device_id=0,  # Camera device ID
    simulation_mode=False  # Set to True for testing without hardware
)

# Connect signals to handlers
thermal_thread.frame_captured.connect(handle_thermal_frame)
thermal_thread.segmentation_result.connect(handle_segmentation)

# Start capturing
thermal_thread.start()
```

### Configuration Options

The TC001_SAMCL tools support various configuration options:

- **Device Selection**: Specify which camera device to use
- **Segmentation Parameters**: Adjust segmentation sensitivity and thresholds
- **Visualization Settings**: Configure display options for the thermal imagery and segmentation results
- **Signal Processing Parameters**: Customize filtering and analysis of extracted signals

### Signal Processing

Once regions are segmented, the system can extract temporal signals from these regions, such as:

- Temperature variations over time
- Thermal patterns that may correlate with physiological processes
- Motion patterns detected in thermal imagery

---

# Supporting Libraries and Tools

## neurokit2

A Python toolbox for neurophysiological signal processing that provides:

- Comprehensive signal processing functions for ECG, EDA, EMG, and other physiological signals
- Advanced analysis tools for heart rate variability, signal quality assessment, and feature extraction
- Integration with the extracted signals from MMRPhys and contact sensors
- Standardized preprocessing and analysis pipelines

## physiokit

Tools for physiological data analysis that complement neurokit2 with:

- Specialized algorithms for physiological signal processing
- Additional metrics and analysis functions
- Custom tools developed specifically for the GSR-RGBT project
- Integration utilities for multi-modal physiological data

## pyshimmer

Interface for Shimmer physiological sensors that provides:

- Direct communication with Shimmer devices
- Data acquisition and real-time streaming
- Sensor configuration and calibration tools
- Ground truth data collection for validation of remote sensing methods

---

# Future Integration Opportunities

The current system architecture provides a solid foundation for future enhancements and extensions:

## Real-time Integration

Connecting the data capture components directly to the processing and analysis components for real-time physiological monitoring:

1. **Streaming Pipeline**: Direct data flow from capture to processing without intermediate storage
2. **Real-time Feedback**: Immediate physiological state assessment and feedback
3. **Adaptive Capture**: Dynamic adjustment of capture parameters based on signal quality

## Multi-modal Fusion

Combining signals from different modalities for more robust physiological measurements:

1. **Sensor Fusion**: Combining RGB, thermal, and contact sensor data
2. **Cross-modal Validation**: Using multiple modalities to validate measurements
3. **Redundancy and Reliability**: Improved system reliability through multiple measurement approaches

## Extended Analysis

Applying advanced analysis techniques to the integrated system:

1. **Machine Learning Integration**: Advanced ML models for physiological state classification
2. **Longitudinal Analysis**: Long-term physiological monitoring and trend analysis
3. **Personalization**: Adaptive algorithms that learn individual physiological patterns

## Feedback Systems

Creating closed-loop systems that use extracted physiological signals:

1. **Biofeedback Applications**: Real-time feedback for stress management and relaxation
2. **Adaptive Interfaces**: User interfaces that adapt based on physiological state
3. **Health Monitoring**: Continuous health monitoring with alert systems

---

# System Configuration and Management

## Unified Configuration System

The GSR-RGBT project implements a unified configuration system that coordinates all components:

### Configuration Hierarchy

1. **Global Configuration**: System-wide settings and defaults
2. **Component Configuration**: Specific settings for each repository/component
3. **Experiment Configuration**: Settings specific to experimental conditions
4. **User Configuration**: User-specific preferences and overrides

### Configuration Files

- **Main Configuration**: `config/main_config.yaml`
- **FactorizePhys Configuration**: `third_party/FactorizePhys/configs/`
- **MMRPhys Configuration**: `third_party/MMRPhys/configs/`
- **TC001_SAMCL Configuration**: `third_party/TC001_SAMCL/configs/`

### Configuration Management

The system provides tools for:

- Configuration validation and error checking
- Dynamic configuration updates during runtime
- Configuration versioning and backup
- Template configurations for common use cases

## Data Management

### Data Organization

The system organizes data in a hierarchical structure:

```
data/
├── recordings/
│   ├── participant_id/
│   │   ├── condition/
│   │   │   ├── rgb_video/
│   │   │   ├── thermal_video/
│   │   │   ├── physiological_data/
│   │   │   └── metadata/
├── processed/
│   ├── participant_id/
│   │   ├── condition/
│   │   │   ├── extracted_signals/
│   │   │   ├── analysis_results/
│   │   │   └── visualizations/
└── models/
    ├── pretrained/
    ├── trained/
    └── checkpoints/
```

### Data Synchronization

The system ensures data synchronization through:

- High-precision timestamps across all data streams
- Synchronization markers (LED flashes, audio beeps)
- Post-processing alignment algorithms
- Quality assessment and validation tools

### Data Validation

Comprehensive data validation includes:

- Timestamp consistency checks
- Signal quality assessment
- Missing data detection and handling
- Metadata validation and completeness

---

# Performance and Scalability

## System Performance

The GSR-RGBT system is designed for high-performance operation:

### Real-time Processing

- Multi-threaded capture and processing
- Optimized algorithms for real-time signal extraction
- GPU acceleration for deep learning components
- Efficient memory management and data streaming

### Scalability Considerations

- Modular architecture allows for component scaling
- Support for distributed processing
- Cloud-based processing capabilities
- Batch processing for large datasets

## Resource Requirements

### Minimum System Requirements

- **CPU**: Intel i5 or AMD Ryzen 5 (8 cores recommended)
- **RAM**: 16GB (32GB recommended for real-time processing)
- **GPU**: NVIDIA GTX 1660 or better (for MMRPhys processing)
- **Storage**: 500GB SSD (high-speed storage for video capture)
- **Network**: Gigabit Ethernet (for FLIR thermal camera)

### Recommended System Configuration

- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 32GB or higher
- **GPU**: NVIDIA RTX 3070 or better
- **Storage**: 1TB+ NVMe SSD
- **Network**: Dedicated Gigabit Ethernet for thermal camera

---

# Conclusion

The GSR-RGBT project's system architecture represents a comprehensive approach to physiological sensing that combines the strengths of multiple technologies and methodologies. The integration of FactorizePhys, MMRPhys, TC001_SAMCL, and supporting libraries creates a powerful platform for research and applications in remote physiological monitoring.

The modular design ensures that each component can be developed, tested, and improved independently while maintaining compatibility with the overall system. This architecture provides a solid foundation for future enhancements and extensions, enabling the system to evolve with advancing technology and research needs.

The system's ability to provide both contact-based and non-contact physiological measurements, combined with advanced signal processing and machine learning capabilities, makes it a valuable tool for researchers and practitioners in healthcare monitoring, affective computing, human-computer interaction, and biometric authentication.

For detailed usage instructions and tutorials, please refer to the [Integrated System Tutorial](integrated_system_tutorial.md) and other project documentation.