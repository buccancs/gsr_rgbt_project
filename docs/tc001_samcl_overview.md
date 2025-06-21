# TC001_SAMCL Repository Overview

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

## Integration with the Main Project

TC001_SAMCL integrates with the GSR-RGBT project as a submodule, providing specialized thermal imaging and segmentation capabilities. The main project uses this library to:

1. Capture thermal data from the TOPDON TC001 camera
2. Automatically segment regions of interest in the thermal imagery
3. Extract physiological signals from specific thermal regions
4. Provide real-time visualization of thermal data and derived signals

This integration complements the other components by adding advanced thermal segmentation capabilities, which can be particularly valuable for isolating specific body regions for non-contact physiological monitoring.

### Python Integration via tc001_thermal_capture.py

The main project includes a Python wrapper class `TC001ThermalCaptureThread` in `src/capture/tc001_thermal_capture.py` that provides a convenient interface to the TC001_SAMCL library. This class:

1. Extends the `BaseCaptureThread` class to maintain consistency with the project's architecture
2. Handles initialization and configuration of the TOPDON TC001 thermal camera
3. Provides both real capture and simulation modes for development and testing
4. Integrates the segmentation capabilities from TC001_SAMCL
5. Emits PyQt signals with captured thermal frames and segmentation results

Example usage of the Python wrapper:

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

# ... your application code ...

# Stop capturing
thermal_thread.stop()
```

For simulation mode, you can use:

```python
# Create a simulation capture thread
simulation_thread = TC001ThermalCaptureThread(
    config_path="third_party/TC001_SAMCL/configs/default_config.yaml",
    device_id=0,
    simulation_mode=True
)

# The simulation will generate synthetic thermal data
simulation_thread.start()
```

### Installation and Setup

To use TC001_SAMCL in your project:

1. **Add as a Submodule**:
   ```bash
   git submodule add https://github.com/your-organization/TC001_SAMCL.git third_party/TC001_SAMCL
   git submodule update --init --recursive
   ```

2. **Install Dependencies**:
   ```bash
   cd third_party/TC001_SAMCL
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:
   ```bash
   mkdir -p ckpt
   # Download pre-trained SAM model
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P ckpt/
   ```

4. **Configure Camera**:
   - Ensure the TOPDON TC001 camera is properly connected
   - Install any required camera drivers
   - Update the configuration file with the correct camera parameters:
     ```yaml
     camera:
       device_id: 0  # Adjust based on your system
       resolution: [640, 512]
       fps: 30
     ```

5. **Test Installation**:
   ```bash
   python tc001_live_seg.py --device 0 --config configs/default_config.yaml
   ```

## Usage Examples

### Basic Live Segmentation

```python
# Run live thermal capture and segmentation
python tc001_live_seg.py --device 2
```

Where `--device 2` specifies the camera device index.

### Configuration Options

The TC001_SAMCL tools support various configuration options:

- **Device Selection**: Specify which camera device to use
- **Segmentation Parameters**: Adjust segmentation sensitivity and thresholds
- **Visualization Settings**: Configure display options for the thermal imagery and segmentation results
- **Signal Processing Parameters**: Customize filtering and analysis of extracted signals

### Advanced Usage

For more advanced applications, the repository provides components that can be integrated into custom pipelines:

```python
# Example of using the ThermSeg class in a custom application
from inference import ThermSeg
from utils.configer import Configer

# Initialize the segmentation model
config_path = "configs/default_config.yaml"
configer = Configer(config_path=config_path)
segmenter = ThermSeg(configer)

# Capture and process a thermal frame
frame = capture_thermal_frame()  # Your frame capture function
segmentation_result = segmenter.process_frame(frame)

# Use the segmentation result
process_segmentation(segmentation_result)  # Your processing function
```

## Implementation Details

### Thermal Camera Setup

The repository is specifically designed for the TOPDON TC001 thermal camera, which provides temperature data as well as visual thermal imagery. The camera interface handles:

- Camera initialization and configuration
- Frame acquisition at the desired frame rate
- Temperature data extraction from thermal frames

### Segmentation Approach

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

This contrastive learning approach significantly improves the model's ability to segment thermal imagery, particularly in challenging conditions with low contrast or unusual temperature distributions.

### Signal Processing

Once regions are segmented, the system can extract temporal signals from these regions, such as:

- Temperature variations over time
- Thermal patterns that may correlate with physiological processes
- Motion patterns detected in thermal imagery

## Future Improvements

Potential improvements for the TC001_SAMCL repository could include:

1. Support for additional thermal camera models beyond the TOPDON TC001
2. Enhanced segmentation algorithms specifically optimized for physiological monitoring
3. Integration with other sensing modalities for multi-modal analysis
4. Improved real-time performance for edge computing applications
5. Extended signal processing capabilities for extracting more types of physiological information
