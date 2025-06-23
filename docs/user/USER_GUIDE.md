# User Guide: GSR-RGBT Physiological Monitoring System

This comprehensive user guide provides step-by-step instructions for using the GSR-RGBT system that combines FactorizePhys, MMRPhys, and TC001_SAMCL for comprehensive physiological sensing and analysis.

## Overview

The GSR-RGBT system provides a complete pipeline for:

1. **Data Acquisition**: Capturing synchronized RGB video, thermal video, and physiological data using FactorizePhys
2. **Thermal Segmentation**: Identifying regions of interest in thermal imagery using TC001_SAMCL
3. **Signal Extraction**: Extracting physiological signals from video data using MMRPhys
4. **Analysis**: Processing and visualizing the extracted signals

For a comprehensive overview of how these repositories are integrated and work together, please refer to the [System Architecture](ARCHITECTURE.md) document.

## Prerequisites

Before using the integrated system, ensure you have:

1. Installed all required dependencies for each component
2. Set up the hardware (if using real hardware)
   - RGB camera
   - TOPDON TC001 thermal camera (or compatible)
   - Shimmer GSR+ device (or other physiological sensors)
3. Downloaded pre-trained models for MMRPhys

## Installation

1. Clone the GSR-RGBT project repository:
   ```bash
   git clone https://github.com/your-username/gsr_rgbt_project.git
   cd gsr_rgbt_project
   ```

2. Initialize and update all submodules:
   ```bash
   git submodule update --init --recursive
   ```

3. Install the required dependencies:
   ```bash
   ./gsr_rgbt_tools.sh setup
   ```

4. Build the C++ components (FactorizePhys and RGBTPhys_CPP):
   ```bash
   cd third_party/RGBTPhys_CPP
   make
   cd ../../third_party/FactorizePhys
   make
   cd ../..
   ```

## Quick Start

The easiest way to get started is using the unified tool script:

1. **Set up the environment**:
   ```bash
   ./gsr_rgbt_tools.sh setup
   ```

2. **Start data collection**:
   ```bash
   ./gsr_rgbt_tools.sh collect
   ```

3. **Train models on collected data**:
   ```bash
   ./gsr_rgbt_tools.sh train
   ```

4. **Evaluate results**:
   ```bash
   ./gsr_rgbt_tools.sh evaluate --visualize
   ```

## Usage Scenarios

### Scenario 1: Real-Time Monitoring

The real-time monitoring application provides a graphical interface for live physiological monitoring.

1. Run the real-time monitoring application:
   ```bash
   python src/scripts/real_time_monitoring.py --simulation
   ```

   For real hardware:
   ```bash
   python src/scripts/real_time_monitoring.py --factorize-config third_party/FactorizePhys/default_config --thermal-device 0 --tc001-config third_party/TC001_SAMCL/configs/default_config.yaml --mmrphys-model MMRPhysLEF --mmrphys-weights path/to/pretrained/weights.pth
   ```

2. The application window will display:
   - RGB video feed
   - Thermal video feed
   - Segmentation mask
   - Real-time physiological measurements (heart rate, GSR, temperature, signal quality)

3. Use the control buttons to:
   - Start/stop recording
   - Save snapshots
   - Exit the application

4. Recorded data will be saved in the `data/real_time_monitoring` directory.

### Scenario 2: Batch Processing

For processing pre-recorded data or running experiments:

1. Capture synchronized data using FactorizePhys:
   ```bash
   cd third_party/FactorizePhys
   ./RGBTPhys.exe default_config /path/to/save/data participant_id
   cd ../..
   ```

2. Process the thermal data with TC001_SAMCL:
   ```bash
   python third_party/TC001_SAMCL/tc001_process.py --input /path/to/thermal/data --output /path/to/output/segmentation
   ```

3. Extract physiological signals with MMRPhys:
   ```bash
   python third_party/MMRPhys/main.py --config_file configs/infer_configs/BVP/Cross/RGB/iBVP_UBFC-rPPG_MMRPhys_SFSAM_Label.yaml --input /path/to/rgb/data
   ```

4. Analyze the results:
   ```bash
   python src/scripts/analyze_results.py --phys /path/to/phys/data --mmrphys /path/to/mmrphys/results --output /path/to/analysis/output
   ```

### Scenario 3: End-to-End Processing

For a complete end-to-end processing pipeline:

1. Run the integrated capture and analysis script:
   ```bash
   python src/scripts/integrated_capture_and_analysis.py --simulation
   ```

   For real hardware:
   ```bash
   python src/scripts/integrated_capture_and_analysis.py --factorize-config third_party/FactorizePhys/default_config --thermal-device 0 --tc001-config third_party/TC001_SAMCL/configs/default_config.yaml --mmrphys-model MMRPhysLEF --mmrphys-weights path/to/pretrained/weights.pth --participant-id subject_01 --duration 60
   ```

2. This script will:
   - Capture synchronized data for the specified duration
   - Process the thermal data to identify regions of interest
   - Extract physiological signals from the RGB and thermal data
   - Analyze the results and generate visualizations
   - Save all data and results to the specified output directory

## Configuration Options

### FactorizePhys Configuration

The FactorizePhys configuration file controls the data capture parameters:

```
# Example configuration
configure_thermal_camera = true
capture_thermal = true
save_thermal = true
show_thermal = true
thermal_fps = 30
thermal_im_width = 640
thermal_im_height = 512
capture_rgb = true
save_rgb = true
show_rgb = true
rgb_camera_number = 1
rgb_im_width = 1280
rgb_im_height = 720
rgb_fps = 30
thread_sleep_interval_acquisition = 500
exp_condition = baseline
acquisition_duration = 180
capture_phys = true
com_port = COM4
baud_rate = 2000000
phys_channels = EDA,Resp,PPG Finger,PPG Ear,arduino_ts,EventCode
is_shimmer_device = true
shimmer_device_type = GSR+
```

### TC001_SAMCL Configuration

The TC001_SAMCL configuration file controls the thermal segmentation parameters:

```yaml
# Example configuration
model:
  type: sam
  backbone: vit_b
  checkpoint: path/to/sam_vit_b_01ec64.pth

segmentation:
  threshold: 0.5
  min_area: 100
  max_area: 10000

roi:
  face: true
  hands: true
  custom: []
```

### MMRPhys Configuration

The MMRPhys configuration file controls the physiological signal extraction parameters:

```yaml
# Example configuration
model:
  name: MMRPhysLEF
  pretrained: path/to/pretrained/weights.pth

inference:
  batch_size: 4
  window_size: 300
  stride: 10

output:
  save_predictions: true
  visualize: true
```

## Advanced Usage

### Custom Region of Interest Selection

You can define custom regions of interest for thermal segmentation:

1. In the TC001_SAMCL configuration file, add custom ROI definitions:
   ```yaml
   roi:
     face: true
     hands: true
     custom:
       - name: forehead
         points: [[100, 50], [200, 50], [200, 100], [100, 100]]
       - name: cheeks
         points: [[150, 150], [250, 150], [250, 200], [150, 200]]
   ```

2. The custom ROIs will be used in addition to the automatically detected regions.

### Multi-modal Signal Fusion

To combine signals from different modalities:

1. Capture data from all sources (RGB, thermal, contact sensors).
2. Extract signals from each modality.
3. Use the signal fusion script:
   ```bash
   python src/scripts/fuse_signals.py --rgb /path/to/rgb/signals --thermal /path/to/thermal/signals --contact /path/to/contact/signals --output /path/to/output
   ```

4. The fused signals will provide more robust physiological measurements.

## Command Line Interface

The project includes a unified tool script for easy access to all functionality:

### Basic Commands

```bash
# Get help
./gsr_rgbt_tools.sh help

# Set up environment
./gsr_rgbt_tools.sh setup

# Start data collection
./gsr_rgbt_tools.sh collect

# Train models
./gsr_rgbt_tools.sh train

# Evaluate models
./gsr_rgbt_tools.sh evaluate

# Run tests
./gsr_rgbt_tools.sh test
```

### Advanced Commands

```bash
# Train specific model with custom configuration
./gsr_rgbt_tools.sh train --model=lstm --config=configs/custom.yaml

# Collect data in simulation mode
./gsr_rgbt_tools.sh collect --simulate

# Evaluate with visualizations
./gsr_rgbt_tools.sh evaluate --visualize

# Get detailed help for specific commands
./gsr_rgbt_tools.sh help collect
./gsr_rgbt_tools.sh help train
./gsr_rgbt_tools.sh help troubleshoot
```

## Troubleshooting

### Common Issues

1. **Hardware Connection Problems**:
   - Ensure all devices are properly connected and powered.
   - Check that the correct COM ports and device IDs are specified in the configuration files.
   - Run `./gsr_rgbt_tools.sh test` to validate hardware connectivity.

2. **Missing Dependencies**:
   - Run `./gsr_rgbt_tools.sh setup --force` to reinstall all dependencies.
   - For CUDA support, ensure you have the correct version of PyTorch installed.

3. **Segmentation Issues**:
   - Adjust the segmentation parameters in the TC001_SAMCL configuration file.
   - Try different thresholds and ROI settings.

4. **Signal Quality Problems**:
   - Ensure good lighting conditions for RGB capture.
   - Minimize subject movement during capture.
   - Check that the thermal camera is properly calibrated.

5. **Python Import Errors**:
   - Activate the virtual environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
   - Rebuild Cython extensions: `python setup.py build_ext --inplace`

6. **GUI Application Crashes**:
   - Check console output for error messages
   - Try running in simulation mode first: `./gsr_rgbt_tools.sh collect --simulate`
   - Verify all hardware is properly connected

### Getting Help

If you encounter issues not covered in this guide:

1. **Use the built-in troubleshooting guide**:
   ```bash
   ./gsr_rgbt_tools.sh help troubleshoot
   ```

2. **Check the documentation for each component**:
   - [System Architecture](ARCHITECTURE.md)
   - [Developer Guide](DEVELOPER_GUIDE.md)
   - [Technical Guide](technical_guide.md)

3. **Run system diagnostics**:
   ```bash
   ./gsr_rgbt_tools.sh test
   ```

4. **Check the issue tracker on GitHub** for known issues and solutions.

## Data Management

### Data Storage Structure

The system organizes data in a structured format:

```
data/
├── recordings/
│   ├── subject_01/
│   │   ├── session_001/
│   │   │   ├── rgb_video.avi
│   │   │   ├── thermal_video.avi
│   │   │   ├── gsr_data.csv
│   │   │   └── timestamps.csv
│   │   └── session_002/
│   └── subject_02/
├── processed/
│   ├── features/
│   ├── models/
│   └── results/
└── outputs/
    ├── models/
    ├── logs/
    └── visualizations/
```

### Data Export and Analysis

1. **Export data for external analysis**:
   ```bash
   python src/scripts/export_data.py --input data/recordings/subject_01 --format csv --output exports/
   ```

2. **Generate analysis reports**:
   ```bash
   python src/scripts/generate_report.py --data data/recordings --output reports/
   ```

3. **Visualize results**:
   ```bash
   ./gsr_rgbt_tools.sh evaluate --visualize --output visualizations/
   ```

## Best Practices

### Data Collection

1. **Environment Setup**:
   - Ensure consistent lighting conditions
   - Minimize background noise and movement
   - Maintain comfortable room temperature

2. **Subject Preparation**:
   - Allow subjects to acclimate to the environment
   - Ensure proper sensor placement
   - Provide clear instructions to subjects

3. **Quality Control**:
   - Monitor signal quality in real-time
   - Record environmental conditions
   - Document any issues or anomalies

### Data Processing

1. **Preprocessing**:
   - Apply appropriate filtering to remove noise
   - Synchronize data streams properly
   - Handle missing data appropriately

2. **Feature Extraction**:
   - Use validated feature extraction methods
   - Consider multiple time windows
   - Validate extracted features

3. **Model Training**:
   - Use appropriate cross-validation strategies
   - Monitor for overfitting
   - Validate on independent test sets

## Conclusion

The GSR-RGBT system provides a powerful platform for physiological sensing and analysis. By combining the strengths of FactorizePhys, MMRPhys, and TC001_SAMCL, you can capture synchronized multi-modal data, extract physiological signals, and perform advanced analysis.

This user guide has covered the basic usage scenarios, but the system is highly configurable and can be adapted to a wide range of research and application needs. Experiment with different configurations and processing pipelines to find the optimal setup for your specific use case.

For more advanced topics and development information, please refer to the [Developer Guide](DEVELOPER_GUIDE.md) and [Technical Guide](technical_guide.md).