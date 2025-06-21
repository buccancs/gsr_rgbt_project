# Integrated System Tutorial: Using FactorizePhys, MMRPhys, and TC001_SAMCL Together

This tutorial provides a step-by-step guide for using the integrated system that combines FactorizePhys, MMRPhys, and TC001_SAMCL for comprehensive physiological sensing and analysis.

## Overview

The integrated system provides a complete pipeline for:

1. **Data Acquisition**: Capturing synchronized RGB video, thermal video, and physiological data using FactorizePhys
2. **Thermal Segmentation**: Identifying regions of interest in thermal imagery using TC001_SAMCL
3. **Signal Extraction**: Extracting physiological signals from video data using MMRPhys
4. **Analysis**: Processing and visualizing the extracted signals

For a comprehensive overview of how these repositories are integrated and work together, please refer to the [New Repositories Integration](new_repositories_integration.md) document.

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
   pip install -r requirements.txt
   ```

4. Build the C++ components (FactorizePhys and RGBTPhys_CPP):
   ```bash
   cd third_party/RGBTPhys_CPP
   make
   cd ../../third_party/FactorizePhys
   make
   cd ../..
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

## Troubleshooting

### Common Issues

1. **Hardware Connection Problems**:
   - Ensure all devices are properly connected and powered.
   - Check that the correct COM ports and device IDs are specified in the configuration files.

2. **Missing Dependencies**:
   - Run `pip install -r requirements.txt` to install all required dependencies.
   - For CUDA support, ensure you have the correct version of PyTorch installed.

3. **Segmentation Issues**:
   - Adjust the segmentation parameters in the TC001_SAMCL configuration file.
   - Try different thresholds and ROI settings.

4. **Signal Quality Problems**:
   - Ensure good lighting conditions for RGB capture.
   - Minimize subject movement during capture.
   - Check that the thermal camera is properly calibrated.

### Getting Help

If you encounter issues not covered in this tutorial:

1. Check the documentation for each individual component:
   - [FactorizePhys Documentation](factorizephys_overview.md)
   - [MMRPhys Documentation](mmrphys_overview.md)
   - [TC001_SAMCL Documentation](tc001_samcl_overview.md)

2. Review the integration documentation:
   - [New Repositories Integration](new_repositories_integration.md)
   - [RGBTPhys_CPP Integration](rgbt_phys_integration.md)
   - [Shimmer Integration](shimmer_integration.md)

3. Check the issue tracker on GitHub for known issues and solutions.

## Conclusion

The integrated system provides a powerful platform for physiological sensing and analysis. By combining the strengths of FactorizePhys, MMRPhys, and TC001_SAMCL, you can capture synchronized multi-modal data, extract physiological signals, and perform advanced analysis.

This tutorial has covered the basic usage scenarios, but the system is highly configurable and can be adapted to a wide range of research and application needs. Experiment with different configurations and processing pipelines to find the optimal setup for your specific use case.
