# Integration of New Repositories in the GSR-RGBT Project

## Overview

This document provides an overview of the newly added repositories to the GSR-RGBT project and explains how they work together to form a comprehensive system for physiological sensing using RGB and thermal imaging. The integration of these repositories enhances the project's capabilities for data capture, processing, and analysis.

## Newly Added Repositories

Three new repositories have been added to the GSR-RGBT project:

1. **FactorizePhys**: A C++ library for synchronized capture of RGB video, thermal video, and physiological data
2. **MMRPhys**: A deep learning framework for remote physiological sensing using multidimensional attention and target signal constrained factorization
3. **TC001_SAMCL**: A specialized repository for live capture and segmentation using the TOPDON TC001 thermal camera

These repositories complement the existing submodules:

- **RGBTPhys_CPP**: The original library for synchronized data stream capture
- **neurokit2**: A Python toolbox for neurophysiological signal processing
- **physiokit**: Tools for physiological data analysis
- **pyshimmer**: Interface for Shimmer physiological sensors

## Integration Architecture

The integration of these repositories creates a complete pipeline for physiological sensing:

```
Data Acquisition → Preprocessing → Signal Extraction → Analysis
[FactorizePhys]    [TC001_SAMCL]    [MMRPhys]         [neurokit2/physiokit]
[RGBTPhys_CPP]
```

### Data Flow

1. **Data Acquisition Layer**:
   - **FactorizePhys** and **RGBTPhys_CPP** handle the synchronized capture of RGB video, thermal video, and contact-based physiological signals
   - These libraries ensure that all data streams are properly time-aligned
   - Data is saved in a structured format organized by participant ID and experimental condition

2. **Preprocessing Layer**:
   - **TC001_SAMCL** provides specialized thermal image processing and segmentation
   - It identifies regions of interest in thermal imagery that are relevant for physiological signal extraction
   - The segmentation results can be used to focus the analysis on specific body regions

3. **Signal Extraction Layer**:
   - **MMRPhys** applies deep learning techniques to extract physiological signals from the video data
   - It can work with both RGB and thermal video inputs
   - The extracted signals provide non-contact alternatives to traditional physiological measurements

4. **Analysis Layer**:
   - **neurokit2** and **physiokit** provide tools for analyzing the extracted physiological signals
   - **pyshimmer** allows for comparison with ground truth data from contact-based sensors

## Use Cases and Workflows

### Complete Physiological Monitoring Workflow

1. Use **FactorizePhys** to capture synchronized RGB video, thermal video, and contact-based physiological data
2. Apply **TC001_SAMCL** to segment regions of interest in the thermal imagery
3. Process the video data with **MMRPhys** to extract non-contact physiological signals
4. Analyze both contact-based and non-contact signals using **neurokit2** and **physiokit**
5. Validate the results by comparing the extracted signals with the ground truth from contact sensors

### Remote Physiological Sensing Only

1. Use **RGBTPhys_CPP** or **FactorizePhys** to capture RGB and thermal video
2. Apply **MMRPhys** directly to the video data to extract physiological signals
3. Analyze the extracted signals using **neurokit2** and **physiokit**

### Thermal-Focused Analysis

1. Use **TC001_SAMCL** for specialized thermal imaging with the TOPDON TC001 camera
2. Apply the segmentation capabilities to identify specific regions of interest
3. Extract temporal signals from these regions
4. Process the signals with **MMRPhys** or analyze them directly with **neurokit2**

## Technical Integration Details

### Data Formats and Compatibility

The repositories have been designed to work with compatible data formats:

- **FactorizePhys** and **RGBTPhys_CPP** save data in formats that can be directly used by **MMRPhys**
- **TC001_SAMCL** produces segmentation masks that can be applied to the thermal data before processing with **MMRPhys**
- All extracted signals can be analyzed using **neurokit2** and **physiokit**

### Configuration and Setup

Each repository has its own configuration system:

- **FactorizePhys** uses text-based configuration files for capture parameters
- **MMRPhys** uses YAML configuration files for model and training parameters
- **TC001_SAMCL** uses command-line arguments and configuration files for camera and segmentation settings

When integrating these repositories, ensure that the configurations are compatible, particularly for aspects like frame rates, resolutions, and file paths.

## Future Integration Opportunities

The addition of these repositories opens up several opportunities for future enhancements:

1. **Real-time Integration**: Connecting the data capture components directly to the processing and analysis components for real-time physiological monitoring
2. **Multi-modal Fusion**: Combining signals from different modalities (RGB, thermal, contact sensors) for more robust physiological measurements
3. **Extended Analysis**: Applying advanced analysis techniques from **neurokit2** and **physiokit** to the signals extracted by **MMRPhys**
4. **Feedback Systems**: Creating closed-loop systems that use the extracted physiological signals to adapt the data capture or provide feedback to users

## Detailed Usage Instructions

For detailed step-by-step instructions on how to use the integrated system, please refer to the [Integrated System Tutorial](integrated_system_tutorial.md). This tutorial provides comprehensive guidance on:

1. Setting up the environment and hardware
2. Running the system in different scenarios (real-time monitoring, batch processing, end-to-end processing)
3. Configuring each component for optimal performance
4. Troubleshooting common issues

## Conclusion

The integration of FactorizePhys, MMRPhys, and TC001_SAMCL significantly enhances the capabilities of the GSR-RGBT project. Together with the existing submodules, these repositories form a comprehensive system for physiological sensing that combines the strengths of contact-based and non-contact approaches, RGB and thermal imaging, and traditional signal processing with deep learning techniques.

This integrated system provides a powerful platform for research and applications in areas such as healthcare monitoring, affective computing, human-computer interaction, and biometric authentication.
