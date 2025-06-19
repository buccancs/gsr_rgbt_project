# Shimmer3 GSR+ Integration

## Overview
This document describes the integration of the Shimmer3 GSR+ unit into the GSR-RGBT project. The Shimmer3 GSR+ unit is a wearable sensor that can measure Galvanic Skin Response (GSR) and Photoplethysmography (PPG) signals. This integration allows the project to use data collected from the Shimmer3 GSR+ unit for GSR prediction from RGB-Thermal video.

## Shimmer3 GSR+ Specifications

### GSR Channel
- **Source**: Shimmer circuit design
- **Channels**: 1 Channel (GSR)
- **Sampling Rate**: 128 Hz
- **Format**: 16 bits, signed
- **Units**: kOhms
- **Filtering**: None

### PPG Channel
- **Source**: Shimmer circuit design
- **Channels**: 1 Channel (PPG)
- **Sampling Rate**: 128 Hz
- **Format**: 16 bits, signed
- **Units**: mV
- **Filtering**: None

### PPG-to-HR Algorithm
The PPG data can be passed through the PPG-to-HR algorithm, which is available in the Consensys software application. This algorithm converts the PPG signal to a heart rate (bpm). The heart rate has a value of -1 for the first few samples as the algorithm enters a training period.

## Data Collection Protocol
During data collection, the subject connects the GSR electrodes to the index and middle finger on the left hand and the PPG probe onto the left ear lobe. The data collection protocol includes:

1. First minute: Subject is sitting down
2. Second minute: Subject is walking

## Data Format
The Shimmer3 GSR+ unit outputs data in a tab-separated CSV file with the following columns:
- Timestamp (yyyy/mm/dd hh:mm:ss.000)
- Accelerometer X, Y, Z (m/s^2)
- GSR (kOhms)
- PPG-to-HR (BPM)
- PPG (mV)

## Integration Changes
The following changes were made to integrate the Shimmer3 GSR+ unit into the project:

1. **Data Loading**: Updated `data_loader.py` to handle the Shimmer data format, including extracting GSR, PPG, and heart rate data.
2. **Signal Processing**: Enhanced `preprocessing.py` to process both GSR and PPG signals, and to include heart rate data in the processed output.
3. **Configuration**: Updated `config.py` to set the GSR sampling rate to 128 Hz to match the Shimmer3 GSR+ unit.
4. **Testing**: Added tests for loading and processing Shimmer data, including tests for PPG and heart rate data.

## Usage
To use Shimmer data in the project:

1. Place the Shimmer data file (e.g., `SampleGSRPPG_Session1_Shimmer_B640_Calibrated_SD.csv`) in the session directory.
2. The data loader will automatically detect and load the Shimmer data file if present.
3. The processed data will include GSR, PPG, and heart rate data if available.

## Sample Data
A sample Shimmer data file is included in the project at `data/sample/SampleGSRPPG_Session1_Shimmer_B640_Calibrated_SD.csv`. This file can be used for testing and development purposes.