# GSR-RGBT Project Plan Timeline

## Introduction

This document outlines the comprehensive project plan for the Contactless GSR Prediction project, based on our conversation that started on May 15, 2023. The plan is organized into sequential phases with specific tasks and timelines to transform the current codebase into a validated, research-grade system.

## Phase 1: Documentation Update (May 15-16, 2023)

### May 15, 2023: Review and Update Documentation

1. **Update Information Sheet and Consent Form**
   - Update docs/information_sheet.tex and docs/consent_form.tex
   - Clearly state that one hand will be recorded by video while the other will have sensors attached
   - Remove any ambiguity in the experimental setup description

2. **Update Research Proposal**
   - Revise the "Methods" and "Data Acquisition" sections in docs/proposal.tex
   - Update text to describe the contralateral ("mirrored") experimental design
   - Add a paragraph detailing the advanced preprocessing pipeline

3. **Update README.md**
   - Update the main README.md file to include instructions for the entire workflow
   - Include information about the create_mock_data.py script and test suite
   - Ensure all repository references use consistent naming (gsr_rgbt_project)

## Phase 2: Data Preprocessing Pipeline Development (May 17-19, 2023)

### May 17, 2023: Implement Multi-ROI Approach

1. **Enhance ROI Detection**
   - Update src/processing/preprocessing.py to use MediaPipe for hand landmark detection
   - Implement the Multi-ROI approach to extract features from multiple regions:
     - Index finger base (MediaPipe landmark 5)
     - Ring finger base (MediaPipe landmark 13)
     - Center of the palm (average of landmarks 0, 5, 9, 13, 17)

2. **Update Feature Engineering**
   - Modify src/processing/feature_engineering.py to process signals from multiple ROIs
   - Create a richer feature set by combining signals from all ROIs
   - Remove GSR_Tonic from features to avoid data leakage

### May 18, 2023: Improve Data Synchronization

1. **Implement Centralized Timestamp Authority**
   - Create a new TimestampThread class in src/utils/timestamp_thread.py
   - Emit timestamps at a high frequency (200Hz) for precise temporal alignment
   - Use time.perf_counter_ns() for nanosecond precision

2. **Update Data Logger**
   - Enhance src/utils/data_logger.py to log timestamps for each frame
   - Implement proper error handling and resource cleanup
   - Create CSV files for RGB and thermal frame timestamps

### May 19, 2023: Refine Thermal Data Processing

1. **Enhance Thermal Feature Extraction**
   - Update the feature engineering pipeline to better utilize thermal data
   - Apply the same Multi-ROI approach to thermal frames
   - Create separate feature columns for RGB and thermal data

2. **Improve Dual-Stream Model Support**
   - Update the feature reshaping code for dual-stream models
   - Use column name prefixes to separate RGB and thermal features
   - Ensure proper handling of the new feature structure

## Phase 3: Model Development and Validation (May 20-23, 2023)

### May 20, 2023: Implement Advanced Model Architectures

1. **Develop Dual-Stream CNN-LSTM Model**
   - Implement a model with separate CNN streams for RGB and thermal data
   - Add a fusion layer to combine features from both streams
   - Use LSTM layers to capture temporal dependencies

2. **Implement Transformer-Based Models**
   - Create a PyTorchTransformer model for sequence data
   - Implement self-attention mechanisms for better temporal modeling
   - Add positional encoding for sequence order information

### May 21-22, 2023: Implement Robust Validation Strategy

1. **Implement Leave-One-Subject-Out Cross-Validation**
   - Update src/scripts/train_model.py to use LOSO cross-validation
   - Ensure proper train/validation/test splits with subject-aware validation
   - Prevent data leakage between subjects

2. **Implement Early Stopping and Model Checkpointing**
   - Add early stopping to prevent overfitting
   - Implement model checkpointing to save the best model
   - Save detailed metadata about the training process

### May 23, 2023: Implement Evaluation and Visualization

1. **Create Comprehensive Evaluation Metrics**
   - Implement MSE, MAE, and RMSE metrics
   - Add correlation coefficient and Bland-Altman analysis
   - Create functions to evaluate model performance across subjects

2. **Develop Visualization Tools**
   - Create scripts to visualize predictions vs. ground truth
   - Implement ROI contribution analysis
   - Generate model comparison reports

## Phase 4: Data Collection and Analysis (May 24-26, 2023)

### May 24, 2023: Prepare for Data Collection

1. **Finalize Experimental Protocol**
   - Create a detailed protocol for data collection
   - Define baseline, stressor, and recovery periods
   - Implement protocol timing in the GUI application

2. **Test the Complete Pipeline**
   - Generate mock data for end-to-end testing
   - Run the full pipeline on mock data
   - Verify that all components work together correctly

### May 25, 2023: Collect and Process Data

1. **Collect Data from Participants**
   - Use the validated GUI application to collect data
   - Ensure proper synchronization between all data streams
   - Follow the experimental protocol consistently

2. **Process the Collected Data**
   - Run the preprocessing pipeline on the collected data
   - Extract features from all ROIs
   - Align GSR and video signals

### May 26, 2023: Final Analysis and Documentation

1. **Train and Evaluate Models**
   - Run the full Leave-One-Subject-Out cross-validation
   - Generate performance metrics and visualizations
   - Compare different model architectures

2. **Document Results and Findings**
   - Create a comprehensive report of the results
   - Document the effectiveness of the Multi-ROI approach
   - Provide recommendations for future improvements

## Conclusion

This timeline provides a structured approach to transforming the GSR-RGBT project into a robust, research-grade system for contactless GSR prediction. By following this plan, we will address the critical issues in ROI detection, data synchronization, and feature engineering, resulting in a system capable of producing high-quality results.

The plan is designed to be flexible, allowing for adjustments based on findings during implementation. Regular testing and validation throughout the process will ensure that each component works correctly and contributes to the overall goal of accurate contactless GSR prediction.
