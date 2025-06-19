# Cross-Referencing and Deep Research on GSR Measurement Methods and Neural Networks for Multimodal Signal Prediction

## 1. Introduction

This research report provides a comprehensive analysis of the measurement methods, data extraction techniques, and neural network architectures used in the GSR-RGBT project. The project aims to predict Galvanic Skin Response (GSR) from synchronized RGB and thermal video streams, using a contactless approach that leverages computer vision and machine learning techniques.

## 2. Measurement Methods

### 2.1 GSR Measurement Principles

Galvanic Skin Response (GSR), also known as Electrodermal Activity (EDA), measures changes in the electrical conductance of the skin caused by variations in sweat gland activity. These changes are controlled by the sympathetic nervous system, making GSR a reliable indicator of psychological or physiological arousal.

The project employs two approaches to GSR measurement:

1. **Contact-based measurement**: Using the Shimmer3 GSR+ sensor, which applies a small constant voltage to the skin and measures the resulting current. The sensor is attached to the fingers of one hand and provides ground-truth GSR data for training and validation.

2. **Contactless prediction**: Using RGB and thermal video of the opposite hand to predict GSR values. This novel approach is based on the "mirror effect" of contralateral sympathetic responses.

The Shimmer3 GSR+ sensor captures both tonic (slow-changing baseline) and phasic (rapid changes in response to stimuli) components of the GSR signal. The NeuroKit2 library is used to process the raw GSR signal and decompose it into these components:

```python
signals, info = nk.eda_process(gsr_df["gsr_value"], sampling_rate=sampling_rate)
```

### 2.2 RGB and Thermal Video Capture

The project uses a dual-camera setup to capture both RGB and thermal video streams:

1. **RGB Camera**: A standard RGB camera captures visible light reflections from the hand. The `VideoCaptureThread` class handles RGB video capture using OpenCV:

```python
self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # On Windows
```

2. **Thermal Camera**: A FLIR thermal camera captures infrared radiation emitted by the hand, which correlates with skin temperature. The `ThermalCaptureThread` class interfaces with the camera using the PySpin library:

```python
self.camera = cam_list.GetByIndex(self.camera_index)
self.camera.Init()
self.camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
```

Both capture threads run asynchronously and emit frames with high-resolution timestamps:

```python
current_capture_time = time.perf_counter_ns()  # High-resolution timestamp
self.frame_captured.emit(frame, current_capture_time)
```

### 2.3 Data Synchronization Mechanisms

Accurate synchronization between different data streams is crucial for this project. The system employs several mechanisms to ensure temporal alignment:

1. **High-resolution timestamps**: Each data stream (RGB video, thermal video, GSR) is tagged with high-precision timestamps using `time.perf_counter_ns()`.

2. **Timestamp logging**: The `DataLogger` class logs timestamps for each frame and GSR sample:

```python
# For video frames
self.rgb_timestamps_writer.writerow([frame_num, timestamp])

# For GSR data
self.gsr_writer.writerow([system_timestamp, shimmer_timestamp, gsr_value])
```

3. **Signal alignment**: During processing, the `align_signals` function aligns video-derived signals to GSR timestamps using interpolation:

```python
aligned_data = cy_align_signals(gsr_data, video_data, gsr_timestamps, video_timestamps)
```

### 2.4 The "Mirror Effect" of Contralateral Sympathetic Responses

A key innovation in this project is the exploration of the "mirror effect" of contralateral sympathetic responses. This refers to the phenomenon where sympathetic nervous system responses (like sweating) occur bilaterally across the body.

The project leverages this effect by:
- Recording RGB and thermal video of one hand
- Measuring ground-truth GSR from the opposite hand
- Training models to predict the GSR values from the video data

This approach allows for contactless GSR prediction while still having ground-truth measurements for training and validation. The scientific basis for this approach is that sympathetic nervous system activation affects both hands similarly, so visual cues from one hand can predict GSR measured from the other hand.

## 3. Data Extraction Techniques

### 3.1 Multi-ROI Approach Using MediaPipe Hand Landmarks

The project employs a Multi-ROI (Multiple Regions of Interest) approach to extract features from hand videos. This technique uses MediaPipe's hand landmark detection to identify key points on the hand:

```python
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:
    results = hands.process(rgb_frame)
```

Three specific regions are targeted based on their physiological significance:

1. **Index finger base** (MediaPipe landmark 5): Contains a high concentration of sweat glands
2. **Ring finger base** (MediaPipe landmark 13): Shows strong vascular patterns
3. **Center of the palm** (average of landmarks 0, 5, 9, 13, 17): Provides a stable reference point with consistent blood flow

These ROIs are defined in the `define_multi_roi` function:

```python
rois = {}
if 5 in hand_landmarks:
    x, y, _ = hand_landmarks[5]
    rois["index_finger_base"] = (max(0, x - roi_size // 2), max(0, y - roi_size // 2), 
                                min(roi_size, w - x + roi_size // 2), min(roi_size, h - y + roi_size // 2))
```

### 3.2 Feature Engineering Process

The feature engineering pipeline transforms raw video and GSR data into features suitable for machine learning:

1. **ROI Signal Extraction**: For each ROI, the mean pixel values are extracted:

```python
mean_signal = extract_roi_signal(frame, roi)  # Returns [B, G, R] values
```

2. **GSR Signal Processing**: The raw GSR signal is processed to extract tonic and phasic components:

```python
signals, info = nk.eda_process(gsr_df["gsr_value"], sampling_rate=sampling_rate)
```

3. **Signal Alignment**: Video-derived signals are aligned to GSR timestamps:

```python
aligned_df = align_signals(processed_gsr, video_df)
```

4. **Feature Windowing**: The aligned data is segmented into overlapping windows for sequence modeling:

```python
X, y = create_feature_windows(aligned_df, feature_columns, target_column, window_size_samples, step_size)
```

The complete pipeline is implemented in the `create_dataset_from_session` function, which handles both RGB-only and dual-stream (RGB + thermal) data.

### 3.3 Signal Processing Techniques for GSR Data

The project uses NeuroKit2 for GSR signal processing, which applies several techniques:

1. **Signal cleaning**: Removes noise and artifacts from the raw GSR signal
2. **Tonic-phasic decomposition**: Separates the GSR signal into:
   - **Tonic component**: Slow-changing baseline level of skin conductance
   - **Phasic component**: Rapid changes in skin conductance in response to stimuli

The processed GSR data includes columns for raw, cleaned, tonic, and phasic components:

```python
processed_df = signals.rename(
    columns={
        "EDA_Raw": "GSR_Raw",
        "EDA_Clean": "GSR_Clean",
        "EDA_Tonic": "GSR_Tonic",
        "EDA_Phasic": "GSR_Phasic",
    }
)
```

### 3.4 Alignment and Windowing Techniques

The project uses sophisticated techniques for aligning and windowing time-series data:

1. **Signal alignment**: The `align_signals` function aligns video-derived signals to GSR timestamps using interpolation. It has both Cython and Python implementations for performance optimization:

```python
# Cython implementation
aligned_data = cy_align_signals(gsr_data, video_data, gsr_timestamps, video_timestamps)

# Python implementation
combined_df = pd.concat([gsr_df.set_index("timestamp"), video_signals.set_index("timestamp")], axis=1)
aligned_df = combined_df.interpolate(method="time").reindex(gsr_df.set_index("timestamp").index)
```

2. **Feature windowing**: The `create_feature_windows` function creates overlapping windows for sequence modeling:

```python
# Window size: 5 seconds of data at the GSR sampling rate
window_size_samples = 5 * gsr_sampling_rate
step_size = gsr_sampling_rate // 2  # 50% overlap
```

These techniques ensure that the data is properly aligned and formatted for the neural network models.

## 4. Neural Network Architectures

### 4.1 Overview of Model Architectures

The project implements several neural network architectures for GSR prediction:

1. **LSTM Models**: Long Short-Term Memory networks for sequence modeling
   - `PyTorchLSTM`: Basic LSTM model
   - `PyTorchLSTMModel`: Wrapper implementing the `BaseModel` interface

2. **CNN Models**: Convolutional Neural Networks for feature extraction
   - `PyTorchCNN`: Basic CNN model
   - `PyTorchCNNModel`: Wrapper implementing the `BaseModel` interface

3. **CNN-LSTM Models**: Hybrid models combining CNN and LSTM layers
   - `PyTorchCNNLSTM`: Hybrid CNN-LSTM model
   - `PyTorchCNNLSTMModel`: Wrapper implementing the `BaseModel` interface

4. **Dual-Stream Models**: Models that process RGB and thermal inputs separately
   - `PyTorchDualStreamCNNLSTM`: Dual-stream CNN-LSTM model
   - `PyTorchDualStreamCNNLSTMModel`: Wrapper implementing the `BaseModel` interface

5. **Transformer Models**: Self-attention based models for sequence data
   - `PyTorchTransformer`: Transformer model with positional encoding
   - `PyTorchTransformerModel`: Wrapper implementing the `BaseModel` interface

6. **ResNet Models**: Residual Networks with skip connections
   - `PyTorchResNet`: ResNet model with residual blocks
   - `PyTorchResNetModel`: Wrapper implementing the `BaseModel` interface

7. **Autoencoder and VAE Models**: For unsupervised feature learning and generative modeling
   - `PyTorchAutoencoder`: Autoencoder model
   - `PyTorchVAE`: Variational Autoencoder model

### 4.2 Dual-Stream Models for RGB and Thermal Data

The `PyTorchDualStreamCNNLSTM` model is particularly relevant for this project as it processes both RGB and thermal video streams:

```python
class PyTorchDualStreamCNNLSTM(nn.Module):
    """
    PyTorch Dual-Stream CNN-LSTM hybrid model for processing RGB and thermal video streams.

    This model has two separate CNN streams for processing RGB and thermal video frames,
    followed by a fusion layer that combines the features from both streams.
    The fused features are then passed to an LSTM to capture temporal dependencies.
    """
```

Key components of this architecture:

1. **Separate CNN streams**: The model has two parallel CNN streams, one for RGB and one for thermal data:

```python
# Build RGB CNN stream
self.rgb_cnn_layers = self._build_cnn_stream(
    input_channels=rgb_input_shape[0],
    cnn_filters=cnn_filters,
    cnn_kernel_sizes=cnn_kernel_sizes,
    cnn_strides=cnn_strides,
    cnn_pool_sizes=cnn_pool_sizes,
    activations=activations,
    dropout_rate=dropout_rate,
    stream_name="rgb"
)

# Build thermal CNN stream
self.thermal_cnn_layers = self._build_cnn_stream(
    input_channels=thermal_input_shape[0],
    cnn_filters=cnn_filters,
    cnn_kernel_sizes=cnn_kernel_sizes,
    cnn_strides=cnn_strides,
    cnn_pool_sizes=cnn_pool_sizes,
    activations=activations,
    dropout_rate=dropout_rate,
    stream_name="thermal"
)
```

2. **Feature fusion**: The features from both streams are concatenated and fused:

```python
# Concatenate features from both streams
combined_features = torch.cat([rgb_features, thermal_features], dim=1)

# Fuse features
fused = self.fusion_layer(combined_features)
```

3. **Temporal modeling**: The fused features are processed by an LSTM to capture temporal dependencies:

```python
# Stack features from all frames
sequence = torch.stack(fused_features, dim=1)  # (batch_size, seq_len, fused_dim)

# Pass through LSTM
lstm_out, _ = self.lstm(sequence)
```

4. **Fully connected layers**: The LSTM output is passed through fully connected layers to produce the final prediction:

```python
# Pass through fully connected layers
for layer in self.fc_layers:
    lstm_out = layer(lstm_out)
```

This architecture effectively combines spatial features from both RGB and thermal modalities with temporal dynamics to predict GSR values.

### 4.3 Multimodal Fusion Approaches

The project employs several approaches to multimodal fusion:

1. **Early fusion**: Combining raw features from different modalities before processing
2. **Late fusion**: Processing each modality separately and combining the outputs
3. **Hybrid fusion**: The dual-stream architecture uses a hybrid approach:
   - Each modality is processed by its own CNN stream
   - Features are fused at an intermediate level
   - The fused features are processed by an LSTM for temporal modeling

The fusion layer in the dual-stream model is a simple linear layer:

```python
self.fusion_layer = nn.Linear(rgb_cnn_output_size + thermal_cnn_output_size, cnn_filters[-1])
```

More sophisticated fusion techniques could be explored, such as attention mechanisms or cross-modal transformers.

### 4.4 Evaluation Metrics and Validation Strategies

The project uses several metrics to evaluate model performance:

1. **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values
2. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values
3. **Root Mean Squared Error (RMSE)**: The square root of MSE, in the same units as the target variable

These metrics are calculated in the `evaluate` method of each model:

```python
mse_loss = nn.functional.mse_loss(predictions, y_tensor).item()
mae_loss = nn.functional.l1_loss(predictions, y_tensor).item()

return {
    "mse": mse_loss,
    "mae": mae_loss,
    "rmse": np.sqrt(mse_loss)
}
```

For validation, the project uses:

1. **Leave-One-Subject-Out (LOSO) cross-validation**: Training on all subjects except one, then testing on the left-out subject
2. **K-fold cross-validation**: Splitting the data into k folds and using each fold as a test set once
3. **Early stopping**: Monitoring validation loss and stopping training when it stops improving:

```python
early_stopping = EarlyStopping(
    patience=patience,
    min_delta=min_delta,
    monitor=monitor
)
```

These validation strategies help ensure that the models generalize well to new subjects and prevent overfitting.

## 5. Academic Literature Review

### 5.1 GSR Measurement and Contactless Alternatives

The field of contactless physiological monitoring has seen significant advances in recent years, but GSR remains one of the most challenging signals to measure without contact.

Jo et al. (2021) demonstrated the concept of non-contact GSR estimation using infrared imaging of the face. They found correlations between facial image intensities and GSR sensor output, suggesting that facial thermal patterns contain information about sympathetic arousal.

Gioia et al. (2021) used contactless thermal imaging to discriminate between stress and cognitive load, using GSR as a ground-truth reference. Their work supports the idea that thermal imaging can capture physiological changes related to stress.

### 5.2 Contralateral Sympathetic Responses

The "mirror effect" of contralateral sympathetic responses is based on the bilateral nature of the sympathetic nervous system. When one part of the body experiences sympathetic activation (e.g., increased sweating), similar responses occur in corresponding areas on the opposite side.

This phenomenon has been studied in the context of various physiological signals, including GSR. Research has shown that GSR measurements from left and right hands are highly correlated, supporting the project's approach of measuring GSR on one hand while recording video of the other.

### 5.3 Multi-ROI Approaches in Physiological Sensing

The Multi-ROI approach used in this project is inspired by research in remote photoplethysmography (rPPG), where multiple regions of the face or body are analyzed to extract physiological signals.

Chen et al. (2024) reviewed deep learning approaches for contactless physiological measurement, highlighting the importance of selecting appropriate regions of interest for signal extraction. They noted that different regions may contain complementary information, supporting the project's use of multiple ROIs.

### 5.4 Deep Learning for Multimodal Fusion

Multimodal fusion is an active area of research in deep learning. Various approaches have been proposed for combining information from different modalities:

1. **Feature-level fusion**: Combining features from different modalities before classification or regression
2. **Decision-level fusion**: Making separate predictions for each modality and then combining them
3. **Model-level fusion**: Using specialized architectures that process each modality separately but share information

The dual-stream architecture used in this project is an example of model-level fusion, where separate CNN streams process RGB and thermal data before fusion.

Recent advances in multimodal transformers and cross-attention mechanisms could potentially improve the fusion of RGB and thermal features in this project.

## 6. Recommendations for Improvement

Based on the analysis of the current implementation and recent advances in the field, several improvements could be considered:

### 6.1 Measurement and Data Collection

1. **Hardware synchronization**: Implement hardware-level synchronization between cameras and GSR sensors using trigger signals for more precise temporal alignment.

2. **Expanded dataset**: Collect data from a more diverse set of participants, covering different skin tones, ages, and stress conditions to improve generalization.

3. **Additional physiological signals**: Include other contactless physiological measurements (e.g., heart rate, respiration) to provide a more comprehensive picture of autonomic arousal.

### 6.2 Data Processing and Feature Engineering

1. **Advanced ROI selection**: Implement adaptive ROI selection based on physiological significance and signal quality, rather than using fixed landmarks.

2. **Deep feature extraction**: Use pre-trained CNNs (e.g., ResNet, EfficientNet) for feature extraction from ROIs instead of simple color averaging.

3. **Temporal feature engineering**: Extract temporal features (e.g., frequency domain features, wavelet coefficients) from the ROI signals to capture dynamic patterns.

### 6.3 Neural Network Architectures

1. **Attention mechanisms**: Incorporate attention mechanisms in the dual-stream model to focus on the most informative regions and time points.

2. **Cross-modal transformers**: Implement cross-modal transformer architectures that can better capture interactions between RGB and thermal modalities.

3. **Graph neural networks**: Use graph neural networks to model the spatial relationships between different hand landmarks and ROIs.

4. **Ensemble methods**: Combine predictions from multiple models (e.g., LSTM, Transformer, ResNet) to improve robustness and accuracy.

### 6.4 Evaluation and Validation

1. **Physiological validation**: Validate the model's predictions against other physiological measures of stress (e.g., cortisol levels, heart rate variability).

2. **Real-world testing**: Evaluate the system in real-world scenarios with natural movements and lighting conditions.

3. **Longitudinal studies**: Conduct longitudinal studies to assess the stability of the model's predictions over time and across different contexts.

## 7. Conclusion

The GSR-RGBT project represents a significant step forward in contactless physiological monitoring. By combining RGB and thermal video with advanced machine learning techniques, it aims to predict GSR without physical contact, opening new possibilities for stress and emotion monitoring in various applications.

The project's innovative use of the "mirror effect" of contralateral sympathetic responses, along with the Multi-ROI approach and dual-stream neural network architecture, provides a solid foundation for contactless GSR prediction. The comprehensive data processing pipeline and flexible model architecture enable accurate and robust predictions.

While there are opportunities for improvement in various aspects of the system, the current implementation demonstrates the feasibility of contactless GSR prediction and establishes a framework for future research in this area.

## References

1. Jo, G., Lee, S., & Lee, E. C. (2021). A Study on the Possibility of Measuring the Non-contact Galvanic Skin Response Based on Near-Infrared Imaging. In Int. Conf. Intelligent Human Computer Interaction (IHCI), 110-119.

2. Gioia, F., Pascali, M. A., Greco, A., Colantonio, S., & Scilingo, E. P. (2021). Discriminating Stress From Cognitive Load Using Contactless Thermal Imaging Devices. In IEEE EMBC, 608-611.

3. Chen, W., et al. (2024). Deep Learning and Remote Photoplethysmography Powered Advancements in Contactless Physiological Measurement. Front. Med., 11.

4. Huang, B., et al. (2023). Challenges and Prospects of Visual Contactless Physiological Monitoring in Clinical Study. npj Digit. Med., 6, Article 231.

5. Nechyporenko, A., et al. (2024). Galvanic Skin Response and Photoplethysmography for Stress Recognition Using Machine Learning and Wearable Sensors. Appl. Sci., 14(24), Article 11997.

6. Al-Nafjan, A., & Aldayel, M. (2023). Anxiety Detection System Based on Galvanic Skin Response Signals. Appl. Sci., 14(23), Article 10788.

7. Tagnithammou, T., Monacelli, E., Ferszterowski, A., & Trénor, L. (2021). Emotional State Detection on Mobility Vehicle Using Camera: Feasibility and Evaluation Study. Proc. Int. Symp. Affective Comput. Intell. Interact. (ACII).