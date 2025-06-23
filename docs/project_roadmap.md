# GSR-RGBT Project Roadmap

## Introduction

This comprehensive roadmap provides a complete picture of the GSR-RGBT project's trajectory, combining planned engineering enhancements with scientific exploration directions. It serves as a unified vision for the project's evolution, encompassing both technical improvements and research opportunities that will advance the field of contactless physiological monitoring.

The roadmap is organized into two main sections: **Planned Improvements** focusing on engineering enhancements and system development, and **Future Research Directions** exploring scientific opportunities and cutting-edge approaches.

---

# Planned Improvements

## Introduction

This section outlines the planned engineering improvements for the GSR-RGBT (Galvanic Skin Response - RGB-Thermal) project. These improvements are based on a thorough analysis of the current implementation, recent enhancements, and identified opportunities for further development. The goal is to provide a clear roadmap for enhancing the project's capabilities, robustness, and usability.

## 1. Data Acquisition and Synchronization

### 1.1 Hardware Synchronization Implementation

**Current State:** The project currently uses a software-based approach for synchronizing data from different sources (RGB video, thermal video, and GSR sensor). While recent improvements have added a centralized timestamp authority, hardware synchronization would provide more precise alignment.

**Proposed Improvements:**
- Implement the Arduino-based LED flash system for precise synchronization
- Create a hardware trigger mechanism that can be detected in all data streams
- Develop a calibration procedure to measure and account for device-specific latencies

**Rationale:** Hardware synchronization will significantly improve the temporal alignment of multimodal data, which is critical for accurate GSR prediction from video streams. This addresses one of the key challenges identified in the timestamp synchronization research.

### 1.2 Enhanced Real-time Visualization

**Current State:** The GUI has been enhanced to include real-time visualization of GSR data, but there's potential for more comprehensive visualization of all data streams.

**Proposed Improvements:**
- Add real-time visualization of extracted ROI features from video streams
- Implement split-screen views showing the detected hand landmarks and ROIs
- Create a dashboard-style interface with customizable visualization panels
- Add real-time quality indicators for signal strength and synchronization status

**Rationale:** Enhanced visualization will provide immediate feedback during data collection, helping researchers identify and address issues promptly. This will improve the quality of collected data and reduce the need for post-collection data cleaning.

## 2. Data Processing Pipeline

### 2.1 Further Modularization of Feature Engineering

**Current State:** The feature engineering pipeline has been improved to better utilize the Multi-ROI approach and avoid data leakage, but there's room for further modularization.

**Proposed Improvements:**
- Refactor the feature extraction code into a plugin-based architecture
- Create a standardized interface for feature extractors
- Implement additional feature extraction methods (e.g., optical flow, texture analysis)
- Develop a feature selection module to automatically identify the most informative features

**Rationale:** A more modular feature engineering pipeline will make it easier to experiment with different feature extraction methods and adapt to new research questions. This addresses the future consideration mentioned in implementation notes.

### 2.2 Improved Thermal Data Utilization

**Current State:** Support for thermal video in dual-stream models has been added, and the MMRPhysProcessor has been integrated to extract physiological signals from RGB and thermal videos. However, the thermal data processing could be further enhanced.

**Proposed Improvements:**
- Implement specialized preprocessing techniques for thermal imagery
- Develop thermal-specific feature extractors that leverage temperature gradients
- Create fusion methods that optimally combine information from RGB and thermal streams
- Research and implement temperature calibration for more accurate thermal readings
- Extend the MMRPhysProcessor to better utilize thermal data for physiological signal extraction
- Implement cross-modal attention mechanisms for RGB-thermal fusion in the MMRPhysProcessor

**Rationale:** Thermal data provides unique physiological information that complements RGB data. Better utilization of thermal data could significantly improve GSR prediction accuracy. The MMRPhysProcessor provides a foundation for advanced physiological signal extraction, which can be further enhanced with thermal-specific techniques.

## 3. Machine Learning Models

### 3.1 Advanced Model Architectures

**Current State:** The project supports several model architectures (LSTM, Autoencoder, VAE, CNN, CNN-LSTM, Transformer, ResNet), but there's potential for exploring more advanced architectures.

**Proposed Improvements:**
- Implement attention mechanisms for better temporal modeling
- Develop multimodal fusion architectures specifically designed for RGB-T data
- Explore graph neural networks for modeling relationships between different ROIs
- Implement contrastive learning approaches for better feature representations

**Rationale:** Advanced model architectures could better capture the complex relationships between video features and GSR signals, potentially improving prediction accuracy.

### 3.2 Experiment Tracking Integration

**Current State:** The project has improved model run ID extraction in experiment comparison, but a more comprehensive experiment tracking solution would be beneficial.

**Proposed Improvements:**
- Integrate a dedicated experiment tracking tool like MLflow or Weights & Biases
- Implement automatic logging of hyperparameters, metrics, and artifacts
- Create a web-based dashboard for comparing experiments
- Develop a standardized reporting system for experiment results

**Rationale:** Proper experiment tracking will make it easier to organize, compare, and reproduce experiments. This addresses the future consideration mentioned in implementation notes.

## 4. Testing and Validation

### 4.1 Automated Testing Framework

**Current State:** The project has implemented a comprehensive automated testing framework with unit, smoke, and regression tests for core components, including the newly added MMRPhysProcessor. This framework helps ensure code quality and prevent regressions.

**Proposed Improvements:**
- Expand test coverage to include more edge cases and error conditions
- Implement property-based testing for complex algorithms
- Add performance benchmarks to regression tests
- Implement continuous integration to automatically run tests on code changes
- Create visual regression tests for GUI components
- Develop automated end-to-end tests with real hardware

**Rationale:** While the current testing framework provides good coverage, expanding it further will help ensure that the codebase remains robust as it evolves. Continuous integration will automate the testing process, making it easier to catch issues early.

### 4.2 Validation Tools for Synchronization

**Current State:** While the project has improved data synchronization, tools for quantitatively assessing synchronization accuracy are needed.

**Proposed Improvements:**
- Develop metrics for measuring synchronization accuracy
- Create visualization tools for inspecting temporal alignment
- Implement automated detection of synchronization issues
- Design experiments to validate synchronization methods

**Rationale:** Validation tools will help ensure that the synchronization methods are working correctly and provide a way to compare different approaches.

## 5. User Experience and Documentation

### 5.1 Enhanced User Interface

**Current State:** The GUI has been improved with real-time visualization and better organization, but there's potential for further enhancements.

**Proposed Improvements:**
- Implement a wizard-style interface for guiding users through the data collection process
- Create a session management system for organizing recordings
- Add user authentication and role-based access control
- Develop a more intuitive interface for configuring hardware settings

**Rationale:** An enhanced user interface will make the application more accessible to researchers with varying levels of technical expertise.

### 5.2 Comprehensive Documentation

**Current State:** The project has good documentation of recent improvements and the project's evolution, but a more comprehensive documentation system would be beneficial.

**Proposed Improvements:**
- Create a user manual with step-by-step instructions for common tasks
- Develop API documentation for all modules and classes
- Implement interactive tutorials for new users
- Create a knowledge base for troubleshooting common issues

**Rationale:** Comprehensive documentation will make it easier for new users to get started with the project and for existing users to make the most of its capabilities.

## 6. Deployment and Scalability

### 6.1 Containerization and Cloud Deployment

**Current State:** The project is designed for local deployment, but containerization and cloud deployment would enhance scalability and reproducibility.

**Proposed Improvements:**
- Create Docker containers for the application and its dependencies
- Develop Kubernetes configurations for cloud deployment
- Implement cloud storage integration for data and models
- Create a web-based interface for remote access

**Rationale:** Containerization and cloud deployment will make it easier to scale the project and collaborate with researchers at different institutions.

### 6.2 Performance Optimization

**Current State:** The project has implemented Cython optimizations for some components, but further performance improvements are possible.

**Proposed Improvements:**
- Profile the application to identify performance bottlenecks
- Implement GPU acceleration for more components
- Optimize memory usage for handling large datasets
- Develop distributed processing capabilities for parallel computation

**Rationale:** Performance optimization will enable the project to handle larger datasets and more complex models, enhancing its research capabilities.

## 7. Research Extensions

### 7.1 Multi-subject Analysis

**Current State:** The project focuses on individual subject analysis, but multi-subject analysis would enable broader research questions. The integration of the MMRPhysProcessor provides additional physiological signals that could be valuable for multi-subject analysis.

**Proposed Improvements:**
- Develop methods for normalizing data across subjects
- Implement transfer learning approaches for adapting models to new subjects
- Create visualization tools for comparing results across subjects
- Research population-level patterns in GSR responses
- Investigate how MMRPhys-extracted physiological signals vary across different subjects
- Develop subject-invariant features using the combined RGB, thermal, and MMRPhys signals

**Rationale:** Multi-subject analysis will enable researchers to identify common patterns and individual differences in GSR responses. The additional physiological signals from the MMRPhysProcessor could provide valuable insights into individual differences and common patterns across subjects.

### 7.2 Real-time GSR Prediction

**Current State:** The project currently focuses on offline analysis, but real-time GSR prediction would open up new application areas.

**Proposed Improvements:**
- Optimize the pipeline for real-time processing
- Implement streaming data handling
- Develop lightweight models suitable for real-time inference
- Create a real-time feedback system based on GSR predictions

**Rationale:** Real-time GSR prediction would enable applications in areas such as affective computing, human-computer interaction, and biofeedback.

## Implementation Timeline

The proposed improvements are organized into short-term (1-3 months), medium-term (3-6 months), and long-term (6-12 months) goals:

### Short-term Goals (1-3 months)
- Hardware synchronization implementation
- Enhanced real-time visualization
- Automated testing framework
- Validation tools for synchronization

### Medium-term Goals (3-6 months)
- Further modularization of feature engineering
- Improved thermal data utilization
- Experiment tracking integration
- Enhanced user interface
- Comprehensive documentation

### Long-term Goals (6-12 months)
- Advanced model architectures
- Containerization and cloud deployment
- Performance optimization
- Multi-subject analysis
- Real-time GSR prediction

---

# Future Research Directions

## Introduction

This section outlines the future research directions for the GSR-RGBT (Galvanic Skin Response - RGB-Thermal) project. It identifies promising areas for further investigation, references relevant current research, and discusses bleeding-edge approaches that could advance the field of contactless physiological monitoring. The goal is to provide a roadmap for future research that builds on the project's current foundation.

## 1. Advanced Contactless GSR Estimation Techniques

### 1.1 Thermal-Based GSR Estimation

**Current Research Context:**
Jo et al. (2021) demonstrated the concept of non-contact GSR estimation using infrared imaging of the face, finding correlations between facial image intensities and GSR sensor output. Gioia et al. (2021) used contactless thermal imaging to discriminate between stress and cognitive load, using GSR as a ground-truth reference.

**Research Directions:**
- Investigate the relationship between thermal patterns on the hand and GSR measurements
- Develop thermal-specific features that directly correlate with sympathetic arousal
- Research the impact of ambient temperature and individual differences on thermal signatures
- Explore the use of thermal gradient analysis for improved GSR prediction

**Potential Impact:**
Advancing thermal-based GSR estimation could lead to more accurate contactless measurements, potentially eliminating the need for RGB video in some applications. This would simplify the hardware requirements and potentially improve performance in varying lighting conditions.

### 1.2 Multi-Spectral Imaging for GSR Estimation

**Current Research Context:**
Recent advances in multi-spectral imaging have shown promise for various physiological measurements. Huang et al. (2023) reviewed visual contactless physiological monitoring techniques, highlighting the potential of multi-spectral approaches for capturing physiological signals that are not visible in standard RGB or thermal imaging.

**Research Directions:**
- Investigate specific spectral bands that correlate with sympathetic arousal
- Develop multi-spectral imaging setups optimized for GSR estimation
- Research fusion techniques for combining information from different spectral bands
- Explore the use of hyperspectral imaging for more detailed physiological information

**Potential Impact:**
Multi-spectral imaging could provide access to physiological information not available in RGB or thermal video alone, potentially improving the accuracy of contactless GSR estimation and enabling the detection of more subtle changes in sympathetic arousal.

## 2. Advanced Machine Learning Approaches

### 2.1 Cross-Modal Attention Mechanisms

**Current Research Context:**
Attention mechanisms have revolutionized natural language processing and computer vision, and are increasingly being applied to multimodal data fusion. Recent research has shown the effectiveness of cross-modal attention for aligning and integrating information from different modalities.

**Research Directions:**
- Develop cross-modal attention mechanisms for RGB-thermal fusion
- Investigate temporal attention for capturing dynamic patterns in physiological signals
- Research spatial attention for focusing on the most informative regions of the hand
- Explore hierarchical attention mechanisms for multi-level feature integration

**Potential Impact:**
Cross-modal attention mechanisms could significantly improve the fusion of RGB and thermal data, enabling the model to focus on the most relevant features in each modality and better capture the complex relationships between visual cues and GSR.

### 2.2 Self-Supervised Learning for Physiological Signal Prediction

**Current Research Context:**
Self-supervised learning has emerged as a powerful paradigm for learning representations from unlabeled data. This approach could be particularly valuable for physiological signal prediction, where labeled data is often limited.

**Research Directions:**
- Develop self-supervised pretraining tasks for RGB and thermal hand videos
- Investigate contrastive learning approaches for learning physiological signal representations
- Research temporal consistency objectives for capturing physiological dynamics
- Explore cross-modal self-supervised learning for RGB-thermal alignment

**Potential Impact:**
Self-supervised learning could enable more effective use of unlabeled data, potentially improving model performance when labeled data is limited. This could make the system more practical for real-world applications where collecting ground-truth GSR data is challenging.

### 2.3 Federated Learning for Privacy-Preserving GSR Prediction

**Current Research Context:**
Federated learning allows models to be trained across multiple devices or servers while keeping the data localized, addressing privacy concerns in sensitive applications like physiological monitoring.

**Research Directions:**
- Develop federated learning protocols for GSR prediction models
- Investigate privacy-preserving techniques for sharing physiological data
- Research personalization methods in federated learning settings
- Explore the trade-offs between privacy, performance, and computational requirements

**Potential Impact:**
Federated learning could enable the development of more robust GSR prediction models by leveraging data from multiple sources without compromising privacy. This could be particularly important for applications in healthcare or affective computing.

## 3. Physiological Validation and Applications

### 3.1 Comprehensive Physiological Validation

**Current Research Context:**
Nechyporenko et al. (2024) and Al-Nafjan & Aldayel (2023) have conducted studies on GSR and other physiological signals for stress and anxiety detection, highlighting the importance of comprehensive validation against multiple physiological measures.

**Research Directions:**
- Conduct studies comparing contactless GSR predictions with multiple physiological measures
- Investigate the correlation between contactless GSR and other stress indicators (cortisol, heart rate variability)
- Research the impact of individual differences on GSR prediction accuracy
- Explore the temporal dynamics of GSR in relation to other physiological signals

**Potential Impact:**
Comprehensive physiological validation would establish the scientific validity of contactless GSR prediction, potentially leading to wider acceptance in research and clinical applications. It would also provide insights into the relationship between different physiological measures of stress and emotion.

### 3.2 Real-World Applications and Ecological Validity

**Current Research Context:**
Tagnithammou et al. (2021) conducted a feasibility study on emotional state detection using cameras in mobility vehicles, highlighting the challenges and potential of real-world applications of contactless physiological monitoring.

**Research Directions:**
- Develop and test GSR prediction systems in real-world environments (e.g., driving, workplace)
- Investigate the impact of natural movements and lighting conditions on prediction accuracy
- Research methods for adapting to changing environmental conditions
- Explore applications in stress management, human-computer interaction, and healthcare

**Potential Impact:**
Research on real-world applications would bridge the gap between laboratory studies and practical implementations, potentially leading to new applications of contactless GSR prediction in fields such as automotive safety, workplace wellness, and mental health monitoring.

## 4. Multi-Modal and Multi-Signal Approaches

### 4.1 Integration with Other Contactless Physiological Measurements

**Current Research Context:**
Chen et al. (2024) reviewed deep learning approaches for contactless physiological measurement, highlighting the potential of combining multiple physiological signals for more comprehensive monitoring.

**Research Directions:**
- Develop systems that combine contactless GSR with remote photoplethysmography (rPPG)
- Investigate the integration of respiratory monitoring through video analysis
- Research multi-signal fusion techniques for comprehensive stress assessment
- Explore the use of facial expression and body posture as complementary signals

**Potential Impact:**
Integrating multiple contactless physiological measurements could provide a more comprehensive picture of emotional and physiological state, potentially improving the accuracy and robustness of stress and emotion detection systems.

### 4.2 Contextual and Behavioral Integration

**Current Research Context:**
Recent research has highlighted the importance of contextual and behavioral information for interpreting physiological signals. Understanding the context in which physiological changes occur can provide valuable insights into their meaning and significance.

**Research Directions:**
- Develop methods for integrating contextual information with physiological signals
- Investigate the relationship between hand movements and GSR responses
- Research the impact of social context on physiological responses
- Explore the use of multimodal context (audio, video, text) for interpreting GSR

**Potential Impact:**
Integrating contextual and behavioral information could lead to more nuanced interpretations of GSR signals, potentially enabling more accurate assessment of emotional states and stress levels in complex real-world situations.

## 5. Longitudinal and Population-Level Research

### 5.1 Longitudinal Studies of GSR Patterns

**Current Research Context:**
Most current research on GSR focuses on short-term measurements in controlled settings. Longitudinal studies could provide valuable insights into how GSR patterns change over time and across different contexts.

**Research Directions:**
- Conduct longitudinal studies of GSR patterns over days, weeks, or months
- Investigate the stability and variability of GSR responses over time
- Research the impact of habituation and adaptation on GSR
- Explore the relationship between long-term GSR patterns and health outcomes

**Potential Impact:**
Longitudinal research could provide insights into the temporal dynamics of GSR and its relationship to long-term stress and health outcomes. This could lead to new applications in preventive healthcare and wellness monitoring.

### 5.2 Population-Level Analysis and Individual Differences

**Current Research Context:**
Individual differences in GSR responses are well-documented but not fully understood. Population-level analysis could help identify patterns and factors that influence GSR responses across different individuals and groups.

**Research Directions:**
- Develop methods for normalizing GSR data across individuals
- Investigate demographic and physiological factors that influence GSR responses
- Research cultural and contextual influences on GSR patterns
- Explore the use of population-level models with individual adaptation

**Potential Impact:**
Population-level analysis could lead to more personalized and accurate GSR prediction models that account for individual differences. This could improve the applicability of contactless GSR prediction across diverse populations and contexts.

## 6. Ethical and Privacy Considerations

### 6.1 Ethical Framework for Contactless Physiological Monitoring

**Current Research Context:**
As contactless physiological monitoring technologies advance, ethical considerations become increasingly important. Issues of consent, privacy, and potential misuse need to be addressed proactively.

**Research Directions:**
- Develop ethical guidelines for contactless physiological monitoring
- Investigate user perceptions and concerns regarding contactless GSR prediction
- Research methods for ensuring informed consent in various applications
- Explore the ethical implications of emotion detection in different contexts

**Potential Impact:**
Developing a robust ethical framework would help ensure that contactless GSR prediction technologies are developed and deployed in ways that respect privacy, autonomy, and human dignity. This could increase user acceptance and trust in these technologies.

### 6.2 Privacy-Preserving Techniques for Physiological Data

**Current Research Context:**
Physiological data is highly sensitive and personal. Privacy-preserving techniques are essential for protecting this data while still enabling useful applications.

**Research Directions:**
- Develop on-device processing methods that avoid storing raw video data
- Investigate differential privacy approaches for physiological data
- Research anonymization techniques for GSR and related signals
- Explore the trade-offs between privacy, utility, and performance

**Potential Impact:**
Privacy-preserving techniques could enable the widespread adoption of contactless GSR prediction while protecting user privacy. This could be particularly important for applications in healthcare, workplace monitoring, and public spaces.

## 7. Technical Advancements

### 7.1 Hardware Innovations for Improved Sensing

**Current Research Context:**
Advances in camera technology, thermal imaging, and computational capabilities continue to create new opportunities for contactless physiological monitoring.

**Research Directions:**
- Investigate the use of depth cameras for improved hand tracking and ROI detection
- Research low-cost thermal imaging solutions for wider accessibility
- Explore the potential of event-based cameras for high-temporal-resolution capture
- Develop specialized hardware configurations optimized for GSR prediction

**Potential Impact:**
Hardware innovations could improve the accuracy, reliability, and accessibility of contactless GSR prediction systems. Lower-cost solutions could enable wider adoption, while specialized hardware could improve performance in specific applications.

### 7.2 Edge Computing for Real-Time Processing

**Current Research Context:**
Edge computing is increasingly enabling sophisticated processing on resource-constrained devices. This could be particularly valuable for real-time contactless GSR prediction in mobile or embedded applications.

**Research Directions:**
- Develop efficient model architectures for edge deployment
- Investigate model compression techniques for resource-constrained devices
- Research hardware-software co-design for optimized performance
- Explore the use of specialized neural processing units for physiological signal processing

**Potential Impact:**
Edge computing solutions could enable real-time contactless GSR prediction on mobile devices, wearables, or embedded systems. This could open up new applications in mobile health, automotive safety, and ubiquitous computing.

---

# Integrated Roadmap and Priorities

## Strategic Alignment

The planned improvements and future research directions are strategically aligned to create a comprehensive advancement of the GSR-RGBT project. The engineering improvements provide the foundation and infrastructure necessary to support advanced research, while the research directions push the boundaries of what's possible in contactless physiological monitoring.

### Synergistic Opportunities

Several areas where planned improvements and research directions complement each other:

1. **Real-time Processing**: Engineering improvements in performance optimization and edge computing directly support research into real-time GSR prediction and ecological validity studies.

2. **Multi-modal Integration**: Technical improvements in thermal data utilization and advanced model architectures enable research into multi-spectral imaging and cross-modal attention mechanisms.

3. **Privacy and Ethics**: Engineering work on containerization and cloud deployment must be aligned with research into privacy-preserving techniques and ethical frameworks.

4. **Validation and Testing**: Improvements in automated testing frameworks support research into comprehensive physiological validation and longitudinal studies.

## Priority Matrix

### High Priority (Immediate Focus)
- Hardware synchronization implementation
- Enhanced real-time visualization
- Thermal-based GSR estimation research
- Cross-modal attention mechanisms
- Comprehensive physiological validation

### Medium Priority (6-12 months)
- Advanced model architectures
- Multi-spectral imaging research
- Self-supervised learning approaches
- Real-world applications research
- Performance optimization

### Long-term Priority (12+ months)
- Containerization and cloud deployment
- Federated learning research
- Longitudinal studies
- Edge computing solutions
- Ethical framework development

## Resource Allocation

### Technical Development (60%)
- Software engineering improvements
- Algorithm development
- System optimization
- Testing and validation

### Research Activities (30%)
- Experimental studies
- Algorithm research
- Validation studies
- Publication and dissemination

### Infrastructure and Support (10%)
- Documentation
- User interface improvements
- Community building
- Collaboration tools

## Success Metrics

### Technical Metrics
- Synchronization accuracy improvements
- Model performance gains
- Real-time processing capabilities
- System reliability and robustness

### Research Metrics
- Publication of research findings
- Validation against gold standards
- Novel algorithm development
- Community adoption and impact

### User Experience Metrics
- Ease of use improvements
- Documentation completeness
- User satisfaction
- Adoption by research community

---

# Conclusion

This comprehensive roadmap provides a unified vision for the GSR-RGBT project's evolution, combining planned engineering enhancements with scientific exploration. The integration of technical improvements and research directions creates a synergistic approach that will advance both the practical capabilities of the system and the scientific understanding of contactless physiological monitoring.

The roadmap emphasizes the importance of:

1. **Technical Excellence**: Continuous improvement of the system's capabilities, performance, and reliability
2. **Scientific Rigor**: Pursuing cutting-edge research that advances the field
3. **Practical Impact**: Developing solutions that can be applied in real-world scenarios
4. **Ethical Responsibility**: Ensuring that advances are made with consideration for privacy, consent, and human dignity

By following this roadmap, the GSR-RGBT project can continue to push the boundaries of contactless physiological monitoring while maintaining its commitment to technical excellence and scientific integrity. The project's evolution will contribute to advancing the field and enabling new applications in healthcare, human-computer interaction, and affective computing.

Regular reviews and updates of this roadmap will ensure it remains aligned with evolving project requirements, technological advancements, and research opportunities. Collaboration with the broader research community will be essential for achieving these ambitious goals and maximizing the project's impact.

## References

1. Jo, G., Lee, S., & Lee, E. C. (2021). A Study on the Possibility of Measuring the Non-contact Galvanic Skin Response Based on Near-Infrared Imaging. In Int. Conf. Intelligent Human Computer Interaction (IHCI), 110-119.

2. Gioia, F., Pascali, M. A., Greco, A., Colantonio, S., & Scilingo, E. P. (2021). Discriminating Stress From Cognitive Load Using Contactless Thermal Imaging Devices. In IEEE EMBC, 608-611.

3. Chen, W., et al. (2024). Deep Learning and Remote Photoplethysmography Powered Advancements in Contactless Physiological Measurement. Front. Med., 11.

4. Huang, B., et al. (2023). Challenges and Prospects of Visual Contactless Physiological Monitoring in Clinical Study. npj Digit. Med., 6, Article 231.

5. Nechyporenko, A., et al. (2024). Galvanic Skin Response and Photoplethysmography for Stress Recognition Using Machine Learning and Wearable Sensors. Appl. Sci., 14(24), Article 11997.

6. Al-Nafjan, A., & Aldayel, M. (2023). Anxiety Detection System Based on Galvanic Skin Response Signals. Appl. Sci., 14(23), Article 10788.

7. Tagnithammou, T., Monacelli, E., Ferszterowski, A., & Trénor, L. (2021). Emotional State Detection on Mobility Vehicle Using Camera: Feasibility and Evaluation Study. Proc. Int. Symp. Affective Comput. Intell. Interact. (ACII).