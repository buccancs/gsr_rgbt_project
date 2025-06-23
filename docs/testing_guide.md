# GSR-RGBT Project Testing Guide

## Introduction

This comprehensive guide outlines the testing strategy, execution procedures, coverage analysis, and mocking approaches for the GSR-RGBT project. It serves as a unified resource for developers to understand and implement effective testing practices throughout the project lifecycle.

The goal is to provide a complete testing framework that ensures the reliability and correctness of the codebase through multiple layers of testing, from individual component verification to end-to-end system validation.

---

# Testing Strategy

## Testing Philosophy

The GSR-RGBT project follows a multi-layered testing approach to ensure code quality and prevent regressions:

1. **Unit Tests**: Test individual components in isolation to verify their correctness.
2. **Smoke Tests**: Verify that the main functionality of the system runs without errors.
3. **Regression Tests**: Ensure that changes to the codebase don't break existing functionality.

This approach allows us to catch issues at different levels of abstraction and provides confidence in the correctness of the codebase.

## Test Organization

The tests are organized in the following directory structure:

```
src/tests/
├── unit/             # Unit tests for individual components
│   ├── capture/      # Tests for capture components
│   ├── data_collection/ # Tests for data collection components
│   ├── evaluation/   # Tests for evaluation components
│   ├── gui/          # Tests for GUI components
│   ├── system/       # Tests for system components
│   └── utils/        # Tests for utility components
├── smoke/            # Smoke tests for basic functionality
├── regression/       # Regression tests for end-to-end functionality
├── run_tests.py      # Script for running tests
└── __init__.py       # Package initialization
```

Each test file follows the naming convention `test_*.py` to ensure it's discovered by the test runner.

## Test Types

### Unit Tests

Unit tests focus on testing individual components in isolation. They verify that each component behaves correctly according to its specification. Unit tests should be:

- **Fast**: Unit tests should run quickly to provide immediate feedback.
- **Isolated**: Unit tests should not depend on external systems or other components.
- **Deterministic**: Unit tests should produce the same result every time they run.

Example unit tests include:
- Tests for the MMRPhysProcessor class that verify its initialization, frame preprocessing, and heart rate extraction functionality.
- Tests for the PyTorch models that verify their initialization, training, and prediction functionality.
- Tests for the feature engineering pipeline that verify its data processing functionality.

### Smoke Tests

Smoke tests verify that the main functionality of the system runs without errors. They don't test the correctness of the results in detail but ensure that the system doesn't "smoke" when run. Smoke tests should:

- **Cover Critical Paths**: Smoke tests should cover the most important functionality of the system.
- **Be Quick**: Smoke tests should run quickly to provide immediate feedback.
- **Be Simple**: Smoke tests should be simple and easy to understand.

Example smoke tests include:
- Tests that verify the MMRPhysProcessor can be initialized and process a frame sequence.
- Tests that verify the feature engineering pipeline can process a dataset.
- Tests that verify the model training pipeline can train a model.

### Regression Tests

Regression tests ensure that changes to the codebase don't break existing functionality. They test the system end-to-end and verify that it behaves correctly in realistic scenarios. Regression tests should:

- **Be Comprehensive**: Regression tests should cover all critical functionality of the system.
- **Use Realistic Data**: Regression tests should use data that's representative of real-world usage.
- **Be Maintainable**: Regression tests should be easy to update when the system changes.

Example regression tests include:
- Tests that verify the MMRPhysProcessor can be integrated with the feature engineering pipeline.
- Tests that verify the feature engineering pipeline can be integrated with the model training pipeline.
- Tests that verify the model training pipeline can train a model and make predictions.

---

# Test Execution

## Prerequisites

Before running tests, ensure that you have the following prerequisites installed:

1. **Python 3.8+**: The project requires Python 3.8 or higher.
2. **Required Python Packages**: Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```
3. **Test-Specific Requirements**: Some tests may require additional packages:
   ```
   pip install pytest coverage pytest-cov
   ```

## Test Environment Setup

### Setting Up the Development Environment

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/gsr_rgbt_project.git
   cd gsr_rgbt_project
   ```

2. **Create a Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   pip install -e .  # Install the package in development mode
   ```

### Setting Up Test Data

Some tests require test data. You can set up the test data as follows:

1. **Download Sample Data**:
   ```
   python scripts/download_test_data.py
   ```

2. **Generate Synthetic Data**:
   ```
   python scripts/generate_synthetic_data.py
   ```

## Running Tests

The GSR-RGBT project uses the `unittest` framework for testing. Tests can be run using the `run_tests.py` script in the `src/tests` directory.

### Running All Tests

To run all tests:
```
python src/tests/run_tests.py
```

### Running Specific Test Types

To run only unit tests:
```
python src/tests/run_tests.py --type unit
```

To run only smoke tests:
```
python src/tests/run_tests.py --type smoke
```

To run only regression tests:
```
python src/tests/run_tests.py --type regression
```

### Running Tests with Less Verbose Output

To run tests with less verbose output:
```
python src/tests/run_tests.py --quiet
```

### Running Individual Test Files

To run a specific test file:
```
python -m unittest src/tests/unit/test_mmrphys_processor.py
```

### Running Individual Test Cases

To run a specific test case:
```
python -m unittest src.tests.unit.test_mmrphys_processor.TestMMRPhysProcessor
```

### Running Individual Test Methods

To run a specific test method:
```
python -m unittest src.tests.unit.test_mmrphys_processor.TestMMRPhysProcessor.test_init_with_different_model_types
```

## Running Tests with Coverage

To run tests with coverage measurement:
```
coverage run --source=src src/tests/run_tests.py
coverage report
```

To generate an HTML coverage report:
```
coverage html
```

This will create a directory named `htmlcov` with an HTML report that you can view in a web browser.

## Continuous Integration

The project uses continuous integration to automatically run tests on code changes. The CI pipeline is configured to:

1. Run all tests on every pull request
2. Measure test coverage and report it
3. Fail the build if tests fail or if coverage drops below a threshold

You can view the CI pipeline configuration in the `.github/workflows` directory.

## Troubleshooting

### Common Issues

1. **Import Errors**: If you encounter import errors, make sure you've installed the package in development mode:
   ```
   pip install -e .
   ```

2. **Missing Dependencies**: If tests fail due to missing dependencies, make sure you've installed all required packages:
   ```
   pip install -r requirements.txt
   pip install pytest coverage pytest-cov
   ```

3. **Test Data Issues**: If tests fail due to missing test data, make sure you've set up the test data:
   ```
   python scripts/download_test_data.py
   python scripts/generate_synthetic_data.py
   ```

4. **Hardware-Related Issues**: Some tests may require specific hardware. If you don't have the required hardware, you can skip these tests:
   ```
   python src/tests/run_tests.py --skip-hardware
   ```

### Getting Help

If you encounter issues running tests, you can:

1. Check the project's issue tracker for known issues
2. Ask for help in the project's discussion forum
3. Contact the project maintainers

## Best Practices

When running tests, follow these best practices:

1. **Run Tests Regularly**: Run tests regularly during development to catch issues early.
2. **Run All Tests Before Committing**: Run all tests before committing changes to ensure you haven't broken anything.
3. **Check Coverage**: Regularly check test coverage to identify areas that need more testing.
4. **Fix Failing Tests**: Don't ignore failing tests. Fix them or update them if the expected behavior has changed.
5. **Add Tests for New Features**: Add tests for new features as you develop them.

---

# Test Coverage

## Test Coverage Overview

The GSR-RGBT project has a comprehensive test suite that covers the following components:

1. **Data Collection**: Tests for capturing data from RGB cameras, thermal cameras, and GSR sensors.
2. **Feature Engineering**: Tests for processing raw data into features for machine learning models.
3. **Machine Learning Models**: Tests for training and evaluating machine learning models.
4. **Evaluation**: Tests for evaluating the performance of the system.
5. **Utilities**: Tests for utility functions and classes used throughout the project.
6. **MMRPhysProcessor**: Tests for the MMRPhysProcessor component that extracts physiological signals from RGB and thermal videos.

## Component Coverage

### Data Collection

The data collection components are tested with the following coverage:

| Component | Unit Tests | Smoke Tests | Regression Tests |
|-----------|------------|-------------|------------------|
| RGB Camera Capture | ✓ | ✓ | ✓ |
| Thermal Camera Capture | ✓ | ✓ | ✓ |
| GSR Sensor Capture | ✓ | ✓ | ✓ |
| Data Synchronization | ✓ | ✓ | ✓ |
| GUI Components | ✓ | ✓ | ✓ |

Key aspects tested:
- Initialization of capture devices
- Frame capture and processing
- Error handling for device failures
- Data synchronization between different sources
- GUI interaction with capture devices

### Feature Engineering

The feature engineering components are tested with the following coverage:

| Component | Unit Tests | Smoke Tests | Regression Tests |
|-----------|------------|-------------|------------------|
| Data Loading | ✓ | ✓ | ✓ |
| Signal Processing | ✓ | ✓ | ✓ |
| Feature Extraction | ✓ | ✓ | ✓ |
| Feature Selection | ✓ | ✓ | ✓ |
| Data Windowing | ✓ | ✓ | ✓ |

Key aspects tested:
- Loading data from different sources
- Processing raw signals (filtering, normalization)
- Extracting features from processed signals
- Selecting relevant features for model training
- Creating windowed data for sequence models

### Machine Learning Models

The machine learning model components are tested with the following coverage:

| Component | Unit Tests | Smoke Tests | Regression Tests |
|-----------|------------|-------------|------------------|
| LSTM Models | ✓ | ✓ | ✓ |
| Autoencoder Models | ✓ | ✓ | ✓ |
| VAE Models | ✓ | ✓ | ✓ |
| CNN Models | ✓ | ✓ | ✓ |
| CNN-LSTM Models | ✓ | ✓ | ✓ |
| Transformer Models | ✓ | ✓ | ✓ |
| ResNet Models | ✓ | ✓ | ✓ |
| Model Registry | ✓ | ✓ | ✓ |

Key aspects tested:
- Model initialization with different configurations
- Model training with synthetic data
- Model evaluation with synthetic data
- Model saving and loading
- Model prediction with new data
- Model registry for creating models by name

### Evaluation

The evaluation components are tested with the following coverage:

| Component | Unit Tests | Smoke Tests | Regression Tests |
|-----------|------------|-------------|------------------|
| Metrics Calculation | ✓ | ✓ | ✓ |
| Visualization | ✓ | ✓ | ✓ |
| Experiment Comparison | ✓ | ✓ | ✓ |

Key aspects tested:
- Calculation of evaluation metrics (MSE, MAE, etc.)
- Generation of visualizations (plots, graphs)
- Comparison of results from different experiments

### Utilities

The utility components are tested with the following coverage:

| Component | Unit Tests | Smoke Tests | Regression Tests |
|-----------|------------|-------------|------------------|
| Logging | ✓ | ✓ | ✓ |
| Configuration | ✓ | ✓ | ✓ |
| File I/O | ✓ | ✓ | ✓ |
| Data Structures | ✓ | ✓ | ✓ |

Key aspects tested:
- Logging functionality with different log levels
- Configuration loading and validation
- File I/O operations (reading, writing)
- Custom data structures and their operations

### MMRPhysProcessor

The MMRPhysProcessor component is tested with the following coverage:

| Component | Unit Tests | Smoke Tests | Regression Tests |
|-----------|------------|-------------|------------------|
| Initialization | ✓ | ✓ | ✓ |
| Frame Preprocessing | ✓ | ✓ | ✓ |
| Frame Sequence Processing | ✓ | ✓ | ✓ |
| Heart Rate Extraction | ✓ | ✓ | ✓ |
| Video Processing | ✓ | ✓ | ✓ |
| Result Saving | ✓ | ✓ | ✓ |

Key aspects tested:
- Initialization with different model types
- Preprocessing of video frames
- Processing sequences of frames
- Extraction of heart rate from pulse signals
- Processing entire videos
- Saving results to files

## Test Coverage Gaps

While the project has good test coverage overall, there are some areas that could benefit from additional testing:

1. **Edge Cases**: Some components could benefit from more thorough testing of edge cases and error conditions.
2. **Performance Testing**: The project lacks dedicated performance tests to ensure that components meet performance requirements.
3. **Integration with Hardware**: Testing with actual hardware devices is limited and could be expanded.
4. **Long-Term Stability**: The project lacks tests for long-term stability and resource usage.
5. **Cross-Platform Testing**: Testing on different platforms (Windows, Linux, macOS) is limited.

## Test Coverage Improvement Plan

To address the identified gaps in test coverage, the following improvements are planned:

1. **Add Edge Case Tests**: Add more tests for edge cases and error conditions, particularly for the MMRPhysProcessor and feature engineering components.
2. **Implement Performance Tests**: Add performance tests to ensure that components meet performance requirements, particularly for real-time processing.
3. **Expand Hardware Integration Tests**: Add more tests with actual hardware devices, using mocking where necessary.
4. **Add Stability Tests**: Add tests for long-term stability and resource usage, particularly for components that run for extended periods.
5. **Implement Cross-Platform Tests**: Add tests that run on different platforms to ensure cross-platform compatibility.

---

# Test Mocking

## Introduction

This section outlines the mocking strategy for the GSR-RGBT project, including how to mock external dependencies such as hardware devices, third-party libraries, and system resources. The goal is to provide a consistent approach to mocking that makes tests reliable, fast, and independent of external factors.

## Why Mock?

Mocking is essential for effective testing in the GSR-RGBT project for several reasons:

1. **Hardware Independence**: The project interacts with various hardware devices (RGB cameras, thermal cameras, GSR sensors) that may not be available in all development environments.
2. **Reproducibility**: Mocking ensures that tests produce the same results regardless of the environment they run in.
3. **Speed**: Tests that interact with real hardware or external services can be slow. Mocking makes tests run faster.
4. **Isolation**: Mocking allows testing components in isolation, making it easier to identify the source of issues.
5. **Control**: Mocking gives precise control over the behavior of dependencies, allowing testing of edge cases and error conditions.

## Mocking Tools

The GSR-RGBT project uses the following tools for mocking:

1. **unittest.mock**: The primary mocking library, part of the Python standard library.
2. **MagicMock**: A subclass of Mock with default implementations of most magic methods.
3. **patch**: A decorator or context manager for replacing objects with mocks during a test.
4. **mock_open**: A helper function for mocking the built-in `open` function.

## Mocking Strategies

### Hardware Devices

#### Cameras (RGB and Thermal)

Cameras are mocked using the `unittest.mock` library to replace the `cv2.VideoCapture` class:

```python
@patch('cv2.VideoCapture')
def test_camera_capture(self, mock_video_capture):
    # Setup mock
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    
    # Test code that uses cv2.VideoCapture
    ...
```

For thermal cameras, additional mocking may be needed for specific thermal camera libraries:

```python
@patch('thermal_camera_library.ThermalCamera')
def test_thermal_camera_capture(self, mock_thermal_camera):
    # Setup mock
    mock_cam = MagicMock()
    mock_thermal_camera.return_value = mock_cam
    mock_cam.get_frame.return_value = np.zeros((240, 320), dtype=np.float32)
    
    # Test code that uses thermal_camera_library.ThermalCamera
    ...
```

#### GSR Sensors

GSR sensors are mocked using the `unittest.mock` library to replace the sensor interface:

```python
@patch('shimmer.Shimmer3')
def test_gsr_sensor_capture(self, mock_shimmer):
    # Setup mock
    mock_sensor = MagicMock()
    mock_shimmer.return_value = mock_sensor
    mock_sensor.connect.return_value = True
    mock_sensor.get_data.return_value = {
        'timestamp': [1, 2, 3],
        'GSR': [0.1, 0.2, 0.3]
    }
    
    # Test code that uses shimmer.Shimmer3
    ...
```

### Third-Party Libraries

#### PyTorch

PyTorch models and functions are mocked to avoid the need for GPU resources and to make tests faster:

```python
@patch('torch.load')
def test_model_loading(self, mock_torch_load):
    # Setup mock
    mock_torch_load.return_value = {'state_dict': {}}
    
    # Test code that uses torch.load
    ...
```

For PyTorch models, create mock instances with the necessary methods:

```python
@patch('src.ml_models.pytorch_models.PyTorchLSTMModel')
def test_model_training(self, mock_model_class):
    # Setup mock
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.fit.return_value = {"train_loss": [0.1, 0.05], "val_loss": [0.2, 0.1]}
    mock_model.predict.return_value = np.random.randn(10)
    
    # Test code that uses PyTorchLSTMModel
    ...
```

#### OpenCV

OpenCV functions are mocked to avoid dependencies on image processing capabilities:

```python
@patch('cv2.resize')
@patch('cv2.cvtColor')
def test_image_processing(self, mock_cvtcolor, mock_resize):
    # Setup mocks
    mock_resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_cvtcolor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Test code that uses cv2.resize and cv2.cvtColor
    ...
```

### System Resources

#### File System

File operations are mocked to avoid dependencies on the file system:

```python
@patch('builtins.open', new_callable=mock_open, read_data='test data')
def test_file_reading(self, mock_file):
    # Test code that reads files
    ...
```

For writing files:

```python
@patch('builtins.open', new_callable=mock_open)
def test_file_writing(self, mock_file):
    # Test code that writes files
    ...
    mock_file.assert_called_with('output.txt', 'w')
    handle = mock_file()
    handle.write.assert_called_with('output data')
```

#### Time and Randomness

Time-dependent functions are mocked to ensure deterministic behavior:

```python
@patch('time.time')
def test_time_dependent_function(self, mock_time):
    # Setup mock to return a sequence of times
    mock_time.side_effect = [0, 1, 2, 3]
    
    # Test code that uses time.time
    ...
```

Random number generators are seeded or mocked for reproducibility:

```python
def test_random_function(self):
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    
    # Test code that uses np.random
    ...
```

## Mocking Best Practices

1. **Mock at the Right Level**: Mock at the boundary of your system, not within it. For example, mock the camera interface, not the internal functions that process camera data.
2. **Keep Mocks Simple**: Mocks should be as simple as possible while still providing the necessary behavior.
3. **Verify Interactions**: Use `assert_called_with` and similar methods to verify that your code interacts with mocks correctly.
4. **Reset Mocks Between Tests**: Use `setUp` and `tearDown` methods to reset mocks between tests.
5. **Document Mock Behavior**: Document the expected behavior of mocks so that other developers understand how they should work.
6. **Test with Real Objects When Possible**: While mocking is essential for many tests, also include tests that use real objects when practical.

## Example: Mocking the MMRPhysProcessor

The MMRPhysProcessor component interacts with several external dependencies, including PyTorch models, OpenCV functions, and the file system. Here's how to mock these dependencies for testing:

```python
@patch('src.processing.mmrphys_processor.MMRPhysLEF')
@patch('torch.load')
@patch('os.path.exists')
@patch('sys.path.append')
def test_mmrphys_processor_initialization(self, mock_path_append, mock_exists, 
                                         mock_torch_load, mock_lef):
    # Setup mocks
    mock_exists.return_value = True
    mock_torch_load.return_value = {'state_dict': {}}
    
    mock_lef_instance = MagicMock()
    mock_lef.return_value = mock_lef_instance
    
    # Initialize the processor
    processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
    
    # Verify the processor was initialized correctly
    self.assertEqual(processor.model_type, 'MMRPhysLEF')
    self.assertEqual(processor.device.type, 'cpu')
    mock_lef.assert_called_once()
```

For testing video processing:

```python
@patch('cv2.VideoCapture')
def test_process_video(self, mock_video_capture):
    # Setup mock
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 90 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
    
    # Mock read method to return frames then stop
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, test_frame) for _ in range(30)] + [(False, None)]
    
    # Initialize the processor with mocked dependencies
    processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
    
    # Mock the model's forward pass
    processor.model.forward = MagicMock(return_value=torch.tensor([0.1, 0.2, 0.3]))
    
    # Process a video
    results = processor.process_video("test_video.mp4")
    
    # Verify the results
    self.assertIsNotNone(results)
    self.assertIn('pulse_signal', results)
```

---

# Conclusion

This comprehensive testing guide provides a unified approach to testing in the GSR-RGBT project. By following the strategies, execution procedures, coverage analysis, and mocking approaches outlined in this document, developers can ensure the reliability and correctness of the codebase.

The testing framework is designed to be flexible and adaptable, allowing it to evolve as the project grows and changes. Regular reviews of the testing strategy will help ensure it remains effective and aligned with the project's goals.

If you have suggestions for improving the testing process, please share them with the project maintainers.