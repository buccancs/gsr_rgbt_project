# GSR-RGBT Project Test Mocking Strategy

## Introduction

This document outlines the mocking strategy for the GSR-RGBT project, including how to mock external dependencies such as hardware devices, third-party libraries, and system resources. The goal is to provide a consistent approach to mocking that makes tests reliable, fast, and independent of external factors.

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

## Conclusion

This mocking strategy provides a consistent approach to mocking external dependencies in the GSR-RGBT project. By following these guidelines, developers can write tests that are reliable, fast, and independent of external factors.

The strategy is designed to be flexible and adaptable, allowing it to evolve as the project grows and changes. Regular reviews of the mocking strategy will help ensure it remains effective and aligned with the project's goals.