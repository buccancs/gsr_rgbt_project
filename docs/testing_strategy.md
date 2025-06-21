# GSR-RGBT Project Testing Strategy

## Introduction

This document outlines the testing strategy for the GSR-RGBT project, including the types of tests, their organization, and guidelines for writing and maintaining tests. The goal is to provide a comprehensive testing framework that ensures the reliability and correctness of the codebase.

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

## Test Execution

Tests can be run using the `run_tests.py` script in the `src/tests` directory. The script provides options for running specific types of tests and controlling the verbosity of the output.

To run all tests:
```
python src/tests/run_tests.py
```

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

To run tests with less verbose output:
```
python src/tests/run_tests.py --quiet
```

## Test Mocking

Many components in the GSR-RGBT project depend on external systems or hardware devices. To test these components in isolation, we use mocking to simulate these dependencies. The project uses the `unittest.mock` module from the Python standard library for mocking.

Common mocking patterns include:
- Mocking hardware devices like cameras and GSR sensors.
- Mocking file I/O operations.
- Mocking network requests.
- Mocking time-dependent operations.

Example mocking code:
```python
@patch('cv2.VideoCapture')
def test_video_capture(self, mock_video_capture):
    # Setup mock
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    
    # Test code that uses cv2.VideoCapture
    ...
```

## Test Data

The project uses a combination of real and synthetic data for testing:
- **Synthetic Data**: Generated programmatically for unit tests.
- **Sample Data**: Small, representative samples of real data for smoke and regression tests.
- **Mock Data**: Simulated data for testing components that depend on external systems.

Test data should be:
- **Representative**: Test data should be representative of real-world data.
- **Minimal**: Test data should be as small as possible while still being representative.
- **Versioned**: Test data should be versioned along with the code.

## Test Coverage

The project aims for high test coverage, with a focus on covering critical components and functionality. Test coverage is measured using the `coverage` Python package.

To run tests with coverage measurement:
```
coverage run --source=src src/tests/run_tests.py
coverage report
```

## Writing New Tests

When writing new tests, follow these guidelines:
1. **Test One Thing**: Each test should test one specific aspect of the component.
2. **Use Clear Names**: Test names should clearly describe what's being tested.
3. **Use Setup and Teardown**: Use the `setUp` and `tearDown` methods to set up and clean up test data.
4. **Mock External Dependencies**: Use mocking to isolate the component being tested.
5. **Test Edge Cases**: Include tests for edge cases and error conditions.
6. **Keep Tests Fast**: Tests should run quickly to provide immediate feedback.

Example test structure:
```python
class TestMyComponent(unittest.TestCase):
    def setUp(self):
        # Set up test data
        ...
    
    def tearDown(self):
        # Clean up test data
        ...
    
    def test_normal_case(self):
        # Test normal operation
        ...
    
    def test_edge_case(self):
        # Test edge case
        ...
    
    def test_error_condition(self):
        # Test error condition
        ...
```

## Maintaining Tests

As the codebase evolves, tests need to be maintained to ensure they remain relevant and effective. Follow these guidelines for maintaining tests:

1. **Update Tests When Code Changes**: When changing code, update the corresponding tests.
2. **Remove Obsolete Tests**: Remove tests for functionality that's no longer present.
3. **Refactor Tests**: Refactor tests to improve their clarity and maintainability.
4. **Add Tests for New Functionality**: Add tests for new functionality as it's developed.
5. **Review Test Coverage**: Regularly review test coverage to identify areas that need more testing.

## Continuous Integration

The project uses continuous integration to automatically run tests on code changes. This helps catch issues early and ensures that the codebase remains in a working state.

The continuous integration setup:
1. Runs all tests on every pull request.
2. Measures test coverage and reports it.
3. Fails the build if tests fail or if coverage drops below a threshold.

## Conclusion

This testing strategy provides a comprehensive approach to testing the GSR-RGBT project. By following this strategy, we can ensure the reliability and correctness of the codebase, making it easier to develop and maintain the project over time.

The strategy is designed to be flexible and adaptable, allowing it to evolve as the project grows and changes. Regular reviews of the testing strategy will help ensure it remains effective and aligned with the project's goals.