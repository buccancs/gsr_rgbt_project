# GSR-RGBT Project Test Execution

## Introduction

This document provides instructions for running tests in the GSR-RGBT project, including setup, configuration, and execution of different types of tests. The goal is to make it easy for developers to run tests and verify the correctness of the codebase.

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

## Conclusion

This document provides instructions for running tests in the GSR-RGBT project. By following these instructions, you can ensure that the codebase remains correct and reliable.

If you have suggestions for improving the testing process, please share them with the project maintainers.