# GSR-RGBT Project Continuous Integration Setup

## Introduction

This document outlines the continuous integration (CI) setup for the GSR-RGBT project. Continuous integration is a development practice where developers integrate code into a shared repository frequently, and each integration is verified by an automated build and test process. The goal is to detect and address integration issues early, ensuring that the codebase remains in a working state.

## CI Platform

The GSR-RGBT project uses GitHub Actions as its CI platform. GitHub Actions is integrated with the project's GitHub repository and provides automated workflows for building, testing, and deploying the project.

## CI Workflow

The CI workflow for the GSR-RGBT project consists of the following steps:

1. **Code Checkout**: The workflow checks out the latest code from the repository.
2. **Environment Setup**: The workflow sets up the Python environment and installs dependencies.
3. **Code Linting**: The workflow runs linting tools to check code style and quality.
4. **Unit Tests**: The workflow runs unit tests to verify the correctness of individual components.
5. **Smoke Tests**: The workflow runs smoke tests to verify that the main functionality works.
6. **Regression Tests**: The workflow runs regression tests to ensure that changes don't break existing functionality.
7. **Coverage Analysis**: The workflow measures test coverage and reports it.
8. **Documentation Build**: The workflow builds the documentation to ensure it's up-to-date.
9. **Artifact Publishing**: The workflow publishes build artifacts for further inspection.

## Workflow Configuration

The CI workflow is configured in the `.github/workflows/ci.yml` file. Here's an example configuration:

```yaml
name: GSR-RGBT CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest pytest-cov
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install -e .

    - name: Lint with flake8
      run: |
        flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Test with pytest
      run: |
        pytest src/tests/unit --cov=src
        pytest src/tests/smoke
        pytest src/tests/regression

    - name: Upload coverage report
      uses: codecov/codecov-action@v1

    - name: Build documentation
      run: |
        pip install sphinx sphinx_rtd_theme
        cd docs
        make html

    - name: Upload documentation
      uses: actions/upload-artifact@v2
      with:
        name: documentation
        path: docs/_build/html
```

## CI Status Badges

The CI status is displayed using badges in the project's README.md file:

```markdown
![CI Status](https://github.com/username/gsr_rgbt_project/workflows/GSR-RGBT%20CI/badge.svg)
[![codecov](https://codecov.io/gh/username/gsr_rgbt_project/branch/main/graph/badge.svg)](https://codecov.io/gh/username/gsr_rgbt_project)
```

## CI Best Practices

To ensure the CI process is effective, follow these best practices:

1. **Keep CI Fast**: The CI process should complete quickly to provide timely feedback.
2. **Fix Broken Builds Immediately**: If the CI build fails, fix it immediately to prevent blocking other developers.
3. **Write Good Tests**: Write comprehensive tests that cover all critical functionality.
4. **Use Mocking**: Use mocking to isolate components and make tests faster and more reliable.
5. **Monitor Test Coverage**: Regularly review test coverage to identify areas that need more testing.
6. **Automate Everything**: Automate as much of the build and test process as possible.
7. **Use Branch Protection**: Configure branch protection rules to require passing CI checks before merging.

## Setting Up Local CI

Developers can run the CI process locally before pushing changes to the repository. This helps catch issues early and reduces the load on the CI server.

To run the CI process locally:

1. Install the required tools:
   ```
   pip install flake8 pytest pytest-cov
   ```

2. Run the linting checks:
   ```
   flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
   flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
   ```

3. Run the tests:
   ```
   pytest src/tests/unit --cov=src
   pytest src/tests/smoke
   pytest src/tests/regression
   ```

4. Build the documentation:
   ```
   cd docs
   make html
   ```

## Troubleshooting CI Issues

If the CI build fails, follow these steps to troubleshoot:

1. **Check the CI Logs**: Review the CI logs to identify the specific error.
2. **Reproduce Locally**: Try to reproduce the issue locally to debug it.
3. **Check for Environment Differences**: Look for differences between your local environment and the CI environment.
4. **Isolate the Issue**: Determine which specific test or step is failing.
5. **Fix and Verify**: Fix the issue and verify that it resolves the CI failure.

## Viewing Workflow Files in GitHub

If you're having trouble viewing the workflow files in GitHub, please refer to the [GitHub Workflows Guide](github_workflows_guide.md) for detailed instructions on how to:
1. Commit and push workflow files to GitHub
2. Navigate to the Actions tab in your repository
3. Troubleshoot common issues with GitHub Actions

## Conclusion

Continuous integration is an essential practice for maintaining code quality and preventing regressions in the GSR-RGBT project. By automatically building and testing the code on every change, we can detect and address issues early, ensuring that the codebase remains in a working state.

The CI setup described in this document provides a robust framework for automating the build and test process, making it easier to develop and maintain the project over time.
