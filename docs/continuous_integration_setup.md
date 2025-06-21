# GSR-RGBT Project Continuous Integration Setup

## Introduction

This document outlines the continuous integration (CI) setup for the GSR-RGBT project. Continuous integration is a development practice where developers integrate code into a shared repository frequently, and each integration is verified by an automated build and test process. The goal is to detect and address integration issues early, ensuring that the codebase remains in a working state.

## CI Platform

The GSR-RGBT project uses GitHub Actions as its CI platform. GitHub Actions is integrated with the project's GitHub repository and provides automated workflows for building, testing, and deploying the project.

## GitHub Actions Workflows

The GSR-RGBT project uses several GitHub Actions workflows to automate different aspects of the development process:

1. **CI Workflow** (ci.yml): The main continuous integration workflow that runs tests and builds documentation.
2. **Security Scanning** (codeql-analysis.yml): A workflow that performs security analysis using GitHub's CodeQL.
3. **Automated Releases** (release.yml): A workflow that automates the creation of releases and publishing to PyPI.
4. **Performance Benchmarking** (benchmark.yml): A workflow that runs performance benchmarks and tracks results over time.

### CI Workflow

The main CI workflow consists of the following jobs:

1. **Lint**: Checks code style and quality using flake8.
2. **Test**: Runs unit, smoke, and regression tests on multiple Python versions (3.8, 3.9, 3.10).
3. **Docs**: Builds the documentation and uploads it as an artifact.

Key features of the CI workflow include:

- **Matrix Testing**: Tests are run on multiple Python versions to ensure compatibility.
- **Dependency Caching**: Dependencies are cached to speed up builds.
- **Scheduled Runs**: The workflow runs automatically on a schedule (nightly) to catch issues early.
- **Manual Triggering**: The workflow can be triggered manually using the workflow_dispatch event.
- **Coverage Reporting**: Test coverage is measured and reported to Codecov.

### Security Scanning Workflow

The security scanning workflow uses GitHub's CodeQL to identify security vulnerabilities in the codebase. It runs:

- On pushes to the main branch
- On pull requests to the main branch
- On a weekly schedule
- When manually triggered

### Automated Release Workflow

The release workflow automates the process of creating releases and publishing packages to PyPI. It is triggered when a tag starting with 'v' is pushed to the repository (e.g., v1.0.0). The workflow:

1. Builds the Python package
2. Generates a changelog based on commits
3. Creates a GitHub release with the changelog
4. Uploads the package as a release asset
5. Publishes the package to PyPI

### Performance Benchmarking Workflow

The benchmarking workflow runs performance tests and tracks results over time. It:

1. Runs on pushes to the main branch
2. Runs on pull requests to the main branch
3. Runs on a weekly schedule
4. Can be triggered manually

The workflow uses pytest-benchmark to run benchmarks and the github-action-benchmark action to visualize and track results. It also sets up alerts for performance regressions.

## Workflow Configurations

### CI Workflow Configuration

The main CI workflow is configured in the `.github/workflows/ci.yml` file. Here's an example configuration:

```yaml
name: GSR-RGBT CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run nightly at midnight UTC
    - cron: '0 0 * * *'
  workflow_dispatch:  # Allow manual triggering

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

      - name: Lint with flake8
        run: |
          flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  test:
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8', '3.9', '3.10']

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}-${{ matrix.python-version }}
          restore-keys: |
            ${{ runner.os }}-pip-${{ matrix.python-version }}-
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -e .

      - name: Test with pytest
        run: |
          pytest src/tests/unit --cov=src
          pytest src/tests/smoke
          pytest src/tests/regression

      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.python-version }}
          fail_ci_if_error: false

  docs:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-docs-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-docs-
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install sphinx sphinx_rtd_theme
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -e .

      - name: Build documentation
        run: |
          cd docs
          make html

      - name: Upload documentation
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/_build/html
```

### Security Scanning Workflow Configuration

The security scanning workflow is configured in the `.github/workflows/codeql-analysis.yml` file:

```yaml
name: "CodeQL Security Scan"

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run once a week on Sunday at midnight UTC
    - cron: '0 0 * * 0'
  workflow_dispatch:  # Allow manual triggering

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

### Automated Release Workflow Configuration

The release workflow is configured in the `.github/workflows/release.yml` file:

```yaml
name: Create Release

on:
  push:
    tags:
      - 'v*' # Push events to matching v*, i.e. v1.0, v20.15.10

jobs:
  build:
    name: Create Release
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine wheel
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -e .

      - name: Build package
        run: |
          python -m build
          twine check dist/*

      - name: Get version from tag
        id: get_version
        run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Generate changelog
        id: changelog
        uses: metcalfc/changelog-generator@v4.1.0
        with:
          myToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Create Release
        id: create_release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ steps.get_version.outputs.VERSION }}
          body: |
            ## Changes in this Release
            ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: false

      - name: Upload to PyPI
        if: startsWith(github.ref, 'refs/tags/')
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
          skip_existing: true
```

### Performance Benchmarking Workflow Configuration

The benchmarking workflow is configured in the `.github/workflows/benchmark.yml` file:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run weekly on Monday at midnight UTC
    - cron: '0 0 * * 1'
  workflow_dispatch:  # Allow manual triggering

jobs:
  benchmark:
    name: Run Benchmarks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-benchmark-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-benchmark-
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-benchmark
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -e .

      - name: Run benchmarks
        run: |
          mkdir -p benchmark_results
          pytest src/tests/benchmarks --benchmark-json benchmark_results/output.json

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          name: Python Benchmarks
          tool: 'pytest'
          output-file-path: benchmark_results/output.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '200%'
          comment-on-alert: true
          fail-on-alert: true
          summary-always: true
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
