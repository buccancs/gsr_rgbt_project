# Developer Guide

Thank you for your interest in contributing to the GSR-RGBT project! This comprehensive guide provides everything you need to know about our development process, from getting started to advanced CI/CD workflows.

## Table of Contents

- [Getting Started](#getting-started)
- [Code of Conduct](#code-of-conduct)
- [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Continuous Integration](#continuous-integration)
- [GitHub Actions Workflows](#github-actions-workflows)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Community](#community)

## Getting Started

### Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/gsr-rgbt-project.git
   cd gsr-rgbt-project
   ```
3. **Set up the development environment**:
   ```bash
   ./gsr_rgbt_tools.sh setup
   ```
4. **Set up pre-commit hooks**:
   ```bash
   python setup_hooks.py
   ```
5. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
6. **Make your changes** and commit them
7. **Run tests** to ensure your changes don't break existing functionality:
   ```bash
   ./gsr_rgbt_tools.sh test
   ```
8. **Submit a pull request**

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- FLIR Spinnaker SDK (for thermal camera support)
- Make (optional, for using Makefile commands)

### Automated Setup

The project includes an automated setup script that handles all dependencies:

```bash
./gsr_rgbt_tools.sh setup
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Build Cython extensions
- Validate system dependencies
- Check hardware connectivity
- Provide guidance for FLIR Spinnaker SDK installation

### Manual Setup

If you prefer manual setup:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Build Cython extensions
python setup.py build_ext --inplace
```

## Development Workflow

We follow a feature branch workflow designed to ensure code quality and maintainability:

### Branch Strategy

1. **Main Branch**: The `main` branch contains the stable version of the code
2. **Feature Branches**: Create a new branch for each feature or bug fix
   - Format: `feature/feature-name` or `bugfix/issue-description`
   - Keep branches focused and small
   - Regularly sync with main branch

### Workflow Steps

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**:
   - Pre-commit hooks will run automatically
   - Follow our commit message guidelines

4. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Code review and merge**:
   - All PRs require at least one review
   - CI pipeline must pass
   - Maintainer will merge approved PRs

## Pre-commit Hooks

Pre-commit hooks are automated checks that run before each commit, ensuring code quality and consistency.

### What Our Hooks Do

#### Code Quality Hooks
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml/json/toml**: Validates file syntax
- **check-added-large-files**: Prevents large files (>500KB)
- **check-merge-conflict**: Detects merge conflict markers
- **detect-private-key**: Prevents committing private keys

#### Code Formatting Hooks
- **black**: Formats Python code
- **isort**: Sorts Python imports
- **prettier**: Formats YAML, JSON, Markdown files

#### Linting Hooks
- **flake8**: Checks Python code style and errors
- **mypy**: Static type checking
- **bandit**: Security vulnerability scanning

#### Testing Hooks
- **pytest-check**: Runs unit tests

### Setup Pre-commit Hooks

```bash
python setup_hooks.py
```

This script will:
- Install pre-commit if not already installed
- Install all required dependencies
- Set up hooks to run automatically
- Optionally run hooks on all files

### Manual Hook Execution

```bash
# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black
```

### Skipping Hooks (Use Sparingly)

```bash
git commit -m "Your message" --no-verify
```

## Coding Standards

### Python Code Style

- **Follow PEP 8** for Python code style
- **Use 4 spaces** for indentation (no tabs)
- **Maximum line length**: 127 characters
- **Use meaningful names** for variables and functions
- **Write docstrings** for all functions, classes, and modules (Google style)

### Type Hints

- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `Optional[Type]` for optional parameters

### Example Code Style

```python
from typing import Optional, List, Dict, Any

def process_gsr_data(
    data: List[float],
    sampling_rate: int,
    filter_params: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Process GSR data and extract features.
    
    Args:
        data: Raw GSR data points
        sampling_rate: Data sampling rate in Hz
        filter_params: Optional filtering parameters
        
    Returns:
        Dictionary containing extracted features
        
    Raises:
        ValueError: If data is empty or sampling_rate is invalid
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Implementation here
    return {"mean": 0.0, "std": 0.0}
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs after the first line
- Use conventional commit format when applicable

#### Commit Message Examples

```
✨ Add real-time GSR data visualization

🐛 Fix thermal camera initialization timeout

📚 Update installation guide for Windows users

♻️ Refactor data processing pipeline for better performance

🧪 Add unit tests for feature extraction module
```

## Testing Guidelines

### Test Types

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Smoke Tests**: Verify main functionality runs without errors
- **Regression Tests**: Ensure changes don't break existing functionality

### Writing Tests

```python
import pytest
from src.processing.feature_extraction import extract_gsr_features

class TestGSRFeatureExtraction:
    def test_extract_features_valid_data(self):
        """Test feature extraction with valid data."""
        data = [1.0, 2.0, 3.0, 2.0, 1.0]
        features = extract_gsr_features(data, sampling_rate=10)
        
        assert "mean" in features
        assert "std" in features
        assert features["mean"] == pytest.approx(1.8, rel=1e-2)
    
    def test_extract_features_empty_data(self):
        """Test feature extraction with empty data."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            extract_gsr_features([], sampling_rate=10)
```

### Running Tests

```bash
# Run all tests
./gsr_rgbt_tools.sh test

# Run specific test file
pytest src/tests/test_feature_extraction.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto
```

## Pull Request Process

### Before Submitting

1. **Update documentation** with details of interface changes
2. **Add tests** for new functionality
3. **Ensure all tests pass** and CI pipeline is successful
4. **Update CHANGELOG.md** with details of changes
5. **Rebase on latest main** to avoid merge conflicts

### PR Requirements

- **Clear title and description** explaining the changes
- **Link to related issues** using GitHub keywords (fixes #123)
- **Screenshots or demos** for UI changes
- **Performance impact** assessment for significant changes
- **Breaking changes** clearly documented

### Review Process

1. **Automated checks** must pass (CI, pre-commit hooks)
2. **At least one review** from a maintainer required
3. **Address feedback** promptly and professionally
4. **Maintainer merges** approved PRs

## Continuous Integration

Our CI pipeline runs on every pull request and push to main branch.

### CI Pipeline Stages

1. **Code Quality Checks**
   - Linting (flake8, mypy)
   - Code formatting (black, isort)
   - Security scanning (bandit)

2. **Testing**
   - Unit tests
   - Integration tests
   - Coverage reporting

3. **Build Verification**
   - Package building
   - Dependency checking
   - Documentation building

4. **Deployment** (main branch only)
   - Artifact creation
   - Documentation deployment

### Local CI Simulation

```bash
# Run the same checks as CI locally
make ci-check

# Or run individual components
make lint
make test
make build
```

## GitHub Actions Workflows

### Main Workflows

#### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)

Runs on every push and pull request:

```yaml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### 2. Code Quality (`.github/workflows/code-quality.yml`)

Runs linting and security checks:

```yaml
name: Code Quality
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9
      - name: Run pre-commit
        uses: pre-commit/action@v3.0.0
```

### Workflow Triggers

- **Push to main**: Full CI/CD pipeline
- **Pull requests**: CI checks and tests
- **Release tags**: Build and publish artifacts
- **Schedule**: Nightly dependency updates

## Issue Reporting Guidelines

### Using Issue Templates

We provide templates for common issue types:

- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features
- **Documentation**: For documentation improvements
- **Question**: For asking questions

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Windows 10, Ubuntu 20.04]
 - Python Version: [e.g. 3.9.7]
 - Project Version: [e.g. 1.2.3]

**Additional context**
Add any other context about the problem here.
```

## Documentation Guidelines

### Documentation Types

- **User Documentation**: How to use the software
- **Developer Documentation**: How to contribute and extend
- **API Documentation**: Code reference and examples
- **Architecture Documentation**: System design and structure

### Writing Guidelines

- **Use clear, concise language**
- **Include examples** where appropriate
- **Keep documentation up-to-date** with code changes
- **Follow Markdown best practices**
- **Use consistent formatting**

### Documentation Structure

```
docs/
├── user/
│   ├── USER_GUIDE.md
│   └── DEPLOYMENT_GUIDE.md
├── developer/
│   ├── DEVELOPER_GUIDE.md
│   ├── testing_guide.md
│   └── test_report_template.md
├── technical/
│   ├── ARCHITECTURE.md
│   ├── technical_guide.md
│   └── GLOSSARY.md
└── project/
    ├── project_roadmap.md
    └── documentation_strategy.md
```

## Community

### Communication Channels

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and reviews

### Getting Help

1. **Check existing documentation** first
2. **Search GitHub issues** for similar problems
3. **Use the troubleshooting guide**: `./gsr_rgbt_tools.sh help troubleshoot`
4. **Create a new issue** with detailed information

### Contributing to Community

- **Help answer questions** in GitHub Discussions
- **Review pull requests** from other contributors
- **Improve documentation** based on your experience
- **Share your use cases** and success stories

## Advanced Topics

### Custom Development Workflows

For advanced contributors, you can customize the development workflow:

#### Custom Pre-commit Configuration

Create `.pre-commit-config-custom.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: custom-check
        name: Custom Project Check
        entry: python scripts/custom_check.py
        language: system
        files: ^src/
```

#### Custom CI Workflows

For specialized testing or deployment needs, you can extend the GitHub Actions workflows in `.github/workflows/`.

### Performance Testing

```bash
# Run performance benchmarks
python scripts/benchmark.py

# Profile specific functions
python -m cProfile -o profile.stats src/scripts/train_model.py
```

### Security Considerations

- **Never commit secrets** or API keys
- **Use environment variables** for sensitive configuration
- **Regularly update dependencies** to patch security vulnerabilities
- **Follow secure coding practices**

---

Thank you for contributing to the GSR-RGBT project! Your contributions help make this project better for everyone.

For questions or additional help, please don't hesitate to reach out through our community channels.