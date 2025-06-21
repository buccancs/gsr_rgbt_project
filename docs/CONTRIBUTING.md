# Contributing to GSR-RGBT Project

Thank you for your interest in contributing to the GSR-RGBT project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment following the instructions in the [Developer Onboarding Guide](developer_onboarding.md)
4. Create a new branch for your feature or bug fix
5. Make your changes
6. Run tests to ensure your changes don't break existing functionality
7. Submit a pull request

## Development Workflow

We follow a feature branch workflow:

1. **Main Branch**: The `main` branch contains the stable version of the code. All development is done in feature branches.
2. **Feature Branches**: Create a new branch for each feature or bug fix. Branch names should be descriptive and follow the format: `feature/feature-name` or `bugfix/issue-description`.
3. **Pull Requests**: When your feature or bug fix is ready, submit a pull request to merge your changes into the `main` branch.
4. **Code Review**: All pull requests must be reviewed by at least one maintainer before being merged.
5. **Continuous Integration**: All pull requests must pass the CI pipeline before being merged.

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Use 4 spaces for indentation (no tabs)
- Maximum line length is 127 characters
- Use meaningful variable and function names
- Write docstrings for all functions, classes, and modules following the Google style guide

### Documentation Style

- Use Markdown for documentation
- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include examples where appropriate

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - ✨ `:sparkles:` for new features
  - 🐛 `:bug:` for bug fixes
  - 📚 `:books:` for documentation changes
  - ♻️ `:recycle:` for refactoring code
  - 🧪 `:test_tube:` for adding tests
  - 🔧 `:wrench:` for configuration changes

## Pull Request Process

1. Update the README.md and documentation with details of changes to the interface, if applicable
2. Update the CHANGELOG.md with details of changes
3. The version number will be updated by the maintainers following [Semantic Versioning](https://semver.org/)
4. Ensure all tests pass and the CI pipeline is successful
5. Get at least one review from a maintainer
6. Once approved, a maintainer will merge your pull request

## Issue Reporting Guidelines

When reporting issues, please use the issue templates provided in the repository. If no template fits your issue, please include:

1. A clear and descriptive title
2. A detailed description of the issue
3. Steps to reproduce the issue
4. Expected behavior
5. Actual behavior
6. Screenshots or code snippets, if applicable
7. Environment information (OS, Python version, etc.)

## Testing Guidelines

- Write tests for all new features and bug fixes
- Ensure all tests pass before submitting a pull request
- Follow the testing strategy outlined in [testing_strategy.md](testing_strategy.md)
- Use the appropriate test type:
  - **Unit Tests**: Test individual components in isolation
  - **Smoke Tests**: Verify that the main functionality runs without errors
  - **Regression Tests**: Ensure that changes don't break existing functionality

## Documentation Guidelines

- Update documentation for all new features and changes
- Follow the documentation strategy outlined in [documentation_strategy.md](documentation_strategy.md)
- Use clear, concise language
- Include examples where appropriate
- Keep the documentation up-to-date with code changes

## Community

- Join our community discussions on [GitHub Discussions](https://github.com/username/gsr_rgbt_project/discussions)
- Ask questions and get help on our [Slack channel](#)
- Attend our monthly community calls (details in the [community calendar](#))

Thank you for contributing to the GSR-RGBT project!