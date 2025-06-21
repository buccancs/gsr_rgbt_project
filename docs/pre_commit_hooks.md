# Pre-commit Hooks for GSR-RGBT Project

## Introduction

This document describes the pre-commit hooks set up for the GSR-RGBT project. Pre-commit hooks are automated checks that run before each commit, helping to ensure code quality, catch issues early, and maintain a consistent codebase.

## Benefits of Pre-commit Hooks

- **Catch issues early**: Identify problems before they're committed to the repository
- **Ensure code quality**: Maintain consistent code style and quality across the project
- **Save time**: Automate repetitive tasks like code formatting and linting
- **Reduce CI failures**: Fix issues locally before pushing to CI
- **Improve collaboration**: Ensure all contributors follow the same standards

## Hooks Included

The GSR-RGBT project uses the following pre-commit hooks:

### Code Quality Hooks

- **trailing-whitespace**: Trims trailing whitespace from files
- **end-of-file-fixer**: Ensures files end with a single newline
- **check-yaml/json/toml**: Validates syntax of YAML, JSON, and TOML files
- **check-added-large-files**: Prevents committing large files (>500KB)
- **check-merge-conflict**: Checks for merge conflict markers
- **detect-private-key**: Prevents committing private keys
- **debug-statements**: Detects debug statements in Python code

### Code Formatting Hooks

- **black**: Formats Python code according to the Black code style
- **isort**: Sorts Python imports
- **prettier**: Formats YAML, JSON, Markdown, and other non-Python files

### Linting Hooks

- **flake8**: Checks Python code for style and potential errors
  - **flake8-docstrings**: Checks docstring style
  - **flake8-bugbear**: Catches common bugs and design problems
  - **flake8-comprehensions**: Helps write better list/dict/set comprehensions

### Static Type Checking

- **mypy**: Performs static type checking on Python code

### Security Checks

- **bandit**: Finds common security issues in Python code

### Testing

- **pytest-check**: Runs unit tests to ensure code changes don't break existing functionality

## Setup

To set up the pre-commit hooks for the GSR-RGBT project, follow these steps:

1. Ensure you have Python 3.8 or higher installed
2. Run the setup script:

```bash
python setup_hooks.py
```

This script will:
- Install pre-commit if not already installed
- Install all required dependencies
- Set up the hooks to run automatically on each commit
- Optionally run the hooks on all files in the repository

## Usage

Once installed, the pre-commit hooks will run automatically whenever you attempt to commit changes. If any hook fails, the commit will be aborted, and you'll need to fix the issues before trying again.

### Manual Execution

You can also run the hooks manually:

- Run on staged files:
```bash
pre-commit run
```

- Run on all files:
```bash
pre-commit run --all-files
```

- Run a specific hook:
```bash
pre-commit run <hook-id>
```

### Skipping Hooks

In rare cases, you may need to skip the pre-commit hooks:

```bash
git commit -m "Your message" --no-verify
```

However, this should be used sparingly and only when absolutely necessary.

## Configuration

The pre-commit hooks are configured in the following files:

- **.pre-commit-config.yaml**: Defines the hooks and their settings
- **pyproject.toml**: Contains configuration for Black, isort, mypy, and bandit

### Customizing Hook Behavior

To customize the behavior of specific hooks, edit the corresponding section in the configuration files. For example, to change the line length for Black and isort, modify the `line-length` parameter in the pyproject.toml file.

## Troubleshooting

### Common Issues

1. **Hook installation fails**:
   - Ensure you have Python 3.8+ and pip installed
   - Try running `pip install pre-commit` manually

2. **Hooks are not running**:
   - Ensure hooks are installed with `pre-commit install`
   - Check if you're using `--no-verify` flag with git commit

3. **Hooks are too slow**:
   - Consider disabling the pytest hook for regular commits
   - Run comprehensive checks only in CI

### Getting Help

If you encounter issues with the pre-commit hooks:

1. Check the error message for specific guidance
2. Consult the documentation for the specific hook
3. Ask for help from other project contributors

## Best Practices

1. **Run hooks before pushing**: Always run pre-commit hooks before pushing changes
2. **Fix issues promptly**: Address hook failures immediately rather than bypassing them
3. **Keep hooks updated**: Periodically update the hooks with `pre-commit autoupdate`
4. **Add new hooks thoughtfully**: Consider the impact on development workflow before adding new hooks

## Conclusion

Pre-commit hooks are a valuable tool for maintaining code quality and consistency in the GSR-RGBT project. By catching issues early and automating routine checks, they help create a more robust and maintainable codebase.