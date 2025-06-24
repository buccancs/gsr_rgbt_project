# GSR-RGBT Project Style Guidelines

This document outlines the coding style guidelines for the GSR-RGBT project to ensure consistency, readability, and maintainability across the codebase.

## Python Code Style

### Formatting
- **Line Length**: Maximum 88 characters
- **Indentation**: 4 spaces (no tabs)
- **Tool**: Use Black formatter with default settings

### Naming Conventions
- **Modules**: Lowercase with underscores (e.g., `feature_extractor.py`)
- **Packages**: Lowercase, short names (e.g., `ml`, `data`, `core`)
- **Classes**: CamelCase (e.g., `VideoCapture`, `GSRProcessor`)
- **Functions/Methods**: Lowercase with underscores (e.g., `process_frame`, `extract_features`)
- **Variables**: Lowercase with underscores (e.g., `frame_count`, `device_id`)
- **Constants**: UPPERCASE with underscores (e.g., `MAX_FPS`, `DEFAULT_TIMEOUT`)

### Imports
- Sort imports using isort with black profile
- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- Use absolute imports for external packages
- Use relative imports for internal modules

### Type Hints
- Add type hints to all function parameters and return values
- Use `from __future__ import annotations` for forward references
- Use `Optional[Type]` instead of `Type | None`
- Use `Union[Type1, Type2]` instead of `Type1 | Type2`
- Use `Sequence` or `Iterable` instead of specific container types when appropriate

### Docstrings
- Use Google-style docstrings
- Include descriptions for all parameters, return values, and exceptions
- Add examples for complex functions
- Include module-level docstrings

## Code Organization

### File Structure
- Follow the project's defined file structure as outlined in `refactoring_plan.md`
- Keep files focused on a single responsibility
- Split large files (>300 lines) into multiple modules

### Class Design
- Use composition over inheritance when possible
- Implement abstract base classes for common interfaces
- Keep classes focused on a single responsibility
- Use dataclasses for simple data containers

### Error Handling
- Use specific exception types instead of generic exceptions
- Create custom exception classes for domain-specific errors
- Include detailed error messages
- Add proper logging for exceptions

## Performance Considerations

- Use generators for large data processing
- Implement lazy loading for resource-intensive operations
- Add caching for expensive computations
- Use numpy vectorization instead of Python loops for numerical operations

## Testing

- Write unit tests for all new functionality
- Follow test file naming convention: `test_*.py`
- Use pytest fixtures for common test setups
- Aim for high test coverage, especially for critical components

## Documentation

- Keep README.md up to date
- Document architectural decisions
- Update documentation when making significant changes
- Include examples for complex functionality

## Version Control

- Write clear, descriptive commit messages
- Reference issue numbers in commit messages when applicable
- Keep commits focused on a single change
- Use feature branches for new development

## Success Criteria

Code should:
- Pass all linting checks (flake8)
- Pass formatting checks (black)
- Pass type checking (mypy)
- Have comprehensive docstrings
- Follow the naming conventions
- Be well-tested
- Be maintainable and readable