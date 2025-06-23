# GSR-RGBT Project Refactoring Plan

## Overview
This document outlines the comprehensive refactoring plan for the GSR-RGBT project to make it professional, pythonic, and maintainable.

## File Naming and Organization Issues Identified

### Current Issues:
1. **Duplicate files in different locations**:
   - `src/capture/` vs `src/data_collection/capture/`
   - `src/processing/` vs `src/ml_pipeline/preprocessing/`
   - Multiple test directories with similar content

2. **Non-pythonic naming**:
   - `pytorch_cnn_models.py` → should be `cnn.py` in pytorch models package
   - `check_neurokit.py` → should be `check_dependencies.py`
   - `build_project.py` → should be in scripts/ or tools/

3. **Overly long files**:
   - `pytorch_models.py` (1505 lines) → split into multiple files
   - `main.py` (345 lines) → extract components

4. **Mixed responsibilities**:
   - GUI and business logic mixed
   - Configuration scattered across files

## New File Structure

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   └── exceptions.py
├── data/
│   ├── __init__.py
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gsr.py
│   │   ├── thermal.py
│   │   └── video.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── feature_extractor.py
│   └── logger.py
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pytorch/
│   │   │   ├── __init__.py
│   │   │   ├── lstm.py
│   │   │   ├── cnn.py
│   │   │   ├── autoencoder.py
│   │   │   ├── vae.py
│   │   │   ├── resnet.py
│   │   │   └── transformer.py
│   │   └── tensorflow/
│   │       ├── __init__.py
│   │       ├── cnn.py
│   │       ├── resnet.py
│   │       └── transformer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── early_stopping.py
│   │   └── metrics.py
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py
│       └── visualizer.py
├── gui/
│   ├── __init__.py
│   ├── application.py
│   ├── main_window.py
│   └── widgets/
│       ├── __init__.py
│       ├── video_widget.py
│       └── control_panel.py
├── utils/
│   ├── __init__.py
│   ├── device_manager.py
│   ├── timestamp_manager.py
│   └── file_utils.py
└── scripts/
    ├── __init__.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── inference.py
    └── system_check.py
```

## Code Quality Improvements

### 1. Type Hints
- Add strict type hints to all functions and methods
- Use `from __future__ import annotations` for forward references
- Use Union, Optional, and generic types appropriately

### 2. Docstrings
- Add comprehensive Google-style docstrings to all classes and functions
- Include parameter types, return types, and examples where appropriate
- Add module-level docstrings

### 3. Error Handling
- Replace generic `Exception` with specific exception types
- Create custom exception classes for domain-specific errors
- Add proper error messages and logging

### 4. Code Organization
- Extract long methods into smaller, focused functions
- Use composition over inheritance where appropriate
- Implement proper separation of concerns

### 5. Performance Optimizations
- Use generators where appropriate
- Implement lazy loading for large datasets
- Add caching for expensive operations
- Use numpy vectorization where possible

### 6. Testing
- Ensure all test files follow `test_*.py` naming convention
- Add comprehensive unit tests for new modules
- Implement integration tests for critical workflows

## Implementation Steps

1. **Phase 1: File Reorganization**
   - Create new directory structure
   - Move and rename files according to new scheme
   - Update all import statements

2. **Phase 2: Code Refactoring**
   - Split large files into focused modules
   - Add type hints and docstrings
   - Improve error handling

3. **Phase 3: Optimization**
   - Implement performance improvements
   - Add caching and lazy loading
   - Optimize critical paths

4. **Phase 4: Testing and Validation**
   - Run all existing tests
   - Add new tests for refactored code
   - Validate functionality

## Tools and Standards

- **Formatting**: Black with line length 88
- **Linting**: Flake8 with E203, W503 ignored
- **Type Checking**: mypy with strict mode
- **Import Sorting**: isort with black profile
- **Documentation**: Google-style docstrings

## Success Criteria

- [ ] All files follow pythonic naming conventions
- [ ] Code passes flake8 and black checks
- [ ] All functions have type hints and docstrings
- [ ] No duplicate code or files
- [ ] All tests pass
- [ ] Performance is maintained or improved
- [ ] Code is more maintainable and readable