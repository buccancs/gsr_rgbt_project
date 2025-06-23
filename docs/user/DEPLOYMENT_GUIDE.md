# GSR-RGBT Project Deployment Guide

## Introduction

This guide provides instructions for deploying the GSR-RGBT project in different environments. It covers environment setup, deployment steps, configuration options, and troubleshooting tips to ensure consistent and reliable deployments.

## Table of Contents

- [Environment Types](#environment-types)
- [Prerequisites](#prerequisites)
- [Development Environment](#development-environment)
- [Testing Environment](#testing-environment)
- [Production Environment](#production-environment)
- [Configuration Management](#configuration-management)
- [Deployment Process](#deployment-process)
- [Monitoring and Logging](#monitoring-and-logging)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)

## Environment Types

The GSR-RGBT project supports the following environment types:

1. **Development Environment**: Used by developers for active development and testing.
2. **Testing Environment**: Used for integration testing, user acceptance testing, and performance testing.
3. **Production Environment**: Used for the actual data collection and analysis in research settings.

## Prerequisites

Before deploying the GSR-RGBT project, ensure you have the following prerequisites:

- Python 3.8 or higher
- Git
- Required hardware:
  - RGB camera
  - Thermal camera (optional, but recommended)
  - Shimmer3 GSR+ sensor (optional, can use simulated data)
- Operating system: Windows 10/11, macOS, or Linux (Ubuntu 20.04 or higher recommended)
- Sufficient disk space (at least 10GB for the application and data)
- Sufficient RAM (at least 8GB, 16GB recommended for processing large datasets)

## Development Environment

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/gsr_rgbt_project.git
   cd gsr_rgbt_project
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the package in development mode
   ```

4. **Initialize Submodules**:
   ```bash
   git submodule update --init --recursive
   ```

5. **Configure Hardware**:
   - Update `src/config.py` with your camera IDs and GSR sensor port
   - For development without hardware, set `GSR_SIMULATION_MODE = True`

6. **Run System Validation**:
   ```bash
   python src/scripts/check_system.py
   ```

### Running the Application

To run the application in development mode:

```bash
python src/main.py --dev
```

This will start the application with development settings, including more verbose logging and mock data generation if hardware is not available.

### Development Tools

- **Code Linting**:
  ```bash
  flake8 src
  ```

- **Running Tests**:
  ```bash
  python src/tests/run_tests.py
  ```

- **Building Documentation**:
  ```bash
  cd docs
  make html
  ```

## Testing Environment

The testing environment is similar to the development environment but with a focus on testing with real hardware and data.

### Setup

1. Follow steps 1-4 from the Development Environment setup.

2. **Configure Hardware**:
   - Update `src/config.py` with the actual camera IDs and GSR sensor port
   - Set `GSR_SIMULATION_MODE = False` to use real hardware

3. **Configure Test Data Directory**:
   ```bash
   mkdir -p data/test_recordings
   ```

4. **Run System Validation**:
   ```bash
   python src/scripts/check_system.py --thorough
   ```

### Running Tests

To run the full test suite, including hardware integration tests:

```bash
python src/tests/run_tests.py --include-hardware
```

### Performance Testing

To run performance benchmarks:

```bash
pytest src/tests/benchmarks
```

## Production Environment

The production environment is used for actual data collection and analysis in research settings.

### Setup

1. **Clone the Repository**:
   ```bash
   git clone --branch stable https://github.com/username/gsr_rgbt_project.git
   cd gsr_rgbt_project
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Submodules**:
   ```bash
   git submodule update --init --recursive
   ```

5. **Configure Hardware**:
   - Update `src/config.py` with your camera IDs and GSR sensor port
   - Set `GSR_SIMULATION_MODE = False` to use real hardware
   - Set `LOG_LEVEL = 'INFO'` for appropriate logging
   - Configure `DATA_DIR` to point to your preferred data storage location

6. **Run System Validation**:
   ```bash
   python src/scripts/check_system.py --thorough
   ```

7. **Create a Desktop Shortcut** (Optional):
   - Windows: Create a shortcut to `run_app.bat`
   - macOS/Linux: Create a desktop entry file

### Running the Application

To run the application in production mode:

```bash
python src/main.py --prod
```

Or use the provided script:

```bash
./run_app.sh  # On Windows: run_app.bat
```

### Data Management

In production, data is stored in the configured `DATA_DIR`. It's recommended to:

1. Set up regular backups of the data directory
2. Implement a data retention policy
3. Monitor disk space usage

## Configuration Management

### Configuration Files

The GSR-RGBT project uses several configuration files:

1. **src/config.py**: Main application configuration
2. **configs/models/**: Machine learning model configurations
3. **configs/pipeline/**: ML pipeline configurations

### Environment Variables

The following environment variables can be used to override configuration settings:

- `GSR_RGBT_DATA_DIR`: Override the data directory
- `GSR_RGBT_LOG_LEVEL`: Override the logging level
- `GSR_RGBT_GSR_PORT`: Override the GSR sensor port
- `GSR_RGBT_RGB_CAMERA_ID`: Override the RGB camera ID
- `GSR_RGBT_THERMAL_CAMERA_ID`: Override the thermal camera ID

Example:
```bash
export GSR_RGBT_LOG_LEVEL=DEBUG
python src/main.py
```

### Configuration Templates

Template configuration files are provided in the `configs/templates/` directory. Copy and modify these templates for your specific environment.

## Deployment Process

### Manual Deployment

1. Pull the latest code from the appropriate branch:
   ```bash
   git pull origin stable  # For production
   ```

2. Update dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update submodules:
   ```bash
   git submodule update --recursive
   ```

4. Run system validation:
   ```bash
   python src/scripts/check_system.py
   ```

5. Restart the application

### Automated Deployment

For automated deployment, you can use the provided deployment script:

```bash
./deploy.sh [environment]  # environment can be dev, test, or prod
```

This script will:
1. Pull the latest code
2. Update dependencies
3. Update submodules
4. Run system validation
5. Restart the application

### Continuous Deployment

The project supports continuous deployment using GitHub Actions. The workflow is defined in `.github/workflows/deploy.yml`.

To set up continuous deployment:

1. Configure the deployment environment in GitHub repository settings
2. Add the required secrets for the target environment
3. The deployment will be triggered automatically on pushes to the specified branches

## Monitoring and Logging

### Logging

Logs are stored in the `logs/` directory by default. The logging level can be configured in `src/config.py` or using the `GSR_RGBT_LOG_LEVEL` environment variable.

Log rotation is configured to keep logs for 30 days by default.

### Monitoring

For production environments, it's recommended to set up monitoring:

1. **System Monitoring**:
   - Monitor CPU, memory, and disk usage
   - Set up alerts for resource constraints

2. **Application Monitoring**:
   - Monitor application logs for errors
   - Set up alerts for critical errors

3. **Data Collection Monitoring**:
   - Monitor data collection sessions
   - Set up alerts for failed sessions

## Rollback Procedures

If a deployment causes issues, follow these rollback procedures:

### Manual Rollback

1. Identify the last known good version:
   ```bash
   git log --oneline
   ```

2. Checkout that version:
   ```bash
   git checkout <commit-hash>
   ```

3. Update dependencies and submodules:
   ```bash
   pip install -r requirements.txt
   git submodule update --recursive
   ```

4. Restart the application

### Automated Rollback

To roll back to a previous version using the deployment script:

```bash
./deploy.sh [environment] --version <version-tag>
```

## Troubleshooting

### Common Issues

1. **Camera Not Found**:
   - Check camera connections
   - Verify camera IDs in `src/config.py`
   - Ensure no other application is using the camera

2. **GSR Sensor Not Found**:
   - Check sensor connections
   - Verify the correct port in `src/config.py`
   - Check if the Shimmer device is powered on

3. **Application Crashes on Startup**:
   - Check logs in the `logs/` directory
   - Verify all dependencies are installed
   - Check hardware configuration

4. **Data Not Being Saved**:
   - Check write permissions for the data directory
   - Verify disk space availability
   - Check logs for file I/O errors

### Getting Help

If you encounter issues not covered in this guide:

1. Check the project's issue tracker on GitHub
2. Consult the [Troubleshooting Guide](TROUBLESHOOTING.md) for more detailed solutions
3. Contact the development team for support

---

*Last updated: June 21, 2025*