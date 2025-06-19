# Comprehensive Guide for Setting Up and Running the GSR-RGBT Project on macOS

This guide will help you set up the necessary drivers and dependencies for the FLIR thermal camera and Shimmer GSR+ sensor, configure the devices properly, and start the GUI recording application.

## 1. Installing Necessary Drivers and Dependencies

### 1.1 FLIR A65 Thermal Camera Setup

The FLIR thermal camera **requires** the Spinnaker SDK, which provides the PySpin library used by the application. This SDK is **mandatory** for the thermal camera functionality, even if you initially plan to use simulation mode:

1. **Download the FLIR Spinnaker SDK for macOS**:
   - Visit the [FLIR website](https://www.flir.com/products/spinnaker-sdk/) and download the macOS version of the Spinnaker SDK
   - You will need to create a free FLIR account if you don't already have one
   - Select the appropriate version for your operating system (macOS)

2. **Install the Spinnaker SDK**:
   - Open the downloaded package and follow the installation wizard
   - **IMPORTANT**: During installation, make sure to select the option to install the Python bindings (PySpin)
   - You may need administrator privileges to complete the installation
   - After installation, you may need to restart your computer

3. **Verify the installation**:
   ```bash
   # Activate your Python environment first if you're using one
   source .venv/bin/activate

   # Then run this command to verify PySpin is installed
   python -c "import PySpin; system = PySpin.System.GetInstance(); version = system.GetLibraryVersion(); print(f'PySpin installed successfully. Version: {version.major}.{version.minor}.{version.type}.{version.build}'); system.ReleaseInstance()"
   ```
   If this command runs without errors and displays the version information, the Spinnaker SDK is installed correctly.

4. **Troubleshooting SDK Installation**:
   - If you get an import error for PySpin, ensure that you installed the Python bindings during SDK installation
   - If you're using a virtual environment, you may need to reinstall the SDK or create a symlink to the PySpin module
   - On some systems, you might need to install additional dependencies like libusb
   - Check the FLIR Spinnaker SDK documentation for platform-specific installation issues
   - The system validation script (`make test`) will also verify if the SDK is properly installed

### 1.2 Shimmer3 GSR+ Sensor Setup

The Shimmer GSR+ sensor requires the pyshimmer library and potentially additional software:

1. **Install Shimmer Connect or ConsensysPRO**:
   - Download [Shimmer Connect](https://shimmersensing.com/support/wireless-sensor-networks-download/) or [ConsensysPRO](https://shimmersensing.com/support/consensys-download/) for macOS
   - Use this software to:
     - Update the Shimmer's firmware
     - Configure its settings (enable the GSR sensor and set the sampling rate)
     - Pair it via Bluetooth (if using Bluetooth instead of the dock's serial connection)

2. **Identify the Serial Port**:
   - When connecting the Shimmer via its USB dock, it will appear as a serial port
   - On macOS, the port will typically be named like `/dev/tty.usbmodem*` or `/dev/cu.*`
   - To find the port, run this command before and after connecting the device:
     ```bash
     ls /dev/tty.* /dev/cu.*
     ```
     The new entry that appears is your Shimmer device.

3. **USB-to-Serial Driver (if needed)**:
   - macOS usually has built-in drivers for common USB-to-Serial chips
   - If the device isn't recognized, you might need to install a specific driver for the chip used in the Shimmer dock
   - Look for the chip model on the dock (often FTDI or CP210x) and download the appropriate driver

### 1.3 Python Environment Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/gsr-rgbt-project.git
   cd gsr_rgbt_project
   ```

2. **Run the automated setup script**:
   ```bash
   # Make the script executable
   chmod +x setup.sh

   # Run the setup script
   ./setup.sh
   ```

   This script will:
   - Check for required system dependencies
   - Create and activate a Python virtual environment
   - Install required Python packages
   - Build Cython extensions
   - **Check for and help install the FLIR Spinnaker SDK**
   - **Check for and install pyshimmer and other dependencies**
   - Run system validation checks
   - Provide a detailed report of installed components

   The script will guide you through the installation process for the FLIR Spinnaker SDK, which requires downloading from the FLIR website. It will also attempt to install all other required dependencies automatically.

   **Note**: The script will not report success until all required components are installed and verified.

3. **Alternatively, you can perform the setup steps manually**:
   ```bash
   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate

   # Install required Python packages
   pip install -r requirements.txt

   # Build Cython extensions
   python setup.py build_ext --inplace
   ```

   Or use the Makefile:
   ```bash
   make setup
   ```

## 2. Configuring the Devices

### 2.1 Update the Configuration File

Edit the `src/config.py` file to configure the devices properly:

1. **For the FLIR thermal camera**:
   - Set `THERMAL_CAMERA_ID` to the correct camera ID (usually 0 or 1)
   - Set `THERMAL_SIMULATION_MODE = False` to use the real camera

2. **For the Shimmer GSR+ sensor**:
   - Set `GSR_SENSOR_PORT` to the correct serial port (e.g., "/dev/tty.usbmodem1234")
   - Set `GSR_SIMULATION_MODE = False` to use the real sensor

Example configuration:

```python
# Camera settings
RGB_CAMERA_ID = 0
THERMAL_CAMERA_ID = 1
THERMAL_SIMULATION_MODE = False  # Set to False for real camera

# GSR sensor settings
GSR_SENSOR_PORT = "/dev/tty.usbmodem1234"  # Update with your actual port
GSR_SAMPLING_RATE = 128  # Hz
GSR_SIMULATION_MODE = False  # Set to False for real sensor
```

### 2.2 Verify the System Setup

Run the system validation script to verify that all components are properly configured:

```bash
python src/scripts/check_system.py
```

or using the Makefile:

```bash
make test
```

This script will perform comprehensive checks:
- Verify critical Python dependencies are installed
- Check if output directories exist
- Test RGB camera connectivity
- **Verify that the FLIR Spinnaker SDK is installed and get its version**
- Detect available serial ports for the GSR sensor
- Verify if the configured GSR port is valid and the sensor is working
- List all available FLIR cameras and verify the configured camera index
- Provide a detailed validation summary

The script will display a list of available serial ports, which is helpful for identifying the correct port for the GSR sensor. If the configured GSR port is not found in the available ports, the script will warn you and suggest updating the configuration.

Example output when everything is properly installed:
```
--- Validation Summary ---
Dependencies: OK
Output Directories: OK
RGB Camera: OK
Thermal Camera: OK
  → FLIR SDK Version: 3.0.0.118
GSR Sensor: OK

Available Serial Ports:
  - /dev/cu.usbmodem1234
  - /dev/cu.Bluetooth-Incoming-Port
```

Example output when the FLIR SDK is missing:
```
--- Validation Summary ---
Dependencies: OK
Output Directories: OK
RGB Camera: OK
Thermal Camera: FAIL - FLIR SDK NOT INSTALLED
  → FLIR Spinnaker SDK is required but not found
  → Download from: https://www.flir.com/products/spinnaker-sdk/
GSR Sensor: OK
```

If you're running in simulation mode, the script will still check for the FLIR SDK but will skip the hardware checks for the corresponding devices. **Note that the FLIR SDK is required even in simulation mode for production use.**

## 3. Starting the GUI Recording Application

Once everything is set up and configured, you can start the GUI recording application:

1. **Make sure your devices are connected**:
   - Connect the FLIR thermal camera via USB
   - Connect the Shimmer GSR+ sensor via USB dock or ensure it's paired via Bluetooth

2. **Start the application**:
   ```bash
   python src/main.py
   ```

   or using the Makefile:

   ```bash
   make run_app
   ```

3. **Using the GUI**:
   - Enter a Subject ID in the input field
   - Click the "Start Recording" button to begin capturing data
   - The RGB and thermal video feeds will be displayed in the GUI
   - Click the "Stop Recording" button when finished
   - The recorded data will be saved in the `data/recordings` directory

## 4. Troubleshooting

### FLIR Camera Issues

- **FLIR SDK Not Found**: If the validation shows "FLIR SDK NOT INSTALLED":
  - Follow the installation instructions in section 1.1 to install the Spinnaker SDK
  - Make sure you selected the Python bindings (PySpin) during installation
  - Verify the installation using the command provided in section 1.1
  - If using a virtual environment, you may need to create a symlink to the PySpin module

- **FLIR SDK Found But Camera Not Detected**:
  - Make sure the camera is properly connected via USB
  - Try a different USB port or cable
  - Check if the camera is recognized by the system:
    ```bash
    # On macOS
    system_profiler SPUSBDataType | grep -A 10 FLIR
    ```
  - Try running a simple PySpin example to verify the camera connection:
    ```bash
    # Create a test file
    echo "import PySpin; system = PySpin.System.GetInstance(); cam_list = system.GetCameras(); print(f'Found {cam_list.GetSize()} cameras'); cam_list.Clear(); system.ReleaseInstance()" > test_flir.py
    # Run it
    python test_flir.py
    ```

- **Permission Issues**:
  - On macOS, you might need to grant permissions for the application to access the camera
  - Check System Preferences > Security & Privacy > Camera
  - You may need to run the application with sudo for USB device access:
    ```bash
    sudo python src/main.py
    ```

- **Driver Issues**:
  - Some systems may require additional USB drivers
  - Check the FLIR documentation for your specific camera model
  - You may need to install libusb or other dependencies

### Shimmer Sensor Issues

- If the Shimmer sensor is not detected, verify the correct port in `config.py`
- Make sure the sensor is charged and powered on
- Try connecting to the sensor using Shimmer Connect or ConsensysPRO to verify it's working
- Check if the GSR sensor is enabled in the Shimmer's configuration

### Python Environment Issues

- If you encounter import errors, make sure you've activated the virtual environment
- If Cython extensions fail to build, make sure you have a C compiler installed
- If PySpin is not found, make sure the Spinnaker SDK is installed correctly

## 5. Using the All-in-One Script

For convenience, you can use the `run_everything.sh` script to run all components of the system in sequence:

```bash
# Make the script executable
chmod +x run_everything.sh

# Run the script
./run_everything.sh
```

This script will:
- Check if the virtual environment exists and is activated
- **Check for and install the FLIR Spinnaker SDK if not already installed**
- **Check for and install all required dependencies (pyshimmer, tensorflow, neurokit2, etc.)**
- Build Cython extensions
- Run system validation checks
- Run synchronization tests
- Generate mock data (if no data exists)
- Run the full ML pipeline (training, inference, evaluation)
- Optionally run the data collection application

The script includes comprehensive error handling and will:
- Detect missing components and try to install them automatically
- Warn you if any components could not be installed
- Ask for confirmation before continuing with missing components
- Continue with subsequent steps even if some steps fail (with appropriate warnings)

This is especially useful for running the entire system in one go, ensuring all required components are installed and working properly.

## 6. Running in Simulation Mode

If you don't have the physical devices available, you can run the application in simulation mode:

1. **Install the FLIR Spinnaker SDK anyway**:
   - Even in simulation mode, the FLIR Spinnaker SDK should be installed
   - This ensures a smooth transition to real hardware later
   - The system validation will still check for the SDK even in simulation mode
   - The setup script (`setup.sh`) and all-in-one script (`run_everything.sh`) will help you install the SDK
   - For development and testing purposes only, you can proceed without the SDK, but this is not recommended for production

2. In `src/config.py`, set:
   ```python
   THERMAL_SIMULATION_MODE = True
   GSR_SIMULATION_MODE = True
   ```

3. Start the application using the all-in-one script:
   ```bash
   ./run_everything.sh
   ```

   Or start it manually:
   ```bash
   python src/main.py
   ```

4. Verify the simulation is working:
   - The GUI should display simulated thermal and RGB video feeds
   - The system will generate synthetic data that mimics real sensor readings
   - You can record sessions with this simulated data for testing the ML pipeline

This simulation mode is useful for:
- Development and testing without physical hardware
- Demonstrating the application's functionality
- Testing the ML pipeline with synthetic data
- Training new users on the system

However, for accurate physiological measurements, you should use the real hardware with the proper drivers installed.
