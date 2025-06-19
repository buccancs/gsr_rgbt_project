# src/system/validation/check_system.py

import logging
import sys
import os
import glob
from pathlib import Path

import cv2

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src import config

# Try to import device-specific libraries
try:
    import serial.tools.list_ports
    SERIAL_TOOLS_AVAILABLE = True
except ImportError:
    logging.warning("serial.tools.list_ports not available. Serial port detection will be limited.")
    SERIAL_TOOLS_AVAILABLE = False

try:
    import PySpin
    FLIR_AVAILABLE = True
    # Try to get the version to verify the SDK is properly installed
    try:
        system = PySpin.System.GetInstance()
        version = system.GetLibraryVersion()
        FLIR_VERSION = f"{version.major}.{version.minor}.{version.type}.{version.build}"
        system.ReleaseInstance()
        logging.info(f"FLIR Spinnaker SDK version {FLIR_VERSION} found.")
    except Exception as e:
        FLIR_VERSION = "Unknown"
        logging.warning(f"FLIR Spinnaker SDK found but could not get version: {e}")
except ImportError:
    logging.error("FAIL: PySpin library not found. FLIR camera detection will not be available.")
    logging.error("      Please install the FLIR Spinnaker SDK from https://www.flir.com/products/spinnaker-sdk/")
    FLIR_AVAILABLE = False
    FLIR_VERSION = None

try:
    import pyshimmer
    PYSHIMMER_AVAILABLE = True
except ImportError:
    logging.warning("pyshimmer library not found. Shimmer GSR sensor detection will not be available.")
    PYSHIMMER_AVAILABLE = False

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)


def check_camera(camera_id: int, camera_name: str) -> bool:
    """
    Checks if a camera can be opened and can capture a frame.

    Args:
        camera_id (int): The device ID of the camera.
        camera_name (str): The descriptive name of the camera.

    Returns:
        True if the camera is working, False otherwise.
    """
    logging.info(f"Checking {camera_name} camera (ID: {camera_id})...")
    # Use cv2.CAP_DSHOW on Windows for better camera support
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logging.error(f"FAIL: Could not open {camera_name} camera.")
        return False

    ret, _ = cap.read()
    if not ret:
        logging.error(f"FAIL: Could not read a frame from {camera_name} camera.")
        cap.release()
        return False

    logging.info(f"SUCCESS: {camera_name} camera is connected and working.")
    cap.release()
    return True


def check_directories() -> bool:
    """
    Checks if the required data output directories exist and creates them if not.

    Returns:
        True if directories exist or were created successfully, False otherwise.
    """
    logging.info("Checking output directories...")
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"SUCCESS: Output directory is ready at: {config.OUTPUT_DIR}")
        return True
    except OSError as e:
        logging.error(
            f"FAIL: Could not create output directory {config.OUTPUT_DIR}: {e}"
        )
        return False


def check_dependencies() -> bool:
    """
    Checks if critical dependencies are installed.

    Returns:
        True if all checked dependencies are found, False otherwise.
    """
    logging.info("Checking critical Python package dependencies...")
    dependencies = [
        "numpy",
        "pandas",
        "cv2",
        "PyQt5",
        "tensorflow",
        "sklearn",
        "neurokit2",
    ]
    missing = []
    for package in dependencies:
        try:
            if package == "neurokit2":
                # Try both import methods for neurokit2
                try:
                    __import__(package)
                    logging.info(f"  - Found: {package}")
                except ImportError:
                    # Try importing as nk (common alias)
                    import importlib
                    nk = importlib.import_module(package)
                    logging.info(f"  - Found: {package}")
            else:
                __import__(package)
                logging.info(f"  - Found: {package}")
        except ImportError:
            logging.error(f"  - MISSING: {package}")
            missing.append(package)

    if missing:
        logging.error(
            f"FAIL: The following required packages are not installed: {', '.join(missing)}"
        )
        logging.error(
            "Please run 'make setup' or 'pip install -r requirements.txt' to install them."
        )
        return False

    logging.info("SUCCESS: All critical dependencies are installed.")
    return True


def list_serial_ports() -> list:
    """
    Lists all available serial ports on the system.

    Returns:
        A list of available serial port names.
    """
    logging.info("Checking available serial ports...")
    available_ports = []

    if SERIAL_TOOLS_AVAILABLE:
        # Use pyserial's list_ports to get all serial ports
        ports = list(serial.tools.list_ports.comports())
        if ports:
            for port in ports:
                logging.info(f"  - Found port: {port.device} - {port.description}")
                available_ports.append(port.device)
        else:
            logging.warning("  No serial ports detected using serial.tools.list_ports")
    else:
        # Fallback method for Unix-like systems
        if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            # On Linux/macOS, serial ports are in /dev/
            patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/tty.*', '/dev/cu.*']
            for pattern in patterns:
                for port in glob.glob(pattern):
                    logging.info(f"  - Found port: {port}")
                    available_ports.append(port)
        elif sys.platform.startswith('win'):
            # On Windows, try COM ports from COM1 to COM256
            for i in range(1, 257):
                port = f'COM{i}'
                try:
                    s = serial.Serial(port)
                    s.close()
                    logging.info(f"  - Found port: {port}")
                    available_ports.append(port)
                except (OSError, serial.SerialException):
                    pass

    if not available_ports:
        logging.warning("  No serial ports detected")

    return available_ports


def check_gsr_sensor() -> bool:
    """
    Checks if the GSR sensor can be connected to.

    Returns:
        True if the GSR sensor is connected and working, False otherwise.
    """
    if config.GSR_SIMULATION_MODE:
        logging.info("GSR sensor is in simulation mode, skipping hardware check.")
        return True

    logging.info(f"Checking GSR sensor on port {config.GSR_SENSOR_PORT}...")

    if not PYSHIMMER_AVAILABLE:
        logging.error("FAIL: pyshimmer library is not installed. Cannot check GSR sensor.")
        return False

    try:
        # Try to connect to the Shimmer device
        shimmer_device = pyshimmer.Shimmer(config.GSR_SENSOR_PORT)

        # Configure the shimmer device (enable GSR, set sampling rate)
        shimmer_device.set_sampling_rate(config.GSR_SAMPLING_RATE)
        shimmer_device.enable_gsr()

        # Start streaming briefly to verify connection
        shimmer_device.start_streaming()

        # Read a single packet to verify data flow
        packet = shimmer_device.read_data_packet()

        # Stop streaming and close the connection
        shimmer_device.stop_streaming()
        shimmer_device.close()

        if packet:
            logging.info(f"SUCCESS: GSR sensor is connected and working on port {config.GSR_SENSOR_PORT}")
            return True
        else:
            logging.error(f"FAIL: Could not read data from GSR sensor on port {config.GSR_SENSOR_PORT}")
            return False

    except Exception as e:
        logging.error(f"FAIL: Could not connect to GSR sensor on port {config.GSR_SENSOR_PORT}: {e}")
        return False


def check_flir_cameras() -> bool:
    """
    Checks for available FLIR cameras and verifies the configured camera index.

    Returns:
        True if FLIR cameras are detected and the configured index is valid, False otherwise.
    """
    if config.THERMAL_SIMULATION_MODE:
        logging.info("Thermal camera is in simulation mode, skipping hardware check.")
        logging.warning("NOTE: Even in simulation mode, the FLIR Spinnaker SDK should be installed for production use.")
        return True

    logging.info("Checking for FLIR cameras...")

    if not FLIR_AVAILABLE:
        logging.error("FAIL: PySpin library is not installed. Cannot check FLIR cameras.")
        logging.error("      The FLIR Spinnaker SDK is required for thermal camera operation.")
        logging.error("      Please install it from https://www.flir.com/products/spinnaker-sdk/")
        return False

    try:
        # Initialize the PySpin System
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        if num_cameras == 0:
            logging.error("FAIL: No FLIR cameras detected.")
            cam_list.Clear()
            system.ReleaseInstance()
            return False

        logging.info(f"  Found {num_cameras} FLIR camera(s)")

        # List all cameras
        for i in range(num_cameras):
            camera = cam_list.GetByIndex(i)
            try:
                # Get camera info
                nodemap_tldevice = camera.GetTLDeviceNodeMap()
                node_device_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))
                node_device_id = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

                if PySpin.IsAvailable(node_device_name) and PySpin.IsReadable(node_device_name):
                    device_name = node_device_name.GetValue()
                else:
                    device_name = "Unknown Model"

                if PySpin.IsAvailable(node_device_id) and PySpin.IsReadable(node_device_id):
                    device_id = node_device_id.GetValue()
                else:
                    device_id = "Unknown Serial"

                logging.info(f"  - Camera {i}: {device_name} (S/N: {device_id})")
            except PySpin.SpinnakerException as e:
                logging.error(f"  - Camera {i}: Error getting camera info: {e}")
            finally:
                # Release the camera
                del camera

        # Check if the configured camera index is valid
        if config.THERMAL_CAMERA_ID >= num_cameras:
            logging.error(f"FAIL: Configured thermal camera index ({config.THERMAL_CAMERA_ID}) is out of range. Only {num_cameras} camera(s) detected.")
            cam_list.Clear()
            system.ReleaseInstance()
            return False

        logging.info(f"SUCCESS: FLIR camera with index {config.THERMAL_CAMERA_ID} is available.")

        # Clean up
        cam_list.Clear()
        system.ReleaseInstance()
        return True

    except PySpin.SpinnakerException as e:
        logging.error(f"FAIL: Error checking FLIR cameras: {e}")
        return False


def main():
    """
    Runs all system validation checks.
    """
    logging.info("=======================================")
    logging.info("=  GSR-RGBT System Validation Check   =")
    logging.info("=======================================")

    # Run basic checks
    dep_ok = check_dependencies()
    dirs_ok = check_directories()
    rgb_ok = check_camera(config.RGB_CAMERA_ID, "RGB")

    # List available serial ports
    available_ports = list_serial_ports()

    # Check for FLIR cameras
    if config.THERMAL_SIMULATION_MODE:
        thermal_ok = check_camera(config.THERMAL_CAMERA_ID, "Thermal")
    else:
        thermal_ok = check_flir_cameras()

    # Check GSR sensor
    if not config.GSR_SIMULATION_MODE:
        gsr_ok = check_gsr_sensor()
    else:
        gsr_ok = True
        logging.info("GSR sensor is in simulation mode, skipping hardware check.")

    # Final Summary
    print("\n--- Validation Summary ---")
    print("Dependencies:", "OK" if dep_ok else "FAIL")
    print("Output Directories:", "OK" if dirs_ok else "FAIL")
    print("RGB Camera:", "OK" if rgb_ok else "FAIL")

    # More detailed thermal camera status
    if config.THERMAL_SIMULATION_MODE:
        print("Thermal Camera:", "OK (SIMULATION MODE)" if thermal_ok else "FAIL")
    else:
        if not FLIR_AVAILABLE:
            print("Thermal Camera:", "FAIL - FLIR SDK NOT INSTALLED")
            print("  → FLIR Spinnaker SDK is required but not found")
            print("  → Download from: https://www.flir.com/products/spinnaker-sdk/")
        else:
            print("Thermal Camera:", "OK" if thermal_ok else "FAIL")
            if FLIR_VERSION:
                print(f"  → FLIR SDK Version: {FLIR_VERSION}")

    print("GSR Sensor:", "OK" if gsr_ok else "FAIL")

    if available_ports:
        print("\nAvailable Serial Ports:")
        for port in available_ports:
            print(f"  - {port}")

        if not config.GSR_SIMULATION_MODE and config.GSR_SENSOR_PORT not in available_ports:
            print(f"\nWARNING: Configured GSR port '{config.GSR_SENSOR_PORT}' not found in available ports.")
            print(f"Update GSR_SENSOR_PORT in src/config.py to one of the available ports.")
    else:
        print("\nNo serial ports detected.")
        if not config.GSR_SIMULATION_MODE:
            print("Consider setting GSR_SIMULATION_MODE = True in src/config.py if no GSR sensor is available.")

    if all([dep_ok, dirs_ok, rgb_ok, thermal_ok, gsr_ok]):
        logging.info(
            "SUCCESS: System is configured correctly and ready for data collection."
        )
        sys.exit(0)  # Exit with success code
    else:
        logging.error(
            "FAIL: One or more system checks failed. Please review the log messages above."
        )
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()