# src/scripts/check_system.py

import logging
import sys
from pathlib import Path

import cv2

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config

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
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
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


def main():
    """
    Runs all system validation checks.
    """
    logging.info("=======================================")
    logging.info("=  GSR-RGBT System Validation Check   =")
    logging.info("=======================================")

    # Run all checks
    dep_ok = check_dependencies()
    dirs_ok = check_directories()
    rgb_ok = check_camera(config.RGB_CAMERA_ID, "RGB")
    thermal_ok = check_camera(config.THERMAL_CAMERA_ID, "Thermal")

    # Final Summary
    print("\n--- Validation Summary ---")
    if all([dep_ok, dirs_ok, rgb_ok, thermal_ok]):
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
