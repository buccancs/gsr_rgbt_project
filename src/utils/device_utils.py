# src/utils/device_utils.py
import serial.tools.list_ports
from typing import Optional

class DeviceNotFoundError(Exception):
    """Custom exception for when a required device is not found."""
    pass

def find_shimmer_com_port() -> str:
    """
    Scans available COM ports and returns the one connected to a Shimmer device.

    This function iterates through all connected serial devices and checks their
    description for the keyword "Shimmer".

    Returns:
        The COM port name (e.g., "COM3").

    Raises:
        DeviceNotFoundError: If no Shimmer device is found on any COM port.
    """
    print("Scanning for Shimmer device...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # The description for Shimmer devices usually contains "Shimmer".
        # This is more reliable than checking for "Bluetooth" which is too generic.
        if port.description and "Shimmer" in port.description:
            print(f"Found Shimmer device: {port.description} on port {port.device}")
            return port.device

    # If the loop completes without finding the device, raise an error.
    raise DeviceNotFoundError(
        "Could not find a Shimmer device. Please ensure it is paired via "
        "Bluetooth and powered on."
    )