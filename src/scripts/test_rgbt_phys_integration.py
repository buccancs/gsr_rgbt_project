#!/usr/bin/env python3
# src/scripts/test_rgbt_phys_integration.py

import logging
import sys
import time
import os
from pathlib import Path
import numpy as np
import cv2

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.capture.rgbt_phys_capture import RGBTPhysCaptureThread
from PyQt5.QtCore import QCoreApplication

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class RGBTPhysIntegrationTester:
    """
    Tests the integration of RGBTPhys_CPP with the Python framework by capturing
    a short sequence of synchronized data from RGB camera, thermal camera, and
    physiological sensors.
    """

    def __init__(self, test_duration=5, simulation_mode=True):
        """
        Initialize the integration tester.

        Args:
            test_duration (int): The duration of the test in seconds.
            simulation_mode (bool): Whether to run in simulation mode or with real hardware.
        """
        self.test_duration = test_duration
        self.simulation_mode = simulation_mode
        
        # Create output directory for test data
        self.output_dir = os.path.join(project_root, "data", "test_rgbt_phys")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the RGBTPhys capture thread
        self.rgbt_phys = RGBTPhysCaptureThread(
            config_file="default_config",
            base_save_path=self.output_dir,
            participant_id="test_subject",
            simulation_mode=self.simulation_mode
        )
        
        # Counters for received frames/data
        self.rgb_frames_received = 0
        self.thermal_frames_received = 0
        self.phys_data_received = 0
        
        # Lists to store timestamps for analysis
        self.rgb_timestamps = []
        self.thermal_timestamps = []
        self.phys_timestamps = []

    def collect_rgb_frame(self, frame, timestamp):
        """
        Collect RGB frames and timestamps for analysis.

        Args:
            frame: The RGB frame.
            timestamp: The timestamp of the frame.
        """
        self.rgb_frames_received += 1
        self.rgb_timestamps.append(timestamp)
        logging.info(f"Received RGB frame {self.rgb_frames_received} with timestamp {timestamp}")
        
        # Save the frame to disk (every 10th frame to avoid too many files)
        if self.rgb_frames_received % 10 == 0:
            filename = os.path.join(self.output_dir, f"rgb_frame_{self.rgb_frames_received}.jpg")
            cv2.imwrite(filename, frame)
            logging.info(f"Saved RGB frame to {filename}")

    def collect_thermal_frame(self, frame, timestamp):
        """
        Collect thermal frames and timestamps for analysis.

        Args:
            frame: The thermal frame.
            timestamp: The timestamp of the frame.
        """
        self.thermal_frames_received += 1
        self.thermal_timestamps.append(timestamp)
        logging.info(f"Received thermal frame {self.thermal_frames_received} with timestamp {timestamp}")
        
        # Save the frame to disk (every 10th frame to avoid too many files)
        if self.thermal_frames_received % 10 == 0:
            filename = os.path.join(self.output_dir, f"thermal_frame_{self.thermal_frames_received}.jpg")
            cv2.imwrite(filename, frame)
            logging.info(f"Saved thermal frame to {filename}")

    def collect_phys_data(self, data, timestamp):
        """
        Collect physiological data and timestamps for analysis.

        Args:
            data: The physiological data.
            timestamp: The timestamp of the data.
        """
        self.phys_data_received += 1
        self.phys_timestamps.append(timestamp)
        logging.info(f"Received physiological data {self.phys_data_received} with timestamp {timestamp}")
        logging.info(f"Data: {data}")

    def run_test(self):
        """
        Run the integration test.

        Returns:
            True if the test passes, False otherwise.
        """
        try:
            # Connect signals to data collection methods
            self.rgbt_phys.rgb_frame_captured.connect(self.collect_rgb_frame)
            self.rgbt_phys.thermal_frame_captured.connect(self.collect_thermal_frame)
            self.rgbt_phys.phys_data_captured.connect(self.collect_phys_data)
            
            # Start the capture thread
            logging.info("Starting RGBTPhys capture thread")
            self.rgbt_phys.start()
            
            # Wait for the specified duration
            logging.info(f"Capturing data for {self.test_duration} seconds...")
            time.sleep(self.test_duration)
            
            # Stop the capture thread
            self.rgbt_phys.stop()
            
            # Wait for the thread to finish
            if self.rgbt_phys.isRunning():
                self.rgbt_phys.wait()
            
            # Analyze the results
            return self.analyze_results()
            
        except Exception as e:
            logging.error(f"Error during integration test: {e}")
            return False

    def analyze_results(self):
        """
        Analyze the test results to verify integration.

        Returns:
            True if the integration is working properly, False otherwise.
        """
        logging.info("Analyzing integration test results...")
        
        # Check if we received any data
        if not self.rgb_timestamps or not self.thermal_timestamps or not self.phys_timestamps:
            logging.error("FAIL: No data received from one or more sources.")
            return False
        
        # Log the number of frames/data points received
        logging.info(f"RGB frames received: {self.rgb_frames_received}")
        logging.info(f"Thermal frames received: {self.thermal_frames_received}")
        logging.info(f"Physiological data points received: {self.phys_data_received}")
        
        # Calculate frame rates
        if len(self.rgb_timestamps) > 1:
            rgb_fps = (len(self.rgb_timestamps) - 1) / ((self.rgb_timestamps[-1] - self.rgb_timestamps[0]) / 1e9)
            logging.info(f"RGB camera frame rate: {rgb_fps:.2f} fps")
        
        if len(self.thermal_timestamps) > 1:
            thermal_fps = (len(self.thermal_timestamps) - 1) / ((self.thermal_timestamps[-1] - self.thermal_timestamps[0]) / 1e9)
            logging.info(f"Thermal camera frame rate: {thermal_fps:.2f} fps")
        
        if len(self.phys_timestamps) > 1:
            phys_rate = (len(self.phys_timestamps) - 1) / ((self.phys_timestamps[-1] - self.phys_timestamps[0]) / 1e9)
            logging.info(f"Physiological data rate: {phys_rate:.2f} Hz")
        
        # Check synchronization by comparing timestamps
        # In a perfectly synchronized system, the timestamps should be very close to each other
        # For this test, we'll just check if we're receiving data from all sources
        
        # Final result
        if self.rgb_frames_received > 0 and self.thermal_frames_received > 0 and self.phys_data_received > 0:
            logging.info("SUCCESS: RGBTPhys integration is working properly.")
            return True
        else:
            logging.error("FAIL: RGBTPhys integration issues detected.")
            return False


def main():
    """
    Run the RGBTPhys integration test.
    """
    # Create a QApplication instance (required for Qt signals/slots)
    app = QCoreApplication(sys.argv)
    
    logging.info("=======================================")
    logging.info("=  RGBTPhys Integration Test          =")
    logging.info("=======================================")
    
    # Run the integration test in simulation mode by default
    # Change to False to test with real hardware
    simulation_mode = True
    
    tester = RGBTPhysIntegrationTester(test_duration=5, simulation_mode=simulation_mode)
    integration_ok = tester.run_test()
    
    # Final Summary
    print("\n--- RGBTPhys Integration Test Summary ---")
    print("Integration:", "OK" if integration_ok else "FAIL")
    
    if integration_ok:
        logging.info("SUCCESS: Integration test passed.")
        sys.exit(0)  # Exit with success code
    else:
        logging.error("FAIL: Integration test failed.")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()