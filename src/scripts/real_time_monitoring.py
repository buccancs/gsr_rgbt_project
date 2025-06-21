#!/usr/bin/env python3
# src/scripts/real_time_monitoring.py

import logging
import sys
import time
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt5.QtCore import QCoreApplication, QTimer, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QGridLayout
from PyQt5.QtGui import QImage, QPixmap

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the integration classes for each repository
from src.capture.factorize_phys_capture import FactorizePhysCaptureThread
from src.capture.tc001_thermal_capture import TC001ThermalCaptureThread
from src.processing.mmrphys_processor import MMRPhysProcessor

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class RealTimeMonitoringApp(QMainWindow):
    """
    A real-time monitoring application that integrates FactorizePhys, TC001_SAMCL, and MMRPhys
    to provide live physiological monitoring.
    
    This application demonstrates:
    1. Real-time capture of RGB, thermal, and physiological data
    2. Live segmentation of thermal regions of interest
    3. Real-time extraction and visualization of physiological signals
    """
    
    def __init__(self, args):
        """
        Initialize the real-time monitoring application.
        
        Args:
            args: Command-line arguments.
        """
        super().__init__()
        self.args = args
        self.output_dir = os.path.join(project_root, "data", "real_time_monitoring")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup UI
        self._setup_ui()
        
        # Initialize the capture and processing components
        self._initialize_components()
        
        # Data buffers for real-time visualization
        self.buffer_size = 300  # 10 seconds at 30 fps
        self._initialize_data_buffers()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(33)  # ~30 fps
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = None
        
    def _setup_ui(self):
        """
        Setup the user interface.
        """
        self.setWindowTitle("Real-Time Physiological Monitoring")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Video display area
        video_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        
        # RGB video display
        self.rgb_label = QLabel("RGB Video")
        self.rgb_label.setMinimumSize(400, 300)
        self.rgb_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.rgb_label)
        
        # Thermal video display
        self.thermal_label = QLabel("Thermal Video")
        self.thermal_label.setMinimumSize(400, 300)
        self.thermal_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.thermal_label)
        
        # Segmentation mask display
        self.mask_label = QLabel("Segmentation Mask")
        self.mask_label.setMinimumSize(400, 300)
        self.mask_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.mask_label)
        
        # Signal display area
        signal_layout = QGridLayout()
        main_layout.addLayout(signal_layout)
        
        # Heart rate display
        self.heart_rate_label = QLabel("Heart Rate: -- BPM")
        self.heart_rate_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        signal_layout.addWidget(self.heart_rate_label, 0, 0)
        
        # GSR display
        self.gsr_label = QLabel("GSR: -- µS")
        self.gsr_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        signal_layout.addWidget(self.gsr_label, 0, 1)
        
        # Temperature display
        self.temp_label = QLabel("ROI Temp: -- °C")
        self.temp_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        signal_layout.addWidget(self.temp_label, 0, 2)
        
        # Signal quality display
        self.quality_label = QLabel("Signal Quality: -- %")
        self.quality_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        signal_layout.addWidget(self.quality_label, 0, 3)
        
        # Control buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        
        # Start/Stop recording button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self._toggle_recording)
        button_layout.addWidget(self.record_button)
        
        # Save snapshot button
        self.snapshot_button = QPushButton("Save Snapshot")
        self.snapshot_button.clicked.connect(self._save_snapshot)
        button_layout.addWidget(self.snapshot_button)
        
        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)
        
    def _initialize_components(self):
        """
        Initialize the capture and processing components.
        """
        # Initialize FactorizePhys for synchronized data capture
        self.factorize_phys = FactorizePhysCaptureThread(
            config_file=self.args.factorize_config,
            base_save_path=self.output_dir,
            participant_id=self.args.participant_id,
            simulation_mode=self.args.simulation
        )
        
        # Initialize TC001 thermal capture with segmentation
        self.tc001_thermal = TC001ThermalCaptureThread(
            device_id=self.args.thermal_device,
            config_file=self.args.tc001_config,
            simulation_mode=self.args.simulation
        )
        
        # Initialize MMRPhys processor for physiological signal extraction
        self.mmrphys = MMRPhysProcessor(
            model_type=self.args.mmrphys_model,
            model_path=self.args.mmrphys_weights,
            use_gpu=not self.args.no_gpu
        )
        
        # Connect signals to data collection methods
        self._connect_signals()
        
        # Start the capture threads
        self.factorize_phys.start()
        self.tc001_thermal.start()
        
    def _initialize_data_buffers(self):
        """
        Initialize data buffers for real-time visualization.
        """
        # Frame buffers
        self.latest_rgb_frame = None
        self.latest_thermal_frame = None
        self.latest_mask = None
        
        # Signal buffers
        self.heart_rate_buffer = []
        self.gsr_buffer = []
        self.temp_buffer = []
        self.quality_buffer = []
        
        # Timestamp buffers
        self.heart_rate_timestamps = []
        self.gsr_timestamps = []
        self.temp_timestamps = []
        self.quality_timestamps = []
        
        # Latest values
        self.latest_heart_rate = 0
        self.latest_gsr = 0
        self.latest_temp = 0
        self.latest_quality = 0
        
    def _connect_signals(self):
        """
        Connect signals from capture threads to data collection methods.
        """
        # Connect FactorizePhys signals
        self.factorize_phys.rgb_frame_captured.connect(self._process_rgb_frame)
        self.factorize_phys.thermal_frame_captured.connect(self._process_thermal_frame)
        self.factorize_phys.phys_data_captured.connect(self._process_phys_data)
        self.factorize_phys.factorized_signals.connect(self._process_factorized_signals)
        
        # Connect TC001 thermal signals
        self.tc001_thermal.frame_captured.connect(self._process_tc001_frame)
        self.tc001_thermal.segmentation_captured.connect(self._process_segmentation_mask)
        self.tc001_thermal.roi_signal_captured.connect(self._process_roi_signal)
    
    @pyqtSlot(np.ndarray, float)
    def _process_rgb_frame(self, frame, timestamp):
        """
        Process RGB frames from FactorizePhys.
        """
        self.latest_rgb_frame = frame
        
        # If recording, save the frame
        if self.is_recording:
            frame_path = os.path.join(
                self.output_dir, 
                f"recording_{self.recording_start_time}_rgb_{timestamp}.jpg"
            )
            cv2.imwrite(frame_path, frame)
    
    @pyqtSlot(np.ndarray, float)
    def _process_thermal_frame(self, frame, timestamp):
        """
        Process thermal frames from FactorizePhys.
        """
        self.latest_thermal_frame = frame
        
        # If recording, save the frame
        if self.is_recording:
            frame_path = os.path.join(
                self.output_dir, 
                f"recording_{self.recording_start_time}_thermal_{timestamp}.jpg"
            )
            cv2.imwrite(frame_path, frame)
    
    @pyqtSlot(np.ndarray, float)
    def _process_tc001_frame(self, frame, timestamp):
        """
        Process thermal frames from TC001.
        """
        # We already have thermal frames from FactorizePhys, so we'll just use those
        pass
    
    @pyqtSlot(np.ndarray, float)
    def _process_segmentation_mask(self, mask, timestamp):
        """
        Process segmentation masks from TC001.
        """
        self.latest_mask = mask
        
        # If recording, save the mask
        if self.is_recording:
            mask_path = os.path.join(
                self.output_dir, 
                f"recording_{self.recording_start_time}_mask_{timestamp}.png"
            )
            cv2.imwrite(mask_path, mask)
    
    @pyqtSlot(object, float)
    def _process_phys_data(self, data, timestamp):
        """
        Process physiological data from FactorizePhys.
        """
        # Extract GSR value
        gsr = data.get('gsr', 0)
        
        # Update GSR buffer
        self.gsr_buffer.append(gsr)
        self.gsr_timestamps.append(timestamp)
        
        # Keep buffer at the specified size
        if len(self.gsr_buffer) > self.buffer_size:
            self.gsr_buffer.pop(0)
            self.gsr_timestamps.pop(0)
        
        # Update latest value
        self.latest_gsr = gsr
        
        # If recording, save the data
        if self.is_recording:
            with open(os.path.join(self.output_dir, f"recording_{self.recording_start_time}_phys.csv"), 'a') as f:
                f.write(f"{timestamp},{gsr}\n")
    
    @pyqtSlot(object, float)
    def _process_factorized_signals(self, signals, timestamp):
        """
        Process factorized signals from FactorizePhys.
        """
        # Extract heart rate and quality
        pulse_signal = signals.get('pulse_signal', 0)
        quality_index = signals.get('quality_index', 0)
        
        # Convert pulse signal to heart rate (BPM)
        heart_rate = pulse_signal * 60
        
        # Update heart rate buffer
        self.heart_rate_buffer.append(heart_rate)
        self.heart_rate_timestamps.append(timestamp)
        
        # Update quality buffer
        self.quality_buffer.append(quality_index)
        self.quality_timestamps.append(timestamp)
        
        # Keep buffers at the specified size
        if len(self.heart_rate_buffer) > self.buffer_size:
            self.heart_rate_buffer.pop(0)
            self.heart_rate_timestamps.pop(0)
        
        if len(self.quality_buffer) > self.buffer_size:
            self.quality_buffer.pop(0)
            self.quality_timestamps.pop(0)
        
        # Update latest values
        self.latest_heart_rate = heart_rate
        self.latest_quality = quality_index
        
        # If recording, save the data
        if self.is_recording:
            with open(os.path.join(self.output_dir, f"recording_{self.recording_start_time}_factorized.csv"), 'a') as f:
                f.write(f"{timestamp},{heart_rate},{quality_index}\n")
    
    @pyqtSlot(object, float)
    def _process_roi_signal(self, signal, timestamp):
        """
        Process ROI signals from TC001.
        """
        # Extract temperature
        temp = signal.get('mean_temperature', 0)
        
        # Update temperature buffer
        self.temp_buffer.append(temp)
        self.temp_timestamps.append(timestamp)
        
        # Keep buffer at the specified size
        if len(self.temp_buffer) > self.buffer_size:
            self.temp_buffer.pop(0)
            self.temp_timestamps.pop(0)
        
        # Update latest value
        self.latest_temp = temp
        
        # If recording, save the data
        if self.is_recording:
            with open(os.path.join(self.output_dir, f"recording_{self.recording_start_time}_roi.csv"), 'a') as f:
                f.write(f"{timestamp},{temp}\n")
    
    def _update_ui(self):
        """
        Update the UI with the latest data.
        """
        # Update video displays
        self._update_video_displays()
        
        # Update signal displays
        self._update_signal_displays()
    
    def _update_video_displays(self):
        """
        Update the video display widgets with the latest frames.
        """
        # Update RGB display
        if self.latest_rgb_frame is not None:
            rgb_qimg = self._convert_cv_to_qimage(self.latest_rgb_frame)
            self.rgb_label.setPixmap(QPixmap.fromImage(rgb_qimg).scaled(
                self.rgb_label.width(), self.rgb_label.height(), 
                aspectRatioMode=1
            ))
        
        # Update thermal display
        if self.latest_thermal_frame is not None:
            thermal_qimg = self._convert_cv_to_qimage(self.latest_thermal_frame)
            self.thermal_label.setPixmap(QPixmap.fromImage(thermal_qimg).scaled(
                self.thermal_label.width(), self.thermal_label.height(), 
                aspectRatioMode=1
            ))
        
        # Update mask display
        if self.latest_mask is not None:
            # Convert single-channel mask to RGB for display
            mask_rgb = cv2.cvtColor(self.latest_mask, cv2.COLOR_GRAY2RGB)
            mask_qimg = self._convert_cv_to_qimage(mask_rgb)
            self.mask_label.setPixmap(QPixmap.fromImage(mask_qimg).scaled(
                self.mask_label.width(), self.mask_label.height(), 
                aspectRatioMode=1
            ))
    
    def _update_signal_displays(self):
        """
        Update the signal display widgets with the latest values.
        """
        # Update heart rate display
        self.heart_rate_label.setText(f"Heart Rate: {self.latest_heart_rate:.1f} BPM")
        
        # Update GSR display
        self.gsr_label.setText(f"GSR: {self.latest_gsr:.3f} µS")
        
        # Update temperature display
        self.temp_label.setText(f"ROI Temp: {self.latest_temp:.1f}")
        
        # Update quality display
        quality_percent = self.latest_quality * 100
        self.quality_label.setText(f"Signal Quality: {quality_percent:.1f}%")
    
    def _convert_cv_to_qimage(self, cv_img):
        """
        Convert an OpenCV image to a QImage for display.
        
        Args:
            cv_img (numpy.ndarray): The OpenCV image.
            
        Returns:
            QImage: The converted image.
        """
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def _toggle_recording(self):
        """
        Toggle recording state.
        """
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.recording_start_time = int(time.time())
            self.record_button.setText("Stop Recording")
            logging.info(f"Started recording with ID {self.recording_start_time}")
            
            # Create recording directory
            recording_dir = os.path.join(self.output_dir, f"recording_{self.recording_start_time}")
            os.makedirs(recording_dir, exist_ok=True)
            
            # Initialize CSV files with headers
            with open(os.path.join(self.output_dir, f"recording_{self.recording_start_time}_phys.csv"), 'w') as f:
                f.write("timestamp,gsr\n")
            
            with open(os.path.join(self.output_dir, f"recording_{self.recording_start_time}_factorized.csv"), 'w') as f:
                f.write("timestamp,heart_rate,quality_index\n")
            
            with open(os.path.join(self.output_dir, f"recording_{self.recording_start_time}_roi.csv"), 'w') as f:
                f.write("timestamp,temperature\n")
            
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.setText("Start Recording")
            logging.info(f"Stopped recording with ID {self.recording_start_time}")
            
            # Process the recorded data with MMRPhys
            self._process_recording()
    
    def _save_snapshot(self):
        """
        Save a snapshot of the current state.
        """
        snapshot_time = int(time.time())
        snapshot_dir = os.path.join(self.output_dir, f"snapshot_{snapshot_time}")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Save frames
        if self.latest_rgb_frame is not None:
            cv2.imwrite(os.path.join(snapshot_dir, "rgb.jpg"), self.latest_rgb_frame)
        
        if self.latest_thermal_frame is not None:
            cv2.imwrite(os.path.join(snapshot_dir, "thermal.jpg"), self.latest_thermal_frame)
        
        if self.latest_mask is not None:
            cv2.imwrite(os.path.join(snapshot_dir, "mask.png"), self.latest_mask)
        
        # Save current values
        with open(os.path.join(snapshot_dir, "values.txt"), 'w') as f:
            f.write(f"Heart Rate: {self.latest_heart_rate:.1f} BPM\n")
            f.write(f"GSR: {self.latest_gsr:.3f} µS\n")
            f.write(f"ROI Temperature: {self.latest_temp:.1f}\n")
            f.write(f"Signal Quality: {self.latest_quality * 100:.1f}%\n")
        
        logging.info(f"Saved snapshot to {snapshot_dir}")
    
    def _process_recording(self):
        """
        Process the recorded data with MMRPhys.
        """
        if not self.recording_start_time:
            return
        
        recording_dir = os.path.join(self.output_dir, f"recording_{self.recording_start_time}")
        
        # Check if we have RGB frames to process
        rgb_frames = [f for f in os.listdir(self.output_dir) if f.startswith(f"recording_{self.recording_start_time}_rgb_")]
        
        if not rgb_frames:
            logging.warning("No RGB frames found for processing with MMRPhys")
            return
        
        logging.info("Processing recorded frames with MMRPhys...")
        
        # Sort frames by timestamp
        rgb_frames.sort()
        
        # Create a video from the frames
        video_path = os.path.join(recording_dir, "rgb_video.avi")
        
        # Load the first frame to get dimensions
        first_frame = cv2.imread(os.path.join(self.output_dir, rgb_frames[0]))
        height, width, _ = first_frame.shape
        
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        # Write frames to the video file
        for frame_file in rgb_frames:
            frame = cv2.imread(os.path.join(self.output_dir, frame_file))
            out.write(frame)
        
        out.release()
        
        # Process the video with MMRPhys
        results = self.mmrphys.process_video(
            video_path,
            output_dir=recording_dir
        )
        
        # Save the results
        if results and 'heart_rate' in results:
            with open(os.path.join(recording_dir, "mmrphys_results.txt"), 'w') as f:
                f.write(f"MMRPhys Heart Rate: {results['heart_rate']:.1f} BPM\n")
            
            logging.info(f"MMRPhys estimated heart rate: {results['heart_rate']:.1f} BPM")
    
    def closeEvent(self, event):
        """
        Handle the window close event.
        """
        # Stop the capture threads
        self.factorize_phys.stop()
        self.tc001_thermal.stop()
        
        # Wait for threads to finish
        if self.factorize_phys.isRunning():
            self.factorize_phys.wait()
        if self.tc001_thermal.isRunning():
            self.tc001_thermal.wait()
        
        # Accept the close event
        event.accept()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Real-Time Physiological Monitoring")
    
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode without real hardware")
    parser.add_argument("--participant-id", type=str, default="real_time_subject",
                        help="Participant ID for data organization")
    
    # FactorizePhys options
    parser.add_argument("--factorize-config", type=str, default=None,
                        help="Path to FactorizePhys configuration file")
    
    # TC001 options
    parser.add_argument("--thermal-device", type=int, default=0,
                        help="Device ID for the TOPDON TC001 thermal camera")
    parser.add_argument("--tc001-config", type=str, default=None,
                        help="Path to TC001_SAMCL configuration file")
    
    # MMRPhys options
    parser.add_argument("--mmrphys-model", type=str, default="MMRPhysLEF",
                        choices=["MMRPhysLEF", "MMRPhysMEF", "MMRPhysSEF"],
                        help="MMRPhys model type")
    parser.add_argument("--mmrphys-weights", type=str, default=None,
                        help="Path to pre-trained MMRPhys model weights")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU usage for MMRPhys inference")
    
    return parser.parse_args()


def main():
    """
    Main function to run the real-time monitoring application.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create a QApplication instance
    app = QApplication(sys.argv)
    
    logging.info("=======================================")
    logging.info("= Real-Time Physiological Monitoring =")
    logging.info("=======================================")
    
    # Create and show the application window
    window = RealTimeMonitoringApp(args)
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()