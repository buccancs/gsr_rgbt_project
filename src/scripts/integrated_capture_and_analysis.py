#!/usr/bin/env python3
# src/scripts/integrated_capture_and_analysis.py

import logging
import sys
import time
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# --- Add project root to path for absolute imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the integration classes for each repository
from src.capture.rgbt_phys_capture import RGBTPhysCaptureThread
from src.capture.factorize_phys_capture import FactorizePhysCaptureThread
from src.capture.tc001_thermal_capture import TC001ThermalCaptureThread
from src.processing.mmrphys_processor import MMRPhysProcessor

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class IntegratedCaptureAndAnalysis:
    """
    A class that demonstrates the end-to-end workflow of capturing data with FactorizePhys,
    processing thermal data with TC001_SAMCL, and extracting physiological signals with MMRPhys.
    """
    
    def __init__(self, args):
        """
        Initialize the integrated capture and analysis process.
        
        Args:
            args: Command-line arguments.
        """
        self.args = args
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            project_root, 
            "data", 
            f"{args.participant_id}_{self.session_id}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        self.rgb_dir = os.path.join(self.output_dir, "rgb")
        self.thermal_dir = os.path.join(self.output_dir, "thermal")
        self.phys_dir = os.path.join(self.output_dir, "phys")
        self.segmentation_dir = os.path.join(self.output_dir, "segmentation")
        self.mmrphys_dir = os.path.join(self.output_dir, "mmrphys")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.thermal_dir, exist_ok=True)
        os.makedirs(self.phys_dir, exist_ok=True)
        os.makedirs(self.segmentation_dir, exist_ok=True)
        os.makedirs(self.mmrphys_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Initialize data collection
        self.rgb_frames = []
        self.thermal_frames = []
        self.phys_data = []
        self.segmentation_masks = []
        
        # Initialize timestamps
        self.rgb_timestamps = []
        self.thermal_timestamps = []
        self.phys_timestamps = []
        self.segmentation_timestamps = []
        
        # Initialize capture components
        self._initialize_capture_components()
        
        # Initialize processing components
        self._initialize_processing_components()
        
    def _initialize_capture_components(self):
        """
        Initialize the capture components.
        """
        # Choose the appropriate capture method based on availability
        if self.args.use_factorize:
            logging.info("Using FactorizePhys for data capture")
            self.capture_thread = FactorizePhysCaptureThread(
                config_file=self.args.factorize_config,
                base_save_path=self.output_dir,
                participant_id=self.args.participant_id,
                simulation_mode=self.args.simulation
            )
        else:
            logging.info("Using RGBTPhys_CPP for data capture")
            self.capture_thread = RGBTPhysCaptureThread(
                config_file=self.args.rgbtphys_config,
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
        
    def _initialize_processing_components(self):
        """
        Initialize the processing components.
        """
        # Initialize MMRPhys processor for physiological signal extraction
        self.mmrphys = MMRPhysProcessor(
            model_type=self.args.mmrphys_model,
            model_path=self.args.mmrphys_weights,
            use_gpu=not self.args.no_gpu
        )
        
    def run(self):
        """
        Run the integrated capture and analysis process.
        """
        try:
            # Step 1: Capture synchronized data
            self._capture_data()
            
            # Step 2: Process thermal data with TC001_SAMCL
            self._process_thermal_data()
            
            # Step 3: Extract physiological signals with MMRPhys
            self._extract_physiological_signals()
            
            # Step 4: Analyze and visualize the results
            self._analyze_results()
            
            logging.info("Integrated capture and analysis completed successfully")
            
        except Exception as e:
            logging.error(f"Error during integrated capture and analysis: {e}")
            raise
        
    def _capture_data(self):
        """
        Capture synchronized RGB, thermal, and physiological data.
        """
        logging.info("Starting data capture...")
        
        # Connect signals to data collection methods
        self.capture_thread.rgb_frame_captured.connect(self._collect_rgb_frame)
        self.capture_thread.thermal_frame_captured.connect(self._collect_thermal_frame)
        self.capture_thread.phys_data_captured.connect(self._collect_phys_data)
        
        # Connect TC001 thermal signals
        self.tc001_thermal.frame_captured.connect(self._collect_tc001_frame)
        self.tc001_thermal.segmentation_captured.connect(self._collect_segmentation_mask)
        
        # Start the capture threads
        self.capture_thread.start()
        self.tc001_thermal.start()
        
        # Wait for the specified duration
        logging.info(f"Capturing data for {self.args.duration} seconds...")
        time.sleep(self.args.duration)
        
        # Stop the capture threads
        self.capture_thread.stop()
        self.tc001_thermal.stop()
        
        # Wait for threads to finish
        if self.capture_thread.isRunning():
            self.capture_thread.wait()
        if self.tc001_thermal.isRunning():
            self.tc001_thermal.wait()
        
        logging.info("Data capture completed")
        
        # Save metadata
        self._save_capture_metadata()
        
    def _collect_rgb_frame(self, frame, timestamp):
        """
        Collect RGB frames and timestamps.
        
        Args:
            frame: The RGB frame.
            timestamp: The timestamp of the frame.
        """
        self.rgb_frames.append(frame)
        self.rgb_timestamps.append(timestamp)
        
        # Save the frame to disk
        frame_path = os.path.join(self.rgb_dir, f"rgb_{len(self.rgb_frames):06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
    def _collect_thermal_frame(self, frame, timestamp):
        """
        Collect thermal frames and timestamps.
        
        Args:
            frame: The thermal frame.
            timestamp: The timestamp of the frame.
        """
        self.thermal_frames.append(frame)
        self.thermal_timestamps.append(timestamp)
        
        # Save the frame to disk
        frame_path = os.path.join(self.thermal_dir, f"thermal_{len(self.thermal_frames):06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
    def _collect_tc001_frame(self, frame, timestamp):
        """
        Collect thermal frames from TC001.
        
        Args:
            frame: The thermal frame.
            timestamp: The timestamp of the frame.
        """
        # We already have thermal frames from the main capture thread
        pass
        
    def _collect_phys_data(self, data, timestamp):
        """
        Collect physiological data and timestamps.
        
        Args:
            data: The physiological data.
            timestamp: The timestamp of the data.
        """
        self.phys_data.append(data)
        self.phys_timestamps.append(timestamp)
        
        # Save the data to disk
        with open(os.path.join(self.phys_dir, "phys_data.csv"), 'a') as f:
            if len(self.phys_data) == 1:
                # Write header
                f.write("timestamp,")
                f.write(",".join(data.keys()))
                f.write("\n")
            
            # Write data
            f.write(f"{timestamp},")
            f.write(",".join([str(v) for v in data.values()]))
            f.write("\n")
            
    def _collect_segmentation_mask(self, mask, timestamp):
        """
        Collect segmentation masks and timestamps.
        
        Args:
            mask: The segmentation mask.
            timestamp: The timestamp of the mask.
        """
        self.segmentation_masks.append(mask)
        self.segmentation_timestamps.append(timestamp)
        
        # Save the mask to disk
        mask_path = os.path.join(self.segmentation_dir, f"mask_{len(self.segmentation_masks):06d}.png")
        cv2.imwrite(mask_path, mask)
        
    def _save_capture_metadata(self):
        """
        Save metadata about the captured data.
        """
        metadata = {
            "session_id": self.session_id,
            "participant_id": self.args.participant_id,
            "duration": self.args.duration,
            "rgb_frames": len(self.rgb_frames),
            "thermal_frames": len(self.thermal_frames),
            "phys_data_points": len(self.phys_data),
            "segmentation_masks": len(self.segmentation_masks),
            "capture_start_time": min(self.rgb_timestamps + self.thermal_timestamps + self.phys_timestamps),
            "capture_end_time": max(self.rgb_timestamps + self.thermal_timestamps + self.phys_timestamps),
        }
        
        # Save RGB timestamps
        pd.DataFrame({
            "frame_number": range(len(self.rgb_timestamps)),
            "timestamp": self.rgb_timestamps
        }).to_csv(os.path.join(self.rgb_dir, "timestamps.csv"), index=False)
        
        # Save thermal timestamps
        pd.DataFrame({
            "frame_number": range(len(self.thermal_timestamps)),
            "timestamp": self.thermal_timestamps
        }).to_csv(os.path.join(self.thermal_dir, "timestamps.csv"), index=False)
        
        # Save segmentation timestamps
        pd.DataFrame({
            "frame_number": range(len(self.segmentation_timestamps)),
            "timestamp": self.segmentation_timestamps
        }).to_csv(os.path.join(self.segmentation_dir, "timestamps.csv"), index=False)
        
        # Save metadata
        with open(os.path.join(self.output_dir, "metadata.txt"), 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
                
    def _process_thermal_data(self):
        """
        Process thermal data with TC001_SAMCL to extract regions of interest.
        """
        logging.info("Processing thermal data with TC001_SAMCL...")
        
        # Check if we have thermal frames to process
        if not self.thermal_frames:
            logging.warning("No thermal frames to process")
            return
        
        # Create a video from the thermal frames
        thermal_video_path = os.path.join(self.thermal_dir, "thermal_video.avi")
        
        # Get dimensions from the first frame
        height, width = self.thermal_frames[0].shape[:2]
        
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(thermal_video_path, fourcc, 30.0, (width, height))
        
        # Write frames to the video file
        for frame in self.thermal_frames:
            out.write(frame)
        
        out.release()
        
        # Process the thermal video with TC001_SAMCL
        # This would typically be done by calling the TC001_SAMCL processing functions
        # For now, we'll just use the segmentation masks we already collected
        
        # Extract ROIs from the segmentation masks
        self._extract_rois_from_masks()
        
        logging.info("Thermal data processing completed")
        
    def _extract_rois_from_masks(self):
        """
        Extract regions of interest from the segmentation masks.
        """
        logging.info("Extracting ROIs from segmentation masks...")
        
        # Check if we have segmentation masks
        if not self.segmentation_masks:
            logging.warning("No segmentation masks to process")
            return
        
        # Create a directory for ROI data
        roi_dir = os.path.join(self.segmentation_dir, "rois")
        os.makedirs(roi_dir, exist_ok=True)
        
        # Process each mask to extract ROIs
        roi_data = []
        
        for i, mask in enumerate(self.segmentation_masks):
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour as a potential ROI
            for j, contour in enumerate(contours):
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Skip small contours
                if area < 100:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create a mask for this ROI
                roi_mask = np.zeros_like(mask)
                cv2.drawContours(roi_mask, [contour], 0, 255, -1)
                
                # Save the ROI mask
                roi_mask_path = os.path.join(roi_dir, f"roi_{i:06d}_{j:02d}.png")
                cv2.imwrite(roi_mask_path, roi_mask)
                
                # Get the corresponding thermal frame
                if i < len(self.thermal_frames):
                    thermal_frame = self.thermal_frames[i]
                    
                    # Extract temperature data from the ROI
                    # In a real implementation, this would use actual temperature data
                    # For now, we'll just use pixel values as a proxy
                    roi_pixels = thermal_frame[roi_mask > 0]
                    mean_temp = np.mean(roi_pixels) if len(roi_pixels) > 0 else 0
                    
                    # Add to ROI data
                    roi_data.append({
                        "frame": i,
                        "roi_id": j,
                        "area": area,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "mean_temperature": mean_temp,
                        "timestamp": self.thermal_timestamps[i] if i < len(self.thermal_timestamps) else 0
                    })
        
        # Save ROI data
        pd.DataFrame(roi_data).to_csv(os.path.join(roi_dir, "roi_data.csv"), index=False)
        
        logging.info(f"Extracted {len(roi_data)} ROIs from {len(self.segmentation_masks)} masks")
        
    def _extract_physiological_signals(self):
        """
        Extract physiological signals from the RGB and thermal data using MMRPhys.
        """
        logging.info("Extracting physiological signals with MMRPhys...")
        
        # Check if we have RGB frames to process
        if not self.rgb_frames:
            logging.warning("No RGB frames to process")
            return
        
        # Create a video from the RGB frames
        rgb_video_path = os.path.join(self.rgb_dir, "rgb_video.avi")
        
        # Get dimensions from the first frame
        height, width = self.rgb_frames[0].shape[:2]
        
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(rgb_video_path, fourcc, 30.0, (width, height))
        
        # Write frames to the video file
        for frame in self.rgb_frames:
            out.write(frame)
        
        out.release()
        
        # Process the RGB video with MMRPhys
        mmrphys_results = self.mmrphys.process_video(
            rgb_video_path,
            output_dir=self.mmrphys_dir
        )
        
        # Save the results
        if mmrphys_results:
            with open(os.path.join(self.mmrphys_dir, "mmrphys_results.txt"), 'w') as f:
                for key, value in mmrphys_results.items():
                    f.write(f"{key}: {value}\n")
            
            # Save the pulse signal if available
            if 'pulse_signal' in mmrphys_results:
                pulse_signal = mmrphys_results['pulse_signal']
                pd.DataFrame({
                    "frame": range(len(pulse_signal)),
                    "pulse_signal": pulse_signal
                }).to_csv(os.path.join(self.mmrphys_dir, "pulse_signal.csv"), index=False)
        
        logging.info("Physiological signal extraction completed")
        
    def _analyze_results(self):
        """
        Analyze and visualize the results.
        """
        logging.info("Analyzing and visualizing results...")
        
        # Create visualizations of the data
        self._create_visualizations()
        
        # Compare contact-based and non-contact measurements
        self._compare_measurements()
        
        # Generate a summary report
        self._generate_summary_report()
        
        logging.info("Analysis and visualization completed")
        
    def _create_visualizations(self):
        """
        Create visualizations of the data.
        """
        # Create a directory for visualizations
        viz_dir = os.path.join(self.analysis_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize RGB frames
        if self.rgb_frames:
            # Save a montage of selected RGB frames
            montage_frames = self.rgb_frames[::max(1, len(self.rgb_frames) // 9)][:9]
            montage = self._create_montage(montage_frames, 3, 3)
            cv2.imwrite(os.path.join(viz_dir, "rgb_montage.jpg"), montage)
        
        # Visualize thermal frames
        if self.thermal_frames:
            # Save a montage of selected thermal frames
            montage_frames = self.thermal_frames[::max(1, len(self.thermal_frames) // 9)][:9]
            montage = self._create_montage(montage_frames, 3, 3)
            cv2.imwrite(os.path.join(viz_dir, "thermal_montage.jpg"), montage)
        
        # Visualize segmentation masks
        if self.segmentation_masks:
            # Convert masks to RGB for visualization
            rgb_masks = [cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) for mask in self.segmentation_masks]
            
            # Save a montage of selected masks
            montage_frames = rgb_masks[::max(1, len(rgb_masks) // 9)][:9]
            montage = self._create_montage(montage_frames, 3, 3)
            cv2.imwrite(os.path.join(viz_dir, "mask_montage.jpg"), montage)
        
        # Visualize physiological signals
        self._visualize_physiological_signals(viz_dir)
        
    def _create_montage(self, frames, rows, cols):
        """
        Create a montage of frames.
        
        Args:
            frames: List of frames.
            rows: Number of rows in the montage.
            cols: Number of columns in the montage.
            
        Returns:
            The montage image.
        """
        # Get dimensions from the first frame
        h, w = frames[0].shape[:2]
        
        # Create an empty montage
        montage = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        # Fill the montage with frames
        for i, frame in enumerate(frames):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            # Convert grayscale to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Copy the frame to the montage
            montage[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame
        
        return montage
        
    def _visualize_physiological_signals(self, viz_dir):
        """
        Visualize physiological signals.
        
        Args:
            viz_dir: Directory to save visualizations.
        """
        # Check if we have physiological data
        if not self.phys_data:
            logging.warning("No physiological data to visualize")
            return
        
        # Extract GSR data if available
        gsr_values = []
        gsr_times = []
        
        for i, data in enumerate(self.phys_data):
            if 'gsr' in data:
                gsr_values.append(data['gsr'])
                gsr_times.append(self.phys_timestamps[i])
        
        if gsr_values:
            # Convert timestamps to relative time (seconds from start)
            start_time = min(gsr_times)
            gsr_times = [(t - start_time) / 1e9 for t in gsr_times]
            
            # Plot GSR data
            plt.figure(figsize=(10, 6))
            plt.plot(gsr_times, gsr_values)
            plt.xlabel('Time (seconds)')
            plt.ylabel('GSR (µS)')
            plt.title('Galvanic Skin Response')
            plt.grid(True)
            plt.savefig(os.path.join(viz_dir, "gsr_plot.png"))
            plt.close()
        
        # Check if we have MMRPhys results
        mmrphys_results_path = os.path.join(self.mmrphys_dir, "pulse_signal.csv")
        if os.path.exists(mmrphys_results_path):
            # Load the pulse signal
            pulse_df = pd.read_csv(mmrphys_results_path)
            
            if 'pulse_signal' in pulse_df.columns:
                # Plot pulse signal
                plt.figure(figsize=(10, 6))
                plt.plot(pulse_df['frame'], pulse_df['pulse_signal'])
                plt.xlabel('Frame')
                plt.ylabel('Pulse Signal')
                plt.title('Extracted Pulse Signal from MMRPhys')
                plt.grid(True)
                plt.savefig(os.path.join(viz_dir, "pulse_signal_plot.png"))
                plt.close()
        
    def _compare_measurements(self):
        """
        Compare contact-based and non-contact measurements.
        """
        # This would typically involve comparing the physiological signals extracted
        # from the video data with the contact-based measurements
        # For now, we'll just create a placeholder for this functionality
        
        comparison_path = os.path.join(self.analysis_dir, "measurement_comparison.txt")
        
        with open(comparison_path, 'w') as f:
            f.write("Comparison of Contact-Based and Non-Contact Measurements\n")
            f.write("=====================================================\n\n")
            
            # Check if we have MMRPhys results
            mmrphys_results_path = os.path.join(self.mmrphys_dir, "mmrphys_results.txt")
            if os.path.exists(mmrphys_results_path):
                f.write("MMRPhys Results:\n")
                with open(mmrphys_results_path, 'r') as mmr_file:
                    f.write(mmr_file.read())
                f.write("\n")
            
            # Check if we have contact-based heart rate data
            heart_rate_values = []
            
            for data in self.phys_data:
                if 'heart_rate' in data:
                    heart_rate_values.append(data['heart_rate'])
            
            if heart_rate_values:
                mean_hr = np.mean(heart_rate_values)
                f.write(f"Contact-Based Mean Heart Rate: {mean_hr:.1f} BPM\n")
            
            f.write("\nNote: A more detailed comparison would require additional analysis.\n")
        
    def _generate_summary_report(self):
        """
        Generate a summary report of the analysis.
        """
        report_path = os.path.join(self.analysis_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("Integrated Capture and Analysis Summary Report\n")
            f.write("===========================================\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Participant ID: {self.args.participant_id}\n")
            f.write(f"Duration: {self.args.duration} seconds\n\n")
            
            f.write("Data Collection:\n")
            f.write(f"  RGB Frames: {len(self.rgb_frames)}\n")
            f.write(f"  Thermal Frames: {len(self.thermal_frames)}\n")
            f.write(f"  Physiological Data Points: {len(self.phys_data)}\n")
            f.write(f"  Segmentation Masks: {len(self.segmentation_masks)}\n\n")
            
            # Add information about ROIs if available
            roi_data_path = os.path.join(self.segmentation_dir, "rois", "roi_data.csv")
            if os.path.exists(roi_data_path):
                roi_df = pd.read_csv(roi_data_path)
                f.write("Region of Interest Analysis:\n")
                f.write(f"  Total ROIs Detected: {len(roi_df)}\n")
                f.write(f"  Mean ROI Area: {roi_df['area'].mean():.1f} pixels\n")
                f.write(f"  Mean ROI Temperature: {roi_df['mean_temperature'].mean():.1f}\n\n")
            
            # Add information about MMRPhys results if available
            mmrphys_results_path = os.path.join(self.mmrphys_dir, "mmrphys_results.txt")
            if os.path.exists(mmrphys_results_path):
                f.write("MMRPhys Analysis:\n")
                with open(mmrphys_results_path, 'r') as mmr_file:
                    for line in mmr_file:
                        f.write(f"  {line}")
                f.write("\n")
            
            f.write("Files and Directories:\n")
            f.write(f"  Output Directory: {self.output_dir}\n")
            f.write(f"  RGB Data: {self.rgb_dir}\n")
            f.write(f"  Thermal Data: {self.thermal_dir}\n")
            f.write(f"  Physiological Data: {self.phys_dir}\n")
            f.write(f"  Segmentation Data: {self.segmentation_dir}\n")
            f.write(f"  MMRPhys Results: {self.mmrphys_dir}\n")
            f.write(f"  Analysis Results: {self.analysis_dir}\n")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Integrated Capture and Analysis")
    
    parser.add_argument("--simulation", action="store_true",
                        help="Run in simulation mode without real hardware")
    parser.add_argument("--participant-id", type=str, default="test_subject",
                        help="Participant ID for data organization")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of data capture in seconds")
    
    # Capture options
    parser.add_argument("--use-factorize", action="store_true",
                        help="Use FactorizePhys instead of RGBTPhys_CPP for data capture")
    parser.add_argument("--factorize-config", type=str, default=None,
                        help="Path to FactorizePhys configuration file")
    parser.add_argument("--rgbtphys-config", type=str, default=None,
                        help="Path to RGBTPhys_CPP configuration file")
    
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
    Main function to run the integrated capture and analysis process.
    """
    # Parse command-line arguments
    args = parse_args()
    
    logging.info("=======================================")
    logging.info("= Integrated Capture and Analysis     =")
    logging.info("=======================================")
    
    # Create and run the integrated capture and analysis process
    process = IntegratedCaptureAndAnalysis(args)
    process.run()
    
    logging.info("Process completed successfully")


if __name__ == "__main__":
    main()