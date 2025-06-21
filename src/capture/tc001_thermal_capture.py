import logging
import os
import sys
import time
import numpy as np
import cv2
from PyQt5.QtCore import pyqtSignal

from src.capture.base_capture import BaseCaptureThread

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s",
)

class TC001ThermalCaptureThread(BaseCaptureThread):
    """
    A dedicated thread to capture thermal video from a TOPDON TC001 camera
    and perform segmentation using the SAMCL model.

    This class provides an interface to the TC001_SAMCL repository for thermal
    imaging and segmentation within the Python framework of the GSR-RGBT project.

    Attributes:
        frame_captured (pyqtSignal): A signal that emits the captured thermal frame
                                     as a NumPy array (np.ndarray) and timestamp.
        segmentation_captured (pyqtSignal): A signal that emits the segmentation mask
                                           as a NumPy array (np.ndarray) and timestamp.
        roi_signal_captured (pyqtSignal): A signal that emits the signal extracted from
                                         the region of interest and timestamp.
    """

    frame_captured = pyqtSignal(np.ndarray, float)
    segmentation_captured = pyqtSignal(np.ndarray, float)
    roi_signal_captured = pyqtSignal(object, float)

    def __init__(self, device_id=0, config_file=None, simulation_mode=False, parent=None):
        """
        Initializes the TC001 thermal capture thread.

        Args:
            device_id (int): The device ID of the TOPDON TC001 camera.
            config_file (str): Path to the configuration file for the segmentation model.
            simulation_mode (bool): If True, the thread will generate simulated data.
            parent (QObject, optional): The parent object in the Qt hierarchy.
        """
        super().__init__(device_name="TC001Thermal", parent=parent)
        self.device_id = device_id
        self.config_file = config_file
        self.simulation_mode = simulation_mode
        self.camera = None
        self.segmenter = None
        self.roi_extractor = None

    def _initialize_segmenter(self):
        """
        Initialize the thermal segmentation model.
        """
        # Add TC001_SAMCL to the Python path
        tc001_path = os.path.join(os.getcwd(), "third_party", "TC001_SAMCL")
        if tc001_path not in sys.path:
            sys.path.append(tc001_path)

        try:
            # Import the necessary modules from TC001_SAMCL
            from inference import ThermSeg
            from utils.configer import Configer

            # Use the provided config file or the default one
            if self.config_file:
                config_path = self.config_file
            else:
                config_path = os.path.join(tc001_path, "configs", "default_config.yaml")

            # Initialize the segmentation model
            configer = Configer(config_path=config_path)
            self.segmenter = ThermSeg(configer)
            logging.info(f"Initialized TC001 thermal segmentation model with config: {config_path}")

        except ImportError as e:
            logging.error(f"Failed to import TC001_SAMCL modules: {e}")
            logging.error("Make sure the TC001_SAMCL repository is properly initialized.")
            raise

    def _initialize_camera(self):
        """
        Initialize the TOPDON TC001 thermal camera.
        """
        try:
            # Open the camera
            self.camera = cv2.VideoCapture(self.device_id)
            
            # Check if the camera was opened successfully
            if not self.camera.isOpened():
                logging.error(f"Failed to open TOPDON TC001 camera with device ID {self.device_id}")
                return False
            
            # Set camera properties if needed
            # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            logging.info(f"Successfully initialized TOPDON TC001 camera with device ID {self.device_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing TOPDON TC001 camera: {e}")
            return False

    def _run_simulation(self):
        """
        Generates and emits simulated thermal data and segmentation results.
        This is useful for UI development and testing without hardware.
        """
        logging.info("Running in TC001 thermal simulation mode.")
        
        # Simulate at 30 fps
        interval = 1.0 / 30
        
        # Frame size
        frame_size = (480, 640)
        
        # Initialize the segmenter if not in pure simulation mode
        if not self.simulation_mode:
            self._initialize_segmenter()
        
        while self.is_running:
            # Create a simulated thermal image (grayscale with some patterns)
            frame = np.zeros(frame_size, dtype=np.uint8)
            t = time.time()
            
            # Add a gradient from top to bottom
            for i in range(frame_size[0]):
                value = int(255 * i / frame_size[0])
                frame[i, :] = value
            
            # Add some "hot spots" that move over time
            for i in range(3):
                x = int((0.5 + 0.3 * np.sin(t + i)) * frame_size[1])
                y = int((0.5 + 0.3 * np.cos(t + i)) * frame_size[0])
                if 0 <= x < frame_size[1] and 0 <= y < frame_size[0]:
                    # Create a hot spot with a radius of 30 pixels
                    for dx in range(-30, 31):
                        for dy in range(-30, 31):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < frame_size[1] and 0 <= ny < frame_size[0]:
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < 30:
                                    intensity = int(255 * (1 - dist/30))
                                    frame[ny, nx] = min(255, frame[ny, nx] + intensity)
            
            # Convert to 3-channel for display and processing
            color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            
            # Create a simulated segmentation mask
            # In a real scenario, this would come from the segmentation model
            mask = np.zeros(frame_size, dtype=np.uint8)
            
            # Add a circular region of interest
            center_x = int(frame_size[1] / 2)
            center_y = int(frame_size[0] / 2)
            radius = 100
            
            for i in range(frame_size[0]):
                for j in range(frame_size[1]):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    if dist < radius:
                        mask[i, j] = 255
            
            # Extract signal from the region of interest
            # In a real scenario, this would be a temporal signal
            roi_signal = {
                "mean_temperature": np.mean(frame[mask > 0]),
                "max_temperature": np.max(frame[mask > 0]),
                "min_temperature": np.min(frame[mask > 0]),
                "std_temperature": np.std(frame[mask > 0].astype(float))
            }
            
            # Get current timestamp
            current_capture_time = self.get_current_timestamp()
            
            # Emit the frame, mask, and signal
            logging.info(f"TC001ThermalCaptureThread (simulation): Emitting data with timestamp {current_capture_time}")
            self.frame_captured.emit(color_frame, current_capture_time)
            self.segmentation_captured.emit(mask, current_capture_time)
            self.roi_signal_captured.emit(roi_signal, current_capture_time)
            
            # Sleep to maintain frame rate
            self._sleep(interval)

    def _run_real_capture(self):
        """
        Captures frames from the TOPDON TC001 thermal camera and performs segmentation.
        """
        logging.info("Starting TC001 thermal camera capture and segmentation.")
        
        # Initialize the camera
        if not self._initialize_camera():
            logging.error("Failed to initialize TC001 thermal camera. Aborting capture.")
            return
        
        # Initialize the segmentation model
        self._initialize_segmenter()
        
        # Previous segmentation mask for tracking
        prev_mask = None
        
        while self.is_running:
            try:
                # Capture a frame from the camera
                ret, frame = self.camera.read()
                
                if not ret:
                    logging.warning("Failed to capture frame from TC001 thermal camera.")
                    self._sleep(0.1)  # Prevent busy-waiting on error
                    continue
                
                # Get current timestamp
                current_capture_time = self.get_current_timestamp()
                
                # Emit the captured frame
                self.frame_captured.emit(frame, current_capture_time)
                
                # Perform segmentation
                if self.segmenter:
                    # Process the frame with the segmentation model
                    segmentation_result = self.segmenter.process_frame(frame, prev_mask)
                    
                    if segmentation_result:
                        mask = segmentation_result.get('mask')
                        if mask is not None:
                            # Emit the segmentation mask
                            self.segmentation_captured.emit(mask, current_capture_time)
                            
                            # Extract signal from the region of interest
                            roi_signal = self._extract_roi_signal(frame, mask)
                            self.roi_signal_captured.emit(roi_signal, current_capture_time)
                            
                            # Update previous mask for tracking
                            prev_mask = mask
                
            except Exception as e:
                logging.error(f"Error during TC001 thermal capture: {e}")
                self._sleep(0.1)  # Prevent busy-waiting on error
    
    def _extract_roi_signal(self, frame, mask):
        """
        Extract a signal from the region of interest defined by the mask.
        
        Args:
            frame (numpy.ndarray): The thermal frame.
            mask (numpy.ndarray): The segmentation mask.
            
        Returns:
            dict: A dictionary containing the extracted signals.
        """
        # Convert the frame to grayscale if it's not already
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # Apply the mask to extract the region of interest
        roi = gray_frame.copy()
        roi[mask == 0] = 0
        
        # Calculate statistics from the ROI
        roi_pixels = roi[mask > 0]
        if len(roi_pixels) > 0:
            mean_value = np.mean(roi_pixels)
            max_value = np.max(roi_pixels)
            min_value = np.min(roi_pixels)
            std_value = np.std(roi_pixels.astype(float))
        else:
            mean_value = 0
            max_value = 0
            min_value = 0
            std_value = 0
        
        # Return the extracted signals
        return {
            "mean_temperature": mean_value,
            "max_temperature": max_value,
            "min_temperature": min_value,
            "std_temperature": std_value
        }

    def _cleanup(self):
        """
        Clean up resources when the thread is stopping.
        """
        logging.info("Cleaning up TC001 thermal capture resources.")
        
        # Release the camera if it's open
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        
        # Clean up the segmenter if needed
        if self.segmenter:
            # Some segmentation models might need explicit cleanup
            self.segmenter = None

    def _sleep(self, seconds):
        """
        Sleep for the specified number of seconds.
        
        Args:
            seconds (float): The number of seconds to sleep.
        """
        time.sleep(seconds)