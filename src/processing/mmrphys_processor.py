import logging
import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)

class MMRPhysProcessor:
    """
    A processor class for extracting physiological signals from RGB and thermal videos
    using the MMRPhys deep learning framework.
    
    This class provides an interface to the MMRPhys models for remote physiological sensing,
    allowing for extraction of heart rate, respiration, and other physiological signals
    from video data without requiring physical contact with the subject.
    
    Attributes:
        model_type (str): The type of MMRPhys model to use (e.g., 'MMRPhysLEF', 'MMRPhysMEF', 'MMRPhysSEF').
        device (torch.device): The device to run inference on (CPU or GPU).
        model (torch.nn.Module): The loaded MMRPhys model.
    """
    
    def __init__(self, model_type='MMRPhysLEF', model_path=None, use_gpu=True):
        """
        Initialize the MMRPhys processor.
        
        Args:
            model_type (str): The type of MMRPhys model to use.
                Options: 'MMRPhysLEF', 'MMRPhysMEF', 'MMRPhysSEF'
            model_path (str): Path to a pre-trained model file. If None, uses the default model.
            use_gpu (bool): Whether to use GPU for inference if available.
        """
        self.model_type = model_type
        self.device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Add MMRPhys to the Python path
        mmrphys_path = os.path.join(os.getcwd(), "third_party", "MMRPhys")
        if mmrphys_path not in sys.path:
            sys.path.append(mmrphys_path)
        
        # Import MMRPhys modules
        try:
            from neural_methods.model.MMRPhys.MMRPhysLEF import MMRPhysLEF
            from neural_methods.model.MMRPhys.MMRPhysMEF import MMRPhysMEF
            from neural_methods.model.MMRPhys.MMRPhysSEF import MMRPhysSEF
            
            # Initialize the model based on the specified type
            if model_type == 'MMRPhysLEF':
                self.model = MMRPhysLEF()
            elif model_type == 'MMRPhysMEF':
                self.model = MMRPhysMEF()
            elif model_type == 'MMRPhysSEF':
                self.model = MMRPhysSEF()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load pre-trained weights
            if model_path:
                self._load_model(model_path)
            else:
                # Use default pre-trained model
                default_model_path = self._get_default_model_path()
                self._load_model(default_model_path)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"MMRPhys model {model_type} loaded successfully on {self.device}")
            
        except ImportError as e:
            logging.error(f"Failed to import MMRPhys modules: {e}")
            logging.error("Make sure the MMRPhys repository is properly initialized and installed.")
            raise
    
    def _get_default_model_path(self):
        """
        Get the path to the default pre-trained model for the selected model type.
        
        Returns:
            str: Path to the default pre-trained model.
        """
        # Default models are stored in the final_model_release directory
        model_dir = os.path.join(os.getcwd(), "third_party", "MMRPhys", "final_model_release")
        
        # Model filename depends on the model type
        if self.model_type == 'MMRPhysLEF':
            model_file = "mmrphys_lef_rgb_ibvp.pth"
        elif self.model_type == 'MMRPhysMEF':
            model_file = "mmrphys_mef_rgb_ibvp.pth"
        elif self.model_type == 'MMRPhysSEF':
            model_file = "mmrphys_sef_rgb_ibvp.pth"
        else:
            raise ValueError(f"No default model available for {self.model_type}")
        
        model_path = os.path.join(model_dir, model_file)
        
        if not os.path.exists(model_path):
            logging.warning(f"Default model not found at {model_path}")
            logging.warning("Please download the pre-trained models or specify a custom model path.")
            raise FileNotFoundError(f"Default model not found at {model_path}")
        
        return model_path
    
    def _load_model(self, model_path):
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_path (str): Path to the pre-trained model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if the checkpoint contains the state_dict directly or under a key
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logging.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """
        Preprocess a video frame for input to the model.
        
        Args:
            frame (numpy.ndarray): The input frame (RGB or thermal).
            
        Returns:
            torch.Tensor: The preprocessed frame tensor.
        """
        # Resize to the expected input size (224x224 is common for many models)
        resized_frame = cv2.resize(frame, (224, 224))
        
        # Convert to float and normalize to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor and add batch dimension
        # Transpose from (H, W, C) to (C, H, W) format
        frame_tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor.to(self.device)
    
    def process_frame_sequence(self, frames, frame_type='rgb'):
        """
        Process a sequence of frames to extract physiological signals.
        
        Args:
            frames (list): A list of numpy.ndarray frames.
            frame_type (str): The type of frames ('rgb' or 'thermal').
            
        Returns:
            dict: A dictionary containing the extracted physiological signals.
        """
        if not frames:
            logging.warning("Empty frame sequence provided")
            return None
        
        # MMRPhys typically requires a sequence of frames for temporal processing
        # We'll preprocess each frame and stack them into a batch
        preprocessed_frames = [self.preprocess_frame(frame) for frame in frames]
        frame_batch = torch.cat(preprocessed_frames, dim=0)
        
        # Add an extra dimension for the batch if needed by the model
        if len(frame_batch.shape) == 3:
            frame_batch = frame_batch.unsqueeze(0)
        
        try:
            with torch.no_grad():
                # Forward pass through the model
                outputs = self.model(frame_batch)
                
                # Process the outputs based on the model type
                # This will depend on the specific output format of the MMRPhys models
                if isinstance(outputs, tuple):
                    # Some models return multiple outputs
                    pulse_signal = outputs[0].cpu().numpy()
                    quality_index = outputs[1].cpu().numpy() if len(outputs) > 1 else None
                else:
                    # Single output (typically the pulse signal)
                    pulse_signal = outputs.cpu().numpy()
                    quality_index = None
                
                # Create a dictionary of results
                results = {
                    'pulse_signal': pulse_signal,
                    'quality_index': quality_index
                }
                
                return results
                
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return None
    
    def extract_heart_rate(self, pulse_signal, fps=30, window_size=300):
        """
        Extract heart rate from the pulse signal.
        
        Args:
            pulse_signal (numpy.ndarray): The extracted pulse signal.
            fps (int): Frames per second of the video.
            window_size (int): Window size for heart rate calculation.
            
        Returns:
            float: The estimated heart rate in beats per minute.
        """
        # Ensure the signal is 1D
        if len(pulse_signal.shape) > 1:
            pulse_signal = pulse_signal.flatten()
        
        # Use a sliding window approach if the signal is long enough
        if len(pulse_signal) >= window_size:
            # Use the last window_size samples
            signal_window = pulse_signal[-window_size:]
        else:
            signal_window = pulse_signal
        
        # Perform FFT to find the dominant frequency
        fft_result = np.fft.rfft(signal_window)
        fft_freq = np.fft.rfftfreq(len(signal_window), d=1.0/fps)
        
        # Find the peak in the frequency domain (excluding DC component)
        peak_idx = np.argmax(np.abs(fft_result[1:])) + 1
        peak_freq = fft_freq[peak_idx]
        
        # Convert frequency to BPM
        heart_rate = peak_freq * 60
        
        # Typical heart rates are between 40-240 BPM
        if heart_rate < 40 or heart_rate > 240:
            logging.warning(f"Estimated heart rate ({heart_rate:.1f} BPM) is outside normal range")
        
        return heart_rate
    
    def process_video(self, video_path, output_dir=None, frame_limit=None):
        """
        Process a video file to extract physiological signals.
        
        Args:
            video_path (str): Path to the video file.
            output_dir (str): Directory to save the results. If None, results are not saved.
            frame_limit (int): Maximum number of frames to process. If None, process the entire video.
            
        Returns:
            dict: A dictionary containing the extracted physiological signals and metrics.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_limit:
            frame_count = min(frame_count, frame_limit)
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"FPS: {fps}, Frame count: {frame_count}")
        
        # Process frames in batches
        batch_size = 30  # Process 1 second of video at a time
        frames = []
        results = []
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Process when we have enough frames for a batch
            if len(frames) >= batch_size:
                batch_results = self.process_frame_sequence(frames)
                if batch_results:
                    results.append(batch_results)
                frames = []  # Clear the batch
        
        # Process any remaining frames
        if frames:
            batch_results = self.process_frame_sequence(frames)
            if batch_results:
                results.append(batch_results)
        
        # Release the video capture
        cap.release()
        
        # Combine results from all batches
        combined_results = self._combine_batch_results(results)
        
        # Extract heart rate
        if 'pulse_signal' in combined_results and combined_results['pulse_signal'] is not None:
            heart_rate = self.extract_heart_rate(combined_results['pulse_signal'], fps)
            combined_results['heart_rate'] = heart_rate
            logging.info(f"Estimated heart rate: {heart_rate:.1f} BPM")
        
        # Save results if output directory is specified
        if output_dir:
            self._save_results(combined_results, output_dir, os.path.basename(video_path))
        
        return combined_results
    
    def _combine_batch_results(self, batch_results):
        """
        Combine results from multiple batches.
        
        Args:
            batch_results (list): List of result dictionaries from each batch.
            
        Returns:
            dict: Combined results.
        """
        if not batch_results:
            return {}
        
        combined = {}
        
        # Combine pulse signals
        pulse_signals = [batch['pulse_signal'] for batch in batch_results if 'pulse_signal' in batch and batch['pulse_signal'] is not None]
        if pulse_signals:
            combined['pulse_signal'] = np.concatenate(pulse_signals)
        
        # Combine quality indices
        quality_indices = [batch['quality_index'] for batch in batch_results if 'quality_index' in batch and batch['quality_index'] is not None]
        if quality_indices:
            combined['quality_index'] = np.concatenate(quality_indices)
        
        return combined
    
    def _save_results(self, results, output_dir, video_name):
        """
        Save the processing results to files.
        
        Args:
            results (dict): The processing results.
            output_dir (str): Directory to save the results.
            video_name (str): Name of the processed video file.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(video_name)[0]
        
        # Save pulse signal
        if 'pulse_signal' in results and results['pulse_signal'] is not None:
            pulse_path = os.path.join(output_dir, f"{base_name}_pulse_signal.npy")
            np.save(pulse_path, results['pulse_signal'])
            logging.info(f"Saved pulse signal to {pulse_path}")
        
        # Save quality index
        if 'quality_index' in results and results['quality_index'] is not None:
            quality_path = os.path.join(output_dir, f"{base_name}_quality_index.npy")
            np.save(quality_path, results['quality_index'])
            logging.info(f"Saved quality index to {quality_path}")
        
        # Save heart rate
        if 'heart_rate' in results:
            results_path = os.path.join(output_dir, f"{base_name}_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Heart Rate: {results['heart_rate']:.1f} BPM\n")
            logging.info(f"Saved results to {results_path}")