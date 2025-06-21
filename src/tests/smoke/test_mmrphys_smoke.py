import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.processing.mmrphys_processor import MMRPhysProcessor
from src.processing.feature_engineering import create_dataset_from_session


class TestMMRPhysSmoke(unittest.TestCase):
    """
    Smoke tests for the MMRPhysProcessor.
    These tests verify that the MMRPhysProcessor can be integrated with the existing pipeline.
    """

    @patch('src.processing.mmrphys_processor.MMRPhysLEF')
    @patch('src.processing.mmrphys_processor.MMRPhysMEF')
    @patch('src.processing.mmrphys_processor.MMRPhysSEF')
    @patch('torch.load')
    @patch('os.path.exists')
    @patch('sys.path.append')
    def test_mmrphys_processor_initialization(self, mock_path_append, mock_exists, 
                                             mock_torch_load, mock_sef, mock_mef, mock_lef):
        """Smoke test for MMRPhysProcessor initialization."""
        # Setup mocks
        mock_exists.return_value = True
        mock_torch_load.return_value = {'state_dict': {}}
        
        mock_lef_instance = MagicMock()
        mock_lef.return_value = mock_lef_instance
        
        # Initialize the processor
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Check that it was initialized correctly
        self.assertIsNotNone(processor)
        self.assertEqual(processor.model_type, 'MMRPhysLEF')
        self.assertEqual(processor.device.type, 'cpu')

    @patch('src.processing.mmrphys_processor.MMRPhysLEF')
    @patch('torch.load')
    @patch('os.path.exists')
    @patch('sys.path.append')
    def test_mmrphys_basic_functionality(self, mock_path_append, mock_exists, 
                                        mock_torch_load, mock_lef):
        """Smoke test for basic MMRPhysProcessor functionality."""
        # Setup mocks
        mock_exists.return_value = True
        mock_torch_load.return_value = {'state_dict': {}}
        
        mock_lef_instance = MagicMock()
        mock_lef.return_value = mock_lef_instance
        
        # Mock the model's forward pass
        mock_output = torch.tensor([0.1, 0.2, 0.3])
        mock_lef_instance.forward = MagicMock(return_value=mock_output)
        
        # Initialize the processor
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process a single frame
        frame_tensor = processor.preprocess_frame(test_frame)
        
        # Check that the frame was preprocessed correctly
        self.assertIsInstance(frame_tensor, torch.Tensor)
        self.assertEqual(frame_tensor.shape, (1, 3, 224, 224))
        
        # Process a sequence of frames
        frames = [test_frame] * 3
        results = processor.process_frame_sequence(frames)
        
        # Check that results were returned
        self.assertIsNotNone(results)
        self.assertIn('pulse_signal', results)

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    @patch('src.processing.feature_engineering.detect_palm_roi')
    @patch('src.processing.feature_engineering.extract_roi_signal')
    @patch('src.processing.mmrphys_processor.MMRPhysLEF')
    @patch('torch.load')
    @patch('os.path.exists')
    @patch('sys.path.append')
    def test_mmrphys_integration_with_pipeline(self, mock_path_append, mock_exists, 
                                              mock_torch_load, mock_lef, mock_extract_roi, 
                                              mock_detect_roi, mock_process_gsr, mock_loader):
        """Smoke test for integrating MMRPhysProcessor with the existing pipeline."""
        # Setup mocks for MMRPhysProcessor
        mock_exists.return_value = True
        mock_torch_load.return_value = {'state_dict': {}}
        
        mock_lef_instance = MagicMock()
        mock_lef.return_value = mock_lef_instance
        
        # Mock the model's forward pass
        mock_output = torch.tensor([0.1, 0.2, 0.3])
        mock_lef_instance.forward = MagicMock(return_value=mock_output)
        
        # Setup mocks for feature_engineering
        session_path = Path("dummy/path")
        gsr_sampling_rate = 32
        video_fps = 30
        
        # Mock GSR data
        gsr_timestamps = np.arange(0, 10, 1/gsr_sampling_rate)
        mock_gsr_df = MagicMock()
        
        # Setup loader mock
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.get_gsr_data.return_value = mock_gsr_df
        
        # Setup video frame generator
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mock_loader_instance.get_rgb_video_generator.return_value = [
            (True, frame) for _ in range(10)
        ]
        
        # Setup other mocks
        mock_process_gsr.return_value = MagicMock()
        mock_detect_roi.return_value = [(10, 10), (30, 30)]
        mock_extract_roi.return_value = [100, 120, 140]  # RGB values
        
        # Initialize the processor
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Create a dataset using the existing pipeline
        dataset = create_dataset_from_session(session_path, gsr_sampling_rate, video_fps)
        
        # Process a frame from the dataset using MMRPhysProcessor
        # This is a simplified integration test - in a real scenario, we would extract frames from the dataset
        # and pass them to the processor
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_tensor = processor.preprocess_frame(test_frame)
        
        # Check that both components work together
        self.assertIsNotNone(dataset)
        self.assertIsNotNone(frame_tensor)


if __name__ == "__main__":
    unittest.main()