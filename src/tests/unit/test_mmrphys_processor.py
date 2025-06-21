import sys
import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import torch
import cv2

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.processing.mmrphys_processor import MMRPhysProcessor


class TestMMRPhysProcessor(unittest.TestCase):
    """
    Unit tests for the MMRPhysProcessor class.
    
    These tests verify the functionality of the MMRPhysProcessor class,
    which provides an interface to the MMRPhys deep learning framework
    for extracting physiological signals from RGB and thermal videos.
    """

    def setUp(self):
        """Set up test data and mocks."""
        # Create a temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Mock the MMRPhys model classes and torch.load
        self.patcher1 = patch('torch.load')
        self.mock_torch_load = self.patcher1.start()
        self.mock_torch_load.return_value = {'state_dict': {}}
        
        # Mock the MMRPhys model classes
        self.patcher2 = patch('src.processing.mmrphys_processor.MMRPhysLEF')
        self.mock_mmrphys_lef = self.patcher2.start()
        self.mock_mmrphys_lef_instance = MagicMock()
        self.mock_mmrphys_lef.return_value = self.mock_mmrphys_lef_instance
        
        self.patcher3 = patch('src.processing.mmrphys_processor.MMRPhysMEF')
        self.mock_mmrphys_mef = self.patcher3.start()
        self.mock_mmrphys_mef_instance = MagicMock()
        self.mock_mmrphys_mef.return_value = self.mock_mmrphys_mef_instance
        
        self.patcher4 = patch('src.processing.mmrphys_processor.MMRPhysSEF')
        self.mock_mmrphys_sef = self.patcher4.start()
        self.mock_mmrphys_sef_instance = MagicMock()
        self.mock_mmrphys_sef.return_value = self.mock_mmrphys_sef_instance
        
        # Mock os.path.exists to return True for model paths
        self.patcher5 = patch('os.path.exists')
        self.mock_path_exists = self.patcher5.start()
        self.mock_path_exists.return_value = True
        
        # Create test data
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_pulse_signal = np.sin(np.linspace(0, 10*np.pi, 300))  # Simulated pulse signal

    def tearDown(self):
        """Clean up temporary files and stop patches."""
        shutil.rmtree(self.temp_dir)
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()

    @patch('sys.path.append')
    def test_init_with_different_model_types(self, mock_path_append):
        """Test initialization with different model types."""
        # Test with MMRPhysLEF
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        self.assertEqual(processor.model_type, 'MMRPhysLEF')
        self.assertEqual(processor.device.type, 'cpu')
        self.mock_mmrphys_lef.assert_called_once()
        
        # Reset mocks
        self.mock_mmrphys_lef.reset_mock()
        
        # Test with MMRPhysMEF
        processor = MMRPhysProcessor(model_type='MMRPhysMEF', use_gpu=False)
        self.assertEqual(processor.model_type, 'MMRPhysMEF')
        self.assertEqual(processor.device.type, 'cpu')
        self.mock_mmrphys_mef.assert_called_once()
        
        # Reset mocks
        self.mock_mmrphys_mef.reset_mock()
        
        # Test with MMRPhysSEF
        processor = MMRPhysProcessor(model_type='MMRPhysSEF', use_gpu=False)
        self.assertEqual(processor.model_type, 'MMRPhysSEF')
        self.assertEqual(processor.device.type, 'cpu')
        self.mock_mmrphys_sef.assert_called_once()

    @patch('sys.path.append')
    def test_init_with_custom_model_path(self, mock_path_append):
        """Test initialization with a custom model path."""
        custom_path = str(self.temp_dir / "custom_model.pth")
        
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', model_path=custom_path, use_gpu=False)
        
        self.mock_torch_load.assert_called_with(custom_path, map_location=processor.device)
        self.mock_mmrphys_lef_instance.load_state_dict.assert_called_once()

    @patch('sys.path.append')
    def test_preprocess_frame(self, mock_path_append):
        """Test frame preprocessing."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Process a test frame
        frame_tensor = processor.preprocess_frame(self.test_frame)
        
        # Check the output shape and type
        self.assertIsInstance(frame_tensor, torch.Tensor)
        self.assertEqual(frame_tensor.shape, (1, 3, 224, 224))  # Batch, Channels, Height, Width
        self.assertEqual(frame_tensor.device.type, 'cpu')
        
        # Check that values are normalized to [0, 1]
        self.assertTrue(torch.all(frame_tensor >= 0))
        self.assertTrue(torch.all(frame_tensor <= 1))

    @patch('sys.path.append')
    def test_extract_heart_rate(self, mock_path_append):
        """Test heart rate extraction from pulse signal."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Create a synthetic pulse signal with a known frequency
        fps = 30
        duration_seconds = 10
        bpm = 60  # 1 Hz = 60 BPM
        t = np.linspace(0, duration_seconds, fps * duration_seconds)
        pulse_signal = np.sin(2 * np.pi * (bpm / 60) * t)
        
        # Extract heart rate
        heart_rate = processor.extract_heart_rate(pulse_signal, fps=fps)
        
        # Check that the extracted heart rate is close to the expected value
        self.assertAlmostEqual(heart_rate, bpm, delta=5)  # Allow some error due to FFT resolution

    @patch('sys.path.append')
    def test_combine_batch_results(self, mock_path_append):
        """Test combining results from multiple batches."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Create test batch results
        batch1 = {
            'pulse_signal': np.array([1, 2, 3]),
            'quality_index': np.array([0.9, 0.8, 0.7])
        }
        
        batch2 = {
            'pulse_signal': np.array([4, 5, 6]),
            'quality_index': np.array([0.6, 0.5, 0.4])
        }
        
        # Combine batches
        combined = processor._combine_batch_results([batch1, batch2])
        
        # Check the combined results
        np.testing.assert_array_equal(combined['pulse_signal'], np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(combined['quality_index'], np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))

    @patch('sys.path.append')
    def test_empty_batch_results(self, mock_path_append):
        """Test combining empty batch results."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Test with empty list
        combined = processor._combine_batch_results([])
        self.assertEqual(combined, {})
        
        # Test with None values
        batch1 = {
            'pulse_signal': None,
            'quality_index': None
        }
        
        combined = processor._combine_batch_results([batch1])
        self.assertEqual(combined, {})

    @patch('sys.path.append')
    @patch('os.path.exists')
    def test_model_not_found_error(self, mock_exists, mock_path_append):
        """Test error handling when model file is not found."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            MMRPhysProcessor(model_type='MMRPhysLEF', model_path="nonexistent_model.pth", use_gpu=False)

    @patch('sys.path.append')
    def test_process_frame_sequence(self, mock_path_append):
        """Test processing a sequence of frames."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Mock the model's forward pass
        mock_output = torch.tensor([0.1, 0.2, 0.3])
        processor.model.forward = MagicMock(return_value=mock_output)
        
        # Create a sequence of test frames
        frames = [self.test_frame] * 5
        
        # Process the frames
        results = processor.process_frame_sequence(frames)
        
        # Check that the model was called and results were returned
        processor.model.assert_called_once()
        self.assertIn('pulse_signal', results)
        np.testing.assert_array_equal(results['pulse_signal'], mock_output.numpy())

    @patch('sys.path.append')
    def test_empty_frame_sequence(self, mock_path_append):
        """Test handling of empty frame sequences."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Process empty frame sequence
        results = processor.process_frame_sequence([])
        
        # Check that None is returned
        self.assertIsNone(results)

    @patch('sys.path.append')
    @patch('cv2.VideoCapture')
    def test_process_video(self, mock_video_capture, mock_path_append):
        """Test processing a video file."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 90 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
        
        # Mock read method to return 3 frames then stop
        mock_cap.read.side_effect = [
            (True, self.test_frame),
            (True, self.test_frame),
            (True, self.test_frame),
            (False, None)
        ]
        
        # Mock process_frame_sequence
        processor.process_frame_sequence = MagicMock(return_value={
            'pulse_signal': self.test_pulse_signal,
            'quality_index': np.ones_like(self.test_pulse_signal)
        })
        
        # Mock extract_heart_rate
        processor.extract_heart_rate = MagicMock(return_value=72.0)
        
        # Process the video
        results = processor.process_video("test_video.mp4", output_dir=str(self.temp_dir))
        
        # Check that the methods were called correctly
        mock_cap.read.assert_called()
        processor.process_frame_sequence.assert_called()
        processor.extract_heart_rate.assert_called_once()
        
        # Check the results
        self.assertIn('heart_rate', results)
        self.assertEqual(results['heart_rate'], 72.0)

    @patch('sys.path.append')
    @patch('os.makedirs')
    def test_save_results(self, mock_makedirs, mock_path_append):
        """Test saving results to files."""
        processor = MMRPhysProcessor(model_type='MMRPhysLEF', use_gpu=False)
        
        # Create test results
        results = {
            'pulse_signal': self.test_pulse_signal,
            'quality_index': np.ones_like(self.test_pulse_signal),
            'heart_rate': 72.0
        }
        
        # Mock open and np.save
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('numpy.save') as mock_np_save:
            
            # Save the results
            processor._save_results(results, str(self.temp_dir), "test_video.mp4")
            
            # Check that the files were created
            mock_makedirs.assert_called_once_with(str(self.temp_dir), exist_ok=True)
            mock_np_save.assert_called()
            mock_file.assert_called()


if __name__ == "__main__":
    unittest.main()