# src/tests/test_train_model.py

# --- Add project root to path for absolute imports ---
import sys
import unittest
from pathlib import Path
import tempfile
import shutil
import json
import yaml
import os
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.scripts.train_model import (
    load_all_session_data,
    parse_arguments,
    build_model_from_config,
    create_training_metadata,
    save_training_metadata,
    setup_callbacks
)
from src.ml_models.model_config import ModelConfig


class TestTrainModel(unittest.TestCase):
    """
    Test suite for the train_model.py script functions.
    """

    def setUp(self):
        """Set up common objects for all tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Create dummy session directories
        self.session1 = self.data_dir / "Subject_01_20250101_000000"
        self.session2 = self.data_dir / "Subject_02_20250101_000000"
        self.session1.mkdir(exist_ok=True)
        self.session2.mkdir(exist_ok=True)

        # Create a model config for testing
        self.model_config = ModelConfig()
        self.model_config.config = {
            "model_type": "pytorch_cnn",
            "model_params": {
                "conv_channels": [32, 64],
                "kernel_sizes": [3, 3],
                "strides": [1, 1],
                "pool_sizes": [2, 2],
                "fc_layers": [32, 1],
                "activations": ["relu", "relu", "relu", "linear"],
                "dropout_rate": 0.2
            },
            "optimizer_params": {
                "type": "adam",
                "lr": 0.001
            },
            "loss_params": {
                "type": "mse"
            },
            "train_params": {
                "batch_size": 32,
                "epochs": 10,
                "validation_split": 0.2,
                "shuffle": True
            }
        }

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    @patch('src.scripts.train_model.create_dataset_from_session')
    def test_load_all_session_data(self, mock_create_dataset):
        """Test loading data from all sessions."""
        # Setup mock
        X1 = np.random.randn(10, 5, 3)  # 10 samples, 5 timesteps, 3 features
        y1 = np.random.randn(10)
        X2 = np.random.randn(15, 5, 3)  # 15 samples, 5 timesteps, 3 features
        y2 = np.random.randn(15)

        # Mock returns different data for different sessions
        mock_create_dataset.side_effect = [
            (X1, y1),  # First session
            (X2, y2)   # Second session
        ]

        # Call the function
        X, y, subject_ids = load_all_session_data(self.data_dir, 32, 30)

        # Assertions
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(subject_ids)

        # Check that data from both sessions was combined
        self.assertEqual(len(X), len(X1) + len(X2))
        self.assertEqual(len(y), len(y1) + len(y2))
        self.assertEqual(len(subject_ids), len(X1) + len(X2))

        # Check that create_dataset_from_session was called for each session
        self.assertEqual(mock_create_dataset.call_count, 2)

    @patch('src.scripts.train_model.create_dataset_from_session')
    def test_load_all_session_data_with_failures(self, mock_create_dataset):
        """Test loading data when some sessions fail."""
        # Setup mock
        X1 = np.random.randn(10, 5, 3)
        y1 = np.random.randn(10)

        # Mock returns data for first session, None for second session
        mock_create_dataset.side_effect = [
            (X1, y1),  # First session succeeds
            None       # Second session fails
        ]

        # Call the function
        X, y, subject_ids = load_all_session_data(self.data_dir, 32, 30)

        # Assertions
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(subject_ids)

        # Check that only data from the successful session was included
        self.assertEqual(len(X), len(X1))
        self.assertEqual(len(y), len(y1))
        self.assertEqual(len(subject_ids), len(X1))

        # Check that create_dataset_from_session was called for each session
        self.assertEqual(mock_create_dataset.call_count, 2)

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_arguments(self, mock_parse_args):
        """Test argument parsing."""
        # Setup mock
        mock_args = MagicMock()
        mock_args.model_type = "cnn"
        mock_args.config_path = None
        mock_args.data_dir = "data"
        mock_args.output_dir = "output"
        mock_args.gsr_sampling_rate = 32
        mock_args.video_fps = 30
        mock_args.cross_validation = "leave_one_out"
        mock_args.random_seed = 42
        mock_parse_args.return_value = mock_args

        # Call the function
        args = parse_arguments()

        # Assertions
        self.assertEqual(args.model_type, "cnn")
        self.assertIsNone(args.config_path)
        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.output_dir, "output")
        self.assertEqual(args.gsr_sampling_rate, 32)
        self.assertEqual(args.video_fps, 30)
        self.assertEqual(args.cross_validation, "leave_one_out")
        self.assertEqual(args.random_seed, 42)

    @patch('src.scripts.train_model.ModelRegistry')
    @patch('src.scripts.train_model.ModelConfig')
    def test_build_model_from_config(self, mock_model_config, mock_registry):
        """Test building a model from a configuration."""
        # Setup mocks
        mock_model = MagicMock()
        mock_registry.create_model.return_value = mock_model
        mock_config_instance = MagicMock()
        # Configure the mock to return a valid framework
        mock_config_instance.get_framework.return_value = "pytorch"
        mock_model_config.return_value = mock_config_instance

        # Call the function
        input_shape = (50, 4)
        model_type = "cnn"
        config_path = None
        model = build_model_from_config(input_shape, model_type, config_path)

        # Assertions
        self.assertEqual(model, mock_model)
        mock_registry.create_model.assert_called_once()
        mock_model_config.assert_called_once()

    def test_create_training_metadata(self):
        """Test creating training metadata."""
        # Call the function
        model_type = "cnn"
        fold = 0
        subject_id = "Subject_01"
        input_shape = (50, 4)
        preprocessing_params = {"gsr_sampling_rate": 32, "video_fps": 30}
        training_params = {"batch_size": 32, "epochs": 10}
        metrics = {"loss": 0.1, "val_loss": 0.2}

        metadata = create_training_metadata(
            model_type, self.model_config, fold, subject_id, input_shape,
            preprocessing_params, training_params, metrics
        )

        # Assertions
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["model_type"], model_type)
        # The create_training_metadata function adds 1 to the fold number
        self.assertEqual(metadata["fold"], fold + 1)
        # The key is "subject_id", not "test_subject"
        self.assertEqual(metadata["subject_id"], subject_id)
        self.assertEqual(metadata["input_shape"], input_shape)
        # The keys in the metadata dict are "preprocessing" and "training", not "preprocessing_params" and "training_params"
        self.assertEqual(metadata["preprocessing"], preprocessing_params)
        self.assertEqual(metadata["training"], training_params)
        self.assertEqual(metadata["metrics"], metrics)
        self.assertIn("timestamp", metadata)
        self.assertIn("config", metadata)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_training_metadata(self, mock_json_dump, mock_file_open):
        """Test saving training metadata."""
        # Setup
        metadata = {
            "model_type": "cnn",
            "fold": 0,
            "test_subject": "Subject_01",
            "metrics": {"loss": 0.1}
        }
        output_dir = Path(self.temp_dir) / "output"
        model_type = "cnn"
        fold = 0
        subject_id = "Subject_01"

        # Call the function
        save_training_metadata(metadata, output_dir, model_type, fold, subject_id)

        # Assertions
        mock_file_open.assert_called_once()
        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        self.assertEqual(args[0], metadata)

    @patch('src.scripts.train_model.os.makedirs')
    def test_setup_callbacks(self, mock_makedirs):
        """Test setting up callbacks for training."""
        # Call the function
        fold = 0
        subject_id = "Subject_01"
        output_dir = Path(self.temp_dir) / "output"
        callbacks = setup_callbacks(self.model_config, fold, subject_id, output_dir)

        # Assertions
        self.assertIsInstance(callbacks, list)
        # Since TensorFlow is likely not available in the test environment,
        # the function should return an empty list
        self.assertEqual(len(callbacks), 0)


if __name__ == "__main__":
    unittest.main()
