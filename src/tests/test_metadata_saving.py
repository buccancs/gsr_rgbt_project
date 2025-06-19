# src/tests/test_metadata_saving.py

import unittest
import json
import tempfile
from pathlib import Path
import sys
import os
import shutil

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.scripts.train_model import (
    create_training_metadata,
    save_training_metadata
)
from src.ml_models.model_config import ModelConfig


class TestMetadataSaving(unittest.TestCase):
    """Test cases for the metadata saving functionality in train_model.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())

        # Create a model configuration for testing
        self.model_config = ModelConfig("lstm")

        # Create test parameters
        self.model_type = "lstm"
        self.fold = 0
        self.subject_id = "TestSubject01"
        self.input_shape = (32, 10)  # (window_size, features)

        # Create preprocessing parameters
        self.preprocessing_params = {
            "gsr_sampling_rate": 32,
            "video_fps": 30,
            "scaler": "StandardScaler",
            "validation_split": 0.2,
            "train_samples": 100,
            "val_samples": 20,
            "test_samples": 30
        }

        # Create training parameters
        self.training_params = {
            "framework": "pytorch",
            "fit_params": {
                "epochs": 100,
                "batch_size": 32,
                "early_stopping": {
                    "patience": 10,
                    "monitor": "val_loss"
                }
            },
            "history": {
                "loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4]
            }
        }

        # Create metrics
        self.metrics = {
            "mse": 0.25,
            "mae": 0.4,
            "r2": 0.75
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_create_training_metadata(self):
        """Test the create_training_metadata function."""
        # Create metadata
        metadata = create_training_metadata(
            model_type=self.model_type,
            model_config=self.model_config,
            fold=self.fold,
            subject_id=self.subject_id,
            input_shape=self.input_shape,
            preprocessing_params=self.preprocessing_params,
            training_params=self.training_params,
            metrics=self.metrics
        )

        # Check that metadata is a dictionary
        self.assertIsInstance(metadata, dict)

        # Check that all expected keys are present
        expected_keys = [
            "model_type", "model_name", "framework", "fold", "subject_id",
            "input_shape", "preprocessing", "training", "metrics", "timestamp", "config"
        ]
        for key in expected_keys:
            self.assertIn(key, metadata)

        # Check specific values
        self.assertEqual(metadata["model_type"], self.model_type)
        self.assertEqual(metadata["fold"], self.fold + 1)
        self.assertEqual(metadata["subject_id"], self.subject_id)
        self.assertEqual(metadata["input_shape"], self.input_shape)
        self.assertEqual(metadata["preprocessing"], self.preprocessing_params)
        self.assertEqual(metadata["training"], self.training_params)
        self.assertEqual(metadata["metrics"], self.metrics)

        # Check that timestamp is present
        self.assertIn("timestamp", metadata)

        # Check that config is present
        self.assertIn("config", metadata)

    def test_save_training_metadata(self):
        """Test the save_training_metadata function."""
        # Create metadata
        metadata = create_training_metadata(
            model_type=self.model_type,
            model_config=self.model_config,
            fold=self.fold,
            subject_id=self.subject_id,
            input_shape=self.input_shape,
            preprocessing_params=self.preprocessing_params,
            training_params=self.training_params,
            metrics=self.metrics
        )

        # Save metadata
        metadata_path = save_training_metadata(
            metadata=metadata,
            output_dir=self.test_dir,
            model_type=self.model_type,
            fold=self.fold,
            subject_id=self.subject_id
        )

        # Check that the metadata file was created
        self.assertTrue(metadata_path.exists())

        # Check that the metadata file is in the correct location
        expected_path = self.test_dir / "metadata" / f"{self.model_type}_fold_{self.fold+1}_subject_{self.subject_id}_metadata.json"
        self.assertEqual(metadata_path, expected_path)

        # Load the metadata file and check its contents
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)

        # Check that the loaded metadata matches the original metadata
        self.assertEqual(loaded_metadata["model_type"], metadata["model_type"])
        self.assertEqual(loaded_metadata["fold"], metadata["fold"])
        self.assertEqual(loaded_metadata["subject_id"], metadata["subject_id"])

        # JSON serialization converts tuples to lists, so we need to compare the values, not the types
        self.assertEqual(list(metadata["input_shape"]), loaded_metadata["input_shape"])
        self.assertEqual(loaded_metadata["preprocessing"], metadata["preprocessing"])
        self.assertEqual(loaded_metadata["training"], metadata["training"])
        self.assertEqual(loaded_metadata["metrics"], metadata["metrics"])


if __name__ == "__main__":
    unittest.main()
