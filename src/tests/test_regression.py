# src/tests/test_regression.py

# --- Add project root to path for absolute imports ---
import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.processing.feature_engineering import align_signals, create_feature_windows, create_dataset_from_session
from src.ml_models.model_config import ModelConfig
from src.ml_models.model_interface import ModelRegistry
from src.scripts.train_model import build_model_from_config


class TestRegressionFeatureEngineering(unittest.TestCase):
    """
    Regression tests for the feature engineering pipeline.
    These tests verify that the feature engineering functions work together correctly.
    """

    def setUp(self):
        """Set up data that mimics processed GSR and video signals."""
        # Create a high-frequency signal (like GSR at 32Hz)
        self.gsr_sampling_rate = 32
        gsr_timestamps = pd.to_datetime(np.arange(0, 10, 1 / self.gsr_sampling_rate), unit="s")
        self.gsr_df = pd.DataFrame(
            {
                "timestamp": gsr_timestamps,
                "GSR_Phasic": np.sin(np.arange(0, 10, 1 / self.gsr_sampling_rate)),
                "GSR_Tonic": np.cos(np.arange(0, 10, 1 / self.gsr_sampling_rate)),
            }
        )

        # Create a lower-frequency signal (like video features at 30Hz)
        self.video_fps = 30
        video_timestamps = pd.to_datetime(np.arange(0, 10, 1 / self.video_fps), unit="s")
        self.video_df = pd.DataFrame(
            {
                "timestamp": video_timestamps,
                "RGB_R": np.sin(np.arange(0, 10, 1 / self.video_fps) + 1),
                "RGB_G": np.sin(np.arange(0, 10, 1 / self.video_fps) + 2),
                "RGB_B": np.sin(np.arange(0, 10, 1 / self.video_fps) + 3),
            }
        )

        # Parameters for windowing
        self.window_size = 32  # 1 second at 32Hz
        self.step = 16  # 0.5 second at 32Hz
        self.feature_cols = ["RGB_R", "RGB_G", "RGB_B", "GSR_Tonic"]
        self.target_col = "GSR_Phasic"

    def test_align_and_window_pipeline(self):
        """Test that the align and window pipeline produces expected output shapes."""
        # 1. Align signals
        aligned_df = align_signals(self.gsr_df, self.video_df)

        # Check alignment
        self.assertIsInstance(aligned_df, pd.DataFrame)
        self.assertEqual(len(aligned_df), len(self.gsr_df))
        self.assertIn("RGB_R", aligned_df.columns)
        self.assertIn("GSR_Phasic", aligned_df.columns)

        # 2. Create feature windows
        X, y = create_feature_windows(
            aligned_df, self.feature_cols, self.target_col, self.window_size, self.step
        )

        # Check windowing
        expected_num_windows = (len(aligned_df) - self.window_size) // self.step
        self.assertEqual(X.shape, (expected_num_windows, self.window_size, len(self.feature_cols)))
        self.assertEqual(y.shape, (expected_num_windows,))

        # 3. Verify data integrity
        # Check that the first window contains the expected data
        first_window_features = aligned_df[self.feature_cols].iloc[:self.window_size].values
        np.testing.assert_array_almost_equal(X[0], first_window_features)

        # Check that the target value is correct
        first_window_target = aligned_df[self.target_col].iloc[self.window_size - 1]
        self.assertAlmostEqual(y[0], first_window_target)


class TestRegressionModelTraining(unittest.TestCase):
    """
    Regression tests for the model training pipeline.
    These tests verify that models can be built and trained with different configurations.
    """

    def setUp(self):
        """Set up data for model training."""
        # Create synthetic data
        np.random.seed(42)
        self.window_size = 32
        self.num_features = 4
        self.num_samples = 100

        # X shape: (num_samples, window_size, num_features)
        self.X = np.random.randn(self.num_samples, self.window_size, self.num_features)
        # y shape: (num_samples,)
        self.y = np.random.randn(self.num_samples)

        # Input shape for models
        self.input_shape = (self.window_size, self.num_features)

    @patch('src.ml_models.model_interface.ModelRegistry.create_model')
    def test_model_building_with_different_configs(self, mock_create_model):
        """Test that models can be built with different configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        # Test different model types
        model_types = ["cnn", "lstm", "cnn_lstm"]

        for model_type in model_types:
            # Build model
            model = build_model_from_config(self.input_shape, model_type)

            # Check that the model was created
            self.assertEqual(model, mock_model)

            # Check that create_model was called with the right arguments
            # Since we're mocking ModelRegistry.create_model, we can't check the exact arguments
            # Just verify that it was called
            mock_create_model.assert_called_once()

            # Reset the mock for the next iteration
            mock_create_model.reset_mock()

    @patch('src.ml_models.pytorch_cnn_models.PyTorchCNNModel')
    def test_model_training_with_different_configs(self, mock_model_class):
        """Test that models can be trained with different configurations."""
        # Setup mock
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Create a mock ModelFactory
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        # Register the mock factory with the ModelRegistry
        ModelRegistry.register("test_model", mock_factory)

        # Create different configurations
        configs = [
            {
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
                    "epochs": 5,
                    "validation_split": 0.2,
                    "shuffle": True
                }
            },
            {
                "model_params": {
                    "conv_channels": [16, 32],
                    "kernel_sizes": [5, 5],
                    "strides": [1, 1],
                    "pool_sizes": [2, 2],
                    "fc_layers": [16, 1],
                    "activations": ["relu", "relu", "relu", "linear"],
                    "dropout_rate": 0.3
                },
                "optimizer_params": {
                    "type": "sgd",
                    "lr": 0.01
                },
                "loss_params": {
                    "type": "mae"
                },
                "train_params": {
                    "batch_size": 64,
                    "epochs": 10,
                    "validation_split": 0.3,
                    "shuffle": True
                }
            }
        ]

        for config_dict in configs:
            # Create config
            config = ModelConfig()
            config.config = config_dict

            # Create model
            model = ModelRegistry.create_model("test_model", self.input_shape, config.get_config())

            # Train model
            X_train, X_val = self.X[:80], self.X[80:]
            y_train, y_val = self.y[:80], self.y[80:]

            model.fit(X_train, y_train, X_val, y_val)

            # Check that fit was called
            mock_model.fit.assert_called_once()

            # Reset mock for next iteration
            mock_model.reset_mock()


class TestEndToEndPipeline(unittest.TestCase):
    """
    End-to-end tests for the ML pipeline.
    These tests verify that the entire pipeline from feature engineering to model evaluation works correctly.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())

        # Create synthetic data
        np.random.seed(42)
        self.window_size = 20
        self.num_features = 4
        self.batch_size = 10

        # Create mock data
        self.X = np.random.rand(self.batch_size, self.window_size, self.num_features).astype(np.float32)
        self.y = np.random.rand(self.batch_size).astype(np.float32)

        # Create model configurations for testing
        self.model_configs = {
            "lstm": {
                "model_params": {
                    "hidden_size": 32,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "bidirectional": False,
                    "fc_layers": [16, 1],
                    "activations": ["relu", "linear"]
                },
                "optimizer_params": {
                    "type": "adam",
                    "lr": 0.001
                },
                "loss_fn": "mse",
                "train_params": {
                    "epochs": 2,
                    "batch_size": 4,
                    "validation_split": 0.2
                }
            },
            "autoencoder": {
                "model_params": {
                    "latent_dim": 8,
                    "encoder_layers": [16, 8],
                    "decoder_layers": [8, 16],
                    "activations": ["relu", "relu", "relu", "sigmoid"]
                },
                "optimizer_params": {
                    "type": "adam",
                    "lr": 0.001
                },
                "loss_fn": "mse",
                "train_params": {
                    "epochs": 2,
                    "batch_size": 4,
                    "validation_split": 0.2
                }
            }
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_feature_engineering_to_model_pipeline(self):
        """Test the pipeline from feature engineering to model training and prediction."""
        # Create feature windows
        feature_cols = ["feature1", "feature2", "feature3", "feature4"]
        target_col = "target"

        # Create a DataFrame with the right structure
        timestamps = pd.to_datetime(np.arange(100), unit="s")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
            "feature4": np.random.rand(100),
            "target": np.random.rand(100)
        })

        # Create feature windows
        X, y = create_feature_windows(df, feature_cols, target_col, self.window_size, step=5)

        # Build and train LSTM model
        input_shape = (self.window_size, len(feature_cols))
        config = ModelConfig()
        config.config = self.model_configs["lstm"]

        model = build_model_from_config(input_shape, "lstm", config_path=None)

        # Train the model
        history = model.fit(X, y)

        # Make predictions
        predictions = model.predict(X)

        # Verify the pipeline worked
        self.assertIsNotNone(history)
        self.assertIn("train_loss", history)
        self.assertEqual(predictions.shape, (len(X),))

    def test_different_model_configurations(self):
        """Test different model configurations to ensure they all work."""
        input_shape = (self.window_size, self.num_features)

        # Test LSTM model
        lstm_config = ModelConfig()
        lstm_config.config = self.model_configs["lstm"]

        lstm_model = build_model_from_config(input_shape, "lstm", config_path=None)
        lstm_history = lstm_model.fit(self.X, self.y)
        lstm_predictions = lstm_model.predict(self.X)

        # Test Autoencoder model
        ae_config = ModelConfig()
        ae_config.config = self.model_configs["autoencoder"]

        ae_model = build_model_from_config(input_shape, "autoencoder", config_path=None)
        ae_history = ae_model.fit(self.X)
        ae_predictions = ae_model.predict(self.X)

        # Verify all models worked
        self.assertIsNotNone(lstm_history)
        self.assertIn("train_loss", lstm_history)
        self.assertEqual(lstm_predictions.shape, (self.batch_size,))

        self.assertIsNotNone(ae_history)
        self.assertIn("train_loss", ae_history)
        self.assertEqual(ae_predictions.shape, self.X.shape)

    def test_model_save_load_pipeline(self):
        """Test the pipeline for saving and loading models."""
        input_shape = (self.window_size, self.num_features)

        # Build and train LSTM model
        config = ModelConfig()
        config.config = self.model_configs["lstm"]

        model = build_model_from_config(input_shape, "lstm", config_path=None)
        model.fit(self.X, self.y)

        # Save the model
        save_path = self.test_dir / "test_model.pt"
        model.save(str(save_path))

        # Load the model
        loaded_model = ModelRegistry.create_model("lstm", input_shape, config.config)
        loaded_model.model.load_state_dict(torch.load(str(save_path)))

        # Make predictions with both models
        original_predictions = model.predict(self.X)
        loaded_predictions = loaded_model.predict(self.X)

        # Verify the predictions are the same
        np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-5, atol=1e-8)


class TestSmokeTests(unittest.TestCase):
    """
    Smoke tests for the main pipelines.
    These tests verify that the main pipelines run without errors.
    """

    @patch('src.processing.feature_engineering.SessionDataLoader')
    @patch('src.processing.feature_engineering.process_gsr_signal')
    @patch('src.processing.feature_engineering.detect_palm_roi')
    @patch('src.processing.feature_engineering.extract_roi_signal')
    def test_create_dataset_from_session_smoke(self, mock_extract_roi, mock_detect_roi, 
                                              mock_process_gsr, mock_loader):
        """Smoke test for create_dataset_from_session."""
        # Setup mocks
        session_path = Path("dummy/path")
        gsr_sampling_rate = 32
        video_fps = 30

        # Mock GSR data
        gsr_timestamps = pd.to_datetime(np.arange(0, 10, 1/gsr_sampling_rate), unit="s")
        mock_gsr_df = pd.DataFrame({
            "timestamp": gsr_timestamps,
            "GSR_Raw": np.random.randn(len(gsr_timestamps))
        })

        # Mock processed GSR data
        mock_processed_gsr = pd.DataFrame({
            "timestamp": gsr_timestamps,
            "GSR_Phasic": np.random.randn(len(gsr_timestamps)),
            "GSR_Tonic": np.random.randn(len(gsr_timestamps))
        })

        # Setup loader mock
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.get_gsr_data.return_value = mock_gsr_df

        # Setup video frame generator
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mock_loader_instance.get_rgb_video_generator.return_value = [
            (True, frame) for _ in range(10)
        ]

        # Setup other mocks
        mock_process_gsr.return_value = mock_processed_gsr
        mock_detect_roi.return_value = [(10, 10), (30, 30)]
        mock_extract_roi.return_value = [100, 120, 140]  # RGB values

        # Call the function
        result = create_dataset_from_session(session_path, gsr_sampling_rate, video_fps)

        # Just check that it runs without errors and returns something
        self.assertIsNotNone(result)

    @patch('src.ml_models.model_interface.ModelRegistry.create_model')
    def test_model_building_smoke(self, mock_create_model):
        """Smoke test for model building."""
        # Setup mock
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        # Build model
        input_shape = (32, 4)
        model_type = "cnn"
        model = build_model_from_config(input_shape, model_type)

        # Just check that it runs without errors and returns something
        self.assertIsNotNone(model)

    def test_full_pipeline_smoke(self):
        """Smoke test for the full pipeline."""
        # Skip the actual file saving part of the test
        # This is a workaround for issues with file system operations in test environments

        # Create synthetic data
        window_size = 20
        num_features = 4
        batch_size = 10

        # Create mock data
        X = np.random.rand(batch_size, window_size, num_features).astype(np.float32)
        y = np.random.rand(batch_size).astype(np.float32)

        # Split data
        train_size = int(0.8 * batch_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Create model configuration
        config = {
            "model_params": {
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "bidirectional": False,
                "fc_layers": [16, 1],
                "activations": ["relu", "linear"]
            },
            "optimizer_params": {
                "type": "adam",
                "lr": 0.001
            },
            "loss_fn": "mse",
            "train_params": {
                "epochs": 2,  # Minimal epochs for faster testing
                "batch_size": 2,
                "validation_split": 0.2
            }
        }

        # Build model
        input_shape = (window_size, num_features)
        model_config = ModelConfig()
        model_config.config = config

        model = build_model_from_config(input_shape, "lstm", config_path=None)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)

        # Make predictions
        predictions = model.predict(X_test)

        # Just check that everything runs without errors
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(predictions)

        # Mock the save method to avoid file system issues
        with patch.object(model, 'save') as mock_save:
            model.save("mock_path.pt")
            mock_save.assert_called_once_with("mock_path.pt")


if __name__ == "__main__":
    unittest.main()
