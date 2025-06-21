#!/usr/bin/env python3
# src/tests/unit/scripts/test_analyze_and_train.py

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.analyze_and_train import (
    prepare_features_for_training,
    reshape_features_for_model,
    evaluate_model,
    plot_predictions,
    parse_arguments,
    main
)

class TestPrepareFeatures(unittest.TestCase):
    """Test the prepare_features_for_training function."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with features and a target
        self.features_df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'target': np.random.rand(100)
        })
    
    def test_prepare_features_normal_case(self):
        """Test prepare_features_for_training with normal inputs."""
        X_train, X_test, y_train, y_test = prepare_features_for_training(
            self.features_df, 'target', test_size=0.2, random_state=42
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 80)  # 80% of 100
        self.assertEqual(X_test.shape[0], 20)   # 20% of 100
        self.assertEqual(X_train.shape[1], 3)   # 3 features
        self.assertEqual(y_train.shape[0], 80)  # 80% of 100
        self.assertEqual(y_test.shape[0], 20)   # 20% of 100
        
        # Check types
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
    
    def test_prepare_features_invalid_target(self):
        """Test prepare_features_for_training with an invalid target feature."""
        with self.assertRaises(ValueError):
            prepare_features_for_training(
                self.features_df, 'nonexistent_target', test_size=0.2, random_state=42
            )
    
    def test_prepare_features_empty_dataframe(self):
        """Test prepare_features_for_training with an empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            prepare_features_for_training(
                empty_df, 'target', test_size=0.2, random_state=42
            )

class TestReshapeFeatures(unittest.TestCase):
    """Test the reshape_features_for_model function."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample arrays
        self.X_train = np.random.rand(80, 3)
        self.X_test = np.random.rand(20, 3)
    
    def test_reshape_features_lstm(self):
        """Test reshape_features_for_model with LSTM model type."""
        X_train_reshaped, X_test_reshaped = reshape_features_for_model(
            self.X_train, self.X_test, model_type='lstm'
        )
        
        # Check shapes
        self.assertEqual(X_train_reshaped.shape, (80, 1, 3))
        self.assertEqual(X_test_reshaped.shape, (20, 1, 3))
    
    def test_reshape_features_cnn_lstm(self):
        """Test reshape_features_for_model with CNN_LSTM model type."""
        X_train_reshaped, X_test_reshaped = reshape_features_for_model(
            self.X_train, self.X_test, model_type='cnn_lstm'
        )
        
        # Check shapes
        self.assertEqual(X_train_reshaped.shape, (80, 1, 3))
        self.assertEqual(X_test_reshaped.shape, (20, 1, 3))
    
    def test_reshape_features_transformer(self):
        """Test reshape_features_for_model with Transformer model type."""
        X_train_reshaped, X_test_reshaped = reshape_features_for_model(
            self.X_train, self.X_test, model_type='transformer'
        )
        
        # Check shapes
        self.assertEqual(X_train_reshaped.shape, (80, 1, 3))
        self.assertEqual(X_test_reshaped.shape, (20, 1, 3))
    
    def test_reshape_features_non_sequence(self):
        """Test reshape_features_for_model with non-sequence model type."""
        X_train_reshaped, X_test_reshaped = reshape_features_for_model(
            self.X_train, self.X_test, model_type='cnn'
        )
        
        # Check that arrays are unchanged
        np.testing.assert_array_equal(X_train_reshaped, self.X_train)
        np.testing.assert_array_equal(X_test_reshaped, self.X_test)

class TestEvaluateModel(unittest.TestCase):
    """Test the evaluate_model function."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample arrays
        self.X_test = np.random.rand(20, 3)
        self.y_test = np.random.rand(20)
        
        # Create a mock model
        self.model = MagicMock()
        self.model.predict.return_value = self.y_test * 0.9 + 0.05  # Slightly different predictions
    
    def test_evaluate_model(self):
        """Test evaluate_model with normal inputs."""
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        
        # Check that the model's predict method was called
        self.model.predict.assert_called_once_with(self.X_test)
        
        # Check that metrics were calculated
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['r2'], 0)  # R² should be positive for our test data
        self.assertLessEqual(metrics['mse'], 1)    # MSE should be small for our test data

class TestPlotPredictions(unittest.TestCase):
    """Test the plot_predictions function."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample arrays
        self.y_test = np.random.rand(20)
        self.y_pred = self.y_test * 0.9 + 0.05  # Slightly different predictions
        
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_predictions(self, mock_savefig):
        """Test plot_predictions with normal inputs."""
        plot_predictions(
            self.y_test, self.y_pred,
            target_feature='test_feature',
            output_dir=self.output_dir
        )
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
        
        # Check that the filename contains the target feature
        args, kwargs = mock_savefig.call_args
        self.assertIn('test_feature', str(args[0]))

class TestParseArguments(unittest.TestCase):
    """Test the parse_arguments function."""
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_arguments(self, mock_parse_args):
        """Test parse_arguments with normal inputs."""
        # Set up mock return value
        mock_args = MagicMock()
        mock_args.data_dir = 'data/recordings'
        mock_args.output_dir = 'output/results'
        mock_args.gsr_sampling_rate = 32
        mock_args.save_visualizations = True
        mock_args.model_type = 'lstm'
        mock_args.config_path = None
        mock_args.target_feature = 'GSR_mean'
        mock_args.test_size = 0.2
        mock_args.random_state = 42
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = parse_arguments()
        
        # Check that the arguments were parsed correctly
        self.assertEqual(args.data_dir, 'data/recordings')
        self.assertEqual(args.output_dir, 'output/results')
        self.assertEqual(args.gsr_sampling_rate, 32)
        self.assertTrue(args.save_visualizations)
        self.assertEqual(args.model_type, 'lstm')
        self.assertIsNone(args.config_path)
        self.assertEqual(args.target_feature, 'GSR_mean')
        self.assertEqual(args.test_size, 0.2)
        self.assertEqual(args.random_state, 42)

class TestMain(unittest.TestCase):
    """Test the main function."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Create a mock session directory
        self.session_dir = Path(self.temp_dir) / 'Subject_001_20250101_000000'
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.scripts.analyze_and_train.parse_arguments')
    @patch('src.scripts.analyze_and_train.DataAnalyzer')
    @patch('src.scripts.analyze_and_train.prepare_features_for_training')
    @patch('src.scripts.analyze_and_train.reshape_features_for_model')
    @patch('src.scripts.analyze_and_train.build_model_from_config')
    @patch('src.scripts.analyze_and_train.evaluate_model')
    @patch('src.scripts.analyze_and_train.plot_predictions')
    def test_main(self, mock_plot, mock_evaluate, mock_build, mock_reshape, 
                 mock_prepare, mock_analyzer, mock_parse_args):
        """Test main function with mocked dependencies."""
        # Set up mock return values
        mock_args = MagicMock()
        mock_args.data_dir = Path(self.temp_dir)
        mock_args.output_dir = self.output_dir
        mock_args.gsr_sampling_rate = 32
        mock_args.save_visualizations = True
        mock_args.model_type = 'lstm'
        mock_args.config_path = None
        mock_args.target_feature = 'GSR_mean'
        mock_args.test_size = 0.2
        mock_args.random_state = 42
        mock_parse_args.return_value = mock_args
        
        # Mock DataAnalyzer
        mock_analyzer_instance = mock_analyzer.return_value
        mock_results = {'combined_features': pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'GSR_mean': np.random.rand(10)
        })}
        mock_analyzer_instance.analyze_multiple_sessions.return_value = mock_results
        
        # Mock prepare_features_for_training
        mock_X_train = np.random.rand(8, 2)
        mock_X_test = np.random.rand(2, 2)
        mock_y_train = np.random.rand(8)
        mock_y_test = np.random.rand(2)
        mock_prepare.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)
        
        # Mock reshape_features_for_model
        mock_X_train_reshaped = np.random.rand(8, 1, 2)
        mock_X_test_reshaped = np.random.rand(2, 1, 2)
        mock_reshape.return_value = (mock_X_train_reshaped, mock_X_test_reshaped)
        
        # Mock build_model_from_config
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.5, 0.4]}
        mock_model.fit.return_value = mock_history
        mock_model.predict.return_value = np.random.rand(2)
        mock_build.return_value = mock_model
        
        # Mock evaluate_model
        mock_metrics = {'mse': 0.1, 'rmse': 0.3, 'mae': 0.2, 'r2': 0.8}
        mock_evaluate.return_value = mock_metrics
        
        # Call the function
        with patch('matplotlib.pyplot.savefig'), \
             patch('builtins.open'), \
             patch('json.dump'):
            main()
        
        # Check that the functions were called
        mock_parse_args.assert_called_once()
        mock_analyzer.assert_called_once()
        mock_analyzer_instance.analyze_multiple_sessions.assert_called_once()
        mock_prepare.assert_called_once()
        mock_reshape.assert_called_once()
        mock_build.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_plot.assert_called_once()
        mock_model.save.assert_called_once()

if __name__ == '__main__':
    unittest.main()