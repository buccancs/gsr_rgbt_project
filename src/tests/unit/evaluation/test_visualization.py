# src/tests/unit/evaluation/test_visualization.py

import unittest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.visualization import plot_prediction_vs_ground_truth, plot_bland_altman

class TestVisualization(unittest.TestCase):
    """
    Unit tests for the visualization functions.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Create a sample DataFrame for testing
        self.subject_id = "TestSubject01"
        self.time_index = pd.date_range(start='2023-01-01', periods=100, freq='S')
        self.ground_truth = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
        self.prediction = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.2, 100)
        
        self.results_df = pd.DataFrame({
            'ground_truth': self.ground_truth,
            'prediction': self.prediction
        }, index=self.time_index)
        
    def tearDown(self):
        """
        Clean up after each test method.
        """
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
    @patch('src.evaluation.visualization.plt.savefig')
    @patch('src.evaluation.visualization.logging')
    def test_plot_prediction_vs_ground_truth(self, mock_logging, mock_savefig):
        """
        Test that plot_prediction_vs_ground_truth creates and saves a plot correctly.
        """
        # Call the function
        plot_prediction_vs_ground_truth(self.results_df, self.subject_id, self.output_dir)
        
        # Check that savefig was called with the correct path
        expected_path = self.output_dir / f"{self.subject_id}_prediction_vs_truth.png"
        mock_savefig.assert_called_with(expected_path, dpi=300, bbox_inches="tight")
        
        # Check that a success message was logged
        mock_logging.info.assert_called_with(f"Saved prediction plot to {expected_path}")
        
    @patch('src.evaluation.visualization.plt.savefig')
    @patch('src.evaluation.visualization.logging')
    def test_plot_prediction_vs_ground_truth_invalid_df(self, mock_logging, mock_savefig):
        """
        Test that plot_prediction_vs_ground_truth handles invalid DataFrames correctly.
        """
        # Create an invalid DataFrame (missing required columns)
        invalid_df = pd.DataFrame({'some_column': [1, 2, 3]})
        
        # Call the function with the invalid DataFrame
        plot_prediction_vs_ground_truth(invalid_df, self.subject_id, self.output_dir)
        
        # Check that savefig was not called
        mock_savefig.assert_not_called()
        
        # Check that an error was logged
        mock_logging.error.assert_called_with("Invalid DataFrame passed to plotting function.")
        
    @patch('src.evaluation.visualization.plt.savefig')
    @patch('src.evaluation.visualization.logging')
    def test_plot_prediction_vs_ground_truth_save_error(self, mock_logging, mock_savefig):
        """
        Test that plot_prediction_vs_ground_truth handles save errors correctly.
        """
        # Make savefig raise an exception
        mock_savefig.side_effect = Exception("Test exception")
        
        # Call the function
        plot_prediction_vs_ground_truth(self.results_df, self.subject_id, self.output_dir)
        
        # Check that an error was logged
        mock_logging.error.assert_called_with("Failed to save plot: Test exception")
        
    @patch('src.evaluation.visualization.plt.savefig')
    @patch('src.evaluation.visualization.logging')
    def test_plot_bland_altman(self, mock_logging, mock_savefig):
        """
        Test that plot_bland_altman creates and saves a plot correctly.
        """
        # Call the function
        plot_bland_altman(self.results_df, self.subject_id, self.output_dir)
        
        # Check that savefig was called with the correct path
        expected_path = self.output_dir / f"{self.subject_id}_bland_altman_plot.png"
        mock_savefig.assert_called_with(expected_path, dpi=300, bbox_inches="tight")
        
        # Check that a success message was logged
        mock_logging.info.assert_called_with(f"Saved Bland-Altman plot to {expected_path}")
        
    @patch('src.evaluation.visualization.plt.savefig')
    @patch('src.evaluation.visualization.logging')
    def test_plot_bland_altman_invalid_df(self, mock_logging, mock_savefig):
        """
        Test that plot_bland_altman handles invalid DataFrames correctly.
        """
        # Create an invalid DataFrame (missing required columns)
        invalid_df = pd.DataFrame({'some_column': [1, 2, 3]})
        
        # Call the function with the invalid DataFrame
        plot_bland_altman(invalid_df, self.subject_id, self.output_dir)
        
        # Check that savefig was not called
        mock_savefig.assert_not_called()
        
        # Check that an error was logged
        mock_logging.error.assert_called_with("Invalid DataFrame passed to Bland-Altman plotting function.")
        
    @patch('src.evaluation.visualization.plt.savefig')
    @patch('src.evaluation.visualization.logging')
    def test_plot_bland_altman_save_error(self, mock_logging, mock_savefig):
        """
        Test that plot_bland_altman handles save errors correctly.
        """
        # Make savefig raise an exception
        mock_savefig.side_effect = Exception("Test exception")
        
        # Call the function
        plot_bland_altman(self.results_df, self.subject_id, self.output_dir)
        
        # Check that an error was logged
        mock_logging.error.assert_called_with("Failed to save Bland-Altman plot: Test exception")
        
    def test_plot_prediction_vs_ground_truth_integration(self):
        """
        Integration test for plot_prediction_vs_ground_truth.
        
        This test actually creates and saves a plot to the temporary directory.
        """
        # Call the function
        plot_prediction_vs_ground_truth(self.results_df, self.subject_id, self.output_dir)
        
        # Check that the plot file was created
        expected_path = self.output_dir / f"{self.subject_id}_prediction_vs_truth.png"
        self.assertTrue(expected_path.exists())
        
    def test_plot_bland_altman_integration(self):
        """
        Integration test for plot_bland_altman.
        
        This test actually creates and saves a plot to the temporary directory.
        """
        # Call the function
        plot_bland_altman(self.results_df, self.subject_id, self.output_dir)
        
        # Check that the plot file was created
        expected_path = self.output_dir / f"{self.subject_id}_bland_altman_plot.png"
        self.assertTrue(expected_path.exists())

if __name__ == '__main__':
    unittest.main()