#!/usr/bin/env python3
# src/tests/unit/processing/test_data_analysis.py

"""
Unit tests for the data_analysis.py module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import the modules to test
from src.processing.data_analysis import (
    GSRFeatureExtractor,
    PPGFeatureExtractor,
    DataVisualizer,
    DataAnalyzer
)

class TestGSRFeatureExtractor(unittest.TestCase):
    """Test cases for the GSRFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = GSRFeatureExtractor(sampling_rate=32)
        
        # Create a synthetic GSR signal for testing
        t = np.linspace(0, 10, 320)  # 10 seconds at 32 Hz
        self.gsr_signal = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.1, len(t))
    
    def test_extract_statistical_features(self):
        """Test extraction of statistical features."""
        features = self.extractor.extract_statistical_features(self.gsr_signal)
        
        # Check that the expected features are present
        expected_features = ["mean", "std", "min", "max", "range", "median", 
                            "percentile_25", "percentile_75", "iqr", 
                            "skewness", "kurtosis", "mean_derivative", "std_derivative"]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check some basic properties
        self.assertAlmostEqual(features["mean"], np.mean(self.gsr_signal), places=5)
        self.assertAlmostEqual(features["std"], np.std(self.gsr_signal), places=5)
        self.assertAlmostEqual(features["min"], np.min(self.gsr_signal), places=5)
        self.assertAlmostEqual(features["max"], np.max(self.gsr_signal), places=5)
    
    def test_extract_frequency_features(self):
        """Test extraction of frequency domain features."""
        features = self.extractor.extract_frequency_features(self.gsr_signal)
        
        # Check that the expected features are present
        expected_features = ["total_power", "vlf_power", "lf_power", "hf_power", 
                            "vlf_power_rel", "lf_power_rel", "hf_power_rel", 
                            "lf_hf_ratio", "peak_frequency"]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check that relative powers sum to approximately 1
        rel_power_sum = features["vlf_power_rel"] + features["lf_power_rel"] + features["hf_power_rel"]
        self.assertAlmostEqual(rel_power_sum, 1.0, places=5)
    
    def test_extract_nonlinear_features(self):
        """Test extraction of non-linear features."""
        features = self.extractor.extract_nonlinear_features(self.gsr_signal)
        
        # Check that the expected features are present
        expected_features = ["sample_entropy", "dfa_alpha", "poincare_sd1", "poincare_sd2"]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
    
    def test_extract_all_features(self):
        """Test extraction of all features."""
        features = self.extractor.extract_all_features(self.gsr_signal)
        
        # Check that features from all categories are present
        statistical_features = ["mean", "std", "skewness", "kurtosis"]
        frequency_features = ["total_power", "lf_hf_ratio", "peak_frequency"]
        nonlinear_features = ["sample_entropy", "dfa_alpha", "poincare_sd1"]
        
        for feature in statistical_features + frequency_features + nonlinear_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
    
    def test_empty_signal(self):
        """Test behavior with an empty signal."""
        empty_signal = np.array([])
        
        # All feature extraction methods should handle empty signals gracefully
        statistical_features = self.extractor.extract_statistical_features(empty_signal)
        frequency_features = self.extractor.extract_frequency_features(empty_signal)
        nonlinear_features = self.extractor.extract_nonlinear_features(empty_signal)
        all_features = self.extractor.extract_all_features(empty_signal)
        
        # Check that the features are present but have default values
        self.assertEqual(statistical_features["mean"], 0)
        self.assertEqual(frequency_features["total_power"], 0)
        self.assertEqual(nonlinear_features["sample_entropy"], 0)


class TestPPGFeatureExtractor(unittest.TestCase):
    """Test cases for the PPGFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PPGFeatureExtractor(sampling_rate=32)
        
        # Create a synthetic PPG signal for testing
        t = np.linspace(0, 10, 320)  # 10 seconds at 32 Hz
        # Create a signal with peaks at regular intervals (simulating heartbeats)
        self.ppg_signal = np.zeros_like(t)
        for i in range(1, 11):  # 10 heartbeats over 10 seconds (60 BPM)
            peak_loc = i * 32  # Peak every second (32 samples at 32 Hz)
            if peak_loc < len(t):
                # Create a Gaussian peak
                self.ppg_signal += 5 * np.exp(-0.5 * ((t - peak_loc/32) / 0.1) ** 2)
        
        # Add some noise
        self.ppg_signal += np.random.normal(0, 0.1, len(t))
    
    def test_extract_heart_rate(self):
        """Test extraction of heart rate features."""
        features = self.extractor.extract_heart_rate(self.ppg_signal)
        
        # Check that the expected features are present
        expected_features = ["heart_rate", "heart_rate_std"]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check that the heart rate is approximately 60 BPM (our synthetic signal)
        # Allow for some error due to peak detection and noise
        self.assertGreater(features["heart_rate"], 50)
        self.assertLess(features["heart_rate"], 70)
    
    def test_extract_hrv_features(self):
        """Test extraction of heart rate variability features."""
        features = self.extractor.extract_hrv_features(self.ppg_signal)
        
        # Check that the expected features are present
        expected_features = ["rmssd", "sdnn", "pnn50"]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
    
    def test_extract_pulse_wave_features(self):
        """Test extraction of pulse wave features."""
        features = self.extractor.extract_pulse_wave_features(self.ppg_signal)
        
        # Check that the expected features are present
        expected_features = ["pulse_amplitude", "pulse_width", "pulse_rise_time", "pulse_fall_time"]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
    
    def test_extract_all_features(self):
        """Test extraction of all features."""
        features = self.extractor.extract_all_features(self.ppg_signal)
        
        # Check that features from all categories are present
        hr_features = ["heart_rate", "heart_rate_std"]
        hrv_features = ["rmssd", "sdnn", "pnn50"]
        pulse_features = ["pulse_amplitude", "pulse_width", "pulse_rise_time", "pulse_fall_time"]
        
        for feature in hr_features + hrv_features + pulse_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
    
    def test_empty_signal(self):
        """Test behavior with an empty signal."""
        empty_signal = np.array([])
        
        # All feature extraction methods should handle empty signals gracefully
        hr_features = self.extractor.extract_heart_rate(empty_signal)
        hrv_features = self.extractor.extract_hrv_features(empty_signal)
        pulse_features = self.extractor.extract_pulse_wave_features(empty_signal)
        all_features = self.extractor.extract_all_features(empty_signal)
        
        # Check that the features are present but have default values
        self.assertEqual(hr_features["heart_rate"], 0)
        self.assertEqual(hrv_features["rmssd"], 0)
        self.assertEqual(pulse_features["pulse_amplitude"], 0)


class TestDataVisualizer(unittest.TestCase):
    """Test cases for the DataVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        self.visualizer = DataVisualizer(output_dir=self.test_dir)
        
        # Create a test DataFrame
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        self.test_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=n_samples, freq="1s"),
            "GSR": np.random.normal(0, 1, n_samples),
            "PPG": np.sin(np.linspace(0, 10 * np.pi, n_samples)),
            "Temperature": np.random.normal(37, 0.5, n_samples)
        })
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        # Test with a single column
        fig = self.visualizer.plot_time_series(
            self.test_df, ["GSR"], 
            title="GSR Time Series",
            save_as="gsr_time_series.png"
        )
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue((self.test_dir / "gsr_time_series.png").exists())
        
        # Test with multiple columns
        fig = self.visualizer.plot_time_series(
            self.test_df, ["GSR", "PPG", "Temperature"], 
            title="Multiple Time Series",
            save_as="multiple_time_series.png"
        )
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue((self.test_dir / "multiple_time_series.png").exists())
    
    def test_plot_correlation_matrix(self):
        """Test correlation matrix plotting."""
        fig = self.visualizer.plot_correlation_matrix(
            self.test_df,
            title="Correlation Matrix",
            save_as="correlation_matrix.png"
        )
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue((self.test_dir / "correlation_matrix.png").exists())
    
    def test_plot_feature_distributions(self):
        """Test feature distribution plotting."""
        fig = self.visualizer.plot_feature_distributions(
            self.test_df, ["GSR", "PPG", "Temperature"],
            title="Feature Distributions",
            save_as="feature_distributions.png"
        )
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue((self.test_dir / "feature_distributions.png").exists())
    
    def test_plot_pca_visualization(self):
        """Test PCA visualization plotting."""
        fig = self.visualizer.plot_pca_visualization(
            self.test_df, ["GSR", "PPG", "Temperature"],
            n_components=2,
            title="PCA Visualization",
            save_as="pca_visualization.png"
        )
        
        # Check that the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue((self.test_dir / "pca_visualization.png").exists())
    
    def test_invalid_columns(self):
        """Test behavior with invalid column names."""
        # Test with a non-existent column
        fig = self.visualizer.plot_time_series(
            self.test_df, ["NonExistentColumn"], 
            title="Invalid Column",
            save_as="invalid_column.png"
        )
        
        # Check that the figure was created (even if empty)
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the file was saved
        self.assertTrue((self.test_dir / "invalid_column.png").exists())


class TestDataAnalyzer(unittest.TestCase):
    """Test cases for the DataAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        self.analyzer = DataAnalyzer(output_dir=self.test_dir)
        
        # Create a mock session directory structure
        self.session_dir = self.test_dir / "Subject_Test_20250101_000000"
        self.session_dir.mkdir(parents=True)
        
        # Create mock data files
        self.create_mock_data_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def create_mock_data_files(self):
        """Create mock data files for testing."""
        # Create a mock GSR data file
        gsr_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
            "GSR": np.random.normal(0, 1, 100)
        })
        gsr_df.to_csv(self.session_dir / "gsr_data.csv", index=False)
        
        # Create a mock RGB data file
        rgb_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
            "RGB_R": np.random.normal(100, 10, 100),
            "RGB_G": np.random.normal(100, 10, 100),
            "RGB_B": np.random.normal(100, 10, 100)
        })
        rgb_df.to_csv(self.session_dir / "rgb_data.csv", index=False)
        
        # Create a mock thermal data file
        thermal_df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1s"),
            "Temperature": np.random.normal(37, 0.5, 100)
        })
        thermal_df.to_csv(self.session_dir / "thermal_data.csv", index=False)
    
    def test_extract_features(self):
        """Test feature extraction from loaded data."""
        # Create mock data dictionary
        mock_data = {
            'gsr': pd.DataFrame({
                'timestamp': pd.date_range(start="2023-01-01", periods=100, freq="1s"),
                'GSR_Phasic': np.random.normal(0, 1, 100)
            }),
            'aligned': pd.DataFrame({
                'timestamp': pd.date_range(start="2023-01-01", periods=100, freq="1s"),
                'GSR_Phasic': np.random.normal(0, 1, 100),
                'PPG_Signal': np.sin(np.linspace(0, 10 * np.pi, 100))
            })
        }
        
        # Extract features
        features_df = self.analyzer.extract_features(mock_data)
        
        # Check that the features DataFrame is not empty
        self.assertFalse(features_df.empty)
        
        # Check that GSR features are present
        gsr_features = [col for col in features_df.columns if col.startswith('GSR_')]
        self.assertTrue(len(gsr_features) > 0)
    
    def test_analyze_multiple_sessions(self):
        """Test analysis of multiple sessions."""
        # Mock the load_session_data and extract_features methods to avoid actual file loading
        original_load_session_data = self.analyzer.load_session_data
        original_extract_features = self.analyzer.extract_features
        
        def mock_load_session_data(session_path, gsr_sampling_rate=32):
            return {
                'gsr': pd.DataFrame({
                    'timestamp': pd.date_range(start="2023-01-01", periods=100, freq="1s"),
                    'GSR_Phasic': np.random.normal(0, 1, 100)
                }),
                'aligned': pd.DataFrame({
                    'timestamp': pd.date_range(start="2023-01-01", periods=100, freq="1s"),
                    'GSR_Phasic': np.random.normal(0, 1, 100),
                    'PPG_Signal': np.sin(np.linspace(0, 10 * np.pi, 100))
                })
            }
        
        def mock_extract_features(data):
            return pd.DataFrame({
                'GSR_mean': [np.random.normal(0, 1)],
                'GSR_std': [np.random.normal(0, 0.1)],
                'PPG_heart_rate': [np.random.normal(60, 5)]
            })
        
        try:
            # Replace the methods with mocks
            self.analyzer.load_session_data = mock_load_session_data
            self.analyzer.extract_features = mock_extract_features
            
            # Analyze multiple sessions
            results = self.analyzer.analyze_multiple_sessions(
                [self.session_dir],
                gsr_sampling_rate=32,
                save_visualizations=True
            )
            
            # Check that the results dictionary is not empty
            self.assertTrue(len(results) > 0)
            
            # Check that combined features are present
            self.assertIn('combined_features', results)
            self.assertFalse(results['combined_features'].empty)
            
        finally:
            # Restore the original methods
            self.analyzer.load_session_data = original_load_session_data
            self.analyzer.extract_features = original_extract_features


if __name__ == "__main__":
    unittest.main()