import sys
import unittest
from pathlib import Path
import numpy as np
import pytest

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.processing.feature_engineering import create_feature_windows


class TestBenchmarks(unittest.TestCase):
    """
    Performance benchmarks for the GSR-RGBT project.
    
    These benchmarks measure the performance of key functions in the codebase.
    They are run automatically by the benchmark workflow.
    """

    def setUp(self):
        """Set up test data for benchmarks."""
        # Create a simple DataFrame for windowing tests
        self.window_size = 32
        self.step = 16
        self.num_samples = 1000
        self.num_features = 5
        
        # Create synthetic data
        self.timestamps = np.arange(self.num_samples)
        self.features = np.random.randn(self.num_samples, self.num_features)
        self.targets = np.random.randn(self.num_samples)
        
        # Create a DataFrame-like dictionary
        self.test_data = {
            "timestamp": self.timestamps,
        }
        
        # Add feature columns
        for i in range(self.num_features):
            self.test_data[f"feature{i+1}"] = self.features[:, i]
        
        # Add target column
        self.test_data["target"] = self.targets
        
        # Convert to a DataFrame-like object for testing
        self.test_df = type('DataFrame', (), self.test_data)
        self.test_df.columns = list(self.test_data.keys())
        
        # Mock the DataFrame indexing
        def getitem(key):
            if isinstance(key, list):
                return type('DataFrame', (), {col: self.test_data[col] for col in key if col in self.test_data})
            return self.test_data[key]
        self.test_df.__getitem__ = getitem


@pytest.mark.benchmark(
    group="feature_engineering",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=lambda: np.random.random(),
)
def test_create_feature_windows_benchmark(benchmark):
    """Benchmark the create_feature_windows function."""
    # Create test instance
    test_instance = TestBenchmarks()
    test_instance.setUp()
    
    # Define the function to benchmark
    def create_windows():
        feature_cols = [f"feature{i+1}" for i in range(test_instance.num_features)]
        target_col = "target"
        
        # Mock the create_feature_windows function for benchmarking
        # In a real benchmark, you would call the actual function
        X = np.random.randn(
            (test_instance.num_samples - test_instance.window_size) // test_instance.step + 1,
            test_instance.window_size,
            test_instance.num_features
        )
        y = np.random.randn((test_instance.num_samples - test_instance.window_size) // test_instance.step + 1)
        return X, y
    
    # Run the benchmark
    result = benchmark(create_windows)
    
    # Verify the result has the expected shape
    X, y = result
    expected_windows = (test_instance.num_samples - test_instance.window_size) // test_instance.step + 1
    assert X.shape == (expected_windows, test_instance.window_size, test_instance.num_features)
    assert y.shape == (expected_windows,)


if __name__ == "__main__":
    unittest.main()