#!/usr/bin/env python3
# src/tests/run_tests.py

import os
import sys
import unittest
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def run_tests(test_type=None, verbose=True):
    """
    Run the specified type of tests or all tests if no type is specified.
    
    Args:
        test_type (str, optional): The type of tests to run ('unit', 'smoke', 'regression', or None for all).
        verbose (bool, optional): Whether to show verbose output.
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    # Determine which test directories to include
    if test_type is None:
        test_dirs = ['unit', 'smoke', 'regression', '.']
    elif test_type in ['unit', 'smoke', 'regression']:
        test_dirs = [test_type]
    else:
        print(f"Unknown test type: {test_type}")
        print("Valid types are: 'unit', 'smoke', 'regression', or None for all tests")
        return False
    
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add tests from the specified directories
    for test_dir in test_dirs:
        test_path = os.path.join(os.path.dirname(__file__), test_dir)
        if os.path.exists(test_path):
            # Discover tests in the directory
            discovered_tests = unittest.defaultTestLoader.discover(
                start_dir=test_path,
                pattern='test_*.py',
                top_level_dir=os.path.dirname(os.path.dirname(__file__))
            )
            test_suite.addTests(discovered_tests)
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    
    # Return True if all tests pass, False otherwise
    return result.wasSuccessful()


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run tests for the GSR-RGBT project.')
    parser.add_argument('--type', choices=['unit', 'smoke', 'regression'], 
                        help='Type of tests to run (unit, smoke, regression, or all if not specified)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()
    
    # Run the tests
    success = run_tests(test_type=args.type, verbose=not args.quiet)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)