# src/tests/unit/evaluation/test_real_time_visualization.py

import unittest
from unittest.mock import MagicMock, patch
import time
from collections import deque

import numpy as np
import matplotlib
from PyQt5.QtCore import QObject

from src.evaluation.real_time_visualization import RealTimeVisualizer, GSRPlotCanvas

# Use a non-interactive backend for testing
matplotlib.use('Agg')

class TestGSRPlotCanvas(unittest.TestCase):
    """
    Unit tests for the GSRPlotCanvas class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a GSRPlotCanvas instance
        self.buffer_size = 100
        self.canvas = GSRPlotCanvas(buffer_size=self.buffer_size)
        
    def test_initialization(self):
        """
        Test that the GSRPlotCanvas initializes correctly.
        """
        # Check that the figure and axes were created
        self.assertIsNotNone(self.canvas.fig)
        self.assertIsNotNone(self.canvas.axes)
        
        # Check that the data buffers were initialized correctly
        self.assertEqual(self.canvas.buffer_size, self.buffer_size)
        self.assertIsInstance(self.canvas.timestamps, deque)
        self.assertIsInstance(self.canvas.gsr_values, deque)
        self.assertEqual(self.canvas.timestamps.maxlen, self.buffer_size)
        self.assertEqual(self.canvas.gsr_values.maxlen, self.buffer_size)
        
        # Check that the plot was configured correctly
        self.assertEqual(self.canvas.axes.get_title(), 'Real-time GSR Signal')
        self.assertEqual(self.canvas.axes.get_xlabel(), 'Time (s)')
        self.assertEqual(self.canvas.axes.get_ylabel(), 'GSR (µS)')
        
        # Check that the start time was set
        self.assertIsNotNone(self.canvas.start_time)
        
    @patch('src.evaluation.real_time_visualization.time.time')
    def test_update_plot(self, mock_time):
        """
        Test that update_plot correctly updates the plot with a new data point.
        """
        # Mock the time.time() function to return predictable values
        mock_time.return_value = 100.0
        
        # Set the start time to a known value
        self.canvas.start_time = 90.0
        
        # Call update_plot with a test GSR value
        gsr_value = 0.75
        self.canvas.update_plot(gsr_value)
        
        # Check that the data point was added to the buffers
        self.assertEqual(len(self.canvas.timestamps), 1)
        self.assertEqual(len(self.canvas.gsr_values), 1)
        self.assertEqual(self.canvas.timestamps[0], 10.0)  # 100.0 - 90.0
        self.assertEqual(self.canvas.gsr_values[0], gsr_value)
        
        # Check that the plot data was updated
        x_data, y_data = self.canvas.gsr_line.get_data()
        self.assertEqual(list(x_data), [10.0])
        self.assertEqual(list(y_data), [gsr_value])
        
        # Check that the axis limits were adjusted
        x_min, x_max = self.canvas.axes.get_xlim()
        self.assertAlmostEqual(x_min, 0.0)
        self.assertAlmostEqual(x_max, 11.0)
        
        y_min, y_max = self.canvas.axes.get_ylim()
        self.assertAlmostEqual(y_min, 0.65)
        self.assertAlmostEqual(y_max, 0.85)
        
    @patch('src.evaluation.real_time_visualization.time.time')
    def test_update_plot_multiple_points(self, mock_time):
        """
        Test that update_plot correctly handles multiple data points.
        """
        # Mock the time.time() function to return predictable values
        start_time = 100.0
        self.canvas.start_time = start_time
        
        # Add multiple data points
        gsr_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        times = [start_time + i for i in range(1, len(gsr_values) + 1)]
        
        for i, gsr_value in enumerate(gsr_values):
            mock_time.return_value = times[i]
            self.canvas.update_plot(gsr_value)
        
        # Check that all data points were added to the buffers
        self.assertEqual(len(self.canvas.timestamps), len(gsr_values))
        self.assertEqual(len(self.canvas.gsr_values), len(gsr_values))
        
        # Check that the timestamps are correct (relative to start_time)
        expected_timestamps = [i + 1 for i in range(len(gsr_values))]
        self.assertEqual(list(self.canvas.timestamps), expected_timestamps)
        
        # Check that the GSR values are correct
        self.assertEqual(list(self.canvas.gsr_values), gsr_values)
        
        # Check that the plot data was updated
        x_data, y_data = self.canvas.gsr_line.get_data()
        self.assertEqual(list(x_data), expected_timestamps)
        self.assertEqual(list(y_data), gsr_values)
        
        # Check that the axis limits were adjusted for the latest data
        x_min, x_max = self.canvas.axes.get_xlim()
        self.assertAlmostEqual(x_min, 0.0)  # Should show from 0
        self.assertAlmostEqual(x_max, 6.0)  # Last time + 1
        
        y_min, y_max = self.canvas.axes.get_ylim()
        self.assertAlmostEqual(y_min, 0.4)  # min(gsr_values) - 0.1
        self.assertAlmostEqual(y_max, 1.0)  # max(gsr_values) + 0.1

class TestRealTimeVisualizer(unittest.TestCase):
    """
    Unit tests for the RealTimeVisualizer class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a RealTimeVisualizer instance
        self.visualizer = RealTimeVisualizer()
        
    def test_initialization(self):
        """
        Test that the RealTimeVisualizer initializes correctly.
        """
        # Check that the GSR canvas is initially None
        self.assertIsNone(self.visualizer.gsr_canvas)
        
        # Check that the latest GSR value is initialized to 0.0
        self.assertEqual(self.visualizer.latest_gsr_value, 0.0)
        
    def test_create_gsr_canvas(self):
        """
        Test that create_gsr_canvas creates a GSRPlotCanvas instance.
        """
        # Call create_gsr_canvas
        buffer_size = 200
        canvas = self.visualizer.create_gsr_canvas(buffer_size=buffer_size)
        
        # Check that a GSRPlotCanvas was created and returned
        self.assertIsNotNone(canvas)
        self.assertIsInstance(canvas, GSRPlotCanvas)
        
        # Check that the canvas was stored in the visualizer
        self.assertIs(self.visualizer.gsr_canvas, canvas)
        
        # Check that the buffer size was set correctly
        self.assertEqual(canvas.buffer_size, buffer_size)
        
    def test_update_gsr_data(self):
        """
        Test that update_gsr_data updates the GSR canvas with a new data point.
        """
        # Create a mock GSR canvas
        mock_canvas = MagicMock()
        self.visualizer.gsr_canvas = mock_canvas
        
        # Call update_gsr_data with a test GSR value
        gsr_value = 0.75
        self.visualizer.update_gsr_data(gsr_value)
        
        # Check that the latest GSR value was updated
        self.assertEqual(self.visualizer.latest_gsr_value, gsr_value)
        
        # Check that the canvas's update_plot method was called with the GSR value
        mock_canvas.update_plot.assert_called_with(gsr_value)
        
    def test_update_gsr_data_no_canvas(self):
        """
        Test that update_gsr_data handles the case when no canvas is available.
        """
        # Ensure the GSR canvas is None
        self.visualizer.gsr_canvas = None
        
        # Call update_gsr_data with a test GSR value
        gsr_value = 0.75
        self.visualizer.update_gsr_data(gsr_value)
        
        # Check that the latest GSR value was still updated
        self.assertEqual(self.visualizer.latest_gsr_value, gsr_value)
        
    def test_get_latest_gsr_value(self):
        """
        Test that get_latest_gsr_value returns the latest GSR value.
        """
        # Set the latest GSR value
        gsr_value = 0.75
        self.visualizer.latest_gsr_value = gsr_value
        
        # Call get_latest_gsr_value
        result = self.visualizer.get_latest_gsr_value()
        
        # Check that the correct value was returned
        self.assertEqual(result, gsr_value)
        
    def test_reset(self):
        """
        Test that reset clears the data buffers and resets the start time.
        """
        # Create a mock GSR canvas
        mock_canvas = MagicMock()
        mock_canvas.timestamps = deque([1.0, 2.0, 3.0])
        mock_canvas.gsr_values = deque([0.5, 0.6, 0.7])
        self.visualizer.gsr_canvas = mock_canvas
        
        # Call reset
        with patch('src.evaluation.real_time_visualization.time.time') as mock_time:
            mock_time.return_value = 100.0
            self.visualizer.reset()
        
        # Check that the timestamps and GSR values were cleared
        mock_canvas.timestamps.clear.assert_called_once()
        mock_canvas.gsr_values.clear.assert_called_once()
        
        # Check that the start time was reset
        self.assertEqual(mock_canvas.start_time, 100.0)
        
        # Check that the canvas was redrawn
        mock_canvas.fig.canvas.draw_idle.assert_called_once()
        
    def test_reset_no_canvas(self):
        """
        Test that reset handles the case when no canvas is available.
        """
        # Ensure the GSR canvas is None
        self.visualizer.gsr_canvas = None
        
        # Call reset (should not raise an exception)
        self.visualizer.reset()

if __name__ == '__main__':
    unittest.main()