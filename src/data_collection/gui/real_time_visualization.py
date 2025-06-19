# src/data_collection/gui/real_time_visualization.py

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class GSRPlotCanvas(FigureCanvasQTAgg):
    """
    A Matplotlib canvas for real-time GSR data visualization.

    This class extends FigureCanvasQTAgg to create a Qt widget that displays
    real-time GSR data using Matplotlib. It maintains a buffer of recent data
    points and updates the plot when new data arrives.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100, buffer_size=500):
        """
        Initialize the GSR plot canvas.

        Args:
            parent: The parent widget.
            width (int): The width of the figure in inches.
            height (int): The height of the figure in inches.
            dpi (int): The resolution of the figure in dots per inch.
            buffer_size (int): The number of data points to keep in the buffer.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(GSRPlotCanvas, self).__init__(self.fig)

        # Initialize data buffers
        self.buffer_size = buffer_size
        self.timestamps: Deque[float] = deque(maxlen=buffer_size)
        self.gsr_values: Deque[float] = deque(maxlen=buffer_size)

        # Initialize the plot
        self.gsr_line, = self.axes.plot([], [], 'b-', label='GSR Signal')

        # Configure the plot
        self.axes.set_title('Real-time GSR Signal')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('GSR (µS)')
        self.axes.grid(True)
        self.axes.legend(loc='upper right')

        # Set initial y-axis limits
        self.axes.set_ylim(0, 1)

        # Enable tight layout
        self.fig.tight_layout()

        # Start time reference
        self.start_time = time.time()

    def update_plot(self, gsr_value: float):
        """
        Update the plot with a new GSR data point.

        Args:
            gsr_value (float): The new GSR value to add to the plot.
        """
        # Add the new data point to the buffers
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        self.gsr_values.append(gsr_value)

        # Update the plot data
        self.gsr_line.set_data(list(self.timestamps), list(self.gsr_values))

        # Adjust the x-axis limits to show the most recent data
        if len(self.timestamps) > 0:
            x_min = max(0, current_time - 30)  # Show the last 30 seconds
            x_max = current_time + 1
            self.axes.set_xlim(x_min, x_max)

        # Adjust the y-axis limits if needed
        if len(self.gsr_values) > 0:
            y_min = max(0, min(self.gsr_values) - 0.1)
            y_max = max(self.gsr_values) + 0.1
            if y_max > y_min:
                self.axes.set_ylim(y_min, y_max)

        # Redraw the canvas
        self.fig.canvas.draw_idle()


class RealTimeVisualizer(QObject):
    """
    A class for real-time visualization of GSR and video data.

    This class manages the real-time visualization of GSR data and provides
    methods for updating the visualization with new data points.
    """

    # Signal to request a plot update (to be connected to the main thread)
    update_requested = pyqtSignal()

    def __init__(self, parent=None):
        """
        Initialize the real-time visualizer.

        Args:
            parent: The parent QObject.
        """
        super().__init__(parent)
        self.gsr_canvas = None
        self.latest_gsr_value = 0.0

    def create_gsr_canvas(self, buffer_size=500) -> GSRPlotCanvas:
        """
        Create a GSR plot canvas for real-time visualization.

        Args:
            buffer_size (int): The number of data points to keep in the buffer.

        Returns:
            GSRPlotCanvas: The created canvas widget.
        """
        self.gsr_canvas = GSRPlotCanvas(buffer_size=buffer_size)
        return self.gsr_canvas

    @pyqtSlot(float)
    def update_gsr_data(self, gsr_value: float):
        """
        Update the GSR visualization with a new data point.

        This method is designed to be connected to the gsr_data_point signal
        from the GsrCaptureThread.

        Args:
            gsr_value (float): The new GSR value to add to the visualization.
        """
        if self.gsr_canvas is not None:
            self.latest_gsr_value = gsr_value
            self.gsr_canvas.update_plot(gsr_value)

    def get_latest_gsr_value(self) -> float:
        """
        Get the latest GSR value.

        Returns:
            float: The latest GSR value.
        """
        return self.latest_gsr_value

    def reset(self):
        """
        Reset the visualization, clearing all data buffers.
        """
        if self.gsr_canvas is not None:
            self.gsr_canvas.timestamps.clear()
            self.gsr_canvas.gsr_values.clear()
            self.gsr_canvas.start_time = time.time()
            self.gsr_canvas.fig.canvas.draw_idle()
