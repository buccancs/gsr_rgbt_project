# src/tests/unit/data_collection/gui/test_main_window.py

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt

from src.data_collection.gui.main_window import MainWindow

# Create a QApplication instance for testing Qt widgets
app = QApplication.instance()
if app is None:
    app = QApplication([])

class TestMainWindow(unittest.TestCase):
    """
    Unit tests for the MainWindow class.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a mock for the RealTimeVisualizer
        self.patcher = patch('src.data_collection.gui.main_window.RealTimeVisualizer')
        self.mock_visualizer_class = self.patcher.start()
        self.mock_visualizer = MagicMock()
        self.mock_visualizer_class.return_value = self.mock_visualizer
        
        # Create the MainWindow instance
        self.window = MainWindow()
        
    def tearDown(self):
        """
        Clean up after each test method.
        """
        self.patcher.stop()
        self.window.close()
        
    def test_initialization(self):
        """
        Test that the MainWindow initializes correctly.
        """
        # Check that the window title is set
        self.assertEqual(self.window.windowTitle(), "GSR-RGBT Data Collection")
        
        # Check that the visualizer was created
        self.mock_visualizer_class.assert_called_once()
        
    def test_create_video_label(self):
        """
        Test that _create_video_label creates a QLabel with the correct properties.
        """
        placeholder_text = "Test Video Feed"
        label = self.window._create_video_label(placeholder_text)
        
        # Check that a QLabel was created
        self.assertIsInstance(label, QLabel)
        
        # Check that the label has the correct text
        self.assertEqual(label.text(), placeholder_text)
        
        # Check that the label has the correct alignment
        self.assertEqual(label.alignment(), Qt.AlignCenter)
        
    @patch('src.data_collection.gui.main_window.QImage')
    @patch('src.data_collection.gui.main_window.QPixmap')
    def test_update_video_feed(self, mock_pixmap, mock_qimage):
        """
        Test that update_video_feed correctly converts a frame to a QPixmap and sets it on the label.
        """
        # Create a mock frame and label
        frame = np.ones((480, 640, 3), dtype=np.uint8)  # Create a dummy frame
        label = MagicMock()
        
        # Set up the mocks
        mock_qimage_instance = MagicMock()
        mock_qimage.return_value = mock_qimage_instance
        
        mock_pixmap_instance = MagicMock()
        mock_pixmap.fromImage.return_value = mock_pixmap_instance
        
        # Call the method
        self.window.update_video_feed(frame, label)
        
        # Check that QImage was created with the correct parameters
        mock_qimage.assert_called_with(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            mock_qimage.Format_RGB888
        )
        
        # Check that QPixmap.fromImage was called with the QImage
        mock_pixmap.fromImage.assert_called_with(mock_qimage_instance)
        
        # Check that the label's pixmap was set
        label.setPixmap.assert_called_with(mock_pixmap_instance)
        
    @patch('src.data_collection.gui.main_window.QMessageBox')
    def test_show_error_message(self, mock_message_box):
        """
        Test that show_error_message displays a QMessageBox with the correct title and message.
        """
        # Set up the mock
        mock_message_box_instance = MagicMock()
        mock_message_box.return_value = mock_message_box_instance
        
        # Call the method
        title = "Error Title"
        message = "Error Message"
        self.window.show_error_message(title, message)
        
        # Check that QMessageBox.critical was called with the correct parameters
        mock_message_box.critical.assert_called_with(
            self.window,
            title,
            message,
            mock_message_box.Ok
        )
        
    def test_connect_gsr_signal(self):
        """
        Test that connect_gsr_signal connects the GSR signal to the visualizer.
        """
        # Create a mock GSR thread
        mock_gsr_thread = MagicMock()
        
        # Call the method
        self.window.connect_gsr_signal(mock_gsr_thread)
        
        # Check that the GSR signal was connected to the visualizer
        mock_gsr_thread.gsr_data_point.connect.assert_called_with(
            self.mock_visualizer.add_gsr_data_point
        )
        
    def test_reset_visualization(self):
        """
        Test that reset_visualization calls the visualizer's reset method.
        """
        # Call the method
        self.window.reset_visualization()
        
        # Check that the visualizer's reset method was called
        self.mock_visualizer.reset.assert_called_once()

if __name__ == '__main__':
    unittest.main()