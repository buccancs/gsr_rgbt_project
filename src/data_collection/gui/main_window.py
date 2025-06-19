# src/data_collection/gui/main_window.py

import logging

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QGroupBox,
    QSplitter,
)

# --- Import configuration ---
from src import config
from src.data_collection.gui.real_time_visualization import RealTimeVisualizer

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class MainWindow(QMainWindow):
    """
    The main graphical user interface for the data collection application.

    This class sets up all the visual components, including video display areas,
    control buttons, and input fields. It provides methods to update the UI,
    such as displaying new video frames, but does not contain the core
    application logic itself (which resides in the main Application class).
    """

    def __init__(self, parent=None):
        """
        Initializes the main window and all its widgets.
        """
        super().__init__(parent)
        self.setWindowTitle(config.APP_NAME)
        self.setGeometry(*config.GEOMETRY)

        # Initialize the real-time visualizer
        self.visualizer = RealTimeVisualizer(self)

        self._setup_ui()

    def _setup_ui(self):
        """
        Creates and arranges all the widgets in the main window.
        """
        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Create a splitter for video and GSR visualization ---
        splitter = QSplitter(Qt.Vertical)

        # --- Video Display Layout ---
        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # Create video feed group boxes
        rgb_group = QGroupBox("RGB Video Feed")
        rgb_layout = QVBoxLayout(rgb_group)
        self.rgb_video_label = self._create_video_label("")
        rgb_layout.addWidget(self.rgb_video_label)

        thermal_group = QGroupBox("Thermal Video Feed")
        thermal_layout = QVBoxLayout(thermal_group)
        self.thermal_video_label = self._create_video_label("")
        thermal_layout.addWidget(self.thermal_video_label)

        video_layout.addWidget(rgb_group)
        video_layout.addWidget(thermal_group)

        # --- GSR Visualization Layout ---
        gsr_widget = QWidget()
        gsr_layout = QVBoxLayout(gsr_widget)
        gsr_layout.setContentsMargins(0, 0, 0, 0)

        # Create GSR visualization group box
        gsr_group = QGroupBox("Real-time GSR Signal")
        gsr_group_layout = QVBoxLayout(gsr_group)

        # Create and add the GSR plot canvas
        self.gsr_canvas = self.visualizer.create_gsr_canvas(buffer_size=1000)
        gsr_group_layout.addWidget(self.gsr_canvas)

        gsr_layout.addWidget(gsr_group)

        # Add widgets to splitter
        splitter.addWidget(video_widget)
        splitter.addWidget(gsr_widget)

        # Set initial sizes for the splitter
        splitter.setSizes([500, 300])

        # --- Controls Layout ---
        controls_layout = QHBoxLayout()
        self.subject_id_input = QLineEdit("Subject_01")
        self.subject_id_input.setPlaceholderText("Enter Subject ID")

        self.start_button = QPushButton("Start Recording")
        self.start_button.setStyleSheet(
            "background-color: #4CAF50; color: white; height: 30px; font-weight: bold;"
        )

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setStyleSheet(
            "background-color: #f44336; color: white; height: 30px; font-weight: bold;"
        )
        self.stop_button.setEnabled(False)

        controls_layout.addWidget(QLabel("<b>Subject ID:</b>"))
        controls_layout.addWidget(self.subject_id_input)
        controls_layout.addStretch()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        # --- Add layouts to main window ---
        main_layout.addWidget(splitter)
        main_layout.addLayout(controls_layout)

    def _create_video_label(self, placeholder_text: str) -> QLabel:
        """
        Helper function to create and style a video display label.

        Args:
            placeholder_text (str): The text to display before video starts.

        Returns:
            QLabel: A configured QLabel widget for video display.
        """
        label = QLabel(placeholder_text)
        label.setFixedSize(config.FRAME_WIDTH, config.FRAME_HEIGHT)
        label.setStyleSheet(
            "border: 1px solid #AAA; background-color: #111; color: #AAA;"
        )
        label.setAlignment(Qt.AlignCenter)
        return label

    def update_video_feed(self, frame: np.ndarray, label: QLabel):
        """
        Updates a QLabel with a new video frame from an OpenCV numpy array.

        This method handles the conversion from the BGR color format used by
        OpenCV to the RGB format used by PyQt, and scales the image to fit
        the label while maintaining the aspect ratio.

        Args:
            frame (np.ndarray): The video frame to display.
            label (QLabel): The QLabel widget to update.
        """
        if frame is None:
            return

        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            # Convert OpenCV's BGR format to RGB for PyQt
            qt_image = QImage(
                frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(qt_image)
            # Scale the pixmap to fit the label, preserving aspect ratio
            scaled_pixmap = pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            logging.error(f"Error updating video feed: {e}")

    def show_error_message(self, title: str, message: str):
        """
        Displays a standardized error message box.

        Args:
            title (str): The title for the message box window.
            message (str): The error message to display.
        """
        QMessageBox.critical(self, title, message, QMessageBox.Ok)

    def connect_gsr_signal(self, gsr_thread):
        """
        Connect the GSR data signal from the GSR capture thread to the visualizer.

        This method should be called when the GSR capture thread is started to
        ensure that the GSR plot is updated in real-time when new data arrives.

        Args:
            gsr_thread: The GSR capture thread that emits the gsr_data_point signal.
        """
        if gsr_thread:
            # Connect the GSR data signal to the visualizer's update method
            gsr_thread.gsr_data_point.connect(self.visualizer.update_gsr_data)
            logging.info("Connected GSR data signal to visualizer")

    def reset_visualization(self):
        """
        Reset the visualization, clearing all data buffers.

        This method should be called when recording is stopped or a new recording
        is started to ensure that the visualization starts fresh.
        """
        self.visualizer.reset()
        logging.info("Reset visualization")
