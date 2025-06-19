# src/gui/main_window.py

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
)

# --- Import configuration ---
from src import config

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

        self._setup_ui()

    def _setup_ui(self):
        """
        Creates and arranges all the widgets in the main window.
        """
        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Video Display Layout ---
        video_layout = QHBoxLayout()
        self.rgb_video_label = self._create_video_label("RGB Feed")
        self.thermal_video_label = self._create_video_label("Thermal Feed")

        video_layout.addWidget(self.rgb_video_label)
        video_layout.addWidget(self.thermal_video_label)

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
        main_layout.addLayout(video_layout)
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
