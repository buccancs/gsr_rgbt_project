import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.system.validation.test_synchronization import SynchronizationTester


class TestSynchronizationTester(unittest.TestCase):
    """Test suite for the SynchronizationTester class."""

    @patch('src.system.validation.test_synchronization.TimestampThread')
    @patch('src.system.validation.test_synchronization.VideoCaptureThread')
    @patch('src.system.validation.test_synchronization.ThermalCaptureThread')
    @patch('src.system.validation.test_synchronization.GsrCaptureThread')
    @patch('src.system.validation.test_synchronization.DataLogger')
    @patch('src.system.validation.test_synchronization.tempfile.mkdtemp')
    def setUp(self, mock_mkdtemp, mock_data_logger, mock_gsr, mock_thermal, mock_video, mock_timestamp):
        """Set up test fixtures."""
        # Mock the tempfile.mkdtemp function
        mock_mkdtemp.return_value = "/tmp/test_sync"

        # Create the tester instance
        self.tester = SynchronizationTester(test_duration=1)

        # Store the mocks for later use
        self.mock_timestamp = mock_timestamp
        self.mock_video = mock_video
        self.mock_thermal = mock_thermal
        self.mock_gsr = mock_gsr
        self.mock_data_logger = mock_data_logger

        # Mock the thread instances
        self.mock_timestamp_instance = mock_timestamp.return_value
        self.mock_video_instance = mock_video.return_value
        self.mock_thermal_instance = mock_thermal.return_value
        self.mock_gsr_instance = mock_gsr.return_value
        self.mock_data_logger_instance = mock_data_logger.return_value

        # Mock the session_path property of the data logger
        self.mock_data_logger_instance.session_path = Path("/tmp/test_sync/TestSubject_20230101_120000")

    def test_initialization(self):
        """Test that the tester initializes correctly."""
        self.assertEqual(self.tester.test_duration, 1)
        self.assertIsNone(self.tester.latest_timestamp)
        self.assertEqual(self.tester.temp_dir, Path("/tmp/test_sync"))

        # Check that the threads were created with the correct parameters
        self.mock_timestamp.assert_called_once_with(frequency=200)
        self.mock_video.assert_called_once()
        self.mock_thermal.assert_called_once()
        self.mock_gsr.assert_called_once()
        self.mock_data_logger.assert_called_once()

    def test_update_latest_timestamp(self):
        """Test that update_latest_timestamp updates the latest timestamp."""
        self.tester.update_latest_timestamp(12345)
        self.assertEqual(self.tester.latest_timestamp, 12345)

    def test_collect_rgb_frame(self):
        """Test that collect_rgb_frame collects RGB frames and timestamps."""
        frame = MagicMock()
        self.tester.collect_rgb_frame(frame, 12345)
        self.assertEqual(self.tester.rgb_frames, [frame])
        self.assertEqual(self.tester.rgb_timestamps, [12345])

    def test_collect_thermal_frame(self):
        """Test that collect_thermal_frame collects thermal frames and timestamps."""
        frame = MagicMock()
        self.tester.collect_thermal_frame(frame, 12345)
        self.assertEqual(self.tester.thermal_frames, [frame])
        self.assertEqual(self.tester.thermal_timestamps, [12345])

    def test_collect_gsr_data(self):
        """Test that collect_gsr_data collects GSR values and timestamps."""
        self.tester.collect_gsr_data(0.5, 12345)
        self.assertEqual(self.tester.gsr_values, [0.5])
        self.assertEqual(self.tester.gsr_timestamps, [12345])

    @patch('src.system.validation.test_synchronization.time.sleep')
    def test_run_test(self, mock_sleep):
        """Test that run_test runs the synchronization test."""
        # Mock the analyze_synchronization method
        self.tester.analyze_synchronization = MagicMock(return_value=True)

        # Mock the cleanup method
        self.tester.cleanup = MagicMock()

        # Mock the isRunning method for all threads
        type(self.mock_timestamp_instance).isRunning = PropertyMock(return_value=False)
        type(self.mock_video_instance).isRunning = PropertyMock(return_value=False)
        type(self.mock_thermal_instance).isRunning = PropertyMock(return_value=False)
        type(self.mock_gsr_instance).isRunning = PropertyMock(return_value=False)

        # Run the test
        result = self.tester.run_test()

        # Check that the test ran correctly
        self.assertTrue(result)

        # Check that the threads were started
        self.mock_timestamp_instance.start.assert_called_once()
        self.mock_video_instance.start.assert_called_once()
        self.mock_thermal_instance.start.assert_called_once()
        self.mock_gsr_instance.start.assert_called_once()

        # Check that the data logger was started
        self.mock_data_logger_instance.start_logging.assert_called_once()

        # Check that sleep was called with the correct duration
        mock_sleep.assert_called_once_with(1)

        # Check that the threads were stopped
        self.mock_timestamp_instance.stop.assert_called_once()
        self.mock_video_instance.stop.assert_called_once()
        self.mock_thermal_instance.stop.assert_called_once()
        self.mock_gsr_instance.stop.assert_called_once()

        # Check that the data logger was stopped
        self.mock_data_logger_instance.stop_logging.assert_called_once()

        # Check that analyze_synchronization was called
        self.tester.analyze_synchronization.assert_called_once()

        # Check that cleanup was called
        self.tester.cleanup.assert_called_once()

    @patch('src.system.validation.test_synchronization.pd.read_csv')
    def test_analyze_synchronization_success(self, mock_read_csv):
        """Test that analyze_synchronization returns True when synchronization is working properly."""
        # Set up test data
        self.tester.rgb_timestamps = [1000000000, 1033333333, 1066666666, 1100000000]  # 30 fps
        self.tester.thermal_timestamps = [1000000000, 1033333333, 1066666666, 1100000000]  # 30 fps
        self.tester.gsr_timestamps = [1000000000, 1031250000, 1062500000, 1093750000, 1125000000]  # 32 Hz

        # Mock the plot_timestamps method
        self.tester.plot_timestamps = MagicMock()

        # Mock the Path.exists method to return True
        with patch('pathlib.Path.exists', return_value=True):
            # Mock the read_csv function to return DataFrames with the correct length
            mock_read_csv.side_effect = [
                MagicMock(spec=pd.DataFrame, __len__=lambda self: 4),  # RGB timestamps
                MagicMock(spec=pd.DataFrame, __len__=lambda self: 4),  # Thermal timestamps
                MagicMock(spec=pd.DataFrame, __len__=lambda self: 5)  # GSR data
            ]

            # Run the analysis
            result = self.tester.analyze_synchronization()

            # Check that the analysis returned True
            self.assertTrue(result)

            # Check that plot_timestamps was called
            self.tester.plot_timestamps.assert_called_once()

    @patch('src.system.validation.test_synchronization.pd.read_csv')
    def test_analyze_synchronization_no_data(self, mock_read_csv):
        """Test that analyze_synchronization returns False when no data is collected."""
        # Set up test data (empty)
        self.tester.rgb_timestamps = []
        self.tester.thermal_timestamps = []
        self.tester.gsr_timestamps = []

        # Mock the Path.exists method to return False
        with patch('pathlib.Path.exists', return_value=False):
            # Run the analysis
            result = self.tester.analyze_synchronization()

            # Check that the analysis returned False
            self.assertFalse(result)

    @patch('src.system.validation.test_synchronization.pd.read_csv')
    def test_analyze_synchronization_from_files(self, mock_read_csv):
        """Test that analyze_synchronization can read data from files when no data is collected directly."""
        # Set up test data (empty)
        self.tester.rgb_timestamps = []
        self.tester.thermal_timestamps = []
        self.tester.gsr_timestamps = []

        # Mock the Path.exists method to return True
        with patch('pathlib.Path.exists', return_value=True):
            # Mock the read_csv function to return DataFrames with data
            rgb_df = pd.DataFrame({'timestamp': [1000000000, 1033333333, 1066666666, 1100000000]})
            thermal_df = pd.DataFrame({'timestamp': [1000000000, 1033333333, 1066666666, 1100000000]})
            gsr_df = pd.DataFrame({'shimmer_timestamp': [1000000000, 1031250000, 1062500000, 1093750000, 1125000000]})

            mock_read_csv.side_effect = [rgb_df, thermal_df, gsr_df]

            # Mock the plot_timestamps method
            self.tester.plot_timestamps = MagicMock()

            # Run the analysis
            result = self.tester.analyze_synchronization()

            # Check that the analysis returned True
            self.assertTrue(result)

            # Check that the data was loaded from the files
            self.assertEqual(self.tester.rgb_timestamps, [1000000000, 1033333333, 1066666666, 1100000000])
            self.assertEqual(self.tester.thermal_timestamps, [1000000000, 1033333333, 1066666666, 1100000000])
            self.assertEqual(self.tester.gsr_timestamps,
                             [1000000000.0, 1031250000.0, 1062500000.0, 1093750000.0, 1125000000.0])

            # Check that plot_timestamps was called
            self.tester.plot_timestamps.assert_called_once()

    @patch('src.system.validation.test_synchronization.plt')
    def test_plot_timestamps(self, mock_plt):
        """Test that plot_timestamps creates a visualization of the timestamps."""
        # Set up test data
        rgb_times = [0.0, 0.033, 0.067, 0.1]
        thermal_times = [0.0, 0.033, 0.067, 0.1]
        gsr_times = [0.0, 0.031, 0.062, 0.094, 0.125]

        # Mock the figure and savefig methods
        mock_figure = MagicMock()
        mock_plt.figure.return_value = mock_figure

        # Call the method
        self.tester.plot_timestamps(rgb_times, thermal_times, gsr_times)

        # Check that the figure was created
        mock_plt.figure.assert_called_once()

        # Check that the plot was saved
        mock_plt.savefig.assert_called_once()

    def test_cleanup(self):
        """Test that cleanup logs the session path."""
        # Mock the logging.info method
        with patch('src.system.validation.test_synchronization.logging.info') as mock_info:
            # Call the method
            self.tester.cleanup()

            # Check that the session path was logged
            mock_info.assert_called_with(f"Test data saved in: {self.tester.data_logger.session_path}")


if __name__ == "__main__":
    unittest.main()
