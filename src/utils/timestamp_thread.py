import logging
import time
from PyQt5.QtCore import QThread, pyqtSignal

class TimestampThread(QThread):
    """
    A high-priority thread that emits timestamps at a fast, consistent rate.
    
    This thread serves as a centralized timestamp authority for all data capture
    components, ensuring precise synchronization between different data streams.
    
    The thread emits timestamps at a specified frequency (default: 200Hz),
    which is much faster than any of the capture rates to ensure that every
    captured frame or data point can be associated with a recent timestamp.
    """
    
    # Signal emitted with each new timestamp
    timestamp_generated = pyqtSignal(int)  # Timestamp in nanoseconds
    
    def __init__(self, frequency=200):
        """
        Initialize the timestamp thread.
        
        Args:
            frequency (int): The frequency at which to emit timestamps, in Hz.
                Default is 200Hz, which is much faster than typical capture rates.
        """
        super().__init__()
        self.frequency = frequency
        self.interval = 1.0 / frequency  # Time between timestamps in seconds
        self.running = False
        
        # Set this thread to high priority
        self.setPriority(QThread.HighPriority)
        
        logging.info(f"TimestampThread initialized with frequency {frequency}Hz")
    
    def run(self):
        """
        Main thread loop that emits timestamps at the specified frequency.
        """
        self.running = True
        logging.info("TimestampThread started")
        
        while self.running:
            # Get current high-resolution timestamp
            current_time = time.perf_counter_ns()
            
            # Emit the timestamp
            self.timestamp_generated.emit(current_time)
            
            # Sleep for the interval
            time.sleep(self.interval)
    
    def stop(self):
        """
        Stop the timestamp thread.
        """
        self.running = False
        logging.info("TimestampThread stopped")