import tempfile
import time
import numpy as np
from pathlib import Path
import json

# Add the src directory to the path so we can import the modules
import sys
sys.path.append('src')

from data_collection.utils.data_logger import DataLogger, get_git_commit_hash

def test_data_logger_implementation():
    """Test the current DataLogger implementation with buffered writing and session metadata."""
    
    print("Testing current DataLogger implementation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test git commit hash function
        print(f"Git commit hash: {get_git_commit_hash()}")
        
        # Create DataLogger instance
        logger = DataLogger(
            output_dir=temp_path,
            subject_id="test_subject",
            fps=30,
            video_fourcc="mp4v"
        )
        
        print(f"Session path created: {logger.session_path}")
        
        # Test start_logging with session metadata
        frame_size_rgb = (640, 480)
        frame_size_thermal = (320, 240)
        
        try:
            logger.start_logging(frame_size_rgb, frame_size_thermal)
            print("✓ Logging started successfully")
            
            # Check if session_info.json was created
            session_info_path = logger.session_path / "session_info.json"
            if session_info_path.exists():
                print("✓ Session metadata file created")
                with open(session_info_path, 'r') as f:
                    session_info = json.load(f)
                    print(f"  - Participant ID: {session_info.get('participant_id')}")
                    print(f"  - Software version: {session_info.get('software_version')}")
                    print(f"  - Config parameters: {len(session_info.get('config_parameters', {}))}")
            else:
                print("✗ Session metadata file not created")
            
            # Test buffered logging
            print("Testing buffered data logging...")
            
            # Create test frames
            rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            thermal_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            
            # Log some test data
            for i in range(5):
                timestamp = time.time()
                logger.log_rgb_frame(rgb_frame, timestamp)
                logger.log_thermal_frame(thermal_frame, timestamp)
                logger.log_gsr_data(1000.0 + i, timestamp)
                time.sleep(0.1)  # Small delay
            
            print("✓ Test data logged to queue")
            
            # Stop logging
            logger.stop_logging()
            print("✓ Logging stopped successfully")
            
            # Check if files were created
            expected_files = [
                "rgb_video.mp4",
                "thermal_video.mp4", 
                "gsr_data.csv",
                "rgb_timestamps.csv",
                "thermal_timestamps.csv",
                "session_info.json"
            ]
            
            for filename in expected_files:
                filepath = logger.session_path / filename
                if filepath.exists():
                    print(f"✓ {filename} created")
                else:
                    print(f"✗ {filename} not found")
            
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"✗ Error during testing: {e}")
            logger.stop_logging()  # Ensure cleanup
            raise

if __name__ == "__main__":
    test_data_logger_implementation()