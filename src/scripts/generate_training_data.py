# src/scripts/generate_training_data.py

"""
Script to generate multiple mock data sessions for ML training.

This script creates a set of mock data sessions with different characteristics,
providing enough variety for the ML models to learn from. It uses the
create_mock_data.py script as a base and extends it to generate data for
multiple subjects with different physiological patterns.
"""

import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- Add project root to path for absolute imports ---
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config
from src.scripts.create_mock_data import (
    generate_mock_physiological_data,
    generate_mock_hand_video,
    generate_mock_thermal_video,
)

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler("data_generation.log"),
        logging.StreamHandler()
    ]
)

# --- Configuration ---
NUM_SUBJECTS = 10  # Number of subjects to generate
SESSIONS_PER_SUBJECT = 2  # Number of sessions per subject
DURATION_RANGE = (30, 120)  # Range of session durations in seconds
HEART_RATE_RANGE = (60, 100)  # Range of heart rates
GSR_RESPONSE_RANGE = (3, 8)  # Range of GSR responses
GSR_DRIFT_RANGE = (0.05, 0.2)  # Range of GSR baseline drift


def generate_subject_session(subject_id, session_num, duration):
    """
    Generate a complete mock data session for a subject.
    
    Args:
        subject_id (str): ID of the subject
        session_num (int): Session number for this subject
        duration (int): Duration of the session in seconds
        
    Returns:
        Path: Path to the generated session directory
    """
    # Generate random physiological parameters
    heart_rate = random.uniform(*HEART_RATE_RANGE)
    gsr_responses = random.randint(*GSR_RESPONSE_RANGE)
    gsr_drift = random.uniform(*GSR_DRIFT_RANGE)
    
    logging.info(f"Generating session for Subject {subject_id}, Session {session_num}")
    logging.info(f"Parameters: duration={duration}s, heart_rate={heart_rate:.1f}bpm, "
                f"gsr_responses={gsr_responses}, gsr_drift={gsr_drift:.3f}")
    
    # Create timestamp for the session
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    session_dir = config.OUTPUT_DIR / f"Subject_{subject_id}_{timestamp}"
    
    # Create output directory
    logging.info(f"Creating session directory: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Generate physiological data with custom parameters
        logging.info("Step 1/3: Generating physiological data...")
        phys_df = generate_mock_physiological_data(
            duration_seconds=duration,
            sampling_rate=config.GSR_SAMPLING_RATE,
            heart_rate=heart_rate,
            scr_number=gsr_responses,
            drift=gsr_drift
        )
        
        if phys_df.empty:
            logging.error("Failed to generate physiological data. Aborting session creation.")
            return None
        
        # Save the physiological data
        csv_path = session_dir / "gsr_data.csv"
        phys_df.to_csv(csv_path, index=False)
        logging.info(f"Saved physiological data to {csv_path}")
        
        # 2. Generate RGB video based on the PPG signal
        logging.info("Step 2/3: Generating RGB video...")
        rgb_path = session_dir / "rgb_video.mp4"
        generate_mock_hand_video(
            rgb_path, 
            phys_df["ppg_value"].values, 
            duration, 
            config.FPS
        )
        
        # 3. Generate thermal video based on the GSR signal
        logging.info("Step 3/3: Generating thermal video...")
        thermal_path = session_dir / "thermal_video.mp4"
        generate_mock_thermal_video(
            thermal_path,
            phys_df["gsr_value"].values,
            duration,
            config.FPS,
        )
        
        logging.info(f"--- Session successfully generated in: {session_dir} ---")
        return session_dir
        
    except Exception as e:
        logging.error(f"Error during session generation: {e}")
        logging.error("Session generation failed.")
        return None


def main():
    """
    Main function to generate multiple mock data sessions for ML training.
    """
    logging.info("=== Starting Mock Training Data Generation ===")
    logging.info(f"Generating data for {NUM_SUBJECTS} subjects, "
                f"{SESSIONS_PER_SUBJECT} sessions each")
    
    # Create the output directory if it doesn't exist
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track successful sessions
    successful_sessions = 0
    
    # Generate data for each subject
    for i in range(1, NUM_SUBJECTS + 1):
        subject_id = f"MockSubject{i:02d}"
        
        # Generate multiple sessions for each subject
        for j in range(1, SESSIONS_PER_SUBJECT + 1):
            # Randomize session duration
            duration = random.randint(*DURATION_RANGE)
            
            # Generate the session
            session_dir = generate_subject_session(subject_id, j, duration)
            
            if session_dir:
                successful_sessions += 1
            
            # Add a small delay to ensure unique timestamps
            time.sleep(1)
    
    logging.info(f"=== Data Generation Complete: {successful_sessions} sessions generated ===")
    logging.info(f"Data is ready for ML training in: {config.OUTPUT_DIR}")
    
    return successful_sessions


if __name__ == "__main__":
    main()