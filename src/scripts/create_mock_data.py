# src/scripts/create_mock_data.py

import logging

# --- Add project root to path for absolute imports ---
import sys
from pathlib import Path

import cv2
import neurokit2 as nk
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src import config

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def generate_mock_physiological_data(
    duration_seconds: int, sampling_rate: int, heart_rate: float = 75,
    scr_number: int = 5, drift: float = 0.1
) -> pd.DataFrame:
    """
    Generates a realistic, synchronized DataFrame of PPG and GSR signals.

    This function uses NeuroKit2 to simulate realistic physiological signals:
    - Photoplethysmography (PPG): Simulates blood volume changes in tissue
    - Galvanic Skin Response (GSR): Simulates electrodermal activity related to sweating

    The signals are synchronized with the same timestamps to represent data that would
    be collected simultaneously from a subject.

    Args:
        duration_seconds (int): Length of the signals to generate in seconds
        sampling_rate (int): Number of samples per second (Hz)
        heart_rate (float, optional): Heart rate in beats per minute. Defaults to 75.
        scr_number (int, optional): Number of skin conductance responses. Defaults to 5.
        drift (float, optional): Gradual baseline drift for GSR. Defaults to 0.1.

    Returns:
        pd.DataFrame: DataFrame containing 'timestamp', 'ppg_value', and 'gsr_value' columns
                     with synchronized physiological signals

    Raises:
        Exception: If signal generation fails
    """
    logging.info(
        f"Generating {duration_seconds}s of physiological data at {sampling_rate}Hz..."
    )
    logging.info(f"Parameters: heart_rate={heart_rate}bpm, scr_number={scr_number}, drift={drift}")

    try:
        # Generate realistic signals using NeuroKit2
        # PPG: Photoplethysmography signal (blood volume changes)
        ppg = nk.ppg_simulate(
            duration=duration_seconds, 
            sampling_rate=sampling_rate, 
            heart_rate=heart_rate
        )

        # GSR/EDA: Galvanic Skin Response/Electrodermal Activity (sweat gland activity)
        gsr = nk.eda_simulate(
            duration=duration_seconds, 
            sampling_rate=sampling_rate, 
            scr_number=scr_number,
            drift=drift
        )

        # Create timestamps starting from now with proper intervals
        # Convert sample indices to seconds, then to datetime
        timestamps = pd.to_datetime(np.arange(len(ppg)) / sampling_rate, unit="s")

        # Combine all signals into a single DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps, 
            "ppg_value": ppg, 
            "gsr_value": gsr
        })

        logging.info(f"Successfully generated {len(df)} samples of physiological data.")
        return df

    except Exception as e:
        logging.error(f"Error generating physiological data: {e}")
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=["timestamp", "ppg_value", "gsr_value"])


def generate_mock_hand_video(
    video_path: Path, ppg_signal: np.ndarray, duration_seconds: int, fps: int
):
    """
    Generates an RGB video simulating a hand with subtle color changes based on a PPG signal.

    This function creates a synthetic video that mimics photoplethysmography (PPG) effects
    where blood flow causes subtle color changes in skin. The video shows a hand-like shape
    with color pulsing synchronized to the provided PPG signal.

    Args:
        video_path (Path): Path where the output video will be saved
        ppg_signal (np.ndarray): 1D array containing the PPG signal values
        duration_seconds (int): Duration of the video in seconds
        fps (int): Frames per second for the output video

    Returns:
        None: The video is saved to the specified path
    """
    logging.info(f"Generating mock RGB hand video at {video_path}...")

    # Get video dimensions from config
    width, height = config.FRAME_WIDTH, config.FRAME_HEIGHT

    # Setup video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
        out = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

        if not out.isOpened():
            logging.error(f"Failed to open video writer for {video_path}")
            return

        # Create a basic hand-like shape (a beige ellipse)
        hand_color = (180, 200, 220)  # BGR
        center_x, center_y = width // 2, height // 2

        # Normalize PPG to create subtle color pulsing (more efficient to do once)
        ppg_min, ppg_max = np.min(ppg_signal), np.max(ppg_signal)
        ppg_normalized = (ppg_signal - ppg_min) / (ppg_max - ppg_min)

        # Pre-calculate the indices to avoid repeated calculations in the loop
        total_frames = duration_seconds * fps
        indices = np.linspace(0, len(ppg_normalized)-1, total_frames, dtype=int)

        for i in range(total_frames):
            # Create dark background
            frame = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)

            # Modulate the green channel slightly based on the PPG signal
            # Use pre-calculated indices for efficiency
            pulse_intensity = ppg_normalized[indices[i]]

            # Calculate pulsing color - green channel varies with blood flow
            pulsing_color = (
                hand_color[0],
                int(hand_color[1] + (pulse_intensity - 0.5) * 10),  # Pulsing green channel
                hand_color[2],
            )

            # Draw hand-like ellipse with the pulsing color
            cv2.ellipse(
                frame,
                (center_x, center_y),
                (width // 4, height // 3),
                0,
                0,
                360,
                pulsing_color,
                -1,
            )

            # Write frame to video
            out.write(frame)

        logging.info("RGB video generation complete.")
    except Exception as e:
        logging.error(f"Error generating RGB video: {e}")
    finally:
        # Ensure resources are released even if an error occurs
        if 'out' in locals() and out is not None:
            out.release()


def generate_mock_thermal_video(
    video_path: Path, gsr_signal: np.ndarray, duration_seconds: int, fps: int
):
    """
    Generates a thermal video simulating temperature changes based on a GSR signal.

    This function creates a synthetic thermal video that simulates how skin temperature
    changes with galvanic skin response (GSR). Higher GSR values (indicating increased
    sweating) result in cooler skin temperatures in the thermal imagery.

    Args:
        video_path (Path): Path where the output thermal video will be saved
        gsr_signal (np.ndarray): 1D array containing the GSR signal values
        duration_seconds (int): Duration of the video in seconds
        fps (int): Frames per second for the output video

    Returns:
        None: The video is saved to the specified path
    """
    logging.info(f"Generating mock thermal video at {video_path}...")

    # Get video dimensions from config
    width, height = config.FRAME_WIDTH, config.FRAME_HEIGHT

    # Setup video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
        out = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

        if not out.isOpened():
            logging.error(f"Failed to open video writer for {video_path}")
            return

        # Normalize GSR to modulate temperature (higher GSR -> slightly cooler temp due to sweat)
        # More efficient to calculate min/max once
        gsr_min, gsr_max = np.min(gsr_signal), np.max(gsr_signal)
        gsr_normalized = (gsr_signal - gsr_min) / (gsr_max - gsr_min)

        # Pre-calculate the indices to avoid repeated calculations in the loop
        total_frames = duration_seconds * fps
        indices = np.linspace(0, len(gsr_normalized)-1, total_frames, dtype=int)

        for i in range(total_frames):
            # Create a grayscale frame to represent thermal data (baseline temperature)
            frame_gray = np.full((height, width), 128, dtype=np.uint8)

            # Modulate temperature based on GSR using pre-calculated indices
            # Invert for cooling effect (higher GSR = more sweating = cooler skin)
            temp_change = (gsr_normalized[indices[i]] - 0.5) * -20
            hand_temp = int(150 + temp_change)  # Base hand temp

            # Draw hand-like ellipse with the temperature value
            cv2.ellipse(
                frame_gray,
                (width // 2, height // 2),
                (width // 4, height // 3),
                0,
                0,
                360,
                hand_temp,
                -1,
            )

            # Apply a colormap to make it look like a thermal video
            # COLORMAP_INFERNO: dark purple/blue (cold) to bright yellow/white (hot)
            frame_thermal = cv2.applyColorMap(frame_gray, cv2.COLORMAP_INFERNO)
            out.write(frame_thermal)

        logging.info("Thermal video generation complete.")
    except Exception as e:
        logging.error(f"Error generating thermal video: {e}")
    finally:
        # Ensure resources are released even if an error occurs
        if 'out' in locals() and out is not None:
            out.release()


def main():
    """
    Main function to create a complete mock data session.

    This function orchestrates the generation of a full mock dataset that includes:
    1. Physiological data (GSR and PPG signals)
    2. RGB video with simulated blood flow changes
    3. Thermal video with simulated temperature changes

    The data is saved in a timestamped directory under the configured output path.
    All generated files follow the same naming convention as real data to ensure
    compatibility with the processing pipeline.

    Returns:
        Path: The path to the generated session directory, or None if generation failed
    """
    logging.info("--- Starting Mock Data Generation ---")

    try:
        # Configuration
        subject_id = "MockSubject01"
        duration = 30  # seconds
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        session_dir = config.OUTPUT_DIR / f"Subject_{subject_id}_{timestamp}"

        # Create output directory
        logging.info(f"Creating session directory: {session_dir}")
        session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate physiological data
        logging.info("Step 1/3: Generating physiological data...")
        phys_df = generate_mock_physiological_data(duration, config.GSR_SAMPLING_RATE)

        if phys_df.empty:
            logging.error("Failed to generate physiological data. Aborting mock data creation.")
            return None

        # Save the physiological data
        csv_path = session_dir / "gsr_data.csv"
        phys_df.to_csv(csv_path, index=False)
        logging.info(f"Saved physiological data to {csv_path}")
        # Note: Filename implies only GSR, but contains both GSR and PPG for convenience

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

        logging.info(f"--- Mock data successfully generated in: {session_dir} ---")
        return session_dir

    except Exception as e:
        logging.error(f"Error during mock data generation: {e}")
        logging.error("Mock data generation failed.")
        return None


if __name__ == "__main__":
    main()
