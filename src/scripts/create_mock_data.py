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
    duration_seconds: int, sampling_rate: int
) -> pd.DataFrame:
    """
    Generates a realistic, synchronized DataFrame of PPG and GSR signals.
    """
    logging.info(
        f"Generating {duration_seconds}s of physiological data at {sampling_rate}Hz..."
    )

    # Generate realistic signals using NeuroKit2
    ppg = nk.ppg_simulate(
        duration=duration_seconds, sampling_rate=sampling_rate, heart_rate=75
    )
    gsr = nk.eda_simulate(
        duration=duration_seconds, sampling_rate=sampling_rate, scr_number=5, drift=0.1
    )

    # Create timestamps
    timestamps = pd.to_datetime(np.arange(len(ppg)) / sampling_rate, unit="s")

    df = pd.DataFrame({"timestamp": timestamps, "ppg_value": ppg, "gsr_value": gsr})
    logging.info("Physiological data generation complete.")
    return df


def generate_mock_hand_video(
    video_path: Path, ppg_signal: np.ndarray, duration_seconds: int, fps: int
):
    """
    Generates an RGB video simulating a hand with subtle color changes based on a PPG signal.
    """
    logging.info(f"Generating mock RGB hand video at {video_path}...")
    width, height = config.FRAME_WIDTH, config.FRAME_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
    out = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

    # Create a basic hand-like shape (a beige ellipse)
    hand_color = (180, 200, 220)  # BGR
    center_x, center_y = width // 2, height // 2

    # Normalize PPG to create subtle color pulsing
    ppg_normalized = (ppg_signal - np.min(ppg_signal)) / (
        np.max(ppg_signal) - np.min(ppg_signal)
    )

    for i in range(duration_seconds * fps):
        frame = np.full(
            (height, width, 3), (30, 30, 30), dtype=np.uint8
        )  # Dark background

        # Modulate the green channel slightly based on the PPG signal
        pulse_intensity = ppg_normalized[
            int(i * len(ppg_normalized) / (duration_seconds * fps))
        ]
        pulsing_color = (
            hand_color[0],
            int(hand_color[1] + (pulse_intensity - 0.5) * 10),  # Pulsing green channel
            hand_color[2],
        )

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
        out.write(frame)

    out.release()
    logging.info("RGB video generation complete.")


def generate_mock_thermal_video(
    video_path: Path, gsr_signal: np.ndarray, duration_seconds: int, fps: int
):
    """
    Generates a thermal video simulating temperature changes based on a GSR signal.
    """
    logging.info(f"Generating mock thermal video at {video_path}...")
    width, height = config.FRAME_WIDTH, config.FRAME_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
    out = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

    # Normalize GSR to modulate temperature (higher GSR -> slightly cooler temp due to sweat)
    gsr_normalized = (gsr_signal - np.min(gsr_signal)) / (
        np.max(gsr_signal) - np.min(gsr_signal)
    )

    for i in range(duration_seconds * fps):
        # Create a grayscale frame to represent thermal data
        frame_gray = np.full((height, width), 128, dtype=np.uint8)  # Baseline temp

        # Modulate temperature based on GSR
        temp_change = (
            gsr_normalized[int(i * len(gsr_normalized) / (duration_seconds * fps))]
            - 0.5
        ) * -20  # Invert for cooling effect
        hand_temp = int(150 + temp_change)  # Base hand temp

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
        frame_thermal = cv2.applyColorMap(frame_gray, cv2.COLORMAP_INFERNO)
        out.write(frame_thermal)

    out.release()
    logging.info("Thermal video generation complete.")


def main():
    """
    Main function to create a complete mock data session.
    """
    logging.info("--- Starting Mock Data Generation ---")

    # Configuration
    subject_id = "MockSubject01"
    duration = 30  # seconds
    session_dir = (
        config.OUTPUT_DIR
        / f"Subject_{subject_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate physiological data
    phys_df = generate_mock_physiological_data(duration, config.GSR_SAMPLING_RATE)
    phys_df.to_csv(
        session_dir / "gsr_data.csv", index=False
    )  # Note: Name implies only GSR, but contains both

    # 2. Generate videos based on the signals
    generate_mock_hand_video(
        session_dir / "rgb_video.mp4", phys_df["ppg_value"].values, duration, config.FPS
    )
    generate_mock_thermal_video(
        session_dir / "thermal_video.mp4",
        phys_df["gsr_value"].values,
        duration,
        config.FPS,
    )

    logging.info(f"--- Mock data successfully generated in: {session_dir} ---")


if __name__ == "__main__":
    main()
