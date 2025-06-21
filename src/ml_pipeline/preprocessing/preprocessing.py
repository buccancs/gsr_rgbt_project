# src/processing/preprocessing.py

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union

import cv2
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Import from our project ---
from src.ml_pipeline.preprocessing.data_loader import SessionDataLoader

# Try to import Numba optimizations
try:
    from src.ml_pipeline.preprocessing.numba_optimizations import nb_extract_roi_signal
    NUMBA_AVAILABLE = True
    logging.info("Numba optimizations are available and will be used.")
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba optimizations are not available. Using pure Python implementations.")

# Try to import MediaPipe
try:
    import mediapipe as mp
    # Initialize MediaPipe hands module
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
    logging.info("MediaPipe is available and will be used for hand landmark detection.")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe is not available. Using fallback methods for hand landmark detection.")

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def process_gsr_signal(
    gsr_df: pd.DataFrame, sampling_rate: int
) -> Optional[pd.DataFrame]:
    """
    Cleans and processes a raw GSR signal DataFrame.

    This function uses the NeuroKit2 library to perform standard psychophysiological
    signal processing. It cleans the raw signal and decomposes it into its tonic
    and phasic components, which are essential for model training.

    If the DataFrame contains PPG and heart rate data (from Shimmer3 GSR+ device),
    these are also processed and included in the output.

    Args:
        gsr_df (pd.DataFrame): DataFrame containing 'gsr_value' and 'timestamp' columns,
                              and optionally 'ppg_value' and 'hr_value' columns.
        sampling_rate (int): The sampling rate of the GSR signal in Hz.

    Returns:
        pd.DataFrame: The original DataFrame augmented with cleaned, tonic, and
                      phasic GSR signal columns, and processed PPG and HR columns
                      if available. Returns None on error.
    """
    if "gsr_value" not in gsr_df.columns:
        logging.error("GSR DataFrame must contain a 'gsr_value' column.")
        return None

    try:
        # Process the GSR signal using NeuroKit2
        # Note: For Shimmer data, GSR values are in kOhms, but NeuroKit2 can process this
        signals, info = nk.eda_process(gsr_df["gsr_value"], sampling_rate=sampling_rate)

        # Rename columns for clarity and merge back into the original DataFrame
        processed_df = signals.rename(
            columns={
                "EDA_Raw": "GSR_Raw",
                "EDA_Clean": "GSR_Clean",
                "EDA_Tonic": "GSR_Tonic",
                "EDA_Phasic": "GSR_Phasic",
            }
        )

        # Process PPG data if available
        if "ppg_value" in gsr_df.columns:
            try:
                # Process the PPG signal using NeuroKit2
                ppg_signals, ppg_info = nk.ppg_process(
                    gsr_df["ppg_value"], sampling_rate=sampling_rate
                )

                # Rename columns for clarity
                ppg_processed = ppg_signals.rename(
                    columns={
                        "PPG_Raw": "PPG_Raw",
                        "PPG_Clean": "PPG_Clean",
                        "PPG_Rate": "PPG_Rate",
                    }
                )

                # Add PPG columns to the processed DataFrame
                for col in ["PPG_Clean", "PPG_Rate"]:
                    if col in ppg_processed.columns:
                        processed_df[col] = ppg_processed[col]

                logging.info("PPG signal successfully processed.")
            except Exception as e:
                logging.warning(f"Error processing PPG signal: {e}. Continuing without PPG processing.")

        # Include heart rate data if available
        if "hr_value" in gsr_df.columns:
            # The hr_value column contains heart rate in BPM derived from PPG
            # We'll include it directly without additional processing
            processed_df["HR_Value"] = gsr_df["hr_value"]
            logging.info("Heart rate data included in processed DataFrame.")

        # Keep only the essential processed columns and merge with original timestamps
        # First, determine which columns to include from processed_df
        cols_to_include = ["GSR_Clean", "GSR_Tonic", "GSR_Phasic"]
        for col in ["PPG_Clean", "PPG_Rate", "HR_Value"]:
            if col in processed_df.columns:
                cols_to_include.append(col)

        # Merge with original DataFrame
        result_df = pd.concat(
            [
                gsr_df.reset_index(drop=True),
                processed_df[cols_to_include],
            ],
            axis=1,
        )

        logging.info(
            "GSR signal successfully processed into clean, tonic, and phasic components."
        )
        return result_df

    except Exception as e:
        logging.error(
            f"An error occurred during GSR signal processing with NeuroKit2: {e}"
        )
        return None


def detect_hand_landmarks(frame: np.ndarray) -> Optional[List[Dict[str, np.ndarray]]]:
    """
    Detects hand landmarks in a video frame using MediaPipe.

    This function uses the MediaPipe Hands module to detect hand landmarks
    in the input frame. It returns a list of dictionaries, each containing
    the landmarks for one detected hand.

    If MediaPipe is not available, a fallback method is used to create
    dummy landmarks based on simple image processing.

    Args:
        frame (np.ndarray): The input video frame (BGR format).

    Returns:
        Optional[List[Dict[str, np.ndarray]]]: A list of dictionaries, each containing
                                              the landmarks for one detected hand.
                                              Returns None if no hands are found.
    """
    if MEDIAPIPE_AVAILABLE:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # We only need to detect one hand
            min_detection_confidence=0.5
        ) as hands:
            results = hands.process(rgb_frame)

            if not results.multi_hand_landmarks:
                logging.warning("No hands detected in the frame.")
                return None

            # Extract landmarks for each detected hand
            hand_landmarks_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to numpy arrays
                landmarks = {}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[idx] = np.array([landmark.x * frame.shape[1], 
                                              landmark.y * frame.shape[0], 
                                              landmark.z])

                hand_landmarks_list.append(landmarks)

                # Draw landmarks on the frame for visualization (debug only)
                # mp_drawing.draw_landmarks(
                #     frame,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )

            return hand_landmarks_list
    else:
        # Fallback method if MediaPipe is not available
        logging.warning("MediaPipe is not available. Using fallback hand landmark detection.")

        # Create dummy landmarks based on simple image processing
        # This is a very basic approach and won't work well in real scenarios
        h, w, _ = frame.shape

        # Create a dictionary with dummy landmarks
        # The key landmarks for our ROI detection are:
        # 0: Wrist
        # 5: Index finger MCP (base)
        # 9: Middle finger MCP (base)
        # 13: Ring finger MCP (base)
        # 17: Pinky finger MCP (base)
        landmarks = {
            0: np.array([w // 2, h * 0.8, 0]),  # Wrist
            5: np.array([w * 0.4, h * 0.6, 0]),  # Index finger base
            9: np.array([w * 0.5, h * 0.6, 0]),  # Middle finger base
            13: np.array([w * 0.6, h * 0.6, 0]),  # Ring finger base
            17: np.array([w * 0.7, h * 0.6, 0])   # Pinky finger base
        }

        return [landmarks]


def define_multi_roi(frame: np.ndarray, hand_landmarks: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Defines multiple Regions of Interest (ROIs) based on hand landmarks.

    This function defines several ROIs on the palm that are physiologically
    significant for GSR prediction:
    1. Base of the index finger
    2. Base of the ring finger
    3. Center of the palm

    Args:
        frame (np.ndarray): The input video frame.
        hand_landmarks (Dict[str, np.ndarray]): Dictionary of hand landmarks.

    Returns:
        Dict[str, Tuple[int, int, int, int]]: Dictionary mapping ROI names to
                                             bounding boxes (x, y, width, height).
    """
    h, w, _ = frame.shape
    roi_size = int(min(h, w) * 0.1)  # ROI size is 10% of the smaller dimension

    # MediaPipe hand landmark indices
    # 5: Index finger MCP (base)
    # 9: Middle finger MCP (base)
    # 13: Ring finger MCP (base)
    # 17: Pinky finger MCP (base)
    # 0: Wrist

    # Define ROIs
    rois = {}

    # 1. Base of the index finger (landmark 5)
    if 5 in hand_landmarks:
        x, y, _ = hand_landmarks[5]
        x, y = int(x), int(y)
        rois["index_finger_base"] = (
            max(0, x - roi_size // 2),
            max(0, y - roi_size // 2),
            min(roi_size, w - x + roi_size // 2),
            min(roi_size, h - y + roi_size // 2)
        )

    # 2. Base of the ring finger (landmark 13)
    if 13 in hand_landmarks:
        x, y, _ = hand_landmarks[13]
        x, y = int(x), int(y)
        rois["ring_finger_base"] = (
            max(0, x - roi_size // 2),
            max(0, y - roi_size // 2),
            min(roi_size, w - x + roi_size // 2),
            min(roi_size, h - y + roi_size // 2)
        )

    # 3. Center of the palm (average of landmarks 0, 5, 9, 13, 17)
    palm_landmarks = [0, 5, 9, 13, 17]
    if all(idx in hand_landmarks for idx in palm_landmarks):
        x = sum(hand_landmarks[idx][0] for idx in palm_landmarks) / len(palm_landmarks)
        y = sum(hand_landmarks[idx][1] for idx in palm_landmarks) / len(palm_landmarks)
        x, y = int(x), int(y)
        rois["palm_center"] = (
            max(0, x - roi_size // 2),
            max(0, y - roi_size // 2),
            min(roi_size, w - x + roi_size // 2),
            min(roi_size, h - y + roi_size // 2)
        )

    return rois


def detect_palm_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the palm region in a video frame.

    This function is maintained for backward compatibility. It uses the new
    hand landmark detection to find the palm center ROI, or falls back to
    a simple placeholder if no hand is detected.

    Args:
        frame (np.ndarray): The input video frame.

    Returns:
        Optional[Tuple[int, int, int, int]]: A tuple representing the bounding box
                                             (x, y, width, height) of the detected palm.
                                             Returns None if no palm is found.
    """
    # Try to detect hand landmarks
    hand_landmarks_list = detect_hand_landmarks(frame)

    if hand_landmarks_list and len(hand_landmarks_list) > 0:
        # Use the first detected hand
        hand_landmarks = hand_landmarks_list[0]

        # Get the palm center ROI
        rois = define_multi_roi(frame, hand_landmarks)

        if "palm_center" in rois:
            return rois["palm_center"]

    # Fallback to the original placeholder method if hand detection fails
    h, w, _ = frame.shape
    roi_width = int(w * 0.4)
    roi_height = int(h * 0.6)
    roi_x = (w - roi_width) // 2
    roi_y = (h - roi_height) // 2

    logging.warning("Using fallback palm ROI detection method.")
    return (roi_x, roi_y, roi_width, roi_height)


def extract_roi_signal(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extracts the mean pixel value from a specified Region of Interest (ROI).

    Args:
        frame (np.ndarray): The video frame from which to extract the signal.
        roi (Tuple[int, int, int, int]): The bounding box (x, y, w, h) of the ROI.

    Returns:
        np.ndarray: An array containing the mean value for each channel (e.g., [R, G, B]).
    """
    if NUMBA_AVAILABLE:
        # Use the Numba implementation for better performance
        # Ensure frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Call the Numba function
        mean_signal = nb_extract_roi_signal(frame, roi)
        return mean_signal
    else:
        # Use the original OpenCV implementation
        x, y, w, h = roi
        palm_region = frame[y : y + h, x : x + w]

        # Calculate the mean value for each channel within the ROI
        mean_signal = cv2.mean(palm_region)[:3]  # Take only the B, G, R components
        return np.array(mean_signal)


def extract_multi_roi_signals(frame: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
    """
    Extracts mean pixel values from multiple Regions of Interest (ROIs).

    Args:
        frame (np.ndarray): The video frame from which to extract the signals.
        rois (Dict[str, Tuple[int, int, int, int]]): Dictionary mapping ROI names to
                                                    bounding boxes (x, y, w, h).

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping ROI names to arrays containing
                              the mean value for each channel (e.g., [R, G, B]).
    """
    signals = {}
    for roi_name, roi in rois.items():
        signals[roi_name] = extract_roi_signal(frame, roi)
    return signals


def process_frame_with_multi_roi(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Processes a video frame to extract signals from multiple ROIs.

    This function detects hand landmarks in the frame, defines multiple ROIs,
    and extracts signals from each ROI.

    Args:
        frame (np.ndarray): The input video frame.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping ROI names to arrays containing
                              the mean value for each channel (e.g., [R, G, B]).
                              Returns an empty dictionary if no hand is detected.
    """
    # Detect hand landmarks
    hand_landmarks_list = detect_hand_landmarks(frame)

    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        logging.warning("No hands detected in the frame. Cannot extract multi-ROI signals.")
        return {}

    # Use the first detected hand
    hand_landmarks = hand_landmarks_list[0]

    # Define multiple ROIs
    rois = define_multi_roi(frame, hand_landmarks)

    # Extract signals from each ROI
    signals = extract_multi_roi_signals(frame, rois)

    return signals


def visualize_multi_roi(frame: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Visualizes multiple ROIs on a video frame.

    Args:
        frame (np.ndarray): The input video frame.
        rois (Dict[str, Tuple[int, int, int, int]]): Dictionary mapping ROI names to
                                                    bounding boxes (x, y, w, h).

    Returns:
        np.ndarray: The input frame with ROIs visualized.
    """
    # Create a copy of the frame to avoid modifying the original
    vis_frame = frame.copy()

    # Define colors for different ROIs
    colors = {
        "index_finger_base": (0, 255, 0),    # Green
        "ring_finger_base": (0, 0, 255),     # Red
        "palm_center": (255, 0, 0)           # Blue
    }

    # Draw ROIs on the frame
    for roi_name, roi in rois.items():
        x, y, w, h = roi
        color = colors.get(roi_name, (255, 255, 255))  # Default to white
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis_frame, roi_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis_frame


# --- Example Usage ---
# This demonstrates how the functions would be used in a processing script.
if __name__ == "__main__":
    # Assume a dummy session exists from the data_loader example
    dummy_session_path = Path("./dummy_session")
    if not dummy_session_path.exists():
        logging.error(
            "Dummy session directory not found. Please run data_loader.py first to create it."
        )
    else:
        print("--- Testing Preprocessing Functions ---")

        # 1. Load data using the SessionDataLoader
        loader = SessionDataLoader(dummy_session_path)
        gsr_df_raw = loader.get_gsr_data()

        # Add more dummy data for realistic processing
        if gsr_df_raw is not None:
            num_points = 32 * 10  # 10 seconds of data at 32Hz
            dummy_gsr = np.random.randn(num_points) * 0.05 + 0.8
            dummy_timestamps = pd.to_datetime(
                np.arange(num_points) / 32, unit="s", origin="2025-06-19T10:00:00"
            )
            gsr_df_raw = pd.DataFrame(
                {"timestamp": dummy_timestamps, "gsr_value": dummy_gsr}
            )

            # 2. Process the GSR signal
            print("\nProcessing GSR signal...")
            processed_gsr_df = process_gsr_signal(gsr_df_raw, sampling_rate=32)
            if processed_gsr_df is not None:
                print("Processed GSR DataFrame head:")
                print(processed_gsr_df.head())

        # 3. Process a dummy video frame
        print("\nProcessing a dummy video frame...")
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Detect palm ROI (legacy method)
        palm_roi = detect_palm_roi(dummy_frame)
        if palm_roi:
            print(f"Detected palm ROI (legacy method): {palm_roi}")

            # Extract signal from ROI
            roi_signal = extract_roi_signal(dummy_frame, palm_roi)
            print(f"Extracted mean ROI signal (B, G, R): {roi_signal}")

        # 4. Test Multi-ROI processing
        print("\nTesting Multi-ROI processing...")

        # Create a more realistic dummy frame with a hand-like shape
        # (This is just for demonstration - in real use, you'd use actual video frames)
        dummy_hand_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple hand shape
        cv2.circle(dummy_hand_frame, (320, 240), 50, (200, 200, 200), -1)  # Palm
        cv2.rectangle(dummy_hand_frame, (320, 190), (340, 120), (200, 200, 200), -1)  # Index finger
        cv2.rectangle(dummy_hand_frame, (350, 190), (370, 130), (200, 200, 200), -1)  # Middle finger
        cv2.rectangle(dummy_hand_frame, (380, 190), (400, 140), (200, 200, 200), -1)  # Ring finger
        cv2.rectangle(dummy_hand_frame, (410, 190), (430, 150), (200, 200, 200), -1)  # Pinky finger
        cv2.rectangle(dummy_hand_frame, (290, 190), (310, 160), (200, 200, 200), -1)  # Thumb

        # Note: MediaPipe might not detect this artificial hand shape.
        # In real usage, you would use actual video frames with real hands.

        # Process the frame with Multi-ROI
        multi_roi_signals = process_frame_with_multi_roi(dummy_hand_frame)

        if multi_roi_signals:
            print("Extracted Multi-ROI signals:")
            for roi_name, signal in multi_roi_signals.items():
                print(f"  {roi_name}: {signal}")

            # Get the ROIs that were used
            hand_landmarks_list = detect_hand_landmarks(dummy_hand_frame)
            if hand_landmarks_list:
                hand_landmarks = hand_landmarks_list[0]
                rois = define_multi_roi(dummy_hand_frame, hand_landmarks)

                # Visualize the ROIs
                vis_frame = visualize_multi_roi(dummy_hand_frame, rois)

                # Save the visualization (in a real application, you might display it instead)
                cv2.imwrite("multi_roi_visualization.jpg", vis_frame)
                print("Saved Multi-ROI visualization to 'multi_roi_visualization.jpg'")
        else:
            print("MediaPipe could not detect a hand in the dummy frame.")
            print("This is expected for artificial images. In real usage with actual video frames, detection should work.")

            # For demonstration purposes, let's create some dummy ROIs
            dummy_rois = {
                "index_finger_base": (320, 190, 20, 20),
                "ring_finger_base": (380, 190, 20, 20),
                "palm_center": (320, 240, 30, 30)
            }

            # Extract signals from these dummy ROIs
            dummy_signals = extract_multi_roi_signals(dummy_hand_frame, dummy_rois)
            print("\nExtracted signals from dummy ROIs:")
            for roi_name, signal in dummy_signals.items():
                print(f"  {roi_name}: {signal}")

            # Visualize the dummy ROIs
            vis_frame = visualize_multi_roi(dummy_hand_frame, dummy_rois)

            # Save the visualization
            cv2.imwrite("dummy_multi_roi_visualization.jpg", vis_frame)
            print("Saved dummy Multi-ROI visualization to 'dummy_multi_roi_visualization.jpg'")
