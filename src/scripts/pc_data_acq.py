import cv2
import logging
import threading
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from serial import Serial

# Import pyshimmer
from pyshimmer.bluetooth.bt_api import ShimmerBluetooth
from pyshimmer.dev.channels import ESensorGroup
from pyshimmer.dev.base import DEFAULT_BAUDRATE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global stop flag for threads
stop_threads = threading.Event()

# --- RGB Camera Thread ---
def rgb_camera_thread(camera_idx=0, fps=30, resolution=(640, 480), output_filename="brio_output.avi"):
    """
    Thread function to capture video from the Logitech Brio camera and stream frame markers to LSL.
    
    Args:
        camera_idx (int): Camera device index
        fps (int): Target frames per second
        resolution (tuple): Video resolution as (width, height)
        output_filename (str): Filename for saving the video locally
    """
    logging.info(f"Starting RGB camera thread (index: {camera_idx})...")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
    if not cap.isOpened():
        logging.error(f"ERROR: Could not open Logitech Brio camera at index {camera_idx}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    # LSL Stream for RGB frame markers
    info_rgb = StreamInfo('RGB_Markers', 'Markers', 2, 0, 'double64', 'brio_cam_id')
    outlet_rgb = StreamOutlet(info_rgb)
    logging.info("LSL RGB_Markers stream created.")

    # Video writer for local save
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Or 'mp4v' for .mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps, resolution)

    frame_idx = 0
    while not stop_threads.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab RGB frame.")
            time.sleep(0.1)  # Short delay before retrying
            continue

        # Write frame to local file
        out.write(frame)

        # Get LSL timestamp and push marker
        lsl_timestamp = local_clock()
        outlet_rgb.push_sample([frame_idx, lsl_timestamp])

        frame_idx += 1
        
        # Optional: Display frame for debugging (uncomment for visual feedback)
        # cv2.imshow('Brio RGB', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    logging.info("RGB camera thread stopping.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# --- Shimmer Data Thread ---
def shimmer_data_thread(com_port="COM3", sampling_rate=128):
    """
    Thread function to capture physiological data from the Shimmer3 GSR+ unit and stream to LSL.
    
    Args:
        com_port (str): COM port for the Shimmer device
        sampling_rate (int): Sampling rate in Hz
    """
    logging.info(f"Starting Shimmer data thread (COM: {com_port})...")
    
    # Initialize Shimmer
    try:
        # Open serial connection
        serial_conn = Serial(com_port, DEFAULT_BAUDRATE)
        shimmer = ShimmerBluetooth(serial_conn)
        
        # Initialize and configure the Shimmer
        shimmer.initialize()
        
        # Set sampling rate
        shimmer.set_sampling_rate(sampling_rate)
        
        # Enable GSR, PPG (via INTERNAL_ADC_13), and Accelerometer
        enabled_sensors = [
            ESensorGroup.GSR,           # GSR sensor
            ESensorGroup.CH_A13,        # PPG sensor (connected to INTERNAL_ADC_13)
            ESensorGroup.ACCEL_WR       # Wide-range accelerometer
        ]
        shimmer.set_sensors(enabled_sensors)
        
        # Create LSL Stream for Shimmer data
        # 5 channels: GSR, PPG, AccelX, AccelY, AccelZ
        info_shimmer = StreamInfo('Shimmer_Data', 'BioSemi', 5, sampling_rate, 'float32', 'shimmer_id')
        outlet_shimmer = StreamOutlet(info_shimmer)
        logging.info("LSL Shimmer_Data stream created.")
        
        # Data buffer for processing
        data_buffer = []
        
        # Define callback function to process data packets
        def data_callback(data_packet):
            # Extract data from the packet
            # The order of channels depends on the enabled sensors
            # For our configuration: [GSR, PPG, AccelX, AccelY, AccelZ]
            data_array = data_packet.data
            
            # Process the data (convert units if needed)
            gsr_val = data_array[0]  # GSR raw value
            ppg_val = data_array[1] / 1000.0  # PPG value (convert from mV to V)
            accel_x = data_array[2]  # Accelerometer X
            accel_y = data_array[3]  # Accelerometer Y
            accel_z = data_array[4]  # Accelerometer Z
            
            # Push to LSL
            outlet_shimmer.push_sample([gsr_val, ppg_val, accel_x, accel_y, accel_z])
        
        # Register callback for data packets
        shimmer.add_stream_callback(data_callback)
        
        # Start streaming
        shimmer.start_streaming()
        logging.info("Shimmer streaming started.")
        
        # Keep thread alive until stop signal
        while not stop_threads.is_set():
            time.sleep(0.1)
        
    except Exception as e:
        logging.error(f"Error in Shimmer thread: {e}")
    finally:
        # Clean up
        if 'shimmer' in locals():
            try:
                shimmer.stop_streaming()
                shimmer.shutdown()
                logging.info("Shimmer streaming stopped and connection closed.")
            except Exception as e:
                logging.error(f"Error shutting down Shimmer: {e}")
        
        logging.info("Shimmer data thread stopping.")

# --- Main execution block ---
if __name__ == "__main__":
    logging.info("PC-side data acquisition script started.")
    
    # Parse command line arguments (if needed)
    # For simplicity, we're using default values here
    
    # Start threads
    rgb_t = threading.Thread(target=rgb_camera_thread, args=(0, 30, (1920, 1080), "brio_output.avi"))
    shimmer_t = threading.Thread(target=shimmer_data_thread, args=("COM3", 128))  # Adjust COM port as needed

    rgb_t.start()
    shimmer_t.start()

    # Keep main thread alive until stop signal is received
    # This script will be controlled by the run_experiment.py via subprocess.terminate()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Stopping threads.")
    finally:
        stop_threads.set()  # Signal threads to stop
        rgb_t.join()
        shimmer_t.join()
        logging.info("PC-side data acquisition script finished.")