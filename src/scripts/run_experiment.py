import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet, local_clock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Paths should be adjusted based on your system
DEFAULT_CONFIG = {
    "LSL_LABRECORDER_PATH": r"C:\Program Files\LabRecorder\LabRecorder.exe",  # Adjust this path
    "PC_ACQ_SCRIPT_PATH": str(Path(__file__).parent / "pc_data_acq.py"),
    "ANDROID_PACKAGE_NAME": "com.yourcompany.thermalapp",
    "ANDROID_SERVICE_NAME": ".ThermalCaptureService",
    "STIMULI_DIR": str(Path(__file__).parent.parent.parent / "data" / "stimuli"),
    "OUTPUT_DIR": str(Path(__file__).parent.parent.parent / "data" / "recordings"),
    "EXPERIMENT_PROTOCOL": [
        {"condition": "Baseline", "duration": 180, "video": "baseline_video.mp4"},
        {"condition": "Stress_Task", "duration": 300, "video": "stress_video.mp4"},
        {"condition": "Recovery", "duration": 180, "video": "recovery_video.mp4"}
    ]
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the GSR-RGBT experiment protocol")
    
    parser.add_argument("--labrecorder", type=str, 
                        default=DEFAULT_CONFIG["LSL_LABRECORDER_PATH"],
                        help="Path to LabRecorder executable")
    
    parser.add_argument("--pc-acq", type=str, 
                        default=DEFAULT_CONFIG["PC_ACQ_SCRIPT_PATH"],
                        help="Path to PC data acquisition script")
    
    parser.add_argument("--android-pkg", type=str, 
                        default=DEFAULT_CONFIG["ANDROID_PACKAGE_NAME"],
                        help="Android package name for thermal app")
    
    parser.add_argument("--android-svc", type=str, 
                        default=DEFAULT_CONFIG["ANDROID_SERVICE_NAME"],
                        help="Android service name for thermal app")
    
    parser.add_argument("--stimuli-dir", type=str, 
                        default=DEFAULT_CONFIG["STIMULI_DIR"],
                        help="Directory containing stimulus videos")
    
    parser.add_argument("--output-dir", type=str, 
                        default=DEFAULT_CONFIG["OUTPUT_DIR"],
                        help="Directory for saving output data")
    
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run in fullscreen mode")
    
    parser.add_argument("--skip-thermal", action="store_true",
                        help="Skip thermal camera (Android) acquisition")
    
    parser.add_argument("--skip-shimmer", action="store_true",
                        help="Skip Shimmer GSR+ acquisition")
    
    parser.add_argument("--subject-id", type=str, required=True,
                        help="Subject ID for this recording session")
    
    return parser.parse_args()

def send_adb_command(action, package_name, service_name):
    """Send ADB command to Android device."""
    cmd = f"adb shell am startservice -n {package_name}/{service_name} --action {package_name}.{action}"
    logging.info(f"Sending ADB command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logging.info(f"ADB command result: {result.stdout.decode().strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ADB command failed: {e.stderr.decode()}")
        return False

def create_session_directory(base_dir, subject_id):
    """Create a directory for the current recording session."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base_dir) / f"Subject_{subject_id}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def run_experiment(args):
    """Run the experiment protocol."""
    logging.info("Starting experiment orchestration...")
    
    # Create session directory
    session_dir = create_session_directory(args.output_dir, args.subject_id)
    logging.info(f"Created session directory: {session_dir}")
    
    # Initialize processes
    pc_acq_process = None
    labrecorder_process = None
    
    # Setup LSL Marker Stream
    info_markers = StreamInfo('ExperimentMarkers', 'Markers', 1, 0, 'string', 'exp_orchestrator_id')
    outlet_markers = StreamOutlet(info_markers)
    logging.info("LSL ExperimentMarkers stream created.")
    
    try:
        # --- Start the Data Acquisition "Camera Crew" ---
        if not args.skip_shimmer:
            logging.info("Launching PC-side data acquisition (Brio & Shimmer)...")
            pc_acq_process = subprocess.Popen([sys.executable, args.pc_acq])
            time.sleep(5)  # Give it a moment to start
        else:
            logging.info("Skipping Shimmer GSR+ acquisition as requested.")
        
        if not args.skip_thermal:
            logging.info("Sending START command to Android thermal node...")
            if not send_adb_command("START_RECORDING", args.android_pkg, args.android_svc):
                logging.warning("Failed to start Android thermal recording. Continuing anyway...")
            time.sleep(5)  # Give Android app a moment to start
        else:
            logging.info("Skipping thermal camera acquisition as requested.")
        
        logging.info("Launching LSL LabRecorder...")
        # Start LabRecorder - you might need to adjust command line args for auto-start
        labrecorder_process = subprocess.Popen([args.labrecorder])
        logging.info("Please ensure LabRecorder is recording all streams now.")
        time.sleep(10)  # Give user time to start LabRecorder recording
        
        # --- Setup PsychoPy Window for Stimuli ---
        win = visual.Window(
            size=[1920, 1080], 
            fullscr=args.fullscreen, 
            color="black", 
            monitor="testMonitor"
        )
        video_player = visual.MovieStim3(win, noAudio=True)
        text_stim = visual.TextStim(win, text="", height=0.1)
        
        # Send experiment start marker
        outlet_markers.push_sample(["experiment_start"])
        logging.info("LSL Marker sent: experiment_start")
        
        # --- Run the Experiment Protocol ---
        protocol = DEFAULT_CONFIG["EXPERIMENT_PROTOCOL"]
        for trial in protocol:
            logging.info(f"--- Starting Condition: {trial['condition']} ---")
            
            # Load and play video
            video_path = os.path.join(args.stimuli_dir, trial['video'])
            if not os.path.exists(video_path):
                logging.warning(f"Video file not found: {video_path}")
                # Display text instead
                text_stim.text = f"Simulating {trial['condition']} condition\n(video not found)"
                text_stim.draw()
                win.flip()
                video_player = None
            else:
                video_player.loadMovie(video_path)
                video_player.play()
            
            # Send LSL marker for the start of the condition
            outlet_markers.push_sample([f"start_{trial['condition']}"])
            logging.info(f"LSL Marker sent: start_{trial['condition']}")
            
            start_time = core.getTime()  # PsychoPy's high-precision timer
            
            # Main stimulus presentation loop
            while (video_player is None or video_player.status != visual.FINISHED) and \
                  (core.getTime() - start_time) < trial['duration']:
                
                if video_player is not None:
                    video_player.draw()
                else:
                    text_stim.draw()
                
                win.flip()
                
                # Check for escape key to exit early
                if 'escape' in event.getKeys():
                    logging.warning("Experiment manually interrupted.")
                    raise KeyboardInterrupt
            
            # Send LSL marker for the end of the condition
            outlet_markers.push_sample([f"end_{trial['condition']}"])
            logging.info(f"LSL Marker sent: end_{trial['condition']}")
            
            # Inter-trial interval
            win.flip()  # Clear the screen
            text_stim.text = "Please wait for the next condition..."
            text_stim.draw()
            win.flip()
            time.sleep(15)
        
        # Send experiment end marker
        outlet_markers.push_sample(["experiment_end"])
        logging.info("LSL Marker sent: experiment_end")
        
        # --- End the Experiment ---
        text_stim.text = "The experiment is now complete. Thank you."
        text_stim.draw()
        win.flip()
        time.sleep(5)  # Display final message
        
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Clean up PsychoPy
        if 'win' in locals():
            win.close()
        core.quit()
        
        logging.info("Stopping all acquisition processes...")
        
        # Stop Android app
        if not args.skip_thermal:
            send_adb_command("STOP_RECORDING", args.android_pkg, args.android_svc)
        
        # Stop PC-side acquisition
        if pc_acq_process and pc_acq_process.poll() is None:
            logging.info("Terminating PC data acquisition process...")
            try:
                pc_acq_process.terminate()  # Send SIGTERM
                time.sleep(2)
                if pc_acq_process.poll() is None:  # If still running, force kill
                    pc_acq_process.kill()
            except Exception as e:
                logging.error(f"Error terminating PC acquisition: {e}")
        
        # Remind to stop LabRecorder
        if labrecorder_process and labrecorder_process.poll() is None:
            logging.info("Please stop LSL LabRecorder manually if it's still running.")
            # For Windows: subprocess.run(['taskkill', '/F', '/IM', 'LabRecorder.exe'])
        
        logging.info("Experiment orchestration finished.")

if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)