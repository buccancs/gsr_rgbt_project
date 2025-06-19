# Data Synchronization in the GSR-RGBT Project

This document explains the data synchronization approach used in the GSR-RGBT project and how to test it.

## Synchronization Approach

The GSR-RGBT project uses a custom synchronization approach based on a centralized timestamp authority. This approach ensures that all data streams (RGB video, thermal video, and GSR data) are properly synchronized, which is critical for accurate analysis.

### Components

1. **TimestampThread**: A high-priority thread that emits timestamps at a fast, consistent rate (200Hz by default). This thread serves as a centralized timestamp authority for all data capture components.

2. **Capture Threads**: Separate threads for RGB video, thermal video, and GSR data capture. Each thread captures data at its own rate and associates each data point with the latest timestamp from the TimestampThread.

3. **DataLogger**: Writes the captured data to disk, along with the associated timestamps. This allows for precise synchronization during later analysis.

### How It Works

1. The TimestampThread generates high-resolution timestamps at 200Hz (much faster than any capture rate).
2. When a frame or data point is captured, it's associated with the latest timestamp from the TimestampThread.
3. The DataLogger writes the frames/data points to files along with their timestamps.
4. During analysis, the timestamps can be used to align the different data streams.

This approach ensures that all data streams share a common time reference, even if they are captured at different rates or with different latencies.

## Testing Synchronization

The project includes a test script that verifies the data synchronization mechanism is working properly.

### Running the Test

To run the synchronization test:

```bash
python src/scripts/test_synchronization.py
```

Alternatively, if you have Make installed, you can use:

```bash
make test_sync
```

This script:

1. First runs the system validation check to ensure all devices are working properly.
2. Initializes the TimestampThread, capture threads, and DataLogger.
3. Captures data for a short duration (5 seconds by default).
4. Analyzes the collected data to verify synchronization.
5. Creates a visualization plot of the timestamps.
6. Reports success/failure with detailed information.

### Test Results

The test will output a summary of the results, including:

- Whether data was received from all sources
- The calculated frame rates for each data stream
- Whether the frame rates match the configured values
- Whether the timestamp files were created correctly

It also generates a visualization plot of the timestamps, which can be found in the test output directory.

### Interpreting the Results

- **Success**: All data streams are properly synchronized, with frame rates matching the configured values.
- **Failure**: One or more issues were detected, such as missing data, incorrect frame rates, or missing timestamp files.

If the test fails, check the log output for detailed information about the specific issues.

## Comparison with PhysioKit2

The GSR-RGBT project does not currently use PhysioKit2 for data synchronization, although PhysioKit2 is included in the third_party directory for reference. The custom synchronization approach described above provides similar functionality with tight integration into the project's architecture.

If you wish to use PhysioKit2 for data synchronization instead, significant modifications to the codebase would be required. The current approach has been tested and validated for the specific requirements of this project.

## Troubleshooting

If you encounter synchronization issues:

1. **Check device connectivity**: Run `python src/scripts/check_system.py` or `make test` to verify that all devices are properly connected and configured.

2. **Check frame rates**: If the frame rates are inconsistent, it may indicate issues with the devices or their configuration. Try adjusting the FPS settings in `src/config.py`.

3. **Check CPU usage**: High CPU usage can affect the timing of the threads. Make sure your system has sufficient resources to run all components simultaneously.

4. **Check for errors in the logs**: The test script outputs detailed logs that can help identify specific issues.

5. **Try simulation mode**: If you're having issues with the physical devices, you can set `THERMAL_SIMULATION_MODE = True` and `GSR_SIMULATION_MODE = True` in `src/config.py` to test the synchronization with simulated data.
