# Android Thermal Camera App Implementation Guide

This guide provides instructions for implementing the Android app that captures thermal video from the Topdon/InfiRay P2 Pro thermal dongle and streams it to LSL.

## Overview

The Android app will:
1. Connect to the Topdon/InfiRay P2 Pro thermal dongle via USB-C
2. Capture thermal video frames
3. Stream frame data to LSL
4. Save thermal video locally
5. Respond to ADB commands for remote control

## Prerequisites

- Android Studio (latest version)
- Topdon/InfiRay Android SDK (obtain from manufacturer)
- LSL for Android library (liblsl-Java-Release.aar)
- Android device with USB-C port (Samsung Galaxy S21/S22 or similar)
- Topdon/InfiRay P2 Pro thermal dongle

## Project Setup

1. Create a new Android Studio project:
   - Name: ThermalLSLStreamer
   - Package name: com.yourcompany.thermalapp
   - Minimum SDK: API 24 (Android 7.0)
   - Target SDK: Latest stable version
   - Language: Java or Kotlin (Java examples provided below)

2. Add dependencies to your app's build.gradle file:

```gradle
dependencies {
    // Standard Android dependencies
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'com.google.android.material:material:1.10.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    
    // LSL for Android
    implementation files('libs/liblsl-Java-Release.aar')
    
    // Topdon/InfiRay SDK (adjust path as needed)
    implementation files('libs/topdon-thermal-sdk.aar')
    
    // For file operations and permissions
    implementation 'androidx.documentfile:documentfile:1.0.1'
}
```

3. Place the LSL and Topdon/InfiRay SDK files in the app/libs directory.

## Android Manifest Configuration

Add the necessary permissions and service declarations to your AndroidManifest.xml:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.yourcompany.thermalapp">

    <!-- Permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.MANAGE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.USB_PERMISSION" />
    <uses-feature android:name="android.hardware.usb.host" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.ThermalLSLStreamer">

        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
            
            <!-- USB device filter for the thermal dongle -->
            <intent-filter>
                <action android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED" />
            </intent-filter>
            <meta-data
                android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED"
                android:resource="@xml/device_filter" />
        </activity>

        <!-- Service for thermal capture -->
        <service
            android:name=".ThermalCaptureService"
            android:exported="true">
            <intent-filter>
                <action android:name="com.yourcompany.thermalapp.START_RECORDING" />
                <action android:name="com.yourcompany.thermalapp.STOP_RECORDING" />
            </intent-filter>
        </service>
    </application>
</manifest>
```

Create a device filter file at res/xml/device_filter.xml:

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <!-- Add the vendor and product IDs for the Topdon/InfiRay P2 Pro -->
    <!-- Replace these with the actual values from the device -->
    <usb-device vendor-id="XXXX" product-id="YYYY" />
</resources>
```

## Main Activity Implementation

Create MainActivity.java:

```java
package com.yourcompany.thermalapp;

import android.content.Intent;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private TextView statusText;
    private Button startButton;
    private Button stopButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);

        // Set up button click listeners
        startButton.setOnClickListener(v -> startRecording());
        stopButton.setOnClickListener(v -> stopRecording());

        // Check if app was launched by USB device attachment
        Intent intent = getIntent();
        if (intent.getAction().equals(UsbManager.ACTION_USB_DEVICE_ATTACHED)) {
            UsbDevice device = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);
            if (device != null) {
                statusText.setText("Thermal device connected: " + device.getDeviceName());
            }
        }
    }

    private void startRecording() {
        Intent intent = new Intent(this, ThermalCaptureService.class);
        intent.setAction("com.yourcompany.thermalapp.START_RECORDING");
        startService(intent);
        statusText.setText("Recording started");
    }

    private void stopRecording() {
        Intent intent = new Intent(this, ThermalCaptureService.class);
        intent.setAction("com.yourcompany.thermalapp.STOP_RECORDING");
        startService(intent);
        statusText.setText("Recording stopped");
    }
}
```

## Thermal Capture Service Implementation

Create ThermalCaptureService.java:

```java
package com.yourcompany.thermalapp;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.SystemClock;
import android.util.Log;
import androidx.annotation.Nullable;

import edu.ucsd.sccn.LSL;
import edu.ucsd.sccn.LSL.StreamInfo;
import edu.ucsd.sccn.LSL.StreamOutlet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

// Import the Topdon/InfiRay SDK classes
// Note: Replace these with the actual package names from the SDK
import com.topdon.thermal.sdk.ThermalCamera;
import com.topdon.thermal.sdk.ThermalFrame;
import com.topdon.thermal.sdk.ThermalCameraListener;

public class ThermalCaptureService extends Service {
    private static final String TAG = "ThermalCaptureService";
    
    private ThermalCamera thermalCamera;
    private StreamOutlet thermalOutlet;
    private AtomicBoolean isRecording = new AtomicBoolean(false);
    
    private File videoFile;
    private File timestampFile;
    private FileOutputStream videoOutputStream;
    private FileOutputStream timestampOutputStream;
    
    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG, "Service created");
        
        // Initialize the thermal camera
        initializeThermalCamera();
    }
    
    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && intent.getAction() != null) {
            switch (intent.getAction()) {
                case "com.yourcompany.thermalapp.START_RECORDING":
                    startRecording();
                    break;
                case "com.yourcompany.thermalapp.STOP_RECORDING":
                    stopRecording();
                    break;
            }
        }
        return START_STICKY;
    }
    
    private void initializeThermalCamera() {
        try {
            // Initialize the thermal camera using the SDK
            // Note: Replace with actual SDK initialization code
            thermalCamera = new ThermalCamera(this);
            
            thermalCamera.setThermalCameraListener(new ThermalCameraListener() {
                @Override
                public void onFrameReceived(ThermalFrame frame) {
                    if (isRecording.get()) {
                        processFrame(frame);
                    }
                }
                
                @Override
                public void onError(Exception e) {
                    Log.e(TAG, "Thermal camera error", e);
                }
            });
            
            Log.i(TAG, "Thermal camera initialized");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize thermal camera", e);
        }
    }
    
    private void startRecording() {
        if (isRecording.get()) {
            Log.w(TAG, "Already recording");
            return;
        }
        
        try {
            // Create output files
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            File outputDir = new File(getExternalFilesDir(null), "thermal_recordings");
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }
            
            videoFile = new File(outputDir, "thermal_" + timestamp + ".raw");
            timestampFile = new File(outputDir, "thermal_" + timestamp + "_timestamps.csv");
            
            videoOutputStream = new FileOutputStream(videoFile);
            timestampOutputStream = new FileOutputStream(timestampFile);
            
            // Write header to timestamp file
            timestampOutputStream.write("frame_index,system_time_ns,lsl_time\n".getBytes());
            
            // Create LSL outlet for thermal data
            // Adjust channel count and format based on your thermal camera's output
            int width = thermalCamera.getFrameWidth();
            int height = thermalCamera.getFrameHeight();
            StreamInfo info = new StreamInfo(
                    "Thermal_Video",
                    "Video",
                    width * height,  // One value per pixel
                    LSL.IRREGULAR_RATE,
                    LSL.ChannelFormat.float32,
                    "thermal_cam_id");
            thermalOutlet = new StreamOutlet(info);
            
            // Start the camera if not already started
            thermalCamera.start();
            
            isRecording.set(true);
            Log.i(TAG, "Recording started");
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to start recording", e);
            cleanup();
        }
    }
    
    private void processFrame(ThermalFrame frame) {
        try {
            // Get high-precision timestamp
            long systemTimeNs = SystemClock.elapsedRealtimeNanos();
            double lslTime = LSL.local_clock();
            
            // Get frame data
            float[] temperatureData = frame.getTemperatureData();
            int frameIndex = frame.getFrameIndex();
            
            // Write frame data to file
            videoOutputStream.write(convertFloatArrayToBytes(temperatureData));
            
            // Write timestamp to file
            String timestampLine = frameIndex + "," + systemTimeNs + "," + lslTime + "\n";
            timestampOutputStream.write(timestampLine.getBytes());
            
            // Push to LSL
            thermalOutlet.push_sample(temperatureData, lslTime);
            
        } catch (IOException e) {
            Log.e(TAG, "Error processing frame", e);
        }
    }
    
    private byte[] convertFloatArrayToBytes(float[] floatArray) {
        // Convert float array to byte array for storage
        byte[] byteArray = new byte[floatArray.length * 4];
        for (int i = 0; i < floatArray.length; i++) {
            int intBits = Float.floatToIntBits(floatArray[i]);
            byteArray[i * 4] = (byte) (intBits & 0xFF);
            byteArray[i * 4 + 1] = (byte) ((intBits >> 8) & 0xFF);
            byteArray[i * 4 + 2] = (byte) ((intBits >> 16) & 0xFF);
            byteArray[i * 4 + 3] = (byte) ((intBits >> 24) & 0xFF);
        }
        return byteArray;
    }
    
    private void stopRecording() {
        if (!isRecording.get()) {
            Log.w(TAG, "Not recording");
            return;
        }
        
        isRecording.set(false);
        
        // Stop the camera
        thermalCamera.stop();
        
        // Clean up resources
        cleanup();
        
        Log.i(TAG, "Recording stopped");
    }
    
    private void cleanup() {
        // Close file streams
        try {
            if (videoOutputStream != null) {
                videoOutputStream.close();
                videoOutputStream = null;
            }
            
            if (timestampOutputStream != null) {
                timestampOutputStream.close();
                timestampOutputStream = null;
            }
        } catch (IOException e) {
            Log.e(TAG, "Error closing files", e);
        }
        
        // Close LSL outlet
        if (thermalOutlet != null) {
            thermalOutlet.close();
            thermalOutlet = null;
        }
    }
    
    @Override
    public void onDestroy() {
        stopRecording();
        super.onDestroy();
    }
}
```

## Layout Files

Create res/layout/activity_main.xml:

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout 
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <TextView
        android:id="@+id/titleText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Thermal LSL Streamer"
        android:textSize="24sp"
        android:textStyle="bold"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="32dp" />

    <TextView
        android:id="@+id/statusText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Ready"
        android:textSize="18sp"
        app:layout_constraintTop_toBottomOf="@id/titleText"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="32dp" />

    <Button
        android:id="@+id/startButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Start Recording"
        app:layout_constraintTop_toBottomOf="@id/statusText"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="32dp" />

    <Button
        android:id="@+id/stopButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Stop Recording"
        app:layout_constraintTop_toBottomOf="@id/startButton"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="16dp" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

## Testing with ADB Commands

Once the app is installed on the Android device, you can control it using ADB commands:

1. Start recording:
```
adb shell am startservice -n com.yourcompany.thermalapp/.ThermalCaptureService --action com.yourcompany.thermalapp.START_RECORDING
```

2. Stop recording:
```
adb shell am startservice -n com.yourcompany.thermalapp/.ThermalCaptureService --action com.yourcompany.thermalapp.STOP_RECORDING
```

## Important Notes

1. **SDK Integration**: The code above assumes a specific structure for the Topdon/InfiRay SDK. You'll need to adjust the imports and method calls based on the actual SDK documentation.

2. **Permissions**: Modern Android versions require runtime permission requests for storage access. Implement these in the MainActivity.

3. **USB Connection**: The app should handle USB device connection and permission requests. The code above includes basic USB device attachment handling.

4. **Error Handling**: Add more robust error handling for production use.

5. **Power Management**: Consider implementing wake locks to prevent the device from sleeping during recording.

6. **Testing**: Thoroughly test the app with the actual thermal dongle before using it in experiments.

## Troubleshooting

1. **USB Connection Issues**: 
   - Ensure the USB OTG mode is enabled on the Android device
   - Check that the device_filter.xml contains the correct vendor and product IDs

2. **LSL Stream Not Visible**:
   - Ensure the Android device and PC are on the same network
   - Check firewall settings on both devices

3. **Performance Issues**:
   - Reduce the frame rate or resolution if the app becomes unresponsive
   - Ensure the Android device has sufficient storage space

4. **File Storage**:
   - For Android 11+, use the Storage Access Framework or MediaStore API for more reliable file access