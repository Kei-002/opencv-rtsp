# Human Detection with RTSP Camera

This project uses Ultralytics YOLOv8 to detect humans from RTSP camera feeds on a Raspberry Pi 4 Model B.

## Prerequisites

- Raspberry Pi 4 Model B
- RTSP camera(s) - supports both single and dual camera setups
- Python 3.7+
- FFmpeg (required for VLC streaming option)

## Setup with Virtual Environment (Recommended)

The easiest way to set up everything is using the provided setup script:

1. Make the setup script executable:
```
chmod +x setup.sh
```

2. Run the setup script:
```
./setup.sh
```

This script will:
- Update your system
- Install required system dependencies
- Create a Python virtual environment
- Install all required Python packages from requirements.txt
- Download the YOLOv8 model
- Create a run.sh script to activate the environment

3. After setup completes, activate the virtual environment:
```
./run.sh
```

## Manual Setup (Alternative)

If you prefer to set up manually:

1. Update your Raspberry Pi:
```
sudo apt-get update
sudo apt-get upgrade
```

2. Install required system dependencies:
```
sudo apt-get install -y python3-pip python3-dev python3-venv libopencv-dev ffmpeg libgl1-mesa-glx
```

3. Create and activate a virtual environment:
```
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

4. Install the required Python dependencies:
```
pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt
```

5. Download the YOLOv8 model:
```
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Configuration

Edit the script files to update the RTSP URLs with your camera details:

For single camera:
```python
rtsp_url = "rtsp://username:password@camera_ip:port/stream"
```

For dual cameras:
```python
rtsp_url1 = "rtsp://username:password@camera1_ip:port/stream"
rtsp_url2 = "rtsp://username:password@camera2_ip:port/stream"
```

## Usage Options

### Single Camera

#### Option 1: Direct Display (OpenCV Window)

1. Activate the virtual environment (if not already activated):
```
./run.sh
```

2. Run the script:
```
python human_detection.py
```

- Press 'q' to quit the application
- The script will display the camera feed with bounding boxes around detected humans
- FPS is displayed in the top-left corner

#### Option 2: Stream to VLC

1. Activate the virtual environment (if not already activated):
```
./run.sh
```

2. Run the streaming script:
```
python stream_to_vlc.py
```

3. Open VLC media player:
```
vlc
```
   - Go to Media > Open Network Stream
   - Enter `udp://127.0.0.1:5000` as the network URL
   - Click Play

4. To stop the stream, press Ctrl+C in the terminal where the script is running

### Dual Cameras

#### Option 1: Direct Display (OpenCV Window)

1. Activate the virtual environment:
```
./run.sh
```

2. Run the dual camera script:
```
python dual_camera_detection.py
```

- Press 'q' to quit the application
- The script will display both camera feeds side by side with human detection
- FPS is displayed for each camera

#### Option 2: Stream to VLC

1. Activate the virtual environment:
```
./run.sh
```

2. Run the dual camera streaming script:
```
python dual_camera_stream_vlc.py
```

3. Open VLC media player and connect to `udp://127.0.0.1:5000`

## Performance Notes

- The scripts process every 3rd frame to improve performance on the Raspberry Pi
- Using YOLOv8n (nano) model for better performance
- Resolution is set to 640x480 for better performance
- Only detections with confidence > 0.5 are displayed
- The dual camera setup uses threading to process both cameras efficiently

## Troubleshooting

If you encounter issues:

1. Verify your RTSP URL is correct
2. Ensure your camera is accessible on the network
3. Check that all dependencies are installed correctly
4. If you see "ImportError: No module named cv2" or other module errors:
   - Make sure you've activated the virtual environment with `./run.sh`
   - Try reinstalling the package in the virtual environment
5. If you see "ImportError: libGL.so.1: cannot open shared object file":
   ```
   sudo apt-get install libgl1-mesa-glx
   ```
6. For FFmpeg issues:
   ```
   sudo apt-get install --reinstall ffmpeg
   ``` 