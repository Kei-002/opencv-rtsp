# YOLO-based Video Detection System

This repository contains a collection of Python scripts for object detection using YOLO (You Only Look Once) models on various video sources including RTSP streams, video files, and USB cameras.

## Credits

These scripts are based on work by [EdjeElectronics](https://github.com/EdjeElectronics), who has created excellent tutorials and examples for implementing object detection on various platforms. The original concepts have been adapted and extended for this project.

## Features

- **General Object Detection** (`test_yolo.py`): Detect and display all objects from COCO dataset
- **Human Detection** (`human_detection.py`): Focus specifically on detecting people
- **Dual Camera Processing** (`dual_camera_yolo.py`): Process two video streams simultaneously
- **Support for Multiple Sources**:
  - RTSP streams
  - Video files
  - USB cameras
  - Picamera (Raspberry Pi)
- **Recording Capability**: Save detection results as video files
- **Flexible Configuration**: Adjust resolution, confidence thresholds, and more

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- Other dependencies (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Kei-002/yolo-detection.git
   cd yolo-detection
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the setup script (optional):
   ```
   bash setup.sh
   ```

## Usage

### General Object Detection

```bash
# Basic usage with default RTSP stream
python test_yolo.py

# Specify a different source
python test_yolo.py --source your_video.mp4

# Use USB camera
python test_yolo.py --source usb0

# Custom model and resolution
python test_yolo.py --model yolov8m.pt --resolution 1280x720
```

### Human-Only Detection

```bash
# Basic usage with default RTSP stream
python human_detection.py

# Use a USB camera (camera index 0)
python human_detection.py --source 0

# Record detection results
python human_detection.py --source rtsp://your_camera_url --record
```

### Dual Camera Detection

```bash
# Basic usage with default RTSP streams
python dual_camera_yolo.py

# Specify custom sources
python dual_camera_yolo.py --source1 rtsp://camera1_url --source2 rtsp://camera2_url

# Use USB cameras
python dual_camera_yolo.py --source1 0 --source2 1
```

## Setting up RTSP Streams for Testing

### Using VLC for Local Testing

You can create a local RTSP stream using VLC for testing:

1. Open VLC
2. Go to Media > Stream
3. Add your video file
4. Click "Stream"
5. Choose RTSP as the protocol
6. Set path as "/video"
7. Set port to 8554
8. Start streaming

Your stream will be available at `rtsp://127.0.0.1:8554/video`

## Key Controls

- Press `q` to quit
- Press `s` to pause/resume (in single camera modes)
- Press `p` to save a screenshot

## Customization

- Models: The scripts default to YOLOv8n but can use any Ultralytics-compatible model
- Resolution: Adjust with `--resolution WIDTHxHEIGHT` parameter
- Confidence threshold: Set with `--thresh VALUE` (0.0-1.0)

## Troubleshooting

- **Stream Connection Issues**: Check your network connection and verify the RTSP URL
- **Performance Problems**: Try a smaller model (yolov8n) or reduce resolution
- **CUDA Errors**: Ensure your GPU drivers are up to date

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
