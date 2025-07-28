#!/bin/bash

echo "Setting up Human Detection with RTSP Camera for Raspberry Pi 4"
echo "=============================================================="

# Update system
echo "[1/9] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Check if FFmpeg is installed
echo "[2/9] Checking for FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is already installed."
else
    echo "FFmpeg is not installed. Installing now..."
    sudo apt-get install -y ffmpeg
fi

# Install system dependencies
echo "[3/9] Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev python3-venv libgl1-mesa-glx

# Install OpenCV system dependencies
echo "[4/9] Installing OpenCV system dependencies..."
sudo apt-get install -y libopencv-dev python3-opencv

# Create virtual environment
echo "[5/9] Creating Python virtual environment..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install Python dependencies
echo "[6/9] Installing Python dependencies in virtual environment..."
pip3 install --upgrade pip
pip3 install wheel setuptools

# Try to install OpenCV in the virtual environment
echo "[7/9] Installing OpenCV in virtual environment..."
pip3 install opencv-python || echo "Using system OpenCV instead"

# Install other requirements
echo "[8/9] Installing other Python dependencies..."
pip3 install numpy
pip3 install ultralytics
pip3 install ffmpeg-python

# Pre-download YOLOv8 model
echo "[9/9] Downloading YOLOv8 model (this may take a few minutes)..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "Failed to download model - will download on first run"

# Make scripts executable
echo "Making scripts executable..."
chmod +x human_detection.py
chmod +x stream_to_vlc.py

# Create activation script
echo "Creating activation script..."
cat > run.sh << 'EOL'
#!/bin/bash
source venv/bin/activate
echo "Virtual environment activated. You can now run:"
echo "  python human_detection.py"
echo "  or"
echo "  python stream_to_vlc.py"
EOL
chmod +x run.sh

# Create a test script to verify OpenCV
echo "Creating test script..."
cat > test_opencv.py << 'EOL'
#!/usr/bin/env python3
try:
    import cv2
    print("OpenCV is installed correctly! Version:", cv2.__version__)
except ImportError:
    print("ERROR: OpenCV (cv2) module not found")
EOL
chmod +x test_opencv.py

echo ""
echo "Setup completed successfully!"
echo ""
echo "To test if OpenCV is installed correctly:"
echo "  ./run.sh"
echo "  python test_opencv.py"
echo ""
echo "To run human detection:"
echo "  ./run.sh"
echo "  python human_detection.py"
echo ""
echo "Or to run human detection and stream to VLC:"
echo "  ./run.sh"
echo "  python stream_to_vlc.py"
echo "  Then open VLC and connect to: udp://127.0.0.1:5000"
echo ""
echo "Don't forget to update the RTSP URL in the scripts with your camera details!"
echo "==============================================================" 