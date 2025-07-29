#!/bin/bash

echo "==== YOLO Video Detection System Setup ===="
echo "This script will set up the environment for YOLO-based video detection"
echo

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    echo "Detected OS: $OS"
else
    OS="Unknown"
    echo "Could not detect OS, assuming Linux-based system"
fi

# Check if we're on a Raspberry Pi
IS_RASPBERRY_PI=false
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    IS_RASPBERRY_PI=true
    echo "Detected Raspberry Pi hardware"
fi

# Install system dependencies
echo
echo "Installing system dependencies..."
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]] || [[ "$OS" == *"Raspbian"* ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev python3-venv
    sudo apt-get install -y libopencv-dev ffmpeg libgl1-mesa-glx
    
    # Additional packages for Raspberry Pi
    if [ "$IS_RASPBERRY_PI" = true ]; then
        echo "Installing Raspberry Pi specific packages..."
        sudo apt-get install -y python3-picamera2
    fi
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    sudo dnf update -y
    sudo dnf install -y python3-pip python3-devel python3-virtualenv
    sudo dnf install -y opencv-devel ffmpeg mesa-libGL
else
    echo "Unsupported OS for automatic dependency installation"
    echo "Please install the following packages manually:"
    echo "- Python 3.8+ with pip and venv"
    echo "- OpenCV development libraries"
    echo "- FFmpeg"
    echo "- Mesa GL libraries"
fi

# Create virtual environment
echo
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
echo
echo "Upgrading pip and installing wheel..."
pip install --upgrade pip
pip install wheel setuptools

# Install Python dependencies
echo
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Download YOLOv8 model
echo
echo "Downloading YOLOv8n model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create run script
echo
echo "Creating run.sh script for activating the environment..."
cat > run.sh << 'EOL'
#!/bin/bash
source venv/bin/activate
echo "Virtual environment activated. You can now run the detection scripts:"
echo "  python test_yolo.py           # General object detection"
echo "  python human_detection.py     # Human-only detection"
echo "  python dual_camera_yolo.py    # Dual camera detection"
echo
echo "Press Ctrl+D or type 'deactivate' to exit the virtual environment"
exec "${SHELL:-bash}"
EOL

chmod +x run.sh

echo
echo "Setup complete!"
echo
echo "To activate the environment and run the scripts:"
echo "  ./run.sh"
echo
echo "Example commands:"
echo "  python test_yolo.py --source rtsp://your_camera_url"
echo "  python human_detection.py --source 0 --resolution 1280x720"
echo "  python dual_camera_yolo.py --source1 rtsp://camera1_url --source2 rtsp://camera2_url"
echo 