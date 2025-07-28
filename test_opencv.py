#!/usr/bin/env python3
try:
    import cv2
    print("OpenCV is installed correctly! Version:", cv2.__version__)
except ImportError:
    print("ERROR: OpenCV (cv2) module not found")
