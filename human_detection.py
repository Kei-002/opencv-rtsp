#!/usr/bin/env python3

import os
import sys
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

# RTSP URL - replace with your camera's RTSP URL
# Using a local VLC stream for testing
rtsp_url = "rtsp://127.0.0.1:8554/video"
yolo_model = "yolov8n.pt"

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "yolov8n.pt")',
                    default=yolo_model)
parser.add_argument('--source', help='Image source, can be RTSP stream URL ("rtsp://..."), \
                    video file ("testvid.mp4"), or index of USB camera ("0")', 
                    default=rtsp_url)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected people (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default='640x480')
parser.add_argument('--record', help='Record results and save it as "people_detection.avi"',
                    action='store_true')
parser.add_argument('--show-fps', help='Display FPS counter on the video',
                    action='store_true', default=True)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
show_fps = args.show_fps

# Parse resolution if provided
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print(f'Model file {model_path} not found locally. Attempting to download from Ultralytics...')
    try:
        # This will download the model if it doesn't exist locally
        model = YOLO(model_path)
        print(f'Model {model_path} downloaded successfully.')
    except Exception as e:
        print(f'ERROR: Could not download model {model_path}. Error: {e}')
        print('Available models include: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt')
        sys.exit(0)
else:
    # Load the model into memory
    model = YOLO(model_path, task='detect')

# Get the class ID for 'person'
# YOLO models typically have 'person' as class 0, but let's verify
person_class_id = None
for class_id, class_name in model.names.items():
    if class_name.lower() == 'person':
        person_class_id = class_id
        break

if person_class_id is None:
    print("ERROR: This model doesn't have a 'person' class. Please use a COCO-trained model.")
    sys.exit(0)

print(f"Person class ID: {person_class_id}")

# Determine source type
if source.isdigit():
    # If source is a number, treat as camera index
    source_type = 'camera'
    cap = cv2.VideoCapture(int(source))
elif source.startswith('rtsp://'):
    source_type = 'rtsp'
    cap = cv2.VideoCapture(source)
elif os.path.isfile(source):
    source_type = 'video'
    cap = cv2.VideoCapture(source)
else:
    print(f'ERROR: Source {source} is invalid or not found.')
    sys.exit(0)

# Check if camera/video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open source: {source}")
    sys.exit(0)

# Set camera/video resolution if specified
if resize:
    cap.set(3, resW)
    cap.set(4, resH)

# Get actual frame dimensions (might be different from requested)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video dimensions: {actual_width}x{actual_height}")

# Set up recording if enabled
if record:
    record_name = 'people_detection.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (actual_width, actual_height))
    print(f"Recording to {record_name}")

# Person bounding box color (blue)
person_color = (255, 0, 0)  # BGR format

# Initialize control and status variables
fps_buffer = []
fps_avg_len = 30
avg_fps = 0
start_time = time.time()
frames_processed = 0

print("Starting people detection. Press 'q' to quit, 's' to pause, 'p' to save a screenshot.")

# Main detection loop
while True:
    loop_start = time.perf_counter()
    
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        if source_type == 'rtsp':
            print('Failed to read from RTSP stream. Retrying...')
            time.sleep(1)
            continue
        else:
            print('Reached end of video or failed to capture frame. Exiting.')
            break
    
    # Resize frame if needed
    if resize:
        frame = cv2.resize(frame, (resW, resH))
    
    # Run inference on frame
    results = model(frame, verbose=False)
    
    # Extract results
    detections = results[0].boxes
    
    # Count people
    person_count = 0
    
    # Process each detection
    for i in range(len(detections)):
        # Get class ID
        class_id = int(detections[i].cls.item())
        
        # Only process if it's a person
        if class_id == person_class_id:
            # Get confidence
            conf = detections[i].conf.item()
            
            # Check confidence threshold
            if conf > min_thresh:
                # Get bounding box coordinates
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), person_color, 2)
                
                # Add label with confidence
                label = f'Person: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), 
                             person_color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Increment person count
                person_count += 1
    
    # Calculate and display FPS
    frames_processed += 1
    elapsed_time = time.time() - start_time
    current_fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
    
    # Update FPS buffer for smoother display
    fps_buffer.append(current_fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer) / len(fps_buffer)
    
    # Add info text to frame
    if show_fps:
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display person count
    cv2.putText(frame, f'People detected: {person_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('People Detection', frame)
    
    # Record if enabled
    if record:
        recorder.write(frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Press 'q' to quit
        print("Exiting...")
        break
    elif key == ord('s'):  # Press 's' to pause
        print("Paused. Press any key to continue.")
        cv2.waitKey(-1)
    elif key == ord('p'):  # Press 'p' to save a screenshot
        screenshot_name = f'person_detection_{time.strftime("%Y%m%d_%H%M%S")}.png'
        cv2.imwrite(screenshot_name, frame)
        print(f"Screenshot saved as {screenshot_name}")
    
    # Calculate loop time for FPS
    loop_end = time.perf_counter()
    loop_time = loop_end - loop_start

# Clean up
print(f'Average FPS: {avg_fps:.2f}')
print(f'Total frames processed: {frames_processed}')
print(f'Total runtime: {elapsed_time:.2f} seconds')

cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows() 