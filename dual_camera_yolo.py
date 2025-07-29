import os
import sys
import argparse
import time
import threading
import queue

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
# RTSP URL - replace with your camera's RTSP URL
# Using a local VLC stream for testing
rtsp_url1 = "rtsp://127.0.0.1:8554/video"
rtsp_url2 = "rtsp://127.0.0.1:8555/video"
yolo_model = "yolov8n.pt"

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "yolov8n.pt")',
                    default=yolo_model)
parser.add_argument('--source1', help='First camera source: can be RTSP stream URL ("rtsp://..."), \
                    video file ("testvid.mp4"), or index of USB camera ("0")', 
                    default=rtsp_url1)
parser.add_argument('--source2', help='Second camera source: can be RTSP stream URL ("rtsp://..."), \
                    video file ("testvid.mp4"), or index of USB camera ("1")',
                    default=rtsp_url2)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default='640x480')
parser.add_argument('--record', help='Record results from both cameras and save them as "camera1.avi" and "camera2.avi"',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
source1 = args.source1
source2 = args.source2
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Parse resolution
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
    # Load the model into memory and get labelmap
    model = YOLO(model_path, task='detect')

labels = model.names

# Set bounding box colors (using the Tableau 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Function to parse source type
def parse_source(source):
    # Check if it's a number (camera index)
    try:
        return 'camera', int(source)
    except ValueError:
        # Check if it's an RTSP URL
        if source.startswith('rtsp://'):
            return 'rtsp', source
        # Otherwise assume it's a video file
        elif os.path.isfile(source):
            return 'video', source
        else:
            print(f'ERROR: Source {source} is invalid or not found.')
            sys.exit(0)

# Parse sources
source1_type, source1_path = parse_source(source1)
source2_type, source2_path = parse_source(source2)

# Initialize video captures
cap1 = cv2.VideoCapture(source1_path)
cap2 = cv2.VideoCapture(source2_path)

# Set camera resolutions
cap1.set(3, resW)
cap1.set(4, resH)
cap2.set(3, resW)
cap2.set(4, resH)

# Check if cameras opened successfully
if not cap1.isOpened():
    print(f"Error: Could not open source 1: {source1}")
    sys.exit(0)
if not cap2.isOpened():
    print(f"Error: Could not open source 2: {source2}")
    sys.exit(0)

# Set up recording if enabled
if record:
    recorder1 = cv2.VideoWriter('camera1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))
    recorder2 = cv2.VideoWriter('camera2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Create frame queues for each camera
frame_queue1 = queue.Queue(maxsize=1)
frame_queue2 = queue.Queue(maxsize=1)
result_queue1 = queue.Queue(maxsize=1)
result_queue2 = queue.Queue(maxsize=1)

# Flag to signal threads to exit
exit_flag = threading.Event()

# Function to capture frames from a camera
def capture_frames(cap, frame_queue, camera_num):
    fps_buffer = []
    while not exit_flag.is_set():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        
        if not ret:
            if camera_num == 1:
                source_type = source1_type
            else:
                source_type = source2_type
                
            if source_type == 'rtsp':
                print(f'Failed to read from camera {camera_num} RTSP stream. Retrying...')
                time.sleep(1)
                continue
            else:
                print(f'Failed to read from camera {camera_num}. Exiting.')
                exit_flag.set()
                break
        
        # Resize frame
        frame = cv2.resize(frame, (resW, resH))
        
        # Put frame in queue, replacing old frame if queue is full
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame, time.time()))
        
        # Calculate FPS
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        fps_buffer.append(fps)
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)

# Function to process frames with YOLO
def process_frames():
    while not exit_flag.is_set():
        # Process camera 1 if frame available
        try:
            frame1, timestamp1 = frame_queue1.get(timeout=0.1)
            results1 = model(frame1, verbose=False)
            processed_frame1 = draw_detections(frame1, results1[0].boxes, f"Camera 1")
            if result_queue1.full():
                try:
                    result_queue1.get_nowait()
                except queue.Empty:
                    pass
            result_queue1.put(processed_frame1)
        except queue.Empty:
            pass
        
        # Process camera 2 if frame available
        try:
            frame2, timestamp2 = frame_queue2.get(timeout=0.1)
            results2 = model(frame2, verbose=False)
            processed_frame2 = draw_detections(frame2, results2[0].boxes, f"Camera 2")
            if result_queue2.full():
                try:
                    result_queue2.get_nowait()
                except queue.Empty:
                    pass
            result_queue2.put(processed_frame2)
        except queue.Empty:
            pass
        
        # Small delay to prevent CPU overuse
        time.sleep(0.01)

# Function to draw detections on frames
def draw_detections(frame, detections, camera_label):
    # Initialize variable for basic object counting
    object_count = 0
    
    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        
        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        
        # Get bounding box confidence
        conf = detections[i].conf.item()
        
        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Count objects
            object_count += 1
    
    # Add camera label and object count
    cv2.putText(frame, f"{camera_label}: {object_count} objects", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

# Start capture threads
capture_thread1 = threading.Thread(target=capture_frames, args=(cap1, frame_queue1, 1))
capture_thread2 = threading.Thread(target=capture_frames, args=(cap2, frame_queue2, 2))
process_thread = threading.Thread(target=process_frames)

capture_thread1.daemon = True
capture_thread2.daemon = True
process_thread.daemon = True

capture_thread1.start()
capture_thread2.start()
process_thread.start()

print("Processing started. Press 'q' to quit, 'p' to save screenshots.")

# Main display loop
try:
    while not exit_flag.is_set():
        # Get processed frames if available
        frame1 = None
        frame2 = None
        
        try:
            frame1 = result_queue1.get(block=False)
        except queue.Empty:
            pass
            
        try:
            frame2 = result_queue2.get(block=False)
        except queue.Empty:
            pass
            
        # Display frames
        if frame1 is not None:
            cv2.imshow('Camera 1', frame1)
            if record:
                recorder1.write(frame1)
                
        if frame2 is not None:
            cv2.imshow('Camera 2', frame2)
            if record:
                recorder2.write(frame2)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            exit_flag.set()
            break
        elif key == ord('p'):
            if frame1 is not None:
                cv2.imwrite('camera1_capture.png', frame1)
                print("Saved camera1_capture.png")
            if frame2 is not None:
                cv2.imwrite('camera2_capture.png', frame2)
                print("Saved camera2_capture.png")
                
        time.sleep(0.01)  # Small delay to prevent CPU overuse

except KeyboardInterrupt:
    print("Interrupted by user")
    exit_flag.set()

finally:
    # Clean up
    exit_flag.set()
    
    # Wait for threads to finish
    capture_thread1.join(timeout=1.0)
    capture_thread2.join(timeout=1.0)
    process_thread.join(timeout=1.0)
    
    # Release resources
    cap1.release()
    cap2.release()
    if record:
        recorder1.release()
        recorder2.release()
    cv2.destroyAllWindows()
    
    print("Clean exit complete") 