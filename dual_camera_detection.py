#!/usr/bin/env python3
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
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected people (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default='640x480')
parser.add_argument('--record', help='Record results from both cameras and save them as "camera1_people.avi" and "camera2_people.avi"',
                    action='store_true')
parser.add_argument('--show-fps', help='Display FPS counter on the video',
                    action='store_true', default=True)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
source1 = args.source1
source2 = args.source2
min_thresh = args.thresh
user_res = args.resolution
record = args.record
show_fps = args.show_fps

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

# Get actual frame dimensions (might be different from requested)
actual_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera 1 dimensions: {actual_width1}x{actual_height1}")
print(f"Camera 2 dimensions: {actual_width2}x{actual_height2}")

# Set up recording if enabled
if record:
    record_name1 = 'camera1_people.avi'
    record_name2 = 'camera2_people.avi'
    record_fps = 30
    recorder1 = cv2.VideoWriter(record_name1, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (actual_width1, actual_height1))
    recorder2 = cv2.VideoWriter(record_name2, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (actual_width2, actual_height2))
    print(f"Recording to {record_name1} and {record_name2}")

# Person bounding box color (blue)
person_color = (255, 0, 0)  # BGR format

# Create frame queues for each camera
frame_queue1 = queue.Queue(maxsize=1)
frame_queue2 = queue.Queue(maxsize=1)
result_queue1 = queue.Queue(maxsize=1)
result_queue2 = queue.Queue(maxsize=1)

# Flag to signal threads to exit
exit_flag = threading.Event()

# Function to capture frames from a camera
def capture_frames(cap, frame_queue, camera_num):
    while not exit_flag.is_set():
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
        
        # Put frame in queue, replacing old frame if queue is full
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame, time.time()))
        
        # Small delay to prevent CPU overuse
        time.sleep(0.01)

# Function to process frames with YOLO for human detection
def process_frames():
    fps_buffer1 = []
    fps_buffer2 = []
    fps_avg_len = 30
    
    while not exit_flag.is_set():
        # Process camera 1 if frame available
        try:
            frame1, timestamp1 = frame_queue1.get(timeout=0.1)
            t_start1 = time.perf_counter()
            
            # Run inference
            results1 = model(frame1, verbose=False)
            
            # Process results for human detection only
            processed_frame1, person_count1 = process_detections(frame1, results1[0].boxes, "Camera 1")
            
            # Calculate FPS
            t_end1 = time.perf_counter()
            fps1 = 1 / (t_end1 - t_start1)
            fps_buffer1.append(fps1)
            if len(fps_buffer1) > fps_avg_len:
                fps_buffer1.pop(0)
            avg_fps1 = sum(fps_buffer1) / len(fps_buffer1)
            
            # Add FPS to frame if enabled
            if show_fps:
                cv2.putText(processed_frame1, f'FPS: {avg_fps1:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Put processed frame in result queue
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
            t_start2 = time.perf_counter()
            
            # Run inference
            results2 = model(frame2, verbose=False)
            
            # Process results for human detection only
            processed_frame2, person_count2 = process_detections(frame2, results2[0].boxes, "Camera 2")
            
            # Calculate FPS
            t_end2 = time.perf_counter()
            fps2 = 1 / (t_end2 - t_start2)
            fps_buffer2.append(fps2)
            if len(fps_buffer2) > fps_avg_len:
                fps_buffer2.pop(0)
            avg_fps2 = sum(fps_buffer2) / len(fps_buffer2)
            
            # Add FPS to frame if enabled
            if show_fps:
                cv2.putText(processed_frame2, f'FPS: {avg_fps2:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Put processed frame in result queue
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

# Function to process detections and draw only people
def process_detections(frame, detections, camera_label):
    # Initialize person counter
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
    
    # Display person count
    cv2.putText(frame, f'{camera_label}: {person_count} people', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame, person_count

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

print("People detection started on both cameras. Press 'q' to quit, 'p' to save screenshots.")

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
            cv2.imshow('Camera 1 - People Detection', frame1)
            if record:
                recorder1.write(frame1)
                
        if frame2 is not None:
            cv2.imshow('Camera 2 - People Detection', frame2)
            if record:
                recorder2.write(frame2)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            exit_flag.set()
            break
        elif key == ord('p'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if frame1 is not None:
                screenshot_name1 = f'camera1_people_{timestamp}.png'
                cv2.imwrite(screenshot_name1, frame1)
                print(f"Saved {screenshot_name1}")
            if frame2 is not None:
                screenshot_name2 = f'camera2_people_{timestamp}.png'
                cv2.imwrite(screenshot_name2, frame2)
                print(f"Saved {screenshot_name2}")
                
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