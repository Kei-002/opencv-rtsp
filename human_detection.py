#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import torch
import queue

def main():
    # Try to use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # RTSP URL - replace with your camera's RTSP URL
    # Using a local VLC stream for testing
    rtsp_url = "rtsp://127.0.0.1:8554/video"
    
    print("Loading YOLOv8 model...")
    # Load the YOLOv8 model - using the smallest model for better performance on Raspberry Pi
    model = YOLO('yolov8n.pt')
    model.to(device)
    print("Model loaded successfully")
    
    # Open the RTSP stream with FFMPEG backend for better performance
    print(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Set buffer size to 1 for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Check if the stream is opened successfully
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return
    
    # Set an even lower resolution for better performance on Raspberry Pi
    resolution_width = 192   # Further reduced for Pi 4
    resolution_height = 144  # Further reduced for Pi 4
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
    
    # Try to set a lower FPS on the capture
    cap.set(cv2.CAP_PROP_FPS, 3)  # Reduced from 5 FPS to 3 FPS for Pi 4
    
    print("Human detection started. Press 'q' to quit.")
    
    # Create a queue for frames to be processed
    frame_queue = queue.Queue(maxsize=1)
    
    # Create a queue for processed frames
    result_queue = queue.Queue(maxsize=1)
    
    # Initialize variables
    running = True
    frame_count = 0
    start_time = time.time()
    
    # Function to continuously read frames from the camera
    def camera_reader():
        nonlocal running
        nonlocal cap  # Add this line to access the cap variable from the outer scope
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to receive frame from stream.")
                # Try to reconnect
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
                cap.set(cv2.CAP_PROP_FPS, 3)
                continue
                
            # Put the frame in the queue, replacing any old frame
            try:
                # If queue is full, remove the old frame
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Skip this frame if queue is full
            
            # Sleep a tiny bit to avoid hogging CPU
            time.sleep(0.03)  # Increased sleep time to reduce CPU usage
    
    # Function to process frames for detection
    def frame_processor():
        nonlocal running
        detection_count = 0
        last_detection_time = time.time()
        detection_interval = 0.8  # Reduced from 1.5 seconds to 0.8 seconds
        
        # Create a persistent detection frame to avoid reallocating memory
        detection_frame = None
        
        # Store the last detected boxes to display between detections
        last_detected_boxes = []
        # Time when boxes were last updated
        last_boxes_update_time = time.time()
        # Max age for boxes before they're considered stale (in seconds)
        max_box_age = 3.0
        
        while running:
            try:
                # Get a frame from the queue
                frame = frame_queue.get(timeout=0.5)
                
                # Create a copy for display (without detection boxes)
                display_frame = frame.copy()
                
                # Add FPS text to display frame
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed >= 1.0 and frame_count > 0:
                    fps = frame_count / elapsed
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (5, 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Check if it's time to run detection
                if current_time - last_detection_time > detection_interval:
                    last_detection_time = current_time
                    detection_count += 1
                    
                    # Resize to very low resolution for detection (even smaller for Pi 4)
                    if detection_frame is None or detection_frame.shape[0] != 64 or detection_frame.shape[1] != 96:
                        detection_frame = cv2.resize(frame, (96, 64))  # Even smaller for detection on Pi 4
                    else:
                        cv2.resize(frame, (96, 64), dst=detection_frame)
                    
                    # Run detection with higher confidence threshold
                    results = model(detection_frame, classes=0, verbose=False, conf=0.5, iou=0.5)
                    
                    # Process results
                    person_detected = False
                    for result in results:
                        boxes = result.boxes
                        
                        if len(boxes) > 0:
                            person_detected = True
                            print(f"Detected {len(boxes)} person(s)")
                            
                            # Update the last detected boxes
                            last_detected_boxes = []
                            for box in boxes:
                                # Get box coordinates and scale to original frame
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                # Scale coordinates back to original frame size
                                x1 = int(x1 * (frame.shape[1] / detection_frame.shape[1]))
                                y1 = int(y1 * (frame.shape[0] / detection_frame.shape[0]))
                                x2 = int(x2 * (frame.shape[1] / detection_frame.shape[1]))
                                y2 = int(y2 * (frame.shape[0] / detection_frame.shape[0]))
                                conf = float(box.conf[0])
                                last_detected_boxes.append((x1, y1, x2, y2, conf))
                            
                            # Update the timestamp when boxes were last updated
                            last_boxes_update_time = current_time
                    
                    # If no person detected after several attempts, log it
                    if detection_count >= 5 and not person_detected:
                        print("No persons detected in the last 5 attempts. Check camera positioning.")
                        detection_count = 0
                
                # Check if boxes are stale and should be cleared
                if current_time - last_boxes_update_time > max_box_age:
                    last_detected_boxes = []
                
                # Always draw the last detected boxes (persistent display)
                if last_detected_boxes:
                    for x1, y1, x2, y2, conf in last_detected_boxes:
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        
                        # Only add text for high confidence detections
                        if conf > 0.6:
                            # Use even smaller font
                            cv2.putText(display_frame, f"{conf:.2f}", (x1, y1 - 2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Put the processed frame in the result queue
                try:
                    # If queue is full, remove the old frame
                    if result_queue.full():
                        result_queue.get_nowait()
                    result_queue.put_nowait(display_frame)
                except queue.Full:
                    pass  # Skip this frame if queue is full
                
            except queue.Empty:
                # No frame available
                pass
            
            # Sleep a tiny bit to avoid hogging CPU
            time.sleep(0.03)  # Increased sleep time to reduce CPU usage
    
    # Start the camera reader thread
    camera_thread = threading.Thread(target=camera_reader)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Start the frame processor thread
    processor_thread = threading.Thread(target=frame_processor)
    processor_thread.daemon = True
    processor_thread.start()
    
    try:
        while running:
            # Get a frame from the result queue
            try:
                display_frame = result_queue.get(timeout=0.1)
                
                # Increment frame counter for FPS calculation
                frame_count += 1
                
                # Display the frame
                cv2.imshow('Human Detection', display_frame)
                
                # Reset FPS counter periodically
                current_time = time.time()
                if current_time - start_time >= 5.0:
                    start_time = current_time
                    frame_count = 0
                
            except queue.Empty:
                # No frame available, just continue
                pass
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
            
            # Sleep a tiny bit to avoid hogging CPU
            time.sleep(0.03)  # Increased sleep time to reduce CPU usage
    
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        # Signal threads to stop
        running = False
        
        # Wait for threads to finish
        camera_thread.join(timeout=1.0)
        processor_thread.join(timeout=1.0)
        
        # Release resources
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 