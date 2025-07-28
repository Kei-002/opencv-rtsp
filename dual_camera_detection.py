#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import torch
import queue

class RTSPCamera:
    def __init__(self, name, rtsp_url, model):
        self.name = name
        self.rtsp_url = rtsp_url
        self.model = model
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        # Store the last detected boxes to display between detections
        self.last_detected_boxes = []
        self.last_boxes_update_time = time.time()
        self.max_box_age = 3.0  # Max age for boxes before they're considered stale (seconds)
    
    def start(self):
        self.running = True
        # Start camera reader thread
        self.camera_thread = threading.Thread(target=self.camera_reader)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Start frame processor thread
        self.processor_thread = threading.Thread(target=self.frame_processor)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        return self
    
    def camera_reader(self):
        print(f"Starting camera thread for {self.name} with URL: {self.rtsp_url}")
        # Use FFMPEG backend for better RTSP handling
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Set buffer size to 1 for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set a much lower resolution for better performance on Raspberry Pi
        resolution_width = 192   # Further reduced for Pi 4
        resolution_height = 144  # Further reduced for Pi 4
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
        
        # Try to set a lower FPS for Pi 4
        cap.set(cv2.CAP_PROP_FPS, 3)  # Reduced to 3 FPS
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to receive frame from {self.name}")
                time.sleep(0.5)  # Wait before retrying
                # Try to reconnect
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
                cap.set(cv2.CAP_PROP_FPS, 3)
                continue
            
            # Put the frame in the queue, replacing any old frame
            try:
                # If queue is full, remove the old frame
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Skip this frame if queue is full
            
            # Sleep a tiny bit to avoid hogging CPU
            time.sleep(0.03)  # Increased sleep time to reduce CPU usage
        
        cap.release()
    
    def frame_processor(self):
        detection_count = 0
        last_detection_time = time.time()
        detection_interval = 0.8  # Reduced from 1.0 second to 0.8 seconds
        
        # Create a persistent detection frame to avoid reallocating memory
        detection_frame = None
        
        while self.running:
            try:
                # Get a frame from the queue
                frame = self.frame_queue.get(timeout=0.5)
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = current_time
                
                # Add camera name and FPS
                cv2.putText(display_frame, f"{self.name}: {self.fps:.1f}", (5, 15), 
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
                    try:
                        results = self.model(detection_frame, classes=0, verbose=False, conf=0.5, iou=0.5)
                        
                        # Process results
                        person_detected = False
                        for result in results:
                            boxes = result.boxes
                            
                            if len(boxes) > 0:
                                person_detected = True
                                print(f"{self.name} - Detected {len(boxes)} person(s)")
                                
                                # Update the last detected boxes
                                self.last_detected_boxes = []
                                for box in boxes:
                                    # Get box coordinates and scale to original frame
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    # Scale coordinates back to original frame size
                                    x1 = int(x1 * (frame.shape[1] / detection_frame.shape[1]))
                                    y1 = int(y1 * (frame.shape[0] / detection_frame.shape[0]))
                                    x2 = int(x2 * (frame.shape[1] / detection_frame.shape[1]))
                                    y2 = int(y2 * (frame.shape[0] / detection_frame.shape[0]))
                                    conf = float(box.conf[0])
                                    self.last_detected_boxes.append((x1, y1, x2, y2, conf))
                                
                                # Update the timestamp when boxes were last updated
                                self.last_boxes_update_time = current_time
                        
                        # If no person detected after several attempts, log it
                        if detection_count >= 5 and not person_detected:
                            print(f"{self.name} - No persons detected in the last 5 attempts.")
                            detection_count = 0
                    
                    except Exception as e:
                        print(f"Error in detection for {self.name}: {str(e)}")
                
                # Check if boxes are stale and should be cleared
                if current_time - self.last_boxes_update_time > self.max_box_age:
                    self.last_detected_boxes = []
                
                # Always draw the last detected boxes (persistent display)
                if self.last_detected_boxes:
                    for x1, y1, x2, y2, conf in self.last_detected_boxes:
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        
                        # Only add text for high confidence detections
                        if conf > 0.6:
                            # Use smaller font
                            cv2.putText(display_frame, f"{conf:.2f}", (x1, y1 - 2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Put the processed frame in the result queue
                try:
                    # If queue is full, remove the old frame
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait(display_frame)
                except queue.Full:
                    pass  # Skip this frame if queue is full
                
            except queue.Empty:
                # No frame available
                pass
            
            # Sleep a tiny bit to avoid hogging CPU
            time.sleep(0.03)  # Increased sleep time to reduce CPU usage
    
    def get_frame(self):
        try:
            return self.result_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1.0)
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join(timeout=1.0)


def main():
    # Try to use CUDA if available, but on Pi 4 it will use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # RTSP URLs - replace with your camera URLs
    rtsp_url1 = "rtsp://127.0.0.1:8554/video"
    rtsp_url2 = "rtsp://127.0.0.1:8554/video"  # Fallback to pattern if the first one doesn't work
    
    # Load the YOLOv8 model once and share it
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Using the smallest model for better performance on Pi 4
    model.to(device)
    print("Model loaded successfully")
    
    # Create camera objects
    camera1 = RTSPCamera("Cam1", rtsp_url1, model)
    camera2 = RTSPCamera("Cam2", rtsp_url2, model)
    
    # Start the camera threads
    camera1.start()
    camera2.start()
    
    print("Dual camera human detection started. Press 'q' to quit.")
    
    # Give some time for cameras to initialize
    time.sleep(2)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Create a black frame as fallback for when cameras aren't working
    fallback_frame = np.zeros((144, 192, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, "Camera not available", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Create the window early to ensure it's always displayed
    cv2.namedWindow('Dual Camera Human Detection', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Get frames from both cameras
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame()
            
            # Use fallback frame if camera frames are not available
            if frame1 is None:
                frame1 = fallback_frame.copy()
                cv2.putText(frame1, "Cam1: No signal", (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            if frame2 is None:
                frame2 = fallback_frame.copy()
                cv2.putText(frame2, "Cam2: No signal", (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Use a fixed target height for consistency - even smaller for Pi 4
            target_height = 144  # Reduced for Pi 4
            
            # Resize both frames to the same height while maintaining aspect ratio
            aspect_ratio1 = frame1.shape[1] / frame1.shape[0]
            aspect_ratio2 = frame2.shape[1] / frame2.shape[0]
            
            width1_new = int(target_height * aspect_ratio1)
            width2_new = int(target_height * aspect_ratio2)
            
            frame1 = cv2.resize(frame1, (width1_new, target_height))
            frame2 = cv2.resize(frame2, (width2_new, target_height))
            
            # Combine frames horizontally
            combined_frame = np.hstack((frame1, frame2))
            
            # Increment frame counter
            frame_count += 1
            
            # Calculate overall FPS occasionally
            current_time = time.time()
            if current_time - start_time >= 5.0:
                fps = frame_count / (current_time - start_time)
                print(f"Overall FPS: {fps:.2f}")
                frame_count = 0
                start_time = current_time
            
            # Display the combined frame
            cv2.imshow('Dual Camera Human Detection', combined_frame)
            
            # Break the loop if 'q' is pressed - use a small fixed wait time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Sleep a tiny bit to avoid hogging CPU
            time.sleep(0.03)  # Increased sleep time to reduce CPU usage
    
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        # Stop the camera threads
        print("Stopping camera threads...")
        camera1.stop()
        camera2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 