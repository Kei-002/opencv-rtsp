#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import subprocess
import torch
import os
import socket

def main():
    # Try to use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # RTSP URL - replace with your camera's RTSP URL
    rtsp_url = "rtsp://demo:demo@ipvmdemo.dyndns.org:5542/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast"
    
    # Output stream settings - reduced for better performance
    output_width = 320
    output_height = 240
    fps = 10  # Lower FPS for better performance
    
    # Use RTP instead of UDP for better compatibility with VLC
    output_url = "rtp://127.0.0.1:5004"
    
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
    
    # Set a lower resolution for better performance on Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Create an SDP file for VLC
    sdp_content = f"""v=0
o=- 0 0 IN IP4 127.0.0.1
s=No Name
c=IN IP4 127.0.0.1
t=0 0
m=video 5004 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
"""
    
    # Save the SDP file
    sdp_file = "stream.sdp"
    with open(sdp_file, "w") as f:
        f.write(sdp_content)
    
    print(f"Created SDP file: {os.path.abspath(sdp_file)}")
    
    # Initialize the FFmpeg process to stream the output
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{output_width}x{output_height}",
        '-r', str(fps),
        '-i', '-',  # Input from pipe
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',  # Optimize for low latency
        '-crf', '23',  # Better quality for visibility
        '-g', '10',    # GOP size - lower for faster seeking
        '-keyint_min', '10',  # Minimum GOP size
        '-f', 'rtp',
        output_url
    ]
    
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    print(f"Human detection started. Streaming to {output_url}")
    print(f"To view in VLC, open: {os.path.abspath(sdp_file)}")
    print("Instructions for VLC:")
    print("1. Open VLC")
    print("2. Go to Media > Open File")
    print(f"3. Select the SDP file: {os.path.abspath(sdp_file)}")
    print("4. Click Play")
    print("Press Ctrl+C to stop the stream")
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    last_detection_time = 0
    detection_interval = 1.0  # Run detection every 1 second
    detection_running = False
    current_frame = None
    processed_frame = None
    lock = threading.Lock()
    
    # Function to run detection in a separate thread
    def run_detection(frame):
        nonlocal processed_frame, lock, detection_running
        
        try:
            # Resize frame for detection to even lower resolution for faster processing
            detection_frame = cv2.resize(frame, (160, 120))  # Very low resolution for detection
            
            # Run detection
            results = model(detection_frame, classes=0, verbose=False, conf=0.4)  # class 0 is person
            
            # Create a copy of the original frame to draw on
            output_frame = frame.copy()
            
            # Process the results
            person_count = 0
            for result in results:
                boxes = result.boxes
                
                if len(boxes) > 0:
                    person_count = len(boxes)
                
                for box in boxes:
                    # Get box coordinates and scale back to original frame size
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Scale coordinates back to original frame size
                    x1 = int(x1 * (frame.shape[1] / detection_frame.shape[1]))
                    y1 = int(y1 * (frame.shape[0] / detection_frame.shape[0]))
                    x2 = int(x2 * (frame.shape[1] / detection_frame.shape[1]))
                    y2 = int(y2 * (frame.shape[0] / detection_frame.shape[0]))
                    
                    # Get confidence score
                    conf = float(box.conf[0])
                    
                    # Draw bounding box and label on the frame
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thinner rectangle
                    label = f"Person: {conf:.2f}"
                    cv2.putText(output_frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            if person_count > 0:
                print(f"Detected {person_count} person(s)")
            
            # Update the processed frame with detection results
            with lock:
                processed_frame = output_frame
        except Exception as e:
            print(f"Error in detection: {str(e)}")
        finally:
            detection_running = False
    
    try:
        while True:
            # Read a frame from the stream
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to receive frame from stream.")
                # Try to reconnect
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                continue
            
            # Update the current frame
            with lock:
                current_frame = frame.copy()
            
            # Increment frame counter
            frame_count += 1
            
            # Run detection periodically
            current_time = time.time()
            if current_time - last_detection_time > detection_interval and not detection_running:
                last_detection_time = current_time
                detection_running = True
                
                # Start detection in a separate thread
                detection_thread = threading.Thread(target=run_detection, args=(current_frame.copy(),))
                detection_thread.daemon = True
                detection_thread.start()
            
            # Use the processed frame if available, otherwise use the current frame
            display_frame = None
            with lock:
                if processed_frame is not None:
                    display_frame = processed_frame.copy()
                else:
                    display_frame = current_frame.copy()
            
            # Calculate and display FPS
            elapsed_time = current_time - start_time
            if elapsed_time >= 1.0:
                fps_value = frame_count / elapsed_time
                cv2.putText(display_frame, f"FPS: {fps_value:.1f}", (5, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                print(f"Current FPS: {fps_value:.2f}")
                frame_count = 0
                start_time = current_time
            
            # Also show the frame locally (optional)
            cv2.imshow("Stream Preview", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Write the frame to the FFmpeg process
            ffmpeg_process.stdin.write(display_frame.tobytes())
            
    except KeyboardInterrupt:
        print("Stopping the stream...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        # Release resources
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print("Stream ended")

if __name__ == "__main__":
    main() 