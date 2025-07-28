#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import subprocess
import torch
import os

class RTSPCamera:
    def __init__(self, name, rtsp_url, model):
        self.name = name
        self.rtsp_url = rtsp_url
        self.model = model
        self.frame = None
        self.processed_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_detection_time = 0
        self.detection_interval = 1.0  # Run detection every 1 second
        self.detection_running = False  # Flag to prevent multiple detection threads
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def update(self):
        print(f"Starting camera thread for {self.name} with URL: {self.rtsp_url}")
        # Use FFMPEG backend for potentially better RTSP handling
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Set buffer size to 1 for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set a much lower resolution for better performance
        resolution_width = 320   # Reduced from 640
        resolution_height = 240  # Reduced from 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
        
        # Try to set a lower FPS
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        frame_count_for_fps = 0
        start_time_for_fps = time.time()
        
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
                cap.set(cv2.CAP_PROP_FPS, 10)
                continue
            
            self.frame_count += 1
            frame_count_for_fps += 1
            
            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - start_time_for_fps
            if elapsed_time >= 1.0:
                self.fps = frame_count_for_fps / elapsed_time
                frame_count_for_fps = 0
                start_time_for_fps = current_time
                print(f"{self.name} - Current FPS: {self.fps:.2f}")
            
            # Run detection periodically rather than every N frames
            # This ensures consistent detection regardless of frame rate
            if current_time - self.last_detection_time > self.detection_interval and not self.detection_running:
                self.last_detection_time = current_time
                self.detection_running = True
                
                # Create a copy for detection to avoid modifying the original during processing
                detection_frame = frame.copy()
                
                # Run detection in a separate thread to avoid blocking
                detection_thread = threading.Thread(
                    target=self.run_detection,
                    args=(detection_frame,)
                )
                detection_thread.daemon = True
                detection_thread.start()
            
            # Add camera name and FPS
            cv2.putText(frame, f"{self.name}: {self.fps:.1f}", (5, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Update the processed frame
            with self.lock:
                self.processed_frame = frame.copy()
        
        cap.release()
    
    def run_detection(self, frame):
        try:
            # Resize frame for detection to even lower resolution for faster processing
            detection_frame = cv2.resize(frame, (160, 120))  # Very low resolution for detection
            
            # Run detection
            results = self.model(detection_frame, classes=0, verbose=False, conf=0.4)  # class 0 is person
            
            # Process the results
            for result in results:
                boxes = result.boxes
                
                if len(boxes) > 0:
                    print(f"{self.name} - Detected {len(boxes)} person(s)")
                
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
                    with self.lock:
                        if self.processed_frame is not None:
                            cv2.rectangle(self.processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            label = f"Person: {conf:.2f}"
                            cv2.putText(self.processed_frame, label, (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        except Exception as e:
            print(f"Error in detection for {self.name}: {str(e)}")
        finally:
            self.detection_running = False
    
    def get_frame(self):
        with self.lock:
            if self.processed_frame is not None:
                return self.processed_frame.copy()
            return None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)


def main():
    # Try to use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # RTSP URLs - replace with your camera URLs
    # Using test streams more likely to have people
    rtsp_url1 = "rtsp://demo:demo@ipvmdemo.dyndns.org:5542/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast"
    rtsp_url2 = "rtsp://rtsp.stream/pattern"  # Fallback to pattern if the first one doesn't work
    
    # Output stream settings
    output_width = 640  # Combined width of two 320px cameras
    output_height = 240
    fps = 10
    
    # Use RTP instead of UDP for better compatibility with VLC
    output_url = "rtp://127.0.0.1:5004"
    
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
    sdp_file = "dual_stream.sdp"
    with open(sdp_file, "w") as f:
        f.write(sdp_content)
    
    print(f"Created SDP file: {os.path.abspath(sdp_file)}")
    
    # Load the YOLOv8 model once and share it
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    model.to(device)
    print("Model loaded successfully")
    
    # Create camera objects
    camera1 = RTSPCamera("Cam1", rtsp_url1, model)
    camera2 = RTSPCamera("Cam2", rtsp_url2, model)
    
    # Start the camera threads
    camera1.start()
    camera2.start()
    
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
    
    print(f"Dual camera human detection started. Streaming to {output_url}")
    print(f"To view in VLC, open: {os.path.abspath(sdp_file)}")
    print("Instructions for VLC:")
    print("1. Open VLC")
    print("2. Go to Media > Open File")
    print(f"3. Select the SDP file: {os.path.abspath(sdp_file)}")
    print("4. Click Play")
    print("Press Ctrl+C to stop the stream")
    
    # Give some time for cameras to initialize
    time.sleep(2)
    
    try:
        while True:
            # Get frames from both cameras
            frame1 = camera1.get_frame()
            frame2 = camera2.get_frame()
            
            # Create a combined display
            if frame1 is not None and frame2 is not None:
                # Use a fixed target height for consistency
                target_height = output_height
                
                # Resize both frames to the same height while maintaining aspect ratio
                aspect_ratio1 = frame1.shape[1] / frame1.shape[0]
                aspect_ratio2 = frame2.shape[1] / frame2.shape[0]
                
                width1_new = int(target_height * aspect_ratio1)
                width2_new = int(target_height * aspect_ratio2)
                
                # Make sure we don't exceed half the output width
                if width1_new > output_width // 2:
                    width1_new = output_width // 2
                if width2_new > output_width // 2:
                    width2_new = output_width // 2
                
                frame1 = cv2.resize(frame1, (width1_new, target_height))
                frame2 = cv2.resize(frame2, (width2_new, target_height))
                
                # Combine frames horizontally
                combined_frame = np.hstack((frame1, frame2))
                
                # Ensure the combined frame has the expected dimensions
                if combined_frame.shape[1] != output_width or combined_frame.shape[0] != output_height:
                    combined_frame = cv2.resize(combined_frame, (output_width, output_height))
                
                # Also show the frame locally (optional)
                cv2.imshow("Dual Stream Preview", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Write the frame to the FFmpeg process
                ffmpeg_process.stdin.write(combined_frame.tobytes())
            
    except KeyboardInterrupt:
        print("Stopping the stream...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        # Stop the camera threads
        print("Stopping camera threads...")
        camera1.stop()
        camera2.stop()
        cv2.destroyAllWindows()
        
        # Close FFmpeg process
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print("Stream ended")


if __name__ == "__main__":
    main() 