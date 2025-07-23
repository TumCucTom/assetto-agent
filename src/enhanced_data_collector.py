import pygame
import csv
import time
import os
import json
from datetime import datetime
import numpy as np
from mss import mss
import cv2
import threading
import queue

class EnhancedExpertDataCollector:
    def __init__(self, output_dir="expert_data"):
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None
        self.screenshot_dir = None
        self.lane_masks_dir = None
        self.frame_count = 0
        self.session_data = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.screenshot_dir = os.path.join(output_dir, "screenshots")
        self.lane_masks_dir = os.path.join(output_dir, "lane_masks")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.lane_masks_dir, exist_ok=True)
        
        # Initialize pygame for joystick
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick found!")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        # Initialize screen capture
        self.sct = mss()
        self.bounding_box = {'top': 320, 'left': 680, 'width': 600, 'height': 400}
        
        # Data queues for threading
        self.frame_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        
        print(f"Joystick: {self.joystick.get_name()}")
        print(f"Axes: {self.joystick.get_numaxes()}")
        print(f"Buttons: {self.joystick.get_numbuttons()}")
        
    def start_recording(self, session_name=None, track_info=None):
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session directory
        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save session metadata
        self.session_data = {
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'track_info': track_info or {},
            'joystick_info': {
                'name': self.joystick.get_name(),
                'axes': self.joystick.get_numaxes(),
                'buttons': self.joystick.get_numbuttons(),
                'hats': self.joystick.get_numhats()
            }
        }
        
        metadata_path = os.path.join(session_dir, "session_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        # Create session-specific directories
        self.session_screenshot_dir = os.path.join(session_dir, "screenshots")
        self.session_lane_masks_dir = os.path.join(session_dir, "lane_masks")
        os.makedirs(self.session_screenshot_dir, exist_ok=True)
        os.makedirs(self.session_lane_masks_dir, exist_ok=True)
        
        # Create CSV file
        csv_path = os.path.join(session_dir, "driving_data.csv")
        self.csv_file = open(csv_path, 'w', newline='')
        
        # Define CSV headers
        headers = [
            'timestamp', 'frame_id', 'steering', 'throttle', 'brake', 
            'gear', 'speed', 'x_pos', 'y_pos', 'heading', 'lap_time',
            'lap_number', 'track_position', 'screenshot_path', 'lane_mask_path',
            'raw_axes', 'raw_buttons', 'raw_hats'
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
        self.csv_writer.writeheader()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print(f"Started recording to {session_dir}")
        
    def stop_recording(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        # Stop processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.frame_queue.put(None)  # Signal to stop
            self.processing_thread.join(timeout=5)
        
        # Save final metadata
        if self.session_data:
            self.session_data['end_time'] = datetime.now().isoformat()
            self.session_data['total_frames'] = self.frame_count
            
            session_dir = os.path.join(self.output_dir, self.session_data['session_name'])
            metadata_path = os.path.join(session_dir, "session_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        
        print("Recording stopped")
        
    def _process_frames(self):
        """Background thread for processing frames and generating lane masks"""
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:  # Stop signal
                    break
                
                # Process frame for lane detection (placeholder)
                # In a real implementation, you'd run the LaneNet model here
                frame = frame_data['frame']
                frame_id = frame_data['frame_id']
                
                # Generate lane mask (placeholder - replace with actual LaneNet inference)
                lane_mask = self._generate_lane_mask(frame)
                
                # Save lane mask
                lane_mask_path = os.path.join(self.session_lane_masks_dir, f"lane_mask_{frame_id:06d}.png")
                cv2.imwrite(lane_mask_path, lane_mask)
                
                # Update frame data with lane mask path
                frame_data['lane_mask_path'] = lane_mask_path
                
                # Write to CSV
                if self.csv_writer:
                    self.csv_writer.writerow(frame_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
    
    def _generate_lane_mask(self, frame):
        """Generate lane detection mask (placeholder implementation)"""
        # This is a placeholder - replace with actual LaneNet inference
        # For now, just create a simple mask
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a simple lane-like pattern (replace with actual lane detection)
        center_x = width // 2
        lane_width = width // 4
        mask[:, center_x - lane_width//2:center_x + lane_width//2] = 255
        
        return mask
    
    def capture_frame(self):
        """Capture current game state and controls"""
        # Get joystick inputs
        raw_axes = []
        raw_buttons = []
        raw_hats = []
        
        # Get all axes
        for i in range(self.joystick.get_numaxes()):
            axis = self.joystick.get_axis(i)
            raw_axes.append(axis)
            
        # Get all buttons
        for i in range(self.joystick.get_numbuttons()):
            button = self.joystick.get_button(i)
            raw_buttons.append(button)
            
        # Get all hats
        for i in range(self.joystick.get_numhats()):
            hat = self.joystick.get_hat(i)
            raw_hats.append(hat)
        
        # Capture screenshot
        frame = np.array(self.sct.grab(self.bounding_box))[:,:,:3]
        screenshot_path = os.path.join(self.session_screenshot_dir, f"frame_{self.frame_count:06d}.png")
        cv2.imwrite(screenshot_path, frame)
        
        # Map inputs to driving controls (adjust based on your controller)
        # Xbox controller mapping (adjust for your controller)
        steering = raw_axes[0] if len(raw_axes) > 0 else 0.0  # Left stick X
        throttle = max(0, raw_axes[2]) if len(raw_axes) > 2 else 0.0  # Right trigger
        brake = max(0, -raw_axes[5]) if len(raw_axes) > 5 else 0.0  # Left trigger
        
        # Create data row
        data = {
            'timestamp': time.time(),
            'frame_id': self.frame_count,
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'gear': 1,  # You'll need to get this from game state
            'speed': 0.0,  # You'll need to get this from game state
            'x_pos': 0.0,  # You'll need to get this from game state
            'y_pos': 0.0,  # You'll need to get this from game state
            'heading': 0.0,  # You'll need to get this from game state
            'lap_time': 0.0,  # You'll need to get this from game state
            'lap_number': 1,  # You'll need to get this from game state
            'track_position': 0.0,  # Distance along track
            'screenshot_path': screenshot_path,
            'lane_mask_path': '',  # Will be filled by processing thread
            'raw_axes': json.dumps(raw_axes),
            'raw_buttons': json.dumps(raw_buttons),
            'raw_hats': json.dumps(raw_hats)
        }
        
        # Add frame to processing queue
        try:
            self.frame_queue.put_nowait({
                'frame': frame,
                'frame_id': self.frame_count,
                **data
            })
        except queue.Full:
            print("Warning: Frame queue full, dropping frame")
        
        self.frame_count += 1
        return data
    
    def run(self):
        """Main recording loop"""
        print("Enhanced Expert Data Collector")
        print("Press 'R' to start/stop recording, 'Q' to quit")
        print("Make sure Assetto Corsa is running and visible!")
        
        recording = False
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        if recording:
                            self.stop_recording()
                            recording = False
                            print("Recording stopped")
                        else:
                            # Get track info from user
                            track_name = input("Enter track name (or press Enter to skip): ").strip()
                            car_name = input("Enter car name (or press Enter to skip): ").strip()
                            
                            track_info = {}
                            if track_name:
                                track_info['track_name'] = track_name
                            if car_name:
                                track_info['car_name'] = car_name
                            
                            self.start_recording(track_info=track_info)
                            recording = True
                            print("Recording started")
                    elif event.key == pygame.K_q:
                        if recording:
                            self.stop_recording()
                        return
            
            # Capture frame if recording
            if recording:
                data = self.capture_frame()
                print(f"Frame {data['frame_id']}: Steering={data['steering']:.3f}, "
                      f"Throttle={data['throttle']:.3f}, Brake={data['brake']:.3f}")
            
            clock.tick(60)  # 60 FPS
        
        pygame.quit()

if __name__ == "__main__":
    collector = EnhancedExpertDataCollector()
    collector.run() 