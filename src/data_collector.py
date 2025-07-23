import pygame
import csv
import time
import os
from datetime import datetime
import numpy as np
from mss import mss
import cv2

class ExpertDataCollector:
    def __init__(self, output_dir="expert_data"):
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None
        self.screenshot_dir = None
        self.frame_count = 0
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.screenshot_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
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
        
        print(f"Joystick: {self.joystick.get_name()}")
        print(f"Axes: {self.joystick.get_numaxes()}")
        print(f"Buttons: {self.joystick.get_numbuttons()}")
        
    def start_recording(self, session_name=None):
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = os.path.join(self.output_dir, f"{session_name}.csv")
        self.csv_file = open(csv_path, 'w', newline='')
        
        # Define CSV headers
        headers = [
            'timestamp', 'frame_id', 'steering', 'throttle', 'brake', 
            'gear', 'speed', 'x_pos', 'y_pos', 'heading', 'lap_time',
            'screenshot_path'
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
        self.csv_writer.writeheader()
        
        print(f"Started recording to {csv_path}")
        
    def stop_recording(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print("Recording stopped")
        
    def capture_frame(self):
        """Capture current game state and controls"""
        # Get joystick inputs
        inputs = []
        
        # Get all axes (steering, throttle, brake, etc.)
        for i in range(self.joystick.get_numaxes()):
            axis = self.joystick.get_axis(i)
            inputs.append(axis)
            
        # Get all buttons
        for i in range(self.joystick.get_numbuttons()):
            button = self.joystick.get_button(i)
            inputs.append(button)
            
        # Get all hats
        for i in range(self.joystick.get_numhats()):
            hat = self.joystick.get_hat(i)
            inputs.append(hat)
        
        # Capture screenshot
        frame = np.array(self.sct.grab(self.bounding_box))[:,:,:3]
        screenshot_path = os.path.join(self.screenshot_dir, f"frame_{self.frame_count:06d}.png")
        cv2.imwrite(screenshot_path, frame)
        
        # Map inputs to driving controls (adjust based on your controller)
        # This mapping may need adjustment based on your specific controller
        steering = inputs[0] if len(inputs) > 0 else 0.0  # Left stick X
        throttle = max(0, inputs[2]) if len(inputs) > 2 else 0.0  # Right trigger
        brake = max(0, -inputs[5]) if len(inputs) > 5 else 0.0  # Left trigger
        
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
            'screenshot_path': screenshot_path
        }
        
        # Write to CSV
        if self.csv_writer:
            self.csv_writer.writerow(data)
        
        self.frame_count += 1
        return data
    
    def run(self):
        """Main recording loop"""
        print("Starting data collection...")
        print("Press 'R' to start/stop recording, 'Q' to quit")
        
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
                            self.start_recording()
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
    collector = ExpertDataCollector()
    collector.run() 