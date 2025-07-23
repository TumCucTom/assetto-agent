import cv2
import numpy as np
import pandas as pd
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import mss
import pygame
from PIL import Image
import csv

# Import our custom modules
from assetto_corsa_env import AssettoCorsaEnvironment
from enhanced_assetto_corsa_env import EnhancedAssettoCorsaEnvironment
from input import InputHandler

class ExpertDataCollector:
    """
    Collects expert driving data for imitation learning
    Captures screenshots, actions, and telemetry in the required format
    """
    
    def __init__(self, output_dir="expert_data", 
                 screen_region=(0, 0, 1920, 1080),
                 use_enhanced=False,
                 use_telemetry=False,
                 use_lanenet=False,
                 use_road_seg=False,
                 save_format="csv"):
        
        self.output_dir = Path(output_dir)
        self.screen_region = screen_region
        self.use_enhanced = use_enhanced
        self.use_telemetry = use_telemetry
        self.use_lanenet = use_lanenet
        self.use_road_seg = use_road_seg
        self.save_format = save_format
        
        # Create output directories
        self.setup_directories()
        
        # Initialize screen capture
        self.sct = mss.mss()
        
        # Initialize input handler
        self.input_handler = InputHandler()
        
        # Initialize environment (for telemetry if needed)
        if use_enhanced:
            self.env = EnhancedAssettoCorsaEnvironment(
                screen_region=screen_region,
                use_telemetry=use_telemetry,
                use_lanenet=use_lanenet,
                use_road_seg=use_road_seg
            )
        else:
            self.env = AssettoCorsaEnvironment(screen_region=screen_region)
        
        # Data collection state
        self.is_collecting = False
        self.episode_count = 0
        self.frame_count = 0
        self.session_data = []
        
        # Performance tracking
        self.start_time = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def setup_directories(self):
        """Create necessary directories for data storage"""
        # Main directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "screenshots").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Episode-specific directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        (self.session_dir / "screenshots").mkdir(exist_ok=True)
        (self.session_dir / "telemetry").mkdir(exist_ok=True)
        
        print(f"Data will be saved to: {self.session_dir}")
    
    def capture_screenshot(self):
        """Capture screenshot from the specified screen region"""
        try:
            # Capture screen region
            screenshot = self.sct.grab(self.screen_region)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            return img
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def get_input_actions(self):
        """Get current input actions from keyboard/gamepad"""
        try:
            # Get keyboard state
            keys = pygame.key.get_pressed()
            
            # Get gamepad state (if available)
            gamepad_actions = self.input_handler.get_gamepad_state()
            
            # Combine keyboard and gamepad inputs
            actions = {
                'throttle': 0.0,
                'brake': 0.0,
                'steering': 0.0,
                'gear_up': False,
                'gear_down': False,
                'clutch': 0.0
            }
            
            # Keyboard controls
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                actions['throttle'] = 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                actions['brake'] = 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                actions['steering'] = -1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                actions['steering'] = 1.0
            if keys[pygame.K_q]:
                actions['gear_down'] = True
            if keys[pygame.K_e]:
                actions['gear_up'] = True
            if keys[pygame.K_SPACE]:
                actions['clutch'] = 1.0
            
            # Gamepad controls (override keyboard if available)
            if gamepad_actions:
                actions.update(gamepad_actions)
            
            return actions
        except Exception as e:
            print(f"Error getting input actions: {e}")
            return {'throttle': 0.0, 'brake': 0.0, 'steering': 0.0, 
                   'gear_up': False, 'gear_down': False, 'clutch': 0.0}
    
    def get_telemetry_data(self):
        """Get telemetry data if enhanced environment is enabled"""
        if not self.use_enhanced or not self.use_telemetry:
            return {}
        
        try:
            # Get telemetry from environment
            telemetry = self.env.get_telemetry()
            return telemetry
        except Exception as e:
            print(f"Error getting telemetry: {e}")
            return {}
    
    def save_screenshot(self, img, episode, frame):
        """Save screenshot to file"""
        try:
            # Create filename
            filename = f"episode_{episode:03d}_frame_{frame:06d}.png"
            filepath = self.session_dir / "screenshots" / filename
            
            # Save image
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            return filename
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return None
    
    def save_metadata_csv(self, data):
        """Save metadata to CSV file"""
        try:
            csv_path = self.session_dir / "metadata" / "expert_data.csv"
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            print(f"Metadata saved to: {csv_path}")
            return csv_path
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return None
    
    def save_metadata_json(self, data):
        """Save metadata to JSON file"""
        try:
            json_path = self.session_dir / "metadata" / "expert_data.json"
            
            # Save to JSON
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Metadata saved to: {json_path}")
            return json_path
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return None
    
    def start_collection(self):
        """Start data collection"""
        print("Starting expert data collection...")
        print("Controls:")
        print("  W/Up Arrow: Throttle")
        print("  S/Down Arrow: Brake")
        print("  A/Left Arrow: Steer Left")
        print("  D/Right Arrow: Steer Right")
        print("  Q: Gear Down")
        print("  E: Gear Up")
        print("  Space: Clutch")
        print("  ESC: Stop collection")
        print("  R: Reset episode")
        
        self.is_collecting = True
        self.start_time = time.time()
        self.episode_count += 1
        self.frame_count = 0
        
        # Initialize pygame for input handling
        pygame.init()
        pygame.display.set_mode((1, 1))  # Minimal window for input handling
        
        print(f"Episode {self.episode_count} started. Press ESC to stop.")
    
    def stop_collection(self):
        """Stop data collection and save data"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        pygame.quit()
        
        # Save collected data
        if self.session_data:
            if self.save_format == "csv":
                self.save_metadata_csv(self.session_data)
            else:
                self.save_metadata_json(self.session_data)
        
        # Print summary
        elapsed_time = time.time() - self.start_time
        total_frames = len(self.session_data)
        avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n=== Data Collection Summary ===")
        print(f"Episodes: {self.episode_count}")
        print(f"Total frames: {total_frames}")
        print(f"Collection time: {elapsed_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Data saved to: {self.session_dir}")
    
    def reset_episode(self):
        """Reset current episode"""
        print(f"Episode {self.episode_count} ended. Starting new episode...")
        self.episode_count += 1
        self.frame_count = 0
        print(f"Episode {self.episode_count} started.")
    
    def collect_frame(self):
        """Collect data for a single frame"""
        if not self.is_collecting:
            return
        
        try:
            # Capture screenshot
            screenshot = self.capture_screenshot()
            if screenshot is None:
                return
            
            # Get input actions
            actions = self.get_input_actions()
            
            # Get telemetry data
            telemetry = self.get_telemetry_data()
            
            # Save screenshot
            screenshot_filename = self.save_screenshot(screenshot, self.episode_count, self.frame_count)
            
            # Create frame data
            frame_data = {
                'episode': self.episode_count,
                'frame': self.frame_count,
                'timestamp': time.time(),
                'screenshot_filename': screenshot_filename,
                'throttle': actions['throttle'],
                'brake': actions['brake'],
                'steering': actions['steering'],
                'gear_up': actions['gear_up'],
                'gear_down': actions['gear_down'],
                'clutch': actions['clutch']
            }
            
            # Add telemetry data
            frame_data.update(telemetry)
            
            # Store data
            self.session_data.append(frame_data)
            
            # Update counters
            self.frame_count += 1
            self.fps_counter += 1
            
            # Print FPS every second
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.last_fps_time)
                print(f"Episode {self.episode_count}, Frame {self.frame_count}, FPS: {fps:.1f}")
                self.fps_counter = 0
                self.last_fps_time = current_time
            
        except Exception as e:
            print(f"Error collecting frame: {e}")
    
    def run_collection_loop(self):
        """Main collection loop"""
        self.start_collection()
        
        try:
            while self.is_collecting:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.is_collecting = False
                        elif event.key == pygame.K_r:
                            self.reset_episode()
                
                # Collect frame data
                self.collect_frame()
                
                # Small delay to maintain reasonable FPS
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user.")
        
        finally:
            self.stop_collection()
    
    def get_data_summary(self):
        """Get summary of collected data"""
        if not self.session_data:
            return {}
        
        df = pd.DataFrame(self.session_data)
        
        summary = {
            'total_frames': len(self.session_data),
            'episodes': df['episode'].nunique(),
            'duration_seconds': df['timestamp'].max() - df['timestamp'].min(),
            'avg_fps': len(self.session_data) / (df['timestamp'].max() - df['timestamp'].min()),
            'throttle_stats': {
                'mean': df['throttle'].mean(),
                'std': df['throttle'].std(),
                'max': df['throttle'].max()
            },
            'brake_stats': {
                'mean': df['brake'].mean(),
                'std': df['brake'].std(),
                'max': df['brake'].max()
            },
            'steering_stats': {
                'mean': df['steering'].mean(),
                'std': df['steering'].std(),
                'max': df['steering'].max()
            }
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Collect expert driving data for imitation learning')
    parser.add_argument('--output-dir', type=str, default='expert_data',
                       help='Output directory for collected data')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region to capture (x,y,width,height)')
    parser.add_argument('--use-enhanced', action='store_true',
                       help='Use enhanced environment with telemetry/CV')
    parser.add_argument('--use-telemetry', action='store_true',
                       help='Enable telemetry data collection')
    parser.add_argument('--use-lanenet', action='store_true',
                       help='Enable LaneNet data collection')
    parser.add_argument('--use-road-seg', action='store_true',
                       help='Enable road segmentation data collection')
    parser.add_argument('--save-format', choices=['csv', 'json'], default='csv',
                       help='Format for saving metadata')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    # Create collector
    collector = ExpertDataCollector(
        output_dir=args.output_dir,
        screen_region=screen_region,
        use_enhanced=args.use_enhanced,
        use_telemetry=args.use_telemetry,
        use_lanenet=args.use_lanenet,
        use_road_seg=args.use_road_seg,
        save_format=args.save_format
    )
    
    try:
        # Run collection
        collector.run_collection_loop()
        
        # Print summary
        summary = collector.get_data_summary()
        if summary:
            print(f"\n=== Data Summary ===")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Episodes: {summary['episodes']}")
            print(f"Duration: {summary['duration_seconds']:.1f}s")
            print(f"Average FPS: {summary['avg_fps']:.1f}")
            print(f"Throttle (mean/std/max): {summary['throttle_stats']['mean']:.3f}/{summary['throttle_stats']['std']:.3f}/{summary['throttle_stats']['max']:.3f}")
            print(f"Brake (mean/std/max): {summary['brake_stats']['mean']:.3f}/{summary['brake_stats']['std']:.3f}/{summary['brake_stats']['max']:.3f}")
            print(f"Steering (mean/std/max): {summary['steering_stats']['mean']:.3f}/{summary['steering_stats']['std']:.3f}/{summary['steering_stats']['max']:.3f}")
        
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 