import gym
import numpy as np
import cv2
import time
from mss import mss
from PIL import Image
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

class MacAssettoCorsaEnvironment(gym.Env):
    """
    macOS-compatible Assetto Corsa Environment wrapper
    For testing purposes - simulates the environment without actual game control
    """
    
    def __init__(self, 
                 screen_region=(0, 0, 1920, 1080),  # Default 1080p
                 resize_dim=(84, 84),  # Match gym CarRacing dimensions
                 frame_stack=4,
                 action_repeat=4,
                 max_steps=1000,
                 simulation_mode=True):  # Simulate game behavior
        
        super(MacAssettoCorsaEnvironment, self).__init__()
        
        # Configuration
        self.simulation_mode = simulation_mode
        self.screen_region = screen_region
        self.resize_dim = resize_dim
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.max_steps = max_steps
        
        # Screen capture setup (only if not in simulation mode)
        if not self.simulation_mode:
            self.sct = mss()
        else:
            self.sct = None
        
        # Action space: 5 discrete actions (same as gym CarRacing)
        self.action_space = gym.spaces.Discrete(5)
        
        # Observation space: 4 stacked 84x84 grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(frame_stack, resize_dim[0], resize_dim[1]), 
            dtype=np.uint8
        )
        
        # State tracking
        self.step_count = 0
        self.frame_buffer = []
        self.last_action = None
        
        # Simulation state (for testing without actual game)
        self.sim_speed = 0.0
        self.sim_position = [0.0, 0.0]  # x, y position
        self.sim_heading = 0.0  # heading angle
        self.sim_on_track = True
        
        # Initialize frame buffer
        self._initialize_frame_buffer()
        
    def _initialize_frame_buffer(self):
        """Initialize frame buffer with current screen state"""
        self.frame_buffer = []
        for _ in range(self.frame_stack):
            frame = self._capture_screen()
            self.frame_buffer.append(frame)
    
    def _capture_screen(self):
        """Capture and process screen"""
        if self.simulation_mode:
            # Generate synthetic frame for testing
            return self._generate_synthetic_frame()
        else:
            # Real screen capture
            screenshot = self.sct.grab(self.screen_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, self.resize_dim)
            return frame
    
    def _generate_synthetic_frame(self):
        """Generate synthetic frame for testing"""
        # Create a simple synthetic racing frame
        frame = np.random.randint(50, 150, (self.resize_dim[0], self.resize_dim[1]), dtype=np.uint8)
        
        # Add some "road" features
        # Road center line
        center_y = self.resize_dim[0] // 2
        frame[center_y-2:center_y+2, :] = 100  # Road center
        
        # Add some variation based on simulation state
        if not self.sim_on_track:
            # Add "grass" texture when off track
            frame += np.random.randint(-20, 20, frame.shape, dtype=np.uint8)
        
        # Add some "speed lines" based on speed
        if self.sim_speed > 50:
            for i in range(0, self.resize_dim[1], 20):
                frame[:, i:i+2] = 120
        
        return frame
    
    def _get_observation(self):
        """Get current observation (stacked frames)"""
        return np.array(self.frame_buffer, dtype=np.uint8)
    
    def _apply_action(self, action):
        """Apply action (simulated on Mac)"""
        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        # Action mapping (same as gym CarRacing)
        action_map = {
            0: [-1.0, 0.0, 0.0],  # Steer left
            1: [1.0, 0.0, 0.0],   # Steer right
            2: [0.0, 1.0, 0.0],   # Accelerate
            3: [0.0, 0.0, 0.8],   # Brake
            4: [0.0, 0.0, 0.0]    # Do nothing
        }
        
        steering, throttle, brake = action_map[action]
        
        # Update simulation state
        self._update_simulation_state(steering, throttle, brake)
        
        self.last_action = action
    
    def _update_simulation_state(self, steering, throttle, brake):
        """Update simulation state based on actions"""
        # Update speed
        if throttle > 0:
            self.sim_speed = min(200, self.sim_speed + throttle * 5)
        if brake > 0:
            self.sim_speed = max(0, self.sim_speed - brake * 10)
        
        # Update position and heading
        if self.sim_speed > 0:
            # Update heading based on steering
            self.sim_heading += steering * 0.1
            
            # Update position based on heading and speed
            speed_factor = self.sim_speed / 100.0
            self.sim_position[0] += np.cos(self.sim_heading) * speed_factor
            self.sim_position[1] += np.sin(self.sim_heading) * speed_factor
            
            # Simple track boundary check
            track_width = 10.0
            if abs(self.sim_position[0]) > track_width or abs(self.sim_position[1]) > track_width:
                self.sim_on_track = False
            else:
                self.sim_on_track = True
    
    def _calculate_reward(self):
        """Calculate reward based on simulation state"""
        reward = 0.0
        
        # Basic survival reward
        reward += 0.1
        
        # Speed reward
        if self.sim_speed > 50:
            reward += 0.1
        elif self.sim_speed > 20:
            reward += 0.05
        
        # Track staying reward
        if self.sim_on_track:
            reward += 0.1
        else:
            reward -= 0.2
        
        # Progress reward (moving forward)
        if self.sim_position[1] > 0:  # Moving in positive Y direction
            reward += 0.05
        
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        # Episode ends if:
        # 1. Max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # 2. Car goes too far off track
        if not self.sim_on_track and self.step_count > 50:
            return True
        
        # 3. Car stops for too long
        if self.sim_speed < 5 and self.step_count > 100:
            return True
        
        return False
    
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        
        # Reset simulation state
        self.sim_speed = 0.0
        self.sim_position = [0.0, 0.0]
        self.sim_heading = 0.0
        self.sim_on_track = True
        
        # Wait a moment for game to reset (if needed)
        if not self.simulation_mode:
            time.sleep(0.1)
        
        # Capture initial frames
        self._initialize_frame_buffer()
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return next state"""
        self.step_count += 1
        
        # Apply action multiple times (action repeat)
        for _ in range(self.action_repeat):
            self._apply_action(action)
            if not self.simulation_mode:
                time.sleep(0.016)  # ~60 FPS
        
        # Capture new frame
        new_frame = self._capture_screen()
        
        # Update frame buffer
        self.frame_buffer.pop(0)
        self.frame_buffer.append(new_frame)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self._is_done()
        
        # Info dict
        info = {
            'action': action,
            'step': self.step_count,
            'speed': self.sim_speed,
            'position': self.sim_position,
            'on_track': self.sim_on_track
        }
        
        return observation, reward, done, info
    
    def close(self):
        """Clean up resources"""
        if self.sct:
            self.sct.close()
    
    def render(self, mode='human'):
        """Render current state"""
        if mode == 'human':
            # Show current frame
            current_frame = self.frame_buffer[-1]
            cv2.imshow('Mac Assetto Corsa RL Agent', current_frame)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.frame_buffer[-1]
        return None


def create_mac_assetto_corsa_env(screen_region=(0, 0, 1920, 1080), simulation_mode=True):
    """Factory function to create Mac-compatible Assetto Corsa environment"""
    def make_env():
        return MacAssettoCorsaEnvironment(screen_region=screen_region, simulation_mode=simulation_mode)
    return make_env


# Test the environment
if __name__ == "__main__":
    # Create environment in simulation mode
    env = MacAssettoCorsaEnvironment(simulation_mode=True)
    
    # Test environment compatibility
    check_env(env)
    
    print("Mac Assetto Corsa environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.3f}, Speed={info['speed']:.1f}, On Track={info['on_track']}, Done={done}")
        
        if done:
            break
    
    env.close()
    print("Mac Assetto Corsa environment test completed!") 