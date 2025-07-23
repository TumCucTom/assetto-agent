import gym
import numpy as np
import cv2
import time
from mss import mss
from PIL import Image
import ctypes
import struct
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Import existing vJoy interface
from control_test import vJoy

class AssettoCorsaEnvironment(gym.Env):
    """
    Assetto Corsa Environment wrapper for RL training
    Adapts the gym interface to work with the real Assetto Corsa game
    """
    
    def __init__(self, 
                 screen_region=(0, 0, 1920, 1080),  # Default 1080p
                 resize_dim=(84, 84),  # Match gym CarRacing dimensions
                 frame_stack=4,
                 action_repeat=4,
                 max_steps=1000):
        
        super(AssettoCorsaEnvironment, self).__init__()
        
        # Screen capture setup
        self.sct = mss()
        self.screen_region = screen_region
        self.resize_dim = resize_dim
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.max_steps = max_steps
        
        # Initialize vJoy for game control
        self.vjoy = vJoy()
        if not self.vjoy.open():
            raise RuntimeError("Failed to open vJoy device")
        
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
        # Capture screen region
        screenshot = self.sct.grab(self.screen_region)
        
        # Convert to numpy array
        frame = np.array(screenshot)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target dimensions
        frame = cv2.resize(frame, self.resize_dim)
        
        return frame
    
    def _get_observation(self):
        """Get current observation (stacked frames)"""
        return np.array(self.frame_buffer, dtype=np.uint8)
    
    def _apply_action(self, action):
        """Apply action to game via vJoy"""
        # Action mapping (same as gym CarRacing)
        action_map = {
            0: [-1.0, 0.0, 0.0],  # Steer left
            1: [1.0, 0.0, 0.0],   # Steer right
            2: [0.0, 1.0, 0.0],   # Accelerate
            3: [0.0, 0.0, 0.8],   # Brake
            4: [0.0, 0.0, 0.0]    # Do nothing
        }
        
        steering, throttle, brake = action_map[action]
        
        # Convert to vJoy format (-1 to 1 -> 0 to 32767)
        steering_vjoy = int((steering + 1) * 16383.5)
        throttle_vjoy = int(throttle * 32767)
        brake_vjoy = int(brake * 32767)
        
        # Send to vJoy
        joy_pos = self.vjoy.generateJoystickPosition(
            wAxisX=steering_vjoy,      # Steering
            wThrottle=throttle_vjoy,   # Throttle
            wRudder=brake_vjoy         # Brake
        )
        self.vjoy.update(joy_pos)
        
        self.last_action = action
    
    def _calculate_reward(self):
        """Calculate reward based on game state"""
        # This is a simplified reward function
        # In practice, you'd want to extract more game state info
        
        # Basic reward: encourage forward movement and staying on track
        # This is a placeholder - you'll need to implement proper reward calculation
        reward = 0.1  # Small positive reward for each step
        
        # Penalize for going off track (you'd detect this from screen analysis)
        # reward -= 1.0 if off_track else 0
        
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        # Episode ends if:
        # 1. Max steps reached
        # 2. Car goes off track
        # 3. Car crashes
        # 4. Lap completed
        
        if self.step_count >= self.max_steps:
            return True
        
        # Add other termination conditions based on game state
        # This is a placeholder - implement based on your needs
        
        return False
    
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        
        # Wait a moment for game to reset (if needed)
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
            'step': self.step_count
        }
        
        return observation, reward, done, info
    
    def close(self):
        """Clean up resources"""
        if self.vjoy.acquired:
            self.vjoy.close()
        self.sct.close()
    
    def render(self, mode='human'):
        """Render current state"""
        if mode == 'human':
            # Show current frame
            current_frame = self.frame_buffer[-1]
            cv2.imshow('Assetto Corsa RL Agent', current_frame)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.frame_buffer[-1]
        return None


def create_assetto_corsa_env(screen_region=(0, 0, 1920, 1080)):
    """Factory function to create Assetto Corsa environment"""
    def make_env():
        return AssettoCorsaEnvironment(screen_region=screen_region)
    return make_env


# Test the environment
if __name__ == "__main__":
    # Create environment
    env = AssettoCorsaEnvironment()
    
    # Test environment compatibility
    check_env(env)
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.3f}, Done={done}")
        
        if done:
            break
    
    env.close()
    print("Environment test completed!") 