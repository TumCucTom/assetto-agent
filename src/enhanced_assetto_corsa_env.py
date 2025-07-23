import gym
import numpy as np
import cv2
import time
from mss import mss
from PIL import Image
import ctypes
import struct
import json
import socket
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Import existing vJoy interface
from control_test import vJoy

# Import computer vision models
from lanenet.lanenet_model import lanenet
from lanenet.lanenet_postprocess import lanenet_postprocess
import tensorflow as tf

class EnhancedAssettoCorsaEnvironment(gym.Env):
    """
    Enhanced Assetto Corsa Environment with telemetry and computer vision
    Incorporates:
    - Telemetry data (tire temps, gear, speed, etc.)
    - LaneNet for lane detection
    - Road segmentation for track understanding
    - Enhanced observation space with multiple modalities
    """
    
    def __init__(self, 
                 screen_region=(0, 0, 1920, 1080),
                 resize_dim=(84, 84),
                 frame_stack=4,
                 action_repeat=4,
                 max_steps=1000,
                 use_telemetry=True,
                 use_lanenet=True,
                 use_road_seg=True,
                 telemetry_port=9996,
                 lanenet_model_path='lanenet/model/tusimple_lanenet.ckpt',
                 road_seg_model_path='final-road-seg-model-v2.h5'):
        
        super(EnhancedAssettoCorsaEnvironment, self).__init__()
        
        # Configuration flags
        self.use_telemetry = use_telemetry
        self.use_lanenet = use_lanenet
        self.use_road_seg = use_road_seg
        
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
        
        # Telemetry setup
        if self.use_telemetry:
            self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.telemetry_socket.bind(('localhost', telemetry_port))
            self.telemetry_socket.settimeout(0.1)
            self.telemetry_data = self._get_default_telemetry()
        
        # Computer vision models setup
        if self.use_lanenet:
            self.lanenet_model = self._load_lanenet_model(lanenet_model_path)
            self.lanenet_postprocessor = lanenet_postprocess()
        
        if self.use_road_seg:
            self.road_seg_model = self._load_road_seg_model(road_seg_model_path)
        
        # Enhanced action space: 5 discrete actions + gear shifting
        self.action_space = gym.spaces.Discrete(7)  # 5 basic + 2 gear actions
        
        # Enhanced observation space with multiple modalities
        self._setup_observation_space()
        
        # State tracking
        self.step_count = 0
        self.frame_buffer = []
        self.last_action = None
        self.last_telemetry = None
        self.lane_info = None
        self.road_mask = None
        
        # Initialize buffers
        self._initialize_buffers()
        
    def _setup_observation_space(self):
        """Setup observation space with multiple modalities"""
        obs_components = []
        
        # Visual component: stacked frames
        obs_components.append(gym.spaces.Box(
            low=0, high=255,
            shape=(self.frame_stack, self.resize_dim[0], self.resize_dim[1]),
            dtype=np.uint8
        ))
        
        # Telemetry component (if enabled)
        if self.use_telemetry:
            obs_components.append(gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(15,),  # Speed, RPM, gear, tire temps, etc.
                dtype=np.float32
            ))
        
        # Lane detection component (if enabled)
        if self.use_lanenet:
            obs_components.append(gym.spaces.Box(
                low=0, high=1,
                shape=(10,),  # Lane curvature, position, etc.
                dtype=np.float32
            ))
        
        # Road segmentation component (if enabled)
        if self.use_road_seg:
            obs_components.append(gym.spaces.Box(
                low=0, high=1,
                shape=(self.resize_dim[0]//4, self.resize_dim[1]//4),  # Downsampled mask
                dtype=np.float32
            ))
        
        # Use Dict space for multiple modalities
        if len(obs_components) > 1:
            self.observation_space = gym.spaces.Dict({
                'visual': obs_components[0],
                'telemetry': obs_components[1] if self.use_telemetry else gym.spaces.Box((0,), (0,)),
                'lanes': obs_components[2] if self.use_lanenet else gym.spaces.Box((0,), (0,)),
                'road_mask': obs_components[3] if self.use_road_seg else gym.spaces.Box((0,), (0,))
            })
        else:
            self.observation_space = obs_components[0]
    
    def _get_default_telemetry(self):
        """Default telemetry data structure"""
        return {
            'speed': 0.0,
            'rpm': 0.0,
            'gear': 1,
            'tire_temp_fl': 80.0,
            'tire_temp_fr': 80.0,
            'tire_temp_rl': 80.0,
            'tire_temp_rr': 80.0,
            'fuel': 100.0,
            'lap_time': 0.0,
            'lap_count': 1,
            'position': 1,
            'track_position': 0.0,
            'brake_temp': 80.0,
            'oil_temp': 90.0,
            'water_temp': 90.0
        }
    
    def _load_lanenet_model(self, model_path):
        """Load LaneNet model for lane detection"""
        try:
            # Initialize LaneNet model
            model = lanenet()
            model.load_weights(model_path)
            print("LaneNet model loaded successfully")
            return model
        except Exception as e:
            print(f"Failed to load LaneNet model: {e}")
            return None
    
    def _load_road_seg_model(self, model_path):
        """Load road segmentation model"""
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            print("Road segmentation model loaded successfully")
            return model
        except Exception as e:
            print(f"Failed to load road segmentation model: {e}")
            return None
    
    def _initialize_buffers(self):
        """Initialize all observation buffers"""
        self.frame_buffer = []
        for _ in range(self.frame_stack):
            frame = self._capture_screen()
            self.frame_buffer.append(frame)
    
    def _capture_screen(self):
        """Capture and process screen"""
        screenshot = self.sct.grab(self.screen_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.resize_dim)
        return frame
    
    def _get_telemetry_data(self):
        """Get telemetry data from Assetto Corsa"""
        if not self.use_telemetry:
            return self._get_default_telemetry()
        
        try:
            # Receive telemetry data (UDP packet)
            data, addr = self.telemetry_socket.recvfrom(1024)
            telemetry = json.loads(data.decode())
            
            # Extract relevant data
            telemetry_data = np.array([
                telemetry.get('speed', 0.0),
                telemetry.get('rpm', 0.0),
                telemetry.get('gear', 1),
                telemetry.get('tire_temp_fl', 80.0),
                telemetry.get('tire_temp_fr', 80.0),
                telemetry.get('tire_temp_rl', 80.0),
                telemetry.get('tire_temp_rr', 80.0),
                telemetry.get('fuel', 100.0),
                telemetry.get('lap_time', 0.0),
                telemetry.get('lap_count', 1),
                telemetry.get('position', 1),
                telemetry.get('track_position', 0.0),
                telemetry.get('brake_temp', 80.0),
                telemetry.get('oil_temp', 90.0),
                telemetry.get('water_temp', 90.0)
            ], dtype=np.float32)
            
            return telemetry_data
            
        except socket.timeout:
            # Return last known data if no new data
            return self.last_telemetry if self.last_telemetry is not None else np.zeros(15, dtype=np.float32)
        except Exception as e:
            print(f"Telemetry error: {e}")
            return np.zeros(15, dtype=np.float32)
    
    def _detect_lanes(self, frame):
        """Detect lanes using LaneNet"""
        if not self.use_lanenet or self.lanenet_model is None:
            return np.zeros(10, dtype=np.float32)
        
        try:
            # Preprocess frame for LaneNet
            input_frame = cv2.resize(frame, (512, 256))  # LaneNet input size
            input_frame = input_frame / 255.0
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Run LaneNet inference
            binary_seg, instance_seg = self.lanenet_model.predict(input_frame)
            
            # Post-process to get lane information
            lane_info = self.lanenet_postprocessor.postprocess(
                binary_seg[0], instance_seg[0]
            )
            
            # Extract lane features
            lane_features = self._extract_lane_features(lane_info)
            return lane_features
            
        except Exception as e:
            print(f"Lane detection error: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _extract_lane_features(self, lane_info):
        """Extract useful features from lane detection"""
        # This is a simplified feature extraction
        # In practice, you'd extract more sophisticated features
        features = np.zeros(10, dtype=np.float32)
        
        if lane_info and len(lane_info) > 0:
            # Lane curvature
            features[0] = lane_info.get('curvature', 0.0)
            # Lane position
            features[1] = lane_info.get('position', 0.0)
            # Number of lanes detected
            features[2] = len(lane_info.get('lanes', []))
            # Lane width
            features[3] = lane_info.get('width', 0.0)
            # Confidence
            features[4] = lane_info.get('confidence', 0.0)
        
        return features
    
    def _segment_road(self, frame):
        """Segment road using road segmentation model"""
        if not self.use_road_seg or self.road_seg_model is None:
            return np.zeros((self.resize_dim[0]//4, self.resize_dim[1]//4), dtype=np.float32)
        
        try:
            # Preprocess frame for road segmentation
            input_frame = cv2.resize(frame, (400, 600))  # Model input size
            input_frame = input_frame / 255.0
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Run road segmentation
            road_mask = self.road_seg_model.predict(input_frame)
            road_mask = road_mask[0, :, :, 0]  # Remove batch and channel dimensions
            
            # Downsample for observation space
            road_mask = cv2.resize(road_mask, (self.resize_dim[1]//4, self.resize_dim[0]//4))
            
            return road_mask
            
        except Exception as e:
            print(f"Road segmentation error: {e}")
            return np.zeros((self.resize_dim[0]//4, self.resize_dim[1]//4), dtype=np.float32)
    
    def _get_observation(self):
        """Get current observation with all modalities"""
        # Visual component
        visual_obs = np.array(self.frame_buffer, dtype=np.uint8)
        
        if isinstance(self.observation_space, gym.spaces.Dict):
            # Multi-modal observation
            obs = {
                'visual': visual_obs
            }
            
            if self.use_telemetry:
                obs['telemetry'] = self._get_telemetry_data()
            
            if self.use_lanenet:
                obs['lanes'] = self._detect_lanes(visual_obs[-1])  # Use latest frame
            
            if self.use_road_seg:
                obs['road_mask'] = self._segment_road(visual_obs[-1])  # Use latest frame
            
            return obs
        else:
            # Single visual observation (fallback)
            return visual_obs
    
    def _apply_action(self, action):
        """Apply enhanced action to game via vJoy"""
        # Enhanced action mapping with gear shifting
        action_map = {
            0: [-1.0, 0.0, 0.0, 0],  # Steer left
            1: [1.0, 0.0, 0.0, 0],   # Steer right
            2: [0.0, 1.0, 0.0, 0],   # Accelerate
            3: [0.0, 0.0, 0.8, 0],   # Brake
            4: [0.0, 0.0, 0.0, 0],   # Do nothing
            5: [0.0, 0.0, 0.0, 1],   # Shift up
            6: [0.0, 0.0, 0.0, -1]   # Shift down
        }
        
        steering, throttle, brake, gear_shift = action_map[action]
        
        # Convert to vJoy format
        steering_vjoy = int((steering + 1) * 16383.5)
        throttle_vjoy = int(throttle * 32767)
        brake_vjoy = int(brake * 32767)
        
        # Send to vJoy
        joy_pos = self.vjoy.generateJoystickPosition(
            wAxisX=steering_vjoy,
            wThrottle=throttle_vjoy,
            wRudder=brake_vjoy
        )
        self.vjoy.update(joy_pos)
        
        # Handle gear shifting (if supported by game)
        if gear_shift != 0:
            # This would need to be implemented based on game's gear shift mechanism
            pass
        
        self.last_action = action
    
    def _calculate_enhanced_reward(self):
        """Calculate reward using all available data"""
        reward = 0.0
        
        # Basic survival reward
        reward += 0.1
        
        # Telemetry-based rewards
        if self.use_telemetry and self.last_telemetry is not None:
            # Speed reward
            speed = self.last_telemetry[0]
            if speed > 100:
                reward += 0.2
            elif speed > 50:
                reward += 0.1
            
            # RPM reward (avoid redline)
            rpm = self.last_telemetry[1]
            if rpm > 7000:  # Redline
                reward -= 0.1
            
            # Tire temperature reward (optimal range)
            tire_temps = self.last_telemetry[3:7]
            avg_temp = np.mean(tire_temps)
            if 70 < avg_temp < 110:  # Optimal range
                reward += 0.05
        
        # Lane-based rewards
        if self.use_lanenet and self.lane_info is not None:
            # Stay in lane reward
            lane_position = self.lane_info[1]
            if abs(lane_position) < 0.1:  # Centered in lane
                reward += 0.1
            elif abs(lane_position) > 0.5:  # Far from center
                reward -= 0.2
        
        # Road segmentation rewards
        if self.use_road_seg and self.road_mask is not None:
            # On-road reward
            road_ratio = np.mean(self.road_mask)
            if road_ratio > 0.7:  # Mostly on road
                reward += 0.1
            elif road_ratio < 0.3:  # Mostly off road
                reward -= 0.3
        
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        if self.step_count >= self.max_steps:
            return True
        
        # Check telemetry for crash/off-track
        if self.use_telemetry and self.last_telemetry is not None:
            speed = self.last_telemetry[0]
            if speed < 5 and self.step_count > 100:  # Stuck
                return True
        
        return False
    
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        self.last_telemetry = None
        self.lane_info = None
        self.road_mask = None
        
        time.sleep(0.1)
        self._initialize_buffers()
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return next state"""
        self.step_count += 1
        
        # Apply action
        for _ in range(self.action_repeat):
            self._apply_action(action)
            time.sleep(0.016)
        
        # Capture new frame
        new_frame = self._capture_screen()
        self.frame_buffer.pop(0)
        self.frame_buffer.append(new_frame)
        
        # Update telemetry
        if self.use_telemetry:
            self.last_telemetry = self._get_telemetry_data()
        
        # Update lane info
        if self.use_lanenet:
            self.lane_info = self._detect_lanes(new_frame)
        
        # Update road mask
        if self.use_road_seg:
            self.road_mask = self._segment_road(new_frame)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_enhanced_reward()
        
        # Check if done
        done = self._is_done()
        
        # Info dict
        info = {
            'action': action,
            'step': self.step_count,
            'telemetry': self.last_telemetry.tolist() if self.last_telemetry is not None else None,
            'lane_info': self.lane_info.tolist() if self.lane_info is not None else None
        }
        
        return observation, reward, done, info
    
    def close(self):
        """Clean up resources"""
        if self.vjoy.acquired:
            self.vjoy.close()
        self.sct.close()
        if self.use_telemetry:
            self.telemetry_socket.close()
    
    def render(self, mode='human'):
        """Render current state"""
        if mode == 'human':
            current_frame = self.frame_buffer[-1]
            cv2.imshow('Enhanced Assetto Corsa RL Agent', current_frame)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.frame_buffer[-1]
        return None


def create_enhanced_assetto_corsa_env(screen_region=(0, 0, 1920, 1080), 
                                    use_telemetry=True, 
                                    use_lanenet=True, 
                                    use_road_seg=True):
    """Factory function to create enhanced Assetto Corsa environment"""
    def make_env():
        return EnhancedAssettoCorsaEnvironment(
            screen_region=screen_region,
            use_telemetry=use_telemetry,
            use_lanenet=use_lanenet,
            use_road_seg=use_road_seg
        )
    return make_env


# Test the enhanced environment
if __name__ == "__main__":
    # Create environment with all features
    env = EnhancedAssettoCorsaEnvironment(
        use_telemetry=True,
        use_lanenet=True,
        use_road_seg=True
    )
    
    print("Enhanced environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation: {type(obs)}")
    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.3f}, Done={done}")
        
        if done:
            break
    
    env.close()
    print("Enhanced environment test completed!") 