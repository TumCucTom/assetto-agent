import cv2
import numpy as np
from PIL import Image
import re

class AssettoCorsaRewardCalculator:
    """
    Enhanced reward calculator for Assetto Corsa
    Extracts game state information from screenshots to calculate meaningful rewards
    """
    
    def __init__(self):
        # Color ranges for detecting different game elements
        self.track_colors = {
            'asphalt': [(50, 50, 50), (100, 100, 100)],  # Dark gray
            'grass': [(30, 80, 30), (80, 150, 80)],      # Green
            'sand': [(150, 150, 100), (200, 200, 150)],  # Light brown
            'wall': [(100, 100, 100), (150, 150, 150)]   # Light gray
        }
        
        # Speed detection regions (you'll need to adjust these based on your UI)
        self.speed_region = (100, 50, 200, 80)  # Example: (x, y, width, height)
        self.lap_region = (50, 50, 150, 80)     # Example: (x, y, width, height)
        
        # Reward weights
        self.reward_weights = {
            'speed': 0.1,
            'track_stay': 0.3,
            'progress': 0.2,
            'smoothness': 0.1,
            'survival': 0.1
        }
        
        # State tracking
        self.last_speed = 0
        self.last_position = None
        self.step_count = 0
        self.off_track_count = 0
        
    def calculate_reward(self, frame, action, info):
        """
        Calculate reward based on current frame and game state
        
        Args:
            frame: Current screenshot (numpy array)
            action: Current action taken
            info: Additional info from environment
            
        Returns:
            reward: Calculated reward value
            done: Whether episode should end
            info: Updated info dict
        """
        self.step_count += 1
        
        # Extract game state information
        game_state = self._extract_game_state(frame)
        
        # Calculate individual reward components
        speed_reward = self._calculate_speed_reward(game_state)
        track_reward = self._calculate_track_reward(frame, game_state)
        progress_reward = self._calculate_progress_reward(game_state)
        smoothness_reward = self._calculate_smoothness_reward(action, info)
        survival_reward = self._calculate_survival_reward(game_state)
        
        # Combine rewards
        total_reward = (
            self.reward_weights['speed'] * speed_reward +
            self.reward_weights['track_stay'] * track_reward +
            self.reward_weights['progress'] * progress_reward +
            self.reward_weights['smoothness'] * smoothness_reward +
            self.reward_weights['survival'] * survival_reward
        )
        
        # Check if episode should end
        done = self._check_episode_end(game_state)
        
        # Update info
        info.update({
            'speed': game_state.get('speed', 0),
            'off_track': game_state.get('off_track', False),
            'speed_reward': speed_reward,
            'track_reward': track_reward,
            'progress_reward': progress_reward,
            'smoothness_reward': smoothness_reward,
            'survival_reward': survival_reward,
            'total_reward': total_reward
        })
        
        return total_reward, done, info
    
    def _extract_game_state(self, frame):
        """Extract game state information from frame"""
        game_state = {}
        
        # Try to extract speed (OCR would be better, but this is a simplified approach)
        game_state['speed'] = self._extract_speed(frame)
        
        # Detect if car is on track
        game_state['off_track'] = self._detect_off_track(frame)
        
        # Extract lap information
        game_state['lap'] = self._extract_lap_info(frame)
        
        # Detect crashes or collisions
        game_state['crashed'] = self._detect_crash(frame)
        
        return game_state
    
    def _extract_speed(self, frame):
        """Extract speed from UI (simplified - would need OCR for real implementation)"""
        try:
            # This is a placeholder - in practice you'd use OCR to read the speed
            # For now, we'll use a simple heuristic based on frame brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            
            # Simple heuristic: brighter frames might indicate higher speeds
            # This is just a placeholder - you'd want real speed extraction
            estimated_speed = min(200, brightness * 2)
            
            # Smooth the speed reading
            if self.last_speed == 0:
                self.last_speed = estimated_speed
            else:
                self.last_speed = 0.9 * self.last_speed + 0.1 * estimated_speed
            
            return self.last_speed
            
        except Exception as e:
            print(f"Error extracting speed: {e}")
            return self.last_speed
    
    def _detect_off_track(self, frame):
        """Detect if car is off track"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Create mask for grass (off-track)
            grass_lower = np.array([30, 50, 50])
            grass_upper = np.array([80, 255, 255])
            grass_mask = cv2.inRange(hsv, grass_lower, grass_upper)
            
            # Create mask for sand (off-track)
            sand_lower = np.array([15, 50, 100])
            sand_upper = np.array([30, 255, 255])
            sand_mask = cv2.inRange(hsv, sand_lower, sand_upper)
            
            # Combine off-track masks
            off_track_mask = cv2.bitwise_or(grass_mask, sand_mask)
            
            # Check if significant portion of center area is off-track
            height, width = frame.shape[:2]
            center_region = off_track_mask[height//3:2*height//3, width//3:2*width//3]
            
            off_track_ratio = np.sum(center_region > 0) / center_region.size
            
            is_off_track = off_track_ratio > 0.3  # 30% threshold
            
            if is_off_track:
                self.off_track_count += 1
            else:
                self.off_track_count = max(0, self.off_track_count - 1)
            
            return is_off_track
            
        except Exception as e:
            print(f"Error detecting off-track: {e}")
            return False
    
    def _extract_lap_info(self, frame):
        """Extract lap information (placeholder)"""
        # This would use OCR to read lap counter
        # For now, return placeholder
        return {'current_lap': 1, 'total_laps': 3}
    
    def _detect_crash(self, frame):
        """Detect if car has crashed"""
        try:
            # Look for sudden changes in frame that might indicate crash
            # This is a simplified approach - you'd want more sophisticated detection
            
            # Check for very dark areas (might indicate crash screen)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            dark_pixels = np.sum(gray < 30)
            dark_ratio = dark_pixels / gray.size
            
            # Check for red areas (damage indicators)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            red_ratio = np.sum(red_mask > 0) / red_mask.size
            
            # Simple crash detection
            crashed = dark_ratio > 0.8 or red_ratio > 0.1
            
            return crashed
            
        except Exception as e:
            print(f"Error detecting crash: {e}")
            return False
    
    def _calculate_speed_reward(self, game_state):
        """Calculate reward based on speed"""
        speed = game_state.get('speed', 0)
        
        # Reward for maintaining good speed
        if speed > 100:
            return 0.1
        elif speed > 50:
            return 0.05
        else:
            return 0.0
    
    def _calculate_track_reward(self, frame, game_state):
        """Calculate reward for staying on track"""
        off_track = game_state.get('off_track', False)
        
        if off_track:
            return -0.5  # Penalty for being off track
        else:
            return 0.1   # Small reward for staying on track
    
    def _calculate_progress_reward(self, game_state):
        """Calculate reward for making progress"""
        # This would be based on lap progress, position, etc.
        # For now, give small constant reward for continuing
        return 0.05
    
    def _calculate_smoothness_reward(self, action, info):
        """Calculate reward for smooth driving"""
        # Penalize rapid action changes
        last_action = info.get('last_action', None)
        
        if last_action is not None and last_action != action:
            return -0.01  # Small penalty for action changes
        else:
            return 0.0
    
    def _calculate_survival_reward(self, game_state):
        """Calculate reward for surviving"""
        crashed = game_state.get('crashed', False)
        
        if crashed:
            return -1.0  # Large penalty for crashing
        else:
            return 0.05  # Small reward for surviving
    
    def _check_episode_end(self, game_state):
        """Check if episode should end"""
        # End episode if:
        # 1. Car crashed
        if game_state.get('crashed', False):
            return True
        
        # 2. Car has been off track for too long
        if self.off_track_count > 50:  # 50 consecutive frames off track
            return True
        
        # 3. Max steps reached (handled by environment)
        
        return False
    
    def reset(self):
        """Reset the reward calculator state"""
        self.last_speed = 0
        self.last_position = None
        self.step_count = 0
        self.off_track_count = 0


# Example usage
if __name__ == "__main__":
    # Test the reward calculator
    calculator = AssettoCorsaRewardCalculator()
    
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
    
    # Test reward calculation
    reward, done, info = calculator.calculate_reward(
        dummy_frame, 
        action=2, 
        info={'last_action': 1}
    )
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}") 