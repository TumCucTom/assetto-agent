import gym
import os
import numpy as np
import torch
import cv2
import time
from stable_baselines3 import DQN, PPO
import argparse
import sys

# Import our custom modules
from assetto_corsa_env import AssettoCorsaEnvironment
from enhanced_assetto_corsa_env import EnhancedAssettoCorsaEnvironment
from assetto_corsa_trainer import create_assetto_corsa_env, create_assetto_corsa_vec_env

class AssettoCorsaInference:
    """
    Inference engine for Assetto Corsa AI agent
    Runs trained models on the real game
    """
    
    def __init__(self, model_path, model_type="DQN", 
                 screen_region=(0, 0, 1920, 1080),
                 use_enhanced=False,
                 use_telemetry=False,
                 use_lanenet=False,
                 use_road_seg=False,
                 render_mode='human'):
        
        self.model_path = model_path
        self.model_type = model_type
        self.screen_region = screen_region
        self.use_enhanced = use_enhanced
        self.use_telemetry = use_telemetry
        self.use_lanenet = use_lanenet
        self.use_road_seg = use_road_seg
        self.render_mode = render_mode
        
        # Load model
        self.model = self._load_model()
        
        # Create environment
        self.env = self._create_environment()
        
        # Set model environment
        self.model.set_env(self.env)
        
        # Performance tracking
        self.episode_count = 0
        self.total_reward = 0
        self.step_count = 0
        self.start_time = None
        
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading {self.model_type} model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        if self.model_type == "DQN":
            model = DQN.load(self.model_path)
        elif self.model_type == "PPO":
            model = PPO.load(self.model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model loaded successfully!")
        return model
    
    def _create_environment(self):
        """Create Assetto Corsa environment"""
        print("Creating Assetto Corsa environment...")
        
        # Create environment with proper wrappers
        env = create_assetto_corsa_env(
            screen_region=self.screen_region,
            use_enhanced=self.use_enhanced,
            use_telemetry=self.use_telemetry,
            use_lanenet=self.use_lanenet,
            use_road_seg=self.use_road_seg
        )
        
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        return env
    
    def run_episode(self, max_steps=2000, render=True):
        """Run a single episode"""
        print(f"\n=== Starting Episode {self.episode_count + 1} ===")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        self.start_time = time.time()
        
        while not done and step_count < max_steps:
            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            step_count += 1
            self.step_count += 1
            
            # Render if requested
            if render:
                self.env.render(mode=self.render_mode)
            
            # Print progress
            if step_count % 100 == 0:
                elapsed_time = time.time() - self.start_time
                fps = step_count / elapsed_time if elapsed_time > 0 else 0
                print(f"  Step {step_count}: Reward={reward:.3f}, Total={episode_reward:.3f}, FPS={fps:.1f}")
            
            # Add small delay for real-time simulation
            time.sleep(0.016)  # ~60 FPS
        
        # Episode completed
        elapsed_time = time.time() - self.start_time
        fps = step_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Episode {self.episode_count + 1} completed:")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {episode_reward:.3f}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Duration: {elapsed_time:.1f}s")
        
        # Update tracking
        self.episode_count += 1
        self.total_reward += episode_reward
        
        return episode_reward, step_count, done
    
    def run_multiple_episodes(self, num_episodes=5, max_steps_per_episode=2000, render=True):
        """Run multiple episodes"""
        print(f"Running {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            reward, length, done = self.run_episode(max_steps_per_episode, render)
            episode_rewards.append(reward)
            episode_lengths.append(length)
            
            # Small break between episodes
            if episode < num_episodes - 1:
                print("Waiting 3 seconds before next episode...")
                time.sleep(3)
        
        # Print summary
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\n=== Inference Summary ===")
        print(f"Episodes completed: {num_episodes}")
        print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"Average episode length: {avg_length:.1f} steps")
        print(f"Best episode: {max(episode_rewards):.3f}")
        print(f"Worst episode: {min(episode_rewards):.3f}")
        print(f"Total steps: {self.step_count}")
        print(f"Total reward: {self.total_reward:.3f}")
        
        return episode_rewards, episode_lengths
    
    def run_continuous(self, max_episodes=None, max_steps_per_episode=2000, render=True):
        """Run continuously until interrupted"""
        print("Running continuous inference (press Ctrl+C to stop)...")
        
        episode_rewards = []
        episode_lengths = []
        episode = 0
        
        try:
            while max_episodes is None or episode < max_episodes:
                reward, length, done = self.run_episode(max_steps_per_episode, render)
                episode_rewards.append(reward)
                episode_lengths.append(length)
                episode += 1
                
                # Small break between episodes
                print("Waiting 3 seconds before next episode...")
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\nInference stopped by user.")
        
        # Print summary
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            avg_length = np.mean(episode_lengths)
            
            print(f"\n=== Continuous Inference Summary ===")
            print(f"Episodes completed: {len(episode_rewards)}")
            print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"Average episode length: {avg_length:.1f} steps")
            print(f"Best episode: {max(episode_rewards):.3f}")
            print(f"Worst episode: {min(episode_rewards):.3f}")
        
        return episode_rewards, episode_lengths
    
    def get_model_info(self):
        """Get information about the loaded model"""
        info = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'observation_space': str(self.env.observation_space),
            'action_space': str(self.env.action_space),
            'episode_count': self.episode_count,
            'total_steps': self.step_count,
            'total_reward': self.total_reward
        }
        
        return info
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'env'):
            self.env.close()
        print("Inference engine closed.")

def main():
    parser = argparse.ArgumentParser(description='Run AI agent inference on Assetto Corsa')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--model-type', default='DQN', choices=['DQN', 'PPO'],
                       help='Type of model (DQN or PPO)')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region to capture (x,y,width,height)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum steps per episode')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously until interrupted')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--use-enhanced', action='store_true',
                       help='Use enhanced environment with telemetry/CV')
    parser.add_argument('--use-telemetry', action='store_true',
                       help='Enable telemetry data (requires enhanced env)')
    parser.add_argument('--use-lanenet', action='store_true',
                       help='Enable LaneNet (requires enhanced env)')
    parser.add_argument('--use-road-seg', action='store_true',
                       help='Enable road segmentation (requires enhanced env)')
    parser.add_argument('--render-mode', default='human', choices=['human', 'rgb_array'],
                       help='Rendering mode')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    try:
        # Create inference engine
        inference = AssettoCorsaInference(
            model_path=args.model_path,
            model_type=args.model_type,
            screen_region=screen_region,
            use_enhanced=args.use_enhanced,
            use_telemetry=args.use_telemetry,
            use_lanenet=args.use_lanenet,
            use_road_seg=args.use_road_seg,
            render_mode=args.render_mode
        )
        
        # Print model info
        model_info = inference.get_model_info()
        print("\n=== Model Information ===")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        # Run inference
        if args.continuous:
            inference.run_continuous(max_steps_per_episode=args.max_steps, 
                                   render=not args.no_render)
        else:
            inference.run_multiple_episodes(
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                render=not args.no_render
            )
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'inference' in locals():
            inference.close()

if __name__ == "__main__":
    main() 