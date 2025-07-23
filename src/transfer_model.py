import os
import sys
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

# Add the baseline gym path to import the trained model
sys.path.append('../baseline-gym/DeepRL-CarRacing')

from assetto_corsa_env import AssettoCorsaEnvironment, create_assetto_corsa_env

def load_gym_model(model_path):
    """Load the pre-trained DQN model from gym environment"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Load the model
    model = DQN.load(model_path)
    print("Model loaded successfully!")
    
    return model

def adapt_model_for_assetto_corsa(model, assetto_env):
    """Adapt the model to work with Assetto Corsa environment"""
    print("Adapting model for Assetto Corsa...")
    
    # Set the new environment
    model.set_env(assetto_env)
    
    # The model architecture should be compatible since we designed
    # the Assetto Corsa environment to match the gym environment:
    # - Same observation space: 4 stacked 84x84 grayscale frames
    # - Same action space: 5 discrete actions
    # - Same preprocessing pipeline
    
    print("Model adapted successfully!")
    return model

def test_model_on_assetto_corsa(model, env, num_episodes=5):
    """Test the adapted model on Assetto Corsa"""
    print(f"Testing model on Assetto Corsa for {num_episodes} episodes...")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            # Execute action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Action={action}, Reward={reward:.3f}, Total={total_reward:.3f}")
            
            # Safety check
            if step_count > 2000:  # Prevent infinite loops
                print("  Max steps reached, ending episode")
                break
        
        total_rewards.append(total_reward)
        print(f"  Episode {episode + 1} completed: Total Reward={total_reward:.3f}, Steps={step_count}")
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nTest Results:")
    print(f"  Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"  Best Episode: {max(total_rewards):.3f}")
    print(f"  Worst Episode: {min(total_rewards):.3f}")
    
    return total_rewards

def save_adapted_model(model, save_path):
    """Save the adapted model"""
    print(f"Saving adapted model to: {save_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model
    model.save(save_path)
    print("Model saved successfully!")

def main():
    parser = argparse.ArgumentParser(description='Transfer DQN model from gym to Assetto Corsa')
    parser.add_argument('--gym-model-path', type=str, 
                       default='../models/DQN/best_model.zip',
                       help='Path to the pre-trained gym model')
    parser.add_argument('--save-path', type=str,
                       default='models/assetto_corsa_dqn_model.zip',
                       help='Path to save the adapted model')
    parser.add_argument('--test-episodes', type=int, default=3,
                       help='Number of episodes to test')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region to capture (x,y,width,height)')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing the model')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    try:
        # Step 1: Load the gym model
        print("=== Step 1: Loading Gym Model ===")
        gym_model = load_gym_model(args.gym_model_path)
        
        # Step 2: Create Assetto Corsa environment
        print("\n=== Step 2: Creating Assetto Corsa Environment ===")
        assetto_env = AssettoCorsaEnvironment(screen_region=screen_region)
        print("Assetto Corsa environment created successfully!")
        
        # Step 3: Adapt the model
        print("\n=== Step 3: Adapting Model ===")
        adapted_model = adapt_model_for_assetto_corsa(gym_model, assetto_env)
        
        # Step 4: Test the model (optional)
        if not args.no_test:
            print("\n=== Step 4: Testing Model ===")
            test_rewards = test_model_on_assetto_corsa(adapted_model, assetto_env, args.test_episodes)
        
        # Step 5: Save the adapted model
        print("\n=== Step 5: Saving Adapted Model ===")
        save_adapted_model(adapted_model, args.save_path)
        
        print("\n=== Transfer Complete! ===")
        print(f"Model successfully transferred and saved to: {args.save_path}")
        
        if not args.no_test:
            print("Model tested on Assetto Corsa - check the results above.")
        
    except Exception as e:
        print(f"Error during model transfer: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'assetto_env' in locals():
            assetto_env.close()

if __name__ == "__main__":
    main() 