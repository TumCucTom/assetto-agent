#!/usr/bin/env python3
"""
Quick Start Script for Assetto Corsa AI Training
Demonstrates the complete workflow from fine-tuning to inference
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from assetto_corsa_trainer import (
    create_assetto_corsa_vec_env,
    fine_tune_from_gym_model,
    test_model
)
from assetto_corsa_inference import AssettoCorsaInference

def setup_directories():
    """Create necessary directories"""
    dirs = ['logs', 'models', 'data', 'checkpoints']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

def check_gym_model(gym_model_path):
    """Check if gym model exists and is valid"""
    if not os.path.exists(gym_model_path):
        print(f"Warning: Gym model not found at {gym_model_path}")
        print("You can download a pre-trained model from DeepRL-CarRacing repository")
        return False
    
    print(f"Found gym model: {gym_model_path}")
    return True

def run_fine_tuning_workflow(gym_model_path, screen_region=(0, 0, 1920, 1080), 
                           fine_tune_steps=500000, test_episodes=3):
    """
    Complete fine-tuning workflow
    """
    print("=== Assetto Corsa AI Fine-tuning Workflow ===")
    
    # Setup directories
    setup_directories()
    
    # Check gym model
    if not check_gym_model(gym_model_path):
        print("Please provide a valid gym model path")
        return False
    
    try:
        # Step 1: Create environment
        print("\n1. Creating Assetto Corsa environment...")
        env = create_assetto_corsa_vec_env(
            screen_region=screen_region,
            use_enhanced=False  # Start with basic environment
        )
        
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Step 2: Fine-tune model
        print(f"\n2. Fine-tuning model for {fine_tune_steps} steps...")
        fine_tuned_path = fine_tune_from_gym_model(
            gym_model_path=gym_model_path,
            env=env,
            model_name="DQN",
            fine_tune_steps=fine_tune_steps,
            log_path="logs",
            model_path="models"
        )
        
        print(f"Fine-tuning completed! Model saved to: {fine_tuned_path}")
        
        # Step 3: Test model
        print(f"\n3. Testing fine-tuned model...")
        test_rewards = test_model(
            env=env,
            model_name="DQN_finetuned",
            model_path="models",
            episodes=test_episodes
        )
        
        avg_reward = sum(test_rewards) / len(test_rewards)
        print(f"Test completed! Average reward: {avg_reward:.3f}")
        
        # Step 4: Run inference
        print(f"\n4. Running inference on real game...")
        inference = AssettoCorsaInference(
            model_path=fine_tuned_path,
            model_type="DQN",
            screen_region=screen_region
        )
        
        # Run a few episodes
        episode_rewards, episode_lengths = inference.run_multiple_episodes(
            num_episodes=2,
            max_steps_per_episode=1000,
            render=True
        )
        
        inference.close()
        
        print("\n=== Workflow Completed Successfully! ===")
        print(f"Fine-tuned model: {fine_tuned_path}")
        print(f"Test average reward: {avg_reward:.3f}")
        print(f"Inference average reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'env' in locals():
            env.close()

def run_training_workflow(screen_region=(0, 0, 1920, 1080), 
                         training_steps=1000000, test_episodes=3):
    """
    Complete training from scratch workflow
    """
    print("=== Assetto Corsa AI Training from Scratch Workflow ===")
    
    # Setup directories
    setup_directories()
    
    try:
        # Step 1: Create environment
        print("\n1. Creating Assetto Corsa environment...")
        env = create_assetto_corsa_vec_env(
            screen_region=screen_region,
            use_enhanced=False
        )
        
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Step 2: Train model from scratch
        print(f"\n2. Training model from scratch for {training_steps} steps...")
        from assetto_corsa_trainer import train_model
        
        trained_path = train_model(
            model_name="DQN",
            env=env,
            log_path="logs",
            model_path="models",
            num_steps=training_steps
        )
        
        print(f"Training completed! Model saved to: {trained_path}")
        
        # Step 3: Test model
        print(f"\n3. Testing trained model...")
        test_rewards = test_model(
            env=env,
            model_name="DQN",
            model_path="models",
            episodes=test_episodes
        )
        
        avg_reward = sum(test_rewards) / len(test_rewards)
        print(f"Test completed! Average reward: {avg_reward:.3f}")
        
        # Step 4: Run inference
        print(f"\n4. Running inference on real game...")
        inference = AssettoCorsaInference(
            model_path="models/DQN/best_model.zip",
            model_type="DQN",
            screen_region=screen_region
        )
        
        # Run a few episodes
        episode_rewards, episode_lengths = inference.run_multiple_episodes(
            num_episodes=2,
            max_steps_per_episode=1000,
            render=True
        )
        
        inference.close()
        
        print("\n=== Training Workflow Completed Successfully! ===")
        print(f"Trained model: {trained_path}")
        print(f"Test average reward: {avg_reward:.3f}")
        print(f"Inference average reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'env' in locals():
            env.close()

def main():
    parser = argparse.ArgumentParser(description='Quick start for Assetto Corsa AI training')
    parser.add_argument('--mode', choices=['finetune', 'train'], default='finetune',
                       help='Training mode: finetune from gym model or train from scratch')
    parser.add_argument('--gym-model-path', type=str, default='gym_model.zip',
                       help='Path to pre-trained gym model (for fine-tuning)')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region to capture (x,y,width,height)')
    parser.add_argument('--training-steps', type=int, default=500000,
                       help='Number of training/fine-tuning steps')
    parser.add_argument('--test-episodes', type=int, default=3,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    print("Assetto Corsa AI Quick Start")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Screen region: {screen_region}")
    print(f"Training steps: {args.training_steps}")
    print(f"Test episodes: {args.test_episodes}")
    
    if args.mode == 'finetune':
        success = run_fine_tuning_workflow(
            gym_model_path=args.gym_model_path,
            screen_region=screen_region,
            fine_tune_steps=args.training_steps,
            test_episodes=args.test_episodes
        )
    else:
        success = run_training_workflow(
            screen_region=screen_region,
            training_steps=args.training_steps,
            test_episodes=args.test_episodes
        )
    
    if success:
        print("\n🎉 Workflow completed successfully!")
        print("\nNext steps:")
        print("1. Monitor training with TensorBoard: tensorboard --logdir logs")
        print("2. Run inference: python src/assetto_corsa_inference.py --model-path models/DQN/best_model.zip")
        print("3. Experiment with different hyperparameters")
    else:
        print("\n❌ Workflow failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 