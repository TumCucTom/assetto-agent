import os
import sys
import argparse
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import tensorboard
from datetime import datetime

# Import our custom modules
from assetto_corsa_env import AssettoCorsaEnvironment
from reward_calculator import AssettoCorsaRewardCalculator
from transfer_model import load_gym_model, adapt_model_for_assetto_corsa

def create_assetto_corsa_vec_env(screen_region=(0, 0, 1920, 1080), num_envs=1):
    """Create vectorized Assetto Corsa environment"""
    def make_env():
        env = AssettoCorsaEnvironment(screen_region=screen_region)
        env = Monitor(env)
        return env
    
    if num_envs == 1:
        return DummyVecEnv([make_env])
    else:
        return DummyVecEnv([make_env for _ in range(num_envs)])

def train_with_imitation_learning(gym_model_path, assetto_env, 
                                imitation_steps=10000, 
                                save_path="models/imitation_model"):
    """
    Phase 1: Imitation Learning
    Use the gym model as a teacher to learn basic driving skills
    """
    print("=== Phase 1: Imitation Learning ===")
    
    # Load the gym model as teacher
    teacher_model = load_gym_model(gym_model_path)
    
    # Create student model (same architecture)
    student_model = DQN(
        'CnnPolicy',
        assetto_env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="logs/imitation_learning"
    )
    
    # Imitation learning loop
    print("Starting imitation learning...")
    
    for step in range(imitation_steps):
        # Get teacher action
        obs = assetto_env.reset()
        done = False
        
        while not done:
            # Teacher action
            teacher_action, _ = teacher_model.predict(obs, deterministic=True)
            
            # Student action
            student_action, _ = student_model.predict(obs, deterministic=False)
            
            # Execute teacher action and get reward
            obs, reward, done, info = assetto_env.step(teacher_action)
            
            # Store experience with teacher action as target
            # This is a simplified approach - you might want to use more sophisticated IL methods
            student_model.replay_buffer.add(
                obs, teacher_action, reward, obs, done
            )
            
            # Train student occasionally
            if step % 100 == 0 and len(student_model.replay_buffer) > student_model.learning_starts:
                student_model.train()
        
        if step % 1000 == 0:
            print(f"Imitation step {step}/{imitation_steps}")
    
    # Save imitation model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    student_model.save(save_path)
    print(f"Imitation model saved to {save_path}")
    
    return student_model

def train_with_reinforcement_learning(model, assetto_env, 
                                    rl_steps=50000,
                                    save_path="models/rl_model"):
    """
    Phase 2: Reinforcement Learning
    Fine-tune the model using the enhanced reward function
    """
    print("=== Phase 2: Reinforcement Learning ===")
    
    # Set up callbacks
    eval_callback = EvalCallback(
        assetto_env,
        best_model_save_path=f"{save_path}_best",
        log_path=f"{save_path}_logs",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_path}_checkpoints",
        name_prefix="rl_model"
    )
    
    # Continue training with RL
    print("Starting reinforcement learning fine-tuning...")
    model.learn(
        total_timesteps=rl_steps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False,
        tb_log_name="reinforcement_learning"
    )
    
    # Save final model
    model.save(save_path)
    print(f"RL model saved to {save_path}")
    
    return model

def evaluate_model(model, assetto_env, num_episodes=10):
    """Evaluate the trained model"""
    print(f"=== Evaluating Model ({num_episodes} episodes) ===")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        obs = assetto_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = assetto_env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Reward={reward:.3f}, Total={total_reward:.3f}")
        
        total_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        print(f"  Episode {episode + 1}: Total Reward={total_reward:.3f}, Steps={step_count}")
    
    # Print evaluation results
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"  Average Episode Length: {avg_length:.1f} steps")
    print(f"  Best Episode: {max(total_rewards):.3f}")
    print(f"  Worst Episode: {min(total_rewards):.3f}")
    
    return total_rewards, episode_lengths

def main():
    parser = argparse.ArgumentParser(description='Train AI agent for Assetto Corsa')
    parser.add_argument('--gym-model-path', type=str,
                       default='../models/DQN/best_model.zip',
                       help='Path to pre-trained gym model')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region to capture (x,y,width,height)')
    parser.add_argument('--imitation-steps', type=int, default=5000,
                       help='Number of imitation learning steps')
    parser.add_argument('--rl-steps', type=int, default=20000,
                       help='Number of reinforcement learning steps')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--skip-imitation', action='store_true',
                       help='Skip imitation learning phase')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip reinforcement learning phase')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create Assetto Corsa environment
        print("Creating Assetto Corsa environment...")
        assetto_env = create_assetto_corsa_vec_env(screen_region)
        
        # Phase 1: Imitation Learning
        if not args.skip_imitation:
            model = train_with_imitation_learning(
                args.gym_model_path,
                assetto_env,
                imitation_steps=args.imitation_steps,
                save_path=f"{args.output_dir}/imitation_model"
            )
        else:
            # Load pre-trained gym model directly
            print("Loading pre-trained gym model...")
            model = load_gym_model(args.gym_model_path)
            model = adapt_model_for_assetto_corsa(model, assetto_env)
        
        # Phase 2: Reinforcement Learning
        if not args.skip_rl:
            model = train_with_reinforcement_learning(
                model,
                assetto_env,
                rl_steps=args.rl_steps,
                save_path=f"{args.output_dir}/final_model"
            )
        
        # Evaluation
        print("Evaluating final model...")
        rewards, lengths = evaluate_model(model, assetto_env, args.eval_episodes)
        
        # Save final results
        results = {
            'rewards': rewards,
            'episode_lengths': lengths,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths)
        }
        
        np.save(f"{args.output_dir}/evaluation_results.npy", results)
        print(f"Results saved to {args.output_dir}/evaluation_results.npy")
        
        print("\n=== Training Complete! ===")
        print(f"Models saved in: {args.output_dir}")
        print(f"Average reward: {results['avg_reward']:.3f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'assetto_env' in locals():
            assetto_env.close()

if __name__ == "__main__":
    main() 