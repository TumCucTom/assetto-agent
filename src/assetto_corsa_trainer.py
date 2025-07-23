import gym
import os
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import tensorboard
from datetime import datetime
import argparse
import sys

# Import our custom modules
from assetto_corsa_env import AssettoCorsaEnvironment
from enhanced_assetto_corsa_env import EnhancedAssettoCorsaEnvironment
from reward_calculator import AssettoCorsaRewardCalculator

# Training configuration constants (same as DeepRL-CarRacing)
WIDTH = 84
HEIGHT = 84
STACKED_FRAMES = 4
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
TIME_STEPS = 5000000  # 5 million steps
NUM_EPISODES = 10
LEARNING_STARTS = 5000

class AssettoCorsaEnvironmentWrappers:
    """
    Environment wrappers for Assetto Corsa (same approach as DeepRL-CarRacing)
    """
    
    def __init__(self, width, height, num_stacked_frames):
        self.width = width
        self.height = height
        self.num_stacked = num_stacked_frames
    
    def resize(self, environment):
        """Resize observation to target dimensions"""
        return gym.wrappers.ResizeObservation(environment, (self.width, self.height))
    
    def grayscale(self, environment):
        """Convert RGB to grayscale"""
        return gym.wrappers.GrayScaleObservation(environment)
    
    def frame_stack(self, environment):
        """Stack consecutive frames for temporal information"""
        return gym.wrappers.FrameStack(environment, self.num_stacked)
    
    def observation_wrapper(self, environment, functions):
        """Apply multiple wrappers in sequence"""
        from functools import reduce
        return reduce(lambda a, x: x(a), functions, environment)

class AssettoCorsaRewardWrapper(gym.RewardWrapper):
    """
    Reward wrapper for Assetto Corsa (same as DeepRL-CarRacing approach)
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # Clip reward between -1 to 1 (same as DeepRL-CarRacing)
        return np.clip(reward, -1, 1)

def create_assetto_corsa_env(screen_region=(0, 0, 1920, 1080), 
                           use_enhanced=False,
                           use_telemetry=False,
                           use_lanenet=False,
                           use_road_seg=False):
    """
    Create Assetto Corsa environment with proper wrappers
    """
    # Create base environment
    if use_enhanced:
        env = EnhancedAssettoCorsaEnvironment(
            screen_region=screen_region,
            use_telemetry=use_telemetry,
            use_lanenet=use_lanenet,
            use_road_seg=use_road_seg
        )
    else:
        env = AssettoCorsaEnvironment(screen_region=screen_region)
    
    # Apply environment wrappers (same as DeepRL-CarRacing)
    env_wrappers = AssettoCorsaEnvironmentWrappers(WIDTH, HEIGHT, STACKED_FRAMES)
    
    # Apply wrappers in sequence
    funcs = [
        env_wrappers.resize,      # Resize to 84x84
        env_wrappers.grayscale,   # Convert to grayscale
        env_wrappers.frame_stack  # Stack 4 consecutive frames
    ]
    env = env_wrappers.observation_wrapper(env, funcs)
    
    # Apply reward wrapper
    env = AssettoCorsaRewardWrapper(env)
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    return env

def create_assetto_corsa_vec_env(screen_region=(0, 0, 1920, 1080), 
                                num_envs=1,
                                use_enhanced=False,
                                use_telemetry=False,
                                use_lanenet=False,
                                use_road_seg=False):
    """Create vectorized Assetto Corsa environment"""
    def make_env():
        return create_assetto_corsa_env(
            screen_region=screen_region,
            use_enhanced=use_enhanced,
            use_telemetry=use_telemetry,
            use_lanenet=use_lanenet,
            use_road_seg=use_road_seg
        )
    
    if num_envs == 1:
        return DummyVecEnv([make_env])
    else:
        return DummyVecEnv([make_env for _ in range(num_envs)])

def train_model(model_name, env, log_path, model_path, num_steps, 
                buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE):
    """
    Train model using same approach as DeepRL-CarRacing
    """
    print(f"Training {model_name} model...")
    
    if model_name == "DQN":
        model = DQN(
            'CnnPolicy',           # CNN policy for image input
            env,                   # Environment
            verbose=1,             # Verbose output
            device='cuda',         # GPU acceleration
            buffer_size=buffer_size,  # Replay buffer size
            batch_size=batch_size,    # Batch size
            tensorboard_log=log_path,
            learning_starts=LEARNING_STARTS,  # Start learning after 5k steps
            gamma=0.99,            # Discount factor
            train_freq=4,          # Train every 4 steps
            gradient_steps=1,      # Gradient steps per update
            target_update_interval=1000,  # Update target network every 1000 steps
            exploration_fraction=0.1,     # Exploration fraction
            exploration_initial_eps=1.0,  # Initial epsilon
            exploration_final_eps=0.05    # Final epsilon
        )
    elif model_name == "PPO":
        model = PPO(
            'CnnPolicy',           # CNN policy for image input
            env,                   # Environment
            verbose=1,             # Verbose output
            device='cuda',         # GPU acceleration
            tensorboard_log=log_path,
            learning_rate=3e-4,    # Learning rate
            n_steps=2048,          # Steps per update
            batch_size=64,         # Batch size
            n_epochs=10,           # Number of epochs
            gamma=0.99,            # Discount factor
            gae_lambda=0.95,       # GAE lambda
            clip_range=0.2,        # Clip range
            ent_coef=0.01         # Entropy coefficient
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Setup save paths
    save_best_path = os.path.join(model_path, model_name, 'best_model')
    save_final_path = os.path.join(model_path, model_name, 'final_model', 'final_model.zip')
    
    print(f"Best model save path: {save_best_path}")
    print(f"Final model save path: {save_final_path}")
    
    # Create evaluation callback (same as DeepRL-CarRacing)
    eval_callback = EvalCallback(
        eval_env=model.get_env(),
        best_model_save_path=save_best_path,
        log_path=os.path.join(model_path, model_name, 'eval_logs'),
        n_eval_episodes=5,        # Evaluate on 5 episodes
        eval_freq=50000,          # Evaluate every 50k steps
        verbose=1,
        deterministic=True,       # Use deterministic actions for evaluation
        render=False
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,         # Save every 100k steps
        save_path=os.path.join(model_path, model_name, 'checkpoints'),
        name_prefix=f"{model_name.lower()}_model"
    )
    
    # Train the model
    print(f"Starting training for {num_steps} timesteps...")
    model.learn(
        total_timesteps=num_steps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"{model_name.lower()}_training"
    )
    
    # Save final model
    model.save(save_final_path)
    print(f"Training completed. Final model saved to: {save_final_path}")
    
    return save_final_path

def resume_train_model(model_name, log_path, model_path, env, num_steps):
    """
    Resume training from saved model (same as DeepRL-CarRacing)
    """
    print(f"Resuming {model_name} training...")
    
    # Load the best model
    best_model_path = os.path.join(model_path, model_name, "best_model.zip")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")
    
    # Load model
    if model_name == "DQN":
        model = DQN.load(best_model_path, tensorboard_log=log_path)
    elif model_name == "PPO":
        model = PPO.load(best_model_path, tensorboard_log=log_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    print(f"Loaded {model_name} model from: {best_model_path}")
    
    # Set environment
    model.set_env(env)
    
    # Setup save paths
    save_best_path = os.path.join(model_path, model_name, "best_model")
    save_final_path = os.path.join(model_path, model_name, "final_model", "final_model.zip")
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env=model.get_env(),
        best_model_save_path=save_best_path,
        log_path=os.path.join(model_path, model_name, 'eval_logs'),
        n_eval_episodes=5,
        eval_freq=50000,
        verbose=1,
        deterministic=True,
        render=False
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(model_path, model_name, 'checkpoints'),
        name_prefix=f"{model_name.lower()}_model"
    )
    
    # Continue training
    print(f"Continuing training for {num_steps} timesteps...")
    model.learn(
        total_timesteps=num_steps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False,  # Don't reset timestep counter
        tb_log_name=f"{model_name.lower()}_resume_training"
    )
    
    # Save final model
    model.save(save_final_path)
    print(f"Resume training completed. Final model saved to: {save_final_path}")
    
    return save_final_path

def test_model(env, model_name, model_path, episodes=NUM_EPISODES):
    """
    Test model performance (same as DeepRL-CarRacing)
    """
    print(f"Testing {model_name} model...")
    
    # Load the best model
    best_model_path = os.path.join(model_path, model_name, "best_model.zip")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")
    
    print(f"Loading model from: {best_model_path}")
    
    # Load model
    if model_name == "DQN":
        model = DQN.load(best_model_path, env)
    elif model_name == "PPO":
        model = PPO.load(best_model_path, env)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Set environment
    model.set_env(env)
    env = model.get_env()
    
    # Test episodes
    total_rewards = []
    
    for episode in range(1, episodes + 1):
        print(f"\nEpisode {episode}/{episodes}")
        
        obs = env.reset()
        done = False
        score = 0
        step_count = 0
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            score += reward
            step_count += 1
            
            # Print progress
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Score={score:.3f}")
        
        total_rewards.append(score)
        print(f"Episode {episode} completed: Score={score:.3f}, Steps={step_count}")
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nTest Results:")
    print(f"  Average Score: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"  Best Episode: {max(total_rewards):.3f}")
    print(f"  Worst Episode: {min(total_rewards):.3f}")
    
    return total_rewards

def fine_tune_from_gym_model(gym_model_path, env, model_name="DQN", 
                           fine_tune_steps=1000000, log_path=None, model_path=None):
    """
    Fine-tune a pre-trained gym model on Assetto Corsa
    """
    print(f"Fine-tuning {model_name} model from gym...")
    
    # Load the gym model
    if not os.path.exists(gym_model_path):
        raise FileNotFoundError(f"Gym model not found at: {gym_model_path}")
    
    if model_name == "DQN":
        model = DQN.load(gym_model_path, tensorboard_log=log_path)
    elif model_name == "PPO":
        model = PPO.load(gym_model_path, tensorboard_log=log_path)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    print(f"Loaded gym model from: {gym_model_path}")
    
    # Set new environment
    model.set_env(env)
    
    # Setup save paths
    save_best_path = os.path.join(model_path, f"{model_name}_finetuned", 'best_model')
    save_final_path = os.path.join(model_path, f"{model_name}_finetuned", 'final_model', 'final_model.zip')
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env=model.get_env(),
        best_model_save_path=save_best_path,
        log_path=os.path.join(model_path, f"{model_name}_finetuned", 'eval_logs'),
        n_eval_episodes=5,
        eval_freq=25000,  # More frequent evaluation for fine-tuning
        verbose=1,
        deterministic=True,
        render=False
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(model_path, f"{model_name}_finetuned", 'checkpoints'),
        name_prefix=f"{model_name.lower()}_finetuned"
    )
    
    # Fine-tune the model
    print(f"Starting fine-tuning for {fine_tune_steps} timesteps...")
    model.learn(
        total_timesteps=fine_tune_steps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False,  # Don't reset timestep counter
        tb_log_name=f"{model_name.lower()}_finetuning"
    )
    
    # Save fine-tuned model
    model.save(save_final_path)
    print(f"Fine-tuning completed. Model saved to: {save_final_path}")
    
    return save_final_path

def main():
    parser = argparse.ArgumentParser(description='Train AI agent for Assetto Corsa')
    parser.add_argument('--mode', default='test', choices=['train', 'resume', 'test', 'finetune'],
                       help='Training mode')
    parser.add_argument('--agent', default='DQN', choices=['DQN', 'PPO'],
                       help='RL algorithm to use')
    parser.add_argument('--gym-model-path', type=str,
                       help='Path to pre-trained gym model (for fine-tuning)')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region to capture (x,y,width,height)')
    parser.add_argument('--time-steps', type=int, default=TIME_STEPS,
                       help='Number of training timesteps')
    parser.add_argument('--fine-tune-steps', type=int, default=1000000,
                       help='Number of fine-tuning timesteps')
    parser.add_argument('--test-episodes', type=int, default=NUM_EPISODES,
                       help='Number of test episodes')
    parser.add_argument('--log-path', type=str, default='logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--model-path', type=str, default='models',
                       help='Directory for saved models')
    parser.add_argument('--use-enhanced', action='store_true',
                       help='Use enhanced environment with telemetry/CV')
    parser.add_argument('--use-telemetry', action='store_true',
                       help='Enable telemetry data (requires enhanced env)')
    parser.add_argument('--use-lanenet', action='store_true',
                       help='Enable LaneNet (requires enhanced env)')
    parser.add_argument('--use-road-seg', action='store_true',
                       help='Enable road segmentation (requires enhanced env)')
    parser.add_argument('--buffer-size', type=int, default=REPLAY_BUFFER_SIZE,
                       help='Replay buffer size for DQN')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    # Create output directories
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    
    try:
        # Create environment
        print("Creating Assetto Corsa environment...")
        env = create_assetto_corsa_vec_env(
            screen_region=screen_region,
            use_enhanced=args.use_enhanced,
            use_telemetry=args.use_telemetry,
            use_lanenet=args.use_lanenet,
            use_road_seg=args.use_road_seg
        )
        
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Execute based on mode
        if args.mode == "train":
            saved_path = train_model(
                args.agent, env, args.log_path, args.model_path, 
                args.time_steps, args.buffer_size, args.batch_size
            )
            print(f'Model successfully trained after {args.time_steps} timesteps.')
            print(f'Results saved in: {saved_path}')
            
        elif args.mode == "resume":
            saved_path = resume_train_model(
                args.agent, args.log_path, args.model_path, env, args.time_steps
            )
            print(f'Model successfully resumed training after {args.time_steps} timesteps.')
            print(f'Results saved in: {saved_path}')
            
        elif args.mode == "test":
            test_rewards = test_model(env, args.agent, args.model_path, args.test_episodes)
            print(f'Model testing completed. Average reward: {np.mean(test_rewards):.3f}')
            
        elif args.mode == "finetune":
            if not args.gym_model_path:
                parser.error("--mode=finetune requires --gym-model-path")
            
            saved_path = fine_tune_from_gym_model(
                args.gym_model_path, env, args.agent, 
                args.fine_tune_steps, args.log_path, args.model_path
            )
            print(f'Model successfully fine-tuned after {args.fine_tune_steps} timesteps.')
            print(f'Results saved in: {saved_path}')
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main() 