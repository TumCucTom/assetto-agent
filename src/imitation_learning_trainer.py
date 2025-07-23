import gym
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
from pathlib import Path
import argparse
import json
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import tensorboard
from collections import deque
import random

# Import our custom modules
from assetto_corsa_env import AssettoCorsaEnvironment
from enhanced_assetto_corsa_env import EnhancedAssettoCorsaEnvironment
from assetto_corsa_trainer import create_assetto_corsa_env, create_assetto_corsa_vec_env

class ExpertDataset(Dataset):
    """
    Dataset for expert demonstrations
    Loads screenshots and actions from collected expert data
    """
    
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_samples = max_samples
        
        # Load metadata
        self.metadata = self.load_metadata()
        
        # Filter and prepare data
        self.samples = self.prepare_samples()
        
        print(f"Loaded {len(self.samples)} expert samples from {self.data_dir}")
    
    def load_metadata(self):
        """Load metadata from CSV or JSON file"""
        # Try CSV first
        csv_path = self.data_dir / "metadata" / "expert_data.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        # Try JSON
        json_path = self.data_dir / "metadata" / "expert_data.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        
        raise FileNotFoundError(f"No metadata found in {self.data_dir}")
    
    def prepare_samples(self):
        """Prepare samples for training"""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            # Check if screenshot exists
            screenshot_path = self.data_dir / "screenshots" / row['screenshot_filename']
            if not screenshot_path.exists():
                continue
            
            # Create sample
            sample = {
                'screenshot_path': str(screenshot_path),
                'throttle': float(row['throttle']),
                'brake': float(row['brake']),
                'steering': float(row['steering']),
                'gear_up': bool(row['gear_up']),
                'gear_down': bool(row['gear_down']),
                'clutch': float(row['clutch'])
            }
            
            samples.append(sample)
            
            # Limit samples if specified
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        img = cv2.imread(sample['screenshot_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Create action vector
        action = np.array([
            sample['throttle'],
            sample['brake'],
            sample['steering'],
            float(sample['gear_up']),
            float(sample['gear_down']),
            sample['clutch']
        ], dtype=np.float32)
        
        return img, action

class BehavioralCloningTrainer:
    """
    Behavioral Cloning trainer
    Trains a policy to mimic expert demonstrations
    """
    
    def __init__(self, expert_data_dir, model_save_dir="models/bc", 
                 learning_rate=1e-4, batch_size=32, num_epochs=100):
        
        self.expert_data_dir = expert_data_dir
        self.model_save_dir = Path(model_save_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Create model directory
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model and training components
        self.setup_model()
        self.setup_data()
    
    def setup_model(self):
        """Setup the behavioral cloning model"""
        # Simple CNN for action prediction
        self.model = nn.Sequential(
            # CNN layers
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6 actions: throttle, brake, steering, gear_up, gear_down, clutch
        ).to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def setup_data(self):
        """Setup data loading"""
        # Create dataset
        self.dataset = ExpertDataset(self.expert_data_dir, max_samples=10000)
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4
        )
    
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        # Resize to 84x84 (same as DeepRL-CarRacing)
        img = cv2.resize(img, (84, 84))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, actions) in enumerate(self.dataloader):
            # Move to device
            images = images.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            predicted_actions = self.model(images)
            
            # Calculate loss
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def train(self):
        """Train the behavioral cloning model"""
        print("Starting Behavioral Cloning training...")
        
        best_loss = float('inf')
        training_history = []
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            avg_loss = self.train_epoch()
            training_history.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model("best_bc_model.pth")
                print(f"New best model saved with loss: {best_loss:.6f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"bc_model_epoch_{epoch + 1}.pth")
        
        # Save final model
        self.save_model("final_bc_model.pth")
        
        # Save training history
        history_path = self.model_save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'epochs': list(range(1, self.num_epochs + 1)),
                'losses': training_history,
                'best_loss': best_loss
            }, f, indent=2)
        
        print(f"\nTraining completed! Best loss: {best_loss:.6f}")
        return best_loss
    
    def save_model(self, filename):
        """Save model to file"""
        model_path = self.model_save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_architecture': str(self.model)
        }, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, filename):
        """Load model from file"""
        model_path = self.model_save_dir / filename
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {model_path}")

class DQfDTrainer:
    """
    Deep Q-Learning from Demonstrations trainer
    Combines imitation learning with reinforcement learning
    """
    
    def __init__(self, expert_data_dir, env, model_save_dir="models/dqfd",
                 learning_rate=1e-4, batch_size=32, num_epochs=50):
        
        self.expert_data_dir = expert_data_dir
        self.env = env
        self.model_save_dir = Path(model_save_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Create model directory
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load expert data
        self.expert_dataset = ExpertDataset(expert_data_dir, max_samples=5000)
        
        # Initialize DQN model
        self.setup_dqn_model()
    
    def setup_dqn_model(self):
        """Setup DQN model for DQfD"""
        # Use stable-baselines3 DQN
        self.model = DQN(
            'CnnPolicy',
            self.env,
            verbose=1,
            device=self.device,
            buffer_size=100000,
            batch_size=self.batch_size,
            learning_starts=1000,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=str(self.model_save_dir / "logs")
        )
    
    def preload_expert_data(self):
        """Preload expert data into replay buffer"""
        print("Preloading expert data into replay buffer...")
        
        expert_samples = 0
        for idx, (img, action) in enumerate(self.expert_dataset):
            if idx >= 5000:  # Limit expert samples
                break
            
            # Convert action to discrete action space
            discrete_action = self.action_to_discrete(action)
            
            # Add to replay buffer
            self.model.replay_buffer.add(
                obs=img,
                next_obs=img,  # For demonstration data, next_obs is same
                action=discrete_action,
                reward=1.0,  # Expert demonstrations get high reward
                done=False,
                infos={}
            )
            
            expert_samples += 1
        
        print(f"Preloaded {expert_samples} expert samples into replay buffer")
    
    def action_to_discrete(self, continuous_action):
        """Convert continuous action to discrete action space"""
        # Simple discretization based on action values
        throttle, brake, steering, gear_up, gear_down, clutch = continuous_action
        
        # Create discrete action based on dominant input
        if throttle > 0.5:
            return 0  # Forward
        elif brake > 0.5:
            return 1  # Backward
        elif steering < -0.3:
            return 2  # Left
        elif steering > 0.3:
            return 3  # Right
        elif gear_up:
            return 4  # Gear up
        elif gear_down:
            return 5  # Gear down
        else:
            return 0  # Default to forward
    
    def train_with_demonstrations(self, total_timesteps=1000000):
        """Train DQN with expert demonstrations"""
        print("Starting DQfD training...")
        
        # Preload expert data
        self.preload_expert_data()
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env=self.env,
            best_model_save_path=str(self.model_save_dir / "best_model"),
            log_path=str(self.model_save_dir / "eval_logs"),
            n_eval_episodes=5,
            eval_freq=25000,
            verbose=1,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=str(self.model_save_dir / "checkpoints"),
            name_prefix="dqfd_model"
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name="dqfd_training"
        )
        
        # Save final model
        final_model_path = self.model_save_dir / "final_model" / "final_model.zip"
        self.model.save(str(final_model_path))
        
        print(f"DQfD training completed! Model saved to: {final_model_path}")
        return str(final_model_path)

class ImitationLearningTrainer:
    """
    Main imitation learning trainer
    Combines behavioral cloning and DQfD approaches
    """
    
    def __init__(self, expert_data_dir, env, model_save_dir="models/imitation"):
        
        self.expert_data_dir = expert_data_dir
        self.env = env
        self.model_save_dir = Path(model_save_dir)
        
        # Create model directory
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trainers
        self.bc_trainer = BehavioralCloningTrainer(
            expert_data_dir=expert_data_dir,
            model_save_dir=str(self.model_save_dir / "bc")
        )
        
        self.dqfd_trainer = DQfDTrainer(
            expert_data_dir=expert_data_dir,
            env=env,
            model_save_dir=str(self.model_save_dir / "dqfd")
        )
    
    def train_behavioral_cloning(self, num_epochs=100):
        """Train behavioral cloning model"""
        print("=== Training Behavioral Cloning Model ===")
        return self.bc_trainer.train()
    
    def train_dqfd(self, total_timesteps=1000000):
        """Train DQfD model"""
        print("=== Training DQfD Model ===")
        return self.dqfd_trainer.train_with_demonstrations(total_timesteps)
    
    def train_combined(self, bc_epochs=50, dqfd_steps=500000):
        """Train both models in sequence"""
        print("=== Combined Imitation Learning Training ===")
        
        # Step 1: Behavioral Cloning
        print("\nStep 1: Behavioral Cloning")
        bc_loss = self.train_behavioral_cloning(num_epochs=bc_epochs)
        
        # Step 2: DQfD with pre-trained BC model
        print("\nStep 2: DQfD with Behavioral Cloning initialization")
        dqfd_path = self.train_dqfd(total_timesteps=dqfd_steps)
        
        print(f"\n=== Combined Training Completed ===")
        print(f"Behavioral Cloning final loss: {bc_loss:.6f}")
        print(f"DQfD model saved to: {dqfd_path}")
        
        return bc_loss, dqfd_path
    
    def evaluate_models(self, num_episodes=10):
        """Evaluate trained models"""
        print("=== Evaluating Trained Models ===")
        
        results = {}
        
        # Evaluate DQfD model
        dqfd_model_path = self.model_save_dir / "dqfd" / "best_model.zip"
        if dqfd_model_path.exists():
            print("\nEvaluating DQfD model...")
            model = DQN.load(str(dqfd_model_path), env=self.env)
            mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=num_episodes)
            results['dqfd'] = {'mean_reward': mean_reward, 'std_reward': std_reward}
            print(f"DQfD - Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
        
        # Evaluate BC model (if available)
        bc_model_path = self.model_save_dir / "bc" / "best_bc_model.pth"
        if bc_model_path.exists():
            print("\nEvaluating Behavioral Cloning model...")
            # Note: BC model evaluation would require custom evaluation
            results['bc'] = {'status': 'model_available'}
            print("Behavioral Cloning model available for evaluation")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Train imitation learning models on expert data')
    parser.add_argument('--expert-data-dir', type=str, required=True,
                       help='Directory containing expert data')
    parser.add_argument('--mode', choices=['bc', 'dqfd', 'combined'], default='combined',
                       help='Training mode')
    parser.add_argument('--screen-region', type=str, default='0,0,1920,1080',
                       help='Screen region for environment')
    parser.add_argument('--bc-epochs', type=int, default=50,
                       help='Number of epochs for behavioral cloning')
    parser.add_argument('--dqfd-steps', type=int, default=500000,
                       help='Number of timesteps for DQfD')
    parser.add_argument('--model-save-dir', type=str, default='models/imitation',
                       help='Directory to save trained models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models after training')
    
    args = parser.parse_args()
    
    # Parse screen region
    screen_region = tuple(map(int, args.screen_region.split(',')))
    
    # Create environment
    print("Creating Assetto Corsa environment...")
    env = create_assetto_corsa_vec_env(screen_region=screen_region)
    
    # Create imitation learning trainer
    trainer = ImitationLearningTrainer(
        expert_data_dir=args.expert_data_dir,
        env=env,
        model_save_dir=args.model_save_dir
    )
    
    try:
        # Train based on mode
        if args.mode == 'bc':
            trainer.train_behavioral_cloning(num_epochs=args.bc_epochs)
        elif args.mode == 'dqfd':
            trainer.train_dqfd(total_timesteps=args.dqfd_steps)
        elif args.mode == 'combined':
            trainer.train_combined(bc_epochs=args.bc_epochs, dqfd_steps=args.dqfd_steps)
        
        # Evaluate if requested
        if args.evaluate:
            results = trainer.evaluate_models()
            print(f"\nEvaluation results: {results}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()

if __name__ == "__main__":
    main() 