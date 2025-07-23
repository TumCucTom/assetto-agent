# Assetto Corsa AI Training Guide

This guide explains how to train and fine-tune AI agents for Assetto Corsa using the same approach as DeepRL-CarRacing.

## Overview

The training system follows the exact same methodology as DeepRL-CarRacing:
- **Environment Wrappers**: Resize (84x84), grayscale, frame stacking (4 frames)
- **Reward Clipping**: Between -1 and 1
- **Training Algorithms**: DQN and PPO with CNN policies
- **Evaluation**: Regular evaluation with best model saving
- **Checkpointing**: Automatic model saving during training

## Prerequisites

### Windows Requirements
- Assetto Corsa installed
- vJoy installed and configured
- Python 3.8+ with required packages
- NVIDIA GPU (recommended)

### Required Python Packages
```bash
pip install stable-baselines3 gym opencv-python numpy torch tensorboard
```

## Training Modes

### 1. Train from Scratch
Train a new model from scratch on Assetto Corsa:

```bash
python src/assetto_corsa_trainer.py --mode train --agent DQN --time-steps 5000000
```

### 2. Fine-tune from Gym Model
Fine-tune a pre-trained gym model on Assetto Corsa:

```bash
python src/assetto_corsa_trainer.py --mode finetune --agent DQN --gym-model-path path/to/gym_model.zip --fine-tune-steps 1000000
```

### 3. Resume Training
Resume training from a saved checkpoint:

```bash
python src/assetto_corsa_trainer.py --mode resume --agent DQN --time-steps 1000000
```

### 4. Test Model
Test a trained model:

```bash
python src/assetto_corsa_trainer.py --mode test --agent DQN --test-episodes 10
```

## Training Configuration

### Basic Training Parameters
- **Observation Size**: 84x84 (same as DeepRL-CarRacing)
- **Frame Stacking**: 4 consecutive frames
- **Grayscale**: RGB converted to grayscale
- **Reward Clipping**: -1 to 1
- **Replay Buffer**: 50,000 (DQN)
- **Batch Size**: 64
- **Learning Starts**: 5,000 steps

### DQN Specific Parameters
```python
DQN(
    'CnnPolicy',           # CNN policy for image input
    buffer_size=50000,     # Replay buffer size
    batch_size=64,         # Batch size
    learning_starts=5000,  # Start learning after 5k steps
    gamma=0.99,           # Discount factor
    train_freq=4,         # Train every 4 steps
    target_update_interval=1000,  # Update target network every 1000 steps
    exploration_fraction=0.1,     # Exploration fraction
    exploration_initial_eps=1.0,  # Initial epsilon
    exploration_final_eps=0.05    # Final epsilon
)
```

### PPO Specific Parameters
```python
PPO(
    'CnnPolicy',           # CNN policy for image input
    learning_rate=3e-4,    # Learning rate
    n_steps=2048,          # Steps per update
    batch_size=64,         # Batch size
    n_epochs=10,           # Number of epochs
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE lambda
    clip_range=0.2,       # Clip range
    ent_coef=0.01         # Entropy coefficient
)
```

## Environment Options

### Basic Environment
```bash
python src/assetto_corsa_trainer.py --mode train --agent DQN
```

### Enhanced Environment with Telemetry
```bash
python src/assetto_corsa_trainer.py --mode train --agent DQN --use-enhanced --use-telemetry
```

### Enhanced Environment with Computer Vision
```bash
python src/assetto_corsa_trainer.py --mode train --agent DQN --use-enhanced --use-lanenet --use-road-seg
```

### Custom Screen Region
```bash
python src/assetto_corsa_trainer.py --mode train --agent DQN --screen-region "0,0,1920,1080"
```

## Training Process

### 1. Environment Setup
The system automatically applies the same wrappers as DeepRL-CarRacing:
- **ResizeObservation**: Resize to 84x84
- **GrayScaleObservation**: Convert to grayscale
- **FrameStack**: Stack 4 consecutive frames
- **RewardWrapper**: Clip rewards between -1 and 1
- **Monitor**: Log episode information

### 2. Training Loop
- **Exploration**: Starts with epsilon=1.0, decays to 0.05
- **Learning**: Begins after 5,000 steps
- **Evaluation**: Every 50,000 steps on 5 episodes
- **Checkpointing**: Every 100,000 steps
- **Best Model**: Automatically saved when performance improves

### 3. Monitoring
- **TensorBoard**: Training logs saved to `logs/`
- **Evaluation Logs**: Saved to `models/{agent}/eval_logs/`
- **Checkpoints**: Saved to `models/{agent}/checkpoints/`
- **Best Model**: Saved to `models/{agent}/best_model/`

## Fine-tuning Process

### 1. Load Pre-trained Model
```python
# Load gym model
model = DQN.load("path/to/gym_model.zip")
```

### 2. Adapt to New Environment
```python
# Set new environment
model.set_env(assetto_corsa_env)
```

### 3. Fine-tune with Lower Learning Rate
- More frequent evaluation (every 25,000 steps)
- Smaller checkpoint intervals (every 50,000 steps)
- Preserve timestep counter

### 4. Transfer Learning Benefits
- Faster convergence
- Better initial performance
- More stable training

## Inference

### Run Trained Model
```bash
python src/assetto_corsa_inference.py --model-path models/DQN/best_model.zip --episodes 5
```

### Continuous Inference
```bash
python src/assetto_corsa_inference.py --model-path models/DQN/best_model.zip --continuous
```

### Enhanced Inference
```bash
python src/assetto_corsa_inference.py --model-path models/DQN/best_model.zip --use-enhanced --use-telemetry
```

## Performance Optimization

### GPU Acceleration
- Models automatically use CUDA if available
- Set `device='cuda'` for GPU training
- Monitor GPU memory usage

### Training Speed
- **Frame Rate**: Target 60 FPS for real-time training
- **Batch Processing**: Use vectorized environments for parallel training
- **Memory Management**: Monitor replay buffer size

### Hyperparameter Tuning
- **Learning Rate**: Start with default, adjust based on convergence
- **Buffer Size**: Increase for more stable training
- **Batch Size**: Adjust based on GPU memory
- **Exploration**: Tune epsilon decay for better exploration

## Troubleshooting

### Common Issues

1. **Environment Not Found**
   - Ensure Assetto Corsa is running
   - Check screen region coordinates
   - Verify vJoy installation

2. **Poor Performance**
   - Check reward function
   - Adjust exploration parameters
   - Increase training time

3. **Memory Issues**
   - Reduce batch size
   - Decrease replay buffer size
   - Use smaller observation size

4. **Training Instability**
   - Check reward clipping
   - Adjust learning rate
   - Monitor gradient norms

### Debug Mode
```bash
python src/assetto_corsa_trainer.py --mode test --agent DQN --test-episodes 1
```

## Expected Results

### Training Timeline
- **0-50k steps**: Random exploration
- **50k-200k steps**: Basic driving skills
- **200k-1M steps**: Improved lap times
- **1M+ steps**: Competitive performance

### Performance Metrics
- **Average Reward**: Should increase over time
- **Episode Length**: Should stabilize
- **Success Rate**: Percentage of completed laps
- **Lap Times**: Should decrease over time

### Model Quality Indicators
- **Consistent Performance**: Low variance in rewards
- **Smooth Driving**: No erratic behavior
- **Lap Completion**: High success rate
- **Competitive Times**: Comparable to human performance

## Advanced Features

### Multi-GPU Training
```python
# Use multiple GPUs for faster training
model = DQN('CnnPolicy', env, device='cuda:0')
```

### Custom Reward Functions
```python
# Implement custom reward in reward_calculator.py
class CustomRewardCalculator(AssettoCorsaRewardCalculator):
    def calculate_reward(self, state, action, next_state):
        # Custom reward logic
        return reward
```

### Curriculum Learning
```python
# Start with easier tracks, progress to harder ones
tracks = ['easy_track', 'medium_track', 'hard_track']
for track in tracks:
    env.set_track(track)
    model.learn(total_timesteps=1000000)
```

## Best Practices

1. **Start Simple**: Begin with basic environment, add features gradually
2. **Monitor Training**: Use TensorBoard to track progress
3. **Save Checkpoints**: Regular saves prevent loss of progress
4. **Test Frequently**: Regular evaluation ensures quality
5. **Hyperparameter Tune**: Experiment with different settings
6. **Use Expert Data**: Combine with imitation learning for better results

## Example Training Workflow

1. **Setup Environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure vJoy
   # Start Assetto Corsa
   ```

2. **Initial Training**
   ```bash
   # Train from scratch
   python src/assetto_corsa_trainer.py --mode train --agent DQN --time-steps 1000000
   ```

3. **Fine-tune from Gym**
   ```bash
   # Fine-tune pre-trained model
   python src/assetto_corsa_trainer.py --mode finetune --agent DQN --gym-model-path gym_model.zip --fine-tune-steps 500000
   ```

4. **Evaluate Performance**
   ```bash
   # Test trained model
   python src/assetto_corsa_trainer.py --mode test --agent DQN --test-episodes 10
   ```

5. **Run Inference**
   ```bash
   # Run on real game
   python src/assetto_corsa_inference.py --model-path models/DQN/best_model.zip --episodes 5
   ```

This training system provides the same robust approach as DeepRL-CarRacing while being specifically adapted for Assetto Corsa's unique requirements. 