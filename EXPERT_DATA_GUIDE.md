# Expert Data Collection and Imitation Learning Guide

This guide explains how to collect expert driving data and use it to train AI models through imitation learning techniques.

## Overview

The system provides two main components:
1. **Expert Data Collector**: Captures human driving demonstrations
2. **Imitation Learning Trainer**: Trains models using the collected data

## Expert Data Collection

### What Data is Collected

The expert data collector captures:
- **Screenshots**: Game visuals at 60 FPS
- **Actions**: Throttle, brake, steering, gear changes, clutch
- **Telemetry**: Speed, RPM, gear, lap time, etc. (optional)
- **Metadata**: Timestamps, episode information, file references

### Data Format

The data is saved in a structured format:
```
expert_data/
├── session_20241201_143022/
│   ├── screenshots/
│   │   ├── episode_001_frame_000000.png
│   │   ├── episode_001_frame_000001.png
│   │   └── ...
│   ├── metadata/
│   │   └── expert_data.csv
│   └── telemetry/
└── session_20241201_150045/
    └── ...
```

### CSV Metadata Format

```csv
episode,frame,timestamp,screenshot_filename,throttle,brake,steering,gear_up,gear_down,clutch,speed,rpm,gear,lap_time
1,0,1701441022.123,episode_001_frame_000000.png,0.8,0.0,0.1,False,False,0.0,120.5,4500,3,45.2
1,1,1701441022.139,episode_001_frame_000001.png,0.9,0.0,0.0,False,False,0.0,125.2,4700,3,45.1
...
```

## Collecting Expert Data

### Basic Data Collection

```bash
# Start collecting expert data
python src/expert_data_collector.py --output-dir expert_data
```

### Enhanced Data Collection with Telemetry

```bash
# Collect data with telemetry information
python src/expert_data_collector.py --output-dir expert_data --use-enhanced --use-telemetry
```

### Custom Screen Region

```bash
# Specify custom screen region
python src/expert_data_collector.py --output-dir expert_data --screen-region "0,0,1920,1080"
```

### Data Collection Controls

During collection, use these controls:
- **W/Up Arrow**: Throttle
- **S/Down Arrow**: Brake
- **A/Left Arrow**: Steer Left
- **D/Right Arrow**: Steer Right
- **Q**: Gear Down
- **E**: Gear Up
- **Space**: Clutch
- **ESC**: Stop collection
- **R**: Reset episode

### Best Practices for Data Collection

1. **Quality Over Quantity**: Focus on clean, consistent driving
2. **Diverse Scenarios**: Include different tracks, weather, and conditions
3. **Consistent Performance**: Maintain similar skill level throughout
4. **Multiple Sessions**: Collect data across different sessions
5. **Error Recovery**: Include examples of recovering from mistakes

### Recommended Data Collection Strategy

1. **Start Simple**: Begin with easy tracks and good conditions
2. **Progressive Difficulty**: Gradually increase track difficulty
3. **Consistent Style**: Use similar driving style throughout
4. **Sufficient Duration**: Collect at least 10-30 minutes per session
5. **Multiple Episodes**: Use R key to create separate episodes

## Imitation Learning Training

### Training Approaches

The system supports three imitation learning approaches:

1. **Behavioral Cloning (BC)**: Direct policy imitation
2. **Deep Q-Learning from Demonstrations (DQfD)**: RL with expert data
3. **Combined Approach**: BC followed by DQfD

### Training Modes

#### 1. Behavioral Cloning Only

```bash
# Train behavioral cloning model
python src/imitation_learning_trainer.py --expert-data-dir expert_data/session_20241201_143022 --mode bc --bc-epochs 100
```

#### 2. DQfD Only

```bash
# Train DQfD model
python src/imitation_learning_trainer.py --expert-data-dir expert_data/session_20241201_143022 --mode dqfd --dqfd-steps 1000000
```

#### 3. Combined Training (Recommended)

```bash
# Train both BC and DQfD in sequence
python src/imitation_learning_trainer.py --expert-data-dir expert_data/session_20241201_143022 --mode combined --bc-epochs 50 --dqfd-steps 500000
```

### Training Parameters

#### Behavioral Cloning Parameters
- **Learning Rate**: 1e-4 (default)
- **Batch Size**: 32 (default)
- **Epochs**: 50-100 (recommended)
- **Model Architecture**: CNN with 6 output actions

#### DQfD Parameters
- **Learning Rate**: 1e-4 (default)
- **Batch Size**: 32 (default)
- **Timesteps**: 500,000-1,000,000 (recommended)
- **Expert Samples**: Up to 5,000 preloaded into replay buffer

### Model Outputs

Trained models are saved in:
```
models/imitation/
├── bc/
│   ├── best_bc_model.pth
│   ├── final_bc_model.pth
│   └── training_history.json
└── dqfd/
    ├── best_model.zip
    ├── final_model.zip
    ├── checkpoints/
    └── eval_logs/
```

## Complete Workflow Example

### Step 1: Collect Expert Data

```bash
# Start Assetto Corsa and begin driving
python src/expert_data_collector.py --output-dir expert_data --use-enhanced --use-telemetry

# Drive for 20-30 minutes, creating multiple episodes
# Press R to reset episodes when needed
# Press ESC when finished
```

### Step 2: Train Imitation Learning Model

```bash
# Train combined model on collected data
python src/imitation_learning_trainer.py \
    --expert-data-dir expert_data/session_20241201_143022 \
    --mode combined \
    --bc-epochs 50 \
    --dqfd-steps 500000 \
    --evaluate
```

### Step 3: Fine-tune with RL

```bash
# Fine-tune the imitation learning model with RL
python src/assetto_corsa_trainer.py \
    --mode finetune \
    --agent DQN \
    --gym-model-path models/imitation/dqfd/best_model.zip \
    --fine-tune-steps 1000000
```

### Step 4: Test the Model

```bash
# Test the trained model
python src/assetto_corsa_inference.py \
    --model-path models/imitation/dqfd/best_model.zip \
    --episodes 5
```

## Data Quality Assessment

### Check Data Quality

```python
import pandas as pd

# Load metadata
df = pd.read_csv('expert_data/session_20241201_143022/metadata/expert_data.csv')

# Basic statistics
print(f"Total frames: {len(df)}")
print(f"Episodes: {df['episode'].nunique()}")
print(f"Duration: {df['timestamp'].max() - df['timestamp'].min():.1f}s")

# Action statistics
print(f"Throttle (mean/std): {df['throttle'].mean():.3f}/{df['throttle'].std():.3f}")
print(f"Brake (mean/std): {df['brake'].mean():.3f}/{df['brake'].std():.3f}")
print(f"Steering (mean/std): {df['steering'].mean():.3f}/{df['steering'].std():.3f}")
```

### Quality Indicators

Good expert data should have:
- **Consistent Actions**: Reasonable throttle/brake/steering values
- **Smooth Transitions**: Gradual changes in control inputs
- **Diverse Scenarios**: Different driving situations
- **Sufficient Duration**: At least 10,000+ frames
- **Clean Episodes**: No crashes or major errors

## Advanced Features

### Multi-Session Training

```bash
# Train on multiple expert sessions
python src/imitation_learning_trainer.py \
    --expert-data-dir expert_data \
    --mode combined \
    --bc-epochs 100 \
    --dqfd-steps 1000000
```

### Custom Model Architecture

Modify the behavioral cloning model in `imitation_learning_trainer.py`:

```python
def setup_model(self):
    # Custom CNN architecture
    self.model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # ... custom layers
        nn.Linear(512, 6)
    ).to(self.device)
```

### Data Augmentation

Add data augmentation to improve training:

```python
# In ExpertDataset.__getitem__
if self.transform:
    img = self.transform(img)
else:
    # Basic augmentation
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
        action[2] *= -1  # Invert steering
```

## Troubleshooting

### Common Issues

1. **Poor Data Quality**
   - Ensure consistent driving style
   - Avoid crashes and major errors
   - Collect more diverse scenarios

2. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Use more expert data

3. **Overfitting**
   - Reduce model complexity
   - Use data augmentation
   - Collect more diverse data

4. **Poor Generalization**
   - Collect data from multiple tracks
   - Include different weather conditions
   - Use longer training sessions

### Performance Optimization

1. **GPU Acceleration**: Ensure CUDA is available
2. **Data Loading**: Use multiple workers for data loading
3. **Memory Management**: Adjust batch size based on GPU memory
4. **Storage**: Use SSD for faster data loading

## Expected Results

### Training Timeline

- **0-10k frames**: Basic driving patterns
- **10k-50k frames**: Improved consistency
- **50k+ frames**: High-quality imitation

### Performance Metrics

- **Action Accuracy**: How well actions match expert
- **Driving Smoothness**: Consistency of control inputs
- **Lap Completion**: Success rate on tracks
- **Lap Times**: Comparison to expert performance

### Success Indicators

- **Low BC Loss**: < 0.01 for behavioral cloning
- **High DQfD Reward**: > 500 average reward
- **Smooth Driving**: No erratic behavior
- **Consistent Performance**: Low variance in results

## Integration with RL Training

### Hybrid Training Pipeline

1. **Collect Expert Data**: Human demonstrations
2. **Train Imitation Model**: BC + DQfD
3. **Fine-tune with RL**: Transfer to RL environment
4. **Evaluate Performance**: Compare to expert

### Transfer Learning Benefits

- **Faster Convergence**: Pre-trained from expert data
- **Better Initialization**: Good starting policy
- **Improved Exploration**: Guided by expert demonstrations
- **Stable Training**: Reduced training instability

This expert data collection and imitation learning system provides a powerful foundation for training high-quality AI driving agents that can learn from human expertise and then improve through reinforcement learning. 