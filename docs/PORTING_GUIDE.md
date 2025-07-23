# Porting Gym DQN Model to Assetto Corsa - Complete Guide

This guide walks you through the complete process of transferring a pre-trained DQN model from the gym CarRacing environment to work with Assetto Corsa.

## **Overview**

The porting process involves:
1. **Model Transfer**: Adapting the gym model architecture to Assetto Corsa
2. **Environment Wrapper**: Creating a gym-compatible interface for Assetto Corsa
3. **Imitation Learning**: Using the gym model as a teacher
4. **Reinforcement Learning**: Fine-tuning on the real game
5. **Evaluation**: Testing the final model

## **Prerequisites**

### **Software Requirements**
- **Assetto Corsa** (installed and working)
- **Python 3.8+** with required packages
- **vJoy** driver installed and configured
- **x360ce** (optional, for Xbox controller emulation)

### **Hardware Requirements**
- **Windows PC** (vJoy is Windows-only)
- **Game controller** (for manual testing)
- **Sufficient RAM** (8GB+ recommended)

## **Installation Steps**

### **Step 1: Install Dependencies**
```bash
# Install required Python packages
pip install -r requirements.txt

# Install vJoy driver
# Download from: http://vjoystick.sourceforge.net/
# Run installer as administrator
```

### **Step 2: Configure vJoy**
1. Open vJoy Configuration
2. Create a virtual joystick device
3. Configure axes for steering, throttle, brake
4. Test the virtual device

### **Step 3: Configure Assetto Corsa**
1. Launch Assetto Corsa
2. Go to Settings → Controls
3. Set input method to "Joystick"
4. Calibrate the vJoy device
5. Test controls work properly

## **Usage Instructions**

### **Quick Start (Basic Transfer)**

```bash
# Transfer the gym model to Assetto Corsa
python src/transfer_model.py --test-episodes 3
```

This will:
- Load the pre-trained DQN model from gym
- Create an Assetto Corsa environment wrapper
- Test the model on the real game
- Save the adapted model

### **Full Training Pipeline**

```bash
# Complete training with imitation learning + RL
python src/train_assetto_corsa.py \
    --imitation-steps 5000 \
    --rl-steps 20000 \
    --eval-episodes 5
```

This will:
1. **Imitation Learning**: Use gym model as teacher
2. **Reinforcement Learning**: Fine-tune on Assetto Corsa
3. **Evaluation**: Test the final model

### **Custom Screen Region**

If Assetto Corsa doesn't use full screen:

```bash
# Specify custom screen region (x,y,width,height)
python src/transfer_model.py --screen-region 100,50,1600,900
```

### **Skip Training Phases**

```bash
# Skip imitation learning, go straight to RL
python src/train_assetto_corsa.py --skip-imitation

# Skip RL, just test the transferred model
python src/train_assetto_corsa.py --skip-rl
```

## **Configuration Options**

### **Screen Capture Settings**
- **Full Screen**: `--screen-region 0,0,1920,1080`
- **Windowed Mode**: `--screen-region 100,50,1600,900`
- **Multiple Monitors**: Adjust coordinates accordingly

### **Training Parameters**
- **Imitation Steps**: 1000-10000 (more = better imitation)
- **RL Steps**: 10000-100000 (more = better performance)
- **Evaluation Episodes**: 5-20 (more = better statistics)

### **Model Paths**
- **Gym Model**: `../baseline-gym/DeepRL-CarRacing/Training/Saved_Models/DQN/best_model.zip`
- **Output Directory**: `models/` (customizable)

## **File Structure**

```
src/
├── assetto_corsa_env.py      # Assetto Corsa environment wrapper
├── transfer_model.py         # Model transfer script
├── reward_calculator.py      # Enhanced reward function
├── train_assetto_corsa.py    # Main training script
└── control_test.py           # vJoy interface (existing)

models/                       # Output directory
├── imitation_model.zip       # After imitation learning
├── final_model.zip          # After RL fine-tuning
└── evaluation_results.npy   # Performance metrics

logs/                        # TensorBoard logs
├── imitation_learning/
└── reinforcement_learning/
```

## **Troubleshooting**

### **Common Issues**

#### **1. vJoy Connection Failed**
```
Error: Failed to open vJoy device
```
**Solution:**
- Install vJoy driver as administrator
- Check vJoy Configuration
- Ensure virtual device is created

#### **2. Screen Capture Issues**
```
Error: Cannot capture screen region
```
**Solution:**
- Verify screen resolution
- Check if game is in correct window mode
- Adjust screen region coordinates

#### **3. Model Loading Errors**
```
Error: Model not found
```
**Solution:**
- Verify gym model path
- Check file permissions
- Ensure model file exists

#### **4. Poor Performance**
**Symptoms:** Car crashes, goes off track, doesn't drive well

**Solutions:**
- Increase imitation learning steps
- Adjust reward function weights
- Fine-tune screen region
- Check game settings

### **Performance Optimization**

#### **1. Screen Region Optimization**
- Capture only the game view (exclude UI)
- Use smaller resolution for faster processing
- Test different regions for best results

#### **2. Training Parameters**
- Start with fewer steps for testing
- Increase gradually based on results
- Monitor TensorBoard logs for progress

#### **3. Reward Function Tuning**
- Adjust weights in `reward_calculator.py`
- Add more sophisticated game state detection
- Implement OCR for speed/lap reading

## **Advanced Features**

### **Custom Reward Function**
Edit `src/reward_calculator.py` to:
- Add OCR for speed reading
- Implement lap detection
- Add collision detection
- Customize reward weights

### **Multiple Track Training**
```bash
# Train on different tracks
python src/train_assetto_corsa.py --track "monza"
python src/train_assetto_corsa.py --track "nurburgring"
```

### **Model Comparison**
```bash
# Compare different models
python src/evaluate_models.py --model1 "gym_model" --model2 "assetto_model"
```

## **Expected Results**

### **Performance Metrics**
- **Gym Model**: ~900 points (baseline)
- **After Transfer**: ~200-400 points (initial drop expected)
- **After Imitation**: ~400-600 points
- **After RL**: ~600-800 points

### **Driving Behavior**
- **Basic**: Stays on track, avoids crashes
- **Intermediate**: Maintains speed, follows racing line
- **Advanced**: Optimizes lap times, smooth driving

## **Next Steps**

### **Immediate Improvements**
1. **OCR Integration**: Add real speed/lap reading
2. **Multi-Track Training**: Train on multiple circuits
3. **Advanced Rewards**: Implement lap time optimization

### **Long-term Enhancements**
1. **Multi-Agent Racing**: Race against other cars
2. **Weather Conditions**: Train in different weather
3. **Car Physics**: Adapt to different vehicles

## **Support and Resources**

### **Documentation**
- [vJoy Documentation](http://vjoystick.sourceforge.net/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Assetto Corsa Modding](https://www.assettocorsa.net/)

### **Community**
- GitHub Issues for bug reports
- Discord server for discussions
- Reddit r/assettocorsa for game-specific help

---

**Note**: This is a research project. Results may vary based on hardware, game settings, and training parameters. Start with small tests before running full training sessions. 