# Expert Data Collection Guide for AI Assetto Corsa

This guide explains how to collect expert driving data for training the AI agent using imitation learning.

## Overview

The AI system uses a combination of **imitation learning** and **reinforcement learning** to learn how to drive in Assetto Corsa. The first step is collecting high-quality expert driving data.

## Data Format

### Recommended Format: CSV with Screenshots

The system collects data in the following structure:

```
expert_data/
├── session_20231201_143022/
│   ├── session_metadata.json
│   ├── driving_data.csv
│   ├── screenshots/
│   │   ├── frame_000000.png
│   │   ├── frame_000001.png
│   │   └── ...
│   └── lane_masks/
│       ├── lane_mask_000000.png
│       ├── lane_mask_000001.png
│       └── ...
```

### CSV Data Fields

| Field | Description | Range/Format |
|-------|-------------|--------------|
| `timestamp` | Unix timestamp | Float |
| `frame_id` | Sequential frame number | Integer |
| `steering` | Steering input | -1.0 to 1.0 |
| `throttle` | Throttle input | 0.0 to 1.0 |
| `brake` | Brake input | 0.0 to 1.0 |
| `gear` | Current gear | 1-6, R |
| `speed` | Current speed (km/h) | Float |
| `x_pos, y_pos` | Car position | Float |
| `heading` | Car orientation (radians) | Float |
| `lap_time` | Current lap time | Float |
| `lap_number` | Current lap | Integer |
| `track_position` | Distance along track | Float |
| `screenshot_path` | Path to screenshot | String |
| `lane_mask_path` | Path to lane mask | String |
| `raw_axes` | Raw joystick axes | JSON array |
| `raw_buttons` | Raw joystick buttons | JSON array |
| `raw_hats` | Raw joystick hats | JSON array |

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Controller

The system supports various controllers:
- **Xbox Controller** (recommended)
- **PlayStation Controller**
- **Generic USB Controller**

Make sure your controller is connected and recognized by your system.

### 3. Configure Screen Capture

Edit the `bounding_box` in the data collector to match your Assetto Corsa window:

```python
# Adjust these values to match your screen setup
bounding_box = {
    'top': 320,      # Distance from top of screen
    'left': 680,     # Distance from left of screen  
    'width': 600,    # Width of capture area
    'height': 400    # Height of capture area
}
```

## Data Collection Process

### Step 1: Basic Data Collection

Run the basic data collector:

```bash
cd src
python data_collector.py
```

**Controls:**
- Press `R` to start/stop recording
- Press `Q` to quit

### Step 2: Enhanced Data Collection (Recommended)

For more comprehensive data collection:

```bash
cd src
python enhanced_data_collector.py
```

This version includes:
- Session metadata
- Lane detection masks
- Better data organization
- Background processing

## Best Practices for Data Collection

### 1. **Quality Over Quantity**
- Focus on **consistent, smooth driving**
- Avoid erratic movements or crashes
- Drive at **reasonable speeds** for the track

### 2. **Diverse Scenarios**
Collect data for:
- **Different tracks** (Monza, Spa, Nürburgring, etc.)
- **Different cars** (F1, GT3, road cars)
- **Different weather conditions** (dry, wet)
- **Different times of day** (day, night)

### 3. **Recording Sessions**
- **Session length**: 5-15 minutes per session
- **Multiple sessions**: 10-20 sessions per track/car combination
- **Consistent driving style**: Maintain the same driving approach

### 4. **Data Quality Checks**
- **Smooth inputs**: Avoid sudden jerky movements
- **Consistent lap times**: Within 2-3 seconds variation
- **No crashes**: Clean driving data only
- **Proper gear usage**: Appropriate gear selection

### 5. **File Organization**
```
expert_data/
├── monza_f1_dry/
│   ├── session_001/
│   ├── session_002/
│   └── ...
├── spa_gt3_wet/
│   ├── session_001/
│   └── ...
└── nurburgring_roadcar/
    ├── session_001/
    └── ...
```

## Data Validation

### Check Your Data Quality

After collecting data, verify:

1. **Input ranges are correct**:
   - Steering: -1.0 to 1.0
   - Throttle/Brake: 0.0 to 1.0

2. **Screenshots are captured**:
   - Images are clear and readable
   - Game window is properly captured

3. **Data consistency**:
   - No missing frames
   - Timestamps are sequential
   - File paths are valid

### Data Analysis Script

Use this script to analyze your collected data:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('expert_data/session_001/driving_data.csv')

# Plot steering inputs
plt.figure(figsize=(12, 4))
plt.plot(df['steering'])
plt.title('Steering Inputs Over Time')
plt.ylabel('Steering (-1 to 1)')
plt.show()

# Plot throttle and brake
plt.figure(figsize=(12, 4))
plt.plot(df['throttle'], label='Throttle')
plt.plot(df['brake'], label='Brake')
plt.title('Throttle and Brake Inputs')
plt.legend()
plt.show()
```

## Troubleshooting

### Common Issues

1. **No joystick detected**
   - Check controller connection
   - Install controller drivers
   - Test with `python input.py`

2. **Screen capture not working**
   - Adjust `bounding_box` coordinates
   - Ensure Assetto Corsa is visible
   - Run in windowed mode

3. **Poor data quality**
   - Check controller calibration
   - Ensure smooth driving
   - Verify input mappings

4. **Performance issues**
   - Reduce capture resolution
   - Close unnecessary applications
   - Use SSD for data storage

## Next Steps

After collecting expert data:

1. **Preprocess the data** for training
2. **Train the imitation learning model**
3. **Fine-tune with reinforcement learning**
4. **Test the AI agent**

## File Structure

```
AI-Assetto-Corsa/
├── src/
│   ├── data_collector.py          # Basic data collection
│   ├── enhanced_data_collector.py # Advanced data collection
│   ├── input.py                   # Joystick input test
│   └── control_test.py            # Virtual joystick test
├── expert_data/                   # Collected data (created automatically)
├── requirements.txt               # Dependencies
└── DATA_COLLECTION_GUIDE.md      # This guide
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your setup matches the requirements
3. Test with the basic input script first
4. Ensure Assetto Corsa is running properly

---

**Remember**: High-quality expert data is crucial for successful AI training. Take your time to collect clean, consistent driving data! 