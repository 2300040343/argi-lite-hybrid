# =============================================================================
# AgriLite-Hybrid - Raspberry Pi Deployment Scripts
# =============================================================================

This directory contains scripts for deploying AgriLite-Hybrid on Raspberry Pi
devices for edge inference.

## Files

- `predict_pi.py` - Standalone prediction script for offline inference
- `web_server_pi.py` - Lightweight Flask server for local network access
- `requirements_pi.txt` - Minimal dependencies for Pi deployment
- `setup_pi.sh` - Installation and setup script

## Hardware Requirements

- Raspberry Pi 4 (4GB+ RAM recommended)
- 32GB+ microSD card
- Camera Module (v2 or HQ Camera) or USB webcam
- Power supply (5V 3A)

## Quick Setup

```bash
# 1. Clone or copy the scripts to Pi
scp -r scripts/pi_deploy pi@raspberrypi.local:~/agrilite

# 2. SSH into the Pi
ssh pi@raspberrypi.local

# 3. Run setup script
cd ~/agrilite
chmod +x setup_pi.sh
./setup_pi.sh

# 4. Copy your trained model
cp /path/to/agrilite_model.tflite ~/agrilite/models/

# 5. Run the web server
python3 web_server_pi.py
```

## Usage Options

### Option 1: Command Line Prediction
```bash
python3 predict_pi.py --image /path/to/leaf.jpg --crop tomato
```

### Option 2: Local Web Server
```bash
python3 web_server_pi.py --port 8080
# Access at http://raspberrypi.local:8080
```

### Option 3: Camera Capture
```bash
python3 predict_pi.py --camera --crop tomato
```

## Model Files

Place your trained model files in the `models/` directory:
- `agrilite_model.tflite` - TensorFlow Lite model (recommended for Pi)
- `class_labels.json` - Class label mappings

## Performance Optimization

For best performance on Raspberry Pi:
1. Use TFLite model instead of full TensorFlow
2. Enable XNNPACK delegate for ARM acceleration
3. Resize images to 224x224 before inference
4. Use picamera2 for optimized camera access

## Troubleshooting

**Out of Memory**: Try closing other applications or use Pi 4 with 8GB RAM
**Slow Inference**: Ensure using TFLite model with XNNPACK delegate
**Camera Not Found**: Check camera is enabled in `raspi-config`
