# Training Logs

This directory contains training logs for AgriLite-Hybrid model training sessions.

## Log Structure

```
logs/
├── hybrid/                          # Hybrid model training logs
│   ├── training_YYYYMMDD_HHMMSS.log      # Full training log
│   ├── training_metrics_YYYYMMDD_HHMMSS.csv  # CSV metrics
│   └── agrilite_hybrid_YYYYMMDD_HHMMSS.log   # Session log
├── chilli/                           # Chilli model logs
├── tomato/                          # Tomato model logs
└── chilli/                          # Chilli model logs
```

## Log Files

### Training Log (`training_YYYYMMDD_HHMMSS.log`)
Complete training session log including:
- Configuration parameters
- Dataset statistics
- Model architecture details
- Epoch-by-epoch training progress
- Validation metrics
- Errors and warnings
- File save confirmations

### Metrics CSV (`training_metrics_YYYYMMDD_HHMMSS.csv`)
Structured data for analysis:
```csv
epoch,phase,loss,accuracy,val_loss,val_accuracy,lr,duration_sec
1,initial,2.1234,0.3456,2.2345,0.3123,0.0001,125.34
2,initial,1.9876,0.4123,2.1234,0.3789,0.0001,123.45
...
```

## Usage

### Option 1: Run with logging wrapper (Recommended)
```bash
python scripts/train_hybrid_logged.py
```
This automatically captures all output to log files.

### Option 2: Run directly with output redirection
```bash
# PowerShell
python scripts/train_hybrid.py 2>&1 | Tee-Object -FilePath "logs/training_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Bash/Linux
python scripts/train_hybrid.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

### Option 3: Background training with nohup (Linux/Mac)
```bash
nohup python scripts/train_hybrid_logged.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Option 4: Screen session (Server deployment)
```bash
# Start a screen session
screen -S agrilite_training

# Run training
python scripts/train_hybrid_logged.py

# Detach: Ctrl+A, then D
# Reattach: screen -r agrilite_training
# View logs: tail -f logs/hybrid/training_*.log
```

## Analyzing Logs

### View real-time training progress
```bash
# PowerShell
Get-Content logs/hybrid/training_*.log -Wait -Tail 50

# Bash/Linux
tail -f logs/hybrid/training_*.log
```

### Extract accuracy metrics
```bash
# PowerShell
Select-String "accuracy" logs/hybrid/training_*.log

# Bash/Linux
grep "accuracy" logs/hybrid/training_*.log
```

### Plot metrics from CSV
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('logs/hybrid/training_metrics_20260222_123456.csv')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(df['epoch'], df['accuracy'], label='Train')
axes[0].plot(df['epoch'], df['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()

# Loss
axes[1].plot(df['epoch'], df['loss'], label='Train')
axes[1].plot(df['epoch'], df['val_loss'], label='Val')
axes[1].set_title('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_progress.png')
```

## Log Retention

- Training logs are kept indefinitely
- Rotating file handler limits individual log files to 10MB
- Up to 5 backup files are kept per session
- Manually clean old logs if needed:

```bash
# Keep only last 30 days
find logs/ -name "*.log" -mtime +30 -delete
```

## Debugging

### Common issues:

1. **No logs created**: Check write permissions on `logs/` directory
2. **Logs cut off**: Check disk space
3. **Garbled output**: Ensure UTF-8 encoding

### Debug mode:
Set logger level to DEBUG in `logger_config.py`:
```python
logger = setup_logger(
    console_level=logging.DEBUG,
    file_level=logging.DEBUG
)
```

## Remote Monitoring

### Copy logs from server:
```bash
# SCP
scp user@server:/path/to/agri-lite-hybrid/logs/hybrid/*.log ./local_logs/

# Rsync
rsync -avz user@server:/path/to/agri-lite-hybrid/logs/ ./local_logs/
```

### View logs via SSH:
```bash
ssh user@server "tail -f /path/to/agri-lite-hybrid/logs/hybrid/training_*.log"
```

---

**Note**: Logs contain sensitive information about your model and data. Do not commit them to git repositories (already excluded in `.gitignore`).
