"""
AgriLite-Hybrid - Shared Constants
Common constants used by both backend and edge deployment
"""

# =============================================================================
# Supported Crops
# =============================================================================
SUPPORTED_CROPS = ['tomato', 'chilli']

# =============================================================================
# Image Processing Constants
# =============================================================================
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

# Normalization parameters (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# =============================================================================
# Severity Levels
# =============================================================================
SEVERITY_LEVELS = ['mild', 'medium', 'severe']

SEVERITY_THRESHOLDS = {
    'mild': (0.0, 0.33),
    'medium': (0.33, 0.66),
    'severe': (0.66, 1.0)
}

SEVERITY_COLORS = {
    'mild': '#22C55E',      # Green
    'medium': '#F59E0B',    # Amber/Orange
    'severe': '#EF4444'     # Red
}

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_CONFIG = {
    'h5_model': 'agrilite_hybrid.h5',
    'keras_model': 'agrilite_hybrid.keras',
    'tflite_model': 'agrilite_hybrid.tflite',
    'class_labels': 'class_labels.json',
    'confidence_threshold': 0.5
}

# =============================================================================
# Environmental Risk Thresholds (DHT22 Sensor)
# =============================================================================
ENVIRONMENTAL_THRESHOLDS = {
    'temperature': {
        'high_risk': 35,    # °C - Heat stress
        'low_risk': 10      # °C - Cold stress
    },
    'humidity': {
        'high_risk': 85,    # % - Fungal disease risk
        'low_risk': 30      # % - Drought stress
    }
}

# =============================================================================
# API Response Messages
# =============================================================================
MESSAGES = {
    'PREDICTION_SUCCESS': 'Disease prediction completed successfully',
    'INVALID_CROP': 'Invalid crop type. Supported crops: tomato, chilli',
    'INVALID_IMAGE': 'Invalid or corrupt image file',
    'MODEL_ERROR': 'Error during model inference',
    'UPLOAD_ERROR': 'Error processing uploaded image'
}
