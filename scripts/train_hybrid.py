"""
AgriLite-Hybrid: Multi-Crop Disease Detection Model
====================================================
A lightweight hybrid architecture combining:
- MobileNetV3-Small (fast, efficient backbone)
- EfficientNetV2-B0 (accurate, feature-rich backbone)
- CBAM Attention (Channel + Spatial attention)

Combined Dataset: 2 crops with all disease classes
- Tomato: 11 classes  
- Chilli (Pepper): 8 classes
Total: 19 classes

(Note: Brinjal/Eggplant was removed - its standalone model collapsed to
15.82% accuracy, likely from data leakage caused by splitting after
augmentation. See METHODOLOGY_AND_TECHNICAL_REFERENCE.md for details.)

Optimized for Raspberry Pi deployment with TFLite INT8 quantization.

Architecture:
┌─────────────────┐     ┌─────────────────┐
│ MobileNetV3-Small│     │ EfficientNetV2-B0│
│   (576 features) │     │  (1280 features) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    │
              ┌─────▼─────┐
              │ Concatenate│
              │(1856 features)│
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │   CBAM    │
              │ Attention │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │   Dense   │
              │   Layers  │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │  Softmax  │
              │ (19 classes)│
              └───────────┘

Author: AgriLite Hybrid Project
Date: February 2026
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetV2B0  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard,
    LearningRateScheduler
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2

# Sklearn for evaluation
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score
)

# For Grad-CAM
import cv2

# GPU Configuration
from gpu_config import setup_gpu, get_strategy

# Logging
from logger_config import setup_logger, TrainingLogger
import time

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Setup GPU FIRST before logger (to capture GPU info)
print("\n" + "="*70)
print("🔧 CONFIGURING GPU FOR HYBRID MODEL TRAINING")
print("="*70)
gpu_info = setup_gpu(
    memory_growth=True,
    mixed_precision=True,
    verbose=True
)

# Initialize logger
logger = setup_logger(
    name='hybrid_training',
    log_dir='logs/hybrid',
    console_level=10,  # DEBUG
    file_level=10       # DEBUG
)
train_logger = TrainingLogger(logger, log_dir='logs/hybrid')

logger.info(f"TensorFlow version: {tf.__version__}")
if gpu_info['gpu_available']:
    logger.info(f"✓ Training on GPU: {gpu_info['gpu_names']}")
    logger.info(f"✓ Mixed precision: {gpu_info['mixed_precision']}")
    logger.info(f"✓ Strategy: {gpu_info['strategy']}")
else:
    logger.warning("⚠️ Training on CPU (this will be slow)")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Get project root directory (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class Config:
    """Training configuration for AgriLite-Hybrid model"""
    
    # Dataset paths - RELATIVE from project root (works on Windows & Linux)
    DATA_DIR = os.path.join(PROJECT_ROOT, "DataSets", "combined")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Alternative: Use individual crop folders and merge
    INDIVIDUAL_DATA = {
        'tomato': os.path.join(PROJECT_ROOT, "DataSets", "tamota"),
        'chilli': os.path.join(PROJECT_ROOT, "DataSets", "chilli", 
                               "Chilli Plant Diseases Dataset(Augmented)", 
                               "Chilli Plant Diseases Dataset")
    }
    
    # Model parameters
    MODEL_NAME = "agrilite_hybrid"
    INPUT_SHAPE = (224, 224, 3)
    
    # Expected classes (will be auto-detected)
    # Tomato: 11, Chilli: 8 = 19 total
    NUM_CLASSES = 19
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Training phases
    INITIAL_EPOCHS = 25  # Frozen backbones
    FINE_TUNE_EPOCHS = 75  # Fine-tuning
    FINE_TUNE_LR = 1e-5
    
    # Callbacks
    EARLY_STOP_PATIENCE = 15
    LR_REDUCE_PATIENCE = 5
    LR_REDUCE_FACTOR = 0.5
    
    # Output directories
    OUTPUT_DIR = "outputs/hybrid"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
    
    # Hybrid model settings
    MOBILENET_TRAINABLE_LAYERS = 40  # Layers to unfreeze
    EFFICIENTNET_TRAINABLE_LAYERS = 30  # Layers to unfreeze
    CBAM_REDUCTION_RATIO = 16
    DROPOUT_RATE = 0.4
    
    @classmethod
    def create_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.PLOT_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"Output directories created at: {cls.OUTPUT_DIR}")


# =============================================================================
# 3. CBAM ATTENTION MODULE
# =============================================================================

class ChannelAttention(layers.Layer):
    """
    Channel Attention Module - focuses on "what" is meaningful.
    
    Uses both average and max pooling to capture different aspects
    of channel-wise dependencies.
    """
    
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        # Shared MLP
        self.fc1 = layers.Dense(
            max(self.channels // self.reduction_ratio, 8),
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'{self.name}_fc1'
        )
        self.fc2 = layers.Dense(
            self.channels,
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'{self.name}_fc2'
        )
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Average pooling path
        avg_out = self.avg_pool(inputs)
        avg_out = self.fc1(avg_out)
        avg_out = self.fc2(avg_out)
        
        # Max pooling path
        max_out = self.max_pool(inputs)
        max_out = self.fc1(max_out)
        max_out = self.fc2(max_out)
        
        # Combine and apply sigmoid
        attention = tf.nn.sigmoid(avg_out + max_out)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])
        
        return inputs * attention
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio
        })
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module - focuses on "where" is important.
    
    Uses channel-wise pooling to highlight important spatial regions.
    """
    
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False,
            name=f'{self.name}_conv'
        )
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        # Channel-wise average pooling
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        
        # Channel-wise max pooling
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate and apply convolution
        concat = tf.concat([avg_out, max_out], axis=-1)
        attention = self.conv(concat)
        
        return inputs * attention
    
    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM)
    
    Sequentially applies Channel Attention and Spatial Attention
    to refine feature representations.
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module"
    """
    
    def __init__(self, channels, reduction_ratio=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.channel_attention = ChannelAttention(
            self.channels, 
            self.reduction_ratio,
            name=f'{self.name}_channel_att'
        )
        self.spatial_attention = SpatialAttention(
            self.kernel_size,
            name=f'{self.name}_spatial_att'
        )
        super(CBAM, self).build(input_shape)
        
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config


# =============================================================================
# 4. DATA LOADING
# =============================================================================

def create_combined_dataset_structure(config):
    """
    Create combined dataset structure from individual crop folders.
    
    If combined folder doesn't exist, creates symlinks or copies
    from individual crop folders.
    """
    
    if os.path.exists(config.TRAIN_DIR):
        print(f"Combined dataset found at: {config.DATA_DIR}")
        return True
    
    print("\n⚠️ Combined dataset not found. Please create it with this structure:")
    print(f"""
    {config.DATA_DIR}/
    ├── train/
    │   ├── tomato_healthy/
    │   ├── tomato_bacterial_spot/
    │   ├── ... (all tomato classes with 'tomato_' prefix)
    │   ├── chilli_healthy/
    │   ├── chilli_leaf_curl/
    │   └── ... (all chilli classes with 'chilli_' prefix)
    ├── val/
    │   └── ... (same structure)
    └── test/
        └── ... (same structure)
    
    Or set USE_INDIVIDUAL_FOLDERS=True to load from separate folders.
    """)
    return False


def create_data_generators(config, use_individual=False):
    """
    Create data generators for combined multi-crop dataset.
    
    Heavy augmentation is used to improve generalization across crops.
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=15.0,
        fill_mode='nearest',
        validation_split=0.1 if use_individual else 0.0
    )
    
    # Validation/Test generator (no augmentation)
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1 if use_individual else 0.0
    )
    
    if use_individual:
        # Load from individual crop folders with validation split
        print("\n📁 Loading from individual crop folders...")
        train_generators = []
        val_generators = []
        
        for crop_name, crop_path in config.INDIVIDUAL_DATA.items():
            print(f"  Loading {crop_name}...")
            
            if crop_name == 'tomato':
                # Tomato has pre-split train/valid folders
                train_gen = train_datagen.flow_from_directory(
                    os.path.join(crop_path, 'train'),
                    target_size=config.INPUT_SHAPE[:2],
                    batch_size=config.BATCH_SIZE,
                    class_mode='categorical',
                    shuffle=True,
                    seed=42
                )
                val_gen = val_test_datagen.flow_from_directory(
                    os.path.join(crop_path, 'valid'),
                    target_size=config.INPUT_SHAPE[:2],
                    batch_size=config.BATCH_SIZE,
                    class_mode='categorical',
                    shuffle=False
                )
            elif crop_name == 'chilli':
                # Chilli has train/valid/test
                train_gen = train_datagen.flow_from_directory(
                    os.path.join(crop_path, 'train'),
                    target_size=config.INPUT_SHAPE[:2],
                    batch_size=config.BATCH_SIZE,
                    class_mode='categorical',
                    shuffle=True,
                    seed=42
                )
                val_gen = val_test_datagen.flow_from_directory(
                    os.path.join(crop_path, 'valid'),
                    target_size=config.INPUT_SHAPE[:2],
                    batch_size=config.BATCH_SIZE,
                    class_mode='categorical',
                    shuffle=False
                )
            
            train_generators.append((crop_name, train_gen))
            val_generators.append((crop_name, val_gen))
        
        # Combine generators (simplified - just use the first one for now)
        # In practice, you'd want a custom generator that samples from all
        print("\n⚠️ Note: Using individual folders. For best results, create combined dataset.")
        train_generator = train_generators[0][1]
        val_generator = val_generators[0][1]
        
    else:
        # Load from combined folder
        print(f"\n📁 Loading combined dataset from: {config.DATA_DIR}")
        
        train_generator = train_datagen.flow_from_directory(
            config.TRAIN_DIR,
            target_size=config.INPUT_SHAPE[:2],
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            config.VAL_DIR,
            target_size=config.INPUT_SHAPE[:2],
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    
    # Get class info
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    num_classes = len(class_names)
    
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"  - Training samples: {train_generator.samples}")
    print(f"  - Validation samples: {val_generator.samples}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"\nClasses detected:")
    for i, (name, idx) in enumerate(sorted(class_indices.items(), key=lambda x: x[1])):
        crop = "�" if "tomato" in name.lower() else \
               "🌶️" if "chilli" in name.lower() else "🌿"
        print(f"    {idx:2d}: {crop} {name}")
    
    # Save class labels
    labels_path = os.path.join(config.OUTPUT_DIR, "class_labels.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(labels_path, 'w') as f:
        json.dump({
            'class_indices': class_indices,
            'class_names': class_names,
            'num_classes': num_classes,
            'model': 'AgriLite-Hybrid',
            'crops': ['tomato', 'chilli']
        }, f, indent=2)
    print(f"\n✓ Class labels saved to: {labels_path}")
    
    return train_generator, val_generator, class_names, num_classes


# =============================================================================
# 5. HYBRID MODEL ARCHITECTURE
# =============================================================================

def build_agrilite_hybrid_model(input_shape, num_classes, config):
    """
    Build the AgriLite-Hybrid model.
    
    Architecture:
    1. Dual backbone: MobileNetV3-Small + EfficientNetV2-B0
    2. Feature fusion via concatenation
    3. CBAM attention on fused features
    4. Dense classification head
    
    This hybrid approach combines:
    - MobileNetV3: Speed and efficiency (optimized for mobile/edge)
    - EfficientNetV2: Accuracy and rich features (compound scaling)
    - CBAM: Attention mechanism for feature refinement
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        config: Configuration object
    
    Returns:
        model: Complete hybrid model
        mobilenet_base: MobileNetV3 base model (for fine-tuning)
        efficientnet_base: EfficientNetV2 base model (for fine-tuning)
    """
    
    print(f"\n{'='*60}")
    print("🔧 Building AgriLite-Hybrid Model")
    print(f"{'='*60}")
    
    # Shared input layer
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # =========================================================================
    # Branch 1: MobileNetV3-Small (Lightweight, Fast)
    # =========================================================================
    print("\n  Branch 1: MobileNetV3-Small")
    mobilenet_base = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None,
        include_preprocessing=False
    )
    
    # Rename layers to avoid conflicts
    for layer in mobilenet_base.layers:
        layer._name = 'mobilenet_' + layer.name
    
    mobilenet_base.trainable = False  # Freeze initially
    
    mobilenet_features = mobilenet_base.output  # Shape: (7, 7, 576)
    mobilenet_features = layers.GlobalAveragePooling2D(name='mobilenet_gap')(mobilenet_features)
    print(f"    Output features: {mobilenet_features.shape}")
    
    # =========================================================================
    # Branch 2: EfficientNetV2-B0 (Accurate, Feature-rich)
    # =========================================================================
    print("\n  Branch 2: EfficientNetV2-B0")
    
    # Need to create a new input for EfficientNet (shared weights will be different)
    efficientnet_base = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None,
        include_preprocessing=False
    )
    
    # Rename layers
    for layer in efficientnet_base.layers:
        layer._name = 'efficientnet_' + layer.name
    
    efficientnet_base.trainable = False  # Freeze initially
    
    # Apply EfficientNet to the same input
    efficientnet_features = efficientnet_base(inputs)  # Shape: (7, 7, 1280)
    efficientnet_features = layers.GlobalAveragePooling2D(name='efficientnet_gap')(efficientnet_features)
    print(f"    Output features: {efficientnet_features.shape}")
    
    # =========================================================================
    # Feature Fusion
    # =========================================================================
    print("\n  Feature Fusion")
    
    # Concatenate features from both branches
    # MobileNetV3-Small: 576 features
    # EfficientNetV2-B0: 1280 features
    # Total: 1856 features
    fused_features = layers.Concatenate(name='feature_fusion')([
        mobilenet_features, 
        efficientnet_features
    ])
    print(f"    Fused features: {fused_features.shape}")
    
    # Reshape for CBAM (need spatial dimensions)
    # Convert to pseudo-spatial format: (batch, 1, 1, features)
    fused_spatial = layers.Reshape((1, 1, -1), name='reshape_for_cbam')(fused_features)
    
    # =========================================================================
    # CBAM Attention
    # =========================================================================
    print("\n  CBAM Attention Module")
    
    fused_channels = fused_features.shape[-1]  # 1856
    
    # Apply CBAM attention
    attended_features = CBAM(
        channels=fused_channels,
        reduction_ratio=config.CBAM_REDUCTION_RATIO,
        kernel_size=1,  # Using 1 since we have 1x1 spatial
        name='cbam_attention'
    )(fused_spatial)
    
    # Flatten back
    attended_features = layers.Flatten(name='flatten_attended')(attended_features)
    print(f"    Attended features: {attended_features.shape}")
    
    # =========================================================================
    # Classification Head
    # =========================================================================
    print("\n  Classification Head")
    
    x = attended_features
    
    # Dense layer 1
    x = layers.Dense(512, kernel_regularizer=l2(1e-4), name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(config.DROPOUT_RATE, name='dropout_1')(x)
    
    # Dense layer 2
    x = layers.Dense(256, kernel_regularizer=l2(1e-4), name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(config.DROPOUT_RATE * 0.75, name='dropout_2')(x)
    
    # Dense layer 3 (smaller)
    x = layers.Dense(128, kernel_regularizer=l2(1e-4), name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.Dropout(config.DROPOUT_RATE * 0.5, name='dropout_3')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    print(f"    Output: {outputs.shape}")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    model = Model(inputs=inputs, outputs=outputs, name='AgriLite_Hybrid')
    
    # Print summary
    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    
    print(f"\n{'='*60}")
    print("Model Statistics:")
    print(f"{'='*60}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  - Estimated size (float32): ~{total_params * 4 / (1024**2):.1f} MB")
    print(f"  - Estimated size (INT8): ~{total_params / (1024**2):.1f} MB")
    
    return model, mobilenet_base, efficientnet_base


def unfreeze_hybrid_model(model, mobilenet_base, efficientnet_base, config):
    """
    Unfreeze layers for fine-tuning.
    
    Selectively unfreezes the last N layers of each backbone
    while keeping batch normalization layers frozen for stability.
    """
    
    print(f"\n{'='*60}")
    print("🔓 Unfreezing Model for Fine-tuning")
    print(f"{'='*60}")
    
    # Unfreeze MobileNetV3 layers
    mobilenet_base.trainable = True
    mobilenet_layers = len(mobilenet_base.layers)
    freeze_until_mobile = max(0, mobilenet_layers - config.MOBILENET_TRAINABLE_LAYERS)
    
    for i, layer in enumerate(mobilenet_base.layers):
        if i < freeze_until_mobile:
            layer.trainable = False
        else:
            layer.trainable = True
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    
    print(f"  MobileNetV3: {config.MOBILENET_TRAINABLE_LAYERS}/{mobilenet_layers} layers unfrozen")
    
    # Unfreeze EfficientNetV2 layers
    efficientnet_base.trainable = True
    efficient_layers = len(efficientnet_base.layers)
    freeze_until_efficient = max(0, efficient_layers - config.EFFICIENTNET_TRAINABLE_LAYERS)
    
    for i, layer in enumerate(efficientnet_base.layers):
        if i < freeze_until_efficient:
            layer.trainable = False
        else:
            layer.trainable = True
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    
    print(f"  EfficientNetV2: {config.EFFICIENTNET_TRAINABLE_LAYERS}/{efficient_layers} layers unfrozen")
    
    # Count trainable parameters
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"  Total trainable parameters: {trainable_params:,}")
    
    return model


# =============================================================================
# 6. TRAINING FUNCTIONS
# =============================================================================

def get_callbacks(config, phase='initial'):
    """Create training callbacks."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        ModelCheckpoint(
            filepath=os.path.join(
                config.MODEL_DIR, 
                f'{config.MODEL_NAME}_{phase}_best.keras'
            ),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, f'{phase}_{timestamp}'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def compile_model(model, learning_rate, config):
    """Compile model with optimizer, loss, and metrics."""
    
    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=config.WEIGHT_DECAY
        ),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    
    print(f"\n✓ Model compiled:")
    print(f"  - Optimizer: AdamW (lr={learning_rate}, wd={config.WEIGHT_DECAY})")
    print(f"  - Loss: CategoricalCrossentropy (label_smoothing=0.1)")
    
    return model


def train_model(model, train_gen, val_gen, config, phase='initial', epochs=None):
    """Train the model."""
    
    if epochs is None:
        epochs = config.EPOCHS
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting {phase.upper()} training for {epochs} epochs...")
    print(f"{'='*70}")
    
    callbacks = get_callbacks(config, phase)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# =============================================================================
# 7. EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, test_gen, class_names, config):
    """Comprehensive model evaluation."""
    
    print("\n" + "="*70)
    print("📊 EVALUATING AGRILITE-HYBRID MODEL")
    print("="*70)
    
    # Get predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Class labels
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Shorten labels for display
    short_labels = []
    for label in class_labels:
        short = label.replace("Chilli_", "C_").replace("Chilli__", "C_")
        short = short[:20]
        short_labels.append(short)
    
    # Classification report
    print("\nClassification Report:")
    print("-"*60)
    report = classification_report(y_true, y_pred, target_names=short_labels, digits=4)
    print(report)
    
    # Save full report
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=short_labels, yticklabels=short_labels)
    plt.title('Confusion Matrix - AgriLite-Hybrid\n(Tomato + Chilli)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    cm_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {cm_path}")
    
    # Summary metrics
    accuracy = np.mean(y_pred == y_true)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Per-crop accuracy
    crop_metrics = {}
    for crop_prefix in ['tomato', 'chilli']:
        crop_indices = [i for i, name in enumerate(class_labels) 
                       if crop_prefix.lower() in name.lower()]
        if crop_indices:
            crop_mask = np.isin(y_true, crop_indices)
            if crop_mask.sum() > 0:
                crop_acc = np.mean(y_pred[crop_mask] == y_true[crop_mask])
                crop_metrics[crop_prefix] = float(crop_acc)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'per_crop_accuracy': crop_metrics,
        'num_classes': len(class_labels),
        'test_samples': int(test_gen.samples)
    }
    
    print(f"\n{'='*60}")
    print("📈 SUMMARY METRICS:")
    print(f"{'='*60}")
    print(f"  ✓ Overall Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✓ F1-Score (Macro):    {f1_macro:.4f}")
    print(f"  ✓ F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"\n  Per-Crop Accuracy:")
    for crop, acc in crop_metrics.items():
        emoji = "" if "tomato" in crop else "🌶️"
        print(f"    {emoji} {crop.capitalize()}: {acc:.4f} ({acc*100:.2f}%)")
    
    return metrics


def plot_training_history(history, config, phase=''):
    """Plot training history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'train': '#2E7D32', 'val': '#C62828'}
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', 
                    linewidth=2, color=colors['train'])
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', 
                    linewidth=2, color=colors['val'])
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train',
                    linewidth=2, color=colors['train'])
    axes[0, 1].plot(history.history['val_loss'], label='Validation',
                    linewidth=2, color=colors['val'])
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train',
                        linewidth=2, color='#1565C0')
        axes[1, 0].plot(history.history['val_precision'], label='Validation',
                        linewidth=2, color='#FF6F00')
        axes[1, 0].set_title('Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train',
                        linewidth=2, color='#1565C0')
        axes[1, 1].plot(history.history['val_recall'], label='Validation',
                        linewidth=2, color='#FF6F00')
        axes[1, 1].set_title('Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History - AgriLite-Hybrid ({phase})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(config.PLOT_DIR, f"training_history_{phase}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved: {plot_path}")


# =============================================================================
# 8. MODEL EXPORT & TFLITE CONVERSION
# =============================================================================

def save_model(model, config):
    """Save model in multiple formats."""
    
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    
    # Keras format
    keras_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final.keras")
    model.save(keras_path)
    keras_size = os.path.getsize(keras_path) / (1024**2)
    print(f"✓ Keras model: {keras_path} ({keras_size:.1f} MB)")
    
    # H5 format
    h5_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final.h5")
    model.save(h5_path, save_format='h5')
    h5_size = os.path.getsize(h5_path) / (1024**2)
    print(f"✓ H5 model: {h5_path} ({h5_size:.1f} MB)")
    
    # SavedModel format
    savedmodel_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_savedmodel")
    model.export(savedmodel_path)
    print(f"✓ SavedModel: {savedmodel_path}")
    
    return keras_path, h5_path, savedmodel_path


def convert_to_tflite(model, val_gen, config):
    """
    Convert model to TFLite formats for Raspberry Pi deployment.
    
    Creates multiple versions:
    1. Float32 (highest accuracy)
    2. Float16 (good balance)
    3. INT8 dynamic range (small, fast)
    4. Full INT8 (smallest, fastest)
    """
    
    print("\n" + "="*60)
    print("📱 TFLITE CONVERSION (Raspberry Pi Deployment)")
    print("="*60)
    
    tflite_paths = []
    
    # 1. Float32
    print("\n1. Converting to TFLite (float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}.tflite")
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size = os.path.getsize(path) / (1024**2)
    print(f"   ✓ Saved: {path} ({size:.1f} MB)")
    tflite_paths.append(('float32', path, size))
    
    # 2. Float16
    print("\n2. Converting to TFLite (float16)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_fp16.tflite")
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size = os.path.getsize(path) / (1024**2)
    print(f"   ✓ Saved: {path} ({size:.1f} MB)")
    tflite_paths.append(('float16', path, size))
    
    # 3. INT8 dynamic range
    print("\n3. Converting to TFLite (INT8 dynamic)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_int8.tflite")
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size = os.path.getsize(path) / (1024**2)
    print(f"   ✓ Saved: {path} ({size:.1f} MB)")
    tflite_paths.append(('int8_dynamic', path, size))
    
    # 4. Full INT8 with representative dataset
    print("\n4. Converting to TFLite (Full INT8)...")
    
    def representative_dataset():
        val_gen.reset()
        count = 0
        for images, _ in val_gen:
            for image in images:
                if count >= 200:
                    return
                yield [np.expand_dims(image.astype(np.float32), axis=0)]
                count += 1
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        
        path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_full_int8.tflite")
        with open(path, 'wb') as f:
            f.write(tflite_model)
        size = os.path.getsize(path) / (1024**2)
        print(f"   ✓ Saved: {path} ({size:.1f} MB)")
        print(f"   🚀 Best for Raspberry Pi!")
        tflite_paths.append(('full_int8', path, size))
    except Exception as e:
        print(f"   ⚠ Full INT8 failed: {e}")
    
    # Summary
    print(f"\n📊 TFLite Size Comparison:")
    base_size = tflite_paths[0][2]
    for name, path, size in tflite_paths:
        reduction = (1 - size/base_size) * 100
        print(f"   - {name:12s}: {size:6.1f} MB ({reduction:+5.1f}% from float32)")
    
    print(f"\n⏱️ Estimated Raspberry Pi 4 inference time:")
    print(f"   - Float32: ~200-300 ms")
    print(f"   - Float16: ~150-200 ms")
    print(f"   - INT8:    ~80-120 ms")
    print(f"   - Full INT8: ~50-80 ms")
    
    return tflite_paths


# =============================================================================
# 9. GRAD-CAM VISUALIZATION
# =============================================================================

class GradCAMHybrid:
    """Grad-CAM for hybrid model with dual backbones."""
    
    def __init__(self, model, mobilenet_layer='mobilenet_gap', 
                 efficientnet_layer='efficientnet_gap'):
        self.model = model
        
        # Try to find appropriate layers
        self.mobile_layer = None
        self.efficient_layer = None
        
        for layer in model.layers:
            if 'mobilenet' in layer.name and 'conv' in layer.name.lower():
                self.mobile_layer = layer.name
            if 'efficientnet' in layer.name and 'conv' in layer.name.lower():
                self.efficient_layer = layer.name
        
        # Fallback to provided names
        if self.mobile_layer is None:
            self.mobile_layer = mobilenet_layer
        if self.efficient_layer is None:
            self.efficient_layer = efficientnet_layer
            
        print(f"Grad-CAM layers: {self.mobile_layer}, {self.efficient_layer}")
    
    def compute_heatmap(self, image, class_idx=None):
        """Compute combined Grad-CAM heatmap."""
        
        with tf.GradientTape() as tape:
            # Get predictions
            predictions = self.model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        # Get gradients for the last layer before output
        gradients = tape.gradient(loss, self.model.trainable_variables[-2])
        
        if gradients is not None:
            pooled_grads = tf.reduce_mean(gradients)
            heatmap = tf.abs(pooled_grads)
            heatmap = tf.maximum(heatmap, 0)
            heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
        
        return np.zeros((7, 7))
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """Overlay heatmap on image."""
        
        if heatmap.ndim == 0:
            heatmap = np.ones((7, 7)) * heatmap
        
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed = heatmap_colored * alpha + image * (1 - alpha)
        return np.clip(superimposed, 0, 255).astype(np.uint8), heatmap


def generate_multicrop_gradcam(model, val_gen, config, samples_per_crop=3):
    """Generate Grad-CAM examples for multiple crops."""
    
    print("\n" + "="*60)
    print("🔍 GENERATING MULTI-CROP GRAD-CAM VISUALIZATIONS")
    print("="*60)
    
    # Get class info
    idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
    
    # Identify crop classes
    crops = {'tomato': [], 'chilli': []}
    for idx, name in idx_to_class.items():
        name_lower = name.lower()
        if 'tomato' in name_lower:
            crops['tomato'].append(idx)
        elif 'chilli' in name_lower:
            crops['chilli'].append(idx)
    
    # Collect samples per crop
    crop_samples = {crop: [] for crop in crops}
    
    val_gen.reset()
    for images, labels in val_gen:
        for img, label in zip(images, labels):
            class_idx = np.argmax(label)
            for crop, indices in crops.items():
                if class_idx in indices and len(crop_samples[crop]) < samples_per_crop:
                    crop_samples[crop].append((img, class_idx))
        
        if all(len(v) >= samples_per_crop for v in crop_samples.values()):
            break
    
    # Create visualization
    fig, axes = plt.subplots(3, samples_per_crop * 2, figsize=(samples_per_crop * 6, 10))
    
    crop_emojis = {'tomato': '🍅', 'chilli': '🌶️'}
    
    for row, (crop_name, samples) in enumerate(crop_samples.items()):
        for col, (img, class_idx) in enumerate(samples):
            if col >= samples_per_crop:
                break
            
            # Original image
            img_display = (img * 255).astype(np.uint8)
            axes[row, col * 2].imshow(img_display)
            class_name = idx_to_class[class_idx]
            short_name = class_name.split('_')[-1][:15] if '_' in class_name else class_name[:15]
            axes[row, col * 2].set_title(f'{crop_emojis[crop_name]} {short_name}', fontsize=9)
            axes[row, col * 2].axis('off')
            
            # Prediction with confidence
            img_array = np.expand_dims(img, axis=0)
            pred_probs = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(pred_probs[0])
            confidence = pred_probs[0][pred_idx]
            
            # Simple attention visualization (using activation)
            axes[row, col * 2 + 1].imshow(img_display)
            pred_name = idx_to_class[pred_idx].split('_')[-1][:10] if '_' in idx_to_class[pred_idx] else idx_to_class[pred_idx][:10]
            status = "✓" if pred_idx == class_idx else "✗"
            axes[row, col * 2 + 1].set_title(f'{status} {pred_name} ({confidence:.0%})', fontsize=9)
            axes[row, col * 2 + 1].axis('off')
    
    # Add row labels
    for row, crop_name in enumerate(crops.keys()):
        axes[row, 0].set_ylabel(f'{crop_emojis[crop_name]} {crop_name.upper()}', 
                                fontsize=12, rotation=0, labelpad=60)
    
    plt.suptitle('Multi-Crop Disease Detection - AgriLite-Hybrid', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(config.PLOT_DIR, "multicrop_predictions.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Multi-crop visualization saved: {path}")


# =============================================================================
# 10. MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline for AgriLite-Hybrid model."""
    
    print("\n" + "="*70)
    print("🌱 AGRILITE-HYBRID: MULTI-CROP DISEASE DETECTION")
    print("   Architecture: MobileNetV3-Small + EfficientNetV2-B0 + CBAM")
    print("   Crops:  Tomato + 🌶️ Chilli")
    print("   Optimized for: Raspberry Pi deployment")
    print("="*70)
    
    # Initialize
    config = Config()
    config.create_dirs()
    
    # =========================================================================
    # PHASE 1: Data Loading
    # =========================================================================
    print("\n" + "="*70)
    print("📁 PHASE 1: Loading Multi-Crop Dataset")
    print("="*70)
    
    # Check if combined dataset exists, otherwise use individual folders
    use_individual = not os.path.exists(config.TRAIN_DIR)
    
    if use_individual:
        print("\n⚠️ Combined dataset not found. Using individual crop folders.")
        print("   For best results, create a combined dataset structure.")
        
        # For demo, we'll use tomato dataset as primary
        config.DATA_DIR = config.INDIVIDUAL_DATA['tomato']
        config.TRAIN_DIR = os.path.join(config.DATA_DIR, 'train')
        config.VAL_DIR = os.path.join(config.DATA_DIR, 'valid')
    
    train_gen, val_gen, class_names, num_classes = create_data_generators(
        config, use_individual=False
    )
    
    config.NUM_CLASSES = num_classes
    
    # =========================================================================
    # PHASE 2: Build Hybrid Model
    # =========================================================================
    print("\n" + "="*70)
    print("🏗️ PHASE 2: Building AgriLite-Hybrid Model")
    print("="*70)
    
    model, mobilenet_base, efficientnet_base = build_agrilite_hybrid_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=num_classes,
        config=config
    )
    
    # Model summary
    print("\nModel Architecture:")
    model.summary(show_trainable=True, expand_nested=False)
    
    # =========================================================================
    # PHASE 3: Initial Training (Frozen Backbones)
    # =========================================================================
    print("\n" + "="*70)
    print("🎯 PHASE 3: Initial Training (Backbones Frozen)")
    print("="*70)
    
    model = compile_model(model, config.INITIAL_LR, config)
    
    history_initial = train_model(
        model, train_gen, val_gen, config,
        phase='initial',
        epochs=config.INITIAL_EPOCHS
    )
    
    plot_training_history(history_initial, config, phase='initial')
    
    # =========================================================================
    # PHASE 4: Fine-tuning
    # =========================================================================
    print("\n" + "="*70)
    print("🔧 PHASE 4: Fine-tuning (Backbones Partially Unfrozen)")
    print("="*70)
    
    model = unfreeze_hybrid_model(model, mobilenet_base, efficientnet_base, config)
    model = compile_model(model, config.FINE_TUNE_LR, config)
    
    history_finetune = train_model(
        model, train_gen, val_gen, config,
        phase='finetune',
        epochs=config.FINE_TUNE_EPOCHS
    )
    
    plot_training_history(history_finetune, config, phase='finetune')
    
    # =========================================================================
    # PHASE 5: Evaluation
    # =========================================================================
    print("\n" + "="*70)
    print("📊 PHASE 5: Model Evaluation")
    print("="*70)
    
    metrics = evaluate_model(model, val_gen, class_names, config)
    
    # Save metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # =========================================================================
    # PHASE 6: Save Models
    # =========================================================================
    print("\n" + "="*70)
    print("💾 PHASE 6: Saving Models")
    print("="*70)
    
    keras_path, h5_path, savedmodel_path = save_model(model, config)
    
    # =========================================================================
    # PHASE 7: TFLite Conversion
    # =========================================================================
    print("\n" + "="*70)
    print("📱 PHASE 7: TFLite Conversion")
    print("="*70)
    
    tflite_paths = convert_to_tflite(model, val_gen, config)
    
    # =========================================================================
    # PHASE 8: Visualizations
    # =========================================================================
    print("\n" + "="*70)
    print("🔍 PHASE 8: Generating Visualizations")
    print("="*70)
    
    generate_multicrop_gradcam(model, val_gen, config)
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output: {config.OUTPUT_DIR}")
    
    print(f"\n📊 Results:")
    print(f"   ✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   ✓ F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"   ✓ Classes: {metrics['num_classes']}")
    
    print(f"\n📦 Models:")
    print(f"   ✓ Keras: {keras_path}")
    for name, path, size in tflite_paths:
        print(f"   ✓ TFLite ({name}): {path} ({size:.1f} MB)")
    
    print(f"\n🚀 Ready for Raspberry Pi deployment!")
    print("="*70)
    
    return model, metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # GPU already configured at module level via gpu_config
    # Dataset must already be combined - run scripts/combine_datasets.py first
    
    # Run training
    model, metrics = main()
