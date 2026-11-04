"""
AgriLite-Hybrid: Evaluate Trained Model
========================================
Evaluates the hybrid model on the test set and produces:
- final_metrics.json (overall accuracy, F1 scores)
- classification_report.json (per-class precision, recall, F1)
- Per-crop accuracy breakdown (chilli, tomato separately)
- Confusion matrix plots (overall + per-crop)
- Standalone vs Hybrid comparison table

Usage:
    python evaluate_hybrid.py
    python evaluate_hybrid.py --model path/to/model.keras
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sklearn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_MODEL = os.path.join(
    PROJECT_ROOT, "outputs", "outputs", "hybrid", "models",
    "agrilite_hybrid_finetune_best.keras"
)
DEFAULT_LABELS = os.path.join(
    PROJECT_ROOT, "outputs", "outputs", "hybrid", "class_labels.json"
)
DEFAULT_TEST_DIR = os.path.join(
    PROJECT_ROOT, "DataSets", "combined", "test"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "outputs", "hybrid"
)

INPUT_SHAPE = (224, 224)
BATCH_SIZE = 32

# Crop class ranges (from hybrid training)
# Brinjal removed - its standalone model collapsed to 15.82% accuracy
CROP_RANGES = {
    'chilli': (0, 7),     # indices 0-7 (8 classes)
    'tomato': (8, 18),    # indices 8-18 (11 classes)
}

# Standalone model metrics for comparison
STANDALONE_METRICS = {
    'chilli': {
        'model': 'MobileNetV3-Small',
        'accuracy': 0.9292,
        'f1_macro': 0.9060,
        'f1_weighted': 0.9300,
        'status': 'Good'
    },
    'tomato': {
        'model': 'EfficientNetV2-B0',
        'accuracy': 0.9634,
        'f1_macro': 0.8759,
        'f1_weighted': 0.9636,
        'status': 'Excellent'
    }
}


def load_model(model_path):
    """Load the trained Keras model."""
    print(f"Loading model from: {model_path}")
    
    # Register custom objects if CBAM layers are used
    from train_hybrid import ChannelAttention, SpatialAttention, CBAM
    custom_objects = {
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'CBAM': CBAM,
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model loaded. Input shape: {model.input_shape}, Output shape: {model.output_shape}")
    return model


def load_class_labels(labels_path):
    """Load class labels JSON."""
    with open(labels_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {data['num_classes']} class labels")
    return data


def create_test_generator(test_dir, class_names):
    """Create test data generator with same preprocessing as training."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=INPUT_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )
    
    print(f"Test samples: {test_generator.samples}")
    print(f"Classes found: {len(test_generator.class_indices)}")
    return test_generator


def evaluate_model(model, test_generator, class_names, output_dir):
    """Run full evaluation and generate all metrics."""
    
    print("\n" + "=" * 70)
    print("EVALUATING HYBRID MODEL")
    print("=" * 70)
    
    # Get predictions
    print("\nRunning predictions on test set...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*50}")
    print(f"OVERALL METRICS")
    print(f"{'='*50}")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    
    # Per-class classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    print(f"\n{report_str}")
    
    # Save overall metrics
    final_metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report_dict
    }
    
    metrics_path = os.path.join(output_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved: {metrics_path}")
    
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved: {report_path}")
    
    # Per-crop breakdown
    per_crop_metrics = compute_per_crop_metrics(y_true, y_pred, class_names)
    
    crop_metrics_path = os.path.join(output_dir, 'per_crop_metrics.json')
    with open(crop_metrics_path, 'w') as f:
        json.dump(per_crop_metrics, f, indent=2)
    print(f"Saved: {crop_metrics_path}")
    
    # Comparison table
    comparison = build_comparison_table(per_crop_metrics)
    comparison_path = os.path.join(output_dir, 'standalone_vs_hybrid_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {comparison_path}")
    
    # Print comparison table
    print_comparison_table(comparison)
    
    # Generate plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Overall confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title='Confusion Matrix - AgriLite-Hybrid (All Crops)',
        save_path=os.path.join(plot_dir, 'confusion_matrix_hybrid_overall.png')
    )
    
    # Per-crop confusion matrices
    for crop_name, (start, end) in CROP_RANGES.items():
        crop_mask = (y_true >= start) & (y_true <= end)
        if crop_mask.sum() == 0:
            continue
        
        crop_true = y_true[crop_mask] - start
        crop_pred = y_pred[crop_mask] - start
        # Clip predictions that fall outside this crop's range
        crop_pred = np.clip(crop_pred, 0, end - start)
        crop_class_names = class_names[start:end + 1]
        
        plot_confusion_matrix(
            crop_true, crop_pred, crop_class_names,
            title=f'Confusion Matrix - {crop_name.title()} (Hybrid Model)',
            save_path=os.path.join(plot_dir, f'confusion_matrix_hybrid_{crop_name}.png')
        )
    
    # Comparison bar chart
    plot_comparison_chart(comparison, plot_dir)
    
    return final_metrics, per_crop_metrics, comparison


def compute_per_crop_metrics(y_true, y_pred, class_names):
    """Compute metrics separately for each crop."""
    per_crop = {}
    
    for crop_name, (start, end) in CROP_RANGES.items():
        crop_mask = (y_true >= start) & (y_true <= end)
        if crop_mask.sum() == 0:
            per_crop[crop_name] = {'error': 'No test samples found'}
            continue
        
        crop_true = y_true[crop_mask]
        crop_pred = y_pred[crop_mask]
        
        # Accuracy: how many within this crop's range were correctly predicted
        crop_accuracy = accuracy_score(crop_true, crop_pred)
        
        # F1 scores (using original global indices)
        crop_f1_macro = f1_score(
            crop_true, crop_pred, average='macro',
            labels=list(range(start, end + 1)), zero_division=0
        )
        crop_f1_weighted = f1_score(
            crop_true, crop_pred, average='weighted',
            labels=list(range(start, end + 1)), zero_division=0
        )
        
        # Per-class within crop
        crop_class_names = class_names[start:end + 1]
        crop_report = classification_report(
            crop_true, crop_pred,
            labels=list(range(start, end + 1)),
            target_names=crop_class_names,
            output_dict=True,
            zero_division=0
        )
        
        per_crop[crop_name] = {
            'num_classes': end - start + 1,
            'num_samples': int(crop_mask.sum()),
            'accuracy': crop_accuracy,
            'f1_macro': crop_f1_macro,
            'f1_weighted': crop_f1_weighted,
            'classification_report': crop_report
        }
        
        print(f"\n{'='*50}")
        print(f"{crop_name.upper()} ({end - start + 1} classes, {crop_mask.sum()} samples)")
        print(f"{'='*50}")
        print(f"Accuracy:      {crop_accuracy:.4f} ({crop_accuracy*100:.2f}%)")
        print(f"F1 (macro):    {crop_f1_macro:.4f}")
        print(f"F1 (weighted): {crop_f1_weighted:.4f}")
        
        # Per-class within crop
        crop_report_str = classification_report(
            crop_true, crop_pred,
            labels=list(range(start, end + 1)),
            target_names=crop_class_names,
            zero_division=0
        )
        print(crop_report_str)
    
    return per_crop


def build_comparison_table(per_crop_metrics):
    """Build standalone vs hybrid comparison table."""
    comparison = {}
    
    for crop_name in CROP_RANGES:
        hybrid = per_crop_metrics.get(crop_name, {})
        standalone = STANDALONE_METRICS.get(crop_name, {})
        
        hybrid_acc = hybrid.get('accuracy', 0)
        standalone_acc = standalone.get('accuracy', 0)
        improvement = hybrid_acc - standalone_acc
        
        comparison[crop_name] = {
            'standalone': {
                'model': standalone.get('model', 'N/A'),
                'accuracy': standalone_acc,
                'f1_macro': standalone.get('f1_macro', 0),
                'f1_weighted': standalone.get('f1_weighted', 0),
                'status': standalone.get('status', 'N/A')
            },
            'hybrid': {
                'model': 'AgriLite-Hybrid (MobileNetV3 + EfficientNetV2 + CBAM)',
                'accuracy': hybrid_acc,
                'f1_macro': hybrid.get('f1_macro', 0),
                'f1_weighted': hybrid.get('f1_weighted', 0),
                'num_classes': hybrid.get('num_classes', 0),
                'num_samples': hybrid.get('num_samples', 0)
            },
            'improvement': {
                'accuracy_delta': improvement,
                'accuracy_delta_pct': f"{improvement*100:+.2f}%"
            }
        }
    
    return comparison


def print_comparison_table(comparison):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("STANDALONE vs HYBRID COMPARISON")
    print("=" * 90)
    print(f"{'Crop':<12} {'Standalone Model':<35} {'SA Acc':<10} {'Hybrid Acc':<12} {'Delta':<10}")
    print("-" * 90)
    
    for crop_name, data in comparison.items():
        sa = data['standalone']
        hy = data['hybrid']
        delta = data['improvement']['accuracy_delta_pct']
        
        print(
            f"{crop_name:<12} "
            f"{sa['model']:<35} "
            f"{sa['accuracy']:<10.4f} "
            f"{hy['accuracy']:<12.4f} "
            f"{delta:<10}"
        )
    
    print("-" * 90)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine figure size based on number of classes
    n_classes = len(class_names)
    fig_size = max(8, n_classes * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    
    # Shorten class names for display
    display_names = []
    for name in class_names:
        # Remove common prefixes
        short = name
        for prefix in ['chilli_Chilli_', 'chilli_Chilli__', 'tomato_']:
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        display_names.append(short)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_comparison_chart(comparison, plot_dir):
    """Plot standalone vs hybrid accuracy comparison bar chart."""
    crops = list(comparison.keys())
    sa_acc = [comparison[c]['standalone']['accuracy'] * 100 for c in crops]
    hy_acc = [comparison[c]['hybrid']['accuracy'] * 100 for c in crops]
    
    x = np.arange(len(crops))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, sa_acc, width, label='Standalone', color='#EF4444', alpha=0.8)
    bars2 = ax.bar(x + width/2, hy_acc, width, label='Hybrid (AgriLite)', color='#3B82F6', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Standalone vs Hybrid Model - Per-Crop Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in crops])
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'standalone_vs_hybrid_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate AgriLite-Hybrid Model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Path to model file')
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS, help='Path to class_labels.json')
    parser.add_argument('--test-dir', type=str, default=DEFAULT_TEST_DIR, help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"ERROR: Labels not found: {args.labels}")
        sys.exit(1)
    if not os.path.exists(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and labels
    labels_data = load_class_labels(args.labels)
    class_names = labels_data['class_names']
    
    model = load_model(args.model)
    
    # Create test generator
    test_generator = create_test_generator(args.test_dir, class_names)
    
    # Evaluate
    final_metrics, per_crop, comparison = evaluate_model(
        model, test_generator, np.array(class_names), args.output_dir
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Overall accuracy: {final_metrics['accuracy']*100:.2f}%")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - final_metrics.json")
    print(f"  - classification_report.json")
    print(f"  - per_crop_metrics.json")
    print(f"  - standalone_vs_hybrid_comparison.json")
    print(f"  - plots/confusion_matrix_hybrid_overall.png")
    for crop in CROP_RANGES:
        print(f"  - plots/confusion_matrix_hybrid_{crop}.png")
    print(f"  - plots/standalone_vs_hybrid_comparison.png")


if __name__ == '__main__':
    main()
