"""
Dataset Combiner Utility
========================
Merges individual crop datasets into a unified multi-crop dataset
with crop-prefixed class names for AgriLite-Hybrid model training.

Creates structure:
DataSets/combined/
├── train/
│   ├── tomato_bacterial_spot/
│   ├── tomato_healthy/
│   ├── chilli_healthy/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...

Author: AgriLite Hybrid Project
Date: February 2026
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Get project root directory (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Configuration - RELATIVE paths from project root (works on Windows & Linux)
BASE_DIR = os.path.join(PROJECT_ROOT, "DataSets")
OUTPUT_DIR = os.path.join(BASE_DIR, "combined")

# Crop dataset paths and configurations
# NOTE: Brinjal was removed - its standalone model collapsed to 15.82% accuracy
# (likely data leakage from splitting after augmentation). See METHODOLOGY doc.
CROP_CONFIGS = {
    'tomato': {
        'path': os.path.join(BASE_DIR, "tamota"),
        'splits': {
            'train': 'train',
            'val': 'valid',
            'test': None  # Use portion of val for test
        },
        'split_ratio': None,  # Already split
        'prefix': 'tomato'
    },
    'chilli': {
        'path': os.path.join(BASE_DIR, "chilli", "Chilli Plant Diseases Dataset(Augmented)", 
                             "Chilli Plant Diseases Dataset"),
        'splits': {
            'train': 'train',
            'val': 'valid',
            'test': 'test'
        },
        'split_ratio': None,  # Already split
        'prefix': 'chilli'
    }
}


def get_image_files(directory):
    """Get all image files in a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    
    # Use long path for Windows
    dir_path = long_path(directory)
    
    try:
        for file in os.listdir(dir_path):
            if os.path.splitext(file)[1].lower() in extensions:
                images.append(os.path.join(directory, file))
    except Exception as e:
        print(f"    Warning: Could not read {directory}: {e}")
    
    return images


def long_path(path):
    """Convert path to Windows long path format if needed."""
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        # Convert to absolute and add long path prefix
        abs_path = os.path.abspath(path)
        return '\\\\?\\' + abs_path
    return path


def copy_with_unique_name(src_path, dst_dir, prefix_counter):
    """Copy file with unique name to avoid conflicts."""
    filename = os.path.basename(src_path)
    name, ext = os.path.splitext(filename)
    
    # Shorten name to avoid path length issues
    name = name[:30] if len(name) > 30 else name
    
    # Add counter prefix to ensure uniqueness
    new_name = f"{prefix_counter:05d}{ext}"
    dst_path = os.path.join(dst_dir, new_name)
    
    # Handle duplicates
    counter = 1
    while os.path.exists(long_path(dst_path)):
        new_name = f"{prefix_counter:05d}_{counter}{ext}"
        dst_path = os.path.join(dst_dir, new_name)
        counter += 1
    
    # Use long path format for Windows
    shutil.copy2(long_path(src_path), long_path(dst_path))
    return dst_path


def combine_datasets(use_symlinks=False, verbose=True):
    """
    Combine all crop datasets into a single multi-crop dataset.
    
    Args:
        use_symlinks: If True, create symlinks instead of copying
                     (only works on Windows with developer mode)
        verbose: Print detailed progress
    
    Returns:
        Dictionary with statistics
    """
    
    print("\n" + "="*70)
    print("🌱 AGRILITE DATASET COMBINER")
    print("   Merging:  Tomato + 🌶️ Chilli")
    print("="*70)
    
    random.seed(42)  # For reproducible splits
    
    stats = {
        'crops': {},
        'classes': {},
        'totals': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(long_path(os.path.join(OUTPUT_DIR, split)), exist_ok=True)
    
    image_counter = 0
    
    for crop_name, config in CROP_CONFIGS.items():
        print(f"\n{'='*50}")
        emoji = "" if crop_name == "tomato" else "🌶️"
        print(f"{emoji} Processing {crop_name.upper()}")
        print(f"{'='*50}")
        
        crop_path = config['path']
        prefix = config['prefix']
        splits = config['splits']
        split_ratio = config['split_ratio']
        
        if not os.path.exists(long_path(crop_path)):
            print(f"  ⚠️ Path not found: {crop_path}")
            continue
        
        stats['crops'][crop_name] = {'train': 0, 'val': 0, 'test': 0, 'classes': []}
        
        # Case 1: Pre-split dataset
        if split_ratio is None:
            for split_name, split_folder in splits.items():
                if split_folder is None:
                    continue
                
                split_path = os.path.join(crop_path, split_folder)
                if not os.path.exists(long_path(split_path)):
                    print(f"  ⚠️ Split not found: {split_path}")
                    continue
                
                print(f"\n  Processing {split_name}/ from {split_folder}/")
                
                # Process each class
                for class_name in os.listdir(long_path(split_path)):
                    class_src = os.path.join(split_path, class_name)
                    if not os.path.isdir(long_path(class_src)):
                        continue
                    
                    # Clean class name
                    clean_name = class_name.replace(' ', '_').replace('-', '_')
                    clean_name = clean_name.replace('__', '_').strip('_')
                    prefixed_name = f"{prefix}_{clean_name}"
                    
                    # Destination class directory
                    class_dst = os.path.join(OUTPUT_DIR, split_name, prefixed_name)
                    os.makedirs(long_path(class_dst), exist_ok=True)
                    
                    # Get images
                    images = get_image_files(class_src)
                    
                    if verbose:
                        print(f"    {prefixed_name}: {len(images)} images")
                    
                    # Copy images
                    for img_path in images:
                        copy_with_unique_name(img_path, class_dst, image_counter)
                        image_counter += 1
                    
                    stats['crops'][crop_name][split_name] += len(images)
                    stats['totals'][split_name] += len(images)
                    
                    if prefixed_name not in stats['crops'][crop_name]['classes']:
                        stats['crops'][crop_name]['classes'].append(prefixed_name)
        
        # Case 2: Single folder - needs splitting
        else:
            print(f"\n  Auto-splitting with ratio {split_ratio}")
            
            # Process each class
            for class_name in os.listdir(long_path(crop_path)):
                class_src = os.path.join(crop_path, class_name)
                if not os.path.isdir(long_path(class_src)):
                    continue
                
                # Clean class name
                clean_name = class_name.replace(' ', '_').replace('-', '_')
                clean_name = clean_name.replace('__', '_').strip('_')
                prefixed_name = f"{prefix}_{clean_name}"
                
                # Get and shuffle images
                images = get_image_files(class_src)
                random.shuffle(images)
                
                # Calculate split indices
                n_total = len(images)
                n_train = int(n_total * split_ratio[0])
                n_val = int(n_total * split_ratio[1])
                
                train_imgs = images[:n_train]
                val_imgs = images[n_train:n_train + n_val]
                test_imgs = images[n_train + n_val:]
                
                if verbose:
                    print(f"    {prefixed_name}: {len(train_imgs)}/{len(val_imgs)}/{len(test_imgs)}")
                
                # Copy to respective folders
                for split_name, split_imgs in [('train', train_imgs), 
                                                ('val', val_imgs), 
                                                ('test', test_imgs)]:
                    if not split_imgs:
                        continue
                    
                    class_dst = os.path.join(OUTPUT_DIR, split_name, prefixed_name)
                    os.makedirs(long_path(class_dst), exist_ok=True)
                    
                    for img_path in split_imgs:
                        copy_with_unique_name(img_path, class_dst, image_counter)
                        image_counter += 1
                    
                    stats['crops'][crop_name][split_name] += len(split_imgs)
                    stats['totals'][split_name] += len(split_imgs)
                
                if prefixed_name not in stats['crops'][crop_name]['classes']:
                    stats['crops'][crop_name]['classes'].append(prefixed_name)
    
    # Count total classes
    all_classes = set()
    for crop_stats in stats['crops'].values():
        all_classes.update(crop_stats['classes'])
    stats['classes'] = sorted(list(all_classes))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 DATASET COMBINATION COMPLETE")
    print("="*70)
    
    print(f"\n📁 Output: {OUTPUT_DIR}")
    
    print(f"\n📈 Per-Crop Statistics:")
    for crop_name, crop_stats in stats['crops'].items():
        emoji = "" if crop_name == "tomato" else "🌶️"
        total = crop_stats['train'] + crop_stats['val'] + crop_stats['test']
        print(f"   {emoji} {crop_name.capitalize():10s}: {total:6d} images")
        print(f"       Train: {crop_stats['train']:5d} | Val: {crop_stats['val']:5d} | Test: {crop_stats['test']:5d}")
        print(f"       Classes: {len(crop_stats['classes'])}")
    
    print(f"\n📊 Total Statistics:")
    total = stats['totals']['train'] + stats['totals']['val'] + stats['totals']['test']
    print(f"   Total Images: {total:,}")
    print(f"   Train: {stats['totals']['train']:,} ({stats['totals']['train']/total*100:.1f}%)")
    print(f"   Val: {stats['totals']['val']:,} ({stats['totals']['val']/total*100:.1f}%)")
    print(f"   Test: {stats['totals']['test']:,} ({stats['totals']['test']/total*100:.1f}%)")
    print(f"   Total Classes: {len(stats['classes'])}")
    
    print(f"\n📋 All Classes ({len(stats['classes'])}):")
    for i, class_name in enumerate(stats['classes']):
        crop = "�" if "tomato" in class_name else "🌶️"
        print(f"   {i+1:2d}. {crop} {class_name}")
    
    # Save class mapping
    import json
    class_mapping = {
        'classes': stats['classes'],
        'num_classes': len(stats['classes']),
        'crops': {
            'tomato': [c for c in stats['classes'] if 'tomato' in c],
            'chilli': [c for c in stats['classes'] if 'chilli' in c]
        },
        'stats': stats['totals']
    }
    
    mapping_path = os.path.join(OUTPUT_DIR, "class_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"\n✓ Class mapping saved: {mapping_path}")
    
    print("\n" + "="*70)
    print("✅ Ready for AgriLite-Hybrid training!")
    print("   Run: python scripts/train_hybrid.py")
    print("="*70)
    
    return stats


def verify_combined_dataset():
    """Verify the combined dataset structure."""
    
    print("\n" + "="*70)
    print("🔍 VERIFYING COMBINED DATASET")
    print("="*70)
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Combined dataset not found at: {OUTPUT_DIR}")
        return False
    
    valid = True
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split)
        
        if not os.path.exists(split_dir):
            print(f"⚠️ Missing split: {split}")
            valid = False
            continue
        
        classes = [d for d in os.listdir(split_dir) 
                  if os.path.isdir(os.path.join(split_dir, d))]
        
        print(f"\n  {split}/ - {len(classes)} classes")
        
        # Count images per class
        total_images = 0
        for class_name in sorted(classes):
            class_dir = os.path.join(split_dir, class_name)
            images = len(get_image_files(class_dir))
            total_images += images
            
            if images == 0:
                print(f"    ⚠️ {class_name}: Empty!")
                valid = False
        
        print(f"      Total: {total_images} images")
    
    if valid:
        print("\n✅ Dataset verification passed!")
    else:
        print("\n⚠️ Dataset has issues. Consider re-running combiner.")
    
    return valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine crop datasets for AgriLite-Hybrid")
    parser.add_argument('--verify', action='store_true', help="Verify existing dataset")
    parser.add_argument('--force', action='store_true', help="Force recreation")
    args = parser.parse_args()
    
    if args.verify:
        verify_combined_dataset()
    else:
        if os.path.exists(OUTPUT_DIR) and not args.force:
            print(f"Combined dataset already exists at: {OUTPUT_DIR}")
            print("Use --force to recreate, or --verify to check it.")
            verify_combined_dataset()
        else:
            if args.force and os.path.exists(OUTPUT_DIR):
                print(f"Removing existing dataset: {OUTPUT_DIR}")
                # Use long_path() prefix - plain shutil.rmtree fails on Windows
                # with "directory not empty" / FileNotFoundError on long paths
                shutil.rmtree(long_path(OUTPUT_DIR))
            
            combine_datasets()
