# WARNING: template code, may need edits
import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import Config

def setup_data_folders():
    """Create the required folder structure for data"""
    folders = [
        Config.TRAIN_DIR,
        Config.VAL_DIR,
        os.path.join(Config.TRAIN_DIR, 'cat'),
        os.path.join(Config.TRAIN_DIR, 'not_cat'),
        os.path.join(Config.VAL_DIR, 'cat'),
        os.path.join(Config.VAL_DIR, 'not_cat'),
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("Data folder structure created!")
    print("\nPlease organize your images as follows:")
    print(f"  {Config.TRAIN_DIR}/cat/       <- Training cat images")
    print(f"  {Config.TRAIN_DIR}/not_cat/   <- Training non-cat images")
    print(f"  {Config.VAL_DIR}/cat/         <- Validation cat images")
    print(f"  {Config.VAL_DIR}/not_cat/     <- Validation non-cat images")

def split_dataset(source_folder, train_ratio=0.8):
    """Split images from a source folder into train and validation sets"""
    if not os.path.exists(source_folder):
        print(f"Error: Source folder {source_folder} does not exist")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(source_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {source_folder}")
        return
    
    # Shuffle images
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Determine class from folder name
    class_name = os.path.basename(source_folder)
    if class_name not in Config.CLASS_NAMES:
        print(f"Warning: Unknown class name '{class_name}'")
        return
    
    # Create destination folders
    train_dest = os.path.join(Config.TRAIN_DIR, class_name)
    val_dest = os.path.join(Config.VAL_DIR, class_name)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    # Copy files
    print(f"\nSplitting {len(image_files)} images from {source_folder}")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    
    for img in train_files:
        shutil.copy2(os.path.join(source_folder, img), 
                    os.path.join(train_dest, img))
    
    for img in val_files:
        shutil.copy2(os.path.join(source_folder, img), 
                    os.path.join(val_dest, img))
    
    print("Split completed!")

def visualize_samples(data_dir, num_samples=9):
    """Visualize random samples from the dataset"""
    samples = []
    labels = []
    
    for class_idx, class_name in enumerate(Config.CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sample random images
        sample_files = random.sample(image_files, 
                                    min(num_samples // 2, len(image_files)))
        
        for img_file in sample_files:
            img_path = os.path.join(class_dir, img_file)
            samples.append(img_path)
            labels.append(class_name)
    
    # Plot samples
    num_samples = len(samples)
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (img_path, label) in enumerate(zip(samples, labels)):
        if idx >= len(axes):
            break
        
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f"Class: {label}", fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = f"{Config.OUTPUT_DIR}/sample_images.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved to {output_path}")
    plt.close()

def check_dataset_balance():
    """Check and report dataset balance"""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    for split_name, split_dir in [('Training', Config.TRAIN_DIR), 
                                   ('Validation', Config.VAL_DIR)]:
        print(f"\n{split_name} Set:")
        total = 0
        for class_name in Config.CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {class_name}: {count} images")
                total += count
            else:
                print(f"  {class_name}: 0 images (folder not found)")
        print(f"  Total: {total} images")
    
    print("="*50)

if __name__ == "__main__":
    # Setup folders
    setup_data_folders()
    
    # Check dataset
    check_dataset_balance()
