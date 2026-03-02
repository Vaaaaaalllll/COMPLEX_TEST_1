# WARNING: template code, may need edits
#!/usr/bin/env python3
"""
Helper script to download sample cat images for testing.
This downloads a small subset of images to get you started.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import setup_data_folders

def download_sample_images():
    """
    Download sample images for quick testing.
    Note: For production, you'll need a proper dataset.
    """
    
    print("="*60)
    print("SAMPLE DATA DOWNLOADER")
    print("="*60)
    print("\nThis script will download a few sample images for testing.")
    print("For real training, you need to collect your own dataset.")
    print("\nRecommended datasets:")
    print("  - Kaggle: Dogs vs Cats dataset")
    print("  - ImageNet")
    print("  - Your own collected images")
    print("="*60)
    
    # Create folders
    setup_data_folders()
    
    # Sample URLs (placeholder - replace with actual URLs)
    sample_images = {
        'cat': [
            # Add sample cat image URLs here
            # For example from Unsplash, Pexels, etc.
        ],
        'not_cat': [
            # Add sample non-cat image URLs here
        ]
    }
    
    print("\n Note: This script is a template.")
    print("Please add your own image URLs or use a proper dataset.")
    print("\nSuggested approach:")
    print("1. Download Kaggle Dogs vs Cats dataset")
    print("2. Use our split_dataset utility to organize images")
    print("3. Start training!")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"\nData folders created at: {Config.DATA_DIR}")
    print("\nNext steps:")
    print("1. Add your images to the data folders")
    print("2. Run: python src/train.py")
    print("="*60)

if __name__ == "__main__":
    download_sample_images()
