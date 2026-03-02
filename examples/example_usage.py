# WARNING: template code, may need edits
#!/usr/bin/env python3
"""
Example usage of the cat classifier.
This script demonstrates common workflows.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.model import create_model, load_model, count_parameters
from src.inference import Inferencer
from src.utils import check_dataset_balance, visualize_samples

def example_1_check_setup():
    """Example 1: Check if everything is set up correctly"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Checking Setup")
    print("="*60)
    
    # Print configuration
    Config.print_config()
    
    # Check dataset
    check_dataset_balance()
    
    # Create model to verify
    print("\nCreating model to verify setup...")
    model = create_model()
    print(f"44d Model created successfully!")
    print(f"44d Total parameters: {count_parameters(model):,}")
    print(f"44d Device: {Config.DEVICE}")
    
    print("\n44d Setup looks good! Ready to train.")

def example_2_visualize_data():
    """Example 2: Visualize training data"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Visualizing Data")
    print("="*60)
    
    try:
        visualize_samples('data/train', num_samples=9)
        print("\n44d Sample images saved to outputs/sample_images.png")
    except Exception as e:
        print(f"\n6a7 Error: {e}")
        print("Make sure you have images in data/train/cat and data/train/not_cat")

def example_3_quick_inference():
    """Example 3: Quick inference on a single image"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Quick Inference")
    print("="*60)
    
    model_path = Config.BEST_MODEL_PATH
    
    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"\n6a7 Model not found at {model_path}")
        print("Please train the model first: python src/train.py")
        return
    
    # Load inferencer
    print(f"\nLoading model from {model_path}...")
    inferencer = Inferencer(model_path)
    
    # Example prediction (you need to provide an actual image path)
    print("\nTo make a prediction, use:")
    print("  python src/inference.py --image path/to/your/image.jpg")
    print("\nOr in Python:")
    print("  prediction, confidence = inferencer.predict('my_image.jpg')")

def example_4_programmatic_training():
    """Example 4: Training with custom parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Training (Code Example)")
    print("="*60)
    
    print("\nTo train with custom parameters:")
    print("""
# Modify config
from src.config import Config
Config.NUM_EPOCHS = 50
Config.BATCH_SIZE = 16
Config.LEARNING_RATE = 0.0005

# Train
from src.train import Trainer
trainer = Trainer()
trainer.train()
    """)

def example_5_batch_prediction():
    """Example 5: Batch prediction on multiple images"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Prediction (Code Example)")
    print("="*60)
    
    print("\nTo predict on multiple images:")
    print("""
from src.inference import Inferencer

inferencer = Inferencer('models/best_model.pth')
inferencer.predict_batch('path/to/image/folder')
    """)
    
    print("\nOr from command line:")
    print("  python src/inference.py --folder path/to/image/folder")

def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "CAT CLASSIFIER EXAMPLES" + " "*20 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    examples = [
        ("Check Setup", example_1_check_setup),
        ("Visualize Data", example_2_visualize_data),
        ("Quick Inference", example_3_quick_inference),
        ("Custom Training", example_4_programmatic_training),
        ("Batch Prediction", example_5_batch_prediction),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n6a7 Error in {name}: {e}")
    
    print("\n" + "#"*60)
    print("Examples completed!")
    print("\nFor more information, see:")
    print("  - docs/GUIDE.md for detailed guide")
    print("  - docs/API.md for API documentation")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
