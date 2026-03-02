# WARNING: template code, may need edits
import torch
import os

class Config:
    """Configuration class for the cat classifier project"""
    
    # Paths
    DATA_DIR = 'data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    MODEL_DIR = 'models'
    OUTPUT_DIR = 'outputs'
    LOG_DIR = 'runs'
    
    # Model settings
    MODEL_NAME = 'efficientnet_b0'
    NUM_CLASSES = 2  # cat or not_cat
    INPUT_SIZE = 224
    PRETRAINED = True
    
    # Training settings
    BATCH_SIZE = 32  # Optimized for 4-8GB GPU
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Hardware settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Checkpoint settings
    SAVE_FREQUENCY = 5  # Save model every N epochs
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
    LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'last_model.pth')
    
    # Inference settings
    CONFIDENCE_THRESHOLD = 0.5
    
    # Class names
    CLASS_NAMES = ['not_cat', 'cat']
    
    @staticmethod
    def create_dirs():
        """Create necessary directories if they don't exist"""
        dirs = [Config.MODEL_DIR, Config.OUTPUT_DIR, Config.LOG_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @staticmethod
    def print_config():
        """Print current configuration"""
        print("="*50)
        print("CONFIGURATION")
        print("="*50)
        print(f"Device: {Config.DEVICE}")
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Input Size: {Config.INPUT_SIZE}")
        print("="*50)
