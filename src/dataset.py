# WARNING: template code, may need edits
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import Config

class CatDataset(Dataset):
    """Custom dataset for cat images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for class_idx, class_name in enumerate(Config.CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_training=True):
    """Get data transforms for training or validation"""
    
    if is_training and Config.USE_AUGMENTATION:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE + 32, Config.INPUT_SIZE + 32)),
            transforms.RandomCrop(Config.INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def get_dataloaders():
    """Create train and validation dataloaders"""
    
    # Create datasets
    train_dataset = CatDataset(
        root_dir=Config.TRAIN_DIR,
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = CatDataset(
        root_dir=Config.VAL_DIR,
        transform=get_transforms(is_training=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
