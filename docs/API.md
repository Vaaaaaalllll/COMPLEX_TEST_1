# WARNING: template code, may need edits
# API Documentation

## Overview

This document describes how to use the cat classifier programmatically in your Python code.

---

## Quick Start

```python
from src.model import load_model
from src.inference import Inferencer

# Load model
inferencer = Inferencer('models/best_model.pth')

# Predict single image
prediction, confidence = inferencer.predict('my_cat.jpg')
print(f"Prediction: {prediction} ({confidence*100:.1f}%)")
```

---

## Core Modules

### 1. Config Module

```python
from src.config import Config

# Access configuration
print(Config.DEVICE)          # cuda or cpu
print(Config.BATCH_SIZE)      # 32
print(Config.NUM_EPOCHS)      # 30

# Create directories
Config.create_dirs()

# Print configuration
Config.print_config()
```

### 2. Model Module

#### Create Model

```python
from src.model import create_model

model = create_model(
    num_classes=2,
    pretrained=True
)
```

#### Load Model

```python
from src.model import load_model
import torch

model = load_model(
    model_path='models/best_model.pth',
    device=torch.device('cuda')
)
```

#### Save Model

```python
from src.model import save_model

save_model(model, 'models/my_model.pth')
```

#### Count Parameters

```python
from src.model import count_parameters

num_params = count_parameters(model)
print(f"Parameters: {num_params:,}")
```

### 3. Dataset Module

#### Create Dataset

```python
from src.dataset import CatDataset, get_transforms

# Create dataset
transform = get_transforms(is_training=True)
dataset = CatDataset(
    root_dir='data/train',
    transform=transform
)

print(f"Dataset size: {len(dataset)}")

# Get single item
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
```

#### Create DataLoaders

```python
from src.dataset import get_dataloaders

train_loader, val_loader = get_dataloaders()

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Iterate through batches
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    break
```

### 4. Training Module

#### Basic Training

```python
from src.train import Trainer

trainer = Trainer()
trainer.train()
```

#### Custom Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import create_model
from src.dataset import get_dataloaders
from src.config import Config

# Setup
model = create_model().to(Config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader, val_loader = get_dataloaders()

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} completed")
```

### 5. Testing Module

#### Run Testing

```python
from src.test import Tester

tester = Tester(model_path='models/best_model.pth')
accuracy, loss = tester.test()

print(f"Accuracy: {accuracy:.2f}%")
print(f"Loss: {loss:.4f}")
```

### 6. Inference Module

#### Single Image Inference

```python
from src.inference import Inferencer

inferencer = Inferencer('models/best_model.pth')

# Predict with visualization
prediction, confidence = inferencer.predict(
    'my_image.jpg',
    visualize=True
)

print(f"Class: {prediction}")
print(f"Confidence: {confidence*100:.1f}%")
```

#### Batch Inference

```python
inferencer.predict_batch('path/to/image/folder')
```

#### Custom Inference

```python
import torch
from PIL import Image
from src.model import load_model
from src.dataset import get_transforms
from src.config import Config

# Load model
model = load_model('models/best_model.pth')

# Prepare image
transform = get_transforms(is_training=False)
image = Image.open('my_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = probabilities.max(1)

print(f"Predicted class: {Config.CLASS_NAMES[predicted.item()]}")
print(f"Confidence: {confidence.item()*100:.1f}%")
```

### 7. Utilities Module

#### Setup Data Folders

```python
from src.utils import setup_data_folders

setup_data_folders()
```

#### Split Dataset

```python
from src.utils import split_dataset

split_dataset(
    source_folder='raw_data/cats',
    train_ratio=0.8
)
```

#### Visualize Samples

```python
from src.utils import visualize_samples

visualize_samples(
    data_dir='data/train',
    num_samples=9
)
```

#### Check Dataset Balance

```python
from src.utils import check_dataset_balance

check_dataset_balance()
```

---

## Classes

### CatClassifier

```python
class CatClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        # Initialize model
        pass
    
    def forward(self, x):
        # Forward pass
        return output
```

### CatDataset

```python
class CatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Initialize dataset
        pass
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return image, label
```

### Trainer

```python
class Trainer:
    def __init__(self):
        # Initialize trainer
        pass
    
    def train_epoch(self, epoch):
        # Train one epoch
        return loss, accuracy
    
    def validate(self, epoch):
        # Validate model
        return loss, accuracy
    
    def train(self):
        # Main training loop
        pass
```

### Tester

```python
class Tester:
    def __init__(self, model_path):
        # Initialize tester
        pass
    
    def test(self):
        # Run testing
        return accuracy, loss
    
    def plot_confusion_matrix(self, true_labels, predictions):
        # Plot confusion matrix
        pass
```

### Inferencer

```python
class Inferencer:
    def __init__(self, model_path):
        # Initialize inferencer
        pass
    
    def preprocess_image(self, image_path):
        # Preprocess image
        return original_image, image_tensor
    
    def predict(self, image_path, visualize=True):
        # Make prediction
        return predicted_class, confidence
    
    def predict_batch(self, image_folder):
        # Batch prediction
        pass
```

---

## Configuration Options

### Paths
```python
Config.DATA_DIR = 'data'
Config.TRAIN_DIR = 'data/train'
Config.VAL_DIR = 'data/val'
Config.MODEL_DIR = 'models'
Config.OUTPUT_DIR = 'outputs'
Config.LOG_DIR = 'runs'
```

### Model Settings
```python
Config.MODEL_NAME = 'efficientnet_b0'
Config.NUM_CLASSES = 2
Config.INPUT_SIZE = 224
Config.PRETRAINED = True
```

### Training Settings
```python
Config.BATCH_SIZE = 32
Config.NUM_EPOCHS = 30
Config.LEARNING_RATE = 0.001
Config.WEIGHT_DECAY = 1e-4
```

### Hardware Settings
```python
Config.DEVICE = torch.device('cuda')
Config.NUM_WORKERS = 4
Config.PIN_MEMORY = True
```

---

## Examples

### Example 1: Train Custom Model

```python
from src.config import Config
from src.train import Trainer

# Modify config
Config.NUM_EPOCHS = 50
Config.BATCH_SIZE = 16
Config.LEARNING_RATE = 0.0005

# Train
trainer = Trainer()
trainer.train()
```

### Example 2: Evaluate Model

```python
from src.test import Tester

tester = Tester('models/best_model.pth')
accuracy, loss = tester.test()

if accuracy > 90:
    print("Excellent model!")
elif accuracy > 80:
    print("Good model!")
else:
    print("Needs improvement")
```

### Example 3: Batch Prediction

```python
from src.inference import Inferencer
import os

inferencer = Inferencer('models/best_model.pth')

image_folder = 'test_images'
for img_file in os.listdir(image_folder):
    if img_file.endswith('.jpg'):
        img_path = os.path.join(image_folder, img_file)
        pred, conf = inferencer.predict(img_path, visualize=False)
        print(f"{img_file}: {pred} ({conf*100:.1f}%)")
```

### Example 4: Transfer Learning

```python
import torch.nn as nn
from src.model import create_model

# Load pretrained model
model = create_model(pretrained=True)

# Freeze early layers
for param in model.backbone.features.parameters():
    param.requires_grad = False

# Only train classifier
for param in model.backbone.classifier.parameters():
    param.requires_grad = True

print("Transfer learning setup complete")
```

---

## Error Handling

```python
try:
    from src.inference import Inferencer
    
    inferencer = Inferencer('models/best_model.pth')
    prediction, confidence = inferencer.predict('image.jpg')
    
except FileNotFoundError:
    print("Model or image not found")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Best Practices

1. **Always use Config**: Don't hardcode values
2. **Check CUDA availability**: Before training
3. **Save checkpoints**: During long training
4. **Validate regularly**: Monitor overfitting
5. **Use try-except**: Handle errors gracefully
6. **Log everything**: Use tensorboard
7. **Test thoroughly**: Before deployment

---

## Performance Tips

1. **Reduce batch size** if out of memory
2. **Use pin_memory=True** for faster data loading
3. **Increase num_workers** for faster preprocessing
4. **Use mixed precision** for faster training (advanced)
5. **Profile your code** to find bottlenecks

---

For more examples, see the source code in the `src/` directory.