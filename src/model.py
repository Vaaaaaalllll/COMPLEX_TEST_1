# WARNING: template code, may need edits
import torch
import torch.nn as nn
from torchvision import models
from config import Config

class CatClassifier(nn.Module):
    """Cat classifier using EfficientNet-B0 backbone"""
    
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(CatClassifier, self).__init__()
        
        # Load EfficientNet-B0 (lightweight and efficient)
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """Factory function to create the model"""
    model = CatClassifier(num_classes=num_classes, pretrained=pretrained)
    return model

def load_model(model_path, device=Config.DEVICE):
    """Load a saved model from disk"""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_model(model, path):
    """Save model to disk"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created successfully!")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
