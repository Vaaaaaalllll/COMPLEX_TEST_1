# WARNING: template code, may need edits
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import Config
from model import load_model
from dataset import get_dataloaders

class Tester:
    """Tester class for evaluating the cat classifier"""
    
    def __init__(self, model_path=Config.BEST_MODEL_PATH):
        Config.create_dirs()
        
        print("Loading model...")
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        print("\nLoading validation dataset...")
        _, self.val_loader = get_dataloaders()
        
        self.criterion = nn.CrossEntropyLoss()
    
    def test(self):
        """Run testing on validation set"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        running_loss = 0.0
        correct = 0
        total = 0
        
        print("\nRunning evaluation...")
        pbar = tqdm(self.val_loader, desc="Testing")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                _, predicted = outputs.max(1)
                
                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Statistics
                running_loss += loss.item()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / len(all_labels) * Config.BATCH_SIZE,
                    'acc': 100. * correct / total
                })
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Correct: {correct}/{total}")
        print("="*50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, 
            all_predictions, 
            target_names=Config.CLASS_NAMES,
            digits=4
        ))
        
        # Confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        return accuracy, avg_loss
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=Config.CLASS_NAMES,
                    yticklabels=Config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        output_path = f"{Config.OUTPUT_DIR}/confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {output_path}")
        plt.close()

if __name__ == "__main__":
    import sys
    
    # Allow custom model path
    model_path = sys.argv[1] if len(sys.argv) > 1 else Config.BEST_MODEL_PATH
    
    tester = Tester(model_path)
    tester.test()

# Optional main for setuptools console_scripts

def main():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else Config.BEST_MODEL_PATH
    tester = Tester(model_path)
    tester.test()
