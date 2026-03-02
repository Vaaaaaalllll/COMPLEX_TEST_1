# WARNING: template code, may need edits
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model import create_model, save_model, count_parameters
from dataset import get_dataloaders

class Trainer:
    """Trainer class for the cat classifier"""
    
    def __init__(self):
        # Create necessary directories
        Config.create_dirs()
        Config.print_config()
        
        # Initialize model
        print("\nInitializing model...")
        self.model = create_model().to(Config.DEVICE)
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        # Data loaders
        print("\nLoading datasets...")
        self.train_loader, self.val_loader = get_dataloaders()
        
        # Tensorboard writer
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        # Best validation loss for model saving
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50 + "\n")
        
        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print("-" * 50)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(self.model, Config.BEST_MODEL_PATH)
                print(f"496 Best model saved! (Val Loss: {val_loss:.4f})")
            
            # Save checkpoint periodically
            if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
                save_model(self.model, Config.LAST_MODEL_PATH)
                print(f"496 Checkpoint saved at epoch {epoch+1}")
        
        # Save final model
        save_model(self.model, Config.LAST_MODEL_PATH)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved at: {Config.BEST_MODEL_PATH}")
        print("="*50)
        
        self.writer.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

# Optional main for setuptools console_scripts

def main():
    trainer = Trainer()
    trainer.train()
