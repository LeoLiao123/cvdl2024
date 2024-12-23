import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg19_bn, VGG19_BN_Weights
import numpy as np
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('logs', f'training_{timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load datasets
batch_size = 128
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=train_transform)

# Split train dataset into train and validation
train_idx, val_idx = train_test_split(list(range(len(full_train_dataset))), 
                                     test_size=0.2, random_state=42)

train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_train_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                       shuffle=False, num_workers=4, pin_memory=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

# Modify VGG19 for CIFAR10
def create_model():
    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    # Modify first conv layer for 32x32 input
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    # Modify classifier for 10 classes
    model.classifier[-1] = nn.Linear(4096, 10)
    return model

model = create_model().to(device)

# Training hyperparameters
num_epochs = 150  

# Loss and optimizer setup
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), 
                       lr=0.001,              # Lower learning rate for Adam-based optimizers
                       weight_decay=0.05,     # Higher weight decay for AdamW
                       betas=(0.9, 0.999))    # Default momentum parameters

# Use OneCycleLR scheduler, which works well with Adam/AdamW
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,              # Maximum learning rate
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,             # Percentage of training time for warmup
    div_factor=25.0,           # Initial learning rate = max_lr/div_factor
    final_div_factor=1000.0    # Final learning rate = max_lr/(div_factor*final_div_factor)
)

# Mixed precision training
scaler = torch.amp.GradScaler('cuda')

def train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, scheduler, epoch):
    """Train model for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(train_loader)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step the scheduler
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print training progress
        if (batch_idx + 1) % 20 == 0:
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            logging.info(f'Epoch: {epoch+1} | Batch: {batch_idx+1}/{batch_count} | '
                      f'Loss: {current_loss:.3f} | Acc: {current_acc:.2f}% | '
                      f'LR: {lr:.6f}')
    
    final_loss = running_loss / len(train_loader)
    final_acc = 100. * correct / total
    return final_loss, final_acc

def evaluate(model, data_loader, loss_fn):
    """Evaluate model on the provided data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return accuracy, avg_loss

# Initialize history dictionary
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

def plot_training_history(history, save_path=None):
    """Plot training history."""
    if save_path is None:
        save_path = os.path.join('logs', f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    plt.figure(figsize=(10, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='red')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f'Training history plot saved to {save_path}')

# Training loop
best_acc = 0
logging.info(f"Starting training for {num_epochs} epochs")
start_time = time.time()

for epoch in range(num_epochs):
    # Train and evaluate
    epoch_start_time = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, scheduler, epoch)
    val_acc, val_loss = evaluate(model, val_loader, loss_fn)
    
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Plot training history
    if (epoch + 1) % 5 == 0:  # Plot every 5 epochs
        plot_training_history(history)
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    
    # Save best model
    if val_acc > best_acc:
        logging.info(f'New best validation accuracy: {val_acc:.2f}% (previous: {best_acc:.2f}%)')
        state = {
            'model': model.state_dict(),
            'acc': val_acc,
            'epoch': epoch,
        }
        model_path = os.path.join('logs', 'best_model_adam.pth')
        torch.save(state, model_path)
        best_acc = val_acc
    
    # Log epoch summary
    logging.info(f'Epoch: {epoch+1:3d}/{num_epochs} | '
              f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}% | '
              f'Time: {epoch_time:.1f}s')

# Log final training summary and plot final history
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

logging.info(f'Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}')
logging.info(f'Best accuracy: {best_acc:.2f}%')

# Save final plot
plot_training_history(history, os.path.join('logs', 'final_training_history.png'))