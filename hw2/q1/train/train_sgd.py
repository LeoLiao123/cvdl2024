import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import optuna
from torchvision.models import vgg19_bn, VGG19_BN_Weights
import logging
import os
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Setup logging
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloaders(batch_size):
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

    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    
    # Split into train and validation
    train_idx, val_idx = train_test_split(
        list(range(len(full_train_dataset))), test_size=0.2, random_state=42)
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def create_model():
    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    model.classifier[-1] = nn.Linear(4096, 10)
    return model.to(device)

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def objective(trial):
    # Hyperparameters to optimize
    batch_size = trial.suggest_int('batch_size', 64, 256, step=64)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Scheduler parameters
    scheduler_type = trial.suggest_categorical('scheduler', ['onecycle', 'cosine'])
    
    if scheduler_type == 'onecycle':
        pct_start = trial.suggest_float('pct_start', 0.2, 0.4)
        div_factor = trial.suggest_float('div_factor', 10, 30)
        final_div_factor = trial.suggest_float('final_div_factor', 100, 1000)
    
    # Create dataloaders and model
    train_loader, val_loader = create_dataloaders(batch_size)
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, 
                         momentum=momentum, 
                         weight_decay=weight_decay)
    
    if scheduler_type == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=5,  # Using 5 epochs for quick evaluation
            steps_per_epoch=len(train_loader),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5 * len(train_loader)
        )
    
    # Quick training for 5 epochs to evaluate hyperparameters
    best_val_acc = 0
    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def main():
    # Create a study object and specify the direction is 'maximize'
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Get the best parameters
    best_params = study.best_params
    logging.info(f'Best parameters: {best_params}')
    
    # Train the final model with the best parameters
    batch_size = best_params['batch_size']
    train_loader, val_loader = create_dataloaders(batch_size)
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    
    # Setup best optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=best_params['lr'],
        momentum=best_params['momentum'],
        weight_decay=best_params['weight_decay']
    )
    
    # Setup best scheduler
    if best_params['scheduler'] == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_params['lr'],
            epochs=100,  # Full training
            steps_per_epoch=len(train_loader),
            pct_start=best_params['pct_start'],
            div_factor=best_params['div_factor'],
            final_div_factor=best_params['final_div_factor']
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100 * len(train_loader)
        )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Full training with best parameters
    best_acc = 0
    for epoch in range(100):
        # Train and evaluate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'logs/best_model.pth')
        
        logging.info(f'Epoch: {epoch+1}/100 | '
                    f'Train Loss: {train_loss:.3f} | '
                    f'Train Acc: {train_acc:.2f}% | '
                    f'Val Loss: {val_loss:.3f} | '
                    f'Val Acc: {val_acc:.2f}%')
        
        # Plot training curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Val')
            plt.title('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train')
            plt.plot(history['val_acc'], label='Val')
            plt.title('Accuracy')
            plt.legend()
            
            plt.savefig(f'logs/training_curve_epoch_{epoch+1}.png')
            plt.close()

if __name__ == '__main__':
    main()