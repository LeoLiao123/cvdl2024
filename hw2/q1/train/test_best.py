import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg19_bn
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_model():
    model = vgg19_bn(weights=None)  # No pre-trained weights needed for testing
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    model.classifier[-1] = nn.Linear(4096, 10)
    return model

def load_test_data(batch_size=128):
    # Define test transforms - same as training but without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    
    return accuracy, per_class_accuracy, class_names

def main():
    # Create and load model
    model = create_model().to(device)
    
    # Load the saved best model
    checkpoint = torch.load('logs/best_model_adam.pth', weights_only=True)  # Use weights_only=True for security
    model.load_state_dict(checkpoint['model'])  
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['acc']:.2f}%")
    
    # Load test data
    test_loader = load_test_data()
    
    # Evaluate on test set
    accuracy, per_class_accuracy, class_names = evaluate_model(model, test_loader)
    
    # Print results
    print(f"\nTest Set Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("\nPer-class Accuracy:")
    for class_name, class_acc in zip(class_names, per_class_accuracy):
        print(f"{class_name:>10}: {class_acc:.2f}%")
    
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    main()