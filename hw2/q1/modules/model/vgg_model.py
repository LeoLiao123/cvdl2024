import torch
import torchvision
import os
from pathlib import Path

class VGGModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        self.setup_model()

    def setup_model(self):
        # Initialize model instance
        self.model = torchvision.models.vgg19_bn(num_classes=10)
        
        # Specify the exact model path
        model_path = Path('train/logs/best_model_adam.pth')
        
        try:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(
                str(model_path),
                map_location=self.device,
                weights_only=True
            )
            
            # Load model state dict
            self.model.load_state_dict(checkpoint['model'])
            print(f"Successfully loaded model - Epoch: {checkpoint['epoch']}, "
                  f"Accuracy: {checkpoint['acc']:.2f}%")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Warning: Using untrained model!")
        
        # Move model to correct device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_tensor):
        """
        Perform model prediction
        Args:
            image_tensor: Preprocessed image tensor [1, 3, 32, 32]
        Returns:
            predicted_class: Index of predicted class
            probabilities: Probabilities for each class
        """
        with torch.no_grad():
            # Ensure input tensor is on correct device
            if image_tensor.device != self.device:
                image_tensor = image_tensor.to(self.device)
            
            # Get model output
            outputs = self.model(image_tensor)
            
            # Calculate probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get predicted class
            predicted_class = outputs.argmax(1).item()
            
        return predicted_class, probabilities