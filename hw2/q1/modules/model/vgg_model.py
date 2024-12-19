import torch
import torchvision
import os

class VGGModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        self.setup_model()

    def setup_model(self):
        self.model = torchvision.models.vgg19_bn(num_classes=10)
        if os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = outputs.argmax(1).item()
        return predicted_class, probabilities
