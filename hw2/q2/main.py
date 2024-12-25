import os
import sys

# Force using X11 instead of Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Suppress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules.gan_model import Generator, Discriminator
from modules.data_loader import MNISTLoader
from utils.display import show_image_grid, create_comparison_window
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCGAN MNIST Viewer")
        self.setMinimumSize(400, 300)
        
        self._setup_ui()
        self._initialize_components()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.btn_show_images = QPushButton("1. Show Training Images")
        self.btn_show_model = QPushButton("2. Show Model Structure")
        self.btn_show_loss = QPushButton("3. Show Training Loss")
        self.btn_inference = QPushButton("4. Inference")
        
        layout.addWidget(self.btn_show_images)
        layout.addWidget(self.btn_show_model)
        layout.addWidget(self.btn_show_loss)
        layout.addWidget(self.btn_inference)
        
        self.btn_show_images.clicked.connect(self.show_training_images)
        self.btn_show_model.clicked.connect(self.show_model_structure)
        self.btn_show_loss.clicked.connect(self.show_training_loss)
        self.btn_inference.clicked.connect(self.run_inference)
        
    def _initialize_components(self):
        self.data_loader = MNISTLoader()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.load_trained_model()
        
    def load_trained_model(self):
        """Load the pre-trained generator model"""
        try:
            # First try loading with the original key
            checkpoint = torch.load('model/generator.pth', map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['model_state_dict'])
            elif 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                # If it's just the state dict directly
                self.generator.load_state_dict(checkpoint)
            
            self.generator.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def show_training_images(self):
        try:
            original_images, augmented_images = self.data_loader.get_sample_images(64)
            create_comparison_window("Training Images", original_images, 
                                  "Augmented Images", augmented_images)
        except Exception as e:
            print(f"Error showing training images: {e}")
            
    def show_model_structure(self):
        generator = Generator()
        discriminator = Discriminator()
        print("\nGenerator Structure:")
        print(generator)
        print("\nDiscriminator Structure:")
        print(discriminator)
        
    def show_training_loss(self):
        try:
            loss_img = plt.imread('results/loss.png')
            plt.figure(figsize=(10, 6))
            plt.imshow(loss_img)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error showing loss plot: {e}")
            
    def run_inference(self):
        """Generate and display MNIST-like images using the trained generator"""
        try:
            with torch.no_grad():
                # Create noise vector for generating images
                noise = torch.randn(64, 100, 1, 1, device=self.device)
                
                # Generate fake images
                generated_images = self.generator(noise)
                
                # Move images to CPU and denormalize from [-1, 1] to [0, 1] range
                generated_images = generated_images.cpu()
                generated_images = (generated_images + 1) / 2.0

                # Create results directory if it doesn't exist
                os.makedirs('results', exist_ok=True)
                
                # Save individual images
                for idx in range(generated_images.size(0)):
                    img = generated_images[idx].squeeze()  # Remove channel dimension
                    plt.figure(figsize=(2, 2))
                    plt.axis('off')
                    plt.imshow(img, cmap='gray')
                    plt.savefig(f'results/generated_{idx}.png', 
                            bbox_inches='tight', 
                            pad_inches=0)
                    plt.close()
                
                # Display grid of generated images
                plt.figure(figsize=(10, 10))
                plt.suptitle('Generated Images', fontsize=16)
                
                grid_size = int(np.sqrt(generated_images.size(0)))
                for idx in range(grid_size * grid_size):
                    plt.subplot(grid_size, grid_size, idx + 1)
                    plt.axis('off')
                    img = generated_images[idx].squeeze()
                    plt.imshow(img, cmap='gray')
                
                # Save and show the grid
                plt.savefig('results/generated_grid.png', 
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0.1)
                plt.show()
                
                print(f"Generated images have been saved to the 'results' directory")
                
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()

def handle_exception(exc_type, exc_value, exc_traceback):
    """Custom exception handler to log errors"""
    print(f"An error occurred: {exc_value}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)  # Call the default handler

if __name__ == "__main__":
    # Set up custom exception handler
    sys.excepthook = handle_exception
    
    # Create and run application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())