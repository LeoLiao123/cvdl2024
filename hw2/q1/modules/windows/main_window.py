from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import torch
from PIL import Image
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from modules.model.vgg_model import VGGModel
from utils.transforms import ImageTransforms
from utils.visualization import Visualizer
from .augmentation_window import AugmentationWindow
from .history_window import HistoryWindow
from .probability_window import ProbabilityWindow
from torchsummary import summary

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Learning Demo")
        self.setup()
        self.init_ui()

    def setup(self):
        """Initialize model and transforms"""
        self.model = VGGModel()
        self.transform = ImageTransforms.get_inference_transforms()
        self.current_image_path = None
        self.windows = []  # Store references to child windows

    def init_ui(self):
        """Initialize the user interface"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create top control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Create buttons with styled appearance
        buttons_data = [
            ("Load Image", self.load_image),
            ("1. Show Augmented Images", self.show_augmented),
            ("2. Show Model Structure", self.show_structure),
            ("3. Show Accuracy and Loss", self.show_accuracy),
            ("4. Inference", self.do_inference)
        ]
        
        for text, slot in buttons_data:
            btn = QPushButton(text)
            btn.setMinimumHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
                QPushButton:pressed {
                    background-color: #2a5f9e;
                }
            """)
            btn.clicked.connect(slot)
            control_layout.addWidget(btn)

        # Create image display area
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        
        # Image label with border and background
        self.image_label = QLabel()
        self.image_label.setFixedSize(256, 256)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #cccccc;
                border-radius: 5px;
            }
        """)
        
        # Prediction label with styling
        self.prediction_label = QLabel("Predicted = ")
        self.prediction_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: white;
                padding: 10px;
            }
        """)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.prediction_label)
        
        # Add all components to main layout
        layout.addWidget(control_panel)
        layout.addWidget(image_widget)
        
        # Set window properties
        self.setMinimumSize(400, 600)
        self.center_window()

    def center_window(self):
        """Center the window on the screen"""
        frame = self.frameGeometry()
        center_point = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

    def load_image(self):
        """Handle image loading"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            try:
                pixmap = QPixmap(file_name)
                scaled_pixmap = pixmap.scaled(
                    256, 256,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.current_image_path = file_name
                self.prediction_label.setText("Predicted = ")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")

    def show_augmented(self):
        """Show augmented images window"""
        try:
            augmentation_window = AugmentationWindow()
            augmentation_window.show()
            self.windows.append(augmentation_window)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error showing augmented images: {str(e)}")

    def show_structure(self):
        """Display model structure in terminal"""
        try:
            device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
            summary(self.model.model, (3, 32, 32), device=device)
        except Exception as e:
            print(f"Error generating model summary: {str(e)}")

    def show_accuracy(self):
        """Show accuracy and loss plots"""
        try:
            history_window = HistoryWindow()
            history_window.show()
            self.windows.append(history_window)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error showing accuracy plots: {str(e)}")

    def do_inference(self):
        """Perform inference on loaded image"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        try:
            # Prepare image
            image = Image.open(self.current_image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.model.device)
            
            # Get prediction
            predicted_class, probabilities = self.model.predict(image_tensor)
            
            # Update prediction label
            self.prediction_label.setText(
                f"Predicted = {self.model.classes[predicted_class]}"
            )
            
            # Show probability distribution
            prob_window = ProbabilityWindow(probabilities.cpu().numpy(), self.model.classes)
            prob_window.show()
            self.windows.append(prob_window)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Inference error: {str(e)}")

    def closeEvent(self, event):
        """Handle application closure"""
        # Close all child windows
        for window in self.windows:
            window.close()
        event.accept()