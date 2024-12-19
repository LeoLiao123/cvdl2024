# modules/windows/augmentation_window.py
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PIL import Image
import os
from utils.transforms import ImageTransforms

class AugmentationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Augmented Images")
        self.setMinimumSize(800, 600)
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout()
        self.setLayout(layout)
        
        image_folder = "Q1_image/Q1_1/"
        transform = ImageTransforms.get_augmentation_transforms()
        
        try:
            image_files = [f for f in os.listdir(image_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:9]
            
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                augmented_image = transform(image)
                
                # Convert PIL image to QPixmap
                q_image = self.pil_to_pixmap(augmented_image)
                label = QLabel()
                label.setFixedSize(200, 200)
                label.setPixmap(q_image.scaled(200, 200, Qt.KeepAspectRatio))
                layout.addWidget(label, i // 3, i % 3)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading or augmenting images: {str(e)}")
            
    def pil_to_pixmap(self, pil_image):
        bytes_img = QByteArray()
        buffer = QBuffer(bytes_img)
        buffer.open(QBuffer.WriteOnly)
        pil_image.save(buffer, format='PNG')
        pixmap = QPixmap()
        pixmap.loadFromData(bytes_img)
        return pixmap
