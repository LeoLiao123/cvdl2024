import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLineEdit, QGroupBox, QGridLayout
)
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
from utils.file_utils import load_images_from_folder
from modules.calibration import Calibration
from modules.aug_reality import AugmentedRealityProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow - PySide6 UI Example with Custom Button Sizes")
        self.setGeometry(100, 100, 800, 600)
        self.calibration = Calibration(height=11, width=8)
        self.aug_reality = AugmentedRealityProcessor()  
        self.structured_images = None
        self.left_img = None
        self.right_img = None      
        main_layout = QHBoxLayout()

        main_layout.addWidget(self.create_load_image_group())
        main_layout.addWidget(self.create_calibration_group())
        main_layout.addWidget(self.create_ar_group())
        main_layout.addWidget(self.create_stereo_group())
        main_layout.addWidget(self.create_sift_group())

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_button(self, text, size, callback=None):
        button = QPushButton(text)
        button.setFixedSize(*size)
        if callback:
            button.clicked.connect(callback)
        return button

    def create_load_image_group(self):
        group = QGroupBox("Load Image")
        layout = QVBoxLayout()
        layout.addWidget(self.create_button("Load folder", (100, 80), self.load_folder))
        layout.addWidget(self.create_button("Load Image_L", (100, 80), lambda : self.load_image("Left")))
        layout.addWidget(self.create_button("Load Image_R", (100, 80), lambda : self.load_image("Right")))
        group.setLayout(layout)
        return group

    def create_calibration_group(self):
        group = QGroupBox("1. Calibration")
        layout = QGridLayout()
        layout.addWidget(self.create_button("1.1 Find corners", (150, 40), self.find_corners), 0, 0)
        layout.addWidget(self.create_button("1.2 Find intrinsic", (150, 40), self.find_intrinsics), 1, 0)
        layout.addWidget(self.create_button("1.3 Find extrinsic", (150, 40), self.find_extrinsic), 2, 0)
        layout.addWidget(self.create_button("1.4 Find distortion", (150, 40), self.find_distortion), 3, 0)
        layout.addWidget(self.create_button("1.5 Show result", (150, 40),self.show_undistorted_result), 4, 0)
        
        # Set self.spin_box as an attribute
        self.spin_box = QSpinBox()
        self.spin_box.setRange(1, 15)
        layout.addWidget(self.spin_box, 2, 1)
        
        group.setLayout(layout)
        return group

    def create_ar_group(self):
        group = QGroupBox("2. Augmented Reality")
        layout = QGridLayout()
        self.line_edit = QLineEdit()
        # Set the validator
        regex = QRegularExpression("^[a-zA-Z]{0,6}$")  # Only allows 0 to 6 English letters
        validator = QRegularExpressionValidator(regex)
        self.line_edit.setValidator(validator)
        layout.addWidget(self.line_edit, 0, 0, 1, 2)
        layout.addWidget(self.create_button("2.1 Show words on board", (180, 40), self.show_words_on_board), 1, 0)
        layout.addWidget(self.create_button("2.2 Show words vertical", (180, 40), self.show_words_vertical), 1, 1)
        group.setLayout(layout)
        return group

    def create_stereo_group(self):
        group = QGroupBox("3. Stereo disparity map")
        layout = QVBoxLayout()
        layout.addWidget(self.create_button("3.1 Stereo disparity map", (200, 50)))
        group.setLayout(layout)
        return group

    def create_sift_group(self):
        group = QGroupBox("4. SIFT")
        layout = QGridLayout()
        layout.addWidget(self.create_button("Load Image1", (120, 40)), 0, 0)
        layout.addWidget(self.create_button("Load Image2", (120, 40)), 1, 0)
        layout.addWidget(self.create_button("4.1 Keypoints", (120, 40)), 2, 0)
        layout.addWidget(self.create_button("4.2 Matched Keypoints", (120, 40)), 3, 0)
        group.setLayout(layout)
        return group

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            structured_images = load_images_from_folder(folder)
            if structured_images:
                self.structured_images = structured_images
                print("Loaded images by folder structure:")
                for subfolder, images in structured_images.items():
                    print(f"{subfolder}: {len(images)} images")
                    print(images)
            else:
                print("No images found in the selected folder.")

    def load_image(self, side):
        image = QFileDialog.getOpenFileName(self, "Select Image")
        if image:
            print(f"{side} side image : {image[0]}")
            if side == "Left":
                self.left_img = image[0]
            else:
                self.right_img = image[0]
        else:
            print("No images found in the selected folder.")

    def find_corners(self):
        if hasattr(self, 'structured_images'):
            self.calibration.find_and_draw_corners(self.structured_images["Q1_Image"])
        else:
            print("No images loaded. Please load a folder first.")

    def find_intrinsics(self):
        self.calibration.find_intrinsics()

    def find_extrinsic(self):
        image_num = self.spin_box.value()  # Get the current value of the spin box
        self.calibration.find_extrinsic(image_num)

    def find_distortion(self):
        self.calibration.find_distortion()
    
    def show_undistorted_result(self):
        self.calibration.show_undistorted_result(self.structured_images["Q1_Image"])

    def show_words_on_board(self):
        if hasattr(self, 'structured_images'):
            self.aug_reality.project_word_on_board(self.structured_images["Q1_Image"], self.line_edit.text(), "horizontal")
        else:
            print("No images loaded. Please load a folder first.")

    def show_words_vertical(self):
        if hasattr(self, 'structured_images'):
            self.aug_reality.project_word_on_board(self.structured_images["Q1_Image"], self.line_edit.text(), "vertical")
        else:
            print("No images loaded. Please load a folder first.")        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
