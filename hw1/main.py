import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLineEdit, QGroupBox, QGridLayout
)
from utils.file_utils import load_images_from_folder
from modules.calibration import Calibration

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MainWindow - PySide6 UI Example with Custom Button Sizes")
        self.setGeometry(100, 100, 800, 600)
        self.calibration = Calibration(height=11, width=8)        
        main_layout = QHBoxLayout()

        # Load Image Section
        load_image_group = QGroupBox("Load Image")
        load_image_layout = QVBoxLayout()
        load_folder_button = QPushButton("Load folder")
        load_folder_button.setFixedSize(100, 80)  # Set fixed size
        load_folder_button.clicked.connect(self.load_folder)
        load_image_layout.addWidget(load_folder_button)

        load_image_l_button = QPushButton("Load Image_L")
        load_image_l_button.setMinimumSize(100, 80)  # Set minimum size
        load_image_layout.addWidget(load_image_l_button)

        load_image_r_button = QPushButton("Load Image_R")
        load_image_r_button.setMaximumSize(100, 80)  # Set maximum size
        load_image_layout.addWidget(load_image_r_button)

        load_image_group.setLayout(load_image_layout)
        main_layout.addWidget(load_image_group)

        # Calibration Section with Grid Layout for custom positioning
        calibration_group = QGroupBox("1. Calibration")
        calibration_layout = QGridLayout()

        find_corners_button = QPushButton("1.1 Find corners")
        find_corners_button.setFixedSize(150, 40)
        find_corners_button.clicked.connect(self.find_corners)
        calibration_layout.addWidget(find_corners_button, 0, 0)

        find_intrinsic_button = QPushButton("1.2 Find intrinsic")
        find_intrinsic_button.setFixedSize(150, 40)
        find_intrinsic_button.clicked.connect(self.find_instrinsics)
        calibration_layout.addWidget(find_intrinsic_button, 1, 0)

        find_extrinsic_button = QPushButton("1.3 Find extrinsic")
        find_extrinsic_button.setFixedSize(150, 40)
        find_extrinsic_button.clicked.connect(self.find_extrinsic)
        calibration_layout.addWidget(find_extrinsic_button, 2, 0)

        self.spin_box = QSpinBox()
        self.spin_box.setRange(1, 15)
        calibration_layout.addWidget(self.spin_box, 2, 1)

        find_distortion_button = QPushButton("1.4 Find distortion")
        find_distortion_button.setFixedSize(150, 40)
        calibration_layout.addWidget(find_distortion_button, 3, 0)

        show_result_button = QPushButton("1.5 Show result")
        show_result_button.setFixedSize(150, 40)
        calibration_layout.addWidget(show_result_button, 4, 0)

        calibration_group.setLayout(calibration_layout)
        main_layout.addWidget(calibration_group)

        # Augmented Reality Section with custom layout
        ar_group = QGroupBox("2. Augmented Reality")
        ar_layout = QGridLayout()
        input_field = QLineEdit()
        ar_layout.addWidget(input_field, 0, 0, 1, 2)

        show_words_on_board_button = QPushButton("2.1 Show words on board")
        show_words_on_board_button.setFixedSize(180, 40)
        ar_layout.addWidget(show_words_on_board_button, 1, 0)

        show_words_vertical_button = QPushButton("2.2 Show words vertical")
        show_words_vertical_button.setFixedSize(180, 40)
        ar_layout.addWidget(show_words_vertical_button, 1, 1)

        ar_group.setLayout(ar_layout)
        main_layout.addWidget(ar_group)

        # Stereo Disparity Map Section
        stereo_group = QGroupBox("3. Stereo disparity map")
        stereo_layout = QVBoxLayout()
        stereo_disparity_map_button = QPushButton("3.1 Stereo disparity map")
        stereo_disparity_map_button.setFixedSize(200, 50)
        stereo_layout.addWidget(stereo_disparity_map_button)
        stereo_group.setLayout(stereo_layout)
        main_layout.addWidget(stereo_group)

        # SIFT Section with custom layout
        sift_group = QGroupBox("4. SIFT")
        sift_layout = QGridLayout()
        
        load_image1_button = QPushButton("Load Image1")
        load_image1_button.setFixedSize(120, 40)
        sift_layout.addWidget(load_image1_button, 0, 0)
        
        load_image2_button = QPushButton("Load Image2")
        load_image2_button.setFixedSize(120, 40)
        sift_layout.addWidget(load_image2_button, 1, 0)
        
        keypoints_button = QPushButton("4.1 Keypoints")
        keypoints_button.setFixedSize(120, 40)
        sift_layout.addWidget(keypoints_button, 2, 0)
        
        matched_keypoints_button = QPushButton("4.2 Matched Keypoints")
        matched_keypoints_button.setFixedSize(120, 40)
        sift_layout.addWidget(matched_keypoints_button, 3, 0)

        sift_group.setLayout(sift_layout)
        main_layout.addWidget(sift_group)

        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            structured_images = load_images_from_folder(folder)
            if structured_images:
                print("Loaded images by folder structure:")
                for subfolder, images in structured_images.items():
                    print(f"{subfolder}: {len(images)} images")
                    for image in images:
                        print(f" - {image}")
                self.structured_images = structured_images
            else:
                print("No images found in the selected folder.")

    def find_corners(self):
        if hasattr(self, 'structured_images'):
            # Call find_and_draw_corners function
            self.calibration.find_and_draw_corners(self.structured_images["Q1_Image"])
        else:
            print("No images loaded. Please load a folder first.")

    def find_instrinsics(self):
        self.calibration.find_intrinsics()

    def find_extrinsic(self):
        self.calibration.find_extrinsic(self.spin_box.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())