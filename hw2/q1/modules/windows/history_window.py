from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

class HistoryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training History")
        self.setMinimumSize(1000, 500)
        self.init_ui()

    def init_ui(self):
        # Create main layout
        main_layout = QVBoxLayout(self)

        # Check if history plot exists
        # Expand the ~ in the path to full home directory path
        history_path = os.path.expanduser('~/code/ncku/ncku-cvdl-hw/hw2/q1/modules/model/training_history.png')
        
        # Print path for debugging
        print(f"Looking for history plot at: {history_path}")

        if not os.path.exists(history_path):
            label = QLabel("Training history plot not found!")
            label.setStyleSheet("""
                QLabel {
                    color: red;
                    font-size: 16px;
                    padding: 20px;
                }
            """)
            label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(label)
        else:
            # Load and display the training history plot
            pixmap = QPixmap(history_path)
            plot_label = QLabel()
            plot_label.setPixmap(pixmap)
            plot_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(plot_label)
