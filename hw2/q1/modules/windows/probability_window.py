from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

class ProbabilityWindow(QWidget):
    def __init__(self, probabilities, classes):
        super().__init__()
        self.setWindowTitle("Prediction Probabilities")
        self.probabilities = probabilities
        self.classes = classes
        # Set white background with black text
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                color: black;
            }
            QLabel {
                color: black;
            }
            QTableWidget {
                background-color: white;
                color: black;
                gridline-color: #d4d4d4;
                alternate-background-color: #f7f7f7;
            }
            QHeaderView::section {
                background-color: #e1e1e1;
                color: black;
                padding: 4px;
                border: 1px solid #d4d4d4;
                font-weight: bold;
            }
            QTableWidget::item {
                color: black;
            }
        """)
        self.init_ui()
        
    def init_ui(self):
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create the plot
        fig = self.create_plot()
        canvas = FigureCanvas(fig)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar(canvas, self)
        
        # Create table showing exact values
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Class', 'Probability'])
        table.setRowCount(len(self.classes))
        
        # Fill the table with data
        for i, (class_name, prob) in enumerate(zip(self.classes, self.probabilities)):
            class_item = QTableWidgetItem(class_name)
            prob_item = QTableWidgetItem(f"{prob:.4%}")
            # Set text color to black
            class_item.setForeground(QBrush(QColor('black')))
            prob_item.setForeground(QBrush(QColor('black')))
            table.setItem(i, 0, class_item)
            table.setItem(i, 1, prob_item)
        
        # Adjust table properties
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add info label with explicit color
        info_label = QLabel(f"Predicted class: {self.classes[np.argmax(self.probabilities)]}")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: black;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        info_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        main_layout.addWidget(toolbar)
        main_layout.addWidget(canvas)
        main_layout.addWidget(info_label)
        main_layout.addWidget(table)
        
        # Set window properties
        self.resize(800, 800)

    def create_plot(self):
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set background color to white
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Create bars with gradient colors
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(self.probabilities)))
        bars = ax.bar(range(len(self.probabilities)), 
                     self.probabilities, 
                     color=colors,
                     edgecolor='black',
                     linewidth=1)
        
        # Customize the plot with explicit colors
        ax.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.set_axisbelow(True)
        
        # Set labels and title with explicit colors
        ax.set_xticks(range(len(self.probabilities)))
        ax.set_xticklabels(self.classes, rotation=45, ha='right', color='black')
        ax.set_ylabel('Probability', fontsize=12, color='black')
        ax.set_title('Class Probability Distribution', pad=20, fontsize=14, color='black')
        
        # Set tick colors to black
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')
        
        # Add value labels with black text
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                   height,
                   f'{height:.2%}',
                   ha='center',
                   va='bottom',
                   color='black')
        
        plt.tight_layout()
        return fig
