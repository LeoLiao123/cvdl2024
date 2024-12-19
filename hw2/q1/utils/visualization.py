import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Visualizer:
    @staticmethod
    def create_probability_plot(probabilities, classes):
        # Set style
        plt.style.use('seaborn')
        
        # Create figure and axis
        figure, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars
        colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(probabilities)))
        bars = ax.bar(range(len(probabilities)), probabilities, color=colors)
        
        # Customize the plot
        ax.set_xticks(range(len(probabilities)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title('Class Probability Distribution', pad=20)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2%}',
                ha='center',
                va='bottom'
            )
        
        # Adjust layout
        plt.tight_layout()
        
        return figure

    @staticmethod
    def create_multiple_augmentations_plot(original_image, augmented_images):
        rows = 3
        cols = 4
        figure, axes = plt.subplots(rows, cols, figsize=(15, 12))
        figure.suptitle('Original Image and Augmentations', fontsize=16)
        
        # Plot original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Plot augmented images
        for idx, aug_img in enumerate(augmented_images, 1):
            if idx >= rows * cols:
                break
            ax = axes[idx // cols, idx % cols]
            ax.imshow(aug_img)
            ax.set_title(f'Augmentation {idx}')
            ax.axis('off')
        
        plt.tight_layout()
        return figure
