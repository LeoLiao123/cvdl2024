import matplotlib.pyplot as plt
import torch
import numpy as np

def create_image_grid(images, title, position, grid_size=8):
    plt.subplot(1, 2, position)
    plt.title(title, pad=20, fontsize=14)
    
    for idx in range(grid_size * grid_size):
        ax = plt.subplot2grid((grid_size, 2*grid_size), 
                            (idx // grid_size, (idx % grid_size) + (grid_size if position == 2 else 0)))
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        img = images[idx].cpu().detach()
        if img.shape[0] == 1:
            img = img.squeeze()
        
        img = (img + 1) / 2.0
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)

def create_comparison_window(title1, images1, title2, images2):
    """Display two sets of images side by side in a single window at 90% of original size"""
    # Original size was 20,10, new size is 90% of that
    fig = plt.figure(figsize=(18, 9))  # 90% of (20,10)
    
    batch_size = images1.size(0)
    grid_size = int(np.sqrt(batch_size))
    
    create_image_grid(images1, title1, 1, grid_size)
    create_image_grid(images2, title2, 2, grid_size)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.1)
    
    plt.show()

def show_image_grid(title, images):
    """Display a grid of images at 90% of original size"""
    batch_size = images.size(0)
    grid_size = int(np.sqrt(batch_size))
    
    # Original size was 7,7, new size is 90% of that
    plt.figure(figsize=(6.3, 6.3))  # 90% of (7,7)
    plt.suptitle(title, fontsize=16, y=0.95)
    
    for idx in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, idx + 1)
        plt.axis('off')
        
        img = images[idx].cpu().detach()
        if img.shape[0] == 1:
            img = img.squeeze()
        
        img = (img + 1) / 2.0
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    
    plt.tight_layout()
    plt.show()
