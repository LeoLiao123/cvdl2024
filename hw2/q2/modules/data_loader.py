import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MNISTCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        
        original_img = self.transform(image) if self.transform else image
        augmented_img = self.augment_transform(image) if self.augment_transform else None
        
        return original_img, augmented_img if augmented_img is not None else original_img

class MNISTLoader:
    def __init__(self, data_dir="Q2_image/mnist"):
        # Basic transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Augmentation transformation pipeline
        self.augment_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomRotation(60),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Create dataset with both transforms
        self.dataset = MNISTCustomDataset(
            root_dir=data_dir, 
            transform=self.transform,
            augment_transform=self.augment_transform
        )
    
    def get_sample_images(self, num_samples):
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        dataloader = DataLoader(self.dataset, batch_size=num_samples, shuffle=True)
        batch = next(iter(dataloader))
        original_images, augmented_images = batch
        
        # Print shapes for debugging
        print(f"Original images shape: {original_images.shape}")
        print(f"Augmented images shape: {augmented_images.shape}")
        
        return original_images, augmented_images