import torchvision.transforms as transforms
from PIL import Image

class ImageTransforms:
    @staticmethod
    def get_inference_transforms():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    @staticmethod
    def get_augmentation_transforms():
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomResizedCrop(
                32,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            )
        ])

    @staticmethod
    def tensor_to_pil(tensor):
        """Convert a tensor to PIL Image"""
        return transforms.ToPILImage()(tensor)

    @staticmethod
    def pil_to_tensor(image):
        """Convert a PIL Image to tensor"""
        return transforms.ToTensor()(image)