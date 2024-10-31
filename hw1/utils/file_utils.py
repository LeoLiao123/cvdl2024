import os

def load_images_from_folder(folder_path):
    """
    Recursively load all image file paths from a specified folder and organize them by subfolder.
    
    Args:
        folder_path (str): The path to the folder containing images.
        
    Returns:
        dict: A dictionary where keys are subfolder names and values are lists of image paths.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    structured_images = {}

    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        images_in_folder = []

        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                images_in_folder.append(os.path.join(root, file))

        if images_in_folder:
            structured_images[folder_name] = images_in_folder

    return structured_images
