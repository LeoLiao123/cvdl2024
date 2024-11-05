import cv2
import numpy as np

class StereoDisparityProcessor:
    def __init__(self):
        self.imgL = None
        self.imgR = None
    
    def load_image_left(self, filepath):
        """
        Load the left image from the provided file path.
        
        Args:
            filepath (str): Path to the left image file.
        """
        self.imgL = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if self.imgL is None:
            print("Failed to load left image.")
        else:
            print("Left image loaded.")

    def load_image_right(self, filepath):
        """
        Load the right image from the provided file path.

        Args:
            filepath (str): Path to the right image file.
        """
        self.imgR = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if self.imgR is None:
            print("Failed to load right image.")
        else:
            print("Right image loaded.")

    def compute_disparity(self):
        """
        Compute the disparity map using the loaded left and right images.
        """
        if self.imgL is None or self.imgR is None:
            print("Please load both left and right images first.")
            return

        stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)
        disparity = stereo.compute(self.imgL, self.imgR)
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        # Show the disparity map
        cv2.imwrite('disparity_map.png', disparity_normalized)
        cv2.imshow('Disparity Map', disparity_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
# processor = StereoDisparityProcessor()
# processor.load_image_left('path/to/imL.png')
# processor.load_image_right('path/to/imR.png')
# processor.compute_disparity()
