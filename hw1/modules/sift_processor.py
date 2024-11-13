import cv2

class SIFTProcessor:
    def __init__(self):
        self.image1 = None
        self.image2 = None

    def load_image1(self, filepath):
        """
        Load the first image from the provided file path.

        Args:
            filepath (str): Path to the image file.
        """
        self.image1 = cv2.imread(filepath)
        if self.image1 is None:
            print("Failed to load Image 1.")
        else:
            print(f"Loaded Image 1: {filepath}")

    def load_image2(self, filepath):
        """
        Load the second image from the provided file path.

        Args:
            filepath (str): Path to the image file.
        """
        self.image2 = cv2.imread(filepath)
        if self.image2 is None:
            print("Failed to load Image 2.")
        else:
            print(f"Loaded Image 2: {filepath}")

    def show_keypoints(self):
        """
        Show the keypoints detected in the first image.
        
        Note: This method uses the SIFT algorithm to detect keypoints.
        """
        if self.image1 is None:
            print("Image 1 not loaded.")
            return
        
        gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
        
        cv2.imshow("Keypoints", img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_matched_keypoints(self):
        """
        Show the matched keypoints between the two loaded images.

        Note: This method uses the SIFT algorithm to detect and match keypoints.
        """
        if self.image1 is None or self.image2 is None:
            print("Both images need to be loaded.")
            return

        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        matched_img = cv2.drawMatchesKnn(gray1, keypoints1, gray2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matched_img = cv2.resize(matched_img, (800, 600))
        cv2.imshow("Matched Keypoints", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
# processor = SIFTProcessor()
# processor.load_image1('path/to/image1.png')
# processor.load_image2('path/to/image2.png')
# processor.show_keypoints()
# processor.show_matched_keypoints()
