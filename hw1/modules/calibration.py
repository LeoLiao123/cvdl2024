import cv2
import os

class Calibration:
    def __init__(self, width, height):
        self.chessboard_size = (width, height)  # Width and height of the chessboard squares

    def find_and_draw_corners(self, structured_images):
        """
        Finds and draws chessboard corners on each image.
        
        Args:
            structured_images (dict): Dictionary where keys are folder names and values are lists of image paths.
        """
        # Define parameters for corner refinement
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

        for subfolder, images in structured_images.items():
            if subfolder != "Q1_Image":
                continue
            print(f"Processing folder: {subfolder}")
            for img_path in images:
                # Read the image in grayscale
                image = cv2.imread(img_path)
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                if grayimg is None:
                    print(f"Could not read image {img_path}")
                    continue
                
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(grayimg, self.chessboard_size, None)
                if ret:
                    # Refine corner positions for better accuracy
                    corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
                    
                    # Draw corners on the image
                    cv2.drawChessboardCorners(image, self.chessboard_size, corners, ret)
                    
                    cv2.namedWindow(f"Corners in {os.path.basename(img_path)}", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"Corners in {os.path.basename(img_path)}", grayimg)
                    cv2.waitKey(5000)  

                else:
                    print(f"Chessboard corners not found in {img_path}")

        # Close all OpenCV windows
        cv2.destroyAllWindows()
