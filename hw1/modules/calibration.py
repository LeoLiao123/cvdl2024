import cv2
import os
import numpy as np

class Calibration:
    def __init__(self, width, height):
        self.chessboard_size = (width, height)  # Width and height of the chessboard squares
        self.image_points = []  # List of chessboard corners in each image
        self.obj_points = [] # List
        self.rvecs = None
        self.tvecs = None 

    def find_and_draw_corners(self, images):
        """
        Finds and draws chessboard corners on each image.
        
        Args:
            structured_images (dict): Dictionary where keys are folder names and values are lists of image paths.
        """
        # Define parameters for corner refinement
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objp *= 0.02  # Scale points by square size

        image_points = []
        obj_points = []
        for img_path in images:
            image = cv2.imread(img_path)
            grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if image is None:
                print(f"Could not read image {img_path}")
                continue
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(grayimg, self.chessboard_size, None)
            if ret:
                image_points.append(corners)
                obj_points.append(objp)
                # Refine corner positions for better accuracy
                corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
                
                # Draw corners on the image
                cv2.drawChessboardCorners(image, self.chessboard_size, corners, ret)
                
                cv2.namedWindow(f"Corners in {os.path.basename(img_path)}", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Corners in {os.path.basename(img_path)}", image)
                cv2.waitKey(5000)  

            else:
                print(f"Chessboard corners not found in {img_path}")

        cv2.destroyAllWindows()
        self.image_points = image_points
        self.obj_points = obj_points
    
    def find_intrinsics(self):
        """
        Finds camera intrinsics using the chessboard corners.
        """
        # Define image size (width, height)
        imageSize = (2048, 2048)

        # Perform camera calibration
        ret, ins, dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.image_points, imageSize, None, None)

        if ret:
            print("Intrinsic Matrix:\n", ins)

        else:
            print("Calibration failed")

    def find_extrinsic(self, image_num):
        """
        Finds the extrinsic matrix for the specified image.
        """
        rotation_matrix = cv2.Rodrigues(self.rvecs[image_num])[0]
        extrinsic_matrix = np.hstack(rotation_matrix , self.tvecs[image_num])
        print("Extrinsic Matrix:\n", extrinsic_matrix)
                