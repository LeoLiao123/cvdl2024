import cv2
import os
import numpy as np

class Calibration:
    def __init__(self, width, height):
        self.chessboard_size = (width, height)  # Width and height of the chessboard squares
        self.image_points = []  # List of chessboard corners in each image
        self.obj_points = []  # List of 3D points in real world space
        self.ins = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None 

    def find_and_draw_corners(self, images):
        """
        Finds and draws chessboard corners on each image.
        
        Args:
            images (list): List of image paths.
        """
        # Reset points
        self.image_points = []
        self.obj_points = []

        # Define parameters for corner refinement
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= 0.02  # Scale points by square size

        for img_path in images:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image {img_path}")
                continue

            grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(grayimg, self.chessboard_size, None)
            if ret:
                # Refine corner positions for better accuracy
                corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
                self.image_points.append(corners)
                self.obj_points.append(objp)

                # Draw corners on the image
                cv2.drawChessboardCorners(image, self.chessboard_size, corners, ret)
                cv2.imshow(f"Corners in {os.path.basename(img_path)}", image)
                cv2.waitKey(5000)  

            else:
                print(f"Chessboard corners not found in {img_path}")

        cv2.destroyAllWindows()
    
    def find_intrinsics(self):
        """
        Finds camera intrinsics using the chessboard corners.
        """
        if not self.image_points or not self.obj_points:
            print("No chessboard corners detected. Run find_and_draw_corners() first.")
            return

        # Use the first image to determine size
        h, w = self.image_points[0][0].shape[0:2]
        imageSize = (w, h)

        # Perform camera calibration
        ret, self.ins, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.obj_points, self.image_points, imageSize, None, None
        )

        if ret:
            print("Intrinsic Matrix:\n", self.ins)
        else:
            print("Calibration failed")

    def find_extrinsic(self, image_num):
        """
        Finds the extrinsic matrix for the specified image.
        
        Args:
            image_num (int): Index of the image to use for extrinsic calculation.
        """
        if self.rvecs is None or self.tvecs is None:
            print("Intrinsics not calculated. Run find_intrinsics() first.")
            return
        
        rotation_matrix = cv2.Rodrigues(self.rvecs[image_num])[0]
        extrinsic_matrix = np.hstack((rotation_matrix , self.tvecs[image_num]))
        print("Extrinsic Matrix:\n", extrinsic_matrix)

    def find_distortion(self):
        """
        Finds distortion coefficients.
        """
        if self.dist is None:
            print("Distortion not calculated. Run find_intrinsics() first.")
            return

        print("Distortion Coefficients:\n", self.dist)

    def show_undistorted_result(self, images):
        """
        Shows the undistorted images.
        
        Args:
            images (list): List of image paths.
        """
        if self.ins is None or self.dist is None:
            print("Calibration data not found. Run find_intrinsics() first.")
            return

        for image_path in images:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            undistorted_image = cv2.undistort(gray_image, self.ins, self.dist)
            
            cv2.imshow("Distorted image", gray_image)
            cv2.imshow("Undistorted image", undistorted_image)
            cv2.waitKey(5000)

        cv2.destroyAllWindows()
