import cv2
import os
import numpy as np

class AugmentedRealityProcessor:
    def __init__(self, calibration):
        self.calibration = calibration
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_folder_path = os.path.join(base_dir, 'Dataset_CvDl_Hw1', 'Q2_Image', 'Q2_db')
        self.alphabet_db_onboard = os.path.join(db_folder_path, 'alphabet_db_onboard.txt')
        self.alphabet_db_vertical = os.path.join(db_folder_path, 'alphabet_db_vertical.txt')


    def read_character_points(self, char, orientation):
        """
        Read character points from the provided alphabet database.
        """
        alphabet_db_file = self.alphabet_db_onboard if orientation == "horizontal" else self.alphabet_db_vertical
        fs = cv2.FileStorage(alphabet_db_file, cv2.FILE_STORAGE_READ)
        char_points = fs.getNode(char).mat()
        fs.release()
        return char_points

    def project_word_on_board(self, word, orientation = "horizontal"): 
        """
        Project each character of the word onto the chessboard using calibrated parameters.
        """
        if not self.calibration.ins or not self.calibration.dist:
            print("Camera not calibrated. Run calibration first.")
            return

        for char in word:
            char_points = self.read_character_points(char, orientation)
            
            for i, (rvec, tvec) in enumerate(zip(self.calibration.rvecs, self.calibration.tvecs)):
                img = cv2.imread(self.calibration.image_files[i])
                if img is None:
                    print(f"Could not read image {self.calibration.image_files[i]}")
                    continue
                
                projected_points, _ = cv2.projectPoints(char_points, rvec, tvec, self.calibration.ins, self.calibration.dist)

                # Draw the projected points on the image
                for point in projected_points:
                    point = tuple(point.ravel())
                    cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

                # Display each image for 1 second
                cv2.imshow(f'Projected {char} on Image {i+1}', img)
                cv2.waitKey(1000)
                cv2.destroyWindow(f'Projected {char} on Image {i+1}')


