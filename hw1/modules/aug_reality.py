import cv2
import os
import numpy as np
from modules.calibration import Calibration

class AugmentedRealityProcessor:
    def __init__(self):
        self.calibration = Calibration(height=11, width=8)
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
        
        if char_points is not None and char_points.shape[1] == 2:
            char_points = char_points.reshape(-1, 3)  # Flatten into (N*2, 3)

        print(f"Character {char} points: {char_points}")
        return char_points

    def project_word_on_board(self, images, word, orientation="horizontal"):
        """
        Project each character of the word onto the chessboard using calibrated parameters.
        """
        select_images = sorted(images, key=lambda x: int(os.path.basename(x).split('.')[0]))[:5]
        self.calibration.find_and_draw_corners(select_images, display=False)
        self.calibration.find_intrinsics()

        if not self.calibration.ins.any() or not self.calibration.dist.any():
            print("Camera not calibrated. Run calibration first.")
            return

        for char in word:
            char_points = self.read_character_points(char, orientation)
            if char_points is None or char_points.size == 0:
                print(f"No valid points found for character: {char}")
                continue
            
            char_points = np.array(char_points, dtype=np.float32).reshape(-1, 3)
            char_points = char_points[:, np.newaxis, :]
            print(f"Character {char} points: {char_points}")
            print(f"ins: {self.calibration.ins}, dist: {self.calibration.dist}")
            
            for i, (rvec, tvec) in enumerate(zip(self.calibration.rvecs, self.calibration.tvecs)):
                img = cv2.imread(select_images[i])
                if img is None:
                    print(f"Could not read image {select_images[i]}")
                    continue
                print(f"rvec: {rvec}, tvec: {tvec}")

                projected_points, _ = cv2.projectPoints(char_points, rvec, tvec, self.calibration.ins, self.calibration.dist)
                print(f"Projected points for character {char}: {projected_points}")

                for j in range(0, len(projected_points) - 1, 2):  # Step by 2 for each line segment
                    pointA = tuple(map(int, projected_points[j].ravel()))
                    pointB = tuple(map(int, projected_points[j + 1].ravel()))

                    # Ensure points are within image bounds
                    if (0 <= pointA[0] < img.shape[1] and 0 <= pointA[1] < img.shape[0] and
                        0 <= pointB[0] < img.shape[1] and 0 <= pointB[1] < img.shape[0]):
                        print(f"Drawing line from {pointA} to {pointB}")
                        cv2.line(img, pointA, pointB, (0, 255, 0), 2)
                    else:
                        print(f"Skipping line from {pointA} to {pointB}, points out of image bounds.")

                resized_img = cv2.resize(img, (800, 600))
                cv2.imshow(f'Projected {char} on Image {i+1}', resized_img)
                cv2.waitKey(1000)
                cv2.destroyWindow(f'Projected {char} on Image {i+1}')

# Assuming this function gets called from somewhere with the appropriate images and word
# Example usage:
# processor = AugmentedRealityProcessor()
# images = ['1.bmp', '2.bmp', '3.bmp', '4.bmp', '5.bmp']  # Replace with actual paths
# processor.project_word_on_board(images, "CAMERA")
