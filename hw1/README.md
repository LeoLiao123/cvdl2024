# NCKU CVDL2024 Hw1

## Overview

This is the assignment 1 of National Cheng Kung University Computer Vision and Deep Learning course, implementing Q1 Calibration, Q2 Augmented Reality, Q3 Stereo Disparity Map, and Q4 SIFT Feature Extraction. Each task corresponds to the use of the datasets in `Dataset_CvDl_Hw1`, such as `Q1_Image`, `Q2_Image`, etc.

## Features

1. **Load Images**

   - Load a folder containing structured images for processing.
   - Load individual left and right images for stereo disparity computation.

2. **Calibration**

   - Find chessboard corners in the images.
   - Find intrinsic and extrinsic parameters of the camera.
   - Find lens distortion parameters.
   - Show the undistorted version of the images.

3. **Augmented Reality**

   - Project words onto a chessboard in either horizontal or vertical orientation.

4. **Stereo Disparity Map**

   - Compute the disparity map using stereo images.

5. **SIFT Feature Extraction**

   - Load two images for SIFT processing.
   - Display keypoints of an image.
   - Display matched keypoints between two images.

## Requirements

- Python 3.6+
- PySide6
- OpenCV
- NumPy
- Custom modules: `file_utils`, `calibration`, `aug_reality`, `stereo_disparity`, `sift_processor`

## Installation

1. Clone this repository:

   ```sh
   git clone git@github.com:LeoLiao123/ncku-cvdl-hw.git
   ```

2. Install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Make sure the custom modules (`file_utils`, `calibration`, `aug_reality`, `stereo_disparity`, `sift_processor`) are correctly placed within the project directory.

## Usage

1. Run the application:

   ```sh
   python main.py
   ```

2. Use the graphical interface to load images and perform the various operations described above.

## User Interface Guide

- **Load Image Group**: Load a folder or individual images for processing.
- **Calibration Group**: Perform camera calibration tasks.
- **Augmented Reality Group**: Project words onto a chessboard.
- **Stereo Group**: Compute a stereo disparity map.
- **SIFT Group**: Load images and visualize SIFT keypoints and matches.

## Workflow

1. **Load Folder**: Click "Load folder" to load a directory containing structured images.
2. **Calibration**: Find chessboard corners and calculate intrinsic and extrinsic parameters.
3. **Augmented Reality**: Enter a word (max 6 characters) and project it onto the chessboard.
4. **Stereo Disparity**: Load left and right images, and compute the disparity map.
5. **SIFT**: Load two images, view keypoints, and match them.

## File Structure

- `main.py`: Entry point for the application.
- `modules/`: Contains the custom modules used for various functionalities.
  - `calibration.py`: Handles camera calibration.
  - `aug_reality.py`: Projects words onto a chessboard for AR tasks.
  - `stereo_disparity.py`: Computes stereo disparity maps.
  - `sift_processor.py`: Handles SIFT feature extraction and matching.
- `utils/`: Contains utility files, such as `file_utils.py` for loading images.