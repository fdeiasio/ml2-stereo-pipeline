import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np


# Source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

'''
This script performs camera calibration using images of a checkerboard pattern.
It looks for images in the 'data/images' directory, detects the checkerboard corners,
and computes the camera matrix and distortion coefficients. The results are saved to 'data/calibration.json'.

To use this script:
1. Capture images of a checkerboard pattern using the Magic Leap cameras and save them to 'data/images'.
2. Run this script to perform calibration. It will display each image with detected corners for verification

If VISUALIZE_CORNERS is True, you can choose to keep or discard each image after seeing the detected corners.
It is useful to discard images where the checkerboard is not fully visible or corners are not detected accurately.
You can keep an image by pressing the 'y' key or delete it by pressing the 'd' key.

It is assumed all 3 cameras have the same intrinsic parameters, so there is no distinction between left/center/right 
camera images for calibration purposes.
'''

VISUALIZE_CORNERS = True


CHECKERBOARD_SIZE = (7, 6)
IMAGES_DIR = Path("data/calibration")
OUTPUT_PATH = Path("data/calibration.json")
IMAGE_GLOB = "*.png"

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

image_paths = sorted(IMAGES_DIR.glob(IMAGE_GLOB))
if not image_paths:
    raise FileNotFoundError(f"No images found with pattern '{IMAGE_GLOB}' in {IMAGES_DIR}")

image_size = None
for image_path in image_paths:
    img = cv.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image, skipping: {image_path}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    # If found, add object points and image points (after refining them)
    if not ret:
        print(f"[INFO] Checkerboard not found: {image_path.name}")
        continue
    
    print(f"[INFO] Checkerboard detected: {image_path.name}")

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    if VISUALIZE_CORNERS:
        cv.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2, ret)
        cv.imshow("calibration", img)
        #if press y key, keep the image, otherwise delete it
        key = cv.waitKey(0) & 0xFF
        if key == ord('y'):
            print(f"[INFO] Keeping image: {image_path.name}")
        elif key == ord('d'):
            print(f"[INFO] Deleting image: {image_path.name}")
            os.remove(image_path)

if VISUALIZE_CORNERS:
    cv.destroyAllWindows()

if not objpoints or image_size is None:
    raise RuntimeError("No valid calibration frames found. Check image paths and checkerboard visibility.")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
if not ret:
    raise RuntimeError("Calibration failed.")

mean_error = 0.0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

reprojection_error = mean_error / len(objpoints)

print(f"Used {len(objpoints)} valid frames out of {len(image_paths)} images")
print(f"Camera matrix:\n{mtx}")
print(f"Distortion coefficients:\n{dist}")
print(f"Re-projection error: {reprojection_error}")

output = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "reprojection_error": float(reprojection_error),
    "checkerboard_size": list(CHECKERBOARD_SIZE),
    "valid_frames": len(objpoints),
    "total_images": len(image_paths),
}

output_path = OUTPUT_PATH
with output_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"Saved calibration to: {output_path}")