import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np

"""
This script performs stereo camera calibration using pairs of images of a checkerboard pattern.
It looks for images in the 'data/stereo_calibration' directory, detects the checkerboard
corners, and computes the stereo camera parameters. The results are saved to 'data/stereo_calibration_<pair_name>.json' 
and rectification maps are saved to 'data/stereo_rectification_maps_<pair_name>.npz'.

To use this script:
1. Capture pairs of images of a checkerboard pattern using the Magic Leap cameras and save them to 'data/stereo_calibration'.
   The images should be named with prefixes to indicate which camera they belong to, e.g. 'left_001.png' and 'center_001.png' for a left-center pair.
2. Run this script to perform stereo calibration.

If VISUALIZE_CORNERS is True, you can choose to keep or discard each pairafter seeing the detected corners. 
It is useful to discard pairs where the checkerboard is not fully visible or corners are not detected accurately.
You can keep a pair by pressing the 'y' key or delete it by pressing the 'd' key.
"""

VISUALIZE_CORNERS = True


# Edit here to choose which pair to calibrate or to calibrate both pairs
LEFT_CENTER_PREFIXES = ("left_", "center_")
CENTER_RIGHT_PREFIXES = ("center_", "right_")
LOADED_PREFIXES = LEFT_CENTER_PREFIXES 


CHECKERBOARD_SIZE = (7, 6)
SQUARE_SIZE = 25.0 # mm


PAIR_NAME = f"{LOADED_PREFIXES[0]}{LOADED_PREFIXES[1]}pair"


INTRINSIC_CALIBRATION_PATH = Path("data/calibration.json")
IMAGES_DIR = Path("data/stereo_calibration")
OUTPUT_DIR = Path("data/stereo_calibration" + PAIR_NAME + ".json")
IMAGE_GLOB = "*.png"


SUBPIX_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)


def _load_intrinsic(path: Path):
    """Load mono camera intrinsics from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find intrinsics file: {path}. Run mono calibration first to generate this file.")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64)
    reprojection_error = float(data.get("reprojection_error", -1.0))

    return camera_matrix, dist_coeffs, reprojection_error


def collect_stereo_pairs(images_dir: Path, left_prefix: str, right_prefix: str):
    if not images_dir.exists():
        raise FileNotFoundError(f"Stereo image directory does not exist: {images_dir}")

    images = sorted(IMAGES_DIR.glob(IMAGE_GLOB))

    left_map = {}
    right_map = {}

    for image_path in images:
        stem = image_path.stem
        stem_lower = stem.lower()

        if stem_lower.startswith(left_prefix):
            key = stem[len(left_prefix):]
            left_map[key] = image_path
        elif stem_lower.startswith(right_prefix):
            key = stem[len(right_prefix):]
            right_map[key] = image_path

    shared_keys = sorted(set(left_map).intersection(right_map))

    if not shared_keys and left_map and right_map and len(left_map) == len(right_map):
        print("[WARN] No matching suffix keys. Falling back to sorted left-right pairing by index.")
        left_paths = [left_map[k] for k in sorted(left_map)]
        right_paths = [right_map[k] for k in sorted(right_map)]
        return [(left_paths[i], right_paths[i], f"index_{i:04d}") for i in range(len(left_paths))]

    if not shared_keys:
        raise RuntimeError(
            f"No stereo pairs found in {images_dir}. "
            f"Expected files named like '{left_prefix}<id>.png' and '{right_prefix}<id>.png'."
        )

    missing_left = sorted(set(right_map).difference(left_map))
    missing_right = sorted(set(left_map).difference(right_map))
    if missing_left:
        print(f"[WARN] Missing left image for {len(missing_left)} right images")
    if missing_right:
        print(f"[WARN] Missing right image for {len(missing_right)} left images")

    return [(left_map[k], right_map[k], k) for k in shared_keys]


def create_object_points(checkerboard_size, square_size: float):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate_pair(
    pair_name: str,
    prefix_a: str,
    prefix_b: str,
    image_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    mono_error: float,
):
    print(f"\n[INFO] Calibrating pair: {pair_name} ({prefix_a} <-> {prefix_b})")

    pairs = collect_stereo_pairs(image_dir, prefix_a, prefix_b)

    objp = create_object_points(CHECKERBOARD_SIZE, SQUARE_SIZE)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    image_size = None

    for left_path, right_path, pair_id in pairs:
        left_img = cv.imread(str(left_path))
        right_img = cv.imread(str(right_path))
        if left_img is None or right_img is None:
            print(f"[WARN] Could not read pair {pair_id}, skipping")
            continue

        left_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
        right_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

        if left_gray.shape != right_gray.shape:
            print(f"[WARN] Shape mismatch for pair {pair_id}, skipping")
            continue

        image_size = left_gray.shape[::-1]

        left_found, left_corners = cv.findChessboardCorners(left_gray, CHECKERBOARD_SIZE, None)
        right_found, right_corners = cv.findChessboardCorners(right_gray, CHECKERBOARD_SIZE, None)

        if not (left_found and right_found):
            print(f"[INFO] Checkerboard not found in both images for pair {pair_id}, skipping")
            continue

        left_corners_refined = cv.cornerSubPix(
            left_gray,
            left_corners,
            (11, 11),
            (-1, -1),
            SUBPIX_CRITERIA,
        )
        right_corners_refined = cv.cornerSubPix(
            right_gray,
            right_corners,
            (11, 11),
            (-1, -1),
            SUBPIX_CRITERIA,
        )

        objpoints.append(objp.copy())
        imgpoints_left.append(left_corners_refined)
        imgpoints_right.append(right_corners_refined)

        print(f"[INFO] Using pair: {pair_id}")

        if VISUALIZE_CORNERS:
            left_vis = left_img.copy()
            right_vis = right_img.copy()
            cv.drawChessboardCorners(left_vis, CHECKERBOARD_SIZE, left_corners_refined, left_found)
            cv.drawChessboardCorners(right_vis, CHECKERBOARD_SIZE, right_corners_refined, right_found)
            both = np.hstack((left_vis, right_vis))
            cv.imshow("stereo calibration", both)

            key = cv.waitKey(0) & 0xFF
            if key == ord("y"):
                print(f"[INFO] Keeping pair: {pair_id}")
            elif key == ord("d"):
                print(f"[INFO] Deleting pair: {pair_id}")
                os.remove(left_path)
                os.remove(right_path)


    if VISUALIZE_CORNERS:
        cv.destroyAllWindows()

    if not objpoints or image_size is None:
        raise RuntimeError("No valid stereo frames found. Check checkerboard visibility and image pairs.")

    stereo_flags = cv.CALIB_FIX_INTRINSIC

    rms_error, cm1, dc1, cm2, dc2, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        camera_matrix.copy(),
        dist_coeffs.copy(),
        camera_matrix.copy(),
        dist_coeffs.copy(),
        image_size,
        criteria=STEREO_CRITERIA,
        flags=stereo_flags,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        cm1,
        dc1,
        cm2,
        dc2,
        image_size,
        R,
        T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=-1,
    )

    left_map_x, left_map_y = cv.initUndistortRectifyMap(
        cm1,
        dc1,
        R1,
        P1,
        image_size,
        cv.CV_32FC1,
    )
    right_map_x, right_map_y = cv.initUndistortRectifyMap(
        cm2,
        dc2,
        R2,
        P2,
        image_size,
        cv.CV_32FC1,
    )

    output_maps_path = Path(f"data/stereo_rectification_maps_{pair_name}.npz")
    output_json_path = Path(f"data/stereo_calibration_{pair_name}.json")

    output = {
        "pair_name": pair_name,
        "camera_a_prefix": prefix_a,
        "camera_b_prefix": prefix_b,
        "image_size": list(image_size),
        "checkerboard_size": list(CHECKERBOARD_SIZE),
        "square_size": float(SQUARE_SIZE),
        "mono_reprojection_error": float(mono_error),
        "stereo_rms_error": float(rms_error),
        "valid_pairs": len(objpoints),
        "total_pairs": len(pairs),
        "baseline": float(np.linalg.norm(T)),
        "left_camera_matrix": cm1.tolist(),
        "left_distortion_coefficients": dc1.tolist(),
        "right_camera_matrix": cm2.tolist(),
        "right_distortion_coefficients": dc2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "roi1": [int(v) for v in roi1],
        "roi2": [int(v) for v in roi2],
        "rectification_maps": str(output_maps_path),
    }

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    np.savez_compressed(
        output_maps_path,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
    )

    print(f"Used {len(objpoints)} valid stereo pairs out of {len(pairs)} matched pairs")
    print(f"Stereo RMS error: {rms_error}")
    print(f"Baseline: {np.linalg.norm(T)}")
    print(f"Saved stereo calibration to: {output_json_path}")
    print(f"Saved rectification maps to: {output_maps_path}")


def main():
    camera_matrix, dist_coeffs, mono_error = _load_intrinsic(INTRINSIC_CALIBRATION_PATH)

    failures = []
    prefix_a, prefix_b = LOADED_PREFIXES
    try:
        calibrate_pair(
            pair_name=PAIR_NAME,
            prefix_a=prefix_a,
            prefix_b=prefix_b,
            image_dir=IMAGES_DIR,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            mono_error=mono_error,
        )
    except Exception as exc:
        failures.append((PAIR_NAME, str(exc)))

    if failures:
        print("\n[WARN] Some pair calibrations failed:")
        for pair_name, reason in failures:
            print(f"  - {pair_name}: {reason}")
        raise RuntimeError("One or more pair calibrations failed.")


if __name__ == "__main__":
    main()