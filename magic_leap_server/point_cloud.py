import cv2
import numpy as np
import json
from pathlib import Path

from .config import (
    LEFT_CENTER_CALIBRATION_PATH,
    CENTER_RIGHT_CALIBRATION_PATH,
    MAX_POINT_COUNT,
    MIN_VALID_DISPARITY,
    DISPARITY_LOW_PERCENTILE,
    DISPARITY_HIGH_PERCENTILE,
    DEPTH_LOW_PERCENTILE,
    DEPTH_HIGH_PERCENTILE,
)


def _map_to_calibration_path(map_value: str) -> Path | None:
    """Map a rectification map identifier to the corresponding calibration file path."""
    normalized = str(map_value).strip().lower()
    if normalized == "left-center":
        return Path(LEFT_CENTER_CALIBRATION_PATH)
    if normalized == "center-right":
        return Path(CENTER_RIGHT_CALIBRATION_PATH)
    return None


def _load_calibration_file(path: Path) -> dict | None:
    """Load and parse a stereo calibration file in JSON format."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[WARN] Failed to read calibration file {path}: {exc}")
        return None


def load_q_matrix(map_value: str) -> np.ndarray | None:
    """Load the Q matrix for 3D reprojection based on the specified rectification map."""
    calibration_path = _map_to_calibration_path(map_value)
    if calibration_path is None or not calibration_path.exists():
        print(f"[WARN] Calibration file not found for: {map_value}")
        return None

    calibration_data = _load_calibration_file(calibration_path)
    if calibration_data is None:
        return None

    q_values = calibration_data.get("Q")
    if q_values is None:
        print(f"[WARN] Missing Q matrix in calibration file: {calibration_path}")
        return None

    q_matrix = np.asarray(q_values, dtype=np.float32)
    if q_matrix.shape != (4, 4):
        print(f"[WARN] Invalid Q matrix shape {q_matrix.shape} in {calibration_path}")
        return None

    return q_matrix


def create_point_cloud_from_disparity(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
) -> np.ndarray | None:
    """
    Create a 3D point cloud from a disparity map and a Q matrix.
    This function is the heaviest part of the pipeline. It's surely not the most efficent wat of implementing it,
    so I would avise anyone trying to optimze the application to begin here. The most computational intensive
    part is reprojectImageTo3D(), which i think can be offloaded to the GPU, but i wouldn't swear on it.
    """
    
    disparity_f32 = disparity.astype(np.float32, copy=False)
    disparity_mask = np.isfinite(disparity_f32)
    np.greater(
        disparity_f32,
        MIN_VALID_DISPARITY,
        out=disparity_mask,
        where=disparity_mask,
    )
    if not np.any(disparity_mask):
        return None

    finite_disparity = disparity_f32[disparity_mask]
    disp_low, disp_high = np.percentile(
        finite_disparity,
        [DISPARITY_LOW_PERCENTILE, DISPARITY_HIGH_PERCENTILE],
        overwrite_input=True,
    )

    try:
        points_3d = cv2.reprojectImageTo3D(disparity_f32, q_matrix)
    except Exception as exc:
        print(f"[WARN] Failed to reproject disparity to 3D: {exc}")
        return None

    valid_disparity_mask = disparity_mask
    np.greater_equal(
        disparity_f32,
        disp_low,
        out=valid_disparity_mask,
        where=valid_disparity_mask,
    )
    np.less_equal(
        disparity_f32,
        disp_high,
        out=valid_disparity_mask,
        where=valid_disparity_mask,
    )
    points = points_3d[valid_disparity_mask]

    if points.size == 0:
        return None

    z_values = points[:, 2]
    finite_depth_mask = np.isfinite(z_values)
    if not np.any(finite_depth_mask):
        return None

    finite_z_values = z_values[finite_depth_mask]
    z_low, z_high = np.percentile(
        finite_z_values,
        [DEPTH_LOW_PERCENTILE, DEPTH_HIGH_PERCENTILE],
        overwrite_input=True,
    )
    if z_high > z_low:
        np.greater_equal(
            z_values,
            z_low,
            out=finite_depth_mask,
            where=finite_depth_mask,
        )
        np.less_equal(
            z_values,
            z_high,
            out=finite_depth_mask,
            where=finite_depth_mask,
        )

    points = points[finite_depth_mask]

    if points.size == 0:
        return None

    if points.shape[0] > MAX_POINT_COUNT:
        step = (points.shape[0] // MAX_POINT_COUNT) + 1
        points = points[::step]

    return points.astype(np.float32, copy=False)