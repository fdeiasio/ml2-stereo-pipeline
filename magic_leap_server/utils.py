import cv2
import numpy as np
from typing import Any
from .config import *

def parse_mode(mode_value: str) -> str | None:
    """Parse and validate the stream mode from a string value, returning a normalized mode identifier."""
    normalized = str(mode_value).strip().lower()
    valid_modes = {"none", "raw", "rect", "disp", "view3d", "model"}
    if normalized in valid_modes:
        return normalized
    else:
        return None


def camera_pair_to_ids(map_value: str) -> tuple[str, str] | None:
    """Map a rectification map identifier to the corresponding pair of camera IDs."""
    normalized = str(map_value).strip().lower()
    if normalized == "left-center":
        return ("left", "center")
    elif normalized == "center-right":
        return ("center", "right")
    else:
        return None
    

def to_positive_int(value: Any) -> int | None:
    """Convert value to a positive integer or return None if invalid."""
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def to_gray(frame: cv2.Mat) -> cv2.Mat:
	"""Convert a BGR frame to grayscale if needed, otherwise return as-is."""
	if len(frame.shape) == 2:
		return frame
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def as_vector3(value: Any) -> np.ndarray | None:
    """Parse a finite 3D vector from an arbitrary value."""
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] != 3 or not np.all(np.isfinite(array)):
        return None

    return array


def as_quaternion(value: Any) -> np.ndarray | None:
    """Parse and normalize a finite quaternion in xyzw order."""
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] != 4 or not np.all(np.isfinite(array)):
        return None

    norm = np.linalg.norm(array)
    if norm <= 0.0:
        return None

    return array / norm


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion in xyzw order to a 3x3 rotation matrix."""
    x, y, z, w = quaternion

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized quaternion in xyzw order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0

    return quaternion

