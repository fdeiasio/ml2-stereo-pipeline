import cv2
import numpy as np
from typing import Any

from .utils import to_gray
from .config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_NUM_DISPARITIES,
	DISPARITY_SPECKLE_MAX_DIFF,
	DISPARITY_SPECKLE_WINDOW,
)


# Documentation: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html


def _create_stereo_matcher() -> Any:
	"""
	Create and configure a StereoBM matcher for computing disparity maps.
	It would also be possible to use StereoSGBM (https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html) 
	insted of StereoBM. This would with more precision and less noise, but with added computational cost.
	I tried it but running in on the CPU slows the computation too much. It it were offloaded to the GPU
	hoewever i think it could be possible to implement it.
	"""
	matcher = cv2.StereoBM_create(DEFAULT_NUM_DISPARITIES, DEFAULT_BLOCK_SIZE)
	matcher.setSpeckleWindowSize(DISPARITY_SPECKLE_WINDOW)
	matcher.setSpeckleRange(DISPARITY_SPECKLE_MAX_DIFF)
	return matcher


def compute_disparity(
	left_rectified: cv2.Mat,
	right_rectified: cv2.Mat,
	stereo_matcher: Any | None = None,
) -> np.ndarray | None:
	"""Compute raw float32 disparity map from a rectified stereo pair."""
	if cv2 is None or np is None:
		return None
	if left_rectified is None or right_rectified is None:
		return None
	if left_rectified.shape[:2] != right_rectified.shape[:2]:
		return None

	left_gray = to_gray(left_rectified)
	right_gray = to_gray(right_rectified)

	matcher = stereo_matcher or _create_stereo_matcher()
	return matcher.compute(left_gray, right_gray)


def preview_from_disparity(disparity: np.ndarray | None) -> cv2.Mat | None:
	"""Convert a raw disparity map into a grayscale BGR preview image."""
	if cv2 is None or np is None:
		return None
	if disparity is None:
		return None

	disparity_normalized = cv2.normalize(
		disparity,
		None,
		alpha=0,
		beta=255,
		norm_type=cv2.NORM_MINMAX,
	)
	disparity_u8 = disparity_normalized.astype(np.uint8)
	return cv2.cvtColor(disparity_u8, cv2.COLOR_GRAY2BGR)
