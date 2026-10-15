import cv2
import numpy as np
from pathlib import Path
from typing import Any

from .utils import camera_pair_to_ids
from .config import (
    CENTER_RIGHT_MAPS_PATH, 
	LEFT_CENTER_MAPS_PATH
)


def _map_to_path(map_value: str) -> Path | None:
	"""Map a rectification map identifier to the corresponding .npz file path."""
	normalized = str(map_value).strip().lower()
	if normalized == "left-center":
		return Path(LEFT_CENTER_MAPS_PATH)
	elif normalized == "center-right":
		return Path(CENTER_RIGHT_MAPS_PATH)
	else:
		return None


def _load_rectification_file(npz_path: Path) -> tuple[Any, Any, Any, Any] | None:
	"""Load the rectification maps from a .npz file and return them as a tuple."""
	try:
		with np.load(npz_path) as data:
			return (
				data["left_map_x"],
				data["left_map_y"],
				data["right_map_x"],
				data["right_map_y"],
			)
	except Exception as exc:
		print(f"[WARN] Failed to load rectification maps from {npz_path}: {exc}")
		return None


def load_rectification_maps(map_value: str) -> dict[str, tuple[Any, Any]] | None:
	"""Load the rectification maps for the specified map identifier and return them in a dictionary keyed by camera ID."""
	npz_path = _map_to_path(map_value)
	if npz_path is None or not npz_path.exists():
		print(f"[WARN] Unsupported rectification map specified: {map_value}")
		return None

	data = _load_rectification_file(npz_path)
	if data is None:
		print(f"[WARN] Failed to load rectification maps for {map_value} from {npz_path}")
		return None

	map_a_x, map_a_y, map_b_x, map_b_y = data

	camera_ids = camera_pair_to_ids(map_value)
	if camera_ids is None:
		print(f"[WARN] Failed to convert camera pair to IDs for {map_value}")
		return None

	maps_by_camera_id = {
		camera_ids[0]: (map_a_x, map_a_y),
		camera_ids[1]: (map_b_x, map_b_y),
	}

	return maps_by_camera_id


def rectify_frame(frame: cv2.Mat, map: tuple[Any, Any],):
	"""Apply rectification to a single frame using the provided rectification maps."""
	map_x, map_y = map
	return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
