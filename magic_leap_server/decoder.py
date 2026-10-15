import cv2
import numpy as np
from typing import Any

from .utils import to_positive_int


def decode_raw_frame(metadata: dict[str, Any], image_bytes: bytes) -> cv2.Mat | None:
    """Decode an uncompressed grayscale frame using width/height metadata."""

    # Parse width and height
    width = to_positive_int(metadata.get("width"))
    height = to_positive_int(metadata.get("height"))
    if  height is None or width is None:
        return None

    expected_size = width * height
    if len(image_bytes) != expected_size:
        return None

    # Return a BGR image that has all 3 channels set to the same grayscale values
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width))
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
