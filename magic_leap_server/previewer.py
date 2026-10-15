import os
import cv2
import numpy as np  
from datetime import datetime

from .config import SAVED_IMAGES_DIR


def _make_placeholder(width: int = 1016, height: int = 1016) -> cv2.Mat | None:
    """Create placeholder image for missing frames."""
    if cv2 is None or np is None:
        return None
    
    array = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.putText(array, "No Frame", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def _save_images(
    frames: dict[str, cv2.Mat]
) -> None:
    """Save the camera images to data/images directory.""" 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(SAVED_IMAGES_DIR):
        os.makedirs(SAVED_IMAGES_DIR)

    for camera_id, image in frames.items():
        if image is None:
            image = _make_placeholder()

        filename = os.path.join(SAVED_IMAGES_DIR, f"{camera_id}_{timestamp}.png")
        cv2.imwrite(filename, image)
        print(f"Saved image: {filename}")


def show_preview_window(
    frames: dict[str, cv2.Mat]
) -> bool:
    """Display the given frames in a preview window. Returns False if the window should be closed."""
    if cv2 is None or np is None:
        return False

    images = []
    for camera_id, image in frames.items():
        if image is None:
            image = _make_placeholder()

        cv2.putText(image, camera_id.upper(), (10, 24), int(cv2.FONT_HERSHEY_SIMPLEX), 0.75, (0, 255, 255), 2)
        images.append(image)

    if not images:
        return True

    window_name = " ".join([cam_id.upper() for cam_id in frames.keys()])
    composite = cv2.hconcat(images)
    cv2.imshow(window_name, composite)

    # Map the keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    if key == ord("s"):
        _save_images(frames)
    
    return True


def close_preview_window() -> None:
    if cv2 is not None:
        cv2.destroyAllWindows()