from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import Any

_visualizer: Any = None
_geometry: Any = None
_camera_initialized = False


def _ensure_visualizer() -> bool:
    global _visualizer, _geometry

    if o3d is None:
        return False

    if _visualizer is not None and _geometry is not None:
        return True

    window_name = "POINT CLOUD PREVIEW"

    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(window_name=window_name, width=960, height=720, visible=True)
    if not created:
        return False

    geometry = o3d.geometry.PointCloud()
    visualizer.add_geometry(geometry)

    render_option = visualizer.get_render_option()
    if render_option is not None:
        render_option.point_size = 2.0
        render_option.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)

    _visualizer = visualizer
    _geometry = geometry
    return True


def view_3d_preview(point_cloud: np.ndarray | None) -> bool:
    """Display the 3D point cloud in an interactive window."""
    global _camera_initialized

    if not _ensure_visualizer():
        return False

    if _visualizer is None or _geometry is None:
        return False

    if point_cloud is not None:
        points = np.asarray(point_cloud, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 3 and points.size > 0:
            # Convert OpenCV camera coordinates (+y down) to Open3D display convention.
            display_points = points.copy()
            _geometry.points = o3d.utility.Vector3dVector(display_points)
            _geometry.paint_uniform_color([0.85, 0.85, 0.85])

            _visualizer.update_geometry(_geometry)

            if not _camera_initialized:
                _visualizer.reset_view_point(True)
                _camera_initialized = True

    is_open = _visualizer.poll_events()
    _visualizer.update_renderer()
    return bool(is_open)


def close_3d_preview() -> None:
    """Close the 3D preview window."""
    global _visualizer, _geometry, _camera_initialized

    if _visualizer is not None:
        _visualizer.destroy_window()

    _visualizer = None
    _geometry = None
    _camera_initialized = False