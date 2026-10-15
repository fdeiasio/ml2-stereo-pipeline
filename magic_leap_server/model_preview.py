from __future__ import annotations

import copy
from typing import Any

import numpy as np
import open3d as o3d

from .pose_estimation import load_model
from .utils import as_vector3, as_quaternion, quaternion_to_rotation_matrix


_visualizer: Any = None
_point_cloud_geometry: Any = None
_model_geometry: Any = None
_reference_mesh: Any = None
_camera_initialized = False


def _ensure_visualizer() -> bool:
    global _visualizer, _point_cloud_geometry, _model_geometry, _reference_mesh

    if o3d is None:
        return False

    if (
        _visualizer is not None
        and _point_cloud_geometry is not None
        and _model_geometry is not None
        and _reference_mesh is not None
    ):
        return True

    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(window_name="MODEL POSE PREVIEW", width=960, height=720, visible=True)
    if not created:
        return False

    model_data = load_model()
    if model_data is None:
        visualizer.destroy_window()
        return False

    reference_mesh = model_data.get("mesh") if isinstance(model_data, dict) else None
    if reference_mesh is None or reference_mesh.is_empty():
        print("[WARN] Failed to initialize model preview mesh.")
        visualizer.destroy_window()
        return False

    reference_mesh = copy.deepcopy(reference_mesh)
    if not reference_mesh.has_vertex_normals():
        reference_mesh.compute_vertex_normals()

    point_cloud_geometry = o3d.geometry.PointCloud()
    visualizer.add_geometry(point_cloud_geometry)

    model_geometry = copy.deepcopy(reference_mesh)
    visualizer.add_geometry(model_geometry)

    render_option = visualizer.get_render_option()
    if render_option is not None:
        render_option.point_size = 2.0
        render_option.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)
        if hasattr(render_option, "mesh_show_back_face"):
            render_option.mesh_show_back_face = True

    _visualizer = visualizer
    _point_cloud_geometry = point_cloud_geometry
    _model_geometry = model_geometry
    _reference_mesh = reference_mesh
    return True


def _update_point_cloud_geometry(point_cloud: np.ndarray | None) -> None:
    if _point_cloud_geometry is None:
        return

    if point_cloud is None:
        _point_cloud_geometry.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return

    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.size == 0:
        _point_cloud_geometry.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.size == 0:
        _point_cloud_geometry.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return

    _point_cloud_geometry.points = o3d.utility.Vector3dVector(points)
    _point_cloud_geometry.paint_uniform_color([0.82, 0.82, 0.82])


def show_model_preview(
    position: np.ndarray | None,
    orientation: np.ndarray | None,
    point_cloud: np.ndarray | None = None,
) -> bool:
    """Display the model with the estimated position and orientation."""
    global _camera_initialized

    if not _ensure_visualizer():
        return False

    if _visualizer is None or _point_cloud_geometry is None or _model_geometry is None or _reference_mesh is None:
        return False

    _update_point_cloud_geometry(point_cloud)
    _visualizer.update_geometry(_point_cloud_geometry)

    translation = as_vector3(position)
    quaternion = as_quaternion(orientation)

    if translation is not None and quaternion is not None:
        transformed_mesh = copy.deepcopy(_reference_mesh)
        transformed_mesh.rotate(quaternion_to_rotation_matrix(quaternion), center=(0.0, 0.0, 0.0))
        transformed_mesh.translate(translation)

        _model_geometry.vertices = transformed_mesh.vertices
        _model_geometry.triangles = transformed_mesh.triangles
        if transformed_mesh.has_vertex_normals():
            _model_geometry.vertex_normals = transformed_mesh.vertex_normals
        if transformed_mesh.has_vertex_colors():
            _model_geometry.vertex_colors = transformed_mesh.vertex_colors

        _visualizer.update_geometry(_model_geometry)

    if not _camera_initialized:
        _visualizer.reset_view_point(True)
        _camera_initialized = True

    is_open = _visualizer.poll_events()
    _visualizer.update_renderer()
    return bool(is_open)


def close_model_preview() -> None:
    """Close the model pose preview window."""
    global _visualizer, _point_cloud_geometry, _model_geometry, _reference_mesh, _camera_initialized

    if _visualizer is not None:
        _visualizer.destroy_window()

    _visualizer = None
    _point_cloud_geometry = None
    _model_geometry = None
    _reference_mesh = None
    _camera_initialized = False