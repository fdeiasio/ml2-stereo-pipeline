from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .utils import (
    as_quaternion,
    as_vector3,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from .config import (
    ICP_MAX_ITERATIONS,
    ICP_MIN_FITNESS,
    MODEL_PATH,
    MODEL_SAMPLE_POINTS,
    MODEL_SCALE_FACTOR,
    MODEL_VOXEL_SIZE,
    SCENE_VOXEL_SIZE,
)


def _prepare_scene_point_cloud(point_cloud: np.ndarray | None) -> o3d.geometry.PointCloud | None:
    if point_cloud is None:
        return None

    scene_points = np.asarray(point_cloud, dtype=np.float64)
    if scene_points.ndim != 2 or scene_points.shape[1] != 3:
        return None

    finite_mask = np.isfinite(scene_points).all(axis=1)
    scene_points = scene_points[finite_mask]
    if scene_points.shape[0] == 0:
        return None

    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(scene_points)

    downsampled_scene = scene_cloud.voxel_down_sample(SCENE_VOXEL_SIZE)
    if downsampled_scene is not None and not downsampled_scene.is_empty():
        scene_cloud = downsampled_scene

    if len(scene_cloud.points) == 0:
        return None

    return scene_cloud


def _get_model_scale(model_points: o3d.geometry.PointCloud) -> float:
    bounds = model_points.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    scale = float(np.linalg.norm(extent))
    if not np.isfinite(scale) or scale <= 0.0:
        return 0.1

    return scale


def _build_initial_transform(
    prev_position: np.ndarray | None,
    prev_orientation: np.ndarray | None,
    scene_cloud: o3d.geometry.PointCloud,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)

    position = as_vector3(prev_position)
    orientation = as_quaternion(prev_orientation)
    if position is not None and orientation is not None:
        transform[:3, :3] = quaternion_to_rotation_matrix(orientation)
        transform[:3, 3] = position
        return transform

    scene_center = np.asarray(scene_cloud.get_center(), dtype=np.float64)
    if scene_center.shape[0] == 3 and np.all(np.isfinite(scene_center)):
        transform[:3, 3] = scene_center

    return transform


def _crop_scene_around_previous_pose(
    scene_cloud: o3d.geometry.PointCloud,
    prev_position: np.ndarray | None,
    model_scale: float,
) -> o3d.geometry.PointCloud:
    position = as_vector3(prev_position)
    if position is None:
        return scene_cloud

    points = np.asarray(scene_cloud.points, dtype=np.float64)
    if points.size == 0:
        return scene_cloud

    crop_radius = max(model_scale * 1.75, 0.05)
    squared_distances = np.sum((points - position) ** 2, axis=1)
    inside_mask = squared_distances <= (crop_radius * crop_radius)
    if int(np.count_nonzero(inside_mask)) == 0:
        return scene_cloud

    cropped_cloud = o3d.geometry.PointCloud()
    cropped_cloud.points = o3d.utility.Vector3dVector(points[inside_mask])
    return cropped_cloud


def load_model(model_path: str | None = None) -> dict[str, Any] | None:
    """Load and pre-process the model used by ICP."""
    if o3d is None or np is None:
        return None

    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        print(f"[WARN] Model file not found: {path}")
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(str(path))
    except Exception as exc:
        print(f"[WARN] Failed to load model mesh from {path}: {exc}")
        return None

    if mesh is None or mesh.is_empty():
        print(f"[WARN] Loaded model mesh is empty: {path}")
        return None

    if MODEL_SCALE_FACTOR <= 0.0:
        print(f"[WARN] Invalid MODEL_SCALE_FACTOR={MODEL_SCALE_FACTOR}. Falling back to 1.0.")
        scale_factor = 1.0
    else:
        scale_factor = float(MODEL_SCALE_FACTOR)
    mesh.scale(scale_factor, center=(0.0, 0.0, 0.0))

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    try:
        model_points = mesh.sample_points_uniformly(number_of_points=MODEL_SAMPLE_POINTS)
    except Exception as exc:
        print(f"[WARN] Failed to sample model points from {path}: {exc}")
        return None

    if model_points is None or model_points.is_empty():
        print(f"[WARN] Failed to extract point cloud samples from model: {path}")
        return None

    model_center = np.asarray(model_points.get_center(), dtype=np.float64)
    mesh.translate(-model_center)
    model_points.translate(-model_center)

    downsampled_model = model_points.voxel_down_sample(MODEL_VOXEL_SIZE)
    if downsampled_model is not None and not downsampled_model.is_empty():
        model_points = downsampled_model

    return {
        "path": str(path),
        "mesh": mesh,
        "point_cloud": model_points,
    }


def estimate_model_pose(
    point_cloud: np.ndarray | None,
    model: dict[str, Any] | None,
    prev_position: np.ndarray | None = None,
    prev_orientation: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate model pose from a scene point cloud using ICP."""
    if o3d is None or np is None:
        return None, None
    if point_cloud is None or model is None:
        return None, None

    model_points = model.get("point_cloud") if isinstance(model, dict) else None
    if model_points is None or model_points.is_empty():
        print("[WARN] Missing model point cloud for pose estimation.")
        return None, None

    scene_cloud = _prepare_scene_point_cloud(point_cloud)
    if scene_cloud is None:
        return None, None

    model_scale = _get_model_scale(model_points)
    scene_cloud = _crop_scene_around_previous_pose(scene_cloud, prev_position, model_scale)

    initial_transform = _build_initial_transform(prev_position, prev_orientation, scene_cloud)
    max_correspondence_distance = max(model_scale * 0.1, 0.01)

    try:
        registration_result = o3d.pipelines.registration.registration_icp(
            model_points,
            scene_cloud,
            max_correspondence_distance=max_correspondence_distance,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS),
        )
    except Exception as exc:
        print(f"[WARN] ICP failed during pose estimation: {exc}")
        return None, None

    if not np.isfinite(registration_result.fitness) or registration_result.fitness < ICP_MIN_FITNESS:
        print(
            "[WARN] ICP pose estimate rejected "
            f"(fitness={registration_result.fitness:.4f}, rmse={registration_result.inlier_rmse:.4f})."
        )
        return None, None

    transform = np.asarray(registration_result.transformation, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        print("[WARN] ICP returned an invalid transformation matrix.")
        return None, None

    position = transform[:3, 3]
    orientation = rotation_matrix_to_quaternion(transform[:3, :3])

    previous_quaternion = as_quaternion(prev_orientation)
    if previous_quaternion is not None and float(np.dot(orientation, previous_quaternion)) < 0.0:
        orientation *= -1.0

    return position.astype(np.float32, copy=False), orientation.astype(np.float32, copy=False)