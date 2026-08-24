import math
import numpy as np
import torch
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


class InvalidCheckpointMetadataError(ValueError):
    """Raised when released garment metadata references nonexistent mesh vertices."""


def step_interval(interval=50):
    """Factory function: creates a customizable step interval decorator"""

    def decorator(func):
        call_count = 0

        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count % interval == 0:
                return func(*args, **kwargs)
            else:
                # Return False for skipped steps (maintains backward compatibility)
                # For success_checker_garment_fold, this will be handled in _check_success
                return False

        return wrapper

    return decorator


def calculate_distance(point_a, point_b):
    # Calculate distance
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


def _resolve_particle_indices(particle_object, index_list, mesh_point_count):
    """Translate authored USD checkpoints into the cooked PhysX particle order."""

    try:
        authored_indices = np.asarray(index_list)
    except (TypeError, ValueError) as error:
        raise InvalidCheckpointMetadataError(
            "garment checkpoint metadata must be a one-dimensional integer array"
        ) from error
    if authored_indices.ndim != 1 or authored_indices.dtype.kind not in "iu":
        raise InvalidCheckpointMetadataError(
            "garment checkpoint metadata must be a one-dimensional integer array"
        )
    authored_indices = authored_indices.astype(np.int64, copy=False)

    device = str(getattr(particle_object, "_device", "")).lower()
    if device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
        prim = getattr(particle_object, "_prim", None)
        get_attribute = getattr(prim, "GetAttribute", None)
        if not callable(get_attribute):
            raise InvalidCheckpointMetadataError(
                "garment PhysX checkpoint topology cannot prove cooked particle identity"
            )
        if callable(get_attribute):
            def read_map(name):
                attribute = get_attribute(name)
                if attribute is None:
                    return None
                is_valid = getattr(attribute, "IsValid", None)
                if callable(is_valid) and not is_valid():
                    return None
                get_value = getattr(attribute, "Get", None)
                value = get_value() if callable(get_value) else None
                if value is None:
                    return None
                vector = np.asarray(value)
                return None if vector.size == 0 else vector

            remap_to_orig = read_map("physxParticle:weldedVerticesRemapToOrig")
            remap_to_weld = read_map("physxParticle:weldedVerticesRemapToWeld")
            if (remap_to_orig is None) != (remap_to_weld is None):
                raise InvalidCheckpointMetadataError(
                    "garment PhysX checkpoint topology has an incomplete cooked weld map"
                )
            if remap_to_orig is None:
                welded_triangles = read_map(
                    "physxParticle:weldedTriangleIndices"
                )
                if welded_triangles is not None and welded_triangles.size:
                    raise InvalidCheckpointMetadataError(
                        "garment PhysX welded topology is missing cooked vertex maps"
                    )
                get_authored_points = getattr(
                    particle_object, "_get_points_pose", None
                )
                try:
                    authored_points = get_authored_points()
                    for method_name in ("detach", "cpu", "numpy"):
                        method = getattr(authored_points, method_name, None)
                        if callable(method):
                            authored_points = method()
                    authored_points = np.asarray(authored_points, dtype=np.float32)
                except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                    raise InvalidCheckpointMetadataError(
                        "garment PhysX checkpoint topology cannot prove unwelded identity topology"
                    ) from error
                normalized = authored_points.copy()
                normalized[normalized == 0.0] = 0.0
                if (
                    authored_points.ndim != 2
                    or authored_points.shape[1:] != (3,)
                    or len(authored_points) != mesh_point_count
                    or not np.isfinite(authored_points).all()
                    or len(np.unique(normalized, axis=0)) != mesh_point_count
                ):
                    raise InvalidCheckpointMetadataError(
                        "garment PhysX checkpoint topology cannot prove unwelded identity topology"
                    )
            if remap_to_orig is not None:
                if (
                    remap_to_orig.ndim != 1
                    or remap_to_weld.ndim != 1
                    or remap_to_orig.dtype.kind not in "iu"
                    or remap_to_weld.dtype.kind not in "iu"
                    or len(remap_to_orig) != mesh_point_count
                ):
                    raise InvalidCheckpointMetadataError(
                        "garment PhysX checkpoint topology has an invalid cooked weld map"
                    )
                remap_to_orig = remap_to_orig.astype(np.int64, copy=False)
                remap_to_weld = remap_to_weld.astype(np.int64, copy=False)
                if (
                    np.any(remap_to_orig < 0)
                    or np.any(remap_to_orig >= len(remap_to_weld))
                    or np.any(remap_to_weld < 0)
                    or np.any(remap_to_weld >= mesh_point_count)
                    or not np.array_equal(
                        remap_to_weld[remap_to_orig],
                        np.arange(mesh_point_count, dtype=np.int64),
                    )
                ):
                    raise InvalidCheckpointMetadataError(
                        "garment PhysX checkpoint topology has a non-invertible cooked weld map"
                    )
                invalid_authored = authored_indices[
                    (authored_indices < 0) | (authored_indices >= len(remap_to_weld))
                ]
                if invalid_authored.size:
                    raise InvalidCheckpointMetadataError(
                        "garment checkpoint metadata references nonexistent authored vertices: "
                        f"invalid_indices={invalid_authored.tolist()}, "
                        f"authored_point_count={len(remap_to_weld)}"
                    )
                return remap_to_weld[authored_indices]

    invalid_indices = authored_indices[
        (authored_indices < 0) | (authored_indices >= mesh_point_count)
    ]
    if invalid_indices.size:
        raise InvalidCheckpointMetadataError(
            "garment checkpoint metadata references nonexistent mesh vertices: "
            f"invalid_indices={invalid_indices.tolist()}, mesh_point_count={mesh_point_count}"
        )
    return authored_indices


def get_object_particle_position(particle_object, index_list):
    try:
        transformed_mesh_points, _, _, _ = particle_object.get_current_mesh_points()
    except Exception as e1:
        try:
            logger.error(f"Error in get_object_particle_position: {e1}")
            transformed_mesh_points = (
                particle_object._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as e2:
            logger.error(f"Error in get_object_particle_position: {e2}")
            return

    mesh_point_count = len(transformed_mesh_points)
    particle_indices = _resolve_particle_indices(
        particle_object, index_list, mesh_point_count
    )
    positions = (transformed_mesh_points[particle_indices] * 100).tolist()
    return positions


@step_interval(interval=50)
def success_checker_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


def check_top_sleeve(p, success_distance):
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_2_3 = calculate_distance(p[2], p[3])
    dist_1_5 = calculate_distance(p[1], p[5])
    dist_0_1 = calculate_distance(p[0], p[1])
    dist_4_5 = calculate_distance(p[4], p[5])
    cond1 = dist_0_4 <= success_distance[0]
    cond2 = dist_2_3 <= success_distance[1]
    cond3 = dist_1_5 <= success_distance[2]
    cond4 = dist_0_1 >= success_distance[3]
    cond5 = dist_4_5 >= success_distance[4]

    details = {
        "condition_1": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} <= {success_distance[0]}",
            "value": dist_0_4,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[2], p[3]) = {dist_2_3:.2f} <= {success_distance[1]}",
            "value": dist_2_3,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} <= {success_distance[2]}",
            "value": dist_1_5,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[0], p[1]) = {dist_0_1:.2f} >= {success_distance[3]}",
            "value": dist_0_1,
            "threshold": success_distance[3],
            "passed": cond4,
        },
        "condition_5": {
            "description": f"dist(p[4], p[5]) = {dist_4_5:.2f} >= {success_distance[4]}",
            "value": dist_4_5,
            "threshold": success_distance[4],
            "passed": cond5,
        },
    }

    return cond1 and cond2 and cond3 and cond4 and cond5, details

def check_pant_long(p, success_distance):
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_0_2 = calculate_distance(p[0], p[2])   
    dist_1_3 = calculate_distance(p[1], p[3])
    dist_1_5 = calculate_distance(p[1], p[5])
    cond1 = dist_0_4 <= success_distance[0]
    cond2 = dist_0_2 >= success_distance[1]
    cond3 = dist_1_3 >= success_distance[2]
    cond4 = dist_1_5 <= success_distance[3]
    details = {
        "condition_1": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} <= {success_distance[0]}",
            "value": dist_0_4,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[0], p[2]) = {dist_0_2:.2f} >= {success_distance[1]}",
            "value": dist_0_2,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[1], p[3]) = {dist_1_3:.2f} >= {success_distance[2]}",
            "value": dist_1_3,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} <= {success_distance[3]}",
            "value": dist_1_5,
            "threshold": success_distance[3],
            "passed": cond4,
        },
    }
    return cond1 and cond2 and cond3 and cond4, details

def check_pant_short(p, success_distance):
    dist_0_1 = calculate_distance(p[0], p[1])
    dist_4_5 = calculate_distance(p[4], p[5])
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_1_5 = calculate_distance(p[1], p[5])
    cond1 = dist_0_1 <= success_distance[0]
    cond2 = dist_4_5 <= success_distance[1]
    cond3 = dist_0_4 >= success_distance[2]
    cond4 = dist_1_5 >= success_distance[3]

    details = {
        "condition_1": {
            "description": f"dist(p[0], p[1]) = {dist_0_1:.2f} <= {success_distance[0]}",
            "value": dist_0_1,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[4], p[5]) = {dist_4_5:.2f} <= {success_distance[1]}",
            "value": dist_4_5,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} >= {success_distance[2]}",
            "value": dist_0_4,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} >= {success_distance[3]}",
            "value": dist_1_5,
            "threshold": success_distance[3],
            "passed": cond4,
        },
    }
    return cond1 and cond2 and cond3 and cond4, details

def success_checker_garment_fold_unthrottled(particle_object, garment_type: str):
    """Evaluate garment-fold success immediately without changing physics."""
    check_point_indices = particle_object.check_points  # list[int]
    raw_success_distance = particle_object.success_distance  # list[int]
    current_scale = float(particle_object.init_scale[0])
    success_distance = [d * current_scale for d in raw_success_distance]
    try:
        p = get_object_particle_position(particle_object, check_point_indices)
    except InvalidCheckpointMetadataError as error:
        logger.error(f"Invalid garment checkpoint metadata: {error}")
        return {
            "success": False,
            "garment_type": garment_type,
            "thresholds": success_distance,
            "details": {},
            "metadata_valid": False,
            "metadata_error": str(error),
        }

    if garment_type == "top-long-sleeve" or garment_type == "top-short-sleeve":
        success, details = check_top_sleeve(p, success_distance)
    elif garment_type == "short-pant":
        success, details = check_pant_short(p, success_distance)
    elif garment_type == "long-pant":
        success, details = check_pant_long(p, success_distance)
    else:
        raise ValueError(f"Unknown garment_type: {garment_type}")

    result = {
        "success": bool(success),
        "garment_type": garment_type,
        "thresholds": success_distance,
        "details": details,
        "metadata_valid": True,
    }

    return result


@step_interval(interval=50)
def success_checker_garment_fold(particle_object, garment_type: str):
    """Preserve the ordinary evaluation checker cadence."""

    return success_checker_garment_fold_unthrottled(particle_object, garment_type)


@step_interval(interval=50)
def success_checker_fling(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)

    def xy_distance(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def z_distance(a, b):
        return abs(a[2] - b[2])

    success = (
        xy_distance(p[0], p[4]) > 18
        and z_distance(p[0], p[4]) < 2
        and xy_distance(p[1], p[5]) > 18
        and z_distance(p[1], p[5]) < 2
    )

    return bool(success)


@step_interval(interval=30)
def success_checker_burger(beef_pos, plate_pos):
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    # z distance
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # Success condition: xy < 0.045 and z < 0.03
    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2
