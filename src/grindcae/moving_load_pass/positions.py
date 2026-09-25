"""Complete-pass position generation reused from the Phase 4C1 coordinate."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae.pass_loading import PassLoadResult
from grindcae.trajectory import (
    PASS_STATES,
    motion_coordinate_from_physical_x,
    motion_coordinate_tolerance,
    pass_state_from_motion_coordinate,
    physical_x_from_motion_coordinate,
)

from .models import FixedMeshMovingLoadPassCase


class MovingLoadPositionError(RuntimeError):
    """Raised when a complete moving-load position sequence is invalid."""


@dataclass(frozen=True, slots=True)
class MovingLoadPosition:
    position_id: int
    motion_coordinate_m: float
    wheel_lowest_point_x_m: float
    pass_state: str
    generation_roles: tuple[str, ...]


def _base_positions(start: float, end: float, count: int) -> list[float]:
    span = end - start
    values = [start + span * index / (count - 1) for index in range(count)]
    values[0] = start
    values[-1] = end
    return values


def generate_moving_load_positions(
    case: FixedMeshMovingLoadPassCase,
    reference_pass_load: PassLoadResult,
) -> tuple[MovingLoadPosition, ...]:
    if not isinstance(case, FixedMeshMovingLoadPassCase):
        raise TypeError("case must be a FixedMeshMovingLoadPassCase")
    if not isinstance(reference_pass_load, PassLoadResult):
        raise TypeError("reference_pass_load must be a PassLoadResult")
    if reference_pass_load.case != case.reference_case.pass_load:
        raise MovingLoadPositionError("reference pass load does not match reference_case")
    trajectory = reference_pass_load.trajectory_result
    width = trajectory.case.workpiece.length_m
    arc = trajectory.exact_arc_projected_length_m
    start = -arc
    end = width
    direction = trajectory.case.single_pass.relative_feed_direction
    reference_motion = motion_coordinate_from_physical_x(
        trajectory.case.single_pass.wheel_lowest_point_x_m, width, direction
    )
    tolerance = motion_coordinate_tolerance(width, arc)
    if reference_motion < start - tolerance or reference_motion > end + tolerance:
        raise MovingLoadPositionError(
            "reference_case wheel position must lie within the complete single-pass range"
        )
    candidates: list[tuple[float, tuple[str, ...], int]] = [
        (value, ("uniform_base",), 0)
        for value in _base_positions(start, end, case.pass_settings.base_position_count)
    ]
    candidates.extend(
        (
            (start, ("scan_start", "entry_boundary"), 3),
            (-0.5 * arc, ("representative_entry",), 3),
            (0.0, ("full_contact_start",), 3),
            (0.5 * (width - arc), ("representative_full_contact",), 3),
            (width - arc, ("full_contact_end",), 3),
            (width - 0.5 * arc, ("representative_exit",), 3),
            (end, ("scan_end", "exit_boundary"), 3),
            (reference_motion, ("reference_case",), 2),
        )
    )
    candidates.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, tuple[str, ...], int]]] = []
    for item in candidates:
        if not clusters or abs(item[0] - clusters[-1][-1][0]) > tolerance:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    positions: list[MovingLoadPosition] = []
    for cluster in clusters:
        coordinate, _, _ = max(cluster, key=lambda item: item[2])
        roles = tuple(sorted({role for _, item_roles, _ in cluster for role in item_roles}))
        positions.append(
            MovingLoadPosition(
                position_id=len(positions),
                motion_coordinate_m=coordinate,
                wheel_lowest_point_x_m=physical_x_from_motion_coordinate(
                    coordinate, width, direction
                ),
                pass_state=pass_state_from_motion_coordinate(coordinate, width, arc),
                generation_roles=roles,
            )
        )
    if positions[0].motion_coordinate_m != start or positions[-1].motion_coordinate_m != end:
        raise MovingLoadPositionError("complete-pass endpoints were not preserved exactly")
    if any(a.motion_coordinate_m >= b.motion_coordinate_m for a, b in zip(positions, positions[1:])):
        raise MovingLoadPositionError("motion coordinates must be strictly increasing")
    physical = [item.wheel_lowest_point_x_m for item in positions]
    if direction == "positive_x" and any(a >= b for a, b in zip(physical, physical[1:])):
        raise MovingLoadPositionError("positive_x physical positions must increase")
    if direction == "negative_x" and any(a <= b for a, b in zip(physical, physical[1:])):
        raise MovingLoadPositionError("negative_x physical positions must decrease")
    if {item.pass_state for item in positions} != set(PASS_STATES):
        raise MovingLoadPositionError("complete pass must include all five Phase 4C1 states")
    required_roles = (
        "entry_boundary",
        "full_contact_start",
        "full_contact_end",
        "exit_boundary",
        "representative_entry",
        "representative_full_contact",
        "representative_exit",
        "reference_case",
    )
    for role in required_roles:
        if sum(role in item.generation_roles for item in positions) != 1:
            raise MovingLoadPositionError(f"position role {role} must occur exactly once")
    if not all(math.isfinite(item.motion_coordinate_m) for item in positions):
        raise MovingLoadPositionError("motion coordinates must be finite")
    return tuple(positions)
