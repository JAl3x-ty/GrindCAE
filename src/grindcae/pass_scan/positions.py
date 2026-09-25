"""Deterministic complete-pass position generation using phase 4C1 geometry."""

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

from .models import PassScanCase


class PassScanPositionError(RuntimeError):
    """Raised when complete-pass positions cannot meet the scan contract."""


@dataclass(frozen=True, slots=True)
class ScanPosition:
    """One actual solve position ordered along the wheel trajectory."""

    scan_index: int
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


def generate_scan_positions(
    case: PassScanCase,
    reference_pass_load: PassLoadResult,
) -> tuple[ScanPosition, ...]:
    """Generate, insert, de-duplicate, and classify all complete-pass positions."""

    if not isinstance(case, PassScanCase):
        raise TypeError("case must be a PassScanCase")
    if not isinstance(reference_pass_load, PassLoadResult):
        raise TypeError("reference_pass_load must be a PassLoadResult")
    if reference_pass_load.case != case.reference_case.pass_load:
        raise PassScanPositionError("reference pass load does not match reference_case")
    trajectory = reference_pass_load.trajectory_result
    width = trajectory.case.workpiece.length_m
    arc = trajectory.exact_arc_projected_length_m
    start = -arc
    end = width
    direction = trajectory.case.single_pass.relative_feed_direction
    reference_motion = motion_coordinate_from_physical_x(
        trajectory.case.single_pass.wheel_lowest_point_x_m,
        width,
        direction,
    )
    tolerance = motion_coordinate_tolerance(width, arc)
    if reference_motion < start - tolerance or reference_motion > end + tolerance:
        raise PassScanPositionError(
            "reference_case wheel position must lie within the complete single-pass range"
        )

    candidates: list[tuple[float, tuple[str, ...], int]] = [
        (value, ("uniform_base",), 0)
        for value in _base_positions(start, end, case.scan.base_position_count)
    ]
    # 中文导读：在均匀位置中强制插入进刀、全接触、退出和参考位置。
    critical = (
        (start, ("scan_start",), 3),
        (-0.5 * arc, ("representative_entry",), 3),
        (0.0, ("full_contact_start",), 3),
        (0.5 * (width - arc), ("representative_full_contact",), 3),
        (width - arc, ("full_contact_end",), 3),
        (width - 0.5 * arc, ("representative_exit",), 3),
        (end, ("scan_end",), 3),
        (reference_motion, ("reference_case",), 2),
    )
    candidates.extend(critical)
    candidates.sort(key=lambda item: item[0])

    clusters: list[list[tuple[float, tuple[str, ...], int]]] = []
    for candidate in candidates:
        if not clusters or abs(candidate[0] - clusters[-1][-1][0]) > tolerance:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    positions: list[ScanPosition] = []
    for cluster in clusters:
        coordinate, _, _ = max(cluster, key=lambda item: item[2])
        roles = tuple(sorted({role for _, item_roles, _ in cluster for role in item_roles}))
        physical_x = physical_x_from_motion_coordinate(coordinate, width, direction)
        state = pass_state_from_motion_coordinate(coordinate, width, arc)
        positions.append(
            ScanPosition(
                scan_index=len(positions),
                motion_coordinate_m=coordinate,
                wheel_lowest_point_x_m=physical_x,
                pass_state=state,
                generation_roles=roles,
            )
        )

    if positions[0].motion_coordinate_m != start or positions[-1].motion_coordinate_m != end:
        raise PassScanPositionError("scan endpoints were not preserved exactly")
    if any(
        first.motion_coordinate_m >= second.motion_coordinate_m
        for first, second in zip(positions, positions[1:])
    ):
        raise PassScanPositionError("scan positions must be strictly ordered")
    physical = [item.wheel_lowest_point_x_m for item in positions]
    if direction == "positive_x" and any(a >= b for a, b in zip(physical, physical[1:])):
        raise PassScanPositionError("positive_x physical positions must increase")
    if direction == "negative_x" and any(a <= b for a, b in zip(physical, physical[1:])):
        raise PassScanPositionError("negative_x physical positions must decrease")
    states = {item.pass_state for item in positions}
    if states != set(PASS_STATES):
        raise PassScanPositionError("the complete scan must include all five pass states")
    for role in (
        "representative_entry",
        "representative_full_contact",
        "representative_exit",
        "reference_case",
    ):
        if sum(role in item.generation_roles for item in positions) != 1:
            raise PassScanPositionError(f"scan role {role} must occur exactly once")
    if not all(math.isfinite(item.motion_coordinate_m) for item in positions):
        raise PassScanPositionError("scan positions must be finite")
    return tuple(positions)
