"""Phase 4C1 single-pass circular-wheel geometry and surface sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae import __version__

from .models import TrajectoryCase


RESULT_FORMAT = "grindcae_phase_4c1_single_pass_surface_profile"
REGIONS = ("ground", "contact_arc", "unprocessed")
PASS_STATES = (
    "before_entry",
    "entry",
    "full_contact",
    "exit",
    "after_exit",
)


class TrajectoryError(RuntimeError):
    """Raised when valid inputs cannot produce a consistent phase 4C1 result."""


def _trajectory_error(field: str, unit: str, reason: str) -> None:
    raise TrajectoryError(f"{field} [{unit}]: {reason}")


def _finite(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _trajectory_error(field, unit, "must be numeric")
    try:
        number = float(value)
    except OverflowError:
        _trajectory_error(field, unit, "must be finite and representable")
    if not math.isfinite(number):
        _trajectory_error(field, unit, "must be finite")
    return number


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def motion_coordinate_tolerance(width_m: float, arc_length_m: float) -> float:
    """Return the shared tolerance for one-pass motion-coordinate operations."""

    scale = max(abs(width_m), abs(arc_length_m), 1.0e-300)
    return max(64.0 * math.ulp(scale), 1.0e-15 * scale)


def _snap_motion_position(
    value_m: float,
    width_m: float,
    arc_length_m: float,
) -> float:
    boundaries = (-arc_length_m, 0.0, width_m - arc_length_m, width_m)
    minimum_separation = min(
        second - first for first, second in zip(boundaries, boundaries[1:])
    )
    tolerance = min(
        motion_coordinate_tolerance(width_m, arc_length_m),
        0.25 * minimum_separation,
    )
    nearest = min(boundaries, key=lambda boundary: abs(value_m - boundary))
    if math.isclose(value_m, nearest, rel_tol=0.0, abs_tol=tolerance):
        return nearest
    return value_m


def pass_state_from_motion_coordinate(
    motion_position_m: float,
    workpiece_length_m: float,
    arc_length_m: float,
) -> str:
    """Classify one shared motion coordinate using the phase 4C1 rules."""

    # 中文导读：用同一运动坐标划分进刀前、进刀、全接触、退出和退出后。
    if motion_position_m <= -arc_length_m:
        return "before_entry"
    if motion_position_m < 0.0:
        return "entry"
    if motion_position_m <= workpiece_length_m - arc_length_m:
        return "full_contact"
    if motion_position_m < workpiece_length_m:
        return "exit"
    return "after_exit"


def motion_coordinate_from_physical_x(
    x_m: float, width_m: float, direction: str
) -> float:
    """Map physical wheel-bottom x to the phase 4C1 motion coordinate."""

    return x_m if direction == "positive_x" else width_m - x_m


def physical_x_from_motion_coordinate(
    motion_position_m: float, width_m: float, direction: str
) -> float:
    """Map the phase 4C1 motion coordinate back to physical wheel-bottom x."""

    return motion_position_m if direction == "positive_x" else width_m - motion_position_m


def _from_motion_interval(
    start_m: float,
    end_m: float,
    width_m: float,
    direction: str,
) -> tuple[float, float]:
    if direction == "positive_x":
        return start_m, end_m
    return width_m - end_m, width_m - start_m


@dataclass(frozen=True, slots=True)
class SurfaceSegment:
    """One non-empty part of the current ideal workpiece surface."""

    region: str
    x_start_m: float
    x_end_m: float
    surface_description: str

    def __post_init__(self) -> None:
        if self.region not in REGIONS:
            _trajectory_error("surface_segments.region", "region literal", "is invalid")
        start = _finite(self.x_start_m, "surface_segments.x_start_m", "m")
        end = _finite(self.x_end_m, "surface_segments.x_end_m", "m")
        if not start < end:
            _trajectory_error(
                "surface_segments.length_m", "m", "must be strictly greater than zero"
            )
        if not isinstance(self.surface_description, str) or not self.surface_description:
            _trajectory_error(
                "surface_segments.surface_description",
                "description",
                "must be non-empty",
            )
        object.__setattr__(self, "x_start_m", start)
        object.__setattr__(self, "x_end_m", end)

    @property
    def length_m(self) -> float:
        return self.x_end_m - self.x_start_m

    def to_dict(self) -> dict[str, float | str]:
        return {
            "region": self.region,
            "x_start_m": self.x_start_m,
            "x_end_m": self.x_end_m,
            "length_m": self.length_m,
            "surface_description": self.surface_description,
        }


@dataclass(frozen=True, slots=True)
class SurfaceProfilePoint:
    """One deterministic point on the ideal mean surface profile."""

    x_m: float
    surface_y_m: float
    original_surface_height_m: float
    region: str

    def __post_init__(self) -> None:
        x_value = _finite(self.x_m, "surface_profile.x_m", "m")
        y_value = _finite(self.surface_y_m, "surface_profile.surface_y_m", "m")
        height = _finite(
            self.original_surface_height_m,
            "surface_profile.original_surface_height_m",
            "m",
        )
        if self.region not in REGIONS:
            _trajectory_error("surface_profile.region", "region literal", "is invalid")
        object.__setattr__(self, "x_m", x_value)
        object.__setattr__(self, "surface_y_m", y_value)
        object.__setattr__(self, "original_surface_height_m", height)

    @property
    def surface_elevation_relative_to_original_m(self) -> float:
        return self.surface_y_m - self.original_surface_height_m

    @property
    def removed_depth_m(self) -> float:
        return self.original_surface_height_m - self.surface_y_m


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Validated geometry, partition, and sampled ideal surface."""

    case: TrajectoryCase
    wheel_radius_m: float
    ground_surface_height_m: float
    exact_arc_projected_length_m: float
    shallow_cut_approximation_length_m: float
    length_absolute_difference_m: float
    length_relative_difference: float
    motion_position_m: float
    pass_state: str
    theoretical_arc_interval_m: tuple[float, float]
    effective_arc_interval_m: tuple[float, float] | None
    effective_contact_length_m: float
    contact_ratio: float
    surface_segments: tuple[SurfaceSegment, ...]
    sample_points: tuple[SurfaceProfilePoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, TrajectoryCase):
            _trajectory_error("case", "TrajectoryCase", "has the wrong object type")
        if self.pass_state not in PASS_STATES:
            _trajectory_error("pass_state", "state literal", "is invalid")
        for field, unit in (
            ("wheel_radius_m", "m"),
            ("ground_surface_height_m", "m"),
            ("exact_arc_projected_length_m", "m"),
            ("shallow_cut_approximation_length_m", "m"),
            ("length_absolute_difference_m", "m"),
            ("length_relative_difference", "dimensionless"),
            ("motion_position_m", "m"),
            ("effective_contact_length_m", "m"),
            ("contact_ratio", "dimensionless"),
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, unit))

        width = self.case.workpiece.length_m
        depth = self.case.single_pass.depth_of_cut_m
        height = self.case.workpiece.original_surface_height_m
        if self.wheel_radius_m <= 0.0 or self.exact_arc_projected_length_m <= 0.0:
            _trajectory_error("derived_geometry", "m", "lengths must be positive")
        if not 0.0 <= self.contact_ratio <= 1.0:
            _trajectory_error("contact_ratio", "dimensionless", "must lie in [0, 1]")
        if not math.isclose(
            self.effective_contact_length_m,
            self.contact_ratio * self.exact_arc_projected_length_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            _trajectory_error(
                "contact_ratio",
                "dimensionless",
                "must match effective_contact_length_m / exact_arc_projected_length_m",
            )

        segments = tuple(self.surface_segments)
        points = tuple(self.sample_points)
        if not segments or not points:
            _trajectory_error("surface_profile", "partition and points", "must be non-empty")
        if tuple(sorted(segments, key=lambda item: item.x_start_m)) != segments:
            _trajectory_error("surface_segments", "ordered intervals", "must be sorted")
        for index, segment in enumerate(segments):
            if segment.x_start_m < 0.0 or segment.x_end_m > width:
                _trajectory_error("surface_segments", "m", "must lie within [0, W]")
            if index and not math.isclose(
                segments[index - 1].x_end_m,
                segment.x_start_m,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                _trajectory_error(
                    "surface_segments", "m", "must be contiguous and non-overlapping"
                )
        partition_total = math.fsum(segment.length_m for segment in segments)
        if not math.isclose(partition_total, width, rel_tol=1.0e-12, abs_tol=1.0e-15):
            _trajectory_error("surface_segments", "m", "lengths must sum to W")

        if points[0].x_m != 0.0 or points[-1].x_m != width:
            _trajectory_error("surface_profile.x_m", "m", "must include exact 0 and W")
        for first, second in zip(points, points[1:]):
            if not first.x_m < second.x_m:
                _trajectory_error(
                    "surface_profile.x_m", "m", "must be strictly increasing"
                )
        ground_height = height - depth
        for point in points:
            if not -1.0e-15 <= point.x_m <= width + 1.0e-15:
                _trajectory_error("surface_profile.x_m", "m", "must lie in [0, W]")
            if not ground_height - 1.0e-15 <= point.surface_y_m <= height + 1.0e-15:
                _trajectory_error(
                    "surface_profile.surface_y_m", "m", "must lie in [Hground, H]"
                )
            if not -1.0e-15 <= point.removed_depth_m <= depth + 1.0e-15:
                _trajectory_error(
                    "surface_profile.removed_depth_m", "m", "must lie in [0, ae]"
                )
        object.__setattr__(self, "surface_segments", segments)
        object.__setattr__(self, "sample_points", points)

    @property
    def wheel_center_coordinates_m(self) -> tuple[float, float]:
        return (
            self.case.single_pass.wheel_lowest_point_x_m,
            self.ground_surface_height_m + self.wheel_radius_m,
        )

    def region_length_m(self, region: str) -> float:
        return math.fsum(
            segment.length_m for segment in self.surface_segments if segment.region == region
        )

    def to_summary(self, artifacts: dict[str, str]) -> dict[str, object]:
        """Return the deterministic phase 4C1 JSON payload."""

        direction = self.case.single_pass.relative_feed_direction
        height = self.case.workpiece.original_surface_height_m
        depth = self.case.single_pass.depth_of_cut_m
        partition_total = math.fsum(segment.length_m for segment in self.surface_segments)
        state_flags = {state: state == self.pass_state for state in PASS_STATES}
        return {
            "result_format": RESULT_FORMAT,
            "package_version": __version__,
            "trajectory_schema_version": self.case.trajectory_schema_version,
            "unit_system": self.case.unit_system,
            "normalized_input": self.case.to_dict(),
            "units": {
                "length": "m",
                "surface_height": "m",
                "relative_surface_elevation": "m",
                "removed_depth": "m",
                "contact_ratio": "dimensionless",
            },
            "direction_convention": {
                "relative_feed_direction": direction,
                "meaning": "Direction of the wheel-lowest-point trajectory relative to the workpiece x coordinate.",
                "positive_x": "Processed surface trails toward lower x; the circular transition is ahead toward higher x.",
                "negative_x": "Mirror of positive_x: processed surface trails toward higher x; the circular transition is ahead toward lower x.",
                "not_schema_4_tangential_direction": True,
                "not_machine_table_motion_direction": True,
            },
            "derived_geometry": {
                "wheel_radius_m": self.wheel_radius_m,
                "original_surface_height_m": height,
                "ground_surface_height_m": self.ground_surface_height_m,
                "depth_of_cut_m": depth,
                "wheel_lowest_point_x_m": self.case.single_pass.wheel_lowest_point_x_m,
                "wheel_center_coordinates_m": list(self.wheel_center_coordinates_m),
                "motion_coordinate_position_m": self.motion_position_m,
                "exact_arc_projected_length_m": self.exact_arc_projected_length_m,
                "shallow_cut_approximation_length_m": self.shallow_cut_approximation_length_m,
                "length_absolute_difference_m": self.length_absolute_difference_m,
                "length_relative_difference": self.length_relative_difference,
                "length_relative_difference_reference": "exact_arc_projected_length_m",
                "theoretical_arc_interval_m": list(self.theoretical_arc_interval_m),
                "effective_arc_interval_m": (
                    None
                    if self.effective_arc_interval_m is None
                    else list(self.effective_arc_interval_m)
                ),
                "effective_contact_length_m": self.effective_contact_length_m,
                "contact_ratio": self.contact_ratio,
            },
            "pass_state": {"state": self.pass_state, **state_flags},
            "surface_segments": {
                "segments": [segment.to_dict() for segment in self.surface_segments],
                "ground_length_m": self.region_length_m("ground"),
                "contact_arc_length_m": self.region_length_m("contact_arc"),
                "unprocessed_length_m": self.region_length_m("unprocessed"),
                "partition_total_m": partition_total,
                "partition_residual_m": self.case.workpiece.length_m - partition_total,
            },
            "sampling": {
                "requested_base_point_count": self.case.sampling.base_point_count,
                "final_actual_point_count": len(self.sample_points),
                "point_indexing": "0-based contiguous",
                "boundary_point_region_convention": "A point on a non-empty contact-arc boundary is labelled contact_arc so each x coordinate appears once.",
                "geometry_boundaries_inserted_exactly": True,
            },
            "assumptions_and_cautions": [
                "This is a two-dimensional ideal mean surface profile for one prescribed cutting pass.",
                "depth_of_cut_m is a user-prescribed nominal material-removal depth, not a predicted value.",
                "The wheel is an ideal rigid circle with no runout and no wear.",
                "The profile contains no abrasive-grain randomness and predicts no Ra, Rz, waviness, scratches, chatter marks, or grinding burn.",
                "Elastic recovery, plastic pile-up, burrs, and edge chipping are not represented.",
                "Grinding force, contact pressure, FEM stress, and temperature are not calculated.",
                "The PNG vertical scale is enlarged only to show micrometre-scale height changes.",
                "The strict length sqrt(Ds * ae - ae**2) defines the phase 4C1 circular profile; sqrt(Ds * ae) is retained only as the phase 4A shallow-cut reference.",
                "Phase 4C2 can consume the effective contact length and ratio for an independent empirical load snapshot.",
                "A later phase may generate a Gmsh finite-element mesh from an evolved top surface.",
            ],
            "artifacts": artifacts,
        }


def _canonical_segments(
    motion_position_m: float,
    workpiece_length_m: float,
    arc_length_m: float,
) -> list[tuple[str, float, float]]:
    ground_end = _clamp(motion_position_m, 0.0, workpiece_length_m)
    arc_start = _clamp(motion_position_m, 0.0, workpiece_length_m)
    arc_end = _clamp(
        motion_position_m + arc_length_m, 0.0, workpiece_length_m
    )
    unprocessed_start = arc_end
    candidates = (
        ("ground", 0.0, ground_end),
        ("contact_arc", arc_start, arc_end),
        ("unprocessed", unprocessed_start, workpiece_length_m),
    )
    return [(region, start, end) for region, start, end in candidates if end > start]


def _circular_sagitta(radius_m: float, offset_m: float) -> float:
    """Evaluate R - sqrt(R^2 - offset^2) without forming R squared."""

    offset = _clamp(offset_m, 0.0, radius_m)
    root = math.sqrt(max(radius_m - offset, 0.0)) * math.sqrt(radius_m + offset)
    denominator = radius_m + root
    if not math.isfinite(root) or not math.isfinite(denominator) or denominator <= 0.0:
        _trajectory_error("surface_profile.surface_y_m", "m", "circle evaluation failed")
    return (offset / denominator) * offset


def _surface_description(region: str) -> str:
    descriptions = {
        "ground": "Processed flat surface at H - ae.",
        "contact_arc": "Strict lower circular-wheel profile between H - ae and H.",
        "unprocessed": "Original flat surface at H awaiting this pass.",
    }
    return descriptions[region]


def _sample_coordinates(
    width_m: float,
    base_point_count: int,
    important_coordinates_m: tuple[float, ...],
) -> tuple[float, ...]:
    base = [width_m * index / (base_point_count - 1) for index in range(base_point_count)]
    tagged = [(value, False) for value in base]
    tagged.extend(
        (value, True)
        for value in important_coordinates_m
        if 0.0 <= value <= width_m and math.isfinite(value)
    )
    tagged.extend(((0.0, True), (width_m, True)))
    tagged.sort(key=lambda item: item[0])

    tolerance = max(16.0 * math.ulp(width_m), 1.0e-15 * max(1.0, width_m))
    clusters: list[list[tuple[float, bool]]] = []
    for item in tagged:
        if not clusters or item[0] - clusters[-1][-1][0] > tolerance:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    coordinates: list[float] = []
    for cluster in clusters:
        important = [value for value, is_important in cluster if is_important]
        coordinate = important[0] if important else cluster[0][0]
        if coordinate == 0.0:
            coordinate = 0.0
        elif coordinate == width_m:
            coordinate = width_m
        coordinates.append(coordinate)
    if any(not first < second for first, second in zip(coordinates, coordinates[1:])):
        _trajectory_error(
            "sampling.final_coordinates", "m", "could not be made strictly increasing"
        )
    return tuple(coordinates)


def build_surface_profile(case: TrajectoryCase) -> TrajectoryResult:
    """Calculate a strict circular profile without meshing, loads, or FEM."""

    if not isinstance(case, TrajectoryCase):
        _trajectory_error(
            "case", "TrajectoryCase", "build_surface_profile requires a validated case"
        )
    width = case.workpiece.length_m
    height = case.workpiece.original_surface_height_m
    diameter = case.wheel.diameter_m
    depth = case.single_pass.depth_of_cut_m
    direction = case.single_pass.relative_feed_direction
    wheel_bottom_x = case.single_pass.wheel_lowest_point_x_m

    radius = _finite(diameter / 2.0, "wheel_radius_m", "m")
    ground_height = _finite(height - depth, "ground_surface_height_m", "m")
    exact_length = _finite(
        math.sqrt(depth) * math.sqrt(diameter - depth),
        "exact_arc_projected_length_m",
        "m",
    )
    approximate_length = _finite(
        math.sqrt(depth) * math.sqrt(diameter),
        "shallow_cut_approximation_length_m",
        "m",
    )
    absolute_difference = _finite(
        abs(approximate_length - exact_length),
        "length_absolute_difference_m",
        "m",
    )
    relative_difference = _finite(
        absolute_difference / exact_length,
        "length_relative_difference",
        "dimensionless",
    )
    motion_position = _finite(
        wheel_bottom_x if direction == "positive_x" else width - wheel_bottom_x,
        "motion_position_m",
        "m",
    )
    motion_position = _snap_motion_position(motion_position, width, exact_length)
    state = pass_state_from_motion_coordinate(motion_position, width, exact_length)

    # 中文导读：理论圆弧区间与工件范围取交集，得到当前有效投影长度 Leff。
    canonical_theoretical = (motion_position, motion_position + exact_length)
    theoretical_interval = _from_motion_interval(
        *canonical_theoretical, width, direction
    )
    effective_start = max(0.0, canonical_theoretical[0])
    effective_end = min(width, canonical_theoretical[1])
    if effective_end > effective_start:
        effective_interval = _from_motion_interval(
            effective_start, effective_end, width, direction
        )
        effective_length = _finite(
            effective_end - effective_start, "effective_contact_length_m", "m"
        )
    else:
        effective_interval = None
        effective_length = 0.0
    contact_ratio = _clamp(effective_length / exact_length, 0.0, 1.0)

    segment_values: list[SurfaceSegment] = []
    for region, start, end in _canonical_segments(
        motion_position, width, exact_length
    ):
        x_start, x_end = _from_motion_interval(start, end, width, direction)
        segment_values.append(
            SurfaceSegment(
                region=region,
                x_start_m=x_start,
                x_end_m=x_end,
                surface_description=_surface_description(region),
            )
        )
    segments = tuple(sorted(segment_values, key=lambda item: item.x_start_m))

    important_coordinates: list[float] = [
        theoretical_interval[0],
        theoretical_interval[1],
    ]
    if effective_interval is not None:
        important_coordinates.extend(effective_interval)
    for segment in segments:
        important_coordinates.extend((segment.x_start_m, segment.x_end_m))
    coordinates = _sample_coordinates(
        width,
        case.sampling.base_point_count,
        tuple(important_coordinates),
    )

    points: list[SurfaceProfilePoint] = []
    for x_value in coordinates:
        motion_x = motion_coordinate_from_physical_x(x_value, width, direction)
        on_arc = (
            effective_interval is not None
            and effective_start <= motion_x <= effective_end
        )
        if on_arc:
            # 中文导读：contact_arc 采样点高度直接由圆方程计算，不用直线插值。
            delta = _clamp(motion_x - motion_position, 0.0, exact_length)
            surface_y = ground_height + _circular_sagitta(radius, delta)
            surface_y = _clamp(surface_y, ground_height, height)
            region = "contact_arc"
        elif motion_x < motion_position or state == "after_exit":
            surface_y = ground_height
            region = "ground"
        else:
            surface_y = height
            region = "unprocessed"
        points.append(
            SurfaceProfilePoint(
                x_m=x_value,
                surface_y_m=surface_y,
                original_surface_height_m=height,
                region=region,
            )
        )

    return TrajectoryResult(
        case=case,
        wheel_radius_m=radius,
        ground_surface_height_m=ground_height,
        exact_arc_projected_length_m=exact_length,
        shallow_cut_approximation_length_m=approximate_length,
        length_absolute_difference_m=absolute_difference,
        length_relative_difference=relative_difference,
        motion_position_m=motion_position,
        pass_state=state,
        theoretical_arc_interval_m=theoretical_interval,
        effective_arc_interval_m=effective_interval,
        effective_contact_length_m=effective_length,
        contact_ratio=contact_ratio,
        surface_segments=segments,
        sample_points=tuple(points),
    )
