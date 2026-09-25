"""Rigid analytical circular-grain geometry and ordered contact boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


class ContactGeometryError(ValueError):
    """Raised when rigid-grain contact geometry is invalid or ambiguous."""


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ContactGeometryError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContactGeometryError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContactGeometryError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class RigidCircularGrain:
    radius_m: float
    center_x_m: float
    center_y_m: float

    def __post_init__(self) -> None:
        radius = _finite_number(self.radius_m, "radius_m")
        if radius <= 0.0:
            raise ContactGeometryError("radius_m must be strictly positive")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(
            self, "center_x_m", _finite_number(self.center_x_m, "center_x_m")
        )
        object.__setattr__(
            self, "center_y_m", _finite_number(self.center_y_m, "center_y_m")
        )

    @property
    def center_m(self) -> NDArray[np.float64]:
        return np.array([self.center_x_m, self.center_y_m], dtype=float)


@dataclass(frozen=True, slots=True)
class RigidCircularArcGrain:
    radius_m: float
    center_x_m: float
    center_y_m: float
    start_angle_rad: float
    end_angle_rad: float

    def __post_init__(self) -> None:
        radius = _finite_number(self.radius_m, "radius_m")
        start = _finite_number(self.start_angle_rad, "start_angle_rad")
        end = _finite_number(self.end_angle_rad, "end_angle_rad")
        if radius <= 0.0:
            raise ContactGeometryError("radius_m must be strictly positive")
        if start < 0.0 or end <= start or end - start >= 2.0 * math.pi:
            raise ContactGeometryError(
                "circular arc span must be positive, counterclockwise, and below 2*pi"
            )
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "center_x_m", _finite_number(self.center_x_m, "center_x_m"))
        object.__setattr__(self, "center_y_m", _finite_number(self.center_y_m, "center_y_m"))
        object.__setattr__(self, "start_angle_rad", start)
        object.__setattr__(self, "end_angle_rad", end)

    @property
    def center_m(self) -> NDArray[np.float64]:
        return np.array([self.center_x_m, self.center_y_m], dtype=float)

    def point_at(self, angle_rad: float) -> NDArray[np.float64]:
        return self.center_m + self.radius_m * np.array(
            [math.cos(angle_rad), math.sin(angle_rad)], dtype=float
        )


RigidAnalyticalGrain = RigidCircularGrain | RigidCircularArcGrain


@dataclass(frozen=True, slots=True)
class WedgeEffectiveAngles:
    rake_angle_rad: float
    clearance_angle_rad: float
    leading_feature_type: str
    trailing_feature_type: str


@dataclass(frozen=True, slots=True)
class RigidContactQuery:
    signed_gap_m: float
    penetration_m: float
    distance_m: float
    outward_normal: NDArray[np.float64]
    consistent_tangent: NDArray[np.float64]
    normal_derivative_per_m: NDArray[np.float64]
    closest_point_m: NDArray[np.float64]
    feature_id: int
    feature_type: str
    closest_point_unique: bool
    at_feature_junction: bool
    contact_applicable: bool
    diagnostic_codes: tuple[str, ...] = ()

    @property
    def gap_m(self) -> float:
        return self.signed_gap_m

    @property
    def normal(self) -> NDArray[np.float64]:
        return self.outward_normal

    @property
    def tangent(self) -> NDArray[np.float64]:
        return self.consistent_tangent


def _rotate(angle_rad: float) -> NDArray[np.float64]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


@dataclass(frozen=True, slots=True)
class RigidRoundedWedgeGrain:
    tip_radius_m: float
    wedge_height_m: float
    intrinsic_rake_angle_rad: float
    intrinsic_clearance_angle_rad: float
    pose_angle_rad: float
    reference_x_m: float
    reference_y_m: float

    def __post_init__(self) -> None:
        radius = _finite_number(self.tip_radius_m, "tip_radius_m")
        height = _finite_number(self.wedge_height_m, "wedge_height_m")
        rake = _finite_number(self.intrinsic_rake_angle_rad, "intrinsic_rake_angle_rad")
        clearance = _finite_number(self.intrinsic_clearance_angle_rad, "intrinsic_clearance_angle_rad")
        if radius <= 0.0:
            raise ContactGeometryError("tip_radius_m must be strictly positive")
        if height <= radius:
            raise ContactGeometryError("wedge_height_m must be greater than tip_radius_m")
        if not -0.5 * math.pi < rake < 0.5 * math.pi:
            raise ContactGeometryError("intrinsic_rake_angle_rad is outside its legal range")
        if not 0.0 < clearance < 0.5 * math.pi:
            raise ContactGeometryError("intrinsic_clearance_angle_rad is outside its legal range")
        object.__setattr__(self, "tip_radius_m", radius)
        object.__setattr__(self, "wedge_height_m", height)
        object.__setattr__(self, "intrinsic_rake_angle_rad", rake)
        object.__setattr__(self, "intrinsic_clearance_angle_rad", clearance)
        object.__setattr__(self, "pose_angle_rad", _finite_number(self.pose_angle_rad, "pose_angle_rad"))
        object.__setattr__(self, "reference_x_m", _finite_number(self.reference_x_m, "reference_x_m"))
        object.__setattr__(self, "reference_y_m", _finite_number(self.reference_y_m, "reference_y_m"))
        if self.derived_wedge_width_m <= self.geometry_tolerance_m:
            raise ContactGeometryError("rounded wedge has a degenerate derived width")
        if self.boundary_area_m2 <= self.geometry_tolerance_m**2:
            raise ContactGeometryError("rounded wedge has a degenerate boundary area")

    @property
    def reference_m(self) -> NDArray[np.float64]:
        return np.array([self.reference_x_m, self.reference_y_m], dtype=float)

    @property
    def tip_center_local_m(self) -> NDArray[np.float64]:
        return np.array([0.0, self.tip_radius_m], dtype=float)

    @property
    def geometry_tolerance_m(self) -> float:
        scale = max(self.tip_radius_m, self.wedge_height_m, 1.0e-300)
        return max(64.0 * math.ulp(scale), 1.0e-12 * scale)

    @property
    def rake_direction_local(self) -> NDArray[np.float64]:
        gamma = self.intrinsic_rake_angle_rad
        return np.array([math.sin(gamma), math.cos(gamma)], dtype=float)

    @property
    def clearance_direction_local(self) -> NDArray[np.float64]:
        alpha = self.intrinsic_clearance_angle_rad
        return np.array([-math.cos(alpha), math.sin(alpha)], dtype=float)

    @property
    def rake_tangent_point_local_m(self) -> NDArray[np.float64]:
        direction = self.rake_direction_local
        normal = np.array([direction[1], -direction[0]], dtype=float)
        return self.tip_center_local_m + self.tip_radius_m * normal

    @property
    def clearance_tangent_point_local_m(self) -> NDArray[np.float64]:
        direction = self.clearance_direction_local
        normal = np.array([-direction[1], direction[0]], dtype=float)
        return self.tip_center_local_m + self.tip_radius_m * normal

    def _top_point(self, tangent: NDArray[np.float64], direction: NDArray[np.float64]) -> NDArray[np.float64]:
        distance = (self.wedge_height_m - tangent[1]) / direction[1]
        if distance <= self.geometry_tolerance_m:
            raise ContactGeometryError("rounded wedge has a degenerate face")
        return tangent + distance * direction

    @property
    def rake_top_point_local_m(self) -> NDArray[np.float64]:
        return self._top_point(self.rake_tangent_point_local_m, self.rake_direction_local)

    @property
    def clearance_top_point_local_m(self) -> NDArray[np.float64]:
        return self._top_point(self.clearance_tangent_point_local_m, self.clearance_direction_local)

    @property
    def derived_wedge_width_m(self) -> float:
        return float(self.rake_top_point_local_m[0] - self.clearance_top_point_local_m[0])

    @property
    def feature_types(self) -> tuple[str, ...]:
        return ("rake_face", "tip_arc", "clearance_face", "remote_closure")

    @property
    def boundary_perimeter_m(self) -> float:
        rake_length = float(
            np.linalg.norm(self.rake_top_point_local_m - self.rake_tangent_point_local_m)
        )
        clearance_length = float(
            np.linalg.norm(
                self.clearance_top_point_local_m - self.clearance_tangent_point_local_m
            )
        )
        closure_length = float(
            np.linalg.norm(self.rake_top_point_local_m - self.clearance_top_point_local_m)
        )
        center = self.tip_center_local_m
        rake_angle = math.atan2(
            self.rake_tangent_point_local_m[1] - center[1],
            self.rake_tangent_point_local_m[0] - center[0],
        )
        clearance_angle = math.atan2(
            self.clearance_tangent_point_local_m[1] - center[1],
            self.clearance_tangent_point_local_m[0] - center[0],
        )
        arc_angle = (rake_angle - clearance_angle) % (2.0 * math.pi)
        return rake_length + clearance_length + closure_length + self.tip_radius_m * arc_angle

    @property
    def boundary_area_m2(self) -> float:
        points = np.vstack((
            self.rake_top_point_local_m,
            self.rake_tangent_point_local_m,
            np.array([0.0, 0.0]),
            self.clearance_tangent_point_local_m,
            self.clearance_top_point_local_m,
        ))
        x = points[:, 0]
        y = points[:, 1]
        polygon_area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return polygon_area + 0.5 * math.pi * self.tip_radius_m**2

    def local_to_global(self, point_local_m: object) -> NDArray[np.float64]:
        point = np.asarray(point_local_m, dtype=float)
        return self.reference_m + _rotate(self.pose_angle_rad) @ point

    def global_to_local(self, point_global_m: object) -> NDArray[np.float64]:
        point = np.asarray(point_global_m, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ContactGeometryError("contact point must contain two finite coordinates")
        return _rotate(-self.pose_angle_rad) @ (point - self.reference_m)


RigidAnalyticalGrain = RigidCircularGrain | RigidCircularArcGrain | RigidRoundedWedgeGrain


@dataclass(frozen=True, slots=True)
class CircularContactKinematics:
    gap_m: float
    penetration_m: float
    distance_m: float
    normal: NDArray[np.float64]
    tangent: NDArray[np.float64]
    closest_point_m: NDArray[np.float64]


def circular_contact_kinematics(
    grain: RigidCircularGrain, point_m: object
) -> CircularContactKinematics:
    """Return signed gap and the unique radial closest point on a rigid circle."""

    if not isinstance(grain, RigidCircularGrain):
        raise TypeError("grain must be a RigidCircularGrain")
    point = np.asarray(point_m, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ContactGeometryError("contact point must contain two finite coordinates")
    radial = point - grain.center_m
    distance = float(np.linalg.norm(radial))
    scale = max(grain.radius_m, abs(grain.center_x_m), abs(grain.center_y_m), 1.0)
    if distance <= 64.0 * math.ulp(scale):
        raise ContactGeometryError("closest point is not unique at the circle center")
    normal = radial / distance
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    gap = distance - grain.radius_m
    closest = grain.center_m + grain.radius_m * normal
    return CircularContactKinematics(
        gap_m=gap,
        penetration_m=max(0.0, -gap),
        distance_m=distance,
        normal=normal,
        tangent=tangent,
        closest_point_m=closest,
    )


def _normalized_angle(angle_rad: float, reference_rad: float) -> float:
    return reference_rad + ((angle_rad - reference_rad) % (2.0 * math.pi))


def circular_arc_contact_kinematics(
    grain: RigidCircularArcGrain, point_m: object
) -> CircularContactKinematics:
    """Return the unique closest-point kinematics for one analytic circular arc."""

    if not isinstance(grain, RigidCircularArcGrain):
        raise TypeError("grain must be a RigidCircularArcGrain")
    point = np.asarray(point_m, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ContactGeometryError("contact point must contain two finite coordinates")
    radial = point - grain.center_m
    radial_distance = float(np.linalg.norm(radial))
    scale = max(grain.radius_m, abs(grain.center_x_m), abs(grain.center_y_m), 1.0)
    if radial_distance <= 64.0 * math.ulp(scale):
        raise ContactGeometryError("closest point is not unique at the circular arc center")
    angle = _normalized_angle(math.atan2(radial[1], radial[0]), grain.start_angle_rad)
    tolerance = 64.0 * math.ulp(max(abs(grain.start_angle_rad), abs(grain.end_angle_rad), 1.0))
    if angle <= grain.end_angle_rad + tolerance:
        normal = radial / radial_distance
        closest = grain.center_m + grain.radius_m * normal
        distance = radial_distance
        gap = distance - grain.radius_m
    else:
        start = grain.point_at(grain.start_angle_rad)
        end = grain.point_at(grain.end_angle_rad)
        start_distance = float(np.linalg.norm(point - start))
        end_distance = float(np.linalg.norm(point - end))
        ambiguity = 64.0 * math.ulp(max(start_distance, end_distance, grain.radius_m, 1.0))
        if abs(start_distance - end_distance) <= ambiguity:
            raise ContactGeometryError("closest point is not unique between circular arc endpoints")
        if start_distance < end_distance:
            closest = start
            distance = start_distance
        else:
            closest = end
            distance = end_distance
        if distance <= ambiguity:
            raise ContactGeometryError("closest-point normal is not unique at a circular arc endpoint")
        normal = (point - closest) / distance
        gap = distance
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    return CircularContactKinematics(
        gap_m=gap,
        penetration_m=max(0.0, -gap),
        distance_m=distance,
        normal=normal,
        tangent=tangent,
        closest_point_m=closest,
    )


def rigid_contact_kinematics(
    grain: RigidAnalyticalGrain, point_m: object
) -> CircularContactKinematics:
    if isinstance(grain, RigidCircularGrain):
        return circular_contact_kinematics(grain, point_m)
    if isinstance(grain, RigidCircularArcGrain):
        return circular_arc_contact_kinematics(grain, point_m)
    raise TypeError("grain must be a supported rigid analytical grain")


def _segment_closest(
    point: NDArray[np.float64], start: NDArray[np.float64], end: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, bool]:
    vector = end - start
    length_squared = float(np.dot(vector, vector))
    if length_squared <= 0.0:
        raise ContactGeometryError("rigid boundary contains a degenerate segment")
    raw = float(np.dot(point - start, vector) / length_squared)
    parameter = min(1.0, max(0.0, raw))
    closest = start + parameter * vector
    return closest, float(np.linalg.norm(point - closest)), parameter in (0.0, 1.0)


def _wedge_query(grain: RigidRoundedWedgeGrain, point_m: object) -> RigidContactQuery:
    point_local = grain.global_to_local(point_m)
    center = grain.tip_center_local_m
    radius = grain.tip_radius_m
    candidates: list[tuple[int, str, NDArray[np.float64], float, bool, bool]] = []
    for feature_id, feature_type, start, end, applicable in (
        (0, "rake_face", grain.rake_top_point_local_m, grain.rake_tangent_point_local_m, True),
        (2, "clearance_face", grain.clearance_tangent_point_local_m, grain.clearance_top_point_local_m, True),
        (3, "remote_closure", grain.clearance_top_point_local_m, grain.rake_top_point_local_m, False),
    ):
        closest, distance, endpoint = _segment_closest(point_local, start, end)
        candidates.append((feature_id, feature_type, closest, distance, endpoint, applicable))
    radial = point_local - center
    angle = math.atan2(float(radial[1]), float(radial[0]))
    start_angle = math.atan2(
        grain.rake_tangent_point_local_m[1] - center[1],
        grain.rake_tangent_point_local_m[0] - center[0],
    )
    end_angle = math.atan2(
        grain.clearance_tangent_point_local_m[1] - center[1],
        grain.clearance_tangent_point_local_m[0] - center[0],
    )
    clockwise_span = (start_angle - end_angle) % (2.0 * math.pi)
    clockwise_position = (start_angle - angle) % (2.0 * math.pi)
    if float(np.linalg.norm(radial)) > grain.geometry_tolerance_m and clockwise_position <= clockwise_span + 1.0e-14:
        arc_closest = center + radius * radial / np.linalg.norm(radial)
        arc_endpoint = clockwise_position <= 1.0e-14 or abs(clockwise_position - clockwise_span) <= 1.0e-14
    else:
        endpoints = (grain.rake_tangent_point_local_m, grain.clearance_tangent_point_local_m)
        distances = [float(np.linalg.norm(point_local - item)) for item in endpoints]
        arc_closest = endpoints[int(distances[1] < distances[0])]
        arc_endpoint = True
    candidates.append((1, "tip_arc", arc_closest, float(np.linalg.norm(point_local - arc_closest)), arc_endpoint, True))
    candidates.sort(key=lambda item: (item[3], item[0]))
    chosen = candidates[0]
    ambiguity = max(grain.geometry_tolerance_m, 1.0e-12 * max(chosen[3], radius))
    tied = [item for item in candidates if abs(item[3] - chosen[3]) <= ambiguity]
    if len(tied) > 1:
        compatible = {item[1] for item in tied} in (
            {"rake_face", "tip_arc"}, {"clearance_face", "tip_arc"}
        )
        if not compatible:
            representative = chosen
            if representative[1] == "rake_face":
                outward_local = np.array(
                    [grain.rake_direction_local[1], -grain.rake_direction_local[0]]
                )
            elif representative[1] == "clearance_face":
                outward_local = np.array(
                    [-grain.clearance_direction_local[1], grain.clearance_direction_local[0]]
                )
            elif representative[1] == "remote_closure":
                outward_local = np.array([0.0, 1.0])
            else:
                outward_local = (representative[2] - center) / radius
            representative_gap = float(
                np.dot(point_local - representative[2], outward_local)
            )
            rotation = _rotate(grain.pose_angle_rad)
            outward = rotation @ outward_local
            tangent = np.array([-outward[1], outward[0]], dtype=float)
            return RigidContactQuery(
                signed_gap_m=representative_gap,
                penetration_m=max(0.0, -representative_gap),
                distance_m=representative[3],
                outward_normal=outward,
                consistent_tangent=tangent,
                normal_derivative_per_m=np.zeros((2, 2), dtype=float),
                closest_point_m=grain.local_to_global(representative[2]),
                feature_id=representative[0], feature_type=representative[1],
                closest_point_unique=False, at_feature_junction=True,
                contact_applicable=False, diagnostic_codes=("closest_point_not_unique",),
            )
    closest_local = chosen[2]
    vector = point_local - closest_local
    if chosen[1] == "rake_face":
        outward_local = np.array([grain.rake_direction_local[1], -grain.rake_direction_local[0]])
    elif chosen[1] == "clearance_face":
        outward_local = np.array([-grain.clearance_direction_local[1], grain.clearance_direction_local[0]])
    elif chosen[1] == "remote_closure":
        outward_local = np.array([0.0, 1.0])
    else:
        outward_local = (closest_local - center) / radius
    signed_gap = float(np.dot(vector, outward_local))
    rotation = _rotate(grain.pose_angle_rad)
    outward = rotation @ outward_local
    tangent = np.array([-outward[1], outward[0]], dtype=float)
    radial_distance = float(np.linalg.norm(point_local - center))
    if chosen[1] == "tip_arc" and radial_distance > grain.geometry_tolerance_m:
        normal_derivative_local = (
            np.eye(2) - np.outer(outward_local, outward_local)
        ) / radial_distance
        normal_derivative = rotation @ normal_derivative_local @ rotation.T
    else:
        normal_derivative = np.zeros((2, 2), dtype=float)
    applicable = bool(chosen[5])
    return RigidContactQuery(
        signed_gap_m=signed_gap,
        penetration_m=max(0.0, -signed_gap),
        distance_m=chosen[3],
        outward_normal=outward,
        consistent_tangent=tangent,
        normal_derivative_per_m=normal_derivative,
        closest_point_m=grain.local_to_global(closest_local),
        feature_id=chosen[0],
        feature_type=chosen[1],
        closest_point_unique=True,
        at_feature_junction=len(tied) > 1 or chosen[4],
        contact_applicable=applicable,
        diagnostic_codes=() if applicable else ("remote_closure_not_contact_eligible",),
    )


def rigid_contact_query(grain: RigidAnalyticalGrain, point_m: object) -> RigidContactQuery:
    if isinstance(grain, RigidRoundedWedgeGrain):
        return _wedge_query(grain, point_m)
    result = rigid_contact_kinematics(grain, point_m)
    return RigidContactQuery(
        signed_gap_m=result.gap_m,
        penetration_m=result.penetration_m,
        distance_m=result.distance_m,
        outward_normal=result.normal,
        consistent_tangent=result.tangent,
        normal_derivative_per_m=(
            np.eye(2) - np.outer(result.normal, result.normal)
        ) / result.distance_m,
        closest_point_m=result.closest_point_m,
        feature_id=0,
        feature_type="circle" if isinstance(grain, RigidCircularGrain) else "circular_arc",
        closest_point_unique=True,
        at_feature_junction=False,
        contact_applicable=True,
    )


def effective_wedge_angles(
    grain: RigidRoundedWedgeGrain, motion_direction: object
) -> WedgeEffectiveAngles:
    if not isinstance(grain, RigidRoundedWedgeGrain):
        raise TypeError("grain must be a RigidRoundedWedgeGrain")
    motion = np.asarray(motion_direction, dtype=float)
    if motion.shape != (2,) or not np.all(np.isfinite(motion)):
        raise ContactGeometryError("motion_direction must contain two finite coordinates")
    norm = float(np.linalg.norm(motion))
    if norm <= grain.geometry_tolerance_m:
        raise ContactGeometryError("motion_direction must be nonzero")
    m = motion / norm
    q = np.array([-m[1], m[0]], dtype=float)
    rotation = _rotate(grain.pose_angle_rad)
    tangents = {
        "rake_face": rotation @ grain.rake_direction_local,
        "clearance_face": rotation @ grain.clearance_direction_local,
    }
    for name, tangent in tuple(tangents.items()):
        if float(np.dot(tangent, q)) < 0.0:
            tangents[name] = -tangent
    normals = {
        "rake_face": rotation @ np.array([grain.rake_direction_local[1], -grain.rake_direction_local[0]]),
        "clearance_face": rotation @ np.array([-grain.clearance_direction_local[1], grain.clearance_direction_local[0]]),
    }
    ordered = sorted(normals, key=lambda name: float(np.dot(normals[name], m)), reverse=True)
    leading, trailing = ordered
    if abs(float(np.dot(normals[leading] - normals[trailing], m))) <= 1.0e-12:
        raise ContactGeometryError("leading and trailing wedge faces are not unique")
    leading_tangent = tangents[leading]
    trailing_tangent = tangents[trailing]
    return WedgeEffectiveAngles(
        rake_angle_rad=math.atan2(float(np.dot(leading_tangent, m)), float(np.dot(leading_tangent, q))),
        clearance_angle_rad=math.atan2(float(np.dot(trailing_tangent, q)), float(-np.dot(trailing_tangent, m))),
        leading_feature_type=leading,
        trailing_feature_type=trailing,
    )


def ordered_boundary_tributary_lengths(points_m: object) -> NDArray[np.float64]:
    """Return nodal line-integration weights for an x-ordered open polyline."""

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ContactGeometryError("candidate boundary must contain at least two 2D nodes")
    if not np.all(np.isfinite(points)):
        raise ContactGeometryError("candidate boundary coordinates must be finite")
    if not np.all(np.diff(points[:, 0]) > 0.0):
        raise ContactGeometryError("candidate boundary x coordinates must be strictly increasing")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.all(np.isfinite(segment_lengths)) or np.any(segment_lengths <= 0.0):
        raise ContactGeometryError("candidate boundary segments must be finite and positive")
    weights = np.zeros(points.shape[0], dtype=float)
    weights[:-1] += 0.5 * segment_lengths
    weights[1:] += 0.5 * segment_lengths
    return weights
