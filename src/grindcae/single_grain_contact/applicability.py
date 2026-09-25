"""Small-strain applicability diagnostics and fixed-mesh safety gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .solver import PreparedContactMesh


class ApplicabilityError(RuntimeError):
    """Raised when a converged trial violates a hard fixed-mesh safety gate."""


@dataclass(frozen=True, slots=True)
class ApplicabilityDiagnostics:
    status: str
    maximum_displacement_m: float
    maximum_displacement_to_contact_size: float
    maximum_displacement_to_grain_radius: float
    maximum_displacement_gradient: float
    maximum_absolute_principal_strain: float
    maximum_equivalent_total_strain: float
    maximum_local_rotation_rad: float
    minimum_area_ratio: float
    minimum_normalized_jacobian_ratio: float
    surface_self_intersection: bool
    closest_point_unique: bool
    candidate_boundary_order_preserved: bool
    warning_metrics: tuple[str, ...]
    hard_stop_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _segments_intersect(a, b, c, d, tolerance: float) -> bool:
    def cross(p, q, r) -> float:
        left = q - p
        right = r - p
        return float(left[0] * right[1] - left[1] * right[0])

    values = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
    return values[0] * values[1] < -tolerance and values[2] * values[3] < -tolerance


def _surface_self_intersects(points: NDArray[np.float64]) -> bool:
    scale = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    tolerance = 128.0 * math.ulp(scale) * scale
    for first in range(points.shape[0] - 1):
        for second in range(first + 2, points.shape[0] - 1):
            if _segments_intersect(
                points[first], points[first + 1],
                points[second], points[second + 1], tolerance,
            ):
                return True
    return False


def evaluate_contact_applicability(
    prepared: "PreparedContactMesh",
    displacement_vector_m: object,
    *,
    grain_radius_m: float,
    grain_center_m: tuple[float, float] | None = None,
    grain_geometry: object | None = None,
    warning_displacement_over_contact_size: float = 0.10,
    stop_displacement_over_contact_size: float = 0.50,
    warning_displacement_over_radius: float = 0.05,
    stop_displacement_over_radius: float = 0.20,
    warning_strain: float = 0.05,
    stop_strain: float = 0.15,
    raise_on_hard_stop: bool = False,
) -> ApplicabilityDiagnostics:
    """Evaluate the task-book warning and hard-stop metrics on one trial state."""

    displacement = np.asarray(displacement_vector_m, dtype=float)
    if displacement.shape != (prepared.total_degrees_of_freedom,) or not np.all(
        np.isfinite(displacement)
    ):
        raise ApplicabilityError("contact displacement contains non-finite data")
    radius = float(grain_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("grain_radius_m must be finite and positive")

    structural = prepared.structural
    reference = np.asarray(structural.imported_mesh.mesh.p.T, dtype=float)
    nodal = np.column_stack(
        (
            displacement[structural.component_dofs[0]],
            displacement[structural.component_dofs[1]],
        )
    )
    current = reference + nodal
    maximum_displacement = float(np.max(np.linalg.norm(nodal, axis=1)))
    contact_size = float(np.median(prepared.candidate_tributary_lengths_m))

    maximum_gradient = 0.0
    maximum_principal = 0.0
    maximum_equivalent = 0.0
    maximum_rotation = 0.0
    area_ratios: list[float] = []
    for nodes in structural.element_connectivity:
        x0 = reference[nodes]
        x1 = current[nodes]
        reference_edges = np.column_stack((x0[1] - x0[0], x0[2] - x0[0]))
        current_edges = np.column_stack((x1[1] - x1[0], x1[2] - x1[0]))
        gradient = current_edges @ np.linalg.inv(reference_edges) - np.eye(2)
        strain = 0.5 * (gradient + gradient.T)
        rotation = 0.5 * (gradient[1, 0] - gradient[0, 1])
        principal = np.linalg.eigvalsh(strain)
        equivalent = math.sqrt(
            max(
                0.0,
                2.0 / 3.0
                * (
                    strain[0, 0] ** 2
                    + strain[1, 1] ** 2
                    - strain[0, 0] * strain[1, 1]
                    + 3.0 * strain[0, 1] ** 2
                ),
            )
        )
        maximum_gradient = max(maximum_gradient, float(np.linalg.norm(gradient)))
        maximum_principal = max(maximum_principal, float(np.max(np.abs(principal))))
        maximum_equivalent = max(maximum_equivalent, equivalent)
        maximum_rotation = max(maximum_rotation, abs(float(rotation)))
        area_ratios.append(float(np.linalg.det(np.eye(2) + gradient)))

    surface_dofs = prepared.candidate_component_dofs
    surface = prepared.candidate_reference_coordinates_m + displacement[surface_dofs]
    scale = max(float(np.ptp(surface[:, 0])), 1.0)
    order_tolerance = 128.0 * math.ulp(scale)
    reference_surface = prepared.candidate_reference_coordinates_m
    reference_is_strict_x_chain = bool(
        np.all(np.diff(reference_surface[:, 0]) > order_tolerance)
    )
    if reference_is_strict_x_chain:
        order_preserved = bool(np.all(np.diff(surface[:, 0]) > order_tolerance))
    else:
        segment_lengths = np.linalg.norm(np.diff(surface, axis=0), axis=1)
        order_preserved = bool(
            np.all(np.isfinite(segment_lengths))
            and np.all(segment_lengths > order_tolerance)
        )
    self_intersection = _surface_self_intersects(surface)
    closest_unique = True
    if grain_geometry is not None:
        from .geometry import rigid_contact_query
        # The existing solver and v2 reader reject ambiguity at closed/contact
        # points. A remote, strictly open feature tie carries no contact force.
        closest_unique = all(
            query.closest_point_unique or query.signed_gap_m > 0.0
            for query in (rigid_contact_query(grain_geometry, point) for point in surface)
        )
    elif grain_center_m is not None:
        center = np.asarray(grain_center_m, dtype=float)
        distances = np.linalg.norm(surface - center, axis=1)
        closest_unique = bool(np.all(distances > 64.0 * math.ulp(max(radius, 1.0))))

    area_minimum = min(area_ratios)
    warnings: list[str] = []
    hard_stops: list[str] = []
    metrics = {
        "maximum_displacement_to_contact_size": maximum_displacement / contact_size,
        "maximum_displacement_to_grain_radius": maximum_displacement / radius,
        "maximum_displacement_gradient": maximum_gradient,
        "maximum_absolute_principal_strain": maximum_principal,
        "maximum_local_rotation_rad": maximum_rotation,
    }
    warning_limits = {
        "maximum_displacement_to_contact_size": warning_displacement_over_contact_size,
        "maximum_displacement_to_grain_radius": warning_displacement_over_radius,
        "maximum_displacement_gradient": 0.10,
        "maximum_absolute_principal_strain": warning_strain,
        "maximum_local_rotation_rad": 0.05,
    }
    hard_limits = {
        "maximum_displacement_to_contact_size": stop_displacement_over_contact_size,
        "maximum_displacement_to_grain_radius": stop_displacement_over_radius,
        "maximum_displacement_gradient": 0.25,
        "maximum_absolute_principal_strain": stop_strain,
        "maximum_local_rotation_rad": 0.15,
    }
    for name, value in metrics.items():
        if value > hard_limits[name]:
            hard_stops.append(f"{name} exceeds hard-stop threshold")
        elif value > warning_limits[name]:
            warnings.append(name)
    if area_minimum <= 0.0:
        hard_stops.append("non-positive element area or Jacobian")
    elif area_minimum < 0.30:
        hard_stops.append("severe element area or Jacobian degradation")
    elif area_minimum < 0.80:
        warnings.extend(("minimum_area_ratio", "minimum_normalized_jacobian_ratio"))
    if not order_preserved:
        hard_stops.append("candidate boundary order is not preserved")
    if self_intersection:
        hard_stops.append("contact candidate surface self-intersects")
    if not closest_unique:
        hard_stops.append(
            "rigid-grain closest point is not unique" if grain_geometry is not None
            else "rigid-circle closest point is not unique"
        )

    diagnostics = ApplicabilityDiagnostics(
        status="hard_stop" if hard_stops else "warning" if warnings else "within_range",
        maximum_displacement_m=maximum_displacement,
        maximum_displacement_to_contact_size=metrics[
            "maximum_displacement_to_contact_size"
        ],
        maximum_displacement_to_grain_radius=metrics[
            "maximum_displacement_to_grain_radius"
        ],
        maximum_displacement_gradient=maximum_gradient,
        maximum_absolute_principal_strain=maximum_principal,
        maximum_equivalent_total_strain=maximum_equivalent,
        maximum_local_rotation_rad=maximum_rotation,
        minimum_area_ratio=area_minimum,
        minimum_normalized_jacobian_ratio=area_minimum,
        surface_self_intersection=self_intersection,
        closest_point_unique=closest_unique,
        candidate_boundary_order_preserved=order_preserved,
        warning_metrics=tuple(dict.fromkeys(warnings)),
        hard_stop_reasons=tuple(hard_stops),
    )
    if hard_stops and raise_on_hard_stop:
        raise ApplicabilityError("; ".join(hard_stops))
    return diagnostics


__all__ = [
    "ApplicabilityDiagnostics",
    "ApplicabilityError",
    "evaluate_contact_applicability",
]
