"""Recover element-constant P1 strain and plane-stress fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from grindcae.domain.models import SimulationCase
from grindcae.solver import (
    FacetLoadFullFieldSolution,
    FullFieldSolution,
    LocalLoadFullFieldSolution,
    SolverError,
)


class PostprocessError(RuntimeError):
    """Raised when field recovery or result export cannot proceed."""


@dataclass(frozen=True, slots=True)
class RecoveredFields:
    """Nodal displacement fields and element-constant recovered quantities."""

    node_coordinates: NDArray[np.float64]
    nodal_displacements: NDArray[np.float64]
    displacement_magnitude: NDArray[np.float64]
    element_connectivity: NDArray[np.int32]
    element_centroids: NDArray[np.float64]
    element_areas: NDArray[np.float64]
    strain_xx: NDArray[np.float64]
    strain_yy: NDArray[np.float64]
    engineering_shear_strain_xy: NDArray[np.float64]
    derived_strain_zz: NDArray[np.float64]
    stress_xx: NDArray[np.float64]
    stress_yy: NDArray[np.float64]
    shear_stress_xy: NDArray[np.float64]
    von_mises_stress: NDArray[np.float64]
    principal_stress_1_in_plane: NDArray[np.float64]
    principal_stress_2_in_plane: NDArray[np.float64]

    @property
    def node_count(self) -> int:
        return int(self.node_coordinates.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.element_connectivity.shape[0])


@dataclass(frozen=True, slots=True)
class StressStatistics:
    """Area-aware stress statistics for the unsmoothed element fields."""

    von_mises_area_weighted_mean: float
    von_mises_area_weighted_p95: float
    von_mises_area_weighted_p99: float
    von_mises_raw_element_maximum_mesh_dependent: float
    raw_maximum_element_id: int
    raw_maximum_centroid: tuple[float, float]
    principal_stress_1_in_plane_maximum: float
    principal_stress_2_in_plane_minimum: float

    def to_dict(self) -> dict[str, object]:
        return {
            "von_mises_area_weighted_mean_Pa": (
                self.von_mises_area_weighted_mean
            ),
            "von_mises_area_weighted_p95_Pa": self.von_mises_area_weighted_p95,
            "von_mises_area_weighted_p99_Pa": self.von_mises_area_weighted_p99,
            "von_mises_raw_element_maximum_mesh_dependent_Pa": (
                self.von_mises_raw_element_maximum_mesh_dependent
            ),
            "raw_element_maximum_mesh_dependent_element_id": (
                self.raw_maximum_element_id
            ),
            "raw_element_maximum_mesh_dependent_centroid_m": list(
                self.raw_maximum_centroid
            ),
            "principal_stress_1_in_plane_maximum_Pa": (
                self.principal_stress_1_in_plane_maximum
            ),
            "principal_stress_2_in_plane_minimum_Pa": (
                self.principal_stress_2_in_plane_minimum
            ),
        }


def _element_geometry(
    node_coordinates: NDArray[np.float64],
    connectivity: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vertices = node_coordinates[connectivity]
    centroids = np.mean(vertices, axis=1)
    edge_1 = vertices[:, 1] - vertices[:, 0]
    edge_2 = vertices[:, 2] - vertices[:, 0]
    areas = 0.5 * np.abs(
        edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    )
    return centroids, areas


def recover_fields(
    case: (
        SimulationCase
        | FullFieldSolution
        | LocalLoadFullFieldSolution
        | FacetLoadFullFieldSolution
    ),
    solution: (
        FullFieldSolution | LocalLoadFullFieldSolution | FacetLoadFullFieldSolution | None
    ) = None,
) -> RecoveredFields:
    """Recover fields from a solution, optionally checking an explicit case."""

    if solution is None:
        if not isinstance(
            case,
            (FullFieldSolution, LocalLoadFullFieldSolution, FacetLoadFullFieldSolution),
        ):
            raise TypeError(
                "recover_fields(solution) requires a full-field solution"
            )
        solution = case
    else:
        if not isinstance(case, SimulationCase):
            raise TypeError("case must be a SimulationCase")
        if not isinstance(solution, FullFieldSolution):
            raise TypeError("solution must be a FullFieldSolution")
        if case != solution.case:
            raise PostprocessError(
                "simulation case does not match the case used to create the "
                "existing solution; solve the requested case again"
            )

    if isinstance(solution, FullFieldSolution) and not isinstance(
        solution.case, SimulationCase
    ):
        raise PostprocessError(
            "existing solution does not contain valid simulation-case provenance"
        )

    try:
        solution.validate_displacement_consistency()
    except SolverError as exc:
        raise PostprocessError(
            "existing solution has inconsistent displacement representations; "
            "the nodal displacements must match the displacement vector DOF mapping"
        ) from exc

    solve_case = solution.case

    mesh = solution.mesh
    node_coordinates = np.asarray(mesh.p.T, dtype=float)
    connectivity = np.asarray(mesh.t.T, dtype=np.int32)
    nodal_displacements = np.asarray(solution.nodal_displacements, dtype=float)
    if nodal_displacements.shape != node_coordinates.shape:
        raise PostprocessError(
            "nodal displacement shape does not match imported mesh coordinates"
        )

    interpolated = solution.basis.interpolate(solution.displacement_vector)
    gradient = np.asarray(interpolated.grad, dtype=float)
    expected_prefix = (2, 2, connectivity.shape[0])
    if gradient.ndim != 4 or gradient.shape[:3] != expected_prefix:
        raise PostprocessError(
            "unexpected scikit-fem displacement-gradient array shape"
        )
    if gradient.shape[3] == 0:
        raise PostprocessError("scikit-fem returned no element quadrature points")
    if not np.allclose(
        gradient,
        gradient[:, :, :, :1],
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise PostprocessError(
            "vector P1 displacement gradients must be constant per triangle"
        )

    # 中文导读：P1 三角形内位移梯度为常数，因此应变和应力按单元保存。
    element_gradient = gradient[:, :, :, 0]
    strain_xx = element_gradient[0, 0]
    strain_yy = element_gradient[1, 1]
    engineering_shear = element_gradient[0, 1] + element_gradient[1, 0]

    elastic_modulus = solve_case.material.E
    poisson_ratio = solve_case.material.nu
    plane_stress_factor = elastic_modulus / (1.0 - poisson_ratio**2)
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    derived_strain_zz = (
        -poisson_ratio
        / (1.0 - poisson_ratio)
        * (strain_xx + strain_yy)
    )
    stress_xx = plane_stress_factor * (strain_xx + poisson_ratio * strain_yy)
    stress_yy = plane_stress_factor * (poisson_ratio * strain_xx + strain_yy)
    shear_stress = shear_modulus * engineering_shear

    von_mises_squared = (
        stress_xx**2
        - stress_xx * stress_yy
        + stress_yy**2
        + 3.0 * shear_stress**2
    )
    von_mises = np.sqrt(np.maximum(von_mises_squared, 0.0))
    mean_stress = 0.5 * (stress_xx + stress_yy)
    principal_radius = np.hypot(0.5 * (stress_xx - stress_yy), shear_stress)
    principal_1 = mean_stress + principal_radius
    principal_2 = mean_stress - principal_radius

    centroids, areas = _element_geometry(node_coordinates, connectivity)
    displacement_magnitude = np.linalg.norm(nodal_displacements, axis=1)
    arrays = (
        node_coordinates,
        nodal_displacements,
        displacement_magnitude,
        centroids,
        areas,
        strain_xx,
        strain_yy,
        engineering_shear,
        derived_strain_zz,
        stress_xx,
        stress_yy,
        shear_stress,
        von_mises,
        principal_1,
        principal_2,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise PostprocessError("recovered fields contain a non-finite value")
    if np.any(areas <= 0.0):
        raise PostprocessError("recovered fields contain a non-positive area")
    if np.any(von_mises < 0.0):
        raise PostprocessError("von Mises stress must be non-negative")
    if np.any(principal_1 < principal_2):
        raise PostprocessError(
            "first in-plane principal stress must not be below the second"
        )

    return RecoveredFields(
        node_coordinates=node_coordinates,
        nodal_displacements=nodal_displacements,
        displacement_magnitude=displacement_magnitude,
        element_connectivity=connectivity,
        element_centroids=centroids,
        element_areas=areas,
        strain_xx=strain_xx,
        strain_yy=strain_yy,
        engineering_shear_strain_xy=engineering_shear,
        derived_strain_zz=derived_strain_zz,
        stress_xx=stress_xx,
        stress_yy=stress_yy,
        shear_stress_xy=shear_stress,
        von_mises_stress=von_mises,
        principal_stress_1_in_plane=principal_1,
        principal_stress_2_in_plane=principal_2,
    )


def area_weighted_quantile(
    values: NDArray[np.float64],
    areas: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return the first sorted value whose cumulative area reaches quantile."""

    values_array = np.asarray(values, dtype=float).reshape(-1)
    areas_array = np.asarray(areas, dtype=float).reshape(-1)
    if values_array.shape != areas_array.shape or values_array.size == 0:
        raise ValueError("values and areas must be non-empty arrays of equal size")
    if not 0.0 <= quantile <= 1.0 or not math.isfinite(quantile):
        raise ValueError("quantile must be finite and between zero and one")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values must be finite")
    if not np.all(np.isfinite(areas_array)) or np.any(areas_array <= 0.0):
        raise ValueError("areas must be finite and strictly positive")

    order = np.argsort(values_array, kind="stable")
    sorted_values = values_array[order]
    sorted_areas = areas_array[order]
    cumulative_area = np.cumsum(sorted_areas)
    target_area = quantile * float(cumulative_area[-1])
    index = int(np.searchsorted(cumulative_area, target_area, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def compute_stress_statistics(fields: RecoveredFields) -> StressStatistics:
    """Compute area-weighted and raw unsmoothed element stress statistics."""

    total_area = float(math.fsum(float(value) for value in fields.element_areas))
    weighted_mean = float(
        np.dot(fields.von_mises_stress, fields.element_areas) / total_area
    )
    raw_maximum_element = int(np.argmax(fields.von_mises_stress))
    raw_centroid = tuple(
        float(value) for value in fields.element_centroids[raw_maximum_element]
    )
    return StressStatistics(
        von_mises_area_weighted_mean=weighted_mean,
        # 中文导读：p95/p99 按单元面积加权；原始最大值仍可能随网格细化变化。
        von_mises_area_weighted_p95=area_weighted_quantile(
            fields.von_mises_stress, fields.element_areas, 0.95
        ),
        von_mises_area_weighted_p99=area_weighted_quantile(
            fields.von_mises_stress, fields.element_areas, 0.99
        ),
        von_mises_raw_element_maximum_mesh_dependent=float(
            fields.von_mises_stress[raw_maximum_element]
        ),
        raw_maximum_element_id=raw_maximum_element,
        raw_maximum_centroid=raw_centroid,
        principal_stress_1_in_plane_maximum=float(
            np.max(fields.principal_stress_1_in_plane)
        ),
        principal_stress_2_in_plane_minimum=float(
            np.min(fields.principal_stress_2_in_plane)
        ),
    )
