"""Recover nominal, peak-loaded, and residual top surfaces for Phase 6A.3."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from grindcae.elastoplastic_fem import ElastoplasticSnapshotFields
from grindcae.evolved_mesh import EvolvedMeshValidationResult


class SurfaceRecoveryError(RuntimeError):
    """Raised when the three top-surface definitions cannot be recovered."""


@dataclass(frozen=True, slots=True)
class SurfaceRecoveryPoint:
    surface_node_id: int
    x_m: float
    nominal_y_m: float
    peak_loaded_y_m: float
    residual_unloaded_y_m: float
    peak_uy_m: float
    residual_uy_m: float
    elastic_recovery_y_m: float
    surface_region: str


@dataclass(frozen=True, slots=True)
class SurfaceRecoveryResult:
    points: tuple[SurfaceRecoveryPoint, ...]

    @property
    def maximum_absolute_elastic_recovery_y_m(self) -> float:
        return max((abs(point.elastic_recovery_y_m) for point in self.points), default=0.0)

    @property
    def mean_elastic_recovery_y_m(self) -> float:
        if not self.points:
            return 0.0
        return math.fsum(point.elastic_recovery_y_m for point in self.points) / len(
            self.points
        )


def build_surface_recovery(
    evolved_mesh: EvolvedMeshValidationResult,
    peak_fields: ElastoplasticSnapshotFields,
    residual_fields: ElastoplasticSnapshotFields,
) -> SurfaceRecoveryResult:
    """Use the final solve-mesh top nodes and true unscaled displacements."""

    if not isinstance(evolved_mesh, EvolvedMeshValidationResult):
        raise TypeError("evolved_mesh must be an EvolvedMeshValidationResult")
    if not isinstance(peak_fields, ElastoplasticSnapshotFields):
        raise TypeError("peak_fields must be ElastoplasticSnapshotFields")
    if not isinstance(residual_fields, ElastoplasticSnapshotFields):
        raise TypeError("residual_fields must be ElastoplasticSnapshotFields")
    if peak_fields.node_count != evolved_mesh.node_count or residual_fields.node_count != evolved_mesh.node_count:
        raise SurfaceRecoveryError("snapshot node count does not match the evolved mesh")
    coordinates = np.asarray(evolved_mesh.skfem_mesh.p.T, dtype=float)
    if not np.allclose(peak_fields.node_coordinates_m, coordinates, rtol=0.0, atol=1.0e-14):
        raise SurfaceRecoveryError("peak snapshot coordinates do not match the final mesh")
    if not np.allclose(residual_fields.node_coordinates_m, coordinates, rtol=0.0, atol=1.0e-14):
        raise SurfaceRecoveryError("residual snapshot coordinates do not match the final mesh")

    region_priority = {"ground": 1, "unprocessed": 1, "contact_arc": 2}
    regions: dict[int, str] = {}
    for facet in evolved_mesh.top_boundary_facets:
        for node_id in (facet.node_start_id, facet.node_end_id):
            previous = regions.get(node_id)
            if previous is None or region_priority[facet.region] > region_priority[previous]:
                regions[node_id] = facet.region
    ordered_nodes = sorted(regions, key=lambda node_id: (coordinates[node_id, 0], node_id))
    points: list[SurfaceRecoveryPoint] = []
    for node_id in ordered_nodes:
        x_m, nominal_y_m = coordinates[node_id]
        peak_uy = float(peak_fields.nodal_displacements_m[node_id, 1])
        residual_uy = float(residual_fields.nodal_displacements_m[node_id, 1])
        peak_y = float(nominal_y_m + peak_uy)
        residual_y = float(nominal_y_m + residual_uy)
        point = SurfaceRecoveryPoint(
            surface_node_id=int(node_id),
            x_m=float(x_m),
            nominal_y_m=float(nominal_y_m),
            peak_loaded_y_m=peak_y,
            residual_unloaded_y_m=residual_y,
            peak_uy_m=peak_uy,
            residual_uy_m=residual_uy,
            elastic_recovery_y_m=residual_y - peak_y,
            surface_region=regions[node_id],
        )
        if not all(
            math.isfinite(value)
            for value in (
                point.x_m,
                point.nominal_y_m,
                point.peak_loaded_y_m,
                point.residual_unloaded_y_m,
                point.peak_uy_m,
                point.residual_uy_m,
                point.elastic_recovery_y_m,
            )
        ):
            raise SurfaceRecoveryError("surface recovery contains a non-finite value")
        points.append(point)
    return SurfaceRecoveryResult(points=tuple(points))
