"""Unsmoothed element-constant recovery for Phase 6A.2 snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from grindcae.elastoplastic import equivalent_von_mises_stress
from grindcae.postprocess.recovery import area_weighted_quantile

from .models import ElastoplasticFemCase
from .solver import ConvergedIncrementState, PreparedElastoplasticMesh


class ElastoplasticRecoveryError(RuntimeError):
    """Raised when a converged snapshot cannot be recovered consistently."""


@dataclass(frozen=True, slots=True)
class ElastoplasticSnapshotFields:
    node_coordinates_m: NDArray[np.float64]
    nodal_displacements_m: NDArray[np.float64]
    displacement_magnitude_m: NDArray[np.float64]
    element_connectivity: NDArray[np.int32]
    element_centroids_m: NDArray[np.float64]
    element_areas_m2: NDArray[np.float64]
    strain_xx: NDArray[np.float64]
    strain_yy: NDArray[np.float64]
    engineering_shear_strain_xy: NDArray[np.float64]
    strain_zz: NDArray[np.float64]
    stress_xx_Pa: NDArray[np.float64]
    stress_yy_Pa: NDArray[np.float64]
    stress_zz_Pa: NDArray[np.float64]
    shear_stress_xy_Pa: NDArray[np.float64]
    von_mises_stress_Pa: NDArray[np.float64]
    plastic_strain_tensor: NDArray[np.float64]
    equivalent_plastic_strain: NDArray[np.float64]
    current_yield_strength_Pa: NDArray[np.float64]
    increment_class: tuple[str, ...]

    @property
    def node_count(self) -> int:
        return int(self.node_coordinates_m.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.element_connectivity.shape[0])

    def numeric_arrays(self) -> tuple[NDArray[np.float64], ...]:
        return (
            self.node_coordinates_m,
            self.nodal_displacements_m,
            self.displacement_magnitude_m,
            self.element_centroids_m,
            self.element_areas_m2,
            self.strain_xx,
            self.strain_yy,
            self.engineering_shear_strain_xy,
            self.strain_zz,
            self.stress_xx_Pa,
            self.stress_yy_Pa,
            self.stress_zz_Pa,
            self.shear_stress_xy_Pa,
            self.von_mises_stress_Pa,
            self.plastic_strain_tensor,
            self.equivalent_plastic_strain,
            self.current_yield_strength_Pa,
        )


@dataclass(frozen=True, slots=True)
class SnapshotStatistics:
    maximum_displacement_m: float
    maximum_displacement_node_id: int
    von_mises_area_weighted_mean_Pa: float
    von_mises_area_weighted_p95_Pa: float
    von_mises_area_weighted_p99_Pa: float
    von_mises_raw_element_maximum_mesh_dependent_Pa: float
    raw_maximum_element_id: int
    maximum_equivalent_plastic_strain: float
    plastic_element_count: int
    plastic_element_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "maximum_displacement_m": self.maximum_displacement_m,
            "maximum_displacement_node_id": self.maximum_displacement_node_id,
            "von_mises_area_weighted_mean_Pa": self.von_mises_area_weighted_mean_Pa,
            "von_mises_area_weighted_p95_Pa": self.von_mises_area_weighted_p95_Pa,
            "von_mises_area_weighted_p99_Pa": self.von_mises_area_weighted_p99_Pa,
            "von_mises_raw_element_maximum_mesh_dependent_Pa": self.von_mises_raw_element_maximum_mesh_dependent_Pa,
            "raw_maximum_element_id": self.raw_maximum_element_id,
            "maximum_equivalent_plastic_strain": self.maximum_equivalent_plastic_strain,
            "plastic_element_count": self.plastic_element_count,
            "plastic_element_fraction": self.plastic_element_fraction,
        }


def recover_snapshot_fields(
    case: ElastoplasticFemCase,
    prepared: PreparedElastoplasticMesh,
    state: ConvergedIncrementState,
) -> ElastoplasticSnapshotFields:
    """Recover one converged peak or residual state without nodal smoothing."""

    displacement = np.asarray(state.displacement_vector_m, dtype=float)
    nodal = np.column_stack(
        (
            displacement[prepared.component_dofs[0]],
            displacement[prepared.component_dofs[1]],
        )
    )
    coordinates = np.asarray(prepared.imported_mesh.mesh.p.T, dtype=float)
    connectivity = np.asarray(prepared.element_connectivity, dtype=np.int32)
    vertices = coordinates[connectivity]
    centroids = np.mean(vertices, axis=1)
    strains = np.asarray(state.element_total_strain, dtype=float)
    stresses = np.asarray(state.element_stress_Pa, dtype=float)
    stress_zz = np.asarray(state.element_sigma_zz_Pa, dtype=float)
    plastic = np.array(
        [point.plastic_strain_tensor for point in state.element_states], dtype=float
    )
    equivalent_plastic = np.array(
        [point.equivalent_plastic_strain for point in state.element_states], dtype=float
    )
    yield_strength = np.array(
        [point.current_yield_strength_Pa for point in state.element_states], dtype=float
    )
    classes = tuple(point.increment_class for point in state.element_states)
    von_mises = np.array(
        [equivalent_von_mises_stress(point.stress_tensor_Pa) for point in state.element_states]
    )
    result = ElastoplasticSnapshotFields(
        node_coordinates_m=coordinates,
        nodal_displacements_m=nodal,
        displacement_magnitude_m=np.linalg.norm(nodal, axis=1),
        element_connectivity=connectivity,
        element_centroids_m=centroids,
        element_areas_m2=np.asarray(prepared.element_areas_m2, dtype=float),
        strain_xx=strains[:, 0],
        strain_yy=strains[:, 1],
        engineering_shear_strain_xy=strains[:, 2],
        strain_zz=np.zeros(prepared.element_count, dtype=float),
        stress_xx_Pa=stresses[:, 0],
        stress_yy_Pa=stresses[:, 1],
        stress_zz_Pa=stress_zz,
        shear_stress_xy_Pa=stresses[:, 2],
        von_mises_stress_Pa=von_mises,
        plastic_strain_tensor=plastic,
        equivalent_plastic_strain=equivalent_plastic,
        current_yield_strength_Pa=yield_strength,
        increment_class=classes,
    )
    if not all(np.all(np.isfinite(array)) for array in result.numeric_arrays()):
        raise ElastoplasticRecoveryError("recovered snapshot contains a non-finite value")
    if np.any(result.element_areas_m2 <= 0.0):
        raise ElastoplasticRecoveryError("recovered snapshot contains a non-positive area")
    if np.any(result.equivalent_plastic_strain < 0.0):
        raise ElastoplasticRecoveryError("equivalent plastic strain must be non-negative")
    return result


def compute_snapshot_statistics(fields: ElastoplasticSnapshotFields) -> SnapshotStatistics:
    areas = fields.element_areas_m2
    stress = fields.von_mises_stress_Pa
    total_area = float(math.fsum(float(value) for value in areas))
    maximum_node = int(np.argmax(fields.displacement_magnitude_m))
    raw_maximum = int(np.argmax(stress))
    plastic_count = int(np.count_nonzero(fields.equivalent_plastic_strain > 1.0e-15))
    return SnapshotStatistics(
        maximum_displacement_m=float(fields.displacement_magnitude_m[maximum_node]),
        maximum_displacement_node_id=maximum_node,
        von_mises_area_weighted_mean_Pa=float(np.dot(stress, areas) / total_area),
        von_mises_area_weighted_p95_Pa=area_weighted_quantile(stress, areas, 0.95),
        von_mises_area_weighted_p99_Pa=area_weighted_quantile(stress, areas, 0.99),
        von_mises_raw_element_maximum_mesh_dependent_Pa=float(stress[raw_maximum]),
        raw_maximum_element_id=raw_maximum,
        maximum_equivalent_plastic_strain=float(np.max(fields.equivalent_plastic_strain)),
        plastic_element_count=plastic_count,
        plastic_element_fraction=plastic_count / fields.element_count,
    )
