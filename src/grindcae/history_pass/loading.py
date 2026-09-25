"""Exact consistent P1 nodal assembly from Phase 6B.1 partial-facet mappings."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from grindcae.elastoplastic_fem import ElastoplasticSolverError, PreparedElastoplasticMesh
from grindcae.moving_load_pass import FixedTopLoadMapping


def assemble_fixed_top_mapping_load_vector(
    prepared: PreparedElastoplasticMesh,
    mapping: FixedTopLoadMapping,
) -> NDArray[np.float64]:
    if not isinstance(prepared, PreparedElastoplasticMesh):
        raise TypeError("prepared must be a PreparedElastoplasticMesh")
    if not isinstance(mapping, FixedTopLoadMapping):
        raise TypeError("mapping must be a FixedTopLoadMapping")
    vector = np.zeros(prepared.total_degrees_of_freedom, dtype=float)
    contact = set(int(value) for value in prepared.imported_mesh.mesh.boundaries["contact"])
    for facet in mapping.facets:
        if facet.facet_id not in contact:
            raise ElastoplasticSolverError("mapped facet is outside the imported contact boundary")
        xi0 = (facet.overlap_x_start_m - facet.facet_x_start_m) / facet.facet_dx_m
        xi1 = (facet.overlap_x_end_m - facet.facet_x_start_m) / facet.facet_dx_m
        tolerance = 1.0e-12
        if xi0 < -tolerance or xi1 > 1.0 + tolerance or xi1 <= xi0:
            raise ElastoplasticSolverError("partial-facet overlap coordinates are invalid")
        xi0 = max(0.0, xi0)
        xi1 = min(1.0, xi1)
        quadratic = 0.5 * (xi1 * xi1 - xi0 * xi0)
        start_weight_ds = facet.facet_ds_m * ((xi1 - xi0) - quadratic)
        end_weight_ds = facet.facet_ds_m * quadratic
        for component, traction in (
            (0, facet.tx_applied_per_loaded_ds_N_per_m),
            (1, facet.ty_applied_per_loaded_ds_N_per_m),
        ):
            vector[prepared.component_dofs[component, facet.node_start_id]] += traction * start_weight_ds
            vector[prepared.component_dofs[component, facet.node_end_id]] += traction * end_weight_ds
    assembled_x = float(math.fsum(float(value) for value in vector[prepared.component_dofs[0]]))
    assembled_y = float(math.fsum(float(value) for value in vector[prepared.component_dofs[1]]))
    for actual, expected, name in (
        (assembled_x, mapping.assembled_Fx_N, "Fx"),
        (assembled_y, mapping.assembled_Fy_N, "Fy"),
    ):
        scale = max(abs(actual), abs(expected), 1.0)
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12 * scale):
            raise ElastoplasticSolverError(f"consistent partial-facet nodal {name} does not conserve force")
    return vector
