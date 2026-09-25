from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skfem import MeshTri

from grindcae.elastoplastic_fem.models import ElastoplasticFemCase
from grindcae.elastoplastic_fem.solver import prepare_elastoplastic_mesh_with_facet_tractions
from grindcae.history_pass.loading import assemble_fixed_top_mapping_load_vector
from grindcae.moving_load_pass import FixedTopFacetLoad, FixedTopLoadMapping
from grindcae.pass_loading import build_pass_load
from grindcae.solver import ImportedMesh

from test_elastoplastic_fem_models import valid_payload as fem_payload
from test_moving_load_pass_models import valid_payload as moving_payload
from grindcae.moving_load_pass import FixedMeshMovingLoadPassCase


def _prepared():
    payload = fem_payload()
    payload["geometry"] = {"type": "rectangle", "width": 1.0, "height": 1.0}
    payload["mesh"] = {"target_size": 1.0}
    payload["load"] = {
        "boundary": "top",
        "distribution": "uniform_over_interval",
        "x_start_m": 0.0,
        "x_end_m": 1.0,
        "peak_force_x_N": 0.0,
        "peak_force_y_N": 0.0,
    }
    case = ElastoplasticFemCase.from_mapping(payload)
    mesh = MeshTri.init_tensor(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])).with_boundaries(
        {
            "fixed": lambda x: np.isclose(x[1], 0.0),
            "contact": lambda x: np.isclose(x[1], 1.0),
            "free_left": lambda x: np.isclose(x[0], 0.0),
            "free_right": lambda x: np.isclose(x[0], 1.0),
        }
    )
    imported = ImportedMesh(mesh=mesh, source_path=Path("partial-p1.msh"))
    prepared = prepare_elastoplastic_mesh_with_facet_tractions(
        case,
        imported,
        np.empty(0, dtype=np.int32),
        peak_force_x_N=0.0,
        peak_force_y_N=0.0,
        facet_line_load_x_N_per_m=np.empty(0),
        facet_line_load_y_N_per_m=np.empty(0),
    )
    return prepared


def test_off_centre_partial_facet_uses_exact_p1_subsegment_shape_integrals() -> None:
    prepared = _prepared()
    case = FixedMeshMovingLoadPassCase.from_mapping(moving_payload())
    pass_load = build_pass_load(case.reference_case.pass_load)
    top_facet_id = int(prepared.imported_mesh.mesh.boundaries["contact"][0])
    nodes = prepared.imported_mesh.mesh.facets[:, top_facet_id]
    ordered = sorted((int(node) for node in nodes), key=lambda node: prepared.imported_mesh.mesh.p[0, node])
    facet = FixedTopFacetLoad(
        facet_id=top_facet_id,
        node_start_id=ordered[0],
        node_end_id=ordered[1],
        facet_x_start_m=0.0,
        facet_x_end_m=1.0,
        facet_dx_m=1.0,
        facet_ds_m=1.0,
        overlap_x_start_m=0.2,
        overlap_x_end_m=0.5,
        overlap_dx_m=0.3,
        overlap_ds_m=0.3,
        qx_projected_N_per_m=10.0,
        qy_projected_N_per_m=-20.0,
        tx_applied_per_loaded_ds_N_per_m=10.0,
        ty_applied_per_loaded_ds_N_per_m=-20.0,
        integrated_Fx_N=3.0,
        integrated_Fy_N=-6.0,
        is_partial_facet=True,
    )
    mapping = FixedTopLoadMapping(
        pass_load=pass_load,
        facets=(facet,),
        projected_overlap_sum_m=0.3,
        loaded_actual_length_sum_m=0.3,
        assembled_Fx_N=3.0,
        assembled_Fy_N=-6.0,
        force_balance_residual_x_N=0.0,
        force_balance_residual_y_N=0.0,
        force_balance_residual_N=0.0,
    )

    vector = assemble_fixed_top_mapping_load_vector(prepared, mapping)

    # Integral N_start dx over [0.2, 0.5] = 0.195; N_end = 0.105.
    x = vector[prepared.component_dofs[0]]
    y = vector[prepared.component_dofs[1]]
    assert x[ordered[0]] == pytest.approx(1.95)
    assert x[ordered[1]] == pytest.approx(1.05)
    assert y[ordered[0]] == pytest.approx(-3.9)
    assert y[ordered[1]] == pytest.approx(-2.1)
    assert np.sum(x) == pytest.approx(3.0)
    assert np.sum(y) == pytest.approx(-6.0)
    assert x[ordered[0]] != pytest.approx(1.5)
