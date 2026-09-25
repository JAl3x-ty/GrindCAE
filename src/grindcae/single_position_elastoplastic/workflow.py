"""Thin Phase 6A.3 orchestration over the retained 4C and 6A.2 APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from grindcae.elastoplastic_fem import (
    ElastoplasticSnapshotFields,
    LoadingUnloadingResult,
    PreparedElastoplasticMesh,
    SnapshotStatistics,
    compute_snapshot_statistics,
    prepare_elastoplastic_mesh_with_facet_tractions,
    recover_snapshot_fields,
    solve_loading_unloading,
)
from grindcae.evolved_fem import (
    EvolvedFacetLoadResult,
    build_evolved_facet_load,
)
from grindcae.evolved_mesh import (
    EvolvedMeshCase,
    EvolvedMeshValidationResult,
    generate_evolved_surface_mesh,
    validate_evolved_surface_mesh,
)
from grindcae.pass_loading import PassLoadResult, build_pass_load
from grindcae.solver import ImportedMesh

from .models import SinglePositionElastoplasticCase
from .recovery import SurfaceRecoveryResult, build_surface_recovery


class SinglePositionElastoplasticWorkflowError(RuntimeError):
    """Raised when Phase 6A.3 provenance or orchestration is invalid."""


@dataclass(frozen=True, slots=True)
class SinglePositionElastoplasticOperations:
    build_pass_load: Callable[..., object]
    generate_mesh: Callable[..., object]
    validate_mesh: Callable[..., object]
    build_facet_load: Callable[..., object]
    prepare_mesh: Callable[..., object]
    solve_history: Callable[..., object]
    recover_snapshot: Callable[..., object]
    compute_statistics: Callable[..., object]
    build_surface_recovery: Callable[..., object]


def default_operations() -> SinglePositionElastoplasticOperations:
    return SinglePositionElastoplasticOperations(
        build_pass_load=build_pass_load,
        generate_mesh=generate_evolved_surface_mesh,
        validate_mesh=validate_evolved_surface_mesh,
        build_facet_load=build_evolved_facet_load,
        prepare_mesh=prepare_elastoplastic_mesh_with_facet_tractions,
        solve_history=solve_loading_unloading,
        recover_snapshot=recover_snapshot_fields,
        compute_statistics=compute_snapshot_statistics,
        build_surface_recovery=build_surface_recovery,
    )


@dataclass(frozen=True, slots=True)
class SinglePositionElastoplasticResult:
    case: SinglePositionElastoplasticCase
    pass_load: PassLoadResult
    evolved_mesh: EvolvedMeshValidationResult
    facet_load: EvolvedFacetLoadResult
    prepared_mesh: PreparedElastoplasticMesh
    history: LoadingUnloadingResult
    peak_fields: ElastoplasticSnapshotFields
    residual_fields: ElastoplasticSnapshotFields
    peak_statistics: SnapshotStatistics
    residual_statistics: SnapshotStatistics
    surface_recovery: SurfaceRecoveryResult
    msh_source: Path

    def __post_init__(self) -> None:
        source = Path(self.msh_source).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".msh":
            raise SinglePositionElastoplasticWorkflowError(
                "result MSH source is missing or invalid"
            )
        if self.pass_load.case != self.case.pass_load:
            raise SinglePositionElastoplasticWorkflowError(
                "pass-load result provenance is invalid"
            )
        if self.evolved_mesh.trajectory != self.pass_load.trajectory_result:
            raise SinglePositionElastoplasticWorkflowError(
                "mesh trajectory provenance is invalid"
            )
        if self.facet_load.mesh != self.evolved_mesh:
            raise SinglePositionElastoplasticWorkflowError(
                "facet-load mesh provenance is invalid"
            )
        if self.prepared_mesh.imported_mesh.source_path != source:
            raise SinglePositionElastoplasticWorkflowError(
                "prepared-mesh MSH provenance is invalid"
            )
        if self.history.prepared_mesh is not self.prepared_mesh:
            raise SinglePositionElastoplasticWorkflowError(
                "loading and unloading did not retain one prepared mesh"
            )
        object.__setattr__(self, "msh_source", source)


def evolved_mesh_case(case: SinglePositionElastoplasticCase) -> EvolvedMeshCase:
    return EvolvedMeshCase(
        evolved_mesh_schema_version=1,
        unit_system=case.unit_system,
        model_type="single_pass_evolved_surface_gmsh",
        trajectory=case.pass_load.trajectory,
        mesh=case.mesh,
    )


def solve_single_position_elastoplastic(
    case: SinglePositionElastoplasticCase,
    workspace: str | Path,
    *,
    operations: SinglePositionElastoplasticOperations | None = None,
) -> SinglePositionElastoplasticResult:
    """Generate one evolved mesh and solve its single 0 -> 1 -> 0 history."""

    if not isinstance(case, SinglePositionElastoplasticCase):
        raise TypeError("case must be a SinglePositionElastoplasticCase")
    ops = operations or default_operations()
    workspace_path = Path(workspace).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    msh_path = (workspace_path / "solve_input.msh").resolve()
    pass_load = ops.build_pass_load(case.pass_load)
    mesh_case = evolved_mesh_case(case)
    generation = ops.generate_mesh(mesh_case, pass_load.trajectory_result, msh_path)
    evolved_mesh = ops.validate_mesh(
        msh_path,
        mesh_case,
        pass_load.trajectory_result,
        generation,
    )
    facet_load = ops.build_facet_load(pass_load, evolved_mesh)
    imported_mesh = ImportedMesh(mesh=evolved_mesh.skfem_mesh, source_path=msh_path)
    fem_case = case.to_elastoplastic_fem_case(
        peak_force_x_N=pass_load.current_Fx_N,
        peak_force_y_N=pass_load.current_Fy_N,
    )
    prepared = ops.prepare_mesh(
        fem_case,
        imported_mesh,
        facet_load.facet_ids,
        peak_force_x_N=pass_load.current_Fx_N,
        peak_force_y_N=pass_load.current_Fy_N,
        facet_line_load_x_N_per_m=facet_load.tx_per_ds,
        facet_line_load_y_N_per_m=facet_load.ty_per_ds,
    )
    history = ops.solve_history(fem_case, prepared)
    peak_fields = ops.recover_snapshot(fem_case, prepared, history.peak.state)
    residual_fields = ops.recover_snapshot(
        fem_case, prepared, history.residual.state
    )
    peak_statistics = ops.compute_statistics(peak_fields)
    residual_statistics = ops.compute_statistics(residual_fields)
    surface_recovery = ops.build_surface_recovery(
        evolved_mesh, peak_fields, residual_fields
    )
    return SinglePositionElastoplasticResult(
        case=case,
        pass_load=pass_load,
        evolved_mesh=evolved_mesh,
        facet_load=facet_load,
        prepared_mesh=prepared,
        history=history,
        peak_fields=peak_fields,
        residual_fields=residual_fields,
        peak_statistics=peak_statistics,
        residual_statistics=residual_statistics,
        surface_recovery=surface_recovery,
        msh_source=msh_path,
    )
