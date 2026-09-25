"""Phase 6B.2 orchestration over the exact Phase 6B.1 moving loads and 6A.2 solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from grindcae.elastoplastic_fem import (
    ConvergedIncrementState,
    ElastoplasticSnapshotFields,
    PreparedElastoplasticMesh,
    SnapshotStatistics,
    advance_external_load_increment,
    compute_snapshot_statistics,
    initial_structure_state,
    prepare_elastoplastic_mesh_with_facet_tractions,
    recover_snapshot_fields,
)
from grindcae.moving_load_pass import (
    FixedMeshMovingLoadPassResult,
    MovingLoadPositionResult,
    solve_fixed_mesh_moving_load_pass,
)

from .loading import assemble_fixed_top_mapping_load_vector
from .models import FixedMeshElastoplasticHistoryPassCase
from .solver import ExternalLoadTransitionResult, solve_external_load_transition


class HistoryPassWorkflowError(RuntimeError):
    """Raised when fixed-mesh history provenance or state monotonicity is invalid."""


@dataclass(frozen=True, slots=True)
class HistoryPassRow:
    position_id: int
    motion_coordinate_m: float
    pass_state: str
    contact_ratio: float
    Fx_N: float
    Fy_N: float
    assembled_Fx_N: float
    assembled_Fy_N: float
    fixed_reaction_x_N: float
    fixed_reaction_y_N: float
    balance_residual_N: float
    transition_substep_count: int
    retry_count: int
    maximum_newton_iteration_count: int
    maximum_displacement_m: float
    maximum_von_mises_stress_Pa: float
    maximum_equivalent_plastic_strain: float
    plastic_element_count: int
    plastic_element_fraction: float
    accumulated_plastic_dissipation_J: float
    state_committed: bool
    target_load_vector_N: NDArray[np.float64]
    state: ConvergedIncrementState


@dataclass(frozen=True, slots=True)
class HistoryPositionRecord:
    moving_load_position: MovingLoadPositionResult
    target_load_vector_N: NDArray[np.float64]
    transition: ExternalLoadTransitionResult[ConvergedIncrementState] | None
    row: HistoryPassRow


@dataclass(frozen=True, slots=True)
class TargetFieldSnapshot:
    """Read-only recovered fields for one exact committed moving-load target."""

    frame_index: int
    position_id: int
    motion_coordinate_m: float
    wheel_lowest_point_x_m: float
    pass_state: str
    contact_ratio: float
    Fx_N: float
    Fy_N: float
    accumulated_plastic_dissipation_J: float
    fields: ElastoplasticSnapshotFields
    is_final_unloaded_target: bool


@dataclass(frozen=True, slots=True)
class FixedMeshElastoplasticHistoryPassResult:
    case: FixedMeshElastoplasticHistoryPassCase
    moving_load: FixedMeshMovingLoadPassResult
    prepared_mesh: PreparedElastoplasticMesh
    position_records: tuple[HistoryPositionRecord, ...]
    history_rows: tuple[HistoryPassRow, ...]
    final_fields: ElastoplasticSnapshotFields
    final_statistics: SnapshotStatistics
    maximum_plastic_fields: ElastoplasticSnapshotFields
    maximum_plastic_statistics: SnapshotStatistics
    accumulated_plastic_dissipation_J: float

    @property
    def total_transition_substep_count(self) -> int:
        return sum(row.transition_substep_count for row in self.history_rows)

    @property
    def total_retry_count(self) -> int:
        return sum(row.retry_count for row in self.history_rows)

    @property
    def maximum_newton_iterations(self) -> int:
        return max(row.maximum_newton_iteration_count for row in self.history_rows)

    @property
    def maximum_balance_residual_N(self) -> float:
        return max(row.balance_residual_N for row in self.history_rows)


@dataclass(frozen=True, slots=True)
class SharedHistoryAdvanceResult:
    """Internal reusable 6B.2 state-transfer engine result.

    Both the retained uniform route and Phase 7A.3 call this one engine; it
    owns every Newton trial, rollback, state commit, and final zero-load row.
    """

    position_records: tuple[HistoryPositionRecord, ...]
    history_rows: tuple[HistoryPassRow, ...]
    final_fields: ElastoplasticSnapshotFields
    final_statistics: SnapshotStatistics
    maximum_plastic_fields: ElastoplasticSnapshotFields
    maximum_plastic_statistics: SnapshotStatistics
    accumulated_plastic_dissipation_J: float


def _plastic_dissipation_increment_J(
    prepared: PreparedElastoplasticMesh,
    thickness_m: float,
    previous: ConvergedIncrementState,
    current: ConvergedIncrementState,
) -> float:
    total = 0.0
    volumes = prepared.element_areas_m2 * thickness_m
    for element_id, (old, new) in enumerate(zip(previous.element_states, current.element_states)):
        delta = new.equivalent_plastic_strain - old.equivalent_plastic_strain
        if delta < -1.0e-14:
            raise HistoryPassWorkflowError("equivalent plastic strain decreased after convergence")
        if delta > 0.0:
            total += 0.5 * (old.current_yield_strength_Pa + new.current_yield_strength_Pa) * delta * volumes[element_id]
    return total


def _row(
    *,
    position_id: int,
    motion_coordinate_m: float,
    pass_state: str,
    contact_ratio: float,
    Fx_N: float,
    Fy_N: float,
    assembled_Fx_N: float,
    assembled_Fy_N: float,
    transition_substep_count: int,
    retry_count: int,
    maximum_newton_iteration_count: int,
    accumulated_plastic_dissipation_J: float,
    target_load_vector_N: NDArray[np.float64],
    state: ConvergedIncrementState,
    fields: ElastoplasticSnapshotFields,
) -> HistoryPassRow:
    return HistoryPassRow(
        position_id=position_id,
        motion_coordinate_m=motion_coordinate_m,
        pass_state=pass_state,
        contact_ratio=contact_ratio,
        Fx_N=Fx_N,
        Fy_N=Fy_N,
        assembled_Fx_N=assembled_Fx_N,
        assembled_Fy_N=assembled_Fy_N,
        fixed_reaction_x_N=state.fixed_reaction_x_N,
        fixed_reaction_y_N=state.fixed_reaction_y_N,
        balance_residual_N=state.balance_residual_norm_N,
        transition_substep_count=transition_substep_count,
        retry_count=retry_count,
        maximum_newton_iteration_count=maximum_newton_iteration_count,
        maximum_displacement_m=state.maximum_displacement_m,
        maximum_von_mises_stress_Pa=float(np.max(fields.von_mises_stress_Pa)),
        maximum_equivalent_plastic_strain=state.maximum_equivalent_plastic_strain,
        plastic_element_count=state.plastic_element_count,
        plastic_element_fraction=state.plastic_element_fraction,
        accumulated_plastic_dissipation_J=accumulated_plastic_dissipation_J,
        state_committed=True,
        target_load_vector_N=np.asarray(target_load_vector_N, dtype=float).copy(),
        state=state,
    )


def _read_only_array(array: NDArray[np.generic]) -> NDArray[np.generic]:
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


def _read_only_fields(fields: ElastoplasticSnapshotFields) -> ElastoplasticSnapshotFields:
    """Detach callback fields from solver state and make every array immutable."""

    return ElastoplasticSnapshotFields(
        node_coordinates_m=_read_only_array(fields.node_coordinates_m),
        nodal_displacements_m=_read_only_array(fields.nodal_displacements_m),
        displacement_magnitude_m=_read_only_array(fields.displacement_magnitude_m),
        element_connectivity=_read_only_array(fields.element_connectivity),
        element_centroids_m=_read_only_array(fields.element_centroids_m),
        element_areas_m2=_read_only_array(fields.element_areas_m2),
        strain_xx=_read_only_array(fields.strain_xx),
        strain_yy=_read_only_array(fields.strain_yy),
        engineering_shear_strain_xy=_read_only_array(fields.engineering_shear_strain_xy),
        strain_zz=_read_only_array(fields.strain_zz),
        stress_xx_Pa=_read_only_array(fields.stress_xx_Pa),
        stress_yy_Pa=_read_only_array(fields.stress_yy_Pa),
        stress_zz_Pa=_read_only_array(fields.stress_zz_Pa),
        shear_stress_xy_Pa=_read_only_array(fields.shear_stress_xy_Pa),
        von_mises_stress_Pa=_read_only_array(fields.von_mises_stress_Pa),
        plastic_strain_tensor=_read_only_array(fields.plastic_strain_tensor),
        equivalent_plastic_strain=_read_only_array(fields.equivalent_plastic_strain),
        current_yield_strength_Pa=_read_only_array(fields.current_yield_strength_Pa),
        increment_class=tuple(fields.increment_class),
    )


def advance_prepared_history(
    case: FixedMeshElastoplasticHistoryPassCase,
    moving: FixedMeshMovingLoadPassResult,
    prepared: PreparedElastoplasticMesh,
    target_load_vectors_N: tuple[NDArray[np.float64], ...],
    on_target_snapshot: Callable[[TargetFieldSnapshot], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> SharedHistoryAdvanceResult:
    """Advance independent committed J2 states through supplied ordered targets."""
    if len(target_load_vectors_N) != len(moving.positions):
        raise HistoryPassWorkflowError("history target vector count must match moving positions")
    reference = case.reference_case
    fem_case = reference.to_elastoplastic_fem_case(peak_force_x_N=0.0, peak_force_y_N=0.0)
    zero = np.zeros(prepared.total_degrees_of_freedom, dtype=float)
    initial = initial_structure_state(fem_case, prepared)
    committed = advance_external_load_increment(fem_case, prepared, initial, target_external_load_vector_N=zero, target_load_marker=0.0)
    accumulated = 0.0; rows: list[HistoryPassRow] = []; records: list[HistoryPositionRecord] = []
    maximum_state = committed; maximum_fields = recover_snapshot_fields(fem_case, prepared, committed); current_load = zero
    for index, moving_position in enumerate(moving.positions):
        target = np.asarray(target_load_vectors_N[index], dtype=float).copy()
        transition: ExternalLoadTransitionResult[ConvergedIncrementState] | None = None
        retries = 0; substeps = 0; max_iterations = committed.newton_iterations
        if index > 0:
            requested = case.transition.final_unloading_substep_count if index == len(moving.positions) - 1 else case.transition.base_substep_count
            def advance(previous: ConvergedIncrementState, external: NDArray[np.float64], marker: float) -> ConvergedIncrementState:
                return advance_external_load_increment(fem_case, prepared, previous, target_external_load_vector_N=external, target_load_marker=marker)
            transition = solve_external_load_transition(committed=committed, start_load_vector_N=current_load, target_load_vector_N=target, requested_substep_count=requested, settings=case.transition, advance=advance, marker_start=float(index))
            for state in transition.states:
                accumulated += _plastic_dissipation_increment_J(prepared, reference.analysis.thickness, committed, state); committed = state
            retries = transition.retry_count; substeps = transition.achieved_substep_count; max_iterations = transition.maximum_newton_iterations
        current_load = target; fields = recover_snapshot_fields(fem_case, prepared, committed)
        if committed.maximum_equivalent_plastic_strain > maximum_state.maximum_equivalent_plastic_strain:
            maximum_state = committed; maximum_fields = fields
        mapping = moving_position.load_mapping
        row = _row(position_id=moving_position.position.position_id, motion_coordinate_m=moving_position.position.motion_coordinate_m, pass_state=moving_position.position.pass_state, contact_ratio=moving_position.pass_load.contact_ratio, Fx_N=moving_position.pass_load.current_Fx_N, Fy_N=moving_position.pass_load.current_Fy_N, assembled_Fx_N=float(np.sum(target[prepared.component_dofs[0]])), assembled_Fy_N=float(np.sum(target[prepared.component_dofs[1]])), transition_substep_count=substeps, retry_count=retries, maximum_newton_iteration_count=max_iterations, accumulated_plastic_dissipation_J=accumulated, target_load_vector_N=target, state=committed, fields=fields)
        rows.append(row); records.append(HistoryPositionRecord(moving_load_position=moving_position, target_load_vector_N=target, transition=transition, row=row))
        if on_target_snapshot is not None:
            on_target_snapshot(
                TargetFieldSnapshot(
                    frame_index=index,
                    position_id=moving_position.position.position_id,
                    motion_coordinate_m=moving_position.position.motion_coordinate_m,
                    wheel_lowest_point_x_m=moving_position.position.wheel_lowest_point_x_m,
                    pass_state=moving_position.position.pass_state,
                    contact_ratio=moving_position.pass_load.contact_ratio,
                    Fx_N=moving_position.pass_load.current_Fx_N,
                    Fy_N=moving_position.pass_load.current_Fy_N,
                    accumulated_plastic_dissipation_J=accumulated,
                    fields=_read_only_fields(fields),
                    is_final_unloaded_target=index == len(moving.positions) - 1,
                )
            )
        if progress is not None:
            progress(index + 1, len(moving.positions), f"已收敛并提交位置 {index + 1}/{len(moving.positions)}")
    if np.linalg.norm(current_load) != 0.0:
        raise HistoryPassWorkflowError("complete pass did not reach zero external load")
    final_fields = recover_snapshot_fields(fem_case, prepared, committed); final_position = moving.positions[-1]
    rows.append(_row(position_id=len(moving.positions), motion_coordinate_m=final_position.position.motion_coordinate_m, pass_state="final_unloaded", contact_ratio=0.0, Fx_N=0.0, Fy_N=0.0, assembled_Fx_N=0.0, assembled_Fy_N=0.0, transition_substep_count=0, retry_count=0, maximum_newton_iteration_count=0, accumulated_plastic_dissipation_J=accumulated, target_load_vector_N=zero, state=committed, fields=final_fields))
    if not math.isclose(committed.external_force_x_N, 0.0, abs_tol=1.0e-10) or not math.isclose(committed.external_force_y_N, 0.0, abs_tol=1.0e-10):
        raise HistoryPassWorkflowError("final unloaded external force is not zero")
    return SharedHistoryAdvanceResult(tuple(records), tuple(rows), final_fields, compute_snapshot_statistics(final_fields), maximum_fields, compute_snapshot_statistics(maximum_fields), accumulated)


def solve_fixed_mesh_elastoplastic_history_pass(
    case: FixedMeshElastoplasticHistoryPassCase,
    workspace: str | Path,
    on_target_snapshot: Callable[[TargetFieldSnapshot], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> FixedMeshElastoplasticHistoryPassResult:
    """Transfer converged displacement and J2 states through one fixed-mesh pass."""

    if not isinstance(case, FixedMeshElastoplasticHistoryPassCase):
        raise TypeError("case must be a FixedMeshElastoplasticHistoryPassCase")
    workspace_path = Path(workspace).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    moving = solve_fixed_mesh_moving_load_pass(case.moving_load_pass, workspace_path / "moving_load")
    reference = case.reference_case
    fem_case = reference.to_elastoplastic_fem_case(peak_force_x_N=0.0, peak_force_y_N=0.0)
    prepared = prepare_elastoplastic_mesh_with_facet_tractions(
        fem_case,
        moving.reference_mesh,
        np.empty(0, dtype=np.int32),
        peak_force_x_N=0.0,
        peak_force_y_N=0.0,
        facet_line_load_x_N_per_m=np.empty(0),
        facet_line_load_y_N_per_m=np.empty(0),
    )
    targets = tuple(assemble_fixed_top_mapping_load_vector(prepared, item.load_mapping) for item in moving.positions)
    shared = advance_prepared_history(
        case, moving, prepared, targets, on_target_snapshot=on_target_snapshot,
        progress=progress,
    )
    return FixedMeshElastoplasticHistoryPassResult(
        case=case,
        moving_load=moving,
        prepared_mesh=prepared,
        position_records=shared.position_records,
        history_rows=shared.history_rows,
        final_fields=shared.final_fields,
        final_statistics=shared.final_statistics,
        maximum_plastic_fields=shared.maximum_plastic_fields,
        maximum_plastic_statistics=shared.maximum_plastic_statistics,
        accumulated_plastic_dissipation_J=shared.accumulated_plastic_dissipation_J,
    )
