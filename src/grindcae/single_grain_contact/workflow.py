"""Trajectory orchestration for the GrindCAE 3.0.0 contact solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import tempfile
from collections.abc import Callable

import numpy as np

from grindcae.elastoplastic.j2 import equivalent_von_mises_stress
from grindcae.mesh import generate_split_top_rectangle_mesh
from grindcae.solver import ImportedMesh, import_gmsh_mesh

from .geometry import (
    RigidAnalyticalGrain,
    RigidCircularArcGrain,
    RigidCircularGrain,
    RigidRoundedWedgeGrain,
)
from .applicability import (
    ApplicabilityDiagnostics,
    ApplicabilityError,
    evaluate_contact_applicability,
)
from .models import SingleGrainContactCase, TrajectoryTarget
from .solver import (
    _assemble_contact,
    _assemble_damage_trial,
    _committed_damage_fields,
    CommittedContactStructureState,
    ConstrainedSeparationEventError,
    ContactConvergenceError,
    ContactSolverError,
    ConvergedContactIncrement,
    PreparedContactMesh,
    advance_augmented_contact_increment,
    advance_normal_contact_increment,
    advance_separation_constrained_increment,
    initial_contact_structure_state,
    locate_constrained_separation_event,
    prepare_contact_mesh,
    rebase_contact_state_after_topology_transition,
)
from .continuous_separation import (
    SolveMode,
    ZeroForceCommitMeasures,
    begin_separating,
    qualifies_for_removed_commit,
    registered_zero_force_thresholds,
)
from .material_topology import MaterialTopologyError, EventTopologyQualificationError, transition_material_topology
from .separation_events import (
    SeparationEventError,
    event_candidate_background_ids,
    event_aware_substep_scale,
    locate_earliest_separation_event,
)
from .inertial_transition import (
    advance_inertial_transition_window,
    build_inertial_transition_state,
    ideal_edge_active_state_from_history,
)


@dataclass(slots=True)
class _SubstepRetryBudget:
    maximum_consecutive_failures: int
    total_failures: int = 0
    consecutive_failures: int = 0

    @property
    def can_retry(self) -> bool:
        return self.consecutive_failures < self.maximum_consecutive_failures

    def record_failure(self) -> None:
        self.total_failures += 1
        self.consecutive_failures += 1

    def record_successful_commit(self) -> None:
        self.consecutive_failures = 0


class EventPathQualificationError(ContactSolverError):
    """A converged event violates the retained surface-release contract."""

    retryable = False

    def __init__(self, event_marker, rollback_marker=None, candidates=None):
        if isinstance(event_marker, str) and rollback_marker is None and candidates is None:
            super().__init__(event_marker)
            return
        self.event_marker = float(event_marker)
        self.rollback_marker = float(rollback_marker)
        self.candidate_background_ids = tuple(candidates)
        self.failure_stage = 'event_topology_qualification'
        super().__init__(
            f'converged damage event {self.candidate_background_ids} at {self.event_marker} '
            f'cannot be released; trial discarded, rollback marker {self.rollback_marker}'
        )

    def __reduce__(self):
        state = (self.__dict__, self.__cause__, self.__suppress_context__)
        return (type(self), (self.event_marker, self.rollback_marker,
                             self.candidate_background_ids), state)

    def __setstate__(self, state):
        if isinstance(state, dict):
            # BaseException format used by already saved development failures.
            self.__dict__.update(state)
        else:
            attributes, cause, suppress_context = state
            self.__dict__.update(attributes)
            self.__cause__ = cause
            self.__suppress_context__ = suppress_context


def _qualified_event_topology(imported_mesh, topology, state, candidates, committed):
    try:
        return transition_material_topology(imported_mesh, topology, candidates)
    except EventTopologyQualificationError as exc:
        raise EventPathQualificationError(state.marker, committed.marker, candidates) from exc


@dataclass(frozen=True, slots=True)
class ContactTrajectoryRecord:
    index: int
    segment: str
    center_x_m: float
    center_y_m: float
    retry_count: int
    substep_count: int
    normal_reaction_N: float
    tangential_reaction_N: float
    normal_reaction_per_thickness_N_per_m: float
    tangential_reaction_per_thickness_N_per_m: float
    contact_count: int
    open_count: int
    stick_count: int
    slip_count: int
    maximum_pressure_Pa: float
    maximum_penetration_m: float
    newton_iterations: int
    balance_residual_N: float
    friction_dissipation_increment_J: float
    contact_work_increment_J: float
    plastic_dissipation_increment_J: float
    damage_dissipation_increment_J: float
    separation_event_energy_increment_J: float
    elastic_strain_energy_J: float
    numerical_contact_stored_energy_J: float
    cumulative_contact_work_J: float
    cumulative_friction_dissipation_J: float
    cumulative_plastic_dissipation_J: float
    cumulative_damage_dissipation_J: float
    cumulative_separation_event_energy_J: float
    energy_balance_residual_J: float
    applicability: ApplicabilityDiagnostics
    state: ConvergedContactIncrement
    prepared_mesh: PreparedContactMesh | None = None
    material_topology: object | None = None
    hardening_stored_energy_J: float = 0.0


@dataclass(frozen=True, slots=True)
class ContactTrajectoryResult:
    case: SingleGrainContactCase
    prepared_mesh: PreparedContactMesh
    records: tuple[ContactTrajectoryRecord, ...]
    reference_mesh_path: Path | None
    normal_penalty_Pa_per_m: float
    tangential_penalty_Pa_per_m: float
    friction_dissipation_J: float
    contact_work_J: float
    plastic_dissipation_J: float
    damage_dissipation_J: float
    separation_event_energy_J: float
    initial_elastic_strain_energy_J: float
    final_elastic_strain_energy_J: float
    initial_numerical_contact_stored_energy_J: float
    final_numerical_contact_stored_energy_J: float
    energy_balance_residual_J: float
    energy_balance_relative_residual: float
    maximum_balance_residual_N: float
    applicability_status: str
    applicability: ApplicabilityDiagnostics
    material_topology: object | None = None
    background_mesh: ImportedMesh | None = None
    separation_events: tuple[object, ...] = ()
    automatic_removed_element_ids: tuple[int, ...] = ()
    damage_history: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class DamageHistoryRecord:
    record_index: int
    damaged_element_count: int
    maximum_damage: float
    mean_damage: float
    damage_dissipation_J: float
    separation_event_count: int


@dataclass(frozen=True, slots=True)
class SeparationEventRecord:
    event_index: int
    target_index: int
    trajectory_fraction: float
    marker: float
    background_element_ids: tuple[int, ...]
    removed_area_m2: float
    active_degrees_of_freedom_before: int
    active_degrees_of_freedom_after: int
    free_surface_edge_count_before: int
    free_surface_edge_count_after: int
    damage_dissipation_increment_J: float


def _grain(case: SingleGrainContactCase, target: TrajectoryTarget) -> RigidAnalyticalGrain:
    if case.grain.type == "rounded_wedge":
        return RigidRoundedWedgeGrain(
            tip_radius_m=case.grain.tip_radius_m,
            wedge_height_m=case.grain.wedge_height_m,
            intrinsic_rake_angle_rad=case.grain.intrinsic_rake_angle_rad,
            intrinsic_clearance_angle_rad=case.grain.intrinsic_clearance_angle_rad,
            pose_angle_rad=case.grain.pose_angle_rad,
            reference_x_m=target.reference_x_m,
            reference_y_m=target.reference_y_m,
        )
    if case.grain.type == "rounded_circle":
        return RigidCircularGrain(
            radius_m=case.grain.radius_m,
            center_x_m=target.reference_x_m,
            center_y_m=target.reference_y_m,
        )
    if case.grain.type == "rigid_circular_arc":
        assert case.grain.start_angle_rad is not None and case.grain.end_angle_rad is not None
        return RigidCircularArcGrain(
            radius_m=case.grain.radius_m,
            center_x_m=target.center_x_m,
            center_y_m=target.center_y_m,
            start_angle_rad=case.grain.start_angle_rad,
            end_angle_rad=case.grain.end_angle_rad,
        )
    return RigidCircularGrain(
        radius_m=case.grain.radius_m,
        center_x_m=(target.center_x_m if target.center_x_m is not None else target.reference_x_m),
        center_y_m=(target.center_y_m if target.center_y_m is not None else target.reference_y_m),
    )


def _record(
    case: SingleGrainContactCase,
    index: int,
    target: TrajectoryTarget,
    state: ConvergedContactIncrement,
    *,
    retry_count: int,
    substep_count: int,
    previous_state: CommittedContactStructureState | ConvergedContactIncrement | None,
    prepared: PreparedContactMesh,
    applicability: ApplicabilityDiagnostics,
    contact_work_increment_J: float | None = None,
    friction_dissipation_increment_J: float | None = None,
    plastic_dissipation_increment_J: float | None = None,
    normal_penalty_Pa_per_m: float,
    tangential_penalty_Pa_per_m: float,
    baseline_mechanical_energy_J: float | None,
    cumulative_contact_work_J: float,
    cumulative_friction_dissipation_J: float,
    cumulative_plastic_dissipation_J: float,
    cumulative_damage_dissipation_J: float = 0.0,
    cumulative_separation_event_energy_J: float = 0.0,
    damage_dissipation_increment_J: float = 0.0,
    separation_event_energy_increment_J: float = 0.0,
) -> ContactTrajectoryRecord:
    statuses = [point.status for point in state.contact_states]
    reaction = np.asarray(state.rigid_grain_reaction_N, dtype=float)
    balance = math.hypot(state.balance_residual_x_N, state.balance_residual_y_N)
    if previous_state is None or all(
        value is not None
        for value in (
            contact_work_increment_J,
            friction_dissipation_increment_J,
            plastic_dissipation_increment_J,
        )
    ):
        calculated = (0.0, 0.0, 0.0)
    else:
        calculated = _increment_energies(case, prepared, previous_state, state)
    contact_work = calculated[0] if contact_work_increment_J is None else contact_work_increment_J
    friction_dissipation = calculated[1] if friction_dissipation_increment_J is None else friction_dissipation_increment_J
    plastic_dissipation = calculated[2] if plastic_dissipation_increment_J is None else plastic_dissipation_increment_J
    elastic_energy = _elastic_strain_energy(case, prepared, state)
    hardening_energy = _hardening_stored_energy(case, prepared, state)
    contact_stored_energy = _numerical_contact_stored_energy(
        case,
        prepared,
        state,
        normal_penalty_Pa_per_m=normal_penalty_Pa_per_m,
        tangential_penalty_Pa_per_m=tangential_penalty_Pa_per_m,
    )
    mechanical_energy = (elastic_energy + hardening_energy + contact_stored_energy
                         + float(getattr(state, 'kinetic_energy_J', 0.0)))
    baseline_energy = mechanical_energy if baseline_mechanical_energy_J is None else baseline_mechanical_energy_J
    energy_balance_residual = cumulative_contact_work_J - (
        mechanical_energy
        - baseline_energy
        + cumulative_friction_dissipation_J
        + cumulative_plastic_dissipation_J
        + cumulative_damage_dissipation_J
    )
    return ContactTrajectoryRecord(
        index=index,
        segment=target.segment,
        center_x_m=(
            target.center_x_m
            if target.center_x_m is not None
            else target.reference_x_m
        ),
        center_y_m=(
            target.center_y_m
            if target.center_y_m is not None
            else target.reference_y_m
        ),
        retry_count=retry_count,
        substep_count=substep_count,
        normal_reaction_N=float(reaction[1]),
        tangential_reaction_N=float(reaction[0]),
        normal_reaction_per_thickness_N_per_m=float(reaction[1] / case.analysis.thickness),
        tangential_reaction_per_thickness_N_per_m=float(reaction[0] / case.analysis.thickness),
        contact_count=sum(status != "open" for status in statuses),
        open_count=statuses.count("open"),
        stick_count=statuses.count("stick"),
        slip_count=statuses.count("slip"),
        maximum_pressure_Pa=(normal_penalty_Pa_per_m*state.maximum_penetration_m
            if getattr(prepared,'circle_edge_quadrature_order',0) else float(np.max(state.contact_pressure_Pa))),
        maximum_penetration_m=state.maximum_penetration_m,
        newton_iterations=state.newton_iterations,
        balance_residual_N=balance,
        friction_dissipation_increment_J=friction_dissipation,
        contact_work_increment_J=contact_work,
        plastic_dissipation_increment_J=plastic_dissipation,
        damage_dissipation_increment_J=damage_dissipation_increment_J,
        separation_event_energy_increment_J=separation_event_energy_increment_J,
        elastic_strain_energy_J=elastic_energy,
        hardening_stored_energy_J=hardening_energy,
        numerical_contact_stored_energy_J=contact_stored_energy,
        cumulative_contact_work_J=cumulative_contact_work_J,
        cumulative_friction_dissipation_J=cumulative_friction_dissipation_J,
        cumulative_plastic_dissipation_J=cumulative_plastic_dissipation_J,
        cumulative_damage_dissipation_J=cumulative_damage_dissipation_J,
        cumulative_separation_event_energy_J=cumulative_separation_event_energy_J,
        energy_balance_residual_J=energy_balance_residual,
        applicability=applicability,
        state=state,
        prepared_mesh=prepared,
    )


def _verify_segment_history(previous,state):
    """Check surviving background histories before booking any segment energy."""
    if not getattr(previous,'damage_states',()):
        return
    old={int(identity):(material,damage) for identity,material,damage in zip(
        previous.background_element_ids,previous.element_states,previous.damage_states,strict=True)}
    def decreased(a,b):
        return b<a-64*np.finfo(float).eps*max(abs(a),abs(b),np.finfo(float).tiny)
    for identity,material,damage in zip(state.background_element_ids,state.element_states,state.damage_states,strict=True):
        if int(identity) not in old:
            raise ContactSolverError('irreversible segment introduces unknown material identity')
        old_material,old_damage=old[int(identity)]
        if (decreased(old_material.equivalent_plastic_strain,material.equivalent_plastic_strain)
            or decreased(old_damage.damage,damage.damage)
            or decreased(old_damage.damage_dissipation_density_J_per_m3,damage.damage_dissipation_density_J_per_m3)):
            raise ContactSolverError(f'irreversible material history decreased for background {int(identity)}')


def _increment_energies(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    previous_state: CommittedContactStructureState | ConvergedContactIncrement,
    state: ConvergedContactIncrement,
) -> tuple[float, float, float]:
    _verify_segment_history(previous_state,state)
    previous_center = getattr(previous_state, "grain_reference_m", None)
    if previous_center is None:
        previous_center = getattr(previous_state, "grain_center_m", None)
    contact_work = 0.0
    if previous_center is not None:
        state_reference = getattr(state, "grain_reference_m", None)
        if state_reference is None:
            state_reference = state.grain_center_m
        grain_increment = np.asarray(state_reference) - np.asarray(previous_center)
        previous_reaction = np.asarray(
            getattr(previous_state, "rigid_grain_reaction_N", np.zeros(2)), dtype=float
        )
        average_reaction = 0.5 * (
            previous_reaction + np.asarray(state.rigid_grain_reaction_N, dtype=float)
        )
        contact_work = float(np.dot(average_reaction, -grain_increment))
    previous_by_background = {
        int(background): material
        for background, material in zip(
            previous_state.background_element_ids,
            previous_state.element_states,
            strict=True,
        )
    }
    previous_damage_by_background = (
        {
            int(background): damage
            for background, damage in zip(
                previous_state.background_element_ids,
                previous_state.damage_states,
                strict=True,
            )
        }
        if case.damage is not None
        else {}
    )
    plastic_dissipation = 0.0
    for local_index, (background, new, area) in enumerate(zip(
        state.background_element_ids,
        state.element_states,
        prepared.structural.element_areas_m2,
        strict=True,
    )):
        old = previous_by_background.get(int(background))
        if old is None:
            continue
        delta_ep = max(0.0, new.equivalent_plastic_strain - old.equivalent_plastic_strain)
        old_stress = old.current_yield_strength_Pa
        new_stress = new.current_yield_strength_Pa
        plastic_delta_for_energy = delta_ep
        if case.damage is not None:
            old_damage = previous_damage_by_background.get(int(background))
            new_damage = state.damage_states[local_index]
            if (case.damage.evolution == "energetic_rational_fracture"
                    or getattr(new_damage, "energetic_history", None) is not None):
                if old_damage is None:
                    raise ContactSolverError("energetic plastic work lacks accepted damage history")
                material = case.to_elastoplastic_fem_case().material
                plastic_dissipation += (float(area)*case.analysis.thickness
                    * material.yield_strength*delta_ep
                    * (1.0-0.5*(old_damage.damage+new_damage.damage)))
                continue
            if old_damage is not None and new_damage.initiation_equivalent_plastic_strain is not None:
                pre_damage_end = min(
                    new.equivalent_plastic_strain,
                    new_damage.initiation_equivalent_plastic_strain,
                )
                plastic_delta_for_energy = max(
                    0.0,
                    pre_damage_end - old.equivalent_plastic_strain,
                )
        plastic_dissipation += (
            float(area)
            * case.analysis.thickness
            * 0.5
            * (old_stress + new_stress)
            * plastic_delta_for_energy
        )
    return (
        contact_work,
        state.friction_dissipation_increment_J,
        plastic_dissipation,
    )


def _damage_dissipation_increment_J(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    previous_state: CommittedContactStructureState | ConvergedContactIncrement,
    state: CommittedContactStructureState | ConvergedContactIncrement,
) -> float:
    previous_by_background = {
        int(background): damage.damage_dissipation_density_J_per_m3
        for background, damage in zip(
            previous_state.background_element_ids,
            previous_state.damage_states,
            strict=True,
        )
    }
    increment = math.fsum(
        max(
            0.0,
            damage.damage_dissipation_density_J_per_m3
            - previous_by_background.get(int(background), 0.0),
        )
        * float(area)
        * case.analysis.thickness
        for background, damage, area in zip(
            state.background_element_ids,
            state.damage_states,
            prepared.structural.element_areas_m2,
            strict=True,
        )
    )
    if not math.isfinite(increment) or increment < 0.0:
        raise ContactConvergenceError(
            "damage dissipation increment is invalid",
            attempted_marker=state.marker,
            residual_norm_history_N=getattr(state, "residual_norm_history_N", ()),
        )
    return increment


def _separation_event_energy_J(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    state: ConvergedContactIncrement,
    background_element_ids: tuple[int, ...],
) -> float:
    requested = set(background_element_ids)
    return math.fsum(
        damage.damage_dissipation_density_J_per_m3
        * float(area)
        * case.analysis.thickness
        for background, damage, area in zip(
            state.background_element_ids,
            state.damage_states,
            prepared.structural.element_areas_m2,
            strict=True,
        )
        if int(background) in requested
    )


def _hardening_stored_energy(case, prepared, state):
    """Explicit energetic-hardening storage; legacy energy contracts stay isolated."""
    if case.damage is None:
        return 0.0
    material = case.to_elastoplastic_fem_case().material
    return math.fsum(
        0.5*material.internal_hardening_modulus*point.equivalent_plastic_strain**2
        * (1.0-damage.damage)*float(area)*case.analysis.thickness
        for point, damage, area in zip(state.element_states,state.damage_states,
            prepared.structural.element_areas_m2,strict=True)
        if (case.damage.evolution == "energetic_rational_fracture"
            or getattr(damage,"energetic_history",None) is not None)
    )


def _elastic_strain_energy(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    state: ConvergedContactIncrement,
) -> float:
    if case.damage is not None and len(state.damage_states) != len(state.element_states):
        raise ContactConvergenceError(
            "elastic strain energy damage-state count is inconsistent",
            attempted_marker=state.marker,
            residual_norm_history_N=state.residual_norm_history_N,
        )
    total = 0.0
    for element_index, (strain_vector, material_state, area) in enumerate(zip(
        state.element_total_strain,
        state.element_states,
        prepared.structural.element_areas_m2,
        strict=True,
    )):
        total_strain = np.array(
            [
                [strain_vector[0], 0.5 * strain_vector[2], 0.0],
                [0.5 * strain_vector[2], strain_vector[1], 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        elastic_strain = total_strain - np.asarray(
            material_state.plastic_strain_tensor, dtype=float
        )
        effective_stress = np.asarray(material_state.stress_tensor_Pa, dtype=float)
        damage = (
            state.damage_states[element_index].damage
            if case.damage is not None
            else 0.0
        )
        stress = (1.0 - damage) * effective_stress
        total += (
            0.5
            * float(np.sum(stress * elastic_strain))
            * float(area)
            * case.analysis.thickness
        )
    if not math.isfinite(total):
        raise ContactConvergenceError(
            "elastic strain energy is non-finite",
            attempted_marker=state.marker,
            residual_norm_history_N=state.residual_norm_history_N,
        )
    return max(0.0, total)


def _numerical_contact_stored_energy(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    state: ConvergedContactIncrement,
    *,
    normal_penalty_Pa_per_m: float,
    tangential_penalty_Pa_per_m: float,
) -> float:
    if getattr(prepared,'circle_edge_quadrature_order',0):
        from .solver import _circle_edge_integrals
        if case.contact.normal_algorithm!='penalty' or case.contact.friction_coefficient!=0.:
            raise ContactSolverError('circle edge prototype requires frictionless penalty contact')
        target=replace(case.trajectory.targets[0],reference_x_m=state.grain_reference_m[0],
                       reference_y_m=state.grain_reference_m[1])
        return math.fsum(value.energy_J for _,value in _circle_edge_integrals(
            case.to_elastoplastic_fem_case(),prepared,state.displacement_vector_m,
            _grain(case,target),normal_penalty_Pa_per_m))
    if getattr(prepared,'edge_endpoint_keys',()):
        nodes={int(n):i for i,n in enumerate(prepared.candidate_background_node_ids)}
        histories=dict(state.edge_contact_states)
        if set(histories)!=set(prepared.edge_endpoint_keys):
            raise ContactSolverError('edge contact energy history is incomplete')
        total=0.
        for a,b,n in prepared.edge_endpoint_keys:
            weight=.5*float(np.linalg.norm(prepared.candidate_reference_coordinates_m[nodes[a]]-
                                           prepared.candidate_reference_coordinates_m[nodes[b]]))
            q=state.contact_kinematics[nodes[n]]; h=histories[(a,b,n)]
            total+=weight*(.5*normal_penalty_Pa_per_m*q.penetration_m**2+
                (.5*h.tangential_traction_Pa**2/tangential_penalty_Pa_per_m if tangential_penalty_Pa_per_m>0 else 0.))
        return total*case.analysis.thickness
    total_per_thickness = 0.0
    for weight, kinematics, contact_state in zip(
        prepared.candidate_tributary_lengths_m,
        state.contact_kinematics,
        state.contact_states,
    ):
        normal_density_J_per_m2 = (
            0.5 * normal_penalty_Pa_per_m * kinematics.penetration_m**2
        )
        tangential_density_J_per_m2 = (
            0.5 * contact_state.tangential_traction_Pa**2 / tangential_penalty_Pa_per_m
            if tangential_penalty_Pa_per_m > 0.0
            else 0.0
        )
        total_per_thickness += float(weight) * (
            normal_density_J_per_m2 + tangential_density_J_per_m2
        )
    total = total_per_thickness * case.analysis.thickness
    if not math.isfinite(total) or total < 0.0:
        raise ContactConvergenceError(
            "numerical contact stored energy is invalid",
            attempted_marker=state.marker,
            residual_norm_history_N=state.residual_norm_history_N,
        )
    return total


def _applicability(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    state: ConvergedContactIncrement,
) -> ApplicabilityDiagnostics:
    # Rounded-wedge reference coordinates locate its tip, not a circle center.
    # Query the same analytical boundary as contact; retain the circle check.
    geometry = (
        _grain(case, _interpolated_target(state.grain_reference_m, case.trajectory.targets[0], 0.))
        if case.grain.type == 'rounded_wedge' else None
    )
    return evaluate_contact_applicability(
        prepared,
        state.displacement_vector_m,
        grain_radius_m=(case.grain.radius_m or case.grain.tip_radius_m),
        grain_center_m=state.grain_reference_m,
        grain_geometry=geometry,
        warning_displacement_over_contact_size=case.safety.warn_displacement_over_local_size,
        stop_displacement_over_contact_size=case.safety.stop_displacement_over_local_size,
        warning_displacement_over_radius=case.safety.warn_displacement_over_radius,
        stop_displacement_over_radius=case.safety.stop_displacement_over_radius,
        warning_strain=case.safety.warn_strain,
        stop_strain=case.safety.stop_strain,
        raise_on_hard_stop=True,
    )


def _aggregate_applicability(
    records: list[ContactTrajectoryRecord],
) -> ApplicabilityDiagnostics:
    diagnostics = [record.applicability for record in records]
    return ApplicabilityDiagnostics(
        status="warning" if any(item.status == "warning" for item in diagnostics) else "within_range",
        maximum_displacement_m=max(item.maximum_displacement_m for item in diagnostics),
        maximum_displacement_to_contact_size=max(item.maximum_displacement_to_contact_size for item in diagnostics),
        maximum_displacement_to_grain_radius=max(item.maximum_displacement_to_grain_radius for item in diagnostics),
        maximum_displacement_gradient=max(item.maximum_displacement_gradient for item in diagnostics),
        maximum_absolute_principal_strain=max(item.maximum_absolute_principal_strain for item in diagnostics),
        maximum_equivalent_total_strain=max(item.maximum_equivalent_total_strain for item in diagnostics),
        maximum_local_rotation_rad=max(item.maximum_local_rotation_rad for item in diagnostics),
        minimum_area_ratio=min(item.minimum_area_ratio for item in diagnostics),
        minimum_normalized_jacobian_ratio=min(item.minimum_normalized_jacobian_ratio for item in diagnostics),
        surface_self_intersection=any(item.surface_self_intersection for item in diagnostics),
        closest_point_unique=all(item.closest_point_unique for item in diagnostics),
        candidate_boundary_order_preserved=all(item.candidate_boundary_order_preserved for item in diagnostics),
        warning_metrics=tuple(dict.fromkeys(name for item in diagnostics for name in item.warning_metrics)),
        hard_stop_reasons=(),
    )


def _solve_target(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    target: TrajectoryTarget,
    *,
    normal_penalty: float,
    tangential_penalty: float,
    marker: float,
) -> ConvergedContactIncrement:
    grain = _grain(case, target)
    common = dict(
        grain=grain,
        penalty_Pa_per_m=normal_penalty,
        tangential_penalty_Pa_per_m=tangential_penalty,
        friction_coefficient=case.contact.friction_coefficient,
        friction_regularization_ratio=case.contact.friction_regularization_ratio,
        target_marker=marker,
        damage_input=case.damage,
    )
    if case.contact.normal_algorithm == "augmented_lagrangian":
        local_size = float(np.median(prepared.candidate_tributary_lengths_m))
        return advance_augmented_contact_increment(
            case.to_elastoplastic_fem_case(),
            prepared,
            committed,
            relaxation=case.contact.augmented_relaxation,
            maximum_outer_iterations=case.contact.augmented_maximum_iterations,
            penetration_tolerance_m=5.0e-3 * local_size,
            **common,
        )
    return advance_normal_contact_increment(
        case.to_elastoplastic_fem_case(), prepared, committed, **common
    )


def _interpolated_target(
    start: tuple[float, float], target: TrajectoryTarget, fraction: float
) -> TrajectoryTarget:
    if target.reference_x_m is not None:
        return TrajectoryTarget(
            segment=target.segment,
            reference_x_m=start[0] + fraction * (target.reference_x_m - start[0]),
            reference_y_m=start[1] + fraction * (target.reference_y_m - start[1]),
        )
    return TrajectoryTarget(
        segment=target.segment,
        center_x_m=start[0] + fraction * (target.center_x_m - start[0]),
        center_y_m=start[1] + fraction * (target.center_y_m - start[1]),
    )


def _damage_candidates(
    case: SingleGrainContactCase,
    state: ConvergedContactIncrement,
    *,
    accepted_events_at_position: int,
    prepared: PreparedContactMesh | None = None,
) -> tuple[int, ...]:
    if case.damage is None:
        return ()
    candidates = event_candidate_background_ids(
        background_element_ids=state.background_element_ids,
        damage_states=state.damage_states,
        damage_event_tolerance=case.damage.damage_event_tolerance,
        accepted_events_at_position=accepted_events_at_position,
        maximum_events_per_position=case.damage.maximum_separation_events_per_position,
    )
    if state.solve_mode is SolveMode.INERTIAL_TRANSITION and candidates:
        if prepared is None:
            raise ContactSolverError('transient removal candidates require the active mass mesh')
        # An accepted physical step may use Newton's relative contribution.
        # The zero-force transaction has stricter absolute gates. Keep the
        # material and its mass pending until that same state also meets them;
        # do not discard a valid physical step or relax the transaction audit.
        if (state.residual_norm_N > case.newton.residual_absolute_tolerance_N
                or state.correction_norm_m > case.newton.displacement_absolute_tolerance_m):
            return ()
        from .inertial_transition import material_kinetic_energy
        by_id = dict(zip(state.background_element_ids, state.damage_states, strict=True))
        candidates = tuple(i for i in candidates if by_id[i].damage == 1.0)
        energy = material_kinetic_energy(prepared, state, candidates,
            density_kg_per_m3=case.material.density_kg_per_m3, thickness_m=case.analysis.thickness)
        if energy > max(1e-14, 1e-6*state.transient_maximum_energy_scale_J):
            return ()
    return candidates


def _circle_boundary_release_ready(case,prepared,state,new_prepared,*,normal_penalty):
    """Require unloaded retired edges and stress-free newly exposed edges.

    This is a read-only event eligibility test on the current configuration.
    Material, mass and the existing contact boundary continue unchanged while
    release is pending. Contributions cannot cancel between different edges.
    """
    from .edge_contact_history import integrate_circle_penalty_edge
    old_edges={key[:2] for key in prepared.edge_endpoint_keys}
    new_edges={key[:2] for key in new_prepared.edge_endpoint_keys}
    node_index={int(node):index for index,node in enumerate(prepared.active_to_background_node_ids)}
    coordinates=prepared.structural.imported_mesh.mesh.p.T
    displacements=np.asarray(state.displacement_vector_m).reshape(-1,2)
    removed=set(int(i) for i in prepared.active_to_background_element_ids)-set(
        int(i) for i in new_prepared.active_to_background_element_ids)
    fracture_energy=math.fsum(case.damage.fracture_energy_J_per_m2*float(area)*case.analysis.thickness
        /math.sqrt(4.*float(area)/math.pi)
        for i,area in zip(prepared.active_to_background_element_ids,prepared.structural.element_areas_m2,strict=True)
        if int(i) in removed)
    energy_tolerance=max(1e-18,64*np.finfo(float).eps*fracture_energy)
    changed=old_edges.symmetric_difference(new_edges)
    for edge in sorted(changed):
        indices=[node_index[node] for node in edge]
        value=integrate_circle_penalty_edge(coordinates[indices],displacements[indices],
            center=state.grain_reference_m,radius=case.grain.radius_m,penalty=normal_penalty,
            thickness=case.analysis.thickness,order=prepared.circle_edge_quadrature_order)
        if (np.linalg.norm(value.force_N)>case.newton.residual_absolute_tolerance_N
                or value.energy_J>energy_tolerance):
            return False
    return True


def _direct_path_after_newton_failure(case,prepared,committed,*,start,target,achieved,
        marker_offset,normal_penalty,tangential_penalty,initial_failure):
    """Route a failure without using its damage level as a material criterion."""
    target_reference=np.array([target.reference_x_m,target.reference_y_m],dtype=float)
    length=float(np.linalg.norm(target_reference-np.asarray(start,dtype=float)))
    try:
        return locate_constrained_separation_event(case.to_elastoplastic_fem_case(),prepared,
            committed,committed,start_fraction=achieved,seed_fraction=achieved,
            reference_path_length_m=length,
            grain_at_fraction=lambda value:_grain(case,_interpolated_target(start,target,value)),
            marker_offset=marker_offset,normal_penalty_Pa_per_m=normal_penalty,
            tangential_penalty_Pa_per_m=tangential_penalty,
            friction_coefficient=case.contact.friction_coefficient,
            friction_regularization_ratio=case.contact.friction_regularization_ratio,
            damage_input=case.damage,minimum_substep_fraction=case.step_control.minimum_substep_fraction,
            maximum_steps=min(4096,max(128,math.ceil(2./case.step_control.minimum_substep_fraction))))
    except ContactSolverError as error:
        # Routing is not evidence that an augmented correction was executed.
        error.path_entry_executed=getattr(error,'path_entry_executed',False)
        error.path_failure_stage=getattr(error,'path_failure_stage','entry_precheck')
        error.initial_newton_failure=initial_failure
        raise


def _diagnostic_ideal_edge_transition_after_path_failure(
    case,
    prepared,
    committed,
    *,
    start,
    target,
    achieved,
    marker_offset,
    path_failure,
):
    """Advance one bounded real-mass window after the quasi-static route is exhausted."""

    if (
        case.transient is None
        or case.grain.type != "rounded_circle"
        or case.contact.friction_coefficient != 0.0
        or committed.accepted_path_reference is None
        or not prepared.edge_endpoint_keys
    ):
        raise path_failure
    target_reference = np.asarray(
        (target.reference_x_m, target.reference_y_m)
        if target.reference_x_m is not None
        else (target.center_x_m, target.center_y_m),
        dtype=float,
    )
    start_reference = np.asarray(start, dtype=float)
    delta = target_reference - start_reference
    length = float(np.linalg.norm(delta))
    if length <= np.finfo(float).tiny:
        raise path_failure
    direction = delta / length
    displacement_scale = float(np.median(prepared.candidate_tributary_lengths_m))
    mapped = committed.accepted_path_reference.mapped_vector(
        tuple(int(value) for value in prepared.active_to_background_node_ids),
        prepared.structural.free_dofs,
        displacement_scale_m=displacement_scale,
        marker_scale_m=length,
    )
    constraint_keys = tuple(sorted({key[:2] for key in prepared.edge_endpoint_keys}))
    coordinate_by_background = {
        int(background): coordinate
        for background, coordinate in zip(
            prepared.candidate_background_node_ids,
            prepared.candidate_reference_coordinates_m,
            strict=True,
        )
    }
    edge_lengths = np.asarray(
        [
            np.linalg.norm(
                coordinate_by_background[int(second)]
                - coordinate_by_background[int(first)]
            )
            for first, second in constraint_keys
        ],
        dtype=float,
    )
    node_lookup = {
        int(background): index
        for index, background in enumerate(prepared.candidate_background_node_ids)
    }
    constraint_edges = np.asarray(
        [[node_lookup[int(first)], node_lookup[int(second)]] for first, second in constraint_keys],
        dtype=np.int32,
    )
    active, forces = ideal_edge_active_state_from_history(
        edge_endpoint_keys=prepared.edge_endpoint_keys,
        edge_contact_states=committed.edge_contact_states,
        constraint_edge_keys=constraint_keys,
        edge_lengths_m=edge_lengths,
        thickness_m=case.analysis.thickness,
    )
    transition = build_inertial_transition_state(
        case,
        prepared,
        committed,
        center_m=np.asarray(committed.grain_reference_m, dtype=float),
        grain_motion_direction=direction,
        accepted_tangent=np.r_[mapped[:-1], np.zeros(active.size), mapped[-1]],
        active_edges=active,
        normal_forces_N=forces,
        constraint_edges=constraint_edges,
    )
    advanced = advance_inertial_transition_window(
        case,
        prepared,
        committed,
        transition,
        maximum_steps=case.transient.maximum_steps,
        stop_at_terminal_damage=True,
    )
    result = advanced.committed_state
    if not isinstance(result, (CommittedContactStructureState, ConvergedContactIncrement)):
        raise ContactSolverError("inertial transition returned an invalid committed state")
    if not advanced.steps:
        raise ContactSolverError("inertial transition accepted no physical time step")
    step = advanced.steps[-1]
    contact = step.contact
    reference = tuple(float(value) for value in advanced.center_m)
    return ConvergedContactIncrement(
        marker=float(marker_offset + achieved + np.linalg.norm(advanced.center_m-start_reference)/length),
        displacement_vector_m=np.asarray(step.displacement).copy(),
        element_states=step.body.element_states,
        contact_states=tuple(getattr(result, "contact_states", ())),
        contact_kinematics=tuple(getattr(result, "contact_kinematics", ())),
        contact_pressure_Pa=np.asarray(getattr(result, "contact_pressure_Pa", np.empty(0))).copy(),
        contact_force_vector_N=np.asarray(contact.force).copy(),
        internal_force_vector_N=np.asarray(step.body.internal_force_N).copy(),
        element_total_strain=np.asarray(step.body.element_total_strain).copy(),
        element_stress_Pa=np.asarray(step.body.element_stress_Pa).copy(),
        element_sigma_zz_Pa=np.asarray(step.body.element_sigma_zz_Pa).copy(),
        newton_iterations=int(step.body.finite_difference_branch_crossing_count),
        residual_norm_N=float(step.residual_norm_N),
        residual_tolerance_N=float(step.residual_tolerance_N),
        correction_norm_m=float(step.correction_norm_m),
        correction_tolerance_m=float(case.newton.displacement_absolute_tolerance_m),
        reference_force_N=float(max(np.linalg.norm(step.normal_forces_N), case.newton.residual_absolute_tolerance_N)),
        reference_contact_size_m=displacement_scale,
        reference_displacement_m=float(np.linalg.norm(step.displacement)),
        residual_absolute_tolerance_N=float(case.newton.residual_absolute_tolerance_N),
        displacement_absolute_tolerance_m=float(case.newton.displacement_absolute_tolerance_m),
        residual_norm_history_N=(float(step.residual_norm_N),),
        total_contact_force_N=np.asarray(contact.force).reshape(-1,2).sum(axis=0),
        rigid_grain_reaction_N=-np.asarray(contact.force).reshape(-1,2).sum(axis=0),
        fixed_reaction_x_N=0.0,
        fixed_reaction_y_N=0.0,
        balance_residual_x_N=float(np.sum((step.body.internal_force_N-contact.force).reshape(-1,2),axis=0)[0]),
        balance_residual_y_N=float(np.sum((step.body.internal_force_N-contact.force).reshape(-1,2),axis=0)[1]),
        maximum_penetration_m=max(0.0, -float(np.min(contact.all_gaps))),
        grain_center_m=reference,
        candidate_keys=getattr(result, "candidate_keys", ()),
        grain_reference_m=reference,
        friction_dissipation_increment_J=0.0,
        augmented_iterations=0,
        damage_states=step.body.damage_states,
        background_element_ids=np.asarray(result.background_element_ids).copy(),
        solve_mode=SolveMode.INERTIAL_TRANSITION,
        solve_failure=None,
        separating_states=getattr(result, "separating_states", ()),
        edge_contact_states=getattr(result, "edge_contact_states", ()),
        archived_edge_contact_states=getattr(result, "archived_edge_contact_states", ()),
        accepted_path_reference=getattr(result, "accepted_path_reference", None),
        physical_time_s=float(advanced.dynamic_state.time_s),
        velocity_vector_m_per_s=np.asarray(advanced.dynamic_state.velocity_m_per_s).copy(),
        kinetic_energy_J=float(step.kinetic_energy_J),
    )


def _transient_local_error_allowance(*,used_J,scale_J,dt_s,remaining_time_s):
    """Budget allocation for smooth static steps; not a transient minimum gate."""
    if not all(math.isfinite(v) for v in (used_J,scale_J,dt_s,remaining_time_s)) or min(dt_s,remaining_time_s)<=0.:
        raise ValueError('invalid transient error budget time interval')
    return max(0.,max(1e-10,1e-5*scale_J)-used_J)*min(1.,dt_s/remaining_time_s)


def _transient_step_error_allowance(*,used_J,scale_J,accepted_steps,maximum_steps):
    if type(accepted_steps) is not int or type(maximum_steps) is not int or not 0<=accepted_steps<maximum_steps:
        raise ValueError('invalid remaining transient step budget')
    return max(0.,max(1e-10,1e-5*scale_J)-used_J)/(maximum_steps-accepted_steps)


def _audit_transient_commit_window(previous,result,*,work_J,dissipation_J,stored_before_J,stored_after_J,
                                   topology_error_J=0.):
    """Apply the registered budget to the complete physical/topology window."""
    error=work_J-(stored_after_J-stored_before_J)-dissipation_J
    scale=max(previous.transient_maximum_energy_scale_J,result.transient_maximum_energy_scale_J,
              abs(work_J),abs(dissipation_J),abs(stored_before_J),abs(stored_after_J))
    absolute=max(result.transient_absolute_energy_error_J+abs(topology_error_J),
                 previous.transient_absolute_energy_error_J+abs(error))
    if not all(math.isfinite(v) for v in (error,scale,absolute)) or absolute>max(1e-10,1e-5*scale):
        raise ContactSolverError('complete transient topology window exceeds physical energy budget')
    return replace(result,transient_absolute_energy_error_J=absolute,transient_maximum_energy_scale_J=scale)


def _update_inertial_recovery(prepared, previous, result, *, case=None):
    """Recover only after eight accepted quiet, statically balanced steps."""
    def activity(state):
        integrated_activity = ()
        if getattr(prepared, 'circle_edge_quadrature_order', 0):
            if case is None:
                raise ContactSolverError('integrated contact recovery requires circle geometry')
            from .solver import _circle_edge_integrals
            target = replace(case.trajectory.targets[0],
                reference_x_m=state.grain_reference_m[0], reference_y_m=state.grain_reference_m[1])
            edges = sorted({key[:2] for key in prepared.edge_endpoint_keys})
            # Moving patch endpoints are continuous coordinates; only edge
            # activity and clipped endpoint sides define the active set.
            integrated_activity = tuple(
                (edge, value.active_interval[0] == 0., value.active_interval[1] == 1.)
                for edge, (_, value) in zip(edges, _circle_edge_integrals(
                    case.to_elastoplastic_fem_case(), prepared, state.displacement_vector_m,
                    _grain(case, target), 1.), strict=True)
                if value.active_interval is not None)
        return (tuple((int(i),d.status,d.damage) for i,d in zip(
                    state.background_element_ids,state.damage_states,strict=True)),
                tuple((key,value.status) for key,value in state.edge_contact_states),
                tuple((key,value.status) for key,value in zip(state.candidate_keys,state.contact_states,strict=True)),
                integrated_activity)
    static_residual = result.internal_force_vector_N-result.contact_force_vector_N
    material_history_stable = (
        len(previous.element_states)==len(result.element_states)
        and all(a.equivalent_plastic_strain==b.equivalent_plastic_strain
                and np.array_equal(a.plastic_strain_tensor,b.plastic_strain_tensor)
                for a,b in zip(previous.element_states,result.element_states,strict=True)))
    stable = (result.physical_time_s>previous.physical_time_s
        and result.kinetic_energy_J<=max(1e-14,1e-6*result.transient_maximum_energy_scale_J)
        and np.linalg.norm(static_residual[prepared.structural.free_dofs])<=result.residual_tolerance_N
        and result.correction_norm_m<=result.correction_tolerance_m
        and activity(result)==activity(previous) and material_history_stable)
    count = min(8,getattr(previous,'transient_stable_step_count',0)+1) if stable else 0
    return replace(result,transient_stable_step_count=count,
        solve_mode=SolveMode.STANDARD_NEWTON if count==8 else SolveMode.INERTIAL_TRANSITION)


def _locate_inertial_yield_entry(case, prepared, previous, result, *, dt, solve_at):
    """Split elastic reloads at yield using uncommitted physical momentum solves.

    The elastic predictor is evaluated on each solved displacement, never on
    an interpolated displacement or a modified material history. Crossing a
    yield corner with one trapezoid mixes the elastic and plastic branches.
    """
    from scipy.optimize import brentq
    material=case.to_elastoplastic_fem_case().material
    candidates=[]
    for i,(old,new,damage) in enumerate(zip(previous.element_states,result.element_states,
                                           previous.damage_states,strict=True)):
        tolerance=max(1e-7,1e-12*old.current_yield_strength_Pa)
        if (damage.damage<1. and old.yield_function_Pa < -tolerance
                and new.equivalent_plastic_strain>old.equivalent_plastic_strain):
            candidates.append(i)
    if not candidates:return dt,result
    cache={dt:result}
    def solved(time):
        if time not in cache:cache[time]=solve_at(time)
        return cache[time]
    def predictor(time,index):
        old=previous.element_states[index]
        if time==0.:return old.yield_function_Pa
        trial=solved(time)
        strain=prepared.structural.element_B_matrices_per_m[index]@(
            trial.displacement_vector_m-previous.displacement_vector_m)[
                prepared.structural.element_dofs[index]]
        tensor=np.array([[strain[0],.5*strain[2],0.],
                         [.5*strain[2],strain[1],0.],[0.,0.,0.]])
        trace=float(np.trace(tensor))
        stress=np.asarray(old.stress_tensor_Pa)+material.bulk_modulus*trace*np.eye(3)+(
            2.*material.shear_modulus*(tensor-trace/3.*np.eye(3)))
        return equivalent_von_mises_stress(stress)-old.current_yield_strength_Pa
    earliest=dt
    minimum=case.transient.minimum_time_step_s
    for index in candidates:
        # An event before the registered minimum cannot create a hidden
        # subminimum commit; retain the ordinary physical energy rejection.
        if minimum>=earliest or predictor(minimum,index)>=0.:continue
        if predictor(earliest,index)<=0.:continue
        root=brentq(lambda time:predictor(time,index),minimum,earliest,
                    xtol=4*np.finfo(float).eps*dt,rtol=8*np.finfo(float).eps)
        earliest=min(earliest,root)
    return earliest,solved(earliest)


def _locate_inertial_damage_terminal(case,prepared,previous,result,*,dt,solve_at):
    """Locate Y=R(1) with unchanged history before zero-capacity freezing."""
    from scipy.optimize import brentq
    from .thermodynamic_energy import stored_energy,fracture_resistance
    indices=[i for i,(a,b) in enumerate(zip(previous.damage_states,result.damage_states,strict=True))
             if a.damage<1. and b.damage==1. and b.energetic_history is not None]
    if not indices:return dt,result
    material=case.to_elastoplastic_fem_case().material
    cache={dt:result}
    def solved(time):
        if time not in cache:cache[time]=solve_at(time)
        return cache[time]
    def surface(time,index):
        trial=solved(time);h=trial.damage_states[index].energetic_history
        if h is None:
            # No initiation yet: no terminal surface exists on this side.
            return -1.
        point=trial.element_states[index]
        driving=stored_energy(point.stress_tensor_Pa,point.equivalent_plastic_strain,0.,
            young_modulus=material.E,poisson_ratio=material.nu,
            hardening_modulus=material.internal_hardening_modulus)['effective_total']
        resistance=fracture_resistance(initial_driving_energy=h.initial_driving_energy,
            fracture_energy_density=h.fracture_energy_density).resistance(1.)
        return (driving-resistance)/h.initial_driving_energy
    minimum=case.transient.minimum_time_step_s;earliest=dt
    for index in indices:
        if minimum>earliest:
            raise ContactConvergenceError('damage terminal event precedes registered minimum time step',
                attempted_marker=result.marker,residual_norm_history_N=result.residual_norm_history_N)
        low=surface(minimum,index)
        if low>1e-10:
            error = ContactConvergenceError('damage terminal event precedes registered minimum time step',
                attempted_marker=result.marker,residual_norm_history_N=result.residual_norm_history_N)
            # The minimum itself was solved on the upper side. Retrying a
            # larger time cannot create a legal earlier terminal event from
            # this same history; preserve the refusal without another chain.
            error.registered_minimum_exhausted = True
            raise error
        if surface(earliest,index)<0.:continue
        root=(minimum if abs(low)<=1e-10 else brentq(lambda time:surface(time,index),minimum,earliest,
              xtol=4*np.finfo(float).eps*dt,rtol=8*np.finfo(float).eps))
        # A root rounded to the lower side must not be forced to D=1. Find
        # the first upper-side *solved* time, then recheck its physical surface.
        upper=earliest;lower=root
        if solved(root).damage_states[index].damage<1.:
            for _ in range(64):
                middle=.5*(lower+upper)
                if middle==lower or middle==upper:break
                if solved(middle).damage_states[index].damage==1.:upper=middle
                else:lower=middle
            root=upper
        if abs(surface(root,index))>1e-10:
            raise ContactConvergenceError('damage terminal physical surface localization did not converge',
                attempted_marker=result.marker,residual_norm_history_N=result.residual_norm_history_N)
        earliest=min(earliest,root)
    return earliest,solved(earliest)


def _inertial_transition_after_path_failure(
    case, prepared, committed, *, start, target, achieved, marker_offset,
    path_failure, normal_penalty=None, tangential_penalty=None,
):
    """One physical step using the accepted contact representation and history.

    A step returns its actual marker; the trajectory owns prefix accounting,
    topology transactions and subsequent steps. The ideal-edge/impact adapter
    remains an independent diagnostic and is never a loaded-state conversion.
    """
    from .normal_constraint_path import _inertial_material_energy_increment
    from .thermodynamic_energy import DamagePathConstraintRequired
    if case.transient is None or case.damage is None:
        raise path_failure
    if getattr(committed,'transient_step_count',0)>=case.transient.maximum_steps:
        raise ContactSolverError('physical transient maximum_steps exhausted')
    start_reference = np.asarray(start, dtype=float)
    target_reference = np.asarray(
        (target.reference_x_m, target.reference_y_m) if target.reference_x_m is not None
        else (target.center_x_m, target.center_y_m), dtype=float)
    center = np.asarray(committed.grain_reference_m, dtype=float)
    segment = target_reference-start_reference
    length = float(np.linalg.norm(segment))
    remaining = float(np.linalg.norm(target_reference-center))
    if length <= 0.0 or remaining <= 0.0:
        raise path_failure
    direction = segment/length
    local_size = float(np.median(prepared.candidate_tributary_lengths_m))
    if normal_penalty is None:
        normal_penalty = case.contact.normal_penalty_factor*case.effective_modulus_Pa/local_size
    if tangential_penalty is None:
        tangential_penalty = case.contact.tangential_penalty_ratio*normal_penalty
    velocity = np.asarray(getattr(committed, 'velocity_vector_m_per_s', ()), dtype=float)
    missing_velocity = velocity.size == 0
    if velocity.size == 0:
        velocity = np.zeros(prepared.total_degrees_of_freedom)
    speed = case.transient.grain_speed_m_per_s
    time_to_target = remaining/speed
    dt = min(case.transient.initial_time_step_s, time_to_target)
    if getattr(committed,'transient_last_time_step_s',0.)>0.:
        dt=min(dt,2.*committed.transient_last_time_step_s)
    fem = case.to_elastoplastic_fem_case()
    source_grain = _grain(case, _interpolated_target(tuple(center), target, 0.0))
    contact0 = _assemble_contact(fem, prepared, committed, committed.displacement_vector_m,
        source_grain, normal_penalty, tangential_penalty,
        case.contact.friction_coefficient, case.contact.friction_regularization_ratio)
    if missing_velocity and (np.any(committed.displacement_vector_m) or np.any(contact0.force_N)):
        from .solver import close_path_entry_material_side
        from .inertial_transition import assemble_consistent_mass_matrix
        try:
            velocity=close_path_entry_material_side(fem,prepared,committed,grain=source_grain,
                grain_direction_m=direction,penalty=normal_penalty,tangential=tangential_penalty,
                friction_coefficient=case.contact.friction_coefficient,
                regularization=case.contact.friction_regularization_ratio,damage_input=case.damage)*speed
        except ContactSolverError as error:
            raise ContactSolverError('loaded inertial entry lacks an accepted physical velocity') from error
        mass=assemble_consistent_mass_matrix(prepared.structural,
            density_kg_per_m3=case.material.density_kg_per_m3,thickness_m=case.analysis.thickness)
        committed=replace(committed,velocity_vector_m_per_s=velocity,
            kinetic_energy_J=.5*float(velocity@mass@velocity))
    from .thermodynamic_energy import stored_energy
    initial_material_storage = math.fsum(
        float(area)*case.analysis.thickness*stored_energy(
            point.stress_tensor_Pa, point.equivalent_plastic_strain, damage.damage,
            young_modulus=fem.material.E, poisson_ratio=fem.material.nu,
            hardening_modulus=fem.material.internal_hardening_modulus)[
                'stored_total' if case.damage.evolution=='energetic_rational_fracture'
                or getattr(damage,'energetic_history',None) is not None else 'stored_elastic']
        for area, point, damage in zip(prepared.structural.element_areas_m2,
            committed.element_states, committed.damage_states, strict=True))
    # Retry and event localization share one immutable committed state and
    # contact representation. Reuse solved endpoints at identical physical
    # times; none of these trials is committed until all gates below pass.
    solved_trials = {}
    while True:
        next_center = target_reference.copy() if dt == time_to_target else center+dt*speed*direction
        next_target = _interpolated_target(tuple(next_center), target, 0.0)
        next_marker = marker_offset + float(np.dot(next_center-start_reference, direction))/length
        try:
            advance = advance_normal_contact_increment
            outer_controls = {}
            if case.contact.normal_algorithm == 'augmented_lagrangian':
                advance = advance_augmented_contact_increment
                outer_controls = dict(relaxation=case.contact.augmented_relaxation,
                    maximum_outer_iterations=case.contact.augmented_maximum_iterations,
                    penetration_tolerance_m=5e-3*local_size)
            def solve_at(time):
                if time in solved_trials:
                    return solved_trials[time]
                position=target_reference.copy() if time==time_to_target else center+time*speed*direction
                marker=marker_offset+float(np.dot(position-start_reference,direction))/length
                trial = advance(fem, prepared, committed,
                    grain=_grain(case,_interpolated_target(tuple(position),target,0.)),
                    penalty_Pa_per_m=normal_penalty,tangential_penalty_Pa_per_m=tangential_penalty,
                    friction_coefficient=case.contact.friction_coefficient,
                    friction_regularization_ratio=case.contact.friction_regularization_ratio,
                    target_marker=marker,damage_input=case.damage,
                    time_step_s=time,velocity_m_per_s=velocity,
                    density_kg_per_m3=case.material.density_kg_per_m3,**outer_controls)
                solved_trials[time] = trial
                return trial
            result=solve_at(dt)
            dt,result=_locate_inertial_yield_entry(case,prepared,committed,result,dt=dt,solve_at=solve_at)
            dt,result=_locate_inertial_damage_terminal(case,prepared,committed,result,dt=dt,solve_at=solve_at)
            next_center=np.asarray(result.grain_reference_m)
            next_marker=result.marker
            storage, plastic, damage = _inertial_material_energy_increment(fem, prepared, committed, result,
                evolution=case.damage.evolution)
            source_contact_state = replace(result,
                displacement_vector_m=np.asarray(committed.displacement_vector_m),
                grain_reference_m=tuple(center), contact_kinematics=contact0.kinematics,
                edge_contact_states=contact0.edge_contact_states, contact_states=contact0.states)
            def contact_storage(state):
                return _numerical_contact_stored_energy(case, prepared, state,
                    normal_penalty_Pa_per_m=normal_penalty,
                    tangential_penalty_Pa_per_m=tangential_penalty)
            penalty_change = contact_storage(result)-contact_storage(source_contact_state)
            work = float(0.5*(contact0.force_N+result.contact_force_vector_N).reshape(-1,2).sum(axis=0)
                         @ (next_center-center))
            kinetic_change = result.kinetic_energy_J-float(getattr(committed, 'kinetic_energy_J', 0.0))
            error = work-storage-plastic-damage-penalty_change-kinetic_change-result.friction_dissipation_increment_J
            scale = max(abs(work),abs(storage),abs(plastic),abs(damage),abs(penalty_change),
                        abs(kinetic_change),initial_material_storage,initial_material_storage+storage,
                        result.kinetic_energy_J,
                        float(getattr(committed,'transient_maximum_energy_scale_J',0.0)))
            absolute_error = float(getattr(committed,'transient_absolute_energy_error_J',0.0))+abs(error)
            local_allowance=_transient_step_error_allowance(
                used_J=committed.transient_absolute_energy_error_J,scale_J=scale,
                accepted_steps=committed.transient_step_count,maximum_steps=case.transient.maximum_steps)
            # The remaining-step quota proposes refinement. At the registered
            # minimum dt only the original physical absolute budget is binding;
            # an optional allocation must not reject an otherwise valid step.
            can_refine = dt*0.5 >= case.transient.minimum_time_step_s
            if absolute_error > max(1e-10,1e-5*scale) or (abs(error)>local_allowance and can_refine):
                raise ContactConvergenceError('transient physical energy budget exceeded',
                    attempted_marker=next_marker,residual_norm_history_N=result.residual_norm_history_N)
            result=replace(result,transient_absolute_energy_error_J=absolute_error,
                           transient_maximum_energy_scale_J=scale)
            return _update_inertial_recovery(prepared,committed,result,case=case)
        except (ContactConvergenceError, DamagePathConstraintRequired) as error:
            if (dt <= case.transient.minimum_time_step_s
                    or getattr(error, 'registered_minimum_exhausted', False)):
                error.failure_stage = 'inertial_same_contact_step'
                error.inertial_failure_call = ((case, prepared, committed), dict(
                    start=start, target=target, achieved=achieved, marker_offset=marker_offset,
                    path_failure=path_failure, normal_penalty=normal_penalty,
                    tangential_penalty=tangential_penalty))
                raise
            # A localized event need not lie on the original dyadic time grid.
            # Try the legal minimum once before declaring this window exhausted.
            dt = max(dt*0.5, case.transient.minimum_time_step_s)


def _solve_target_or_direct_path(case,prepared,committed,trial_target,*,start,target,
        achieved,marker_offset,normal_penalty,tangential_penalty,marker):
    if committed.solve_mode is SolveMode.INERTIAL_TRANSITION:
        return _inertial_transition_after_path_failure(case,prepared,committed,
            start=start,target=target,achieved=achieved,marker_offset=marker_offset,
            path_failure=ContactSolverError('inertial continuation'),
            normal_penalty=normal_penalty,tangential_penalty=tangential_penalty)
    try:
        result = _solve_target(case,prepared,committed,trial_target,
            normal_penalty=normal_penalty,tangential_penalty=tangential_penalty,marker=marker)
    except ContactConvergenceError as error:
        if (
            case.damage is None
            or case.transient is None
        ):
            raise
        try:
            result = _direct_path_after_newton_failure(case,prepared,committed,start=start,target=target,
                achieved=achieved,marker_offset=marker_offset,normal_penalty=normal_penalty,
                tangential_penalty=tangential_penalty,initial_failure=error)
        except ContactSolverError as path_failure:
            return _inertial_transition_after_path_failure(
                case, prepared, committed, start=start, target=target, achieved=achieved,
                marker_offset=marker_offset, path_failure=path_failure,
                normal_penalty=normal_penalty,tangential_penalty=tangential_penalty,
            )
    # Both accepted quasistatic routes advance along the same prescribed
    # physical trajectory. Path solver coordinates must not reset its clock.
    if case.transient is not None:
        from .inertial_transition import assemble_consistent_mass_matrix
        distance=float(np.linalg.norm(np.asarray(result.grain_reference_m)-committed.grain_reference_m))
        if distance>0.:
            dt=distance/case.transient.grain_speed_m_per_s
            velocity=(result.displacement_vector_m-committed.displacement_vector_m)/dt
            mass=assemble_consistent_mass_matrix(prepared.structural,
                density_kg_per_m3=case.material.density_kg_per_m3,thickness_m=case.analysis.thickness)
            result=replace(result,physical_time_s=committed.physical_time_s+dt,
                velocity_vector_m_per_s=velocity,kinetic_energy_J=.5*float(velocity@mass@velocity),
                transient_absolute_energy_error_J=committed.transient_absolute_energy_error_J,
                transient_maximum_energy_scale_J=committed.transient_maximum_energy_scale_J,
                transient_step_count=committed.transient_step_count)
    return result


def _begin_separating_active_set(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    previous: CommittedContactStructureState | ConvergedContactIncrement,
    critical: ConvergedContactIncrement,
    background_element_ids: tuple[int, ...],
) -> ConvergedContactIncrement:
    if case.damage is None:
        raise ContactSolverError("SEPARATING requires damage input")
    if (not math.isfinite(critical.residual_norm_N) or not math.isfinite(critical.correction_norm_m)
        or critical.residual_norm_N>critical.residual_tolerance_N
        or critical.correction_norm_m>critical.correction_tolerance_m):
        raise ContactSolverError('SEPARATING requires a converged equilibrium and correction')
    requested = set(int(value) for value in background_element_ids)
    dynamic_terminal = critical.solve_mode is SolveMode.INERTIAL_TRANSITION
    if dynamic_terminal and any(d.damage != 1.0 for i,d in zip(
            critical.background_element_ids,critical.damage_states,strict=True) if int(i) in requested):
        raise ContactSolverError('dynamic terminal mapping requires naturally completed damage')
    direction = np.zeros(prepared.total_degrees_of_freedom, dtype=float)
    if dynamic_terminal:
        direction = np.asarray(critical.velocity_vector_m_per_s, dtype=float)
    elif critical.path_state is not None and critical.path_state.tangent.shape == (
        prepared.structural.free_dofs.size + 1,
    ):
        direction[prepared.structural.free_dofs] = critical.path_state.tangent[:-1]
    else:
        direction = (
            np.asarray(critical.displacement_vector_m, dtype=float)
            - np.asarray(previous.displacement_vector_m, dtype=float)
        )
    if not dynamic_terminal and not np.any(direction):
        raise ConstrainedSeparationEventError(
            "SEPARATING entry has no accepted loading-side direction"
        )
    active: list[tuple[int, object]] = []
    for active_id, background in enumerate(critical.background_element_ids):
        background_id = int(background)
        if background_id not in requested:
            continue
        effective_stress = np.asarray(
            critical.element_states[active_id].stress_tensor_Pa, dtype=float
        )
        if dynamic_terminal:
            history=critical.damage_states[active_id].energetic_history
            if history is not None:
                from .thermodynamic_energy import stored_energy,fracture_resistance
                material=case.to_elastoplastic_fem_case().material
                driving=stored_energy(effective_stress,
                    critical.element_states[active_id].equivalent_plastic_strain,1.,
                    young_modulus=material.E,poisson_ratio=material.nu,
                    hardening_modulus=material.internal_hardening_modulus)['effective_total']
                resistance=fracture_resistance(initial_driving_energy=history.initial_driving_energy,
                    fracture_energy_density=history.fracture_energy_density).resistance(1.)
                if abs(driving-resistance)>1e-10*history.initial_driving_energy:
                    raise ContactSolverError('natural terminal damage surface is not localized')
        element_direction = (
            prepared.structural.element_B_matrices_per_m[active_id]
            @ direction[prepared.structural.element_dofs[active_id]]
        )
        active.append(
            (
                background_id,
                begin_separating(
                    damage_input=case.damage,
                    committed=critical.damage_states[active_id],
                    characteristic_length_m=float(
                        np.sqrt(
                            4.0
                            * prepared.structural.element_areas_m2[active_id]
                            / np.pi
                        )
                    ),
                    effective_equivalent_stress_Pa=(
                        equivalent_von_mises_stress(effective_stress)
                    ),
                    entry_strain_direction=element_direction,
                ),
            )
        )
    if tuple(background for background, _ in active) != tuple(sorted(requested)):
        raise ContactSolverError("SEPARATING candidates do not match the active mesh")
    if dynamic_terminal:
        if any(value.damage != 1.0 or value.nominal_equivalent_stress_Pa != 0.0
               or value.remaining_fracture_energy_density_J_per_m3 != 0.0 for _,value in active):
            raise ContactSolverError('natural terminal state has nonzero capacity or fracture remainder')
        # Only map completed material metadata. No constitutive evolution,
        # prescribed progress solve, displacement change or physical-time jump.
        active = [(identity,replace(value,s_e=1.0,alpha_e=0.0)) for identity,value in active]
    return ConvergedContactIncrement(
        **{
            name: getattr(critical, name)
            for name in critical.__dataclass_fields__
            if name not in {"solve_mode", "solve_failure", "separating_states", "damage_states",
                            "friction_dissipation_increment_J"}
        },
        solve_mode=(SolveMode.INERTIAL_TRANSITION if dynamic_terminal else SolveMode.SEPARATION_CONSTRAINED),
        solve_failure=None,
        # This mapping is a zero-time metadata transaction. The preceding
        # physical segment has already accounted for its frictional heat.
        friction_dissipation_increment_J=0.0,
        separating_states=tuple(active),
        damage_states=tuple(replace(damage,status='SEPARATING') if int(background) in requested else damage
            for background,damage in zip(critical.background_element_ids,critical.damage_states,strict=True)),
    )


@dataclass(frozen=True)
class EventEnergyBudget:
    maximum_absolute_work_J: float = 0.
    event_absolute_error_J: float = 0.
    ordinary_absolute_error_J: float = 0.

    def accept(self, cumulative_work, error, *, event):
        if not math.isfinite(cumulative_work) or not math.isfinite(error):
            raise ContactSolverError('nonfinite event energy ledger')
        scale=max(self.maximum_absolute_work_J,abs(cumulative_work))
        updated=EventEnergyBudget(scale,self.event_absolute_error_J+(abs(error) if event else 0.),
                                  self.ordinary_absolute_error_J+(0. if event else abs(error)))
        allowance=.5*max(1e-8,.05*scale)
        if max(updated.event_absolute_error_J,updated.ordinary_absolute_error_J)>allowance:
            raise ContactSolverError(f'accepted-prefix energy budget exceeded: {updated}, allowance={allowance}')
        return updated


class ZeroForceTopologyTransactionError(ContactSolverError):
    """Measured topology discontinuity; retrying must not hide this failure."""

    retryable = False

    def __init__(self, measures, *, zero_force_measures=None, zero_force_thresholds=None):
        self.measures = dict(measures)
        self.zero_force_measures = zero_force_measures
        self.zero_force_thresholds = zero_force_thresholds
        self.failure_stage = 'zero_force_topology_transaction'
        from dataclasses import asdict
        detail = (f'; zero_force_measures={asdict(zero_force_measures)}; '
                  f'zero_force_thresholds={asdict(zero_force_thresholds)}'
                  if zero_force_measures is not None and zero_force_thresholds is not None else '')
        super().__init__(f'zero-force topology transaction rejected: {self.measures}{detail}')


def _measure_zero_force_topology_transaction(
    case, old_prepared, before, new_prepared, rebased, target,
    *, normal_penalty, tangential_penalty,
):
    """Measure the same-displacement transaction, without committing any trial."""
    fem = case.to_elastoplastic_fem_case()
    body = _committed_damage_fields(fem, new_prepared, rebased)
    contact = _assemble_contact(fem, new_prepared, rebased,
        rebased.displacement_vector_m, _grain(case, target), normal_penalty,
        tangential_penalty, case.contact.friction_coefficient,
        case.contact.friction_regularization_ratio)
    mapped = replace(before, displacement_vector_m=rebased.displacement_vector_m,
        element_states=body.element_states, damage_states=body.damage_states,
        element_total_strain=body.element_total_strain,
        contact_states=contact.states, contact_kinematics=contact.kinematics,
        edge_contact_states=contact.edge_contact_states,
        archived_edge_contact_states=getattr(before,'archived_edge_contact_states',()))
    def stored(prepared, state):
        return _elastic_strain_energy(case, prepared, state) + _hardening_stored_energy(case, prepared, state) + _numerical_contact_stored_energy(
            case, prepared, state, normal_penalty_Pa_per_m=normal_penalty,
            tangential_penalty_Pa_per_m=tangential_penalty)
    old_residual = before.internal_force_vector_N - before.contact_force_vector_N
    new_residual = body.internal_force_N - contact.force_N
    def support(prepared, residual):
        if before.solve_mode is SolveMode.INERTIAL_TRANSITION:
            from .inertial_transition import assemble_consistent_mass_matrix
            from scipy.sparse.linalg import spsolve
            mass=assemble_consistent_mass_matrix(prepared.structural,
                density_kg_per_m3=case.material.density_kg_per_m3,thickness_m=case.analysis.thickness)
            free=prepared.structural.free_dofs
            acceleration=np.zeros(prepared.total_degrees_of_freedom)
            acceleration[free]=spsolve(mass[free][:,free],-residual[free])
            residual=residual+mass@acceleration
        return np.array([np.sum(residual[prepared.structural.fixed_x_dofs]),
                         np.sum(residual[prepared.structural.fixed_y_dofs])])
    retained = set(int(n) for n in new_prepared.active_to_background_node_ids)
    lost = [2*i+c for i,n in enumerate(old_prepared.active_to_background_node_ids)
            if int(n) not in retained for c in (0,1)]
    return dict(
        projection_reaction_norm_N=float(np.linalg.norm(old_residual[lost])),
        discarded_dof_count=len(lost),
        support_reaction_jump_N=float(np.linalg.norm(
            support(new_prepared,new_residual)-support(old_prepared,old_residual))),
        contact_force_jump_N=float(np.linalg.norm(
            contact.force_N.reshape(-1,2).sum(axis=0)
            -before.contact_force_vector_N.reshape(-1,2).sum(axis=0))),
        topology_energy_jump_J=stored(new_prepared,mapped)-stored(old_prepared,before),
    )


def _evaluate_dynamic_topology_state(case, prepared, rebased, before, *, normal_penalty, tangential_penalty):
    """Rebuild forces/acceleration at unchanged position, velocity and time."""
    from scipy.sparse.linalg import spsolve
    from .inertial_transition import assemble_consistent_mass_matrix
    fem = case.to_elastoplastic_fem_case()
    body = _committed_damage_fields(fem,prepared,rebased)
    target = TrajectoryTarget(segment='indentation',reference_x_m=rebased.grain_reference_m[0],
                              reference_y_m=rebased.grain_reference_m[1])
    contact = _assemble_contact(fem,prepared,rebased,rebased.displacement_vector_m,_grain(case,target),
        normal_penalty,tangential_penalty,case.contact.friction_coefficient,case.contact.friction_regularization_ratio)
    mass = assemble_consistent_mass_matrix(prepared.structural,
        density_kg_per_m3=case.material.density_kg_per_m3,thickness_m=case.analysis.thickness)
    free = prepared.structural.free_dofs
    acceleration = np.zeros(prepared.total_degrees_of_freedom)
    net = contact.force_N-body.internal_force_N
    acceleration[free] = spsolve(mass[free][:,free],net[free])
    residual = mass@acceleration-net
    if not np.all(np.isfinite(acceleration)) or np.linalg.norm(residual[free])>before.residual_tolerance_N:
        raise ContactSolverError('rebuilt dynamic topology fails physical momentum balance')
    total_contact = contact.force_N.reshape(-1,2).sum(axis=0)
    return replace(before,
        displacement_vector_m=rebased.displacement_vector_m.copy(),element_states=body.element_states,
        damage_states=body.damage_states,background_element_ids=rebased.background_element_ids.copy(),
        separating_states=rebased.separating_states,path_state=None,accepted_path_reference=rebased.accepted_path_reference,
        contact_states=contact.states,edge_contact_states=contact.edge_contact_states,
        archived_edge_contact_states=rebased.archived_edge_contact_states,candidate_keys=prepared.candidate_keys,
        contact_kinematics=contact.kinematics,contact_pressure_Pa=contact.pressures_Pa,
        contact_force_vector_N=contact.force_N,internal_force_vector_N=body.internal_force_N,
        element_total_strain=body.element_total_strain,element_stress_Pa=body.element_stress_Pa,
        element_sigma_zz_Pa=body.element_sigma_zz_Pa,
        total_contact_force_N=total_contact,rigid_grain_reaction_N=-total_contact,
        acceleration_vector_m_per_s2=acceleration,velocity_vector_m_per_s=rebased.velocity_vector_m_per_s.copy(),
        physical_time_s=rebased.physical_time_s,kinetic_energy_J=rebased.kinetic_energy_J,
        transient_absolute_energy_error_J=rebased.transient_absolute_energy_error_J,
        transient_maximum_energy_scale_J=rebased.transient_maximum_energy_scale_J,
        transient_stable_step_count=0,
        residual_norm_N=float(np.linalg.norm(residual[free])),correction_norm_m=0.,
        fixed_reaction_x_N=float(np.sum(residual[prepared.structural.fixed_x_dofs])),
        fixed_reaction_y_N=float(np.sum(residual[prepared.structural.fixed_y_dofs])),
        balance_residual_x_N=float(np.sum(residual[free[free%2==0]])),
        balance_residual_y_N=float(np.sum(residual[free[free%2==1]])),
        friction_dissipation_increment_J=0.,maximum_penetration_m=max(contact.integrated_maximum_penetration_m,
            max((item.penetration_m for item in contact.kinematics),default=0.)))


def _verify_zero_force_separation_commit(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    state: ConvergedContactIncrement,
    *,
    reference_path_length_m: float,
    topology_measures: dict | None = None,
    boundary_change: bool = False,
    removed_dof_kinetic_energy_J: float = 0.0,
    kinetic_energy_absolute_tolerance_J: float = math.inf,
) -> None:
    if topology_measures is None:
        raise ContactSolverError('zero-force commit requires actual topology measurements')
    if case.damage is None:
        raise ContactSolverError("zero-force commit requires damage input")
    separating = dict(state.separating_states)
    if not separating or len(separating)!=len(state.separating_states):
        raise ContactSolverError('zero-force commit requires unique nonempty SEPARATING identities')
    if state.solve_mode is SolveMode.INERTIAL_TRANSITION:
        from .inertial_transition import material_kinetic_energy
        measured_kinetic=material_kinetic_energy(prepared,state,tuple(separating),
            density_kg_per_m3=case.material.density_kg_per_m3,thickness_m=case.analysis.thickness)
        registered_kinetic_gate=max(1e-14,1e-6*state.transient_maximum_energy_scale_J)
        if measured_kinetic>registered_kinetic_gate:
            raise ContactSolverError('zero-force transaction retains excessive removed-material kinetic energy')
        removed_dof_kinetic_energy_J=measured_kinetic
        kinetic_energy_absolute_tolerance_J=registered_kinetic_gate
    active_by_background = {
        int(background): active_id
        for active_id, background in enumerate(state.background_element_ids)
    }
    for background_id, separation in state.separating_states:
        if int(background_id) not in active_by_background:
            raise ContactSolverError('SEPARATING identity is absent from active topology')
        active_id = active_by_background[int(background_id)]
        area = float(prepared.structural.element_areas_m2[active_id])
        length = float(separation.characteristic_length_m)
        stress = np.asarray(state.element_stress_Pa[active_id], dtype=float)
        element_internal = (
            prepared.structural.element_B_matrices_per_m[active_id].T
            @ stress
            * area
            * case.analysis.thickness
        )
        thresholds = registered_zero_force_thresholds(
            residual_absolute_tolerance_N=case.newton.residual_absolute_tolerance_N,
            displacement_absolute_tolerance_m=(
                case.newton.displacement_absolute_tolerance_m
            ),
            damage_event_tolerance=case.damage.damage_event_tolerance,
            minimum_substep_fraction=case.step_control.minimum_substep_fraction,
            reference_path_length_m=reference_path_length_m,
            fracture_energy_J_per_m2=case.damage.fracture_energy_J_per_m2,
            element_area_m2=area,
            thickness_m=case.analysis.thickness,
            characteristic_length_m=length,
        )
        remaining_energy = (
            separation.remaining_fracture_energy_density_J_per_m3
            * area
            * case.analysis.thickness
        )
        measures = ZeroForceCommitMeasures(
            constitutive_capacity_is_exactly_zero=(
                separation.s_e == 1.0
                and separation.damage == 1.0
                and separation.nominal_equivalent_stress_Pa == 0.0
                and separation.alpha_e == 0.0
            ),
            element_internal_force_norm_N=float(np.linalg.norm(element_internal)),
            projection_reaction_norm_N=topology_measures['projection_reaction_norm_N'],
            remaining_fracture_energy_J=remaining_energy,
            # Real boundary activation is checked by the complete event ledger,
            # not by the pure-representation energy identity.
            topology_energy_jump_J=(0. if boundary_change else topology_measures['topology_energy_jump_J']),
            balance_residual_N=float(state.residual_norm_N),
            path_residual_m=abs(float(state.separation_path_residual_m)),
            separation_residual=float(state.separation_constraint_residual),
            displacement_correction_m=float(state.correction_norm_m),
            damage_energy_fully_accounted=(remaining_energy == 0.0),
            removed_dof_kinetic_energy_J=removed_dof_kinetic_energy_J,
            kinetic_energy_absolute_tolerance_J=kinetic_energy_absolute_tolerance_J,
        )
        if (not qualifies_for_removed_commit(measures, thresholds)
            or (not boundary_change and not all(math.isfinite(topology_measures[name]) and
                       0 <= topology_measures[name] <= case.newton.residual_absolute_tolerance_N
                       for name in ('support_reaction_jump_N','contact_force_jump_N')))):
            raise ZeroForceTopologyTransactionError(topology_measures,
                zero_force_measures=measures, zero_force_thresholds=thresholds)
    if set(separating) != {
        int(background) for background, _ in state.separating_states
    }:  # pragma: no cover - defensive identity check
        raise ContactSolverError("SEPARATING identity changed during zero-force verification")


def _damage_dissipation_J(
    case: SingleGrainContactCase,
    prepared: PreparedContactMesh,
    state: CommittedContactStructureState | ConvergedContactIncrement,
) -> float:
    return math.fsum(
        damage.damage_dissipation_density_J_per_m3
        * float(area)
        * case.analysis.thickness
        for damage, area in zip(
            state.damage_states,
            prepared.structural.element_areas_m2,
            strict=True,
        )
    )


def _background_element_areas_m2(imported_mesh: ImportedMesh) -> np.ndarray:
    coordinates = np.asarray(imported_mesh.mesh.p.T, dtype=float)
    connectivity = np.asarray(imported_mesh.mesh.t.T, dtype=np.int32)
    first = coordinates[connectivity[:, 1]] - coordinates[connectivity[:, 0]]
    second = coordinates[connectivity[:, 2]] - coordinates[connectivity[:, 0]]
    return 0.5 * np.abs(
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )


def run_contact_trajectory_on_mesh(
    case: SingleGrainContactCase,
    imported_mesh: ImportedMesh | None,
    *,
    workspace: str | Path | None = None,
    progress: Callable[[int, int, str | None], None] | None = None,
    checkpoint_callback: Callable[[dict], None] | None = None,
    resume_checkpoint: dict | None = None,
) -> ContactTrajectoryResult:
    """Run every requested grain position with private retry substeps."""

    if not isinstance(case, SingleGrainContactCase):
        raise TypeError("case must be a SingleGrainContactCase")
    if resume_checkpoint is not None:
        import copy
        if (resume_checkpoint.get('format') != 'grindcae_trajectory_substep_v1'
                or resume_checkpoint.get('case_input') != case.to_dict()):
            raise ValueError('trajectory checkpoint case identity mismatch')
        resume_checkpoint = copy.deepcopy(resume_checkpoint)
        if imported_mesh is None:
            imported_mesh = resume_checkpoint['background_mesh']
        if (not np.array_equal(imported_mesh.mesh.p,resume_checkpoint['background_mesh'].mesh.p)
                or not np.array_equal(imported_mesh.mesh.t,resume_checkpoint['background_mesh'].mesh.t)):
            raise ValueError('trajectory checkpoint mesh identity mismatch')
        expected_boundaries=resume_checkpoint['background_mesh'].mesh.boundaries or {}
        actual_boundaries=imported_mesh.mesh.boundaries or {}
        if (set(expected_boundaries)!=set(actual_boundaries) or any(
                not np.array_equal(expected_boundaries[name],actual_boundaries[name]) for name in expected_boundaries)):
            raise ValueError('trajectory checkpoint boundary identity mismatch')
        from .state_codec import encode_committed_state, decode_committed_state, validate_material_compatibility
        saved_prepared=resume_checkpoint['prepared']
        mass_audit = (dict(density_kg_per_m3=case.material.density_kg_per_m3,
                           thickness_m=case.analysis.thickness)
                      if case.transient is not None else {})
        decode_committed_state(saved_prepared,encode_committed_state(saved_prepared,resume_checkpoint['committed']),
            **mass_audit)
        validate_material_compatibility(saved_prepared,resume_checkpoint['committed'],case.to_elastoplastic_fem_case().material)
        index=resume_checkpoint['target_index'];achieved=resume_checkpoint['achieved']
        saved_state=resume_checkpoint['committed']
        if (type(index) is not int or not 1<=index<len(case.trajectory.targets)
                or not math.isfinite(achieved) or not 0.<achieved<=1.
                or abs(saved_state.marker-index-achieved)>1e-12):
            raise ValueError('trajectory checkpoint progress is inconsistent')
        point=_interpolated_target(resume_checkpoint['start'],case.trajectory.targets[index],achieved)
        reference=(point.reference_x_m,point.reference_y_m) if point.reference_x_m is not None else (point.center_x_m,point.center_y_m)
        if not np.allclose(saved_state.grain_reference_m,reference,rtol=0.,atol=1e-15):
            raise ValueError('trajectory checkpoint progress geometry is inconsistent')
        if len(resume_checkpoint['records'])!=index:
            raise ValueError('trajectory checkpoint progress lacks completed targets')
        fraction=resume_checkpoint['fraction']
        retries=resume_checkpoint['retry_budget']
        if (type(resume_checkpoint['substep_count']) is not int
                or resume_checkpoint['substep_count']<1
                or not math.isfinite(fraction) or not 0.<=fraction<=1.-achieved+1e-15
                or (achieved<1.-1e-15 and fraction==0.)
                or not isinstance(retries,_SubstepRetryBudget)
                or retries.maximum_consecutive_failures!=case.step_control.maximum_retries
                or type(retries.total_failures) is not int or retries.total_failures<0
                or retries.consecutive_failures!=0):
            raise ValueError('trajectory checkpoint step control is inconsistent')
        def target_reference(point):
            return np.asarray((point.reference_x_m,point.reference_y_m)
                if point.reference_x_m is not None else (point.center_x_m,point.center_y_m))
        previous_reference=target_reference(case.trajectory.targets[index-1])
        if not np.array_equal(np.asarray(resume_checkpoint['start']),previous_reference):
            raise ValueError('trajectory checkpoint progress start disagrees with prior target')
        if case.transient is not None:
            references=[target_reference(point) for point in case.trajectory.targets[:index]]
            distance=math.fsum(float(np.linalg.norm(b-a)) for a,b in zip(references,references[1:]))
            distance+=float(np.linalg.norm(np.asarray(reference)-previous_reference))
            expected_time=distance/case.transient.grain_speed_m_per_s
            if not math.isclose(saved_state.physical_time_s,expected_time,rel_tol=1e-9,abs_tol=1e-15):
                raise ValueError('trajectory checkpoint physical clock disagrees with prescribed path')
        ledger_names=('cumulative_contact_work','cumulative_friction_dissipation','cumulative_plastic_dissipation',
            'cumulative_damage_dissipation','cumulative_separation_event_energy','contact_work_increment',
            'friction_dissipation_increment','plastic_dissipation_increment','damage_dissipation_increment',
            'separation_event_energy_increment','prefix_work')
        if any(not math.isfinite(resume_checkpoint[name]) for name in ledger_names):
            raise ValueError('trajectory checkpoint energy ledger is nonfinite')
        saved_budget=resume_checkpoint['energy_budget']
        if (not isinstance(saved_budget,EventEnergyBudget) or any(
                not math.isfinite(value) or value<0. for value in (
                    saved_budget.maximum_absolute_work_J,saved_budget.event_absolute_error_J,
                    saved_budget.ordinary_absolute_error_J))):
            raise ValueError('trajectory checkpoint event energy budget is invalid')
        work=resume_checkpoint['cumulative_contact_work']+resume_checkpoint['contact_work_increment']
        if abs(work-resume_checkpoint['prefix_work'])>128*np.finfo(float).eps*max(
                abs(work),abs(resume_checkpoint['prefix_work']),1e-300):
            raise ValueError('trajectory checkpoint energy work disagrees with accepted prefix')
        if case.transient is not None:
            initial_row=resume_checkpoint['records'][0]
            baseline=(initial_row.elastic_strain_energy_J+initial_row.hardening_stored_energy_J
                +initial_row.numerical_contact_stored_energy_J+initial_row.state.kinetic_energy_J)
            current_storage=(_elastic_strain_energy(case,saved_prepared,saved_state)
                +_hardening_stored_energy(case,saved_prepared,saved_state)
                +_numerical_contact_stored_energy(case,saved_prepared,saved_state,
                    normal_penalty_Pa_per_m=resume_checkpoint['normal_penalty'],
                    tangential_penalty_Pa_per_m=resume_checkpoint['tangential_penalty'])+saved_state.kinetic_energy_J)
            dissipation=math.fsum(resume_checkpoint['cumulative_'+name]+resume_checkpoint[name+'_increment']
                for name in ('friction_dissipation','plastic_dissipation','damage_dissipation'))
            error=work-current_storage+baseline-dissipation
            rounding=128*np.finfo(float).eps*max(abs(work),current_storage,baseline,dissipation,1e-300)
            if abs(error)>saved_state.transient_absolute_energy_error_J+rounding:
                raise ValueError('trajectory checkpoint energy budget does not cover accepted balance')
    reference_path: Path | None = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if imported_mesh is None:
        if workspace is None:
            temporary = tempfile.TemporaryDirectory(prefix="grindcae-contact-mesh-")
            directory = Path(temporary.name)
        else:
            directory = Path(workspace).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / "reference_mesh.msh"
        fem_case = case.to_elastoplastic_fem_case()
        generate_split_top_rectangle_mesh(
            fem_case,
            reference_path,
            top_split_x_m=(),
            maximum_estimated_degrees_of_freedom=case.safety.maximum_degrees_of_freedom,
        )
        imported_mesh = import_gmsh_mesh(reference_path, fem_case)
    fem_case = case.to_elastoplastic_fem_case()
    if case.damage is not None:
        minimum_background_area = float(np.min(_background_element_areas_m2(imported_mesh)))
        if case.damage.area_absolute_tolerance_m2 > 0.01 * minimum_background_area:
            raise ValueError(
                "damage.area_absolute_tolerance_m2 must not exceed 1% of the minimum background element area"
            )
    material_topology = None
    if case.single_grain_contact_schema_version >= 2:
        from .material_topology import prepare_material_topology

        material_topology = prepare_material_topology(
            imported_mesh, case.material_topology.preset_removal
        )
    prepared = prepare_contact_mesh(
        fem_case, imported_mesh, topology=material_topology,
        edge_history=case.single_grain_contact_schema_version==3,
    )
    if getattr(case,'circle_edge_quadrature_order',0):
        if (case.single_grain_contact_schema_version!=3 or case.grain.type!='rounded_circle'
            or case.contact.normal_algorithm!='penalty' or case.contact.friction_coefficient!=0.):
            raise ContactSolverError('contact_integration requires Schema3 frictionless circle penalty')
        prepared=replace(prepared,circle_edge_quadrature_order=case.circle_edge_quadrature_order)
    local_size = float(np.median(prepared.candidate_tributary_lengths_m))
    normal_penalty = (
        case.contact.normal_penalty_factor * case.effective_modulus_Pa / local_size
    )
    tangential_penalty = case.contact.tangential_penalty_ratio * normal_penalty
    if resume_checkpoint is not None:
        if (resume_checkpoint['normal_penalty']!=normal_penalty
                or resume_checkpoint['tangential_penalty']!=tangential_penalty):
            raise ValueError('trajectory checkpoint penalty disagrees with input mesh')
        topology=resume_checkpoint['material_topology']
        if case.single_grain_contact_schema_version>=2:
            if topology is None or not np.array_equal(topology.active_element_ids,
                    resume_checkpoint['committed'].background_element_ids):
                raise ValueError('trajectory checkpoint topology ownership mismatch')
        expected_prepared=prepare_contact_mesh(fem_case,imported_mesh,topology=topology,
            edge_history=case.single_grain_contact_schema_version==3)
        saved_prepared=resume_checkpoint['prepared']
        if (saved_prepared.circle_edge_quadrature_order!=case.circle_edge_quadrature_order
                or saved_prepared.candidate_keys!=expected_prepared.candidate_keys
                or saved_prepared.edge_endpoint_keys!=expected_prepared.edge_endpoint_keys
                or any(not np.array_equal(getattr(saved_prepared,name),getattr(expected_prepared,name))
                    for name in ('active_to_background_node_ids','active_to_background_element_ids'))
                or any(not np.array_equal(getattr(saved_prepared.structural.imported_mesh.mesh,name),
                    getattr(expected_prepared.structural.imported_mesh.mesh,name)) for name in ('p','t'))
                or not np.array_equal(saved_prepared.structural.fixed_dofs,expected_prepared.structural.fixed_dofs)):
            raise ValueError('trajectory checkpoint active mesh disagrees with background topology')
        from .state_codec import validate_endpoint_momentum
        state=resume_checkpoint['committed']
        restored_target=_interpolated_target(tuple(state.grain_reference_m),
            case.trajectory.targets[resume_checkpoint['target_index']],0.)
        restored_contact=_assemble_contact(fem_case,saved_prepared,state,state.displacement_vector_m,
            _grain(case,restored_target),normal_penalty,tangential_penalty,
            case.contact.friction_coefficient,case.contact.friction_regularization_ratio)
        validate_endpoint_momentum(saved_prepared,state,case,restored_contact.force_N)
        if case.damage is not None:
            restored_internal = _committed_damage_fields(fem_case,saved_prepared,state).internal_force_N
            total_contact = restored_contact.force_N.reshape(-1,2).sum(axis=0)
            for name, expected_force in (
                ('internal_force_vector_N', restored_internal),
                ('contact_force_vector_N', restored_contact.force_N),
                ('total_contact_force_N', total_contact),
                ('rigid_grain_reaction_N', -total_contact),
            ):
                actual_force = np.asarray(getattr(state,name),dtype=float)
                force_tolerance = (case.newton.residual_absolute_tolerance_N
                    + 128*np.finfo(float).eps*float(np.linalg.norm(expected_force)))
                if (actual_force.shape != expected_force.shape or not np.all(np.isfinite(actual_force))
                        or np.linalg.norm(actual_force-expected_force)>force_tolerance):
                    raise ValueError('trajectory checkpoint force field disagrees with committed history: '+name)
    first = case.trajectory.targets[0]
    first_grain = _grain(case, first)
    committed: CommittedContactStructureState | ConvergedContactIncrement = (
        initial_contact_structure_state(
            fem_case, prepared, grain=first_grain, damage_input=case.damage
        )
    )
    initial = resume_checkpoint['records'][0].state if resume_checkpoint is not None else _solve_target(
        case,
        prepared,
        committed,
        first,
        normal_penalty=normal_penalty,
        tangential_penalty=tangential_penalty,
        marker=0.0,
    )
    initial_applicability = _applicability(case, prepared, initial)
    initial_record = _record(
        case,
        0,
        first,
        initial,
        retry_count=0,
        substep_count=0,
        previous_state=None,
        prepared=prepared,
        applicability=initial_applicability,
        normal_penalty_Pa_per_m=normal_penalty,
        tangential_penalty_Pa_per_m=tangential_penalty,
        baseline_mechanical_energy_J=None,
        cumulative_contact_work_J=0.0,
        cumulative_friction_dissipation_J=0.0,
        cumulative_plastic_dissipation_J=0.0,
    )
    if material_topology is not None:
        initial_record = ContactTrajectoryRecord(
            **{
                name: getattr(initial_record, name)
                for name in initial_record.__dataclass_fields__
                if name != "material_topology"
            },
            material_topology=material_topology,
        )
    records = [initial_record]
    separation_events: list[SeparationEventRecord] = []
    baseline_mechanical_energy = (
        initial_record.elastic_strain_energy_J
        + initial_record.hardening_stored_energy_J
        + initial_record.numerical_contact_stored_energy_J
        + initial_record.state.kinetic_energy_J
    )
    cumulative_contact_work = 0.0
    cumulative_friction_dissipation = 0.0
    cumulative_plastic_dissipation = 0.0
    cumulative_damage_dissipation = 0.0
    cumulative_separation_event_energy = 0.0
    energy_budget=EventEnergyBudget()
    prefix_work=0.
    if progress is not None:
        progress(1, len(case.trajectory.targets), "已提交单磨粒初始位置 1/{}".format(len(case.trajectory.targets)))
    committed = initial
    first_target_index=1
    if resume_checkpoint is not None:
        saved=resume_checkpoint
        first_target_index=saved['target_index']
        records=saved['records'];separation_events=saved['separation_events']
        prepared=saved['prepared'];material_topology=saved['material_topology'];committed=saved['committed']
        energy_budget=saved['energy_budget'];prefix_work=saved['prefix_work']
        cumulative_contact_work=saved['cumulative_contact_work']
        cumulative_friction_dissipation=saved['cumulative_friction_dissipation']
        cumulative_plastic_dissipation=saved['cumulative_plastic_dissipation']
        cumulative_damage_dissipation=saved['cumulative_damage_dissipation']
        cumulative_separation_event_energy=saved['cumulative_separation_event_energy']
        normal_penalty=saved['normal_penalty'];tangential_penalty=saved['tangential_penalty']
    for target_index, target in enumerate(case.trajectory.targets[first_target_index:], start=first_target_index):
        target_start_state = committed
        start = tuple(float(value) for value in committed.grain_reference_m)
        future_references = [np.asarray((point.reference_x_m, point.reference_y_m)
            if point.reference_x_m is not None else (point.center_x_m, point.center_y_m))
            for point in case.trajectory.targets[target_index:]]
        future_distance = math.fsum(float(np.linalg.norm(b-a))
            for a,b in zip(future_references, future_references[1:]))
        fraction = 1.0
        achieved = 0.0
        retry_budget = _SubstepRetryBudget(
            maximum_consecutive_failures=case.step_control.maximum_retries
        )
        substep_count = 0
        physical_energy_retry = False
        contact_work_increment = 0.0
        friction_dissipation_increment = 0.0
        plastic_dissipation_increment = 0.0
        damage_dissipation_increment = 0.0
        separation_event_energy_increment = 0.0
        if resume_checkpoint is not None:
            saved=resume_checkpoint
            target_start_state=saved['target_start_state'];start=saved['start']
            fraction=saved['fraction'];achieved=saved['achieved'];retry_budget=saved['retry_budget']
            substep_count=saved['substep_count'];converged_applicability=saved['converged_applicability']
            contact_work_increment=saved['contact_work_increment']
            friction_dissipation_increment=saved['friction_dissipation_increment']
            plastic_dissipation_increment=saved['plastic_dissipation_increment']
            damage_dissipation_increment=saved['damage_dissipation_increment']
            separation_event_energy_increment=saved['separation_event_energy_increment']
            resume_checkpoint=None
        while achieved < 1.0 - 1.0e-15:
            attempted = min(1.0, achieved + fraction)
            trial_target = _interpolated_target(start, target, attempted)
            dynamic_trial_completed = False
            try:
                trial_prepared = prepared
                trial_topology = material_topology
                trial_committed = committed
                trial_events: list[SeparationEventRecord] = []
                topology_absolute_error = 0.0
                energy_segments: list[
                    tuple[
                        CommittedContactStructureState | ConvergedContactIncrement,
                        ConvergedContactIncrement,
                        PreparedContactMesh,
                    ]
                ] = []
                constrained_event_located = False
                try:
                    if physical_energy_retry:
                        converged = _inertial_transition_after_path_failure(
                            case, trial_prepared, trial_committed, start=start, target=target,
                            achieved=achieved, marker_offset=float(target_index),
                            normal_penalty=normal_penalty, tangential_penalty=tangential_penalty,
                            path_failure=ContactSolverError('quasistatic physical energy closure failed'))
                    else:
                        converged = _solve_target_or_direct_path(
                            case,
                            trial_prepared,
                            trial_committed,
                            trial_target,
                            start=start,target=target,achieved=achieved,marker_offset=float(target_index),
                            normal_penalty=normal_penalty,
                            tangential_penalty=tangential_penalty,
                            marker=target_index + attempted,
                        )
                    dynamic_trial_completed = converged.solve_mode is SolveMode.INERTIAL_TRANSITION
                    constrained_event_located=(converged.solve_mode in (
                        SolveMode.PATH_CONTINUATION, SolveMode.INERTIAL_TRANSITION)
                        or trial_committed.solve_mode is SolveMode.INERTIAL_TRANSITION)
                except ContactConvergenceError as initial_failure:
                    if (physical_energy_retry or trial_committed.solve_mode is SolveMode.INERTIAL_TRANSITION
                            or getattr(initial_failure, 'failure_stage', None) == 'inertial_same_contact_step'):
                        raise
                    if (
                        case.damage is None
                        or initial_failure.maximum_damage
                        < 0.9 * (1.0 - case.damage.damage_event_tolerance)
                    ):
                        raise
                    seed_fraction = achieved + case.step_control.minimum_substep_fraction
                    if seed_fraction >= attempted:
                        raise
                    seed_target = _interpolated_target(start, target, seed_fraction)
                    seed = _solve_target(
                        case,
                        trial_prepared,
                        trial_committed,
                        seed_target,
                        normal_penalty=normal_penalty,
                        tangential_penalty=tangential_penalty,
                        marker=target_index + seed_fraction,
                    )
                    target_reference = np.asarray(
                        (
                            target.reference_x_m,
                            target.reference_y_m,
                        )
                        if target.reference_x_m is not None
                        else (target.center_x_m, target.center_y_m),
                        dtype=float,
                    )
                    start_reference = np.asarray(start, dtype=float)
                    reference_path_length = float(
                        np.linalg.norm(target_reference - start_reference)
                    )
                    converged = locate_constrained_separation_event(
                        case.to_elastoplastic_fem_case(),
                        trial_prepared,
                        trial_committed,
                        seed,
                        start_fraction=achieved,
                        seed_fraction=seed_fraction,
                        reference_path_length_m=reference_path_length,
                        grain_at_fraction=lambda value: _grain(
                            case, _interpolated_target(start, target, value)
                        ),
                        marker_offset=float(target_index),
                        normal_penalty_Pa_per_m=normal_penalty,
                        tangential_penalty_Pa_per_m=tangential_penalty,
                        friction_coefficient=case.contact.friction_coefficient,
                        friction_regularization_ratio=case.contact.friction_regularization_ratio,
                        damage_input=case.damage,
                        minimum_substep_fraction=case.step_control.minimum_substep_fraction,
                        maximum_steps=min(
                            4096,
                            max(
                                128,
                                math.ceil(
                                    2.0
                                    / case.step_control.minimum_substep_fraction
                                ),
                            ),
                        ),
                    )
                    constrained_event_located = True
                candidates = _damage_candidates(
                    case, converged, accepted_events_at_position=0, prepared=trial_prepared
                )
                accepted_events_at_position = 0
                event_fraction = (
                    converged.marker - target_index
                    if constrained_event_located
                    else attempted
                )
                if candidates and not constrained_event_located:
                    assert case.damage is not None and trial_topology is not None
                    def solve_at_fraction(value: float):
                        point = _interpolated_target(start, target, value)
                        solved = _solve_target(
                            case,
                            trial_prepared,
                            trial_committed,
                            point,
                            normal_penalty=normal_penalty,
                            tangential_penalty=tangential_penalty,
                            marker=target_index + value,
                        )
                        return solved, solved.background_element_ids, solved.damage_states

                    try:
                        located = locate_earliest_separation_event(
                            lower_fraction=achieved,
                            upper_fraction=attempted,
                            solve_at_fraction=solve_at_fraction,
                            damage_event_tolerance=case.damage.damage_event_tolerance,
                            fraction_tolerance=1.0e-6,
                        )
                        converged = located.solved_state
                        candidates = located.background_element_ids
                        event_fraction = located.fraction
                    except ContactConvergenceError:
                        seed_fraction = (
                            achieved
                            + case.step_control.minimum_substep_fraction
                        )
                        if seed_fraction >= attempted:
                            raise
                        seed_target = _interpolated_target(
                            start, target, seed_fraction
                        )
                        seed = _solve_target(
                            case,
                            trial_prepared,
                            trial_committed,
                            seed_target,
                            normal_penalty=normal_penalty,
                            tangential_penalty=tangential_penalty,
                            marker=target_index + seed_fraction,
                        )
                        target_reference = np.asarray(
                            (
                                target.reference_x_m,
                                target.reference_y_m,
                            )
                            if target.reference_x_m is not None
                            else (target.center_x_m, target.center_y_m),
                            dtype=float,
                        )
                        converged = locate_constrained_separation_event(
                            case.to_elastoplastic_fem_case(),
                            trial_prepared,
                            trial_committed,
                            seed,
                            start_fraction=achieved,
                            seed_fraction=seed_fraction,
                            reference_path_length_m=float(
                                np.linalg.norm(
                                    target_reference - np.asarray(start, dtype=float)
                                )
                            ),
                            grain_at_fraction=lambda value: _grain(
                                case, _interpolated_target(start, target, value)
                            ),
                            marker_offset=float(target_index),
                            normal_penalty_Pa_per_m=normal_penalty,
                            tangential_penalty_Pa_per_m=tangential_penalty,
                            friction_coefficient=case.contact.friction_coefficient,
                            friction_regularization_ratio=case.contact.friction_regularization_ratio,
                            damage_input=case.damage,
                            minimum_substep_fraction=case.step_control.minimum_substep_fraction,
                            maximum_steps=min(
                                4096,
                                max(
                                    128,
                                    math.ceil(
                                        2.0
                                        / case.step_control.minimum_substep_fraction
                                    ),
                                ),
                            ),
                        )
                        candidates = _damage_candidates(
                            case, converged, accepted_events_at_position=0, prepared=trial_prepared
                        )
                        event_fraction = converged.marker - target_index
                energy_segments.append((trial_committed, converged, trial_prepared))

                while candidates:
                    assert case.damage is not None and trial_topology is not None
                    accepted_events_at_position += 1
                    if (
                        accepted_events_at_position
                        > case.damage.maximum_separation_events_per_position
                    ):
                        raise SeparationEventError(
                            "maximum separation events per position exceeded"
                        )
                    old_prepared = trial_prepared
                    old_topology = trial_topology
                    candidate_topology = _qualified_event_topology(
                        imported_mesh, old_topology, converged, candidates, committed
                    )
                    if (converged.solve_mode is SolveMode.INERTIAL_TRANSITION
                            and old_prepared.circle_edge_quadrature_order):
                        proposed_prepared=prepare_contact_mesh(fem_case,imported_mesh,
                            topology=candidate_topology,edge_history=True)
                        if not _circle_boundary_release_ready(case,old_prepared,converged,proposed_prepared,
                                normal_penalty=normal_penalty):
                            # D=1 does not release a loaded contact boundary.
                            # Keep this valid physical step and await release;
                            # no topology or history is committed by the probe.
                            candidates=()
                            break
                    target_reference = np.asarray(
                        (
                            target.reference_x_m,
                            target.reference_y_m,
                        )
                        if target.reference_x_m is not None
                        else (target.center_x_m, target.center_y_m),
                        dtype=float,
                    )
                    reference_path_length = float(
                        np.linalg.norm(
                            target_reference - np.asarray(start, dtype=float)
                        )
                    )
                    separation_commit = _begin_separating_active_set(
                        case,
                        old_prepared,
                        trial_committed,
                        converged,
                        tuple(candidates),
                    )
                    terminal = separation_commit if separation_commit.solve_mode is SolveMode.INERTIAL_TRANSITION else advance_separation_constrained_increment(
                        fem_case,
                        old_prepared,
                        separation_commit,
                        grain_at_marker=lambda marker: _grain(
                            case,
                            _interpolated_target(
                                start, target, marker - target_index
                            ),
                        ),
                        marker_bounds=(float(target_index), float(target_index + 1)),
                        reference_path_length_m=reference_path_length,
                        target_separation_progress=1.0,
                        normal_penalty_Pa_per_m=normal_penalty,
                        tangential_penalty_Pa_per_m=tangential_penalty,
                        friction_coefficient=case.contact.friction_coefficient,
                        friction_regularization_ratio=(
                            case.contact.friction_regularization_ratio
                        ),
                        damage_input=case.damage,
                        minimum_substep_fraction=(
                            case.step_control.minimum_substep_fraction
                        ),
                    )
                    energy_segments.append((converged, terminal, old_prepared))
                    converged = terminal
                    event_fraction = converged.marker - target_index
                    event_damage_energy = _separation_event_energy_J(
                        case,
                        old_prepared,
                        converged,
                        tuple(candidates),
                    )
                    if progress is not None:
                        progress(
                            target_index,
                            len(case.trajectory.targets),
                            f"正在定位并提交损伤分离事件，累计候选 {len(candidates)} 个背景单元",
                        )
                    trial_topology = candidate_topology
                    trial_prepared = prepare_contact_mesh(
                        fem_case, imported_mesh, topology=trial_topology,
                        edge_history=case.single_grain_contact_schema_version==3,
                    )
                    trial_prepared=replace(trial_prepared,circle_edge_quadrature_order=
                        getattr(old_prepared,'circle_edge_quadrature_order',0))
                    rebased = rebase_contact_state_after_topology_transition(
                        fem_case,
                        old_prepared,
                        converged,
                        trial_prepared,
                        damage_input=case.damage,
                        density_kg_per_m3=(case.material.density_kg_per_m3 if case.transient else None),
                        kinetic_energy_absolute_tolerance_J=max(1e-14,1e-6*converged.transient_maximum_energy_scale_J),
                    )
                    event_target = _interpolated_target(
                        start, target, event_fraction
                    )
                    topology_measures = _measure_zero_force_topology_transaction(
                        case, old_prepared, converged, trial_prepared, rebased,
                        event_target, normal_penalty=normal_penalty,
                        tangential_penalty=tangential_penalty,
                    )
                    if converged.solve_mode is SolveMode.INERTIAL_TRANSITION:
                        topology_absolute_error += abs(topology_measures['topology_energy_jump_J'])
                    _verify_zero_force_separation_commit(
                        case, old_prepared, converged,
                        reference_path_length_m=reference_path_length,
                        topology_measures=topology_measures,
                        boundary_change=True,
                        removed_dof_kinetic_energy_J=(converged.kinetic_energy_J-rebased.kinetic_energy_J),
                        kinetic_energy_absolute_tolerance_J=max(1e-14,1e-6*converged.transient_maximum_energy_scale_J),
                    )
                    rebalanced = _evaluate_dynamic_topology_state(case,trial_prepared,rebased,converged,
                        normal_penalty=normal_penalty,tangential_penalty=tangential_penalty
                    ) if converged.solve_mode is SolveMode.INERTIAL_TRANSITION else _solve_target(
                        case,
                        trial_prepared,
                        rebased,
                        event_target,
                        normal_penalty=normal_penalty,
                        tangential_penalty=tangential_penalty,
                        marker=target_index + event_fraction,
                    )
                    energy_segments.append((converged, rebalanced, trial_prepared))
                    coordinates = np.asarray(imported_mesh.mesh.p.T, dtype=float)
                    connectivity = np.asarray(imported_mesh.mesh.t.T, dtype=np.int32)
                    removed_area = math.fsum(
                        0.5
                        * abs(
                            float(first[0] * second[1] - first[1] * second[0])
                        )
                        for element_id in candidates
                        for first, second in [
                            (
                                coordinates[connectivity[element_id, 1]]
                                - coordinates[connectivity[element_id, 0]],
                                coordinates[connectivity[element_id, 2]]
                                - coordinates[connectivity[element_id, 0]],
                            )
                        ]
                    )
                    trial_events.append(
                        SeparationEventRecord(
                            event_index=len(separation_events) + len(trial_events) + 1,
                            target_index=target_index,
                            trajectory_fraction=event_fraction,
                            marker=target_index + event_fraction,
                            background_element_ids=tuple(candidates),
                            removed_area_m2=removed_area,
                            active_degrees_of_freedom_before=old_prepared.total_degrees_of_freedom,
                            active_degrees_of_freedom_after=trial_prepared.total_degrees_of_freedom,
                            free_surface_edge_count_before=old_topology.free_surface.edge_node_ids.shape[0],
                            free_surface_edge_count_after=trial_topology.free_surface.edge_node_ids.shape[0],
                            damage_dissipation_increment_J=max(
                                0.0, event_damage_energy
                            ),
                        )
                    )
                    trial_committed = rebalanced
                    converged = rebalanced
                    candidates = _damage_candidates(
                        case,
                        converged,
                        accepted_events_at_position=accepted_events_at_position,
                        prepared=trial_prepared,
                    )
                converged_applicability = _applicability(
                    case, trial_prepared, converged
                )
            except (
                ContactSolverError,
                ConstrainedSeparationEventError,
                ApplicabilityError,
                MaterialTopologyError,
                SeparationEventError,
            ) as exc:
                if isinstance(exc, (EventPathQualificationError, ZeroForceTopologyTransactionError)):
                    # A converged internal event is not a failed Newton trial.
                    # Preserve its topology cause instead of hiding it behind retries.
                    raise
                if (physical_energy_retry or committed.solve_mode is SolveMode.INERTIAL_TRANSITION
                        or dynamic_trial_completed
                        or getattr(exc, 'failure_stage', None) == 'inertial_same_contact_step'):
                    # The physical integrator already exhausted its registered
                    # dt reductions; changing an unused geometric fraction
                    # would replay the identical failed step.
                    raise
                if isinstance(exc, ContactConvergenceError):
                    exc.substep_fraction = fraction
                reduced_fraction = 0.5 * fraction
                if (
                    isinstance(exc, ContactConvergenceError)
                    and case.damage is not None
                    and len(exc.event_trial_background_element_ids)
                    == len(exc.event_trial_damage_states)
                    == len(committed.damage_states)
                ):
                    proposal = event_aware_substep_scale(
                        background_element_ids=exc.event_trial_background_element_ids,
                        committed_damage_states=committed.damage_states,
                        trial_damage_states=exc.event_trial_damage_states,
                        damage_event_tolerance=case.damage.damage_event_tolerance,
                        fracture_energy_J_per_m2=case.damage.fracture_energy_J_per_m2,
                    )
                    if proposal is not None:
                        reduced_fraction = min(
                            reduced_fraction,
                            fraction * proposal.scale,
                        )
                if (
                    not case.step_control.allow_reduction
                    or not retry_budget.can_retry
                    or reduced_fraction < case.step_control.minimum_substep_fraction
                ):
                    if isinstance(exc, ApplicabilityError):
                        raise ContactConvergenceError(
                            f"contact applicability hard stop: {exc}",
                            attempted_marker=target_index + attempted,
                            residual_norm_history_N=(),
                        ) from exc
                    raise
                fraction = reduced_fraction
                retry_budget.record_failure()
                continue
            saved_increments=(contact_work_increment,friction_dissipation_increment,
                              plastic_dissipation_increment,damage_dissipation_increment)
            window_work=window_dissipation=0.
            for segment_previous, segment_state, segment_prepared in energy_segments:
                substep_energies = _increment_energies(
                    case, segment_prepared, segment_previous, segment_state
                )
                contact_work_increment += substep_energies[0]
                friction_dissipation_increment += substep_energies[1]
                plastic_dissipation_increment += substep_energies[2]
                window_work+=substep_energies[0]
                window_dissipation+=substep_energies[1]+substep_energies[2]
                if case.damage is not None:
                    damage_dissipation_increment += _damage_dissipation_increment_J(
                        case, segment_prepared, segment_previous, segment_state
                    )
                    window_dissipation+=_damage_dissipation_increment_J(case,segment_prepared,segment_previous,segment_state)
            if case.damage is not None:
                def stored_window(p,s):
                    return _elastic_strain_energy(case,p,s)+_hardening_stored_energy(case,p,s)+_numerical_contact_stored_energy(case,p,s,
                        normal_penalty_Pa_per_m=normal_penalty,tangential_penalty_Pa_per_m=tangential_penalty)+float(getattr(s,'kinetic_energy_J',0.0))
                error=window_work-(stored_window(trial_prepared,converged)-stored_window(prepared,committed))-window_dissipation
                try:
                    if case.transient is not None:
                        if (committed.solve_mode is not SolveMode.INERTIAL_TRANSITION
                                and converged.solve_mode is not SolveMode.INERTIAL_TRANSITION):
                            remaining_time = (float(np.linalg.norm(future_references[0]
                                -np.asarray(committed.grain_reference_m)))+future_distance)/case.transient.grain_speed_m_per_s
                            dt = converged.physical_time_s-committed.physical_time_s
                            if dt > 0. and remaining_time > 0.:
                                local_allowance = _transient_local_error_allowance(
                                    used_J=committed.transient_absolute_energy_error_J,
                                    scale_J=max(committed.transient_maximum_energy_scale_J,
                                        abs(stored_window(prepared,committed)),abs(stored_window(trial_prepared,converged)),
                                        abs(window_work),abs(window_dissipation)),
                                    dt_s=dt,remaining_time_s=remaining_time)
                                if abs(error) > local_allowance:
                                    raise ContactSolverError('quasistatic increment would consume future physical energy budget')
                        converged=_audit_transient_commit_window(committed,converged,work_J=window_work,
                            dissipation_J=window_dissipation,stored_before_J=stored_window(prepared,committed),
                            stored_after_J=stored_window(trial_prepared,converged),
                            topology_error_J=topology_absolute_error)
                    next_budget=energy_budget.accept(prefix_work+window_work,error,event=bool(trial_events))
                except ContactSolverError as energy_failure:
                    (contact_work_increment,friction_dissipation_increment,
                     plastic_dissipation_increment,damage_dissipation_increment)=saved_increments
                    if (converged.solve_mode is SolveMode.INERTIAL_TRANSITION
                        or committed.solve_mode is SolveMode.INERTIAL_TRANSITION
                        or physical_energy_retry
                        or not case.step_control.allow_reduction or not retry_budget.can_retry
                        or .5*fraction<case.step_control.minimum_substep_fraction):
                        # The dynamic solver owns physical dt refinement.
                        # This final audit also includes topology and ledger
                        # mapping: preserve its rejected window for diagnosis.
                        # Halving an unused geometric fraction repeats exactly
                        # the same physical trial from the same committed state.
                        if (case.transient is not None and not physical_energy_retry
                                and committed.solve_mode is not SolveMode.INERTIAL_TRANSITION
                                and converged.solve_mode is not SolveMode.INERTIAL_TRANSITION
                                and not trial_events):
                            # Reject the static trial in full. The next solve
                            # advances real time from the unchanged accepted state.
                            physical_energy_retry = True
                            continue
                        energy_failure.energy_failure_snapshot=(case,prepared,committed,trial_prepared,converged,
                            energy_segments,normal_penalty,tangential_penalty,energy_budget)
                        energy_failure.failure_stage='accepted_prefix_energy'
                        energy_failure.substep_fraction=fraction
                        energy_failure.window_work_J=window_work
                        energy_failure.window_dissipation_J=window_dissipation
                        raise
                    fraction*=.5
                    retry_budget.record_failure()
                    continue
                energy_budget=next_budget
                prefix_work+=window_work
            separation_event_energy_increment += math.fsum(
                event.damage_dissipation_increment_J for event in trial_events
            )
            if case.damage is not None and not trial_events:
                from .path_continuation import AcceptedPathReference
                delta_u=np.asarray(converged.displacement_vector_m)-np.asarray(committed.displacement_vector_m)
                delta_marker=float(converged.marker-committed.marker)
                if delta_marker!=0. and np.any(delta_u):
                    distance=float(np.linalg.norm(np.asarray(converged.grain_reference_m)-np.asarray(committed.grain_reference_m)))
                    scale=max(float(np.median(trial_prepared.candidate_tributary_lengths_m)),np.finfo(float).tiny)
                    reference=AcceptedPathReference(
                        background_node_ids=tuple(int(n) for n in trial_prepared.active_to_background_node_ids),
                        displacement_direction_m=tuple(tuple(float(v) for v in row) for row in delta_u.reshape(-1,2)),
                        marker_direction=delta_marker,displacement_scale_m=scale,
                        marker_scale_m=max(distance/abs(delta_marker),np.finfo(float).tiny),
                        source_markers=(float(committed.marker),float(converged.marker)),source='accepted_increment')
                    converged=replace(converged,accepted_path_reference=reference)
            prepared = trial_prepared
            material_topology = trial_topology
            committed = converged
            physical_energy_retry = False
            separation_events.extend(trial_events)
            achieved = event_fraction if trial_events or constrained_event_located else attempted
            substep_count += 1
            retry_budget.record_successful_commit()
            fraction = min(fraction * 2.0, 1.0 - achieved)
            if checkpoint_callback is not None:
                import copy
                checkpoint_callback(copy.deepcopy(dict(
                    format='grindcae_trajectory_substep_v1',case_input=case.to_dict(),background_mesh=imported_mesh,
                    target_index=target_index,target_start_state=target_start_state,start=start,
                    fraction=fraction,achieved=achieved,retry_budget=retry_budget,substep_count=substep_count,
                    converged_applicability=converged_applicability,prepared=prepared,
                    material_topology=material_topology,committed=committed,records=records,separation_events=separation_events,
                    energy_budget=energy_budget,prefix_work=prefix_work,normal_penalty=normal_penalty,tangential_penalty=tangential_penalty,
                    cumulative_contact_work=cumulative_contact_work,cumulative_friction_dissipation=cumulative_friction_dissipation,
                    cumulative_plastic_dissipation=cumulative_plastic_dissipation,cumulative_damage_dissipation=cumulative_damage_dissipation,
                    cumulative_separation_event_energy=cumulative_separation_event_energy,
                    contact_work_increment=contact_work_increment,friction_dissipation_increment=friction_dissipation_increment,
                    plastic_dissipation_increment=plastic_dissipation_increment,damage_dissipation_increment=damage_dissipation_increment,
                    separation_event_energy_increment=separation_event_energy_increment)))
        completed_record = _record(
                case,
                target_index,
                target,
                committed,
                retry_count=retry_budget.total_failures,
                substep_count=substep_count,
                previous_state=target_start_state,
                prepared=prepared,
                applicability=converged_applicability,
                contact_work_increment_J=contact_work_increment,
                friction_dissipation_increment_J=friction_dissipation_increment,
                plastic_dissipation_increment_J=plastic_dissipation_increment,
                normal_penalty_Pa_per_m=normal_penalty,
                tangential_penalty_Pa_per_m=tangential_penalty,
                baseline_mechanical_energy_J=baseline_mechanical_energy,
                cumulative_contact_work_J=cumulative_contact_work + contact_work_increment,
                cumulative_friction_dissipation_J=cumulative_friction_dissipation + friction_dissipation_increment,
                cumulative_plastic_dissipation_J=cumulative_plastic_dissipation + plastic_dissipation_increment,
                cumulative_damage_dissipation_J=cumulative_damage_dissipation + damage_dissipation_increment,
                cumulative_separation_event_energy_J=cumulative_separation_event_energy + separation_event_energy_increment,
                damage_dissipation_increment_J=damage_dissipation_increment,
                separation_event_energy_increment_J=separation_event_energy_increment,
            )
        if material_topology is not None:
            completed_record = ContactTrajectoryRecord(
                **{
                    name: getattr(completed_record, name)
                    for name in completed_record.__dataclass_fields__
                    if name != "material_topology"
                },
                material_topology=material_topology,
            )
        records.append(completed_record)
        cumulative_contact_work += contact_work_increment
        cumulative_friction_dissipation += friction_dissipation_increment
        cumulative_plastic_dissipation += plastic_dissipation_increment
        cumulative_damage_dissipation += damage_dissipation_increment
        cumulative_separation_event_energy += separation_event_energy_increment
        if progress is not None:
            progress(
                target_index + 1,
                len(case.trajectory.targets),
                f"已提交单磨粒轨迹位置 {target_index + 1}/{len(case.trajectory.targets)}",
            )
    aggregate_applicability = _aggregate_applicability(records)
    final_record = records[-1]
    energy_scale = max(
        abs(final_record.cumulative_contact_work_J),
        abs(
            final_record.elastic_strain_energy_J
            + final_record.hardening_stored_energy_J
            + final_record.numerical_contact_stored_energy_J
            + final_record.state.kinetic_energy_J
            - baseline_mechanical_energy
        )
        + final_record.cumulative_friction_dissipation_J
        + final_record.cumulative_plastic_dissipation_J
        + final_record.cumulative_damage_dissipation_J,
        1.0e-30,
    )
    damage_history: tuple[DamageHistoryRecord, ...] = ()
    if case.damage is not None:
        background_areas = _background_element_areas_m2(imported_mesh)
        history_items: list[DamageHistoryRecord] = []
        for record in records:
            states = record.state.damage_states
            dissipation = math.fsum(
                state.damage_dissipation_density_J_per_m3
                * float(area)
                * case.analysis.thickness
                for state, area in zip(
                    states,
                    background_areas[
                        np.asarray(record.state.background_element_ids, dtype=np.int32)
                    ],
                    strict=True,
                )
            )
            history_items.append(
                DamageHistoryRecord(
                    record_index=record.index,
                    damaged_element_count=sum(state.damage > 0.0 for state in states),
                    maximum_damage=max((state.damage for state in states), default=0.0),
                    mean_damage=(
                        math.fsum(state.damage for state in states) / len(states)
                        if states
                        else 0.0
                    ),
                    damage_dissipation_J=record.cumulative_damage_dissipation_J,
                    separation_event_count=sum(
                        event.target_index <= record.index
                        for event in separation_events
                    ),
                )
            )
        damage_history = tuple(history_items)
    result = ContactTrajectoryResult(
        case=case,
        prepared_mesh=prepared,
        records=tuple(records),
        reference_mesh_path=reference_path,
        normal_penalty_Pa_per_m=normal_penalty,
        tangential_penalty_Pa_per_m=tangential_penalty,
        friction_dissipation_J=final_record.cumulative_friction_dissipation_J,
        contact_work_J=final_record.cumulative_contact_work_J,
        plastic_dissipation_J=final_record.cumulative_plastic_dissipation_J,
        damage_dissipation_J=final_record.cumulative_damage_dissipation_J,
        separation_event_energy_J=final_record.cumulative_separation_event_energy_J,
        initial_elastic_strain_energy_J=initial_record.elastic_strain_energy_J,
        final_elastic_strain_energy_J=final_record.elastic_strain_energy_J,
        initial_numerical_contact_stored_energy_J=initial_record.numerical_contact_stored_energy_J,
        final_numerical_contact_stored_energy_J=final_record.numerical_contact_stored_energy_J,
        energy_balance_residual_J=final_record.energy_balance_residual_J,
        energy_balance_relative_residual=abs(final_record.energy_balance_residual_J) / energy_scale,
        maximum_balance_residual_N=max(record.balance_residual_N for record in records),
        applicability_status=aggregate_applicability.status,
        applicability=aggregate_applicability,
        material_topology=material_topology,
        background_mesh=imported_mesh,
        separation_events=tuple(separation_events),
        automatic_removed_element_ids=tuple(
            sorted(
                element_id
                for event in separation_events
                for element_id in event.background_element_ids
            )
        ),
        damage_history=damage_history,
    )
    # Keep an explicitly requested workspace artifact.  An internal temporary
    # mesh is allowed to disappear after all mesh data have been imported.
    if temporary is not None:
        temporary.cleanup()
    return result
