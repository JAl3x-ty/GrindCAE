"""Coupled small-strain J2 and rigid circular-grain normal-contact solver."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from dataclasses import dataclass, field, replace
import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from skfem import Basis, ElementTriP1, ElementVector, MeshTri

from grindcae.elastoplastic import MaterialPointState
from grindcae.elastoplastic_fem.models import ElastoplasticFemCase
from grindcae.elastoplastic_fem.solver import (
    PreparedElastoplasticMesh,
    _prepare_mesh_with_load,
    _assemble_trial,
    initial_structure_state,
    prepare_elastoplastic_mesh,
)
from grindcae.solver import ImportedMesh

from .contact_law import (
    ContactPointState,
    friction_contact_response,
    normal_contact_response,
    update_augmented_normal_multiplier,
)


def _compatible_slot_state(instance: object) -> list[object]:
    values = []
    for definition in dataclasses.fields(instance):
        if hasattr(instance, definition.name):
            values.append(getattr(instance, definition.name))
        elif definition.default is not dataclasses.MISSING:
            values.append(definition.default)
        elif definition.default_factory is not dataclasses.MISSING:
            values.append(definition.default_factory())
        else:
            raise AttributeError(definition.name)
    return values


def _restore_compatible_slot_state(instance: object, state: object) -> None:
    values = list(state)
    definitions = dataclasses.fields(instance)
    if len(values) > len(definitions):
        raise ValueError("saved dataclass state has unexpected fields")
    for index, definition in enumerate(definitions):
        if index < len(values):
            value = values[index]
        elif definition.default is not dataclasses.MISSING:
            value = definition.default
        elif definition.default_factory is not dataclasses.MISSING:
            value = definition.default_factory()
        else:
            raise ValueError(f"saved dataclass state misses {definition.name}")
        object.__setattr__(instance, definition.name, value)
from .geometry import (
    ContactGeometryError,
    RigidAnalyticalGrain,
    RigidContactQuery,
    rigid_contact_query,
    ordered_boundary_tributary_lengths,
)
from .material_topology import PreparedMaterialTopology
from .path_continuation import PathState, AcceptedPathReference
from .path_continuation import (
    PathContinuationError,
    PseudoArcLengthSettings,
    initialize_equilibrium_path_state,
    initialize_path_state,
    locate_path_event,
    pseudo_arclength_step,
)
from .continuous_separation import (
    SeparatingState,
    SolveFailure,
    SolveMode,
    advance_separating,
    initialize_separation_path_direction,
    separation_constraint_terms,
)
from .damage_coupling import (
    damage_element_trial_response,
    separating_element_trial_response,
)
from .damage_material import DamageState, initial_damage_state
from .damage_models import DamageInput
from .separation_events import event_aware_substep_scale
from .thermodynamic_energy import DamagePathConstraintRequired


class ContactSolverError(RuntimeError):
    """Raised when the coupled contact solve cannot proceed."""


class ContactConvergenceError(ContactSolverError):
    """Raised when a contact increment reaches its Newton iteration limit."""

    def __init__(
        self,
        message: str,
        *,
        attempted_marker: float = math.nan,
        residual_norm_history_N: tuple[float, ...] = (),
        line_search_trial_norms_N: tuple[float, ...] = (),
        contact_open_stick_slip_counts: tuple[tuple[int, int, int], ...] = (),
        maximum_damage: float = 0.0,
        near_initiation_background_ids: tuple[int, ...] = (),
        near_separation_background_ids: tuple[int, ...] = (),
        damage_branch_counts: tuple[tuple[tuple[str, int], ...], ...] = (),
        minimum_and_maximum_characteristic_length_m: tuple[float, float] | None = None,
        tangent_symmetry_error: float | None = None,
        tangent_condition_indicator: float | None = None,
        finite_difference_branch_crossing_count: int = 0,
        event_trial_background_element_ids: tuple[int, ...] = (),
        event_trial_damage_states: tuple[DamageState, ...] = (),
        substep_fraction: float | None = None,
    ) -> None:
        # BaseException pickle reconstruction calls the constructor with args
        # (the message) before restoring __dict__. Defaults let existing saved
        # failures recover their full keyword diagnostics without rewriting them.
        super().__init__(message)
        self.attempted_marker = attempted_marker
        self.residual_norm_history_N = residual_norm_history_N
        self.line_search_trial_norms_N = line_search_trial_norms_N
        self.contact_open_stick_slip_counts = contact_open_stick_slip_counts
        self.maximum_damage = maximum_damage
        self.near_initiation_background_ids = near_initiation_background_ids
        self.near_separation_background_ids = near_separation_background_ids
        self.damage_branch_counts = damage_branch_counts
        self.minimum_and_maximum_characteristic_length_m = (
            minimum_and_maximum_characteristic_length_m
        )
        self.tangent_symmetry_error = tangent_symmetry_error
        self.tangent_condition_indicator = tangent_condition_indicator
        self.finite_difference_branch_crossing_count = (
            finite_difference_branch_crossing_count
        )
        self.event_trial_background_element_ids = event_trial_background_element_ids
        self.event_trial_damage_states = event_trial_damage_states
        self.substep_fraction = substep_fraction


class ConstrainedSeparationEventError(ContactSolverError):
    """Raised when the augmented equilibrium path cannot reach separation."""


def _newton_failure_diagnostics(
    prepared: "PreparedContactMesh",
    body_trial: object,
    contact_trial: object,
    tangent_free: csr_matrix,
    damage_input: DamageInput | None,
) -> dict[str, object]:
    statuses = [state.status for state in contact_trial.states]
    damage_states = tuple(getattr(body_trial, "damage_states", ()))
    background_ids = np.asarray(
        prepared.active_to_background_element_ids
        if prepared.active_to_background_element_ids.size
        else np.arange(prepared.element_count, dtype=np.int32),
        dtype=np.int32,
    )
    branches = tuple(getattr(body_trial, "tangent_branches", ()))
    branch_counts = tuple(
        sorted((name, branches.count(name)) for name in set(branches))
    )
    near_initiation: list[int] = []
    near_separation: list[int] = []
    if damage_input is not None:
        terminal = 1.0 - damage_input.damage_event_tolerance
        for background, state in zip(background_ids, damage_states, strict=True):
            if (
                state.initiation_equivalent_plastic_strain is None
                and state.omega >= 0.95
            ) or (
                state.initiation_equivalent_plastic_strain is not None
                and state.damage <= 0.05
            ):
                near_initiation.append(int(background))
            if state.damage >= 0.95 * terminal:
                near_separation.append(int(background))
    dense = tangent_free.toarray()
    tangent_norm = float(np.linalg.norm(dense))
    symmetry_error = float(
        np.linalg.norm(dense - dense.T) / max(tangent_norm, np.finfo(float).tiny)
    )
    try:
        condition_indicator = float(np.linalg.cond(dense))
    except np.linalg.LinAlgError:
        condition_indicator = math.inf
    lengths = np.asarray(
        getattr(body_trial, "characteristic_lengths_m", ()), dtype=float
    )
    return {
        "contact_open_stick_slip_counts": ((
            statuses.count("open"),
            statuses.count("stick"),
            statuses.count("slip"),
        ),),
        "maximum_damage": max((state.damage for state in damage_states), default=0.0),
        "near_initiation_background_ids": tuple(near_initiation),
        "near_separation_background_ids": tuple(near_separation),
        "damage_branch_counts": (branch_counts,),
        "minimum_and_maximum_characteristic_length_m": (
            (float(np.min(lengths)), float(np.max(lengths)))
            if lengths.size
            else None
        ),
        "tangent_symmetry_error": symmetry_error,
        "tangent_condition_indicator": condition_indicator,
        "finite_difference_branch_crossing_count": int(
            getattr(body_trial, "finite_difference_branch_crossing_count", 0)
        ),
    }


@dataclass(frozen=True, slots=True)
class PreparedContactMesh:
    # Schema3 endpoint ownership; legacy nodal candidates remain presentation views.
    structural: PreparedElastoplasticMesh
    candidate_node_ids: NDArray[np.int32] = field(repr=False)
    candidate_component_dofs: NDArray[np.int32] = field(repr=False)
    candidate_reference_coordinates_m: NDArray[np.float64] = field(repr=False)
    candidate_tributary_lengths_m: NDArray[np.float64] = field(repr=False)
    candidate_keys: tuple[tuple[int, tuple[int, ...]], ...] = ()
    candidate_background_node_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    active_to_background_node_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    background_to_active_node_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    active_to_background_element_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    candidate_incident_edge_ids: tuple[tuple[int, ...], ...] = ()
    candidate_reference_normals: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 2), dtype=float), repr=False
    )
    candidate_reference_normal_unique: NDArray[np.bool_] = field(
        default_factory=lambda: np.empty(0, dtype=bool), repr=False
    )
    candidate_surface_component_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    candidate_surface_classifications: tuple[str, ...] = ()
    edge_endpoint_keys: tuple[tuple[int, int, int], ...] = ()
    consistent_edge_friction_tangent: bool = False
    circle_edge_quadrature_order: int = 0

    @property
    def total_degrees_of_freedom(self) -> int:
        return self.structural.total_degrees_of_freedom

    @property
    def element_count(self) -> int:
        return self.structural.element_count


@dataclass(frozen=True, slots=True)
class CommittedContactStructureState:
    marker: float
    displacement_vector_m: NDArray[np.float64] = field(repr=False)
    element_states: tuple[MaterialPointState, ...]
    contact_states: tuple[ContactPointState, ...]
    grain_center_m: tuple[float, float] | None = None
    candidate_keys: tuple[tuple[int, tuple[int, ...]], ...] = ()
    grain_reference_m: tuple[float, float] | None = None
    damage_states: tuple[DamageState, ...] = ()
    background_element_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    path_state: PathState | None = field(default=None, repr=False)
    solve_mode: SolveMode = SolveMode.STANDARD_NEWTON
    solve_failure: SolveFailure | None = None
    separating_states: tuple[tuple[int, SeparatingState], ...] = ()
    edge_contact_states: tuple = ()
    archived_edge_contact_states: tuple = ()
    accepted_path_reference: AcceptedPathReference | None = None
    physical_time_s: float = 0.0
    velocity_vector_m_per_s: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float), repr=False
    )
    acceleration_vector_m_per_s2: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float), repr=False
    )
    kinetic_energy_J: float = 0.0
    exported_kinetic_energy_J: float = 0.0
    exported_momentum_kg_m_per_s: tuple[float, float] = (0.0, 0.0)
    transient_absolute_energy_error_J: float = 0.0
    transient_maximum_energy_scale_J: float = 0.0
    transient_stable_step_count: int = 0
    transient_step_count: int = 0
    transient_last_time_step_s: float = 0.0

    def __getstate__(self):
        return _compatible_slot_state(self)

    def __setstate__(self, state):
        _restore_compatible_slot_state(self, state)


@dataclass(frozen=True, slots=True)
class ConvergedContactIncrement:
    marker: float
    displacement_vector_m: NDArray[np.float64] = field(repr=False)
    element_states: tuple[MaterialPointState, ...]
    contact_states: tuple[ContactPointState, ...]
    contact_kinematics: tuple[RigidContactQuery, ...]
    contact_pressure_Pa: NDArray[np.float64] = field(repr=False)
    contact_force_vector_N: NDArray[np.float64] = field(repr=False)
    internal_force_vector_N: NDArray[np.float64] = field(repr=False)
    element_total_strain: NDArray[np.float64] = field(repr=False)
    element_stress_Pa: NDArray[np.float64] = field(repr=False)
    element_sigma_zz_Pa: NDArray[np.float64] = field(repr=False)
    newton_iterations: int
    residual_norm_N: float
    residual_tolerance_N: float
    correction_norm_m: float
    correction_tolerance_m: float
    reference_force_N: float
    reference_contact_size_m: float
    reference_displacement_m: float
    residual_absolute_tolerance_N: float
    displacement_absolute_tolerance_m: float
    residual_norm_history_N: tuple[float, ...]
    total_contact_force_N: NDArray[np.float64]
    rigid_grain_reaction_N: NDArray[np.float64]
    fixed_reaction_x_N: float
    fixed_reaction_y_N: float
    balance_residual_x_N: float
    balance_residual_y_N: float
    maximum_penetration_m: float
    grain_center_m: tuple[float, float] | None
    candidate_keys: tuple[tuple[int, tuple[int, ...]], ...]
    grain_reference_m: tuple[float, float]
    friction_dissipation_increment_J: float
    augmented_iterations: int
    damage_states: tuple[DamageState, ...] = ()
    background_element_ids: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32), repr=False
    )
    path_state: PathState | None = field(default=None, repr=False)
    solve_mode: SolveMode = SolveMode.STANDARD_NEWTON
    solve_failure: SolveFailure | None = None
    separating_states: tuple[tuple[int, SeparatingState], ...] = ()
    separation_path_residual_m: float = 0.0
    separation_constraint_residual: float = 0.0
    edge_contact_states: tuple = ()
    archived_edge_contact_states: tuple = ()
    accepted_path_reference: AcceptedPathReference | None = None
    physical_time_s: float = 0.0
    velocity_vector_m_per_s: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float), repr=False
    )
    acceleration_vector_m_per_s2: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float), repr=False
    )
    kinetic_energy_J: float = 0.0
    exported_kinetic_energy_J: float = 0.0
    exported_momentum_kg_m_per_s: tuple[float, float] = (0.0, 0.0)
    transient_absolute_energy_error_J: float = 0.0
    transient_maximum_energy_scale_J: float = 0.0
    transient_stable_step_count: int = 0
    transient_step_count: int = 0
    transient_last_time_step_s: float = 0.0

    def __getstate__(self):
        return _compatible_slot_state(self)

    def __setstate__(self, state):
        _restore_compatible_slot_state(self, state)


def _replace_compatible_dataclass(instance: object, /, **changes: object):
    """Replace current fields while tolerating older pickled class snapshots."""

    values = {}
    for name, definition in instance.__dataclass_fields__.items():
        if name in changes:
            values[name] = changes[name]
        elif hasattr(instance, name):
            values[name] = getattr(instance, name)
        elif definition.default is not dataclasses.MISSING:
            values[name] = definition.default
        elif definition.default_factory is not dataclasses.MISSING:
            values[name] = definition.default_factory()
        else:
            raise AttributeError(name)
    return type(instance)(**values)


@dataclass(frozen=True, slots=True)
class _ContactAssembly:
    force_N: NDArray[np.float64]
    residual_tangent_N_per_m: csr_matrix
    pressures_Pa: NDArray[np.float64]
    states: tuple[ContactPointState, ...]
    kinematics: tuple[RigidContactQuery, ...]
    friction_dissipation_increment_J: float
    edge_contact_states: tuple = ()
    integrated_maximum_penetration_m: float = 0.
    integrated_edge_patches: tuple = ()


@dataclass(frozen=True, slots=True)
class _DamageBodyAssembly:
    internal_force_N: NDArray[np.float64]
    tangent_stiffness_N_per_m: csr_matrix
    element_states: tuple[MaterialPointState, ...]
    damage_states: tuple[DamageState, ...]
    element_total_strain: NDArray[np.float64]
    element_stress_Pa: NDArray[np.float64]
    element_sigma_zz_Pa: NDArray[np.float64]
    tangent_branches: tuple[str, ...]
    finite_difference_branch_crossing_count: int
    characteristic_lengths_m: NDArray[np.float64]
    element_internal_force_vectors_N: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 6), dtype=float), repr=False
    )
    separation_background_element_ids: tuple[int, ...] = ()
    separating_states: tuple[tuple[int, SeparatingState], ...] = ()
    internal_force_progress_derivatives_N: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )
    separation_constraint_residuals_m: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float), repr=False
    )
    separation_constraint_displacement_derivatives: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )
    separation_constraint_progress_derivatives_m: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )


@dataclass(frozen=True, slots=True)
class _CommittedDamageFields:
    internal_force_N: NDArray[np.float64]
    element_states: tuple[MaterialPointState, ...]
    damage_states: tuple[DamageState, ...]
    element_total_strain: NDArray[np.float64]
    element_stress_Pa: NDArray[np.float64]
    element_sigma_zz_Pa: NDArray[np.float64]


def _committed_damage_fields(case, prepared, committed):
    """Recover fields from accepted history without a constitutive trial."""
    internal = np.zeros(prepared.total_degrees_of_freedom)
    strains = np.empty((prepared.element_count, 3))
    stresses = np.empty((prepared.element_count, 3))
    sigma_zz = np.empty(prepared.element_count)
    for index, (point, damage, B, dofs, area) in enumerate(zip(
            committed.element_states, committed.damage_states,
            prepared.structural.element_B_matrices_per_m,
            prepared.structural.element_dofs,
            prepared.structural.element_areas_m2, strict=True)):
        stress = (1.0-damage.damage)*np.asarray(point.stress_tensor_Pa)
        stresses[index] = (stress[0, 0], stress[1, 1], stress[0, 1])
        sigma_zz[index] = stress[2, 2]
        strains[index] = B@committed.displacement_vector_m[dofs]
        internal[dofs] += B.T@stresses[index]*float(area)*case.analysis.thickness
    return _CommittedDamageFields(internal, committed.element_states,
        committed.damage_states, strains, stresses, sigma_zz)


def _assemble_damage_trial(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    displacement: NDArray[np.float64],
    damage_input: DamageInput,
    *,
    separation_progress_by_background: dict[int, float] | None = None,
    active_side_direction: NDArray[np.float64] | None = None,
) -> _DamageBodyAssembly:
    total_dofs = prepared.total_degrees_of_freedom
    internal = np.zeros(total_dofs, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    material_states: list[MaterialPointState] = []
    damage_states: list[DamageState] = []
    strains = np.empty((prepared.element_count, 3), dtype=float)
    stresses = np.empty((prepared.element_count, 3), dtype=float)
    sigma_zz = np.empty(prepared.element_count, dtype=float)
    tangent_branches: list[str] = []
    branch_crossing_count = 0
    characteristic_lengths = np.empty(prepared.element_count, dtype=float)
    element_internal_forces = np.empty((prepared.element_count, 6), dtype=float)
    separating_by_background = dict(committed.separating_states)
    evaluated_separating = []
    separating_background_ids = tuple(sorted(separating_by_background))
    progress_by_background = (
        {} if separation_progress_by_background is None
        else {int(key): float(value) for key, value in separation_progress_by_background.items()}
    )
    if set(progress_by_background) - set(separating_background_ids):
        raise ContactSolverError("separation progress references an inactive background element")
    progress_columns = np.zeros(
        (total_dofs, len(separating_background_ids)), dtype=float
    )
    separation_residuals = np.zeros(len(separating_background_ids), dtype=float)
    separation_displacement_derivatives = np.zeros(
        (len(separating_background_ids), total_dofs), dtype=float
    )
    separation_progress_derivatives = np.zeros(
        (len(separating_background_ids), len(separating_background_ids)), dtype=float
    )
    progress_column_by_background = {
        background: column
        for column, background in enumerate(separating_background_ids)
    }
    displacement_increment = displacement - committed.displacement_vector_m
    if len(committed.damage_states) != prepared.element_count:
        raise ContactSolverError("committed damage-state count does not match mesh")
    for element_id in range(prepared.element_count):
        dofs = prepared.structural.element_dofs[element_id]
        background_id = int(committed.background_element_ids[element_id])
        separation = separating_by_background.get(background_id)
        if separation is None:
            response = damage_element_trial_response(
                case.material,
                damage_input,
                committed.element_states[element_id],
                committed.damage_states[element_id],
                prepared.structural.element_B_matrices_per_m[element_id],
                prepared.structural.element_areas_m2[element_id],
                thickness_m=case.analysis.thickness,
                displacement_increment=displacement_increment[dofs],
                active_side_direction=(None if active_side_direction is None else active_side_direction[dofs]),
            )
            material_update = response.material_update.material_update
            damage_state = response.material_update.damage_state
            in_plane_stress = np.asarray(
                response.material_update.in_plane_stress_Pa, dtype=float
            )
            out_of_plane_stress = float(response.material_update.sigma_zz_Pa)
            tangent_branch = response.material_update.tangent_branch
            branch_crossing_count += sum(
                response.material_update.finite_difference_branch_crossings
            )
        else:
            progress = progress_by_background.get(background_id, separation.s_e)
            response = separating_element_trial_response(
                case.material,
                committed.element_states[element_id],
                committed.damage_states[element_id],
                separation,
                prepared.structural.element_B_matrices_per_m[element_id],
                prepared.structural.element_areas_m2[element_id],
                thickness_m=case.analysis.thickness,
                displacement_increment=displacement_increment[dofs],
                separation_progress=progress,
                active_side_direction=(None if active_side_direction is None else active_side_direction[dofs]),
            )
            material_update = response.material_update
            damage_state = response.damage_state
            evaluated_separating.append((background_id,response.separation_state))
            degradation = 1.0 - damage_state.damage
            in_plane_stress = degradation * np.asarray(
                material_update.in_plane_stress_Pa, dtype=float
            )
            out_of_plane_stress = degradation * float(material_update.sigma_zz_Pa)
            tangent_branch = "separating"
            progress_column = progress_column_by_background[background_id]
            progress_columns[dofs, progress_column] += (
                response.internal_force_progress_derivative_N
            )
            plastic_displacement = response.characteristic_length_m * (
                response.material_update.state.equivalent_plastic_strain
                - float(
                    committed.damage_states[
                        element_id
                    ].initiation_equivalent_plastic_strain
                )
            )
            separation_residual, separation_progress_derivative = (
                separation_constraint_terms(
                    response.separation_state,
                    current_post_initiation_plastic_displacement_m=(
                        plastic_displacement
                    ),
                    separation_progress=progress,
                )
            )
            separation_residuals[progress_column] = separation_residual
            alpha_gradient = np.asarray(
                response.material_update.equivalent_plastic_strain_gradient,
                dtype=float,
            )
            separation_displacement_derivatives[progress_column, dofs] = (
                response.characteristic_length_m
                * alpha_gradient
                @ prepared.structural.element_B_matrices_per_m[element_id]
            )
            separation_progress_derivatives[progress_column, progress_column] = (
                separation_progress_derivative
            )
            if response.energetic_constraint is not None:
                joint=response.energetic_constraint
                history=separation.energetic_history
                remaining_damage=1.-history.damage
                coordinate_scale=separation.characteristic_length_m*remaining_damage
                energy_scale=history.initial_driving_energy
                # Store length-scaled g=(Y-R)/Y0, matching the existing
                # augmented coordinate normalization without changing units.
                separation_residuals[progress_column]=coordinate_scale*joint.constraint_residual/energy_scale
                separation_displacement_derivatives[progress_column,dofs]=(
                    coordinate_scale/energy_scale*joint.constraint_strain_gradient
                    @ prepared.structural.element_B_matrices_per_m[element_id])
                separation_progress_derivatives[progress_column,progress_column]=(
                    coordinate_scale/energy_scale*joint.constraint_damage_derivative*remaining_damage)
            elif (separation.energetic_history is not None
                    and separation.energetic_history.damage==1.):
                # The naturally completed entry is metadata, with no remaining
                # plastic-opening or joint damage coordinate to constrain.
                separation_residuals[progress_column]=0.
                separation_displacement_derivatives[progress_column,:]=0.
                separation_progress_derivatives[progress_column,:]=0.
        internal[dofs] += response.internal_force_N
        element_internal_forces[element_id] = response.internal_force_N
        rows.extend(np.repeat(dofs, 6).tolist())
        columns.extend(np.tile(dofs, 6).tolist())
        values.extend(response.tangent_stiffness_N_per_m.reshape(-1).tolist())
        material_states.append(material_update.state)
        damage_states.append(damage_state)
        strains[element_id] = (
            prepared.structural.element_B_matrices_per_m[element_id]
            @ displacement[dofs]
        )
        stresses[element_id] = in_plane_stress
        sigma_zz[element_id] = out_of_plane_stress
        tangent_branches.append(tangent_branch)
        characteristic_lengths[element_id] = response.characteristic_length_m
    tangent = coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(total_dofs, total_dofs),
    ).tocsr()
    if not np.all(np.isfinite(internal)) or not np.all(np.isfinite(tangent.data)):
        raise ContactSolverError("damage trial assembly produced a non-finite value")
    return _DamageBodyAssembly(
        internal_force_N=internal,
        tangent_stiffness_N_per_m=tangent,
        element_states=tuple(material_states),
        damage_states=tuple(damage_states),
        element_total_strain=strains,
        element_stress_Pa=stresses,
        element_sigma_zz_Pa=sigma_zz,
        tangent_branches=tuple(tangent_branches),
        finite_difference_branch_crossing_count=branch_crossing_count,
        characteristic_lengths_m=characteristic_lengths,
        element_internal_force_vectors_N=element_internal_forces,
        separation_background_element_ids=separating_background_ids,
        separating_states=tuple(sorted(evaluated_separating)),
        internal_force_progress_derivatives_N=progress_columns,
        separation_constraint_residuals_m=separation_residuals,
        separation_constraint_displacement_derivatives=(
            separation_displacement_derivatives
        ),
        separation_constraint_progress_derivatives_m=(
            separation_progress_derivatives
        ),
    )


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _scale_separation_constraint_terms(
    *,
    residuals_m: NDArray[np.float64],
    displacement_derivatives: NDArray[np.float64],
    progress_derivatives_m: NDArray[np.float64],
    progress_coordinate_scales_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scale g by its terminal segment when p = s_e * terminal_segment."""

    scales = np.asarray(progress_coordinate_scales_m, dtype=float)
    return (
        np.asarray(residuals_m, dtype=float) / scales,
        np.asarray(displacement_derivatives, dtype=float) / scales[:, None],
        np.asarray(progress_derivatives_m, dtype=float) / scales[:, None] ** 2,
    )


def _compact_structural_mesh(
    case: ElastoplasticFemCase,
    imported_mesh: ImportedMesh,
    topology: PreparedMaterialTopology,
) -> PreparedElastoplasticMesh:
    background_mesh = imported_mesh.mesh
    active_nodes = np.asarray(topology.active_node_ids, dtype=np.int32)
    active_elements = np.asarray(topology.active_element_ids, dtype=np.int32)
    background_to_active = np.asarray(topology.background_to_active_node, dtype=np.int32)
    coordinates = np.asarray(background_mesh.p[:, active_nodes], dtype=float)
    background_connectivity = np.asarray(background_mesh.t[:, active_elements], dtype=np.int32)
    connectivity = background_to_active[background_connectivity]
    if np.any(connectivity < 0):
        raise ContactSolverError("active element references an inactive node")
    compact = MeshTri(coordinates, connectivity)

    fixed_edges = {
        _edge_key(
            int(background_mesh.facets[0, facet]),
            int(background_mesh.facets[1, facet]),
        )
        for facet in np.asarray(background_mesh.boundaries.get("fixed", []), dtype=np.int32)
    }
    contact_edges = {
        _edge_key(int(edge[0]), int(edge[1]))
        for edge in np.asarray(topology.free_surface.edge_node_ids, dtype=np.int32)
    }
    compact_fixed: list[int] = []
    compact_contact: list[int] = []
    for facet_id, active_edge in enumerate(np.asarray(compact.facets.T, dtype=np.int32)):
        background_edge = _edge_key(
            int(active_nodes[active_edge[0]]), int(active_nodes[active_edge[1]])
        )
        if background_edge in fixed_edges:
            compact_fixed.append(facet_id)
        if background_edge in contact_edges:
            compact_contact.append(facet_id)
    compact = compact.with_boundaries(
        {
            "fixed": np.asarray(compact_fixed, dtype=np.int32),
            "contact": np.asarray(compact_contact, dtype=np.int32),
        }
    )
    active_imported = ImportedMesh(mesh=compact, source_path=imported_mesh.source_path)
    basis = Basis(compact, ElementVector(ElementTriP1()))
    return _prepare_mesh_with_load(
        case,
        active_imported,
        np.empty(0, dtype=np.int32),
        np.zeros(basis.N, dtype=float),
        0.0,
    )


def prepare_contact_mesh(
    case: ElastoplasticFemCase,
    imported_mesh: ImportedMesh,
    *,
    topology: PreparedMaterialTopology | None = None,
    edge_history: bool = False,
) -> PreparedContactMesh:
    """Prepare the retained fixed mesh plus an ordered complete top boundary."""

    if topology is not None and not isinstance(topology, PreparedMaterialTopology):
        raise TypeError("topology must be a PreparedMaterialTopology")
    structural = (
        prepare_elastoplastic_mesh(case, imported_mesh)
        if topology is None
        else _compact_structural_mesh(case, imported_mesh, topology)
    )
    if topology is not None:
        surface = topology.free_surface
        background_nodes = np.asarray(surface.node_ids, dtype=np.int32)
        node_ids = np.asarray(
            topology.background_to_active_node[background_nodes], dtype=np.int32
        )
        return PreparedContactMesh(
            structural=structural,
            candidate_node_ids=node_ids,
            candidate_component_dofs=np.asarray(
                surface.candidate_component_dofs, dtype=np.int32
            ),
            candidate_reference_coordinates_m=np.asarray(
                surface.reference_coordinates_m, dtype=float
            ),
            candidate_tributary_lengths_m=np.asarray(
                surface.tributary_lengths_m, dtype=float
            ),
            candidate_keys=surface.candidate_keys,
            edge_endpoint_keys=(
                tuple((min(int(a),int(b)),max(int(a),int(b)),int(n))
                      for a,b in surface.edge_node_ids for n in (a,b))
                if edge_history else ()
            ),
            candidate_background_node_ids=background_nodes,
            active_to_background_node_ids=np.asarray(
                topology.active_node_ids, dtype=np.int32
            ),
            background_to_active_node_ids=np.asarray(
                topology.background_to_active_node, dtype=np.int32
            ),
            active_to_background_element_ids=np.asarray(
                topology.active_element_ids, dtype=np.int32
            ),
            candidate_incident_edge_ids=surface.candidate_incident_edge_ids,
            candidate_reference_normals=np.asarray(
                surface.candidate_reference_normals, dtype=float
            ),
            candidate_reference_normal_unique=np.asarray(
                surface.candidate_reference_normal_unique, dtype=bool
            ),
            candidate_surface_component_ids=np.asarray(
                surface.candidate_surface_component_ids, dtype=np.int32
            ),
            candidate_surface_classifications=surface.candidate_surface_classifications,
        )
    mesh = imported_mesh.mesh
    try:
        contact_facets = np.asarray(mesh.boundaries["contact"], dtype=np.int32)
    except KeyError as exc:
        raise ContactSolverError("mesh does not contain the contact boundary") from exc
    node_ids = np.unique(mesh.facets[:, contact_facets])
    coordinates = np.asarray(mesh.p[:, node_ids].T, dtype=float)
    order = np.argsort(coordinates[:, 0], kind="stable")
    node_ids = np.asarray(node_ids[order], dtype=np.int32)
    coordinates = coordinates[order]
    try:
        tributary = ordered_boundary_tributary_lengths(coordinates)
    except ContactGeometryError as exc:
        raise ContactSolverError("contact candidate boundary is invalid") from exc
    component_dofs = np.column_stack(
        (
            structural.component_dofs[0, node_ids],
            structural.component_dofs[1, node_ids],
        )
    ).astype(np.int32)
    return PreparedContactMesh(
        structural=structural,
        candidate_node_ids=node_ids,
        candidate_component_dofs=component_dofs,
        candidate_reference_coordinates_m=coordinates,
        candidate_tributary_lengths_m=tributary,
        candidate_keys=tuple((int(node), ()) for node in node_ids),
        edge_endpoint_keys=(tuple((min(int(a),int(b)),max(int(a),int(b)),int(n))
            for a,b in mesh.facets[:,contact_facets].T for n in (a,b)) if edge_history else ()),
        candidate_background_node_ids=node_ids.copy(),
        active_to_background_node_ids=np.arange(imported_mesh.node_count, dtype=np.int32),
        background_to_active_node_ids=np.arange(imported_mesh.node_count, dtype=np.int32),
        active_to_background_element_ids=np.arange(imported_mesh.triangle_count, dtype=np.int32),
        candidate_incident_edge_ids=tuple(() for _ in node_ids),
        candidate_reference_normals=np.zeros((node_ids.size, 2), dtype=float),
        candidate_reference_normal_unique=np.ones(node_ids.size, dtype=bool),
        candidate_surface_component_ids=np.zeros(node_ids.size, dtype=np.int32),
        candidate_surface_classifications=tuple("contact_free_surface" for _ in node_ids),
    )


def remap_contact_states_by_key(
    old_keys: tuple[tuple[int, tuple[int, ...]], ...],
    old_states: tuple[ContactPointState, ...],
    new_keys: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[ContactPointState, ...]:
    if len(old_keys) != len(old_states):
        raise ContactSolverError("committed contact keys and states have different counts")
    if len(old_keys) != len(set(old_keys)) or len(new_keys) != len(set(new_keys)):
        raise ContactSolverError("contact candidate keys must be unique")
    previous = dict(zip(old_keys, old_states, strict=True))
    return tuple(previous.get(key, ContactPointState()) for key in new_keys)


def _grain_reference(grain: RigidAnalyticalGrain) -> NDArray[np.float64]:
    reference = getattr(grain, "reference_m", None)
    if reference is None:
        reference = getattr(grain, "center_m", None)
    if reference is None:
        raise ContactSolverError("rigid grain does not expose a reference point")
    result = np.asarray(reference, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ContactSolverError("rigid grain reference point is invalid")
    return result


def _aligned_contact_states(
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
) -> tuple[ContactPointState, ...]:
    old_keys = tuple(getattr(committed, "candidate_keys", ()))
    if not old_keys:
        if len(committed.contact_states) != prepared.candidate_node_ids.size:
            raise ContactSolverError("committed contact-state count does not match boundary")
        return committed.contact_states
    return remap_contact_states_by_key(
        old_keys, committed.contact_states, prepared.candidate_keys
    )


def initial_contact_structure_state(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    *,
    grain: RigidAnalyticalGrain | None = None,
    damage_input: DamageInput | None = None,
) -> CommittedContactStructureState:
    """Return the immutable zero-history state for the contact route."""

    base = initial_structure_state(case, prepared.structural)
    reference = _grain_reference(grain) if grain is not None else None
    center = (
        (float(grain.center_x_m), float(grain.center_y_m))
        if grain is not None and hasattr(grain, "center_x_m")
        else None
    )
    return CommittedContactStructureState(
        marker=0.0,
        displacement_vector_m=np.asarray(base.displacement_vector_m).copy(),
        element_states=base.element_states,
        contact_states=tuple(
            ContactPointState()
            for _ in range(prepared.candidate_node_ids.size)
        ),
        grain_center_m=center,
        candidate_keys=prepared.candidate_keys,
        grain_reference_m=tuple(reference) if reference is not None else None,
        damage_states=(
            tuple(initial_damage_state() for _ in range(prepared.element_count))
            if damage_input is not None
            else ()
        ),
        background_element_ids=np.asarray(
            prepared.active_to_background_element_ids
            if prepared.active_to_background_element_ids.size
            else np.arange(prepared.element_count, dtype=np.int32),
            dtype=np.int32,
        ),
    )


def rebase_contact_state_after_topology_transition(
    case: ElastoplasticFemCase,
    old_prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    new_prepared: PreparedContactMesh,
    *,
    damage_input: DamageInput,
    density_kg_per_m3: float | None = None,
    kinetic_energy_absolute_tolerance_J: float | None = None,
) -> CommittedContactStructureState:
    """Project committed active state through stable background identities."""

    old_node_map = {
        int(background): active
        for active, background in enumerate(old_prepared.active_to_background_node_ids)
    }
    displacement = np.zeros(new_prepared.total_degrees_of_freedom, dtype=float)
    old_displacement = np.asarray(committed.displacement_vector_m, dtype=float)
    dynamic_fields = dict(
        physical_time_s=float(getattr(committed,'physical_time_s',0.0)),
        transient_step_count=int(getattr(committed,'transient_step_count',0)),
        transient_last_time_step_s=float(getattr(committed,'transient_last_time_step_s',0.0)),
        transient_absolute_energy_error_J=float(getattr(committed,'transient_absolute_energy_error_J',0.0)),
        transient_maximum_energy_scale_J=float(getattr(committed,'transient_maximum_energy_scale_J',0.0)),
    )
    for name in ('velocity_vector_m_per_s', 'acceleration_vector_m_per_s2'):
        original = np.asarray(getattr(committed,name,()),dtype=float)
        if original.size:
            if original.shape != old_displacement.shape:
                raise ContactSolverError('dynamic history shape does not match old topology')
            mapped = np.zeros(new_prepared.total_degrees_of_freedom)
            for new_active, background in enumerate(new_prepared.active_to_background_node_ids):
                old_active = old_node_map[int(background)]
                for component in (0,1):
                    mapped[new_prepared.structural.component_dofs[component,new_active]] = original[
                        old_prepared.structural.component_dofs[component,old_active]]
            dynamic_fields[name] = mapped
    if 'velocity_vector_m_per_s' in dynamic_fields:
        # Recompute retained kinetic energy from the active mass; exporting
        # appreciable removed kinetic energy is not an allowed removal route.
        from .inertial_transition import assemble_consistent_mass_matrix
        same_topology = np.array_equal(old_prepared.active_to_background_element_ids,
                                       new_prepared.active_to_background_element_ids)
        if not same_topology:
            if (density_kg_per_m3 is None or kinetic_energy_absolute_tolerance_J is None
                or not math.isfinite(kinetic_energy_absolute_tolerance_J)
                or kinetic_energy_absolute_tolerance_J<0.):
                raise ContactSolverError('dynamic topology rebase requires physical mass audit')
            from .inertial_transition import triangle_consistent_mass_matrix
            remaining=set(int(i) for i in new_prepared.active_to_background_element_ids)
            removed_energy=0.
            velocity=np.asarray(committed.velocity_vector_m_per_s)
            for local,background in enumerate(old_prepared.active_to_background_element_ids):
                if int(background) in remaining:continue
                nodes=old_prepared.structural.element_connectivity[local]
                dofs=old_prepared.structural.element_dofs[local]
                local_mass=triangle_consistent_mass_matrix(
                    old_prepared.structural.imported_mesh.mesh.p.T[nodes],
                    density_kg_per_m3=density_kg_per_m3,thickness_m=case.analysis.thickness)
                removed_energy+=0.5*float(velocity[dofs]@local_mass@velocity[dofs])
            if removed_energy>kinetic_energy_absolute_tolerance_J:
                raise ContactSolverError('removed material retains kinetic energy above the registered gate')
            active_mass=assemble_consistent_mass_matrix(new_prepared.structural,
                density_kg_per_m3=density_kg_per_m3,thickness_m=case.analysis.thickness)
            mapped_velocity=dynamic_fields['velocity_vector_m_per_s']
            dynamic_fields['kinetic_energy_J']=0.5*float(mapped_velocity@active_mass@mapped_velocity)
            dynamic_fields['transient_absolute_energy_error_J'] += removed_energy
            if dynamic_fields['transient_absolute_energy_error_J']>max(
                    1e-10,1e-5*dynamic_fields['transient_maximum_energy_scale_J']):
                raise ContactSolverError('topology kinetic loss exceeds cumulative energy budget')
        else:
            dynamic_fields['kinetic_energy_J'] = float(committed.kinetic_energy_J)
            dynamic_fields['transient_stable_step_count'] = int(getattr(committed,'transient_stable_step_count',0))
    for new_active, background in enumerate(new_prepared.active_to_background_node_ids):
        try:
            old_active = old_node_map[int(background)]
        except KeyError as exc:  # pragma: no cover - topology only removes material
            raise ContactSolverError("new topology introduced an unknown active node") from exc
        displacement[2 * new_active : 2 * new_active + 2] = old_displacement[
            2 * old_active : 2 * old_active + 2
        ]

    old_element_map = {
        int(background): index
        for index, background in enumerate(committed.background_element_ids)
    }
    material_states: list[MaterialPointState] = []
    damage_states: list[DamageState] = []
    for background in new_prepared.active_to_background_element_ids:
        try:
            old_index = old_element_map[int(background)]
        except KeyError as exc:  # pragma: no cover - topology only removes material
            raise ContactSolverError("new topology introduced an unknown active element") from exc
        material_states.append(committed.element_states[old_index])
        damage_states.append(committed.damage_states[old_index])
    aligned_contact = remap_contact_states_by_key(
        tuple(committed.candidate_keys),
        committed.contact_states,
        new_prepared.candidate_keys,
    )
    active_background_ids = {
        int(value) for value in new_prepared.active_to_background_element_ids
    }
    separating_states = tuple(
        (int(background), state)
        for background, state in committed.separating_states
        if int(background) in active_background_ids
    )
    solve_mode = committed.solve_mode
    solve_failure = committed.solve_failure
    if solve_mode is SolveMode.SEPARATION_CONSTRAINED and not separating_states:
        solve_mode = SolveMode.STANDARD_NEWTON
        solve_failure = None
    return CommittedContactStructureState(
        marker=float(committed.marker),
        edge_contact_states=tuple((key,dict(committed.edge_contact_states).get(key,ContactPointState()))
                                  for key in new_prepared.edge_endpoint_keys),
        archived_edge_contact_states=committed.archived_edge_contact_states+tuple(
            (key,value) for key,value in committed.edge_contact_states if key not in new_prepared.edge_endpoint_keys),
        displacement_vector_m=displacement,
        element_states=tuple(material_states),
        accepted_path_reference=getattr(committed,"accepted_path_reference",None),
        contact_states=aligned_contact,
        grain_center_m=committed.grain_center_m,
        candidate_keys=new_prepared.candidate_keys,
        grain_reference_m=committed.grain_reference_m,
        damage_states=tuple(damage_states),
        background_element_ids=np.asarray(
            new_prepared.active_to_background_element_ids, dtype=np.int32
        ).copy(),
        solve_mode=solve_mode,
        solve_failure=solve_failure,
        separating_states=separating_states,
        **dynamic_fields,
    )
def _assemble_contact(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState,
    displacement: NDArray[np.float64],
    grain: RigidAnalyticalGrain,
    penalty_Pa_per_m: float,
    tangential_penalty_Pa_per_m: float | None,
    friction_coefficient: float,
    friction_regularization_ratio: float,
) -> _ContactAssembly:
    if getattr(prepared,'circle_edge_quadrature_order',0):
        return _assemble_circle_edge_contact(case,prepared,committed,displacement,grain,
            penalty_Pa_per_m,tangential_penalty_Pa_per_m,friction_coefficient,friction_regularization_ratio)
    if getattr(prepared,'edge_endpoint_keys',()):
        return _assemble_edge_endpoint_contact(case,prepared,committed,displacement,grain,
            penalty_Pa_per_m,tangential_penalty_Pa_per_m,friction_coefficient,friction_regularization_ratio)
    total_dofs = prepared.total_degrees_of_freedom
    force = np.zeros(total_dofs, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    pressures = np.zeros(prepared.candidate_node_ids.size, dtype=float)
    states: list[ContactPointState] = []
    kinematics: list[RigidContactQuery] = []
    friction_dissipation = 0.0
    current_reference = _grain_reference(grain)
    committed_reference = (
        np.asarray(committed.grain_reference_m, dtype=float)
        if committed.grain_reference_m is not None
        else (
            np.asarray(committed.grain_center_m, dtype=float)
            if committed.grain_center_m is not None
            else current_reference
        )
    )
    rigid_increment = current_reference - committed_reference
    committed_contact_states = _aligned_contact_states(prepared, committed)
    for index, dofs in enumerate(prepared.candidate_component_dofs):
        point = (
            prepared.candidate_reference_coordinates_m[index]
            + displacement[dofs]
        )
        motion = rigid_contact_query(grain, point)
        if not motion.closest_point_unique and motion.signed_gap_m <= 0.0:
            raise ContactSolverError("rigid-grain closest point is not unique")
        if not motion.contact_applicable:
            motion = RigidContactQuery(
                signed_gap_m=max(motion.signed_gap_m, 0.0),
                penetration_m=0.0,
                distance_m=motion.distance_m,
                outward_normal=motion.outward_normal,
                consistent_tangent=motion.consistent_tangent,
                normal_derivative_per_m=motion.normal_derivative_per_m,
                closest_point_m=motion.closest_point_m,
                feature_id=motion.feature_id,
                feature_type=motion.feature_type,
                closest_point_unique=motion.closest_point_unique,
                at_feature_junction=motion.at_feature_junction,
                contact_applicable=False,
                diagnostic_codes=motion.diagnostic_codes,
            )
        response = normal_contact_response(
            gap_m=motion.signed_gap_m,
            normal=motion.outward_normal,
            tributary_length_m=prepared.candidate_tributary_lengths_m[index],
            thickness_m=case.analysis.thickness,
            penalty_Pa_per_m=penalty_Pa_per_m,
            committed_multiplier_Pa=committed_contact_states[index].normal_multiplier_Pa,
        )
        point_force = response.total_force_N.copy()
        # The equilibrium residual is f_internal - f_contact.  The returned
        # normal tangent is therefore added to the body tangent.  Small-strain
        # 3.0.0 deliberately omits the follower-normal geometric term.
        local_tangent = response.tangent_total_N_per_m.copy()
        if response.pressure_Pa > 0.0:
            geometric = (
                -case.analysis.thickness
                * prepared.candidate_tributary_lengths_m[index]
                * response.pressure_Pa
                * motion.normal_derivative_per_m
            )
            local_tangent += geometric
        pressures[index] = response.pressure_Pa
        old = committed_contact_states[index]
        status = response.status
        tangential_traction = 0.0
        accumulated_slip = old.accumulated_slip_m
        if response.pressure_Pa > 0.0 and tangential_penalty_Pa_per_m is not None:
            node_increment = (
                displacement[dofs]
                - np.asarray(committed.displacement_vector_m, dtype=float)[dofs]
            )
            relative_increment = float(
                np.dot(rigid_increment - node_increment, motion.consistent_tangent)
            )
            friction = friction_contact_response(
                old,
                pressure_Pa=response.pressure_Pa,
                tangential_relative_increment_m=relative_increment,
                tangential_penalty_Pa_per_m=tangential_penalty_Pa_per_m,
                friction_coefficient=friction_coefficient,
                friction_regularization_ratio=friction_regularization_ratio,
            )
            tangential_traction = friction.tangential_traction_Pa
            accumulated_slip = friction.accumulated_slip_m
            status = friction.status
            line_weight = prepared.candidate_tributary_lengths_m[index]
            point_force += (
                tangential_traction
                * line_weight
                * case.analysis.thickness
                * motion.consistent_tangent
            )
            local_tangent += (
                friction.tangent_Pa_per_m
                * line_weight
                * case.analysis.thickness
                * np.outer(motion.consistent_tangent, motion.consistent_tangent)
            )
            if prepared.consistent_edge_friction_tangent:
                rotation=np.array([[0.,-1.],[1.,0.]])
                tangent_gradient=rotation@motion.normal_derivative_per_m
                relative_gradient=-motion.consistent_tangent+(rigid_increment-node_increment)@tangent_gradient
                if friction_coefficient==0.:
                    traction_gradient=np.zeros(2)
                elif friction.status=="slip":
                    traction_gradient=(-math.copysign(1.,tangential_traction)
                        * friction_coefficient*penalty_Pa_per_m*motion.outward_normal)
                else:
                    traction_gradient=tangential_penalty_Pa_per_m*relative_gradient
                # Replace the retained stick-only approximation for the v3
                # edge route; v1/v2 preserve their numerical contract.
                local_tangent -= (friction.tangent_Pa_per_m*line_weight*case.analysis.thickness
                    *np.outer(motion.consistent_tangent,motion.consistent_tangent))
                local_tangent -= line_weight*case.analysis.thickness*(
                    np.outer(motion.consistent_tangent,traction_gradient)
                    +tangential_traction*tangent_gradient)
            friction_dissipation += (
                friction.dissipation_increment_J_per_m2
                * line_weight
                * case.analysis.thickness
            )
        force[dofs] += point_force
        rows.extend(np.repeat(dofs, 2).tolist())
        columns.extend(np.tile(dofs, 2).tolist())
        values.extend(local_tangent.reshape(-1).tolist())
        states.append(
            ContactPointState(
                normal_multiplier_Pa=old.normal_multiplier_Pa,
                tangential_traction_Pa=tangential_traction,
                accumulated_slip_m=accumulated_slip,
                status=status,
            )
        )
        kinematics.append(motion)
    return _ContactAssembly(
        force_N=force,
        residual_tangent_N_per_m=coo_matrix(
            (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
            shape=(total_dofs, total_dofs),
        ).tocsr(),
        pressures_Pa=pressures,
        states=tuple(states),
        kinematics=tuple(kinematics),
        friction_dissipation_increment_J=friction_dissipation,
    )


def _circle_edge_integrals(case,prepared,displacement,grain,penalty):
    from .geometry import RigidCircularGrain
    from .edge_contact_history import integrate_circle_penalty_edge
    if not isinstance(grain,RigidCircularGrain) or not prepared.edge_endpoint_keys:
        raise ContactSolverError('circle edge integral requires a full circle and background edges')
    lookup={int(n):i for i,n in enumerate(prepared.candidate_background_node_ids)}
    for a,b in sorted({key[:2] for key in prepared.edge_endpoint_keys}):
        indices=[lookup[a],lookup[b]]
        dofs=prepared.candidate_component_dofs[indices]
        integral=integrate_circle_penalty_edge(prepared.candidate_reference_coordinates_m[indices],
            displacement[dofs],center=grain.center_m,radius=grain.radius_m,penalty=penalty,
            thickness=case.analysis.thickness,order=prepared.circle_edge_quadrature_order)
        yield dofs.ravel(),integral


def _assemble_circle_edge_contact(case,prepared,committed,displacement,grain,penalty,tangential,mu,regularization):
    from dataclasses import replace
    # This stateless pure-penalty prototype must never reinterpret friction or AL history.
    histories=tuple(committed.contact_states)+tuple(v for _,v in committed.edge_contact_states)
    if mu!=0. or any(v.normal_multiplier_Pa!=0. or v.tangential_traction_Pa!=0. or v.accumulated_slip_m!=0. for v in histories):
        raise ContactSolverError('circle edge prototype cannot transfer nonzero contact history')
    display=_assemble_edge_endpoint_contact(case,replace(prepared,circle_edge_quadrature_order=0),
        committed,displacement,grain,penalty,tangential,mu,regularization)
    force=np.zeros(prepared.total_degrees_of_freedom)
    rows=[];columns=[];values=[]
    maximum_penetration=0.
    patches=[]
    edges=sorted({key[:2] for key in prepared.edge_endpoint_keys})
    coordinates=dict(zip(prepared.candidate_background_node_ids,prepared.candidate_reference_coordinates_m))
    for edge,(dofs,integral) in zip(edges,_circle_edge_integrals(case,prepared,displacement,grain,penalty),strict=True):
        force[dofs]+=integral.force_N
        rows.extend(np.repeat(dofs,4));columns.extend(np.tile(dofs,4))
        values.extend(integral.residual_tangent_N_per_m.ravel())
        maximum_penetration=max(maximum_penetration,integral.maximum_penetration_m)
        if integral.active_interval is not None:
            lo,hi=integral.active_interval
            patches.append(dict(background_edge=edge,active_interval=(lo,hi),
                reference_contact_length_m=float(np.linalg.norm(coordinates[edge[1]]-coordinates[edge[0]]))*(hi-lo),
                maximum_pressure_Pa=penalty*integral.maximum_penetration_m,
                force_N=tuple(float(v) for v in integral.force_N.reshape(2,2).sum(axis=0)),
                penalty_energy_J=integral.energy_J))
    tangent=coo_matrix((values,(rows,columns)),shape=(force.size,force.size)).tocsr()
    return replace(display,force_N=force,residual_tangent_N_per_m=tangent,
        integrated_maximum_penetration_m=maximum_penetration,integrated_edge_patches=tuple(patches))


def _assemble_edge_endpoint_contact(case,prepared,committed,displacement,grain,penalty,tangential,mu,regularization):
    from dataclasses import replace
    nodes={int(n):i for i,n in enumerate(prepared.candidate_background_node_ids)}
    keys=prepared.edge_endpoint_keys
    indices=np.array([nodes[key[2]] for key in keys],dtype=int)
    coordinates=prepared.candidate_reference_coordinates_m
    weights=np.array([.5*np.linalg.norm(coordinates[nodes[a]]-coordinates[nodes[b]]) for a,b,_ in keys])
    previous=dict(committed.edge_contact_states)
    if not previous and any(v != ContactPointState() for v in committed.contact_states):
        raise ContactSolverError('nonzero nodal history cannot be split into edge history')
    expanded=_replace_compatible_dataclass(prepared,edge_endpoint_keys=(),consistent_edge_friction_tangent=True,candidate_node_ids=prepared.candidate_node_ids[indices],
        candidate_component_dofs=prepared.candidate_component_dofs[indices],
        candidate_reference_coordinates_m=coordinates[indices],candidate_tributary_lengths_m=weights,
        candidate_keys=tuple((n,(a,b)) for a,b,n in keys))
    history=_replace_compatible_dataclass(committed,candidate_keys=expanded.candidate_keys,
        contact_states=tuple(previous.get(key,ContactPointState()) for key in keys))
    trial=_assemble_contact(case,expanded,history,displacement,grain,penalty,tangential,mu,regularization)
    states=[]; queries=[]; pressures=[]
    for i in range(len(prepared.candidate_keys)):
        points=np.flatnonzero(indices==i); first=int(points[0]); total=float(sum(weights[points]))
        states.append(ContactPointState(
            normal_multiplier_Pa=sum(weights[j]*trial.states[j].normal_multiplier_Pa for j in points)/total,
            tangential_traction_Pa=sum(weights[j]*trial.states[j].tangential_traction_Pa for j in points)/total,
            accumulated_slip_m=sum(weights[j]*trial.states[j].accumulated_slip_m for j in points)/total,
            status=trial.states[first].status))
        queries.append(trial.kinematics[first])
        pressures.append(sum(weights[j]*trial.pressures_Pa[j] for j in points)/total)
    return replace(trial,states=tuple(states),kinematics=tuple(queries),pressures_Pa=np.array(pressures),
                   edge_contact_states=tuple(zip(keys,trial.states,strict=True)))


def contact_residual_and_tangent(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    displacement_vector_m: object,
    *,
    grain: RigidAnalyticalGrain,
    penalty_Pa_per_m: object,
) -> tuple[NDArray[np.float64], csr_matrix]:
    """Expose the frictionless fixed-state contact contribution for verification."""

    displacement = np.asarray(displacement_vector_m, dtype=float)
    if displacement.shape != (prepared.total_degrees_of_freedom,):
        raise ContactSolverError("displacement_vector_m shape is invalid")
    trial = _assemble_contact(
        case,
        prepared,
        committed,
        displacement,
        grain,
        float(penalty_Pa_per_m),
        None,
        0.0,
        1.0e-6,
    )
    return trial.force_N.copy(), trial.residual_tangent_N_per_m.copy()


def advance_separation_constrained_increment(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    *,
    grain_at_marker: Callable[[float], RigidAnalyticalGrain],
    marker_bounds: tuple[float, float],
    reference_path_length_m: object,
    target_separation_progress: object,
    normal_penalty_Pa_per_m: object,
    tangential_penalty_Pa_per_m: object | None,
    friction_coefficient: object,
    friction_regularization_ratio: object,
    damage_input: DamageInput,
    minimum_substep_fraction: object,
) -> ConvergedContactIncrement:
    """Advance a committed SEPARATING active set on a fixed topology."""

    if not committed.separating_states:
        raise ContactSolverError(
            "SEPARATION_CONSTRAINED requires a committed SEPARATING active set"
        )
    lower_marker, upper_marker = (float(value) for value in marker_bounds)
    path_length = float(reference_path_length_m)
    minimum_fraction = float(minimum_substep_fraction)
    penalty = float(normal_penalty_Pa_per_m)
    tangential_penalty = (
        None
        if tangential_penalty_Pa_per_m is None
        else float(tangential_penalty_Pa_per_m)
    )
    coefficient = float(friction_coefficient)
    regularization = float(friction_regularization_ratio)
    if not lower_marker <= committed.marker <= upper_marker:
        raise ValueError("committed marker is outside marker_bounds")
    if not math.isfinite(path_length) or path_length <= 0.0:
        raise ValueError("reference_path_length_m must be finite and positive")
    if not math.isfinite(minimum_fraction) or minimum_fraction <= 0.0:
        raise ValueError("minimum_substep_fraction must be finite and positive")
    active = tuple(sorted(committed.separating_states, key=lambda item: item[0]))
    background_ids = tuple(item[0] for item in active)
    separation_scales = np.asarray(
        [
            state.terminal_displacement_m * (1.0 - state.entry_softening_fraction)
            for _, state in active
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(separation_scales)) or np.any(separation_scales <= 0.0):
        raise ContactSolverError("SEPARATING terminal-segment scale is invalid")
    target = np.asarray(target_separation_progress, dtype=float)
    if target.ndim == 0:
        target = np.full(len(active), float(target), dtype=float)
    if target.shape != (len(active),) or not np.all(np.isfinite(target)):
        raise ValueError("target_separation_progress shape is invalid")
    minimum_progress = np.asarray([state.s_e for _, state in active], dtype=float)
    if np.any(target < minimum_progress) or np.any(target > 1.0):
        raise ValueError("target_separation_progress must be irreversible and within [0, 1]")

    free = prepared.structural.free_dofs
    fixed = prepared.structural.fixed_dofs
    base_displacement = np.asarray(committed.displacement_vector_m, dtype=float).copy()
    current_distance = 0.0
    current_progress_coordinates = minimum_progress * separation_scales
    current = np.concatenate(
        (base_displacement[free], [current_distance], current_progress_coordinates)
    )
    distance_lower = (lower_marker - committed.marker) * path_length
    distance_upper = (upper_marker - committed.marker) * path_length
    material_side = None

    def assemble(combined: NDArray[np.float64]):
        displacement = base_displacement.copy()
        displacement[free] = combined[: free.size]
        displacement[fixed] = 0.0
        distance = float(combined[free.size])
        marker = committed.marker + distance / path_length
        if marker < lower_marker or marker > upper_marker:
            raise ConstrainedSeparationEventError(
                "separation path marker left the registered interval"
            )
        progress = combined[free.size + 1 :] / separation_scales
        if np.any(progress + 64.0 * np.finfo(float).eps < minimum_progress) or np.any(
            progress > 1.0 + 64.0 * np.finfo(float).eps
        ):
            raise ConstrainedSeparationEventError(
                "separation progress left its irreversible interval"
            )
        progress = np.clip(progress, minimum_progress, 1.0)
        progress_map = dict(zip(background_ids, progress, strict=True))
        grain = grain_at_marker(marker)
        body = _assemble_damage_trial(
            case,
            prepared,
            committed,
            displacement,
            damage_input,
            separation_progress_by_background=progress_map,
            active_side_direction=material_side,
        )
        contact = _assemble_contact(
            case,
            prepared,
            committed,
            displacement,
            grain,
            penalty,
            tangential_penalty,
            coefficient,
            regularization,
        )
        if body.separation_background_element_ids != background_ids:
            raise ContactSolverError("SEPARATING active-set order changed during correction")
        equilibrium = contact.force_N - body.internal_force_N
        tangent = -(
            body.tangent_stiffness_N_per_m + contact.residual_tangent_N_per_m
        )
        progress_columns = -body.internal_force_progress_derivatives_N / separation_scales
        separation_values, separation_u, separation_p = (
            _scale_separation_constraint_terms(
                residuals_m=body.separation_constraint_residuals_m,
                displacement_derivatives=(
                    body.separation_constraint_displacement_derivatives
                ),
                progress_derivatives_m=(
                    body.separation_constraint_progress_derivatives_m
                ),
                progress_coordinate_scales_m=separation_scales,
            )
        )
        return (
            displacement,
            marker,
            progress,
            grain,
            body,
            contact,
            equilibrium[free],
            tangent[free][:, free].toarray(),
            progress_columns[free],
            separation_values,
            separation_u[:, free],
            separation_p,
        )

    def marker_derivative(combined: NDArray[np.float64], contact_status, branches):
        if getattr(prepared,'circle_edge_quadrature_order',0):
            first_reference=_grain_reference(grain_at_marker(lower_marker))
            last_reference=_grain_reference(grain_at_marker(upper_marker))
            rigid=np.zeros(prepared.total_degrees_of_freedom)
            rigid[prepared.structural.basis.nodal_dofs]=(
                (last_reference-first_reference)/((upper_marker-lower_marker)*path_length))[:,None]
            contact=assemble(combined)[5]
            return (contact.residual_tangent_N_per_m@rigid)[free]
        perturbation = max(1.0e-13, 1.0e-4 * minimum_fraction * path_length)
        distance = float(combined[free.size])
        lower_distance = max(distance_lower, distance - perturbation)
        upper_distance = min(distance_upper, distance + perturbation)
        if upper_distance <= lower_distance:
            raise ConstrainedSeparationEventError(
                "separation marker derivative has no legal perturbation interval"
            )
        lower = combined.copy()
        upper = combined.copy()
        lower[free.size] = lower_distance
        upper[free.size] = upper_distance
        lower_data = assemble(lower)
        upper_data = assemble(upper)
        if (
            tuple(item.status for item in lower_data[5].states) != contact_status
            or tuple(item.status for item in upper_data[5].states) != contact_status
        ):
            raise ConstrainedSeparationEventError(
                "separation marker derivative crossed a contact active branch"
            )
        if lower_data[4].tangent_branches != branches or upper_data[4].tangent_branches != branches:
            raise ConstrainedSeparationEventError(
                "separation marker derivative crossed a material active branch"
            )
        return (upper_data[6] - lower_data[6]) / (upper_distance - lower_distance)

    initial_data = assemble(current)
    initial_marker_column = marker_derivative(
        current,
        tuple(item.status for item in initial_data[5].states),
        initial_data[4].tangent_branches,
    )
    tangent_system = np.block(
        [
            [initial_data[7], initial_marker_column[:, None], initial_data[8]],
            [initial_data[10], np.zeros((len(active), 1)), initial_data[11]],
        ]
    )
    reference = np.zeros(current.size, dtype=float)
    accepted_reference=getattr(committed,'accepted_path_reference',None)
    if accepted_reference is not None:
        reference[:free.size+1]=accepted_reference.mapped_vector(
            prepared.active_to_background_node_ids,free,
            displacement_scale_m=1.,marker_scale_m=path_length)
    elif committed.path_state is not None and committed.path_state.tangent.shape == (
        free.size + 1,
    ):
        reference[: free.size] = committed.path_state.tangent[:-1]
        reference[free.size] = committed.path_state.tangent[-1]
    else:
        reference[free.size] = 1.0
    reference[free.size + 1 :] = 1.0
    tangent_direction = initialize_separation_path_direction(
        tangent_system, reference_direction=reference
    )
    # A joint tangent must close the current J2 side of both separating and
    # ordinary elements; the entry history is provenance, not the current side.
    if getattr(prepared,'circle_edge_quadrature_order',0):
        seen_sides=set()
        for _ in range(2*prepared.element_count+1):
            if float(np.sum(tangent_direction[free.size+1:]))<0.:
                tangent_direction=-tangent_direction
            material_side=np.zeros(prepared.total_degrees_of_freedom)
            material_side[free]=tangent_direction[:free.size]
            updated=assemble(current)
            column=marker_derivative(current,tuple(item.status for item in updated[5].states),updated[4].tangent_branches)
            closed_system=np.block([[updated[7],column[:,None],updated[8]],
                [updated[10],np.zeros((len(active),1)),updated[11]]])
            if np.array_equal(closed_system,tangent_system):
                initial_data=updated
                break
            signature=closed_system.tobytes()
            if signature in seen_sides:
                raise ConstrainedSeparationEventError('joint entry constitutive active-side cycle')
            seen_sides.add(signature)
            tangent_system=closed_system
            tangent_direction=initialize_separation_path_direction(tangent_system,reference_direction=reference)
        else:
            raise ConstrainedSeparationEventError('joint entry constitutive side did not close')
    progress_slice = tangent_direction[free.size + 1 :]
    if float(np.sum(progress_slice)) < 0.0:
        tangent_direction = -tangent_direction
    requested_coordinates = target * separation_scales
    progress_direction = tangent_direction[free.size + 1 :]
    positive = progress_direction > 64.0 * np.finfo(float).eps
    if not np.any(positive):
        raise ConstrainedSeparationEventError(
            "separation path direction does not advance the active softening segment"
        )
    arc_candidates = (
        requested_coordinates[positive] - current_progress_coordinates[positive]
    ) / progress_direction[positive]
    positive_indexes = np.flatnonzero(positive)
    controlling_progress_index = int(
        positive_indexes[int(np.argmin(arc_candidates))]
    )
    arc_length = float(np.min(arc_candidates))
    if not math.isfinite(arc_length) or arc_length <= 0.0:
        raise ConstrainedSeparationEventError(
            "requested separation progress is not ahead on the current path"
        )
    predicted = current + arc_length * tangent_direction
    path_constraint_direction = np.zeros_like(tangent_direction)
    path_coordinate_index = free.size + 1 + controlling_progress_index
    path_constraint_direction[path_coordinate_index] = 1.0
    combined = predicted.copy()
    correction_norm = math.inf
    path_residual = math.inf
    separation_norm = math.inf
    residual_norm = math.inf
    body = initial_data[4]
    contact = initial_data[5]
    grain = initial_data[3]
    displacement = initial_data[0]
    progress = minimum_progress
    marker = committed.marker
    residual_history: list[float] = []
    for correction_count in range(case.newton.maximum_iterations + 1):
        data = assemble(combined)
        (
            displacement,
            marker,
            progress,
            grain,
            body,
            contact,
            residual_free,
            tangent_free,
            progress_columns,
            separation_values,
            separation_u,
            separation_p,
        ) = data
        residual_norm = float(np.linalg.norm(residual_free))
        residual_history.append(residual_norm)
        contact_scale = float(np.linalg.norm(contact.force_N[free]))
        residual_tolerance = (
            case.newton.residual_absolute_tolerance_N
            + case.newton.residual_relative_tolerance * contact_scale
        )
        path_residual = float(
            combined[path_coordinate_index]
            - requested_coordinates[controlling_progress_index]
        )
        separation_norm = float(np.linalg.norm(separation_values))
        correction_tolerance = (
            case.newton.displacement_absolute_tolerance_m
            + case.newton.displacement_relative_tolerance
            * float(np.linalg.norm(displacement[free]))
        )
        if (
            residual_norm <= residual_tolerance
            and abs(path_residual)
            <= max(1.0e-15, 1.0e-6 * minimum_fraction * path_length)
            and separation_norm <= damage_input.damage_event_tolerance
            and correction_norm <= correction_tolerance
        ):
            break
        if correction_count >= case.newton.maximum_iterations:
            raise ConstrainedSeparationEventError(
                "separation augmented correction did not converge"
            )
        marker_column = marker_derivative(
            combined,
            tuple(item.status for item in contact.states),
            body.tangent_branches,
        )
        augmented = np.block(
            [
                [tangent_free, marker_column[:, None], progress_columns],
                [path_constraint_direction[None, :]],
                [separation_u, np.zeros((len(active), 1)), separation_p],
            ]
        )
        force_scale = max(contact_scale, case.newton.residual_absolute_tolerance_N)
        scaled_values = np.concatenate(
            (
                residual_free / force_scale,
                [path_residual / path_length],
                separation_values,
            )
        )
        scaled_jacobian = augmented.copy()
        scaled_jacobian[: free.size] /= force_scale
        scaled_jacobian[free.size] /= path_length
        try:
            correction = np.linalg.solve(scaled_jacobian, -scaled_values)
        except np.linalg.LinAlgError as exc:
            raise ConstrainedSeparationEventError(
                "separation augmented correction system is singular"
            ) from exc
        if not np.all(np.isfinite(correction)):
            raise ConstrainedSeparationEventError(
                "separation augmented correction is non-finite"
            )
        candidate = combined + correction
        candidate_progress = candidate[free.size + 1 :] / separation_scales
        if np.any(candidate_progress + 64.0 * np.finfo(float).eps < minimum_progress):
            raise ConstrainedSeparationEventError(
                "separation augmented correction violates irreversibility"
            )
        correction_norm = float(np.linalg.norm(correction[: free.size]))
        combined = candidate

    equilibrium = body.internal_force_N - contact.force_N
    fixed_x = float(math.fsum(float(value) for value in equilibrium[prepared.structural.fixed_x_dofs]))
    fixed_y = float(math.fsum(float(value) for value in equilibrium[prepared.structural.fixed_y_dofs]))
    total_contact = np.asarray(
        [
            math.fsum(
                float(contact.force_N[dofs[component]])
                for dofs in prepared.candidate_component_dofs
            )
            for component in (0, 1)
        ],
        dtype=float,
    )
    updated_separating = body.separating_states
    reference = _grain_reference(grain)
    return ConvergedContactIncrement(
        marker=float(marker),
        displacement_vector_m=displacement.copy(),
        element_states=body.element_states,
        contact_states=contact.states,
        edge_contact_states=contact.edge_contact_states,
        archived_edge_contact_states=committed.archived_edge_contact_states,
        contact_kinematics=contact.kinematics,
        contact_pressure_Pa=contact.pressures_Pa,
        contact_force_vector_N=contact.force_N,
        internal_force_vector_N=body.internal_force_N,
        element_total_strain=body.element_total_strain,
        element_stress_Pa=body.element_stress_Pa,
        element_sigma_zz_Pa=body.element_sigma_zz_Pa,
        newton_iterations=correction_count,
        residual_norm_N=residual_norm,
        residual_tolerance_N=residual_tolerance,
        correction_norm_m=correction_norm,
        correction_tolerance_m=correction_tolerance,
        reference_force_N=contact_scale,
        reference_contact_size_m=float(np.median(prepared.candidate_tributary_lengths_m)),
        reference_displacement_m=max(path_length, float(np.linalg.norm(displacement[free]))),
        residual_absolute_tolerance_N=case.newton.residual_absolute_tolerance_N,
        displacement_absolute_tolerance_m=case.newton.displacement_absolute_tolerance_m,
        residual_norm_history_N=tuple(residual_history),
        total_contact_force_N=total_contact,
        rigid_grain_reaction_N=-total_contact,
        fixed_reaction_x_N=fixed_x,
        fixed_reaction_y_N=fixed_y,
        balance_residual_x_N=float(total_contact[0] + fixed_x),
        balance_residual_y_N=float(total_contact[1] + fixed_y),
        maximum_penetration_m=max(contact.integrated_maximum_penetration_m,
            max(item.penetration_m for item in contact.kinematics)),
        grain_center_m=(
            (float(grain.center_x_m), float(grain.center_y_m))
            if hasattr(grain, "center_x_m") else None
        ),
        candidate_keys=prepared.candidate_keys,
        grain_reference_m=(float(reference[0]), float(reference[1])),
        friction_dissipation_increment_J=contact.friction_dissipation_increment_J,
        augmented_iterations=0,
        damage_states=body.damage_states,
        background_element_ids=np.asarray(
            prepared.active_to_background_element_ids, dtype=np.int32
        ).copy(),
        path_state=None,
        solve_mode=SolveMode.SEPARATION_CONSTRAINED,
        solve_failure=None,
        separating_states=updated_separating,
        accepted_path_reference=getattr(committed,'accepted_path_reference',None),
        separation_path_residual_m=path_residual,
        separation_constraint_residual=separation_norm,
    )


def close_path_entry_material_side(case,prepared,committed,*,grain,grain_direction_m,
                                   penalty,tangential,friction_coefficient,regularization,damage_input):
    """Close directional constitutive derivatives at an accepted equilibrium.

    Translation invariance gives the contact marker column analytically:
    d f_contact/d marker = K_contact times uniform rigid translation. Solve
    K_total du = K_contact d x_grain and require its material side to agree.
    This regular-tangent entry is not a substitute for an augmented nullspace
    solve at a singular limit point; rank loss or an active-side cycle is explicit.
    """
    free=prepared.structural.free_dofs
    motion=np.asarray(grain_direction_m,dtype=float)
    if motion.shape!=(2,) or not np.all(np.isfinite(motion)) or np.linalg.norm(motion)==0:
        raise ContactSolverError('path entry requires finite nonzero prescribed motion')
    contact=_assemble_contact(case,prepared,committed,committed.displacement_vector_m,
        grain,penalty,tangential,friction_coefficient,regularization)
    rigid=np.zeros(prepared.total_degrees_of_freedom)
    rigid[prepared.structural.basis.nodal_dofs]=motion[:,None]
    column=(contact.residual_tangent_N_per_m@rigid)[free]
    direction=rigid.copy();direction[prepared.structural.fixed_dofs]=0.
    seen=[]
    for _ in range(2*prepared.element_count+1):
        body=_assemble_damage_trial(case,prepared,committed,committed.displacement_vector_m,
            damage_input,active_side_direction=direction)
        matrix=(body.tangent_stiffness_N_per_m+contact.residual_tangent_N_per_m)[free][:,free].toarray()
        if np.linalg.matrix_rank(matrix)!=len(free):
            raise ConstrainedSeparationEventError('path entry tangent is singular; augmented nullspace required')
        next_direction=np.zeros_like(direction)
        next_direction[free]=np.linalg.solve(matrix,column)
        verified=_assemble_damage_trial(case,prepared,committed,committed.displacement_vector_m,
            damage_input,active_side_direction=next_direction)
        difference=verified.tangent_stiffness_N_per_m-body.tangent_stiffness_N_per_m
        if difference.nnz==0 or np.max(np.abs(difference.data))==0.:
            return next_direction
        signature=matrix.tobytes()
        if signature in seen:
            raise ConstrainedSeparationEventError('path entry constitutive active-side cycle')
        seen.append(signature);direction=next_direction
    raise ConstrainedSeparationEventError('path entry material side did not close')


def locate_constrained_separation_event(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    seed: CommittedContactStructureState | ConvergedContactIncrement,
    *,
    start_fraction: object,
    seed_fraction: object,
    reference_path_length_m: object,
    grain_at_fraction: Callable[[float], RigidAnalyticalGrain],
    marker_offset: object,
    normal_penalty_Pa_per_m: object,
    tangential_penalty_Pa_per_m: object | None,
    friction_coefficient: object,
    friction_regularization_ratio: object,
    damage_input: DamageInput,
    minimum_substep_fraction: object,
    maximum_steps: object,
) -> ConvergedContactIncrement:
    """Trace one fixed-history equilibrium branch to terminal damage."""

    def path_failure(cause: Exception, stage: str, executed: bool):
        error = ConstrainedSeparationEventError(str(cause))
        error.path_failure_stage = stage
        error.path_entry_executed = executed
        return error

    start = float(start_fraction)
    seed_value = float(seed_fraction)
    path_length = float(reference_path_length_m)
    marker_base = float(marker_offset)
    minimum_fraction = float(minimum_substep_fraction)
    penalty = float(normal_penalty_Pa_per_m)
    tangential_penalty = (
        None
        if tangential_penalty_Pa_per_m is None
        else float(tangential_penalty_Pa_per_m)
    )
    coefficient = float(friction_coefficient)
    regularization = float(friction_regularization_ratio)
    if not isinstance(maximum_steps, int) or isinstance(maximum_steps, bool):
        raise ValueError("maximum_steps must be a positive integer")
    if maximum_steps < 1:
        raise ValueError("maximum_steps must be a positive integer")
    if not 0.0 <= start <= seed_value <= 1.0:
        raise ValueError("path fractions are invalid")
    if not math.isfinite(path_length) or path_length <= 0.0:
        raise ValueError("reference_path_length_m must be finite and positive")
    if not math.isfinite(minimum_fraction) or minimum_fraction <= 0.0:
        raise ValueError("minimum_substep_fraction must be finite and positive")
    if len(committed.damage_states) != prepared.element_count:
        raise ContactSolverError("committed damage-state count does not match mesh")

    free = prepared.structural.free_dofs
    fixed = prepared.structural.fixed_dofs
    base_displacement = np.asarray(committed.displacement_vector_m, dtype=float)
    seed_displacement = np.asarray(seed.displacement_vector_m, dtype=float)
    seed_distance = (seed_value - start) * path_length
    accepted_reference=getattr(committed,"accepted_path_reference",None)
    reference_direction=None
    material_side=None
    if accepted_reference is not None:
        reference_direction=accepted_reference.mapped_vector(
            prepared.active_to_background_node_ids,free,
            displacement_scale_m=1.,marker_scale_m=path_length)
        material_side=np.zeros(prepared.total_degrees_of_freedom)
        material_side[free]=reference_direction[:-1]
    if getattr(prepared,'circle_edge_quadrature_order',0):
        first_reference=_grain_reference(grain_at_fraction(start))
        last_reference=_grain_reference(grain_at_fraction(1.))
        prescribed_direction=(last_reference-first_reference)/(1.-start)
        material_side=close_path_entry_material_side(case,prepared,committed,
            grain=grain_at_fraction(start),grain_direction_m=prescribed_direction,
            penalty=penalty,tangential=tangential_penalty,friction_coefficient=coefficient,
            regularization=regularization,damage_input=damage_input)
        # Local tangent selected by the new segment's prescribed motion. The
        # stored reference is retained unchanged as history, not overwritten.
        reference_direction=np.r_[material_side[free],path_length]

    def assemble(
        state: NDArray[np.float64], distance: float
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        _DamageBodyAssembly,
        _ContactAssembly,
    ]:
        displacement = base_displacement.copy()
        displacement[free] = state
        displacement[fixed] = 0.0
        fraction = start + distance / path_length
        grain = grain_at_fraction(float(fraction))
        body = _assemble_damage_trial(
            case, prepared, committed, displacement, damage_input,
            active_side_direction=material_side
        )
        contact = _assemble_contact(
            case,
            prepared,
            committed,
            displacement,
            grain,
            penalty,
            tangential_penalty,
            coefficient,
            regularization,
        )
        residual = contact.force_N - body.internal_force_N
        tangent = -(
            body.tangent_stiffness_N_per_m
            + contact.residual_tangent_N_per_m
        )
        return residual[free], tangent[free][:, free].toarray(), body, contact

    def residual(
        state: NDArray[np.float64], distance: float
    ) -> NDArray[np.float64]:
        return assemble(state, distance)[0]

    def jacobian(
        state: NDArray[np.float64], distance: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        _, tangent, _, contact = assemble(state, distance)
        if getattr(prepared,'circle_edge_quadrature_order',0):
            rigid=np.zeros(prepared.total_degrees_of_freedom)
            rigid[prepared.structural.basis.nodal_dofs]=(prescribed_direction/path_length)[:,None]
            return tangent,(contact.residual_tangent_N_per_m@rigid)[free]
        perturbation = max(
            1.0e-13,
            1.0e-4 * minimum_fraction * path_length,
        )
        lower_distance = max(-start * path_length, distance - perturbation)
        upper_distance = min((1.0 - start) * path_length, distance + perturbation)
        if upper_distance <= lower_distance:
            raise ConstrainedSeparationEventError(
                "path marker derivative has no legal perturbation interval"
            )
        lower_residual, _, lower_body, lower_contact = assemble(
            state, lower_distance
        )
        upper_residual, _, upper_body, upper_contact = assemble(
            state, upper_distance
        )
        current_contact_status = tuple(item.status for item in contact.states)
        if (
            tuple(item.status for item in lower_contact.states)
            != current_contact_status
            or tuple(item.status for item in upper_contact.states)
            != current_contact_status
        ):
            raise ConstrainedSeparationEventError(
                "path marker derivative crossed a contact active branch"
            )
        current_branches = assemble(state, distance)[2].tangent_branches
        if (
            lower_body.tangent_branches != current_branches
            or upper_body.tangent_branches != current_branches
        ):
            raise ConstrainedSeparationEventError(
                "path marker derivative crossed a damage active branch"
            )
        return tangent, (upper_residual - lower_residual) / (
            upper_distance - lower_distance
        )

    cap = 1.0 - damage_input.damage_event_tolerance

    def event_value(point: PathState) -> float:
        body = assemble(point.state, point.marker)[2]
        damage_event=max(item.damage for item in body.damage_states) - cap
        if getattr(prepared,'circle_edge_quadrature_order',0):
            # Either physical terminal damage or the prescribed segment endpoint
            # ends this numerical route; solver failure cannot require damage.
            return max(damage_event,start+point.marker/path_length-1.)
        return damage_event

    seed_state = seed.path_state
    if seed_state is None:
        try:
            seed_state = initialize_equilibrium_path_state(
                state=seed_displacement[free],
                marker=seed_distance,
                residual=residual,
                jacobian=jacobian,
                marker_direction=1.0,
                residual_tolerance=float(seed.residual_tolerance_N),
                reference_direction=reference_direction,
            )
        except PathContinuationError as exc:
            raise path_failure(exc, 'direction_initialization', False) from exc
    settings = PseudoArcLengthSettings(
        arc_length=4.0 * minimum_fraction * path_length,
        maximum_corrections=case.newton.maximum_iterations,
        residual_tolerance=float(seed.residual_tolerance_N),
        constraint_tolerance=max(1.0e-15, 1.0e-6 * minimum_fraction * path_length),
        correction_tolerance=float(seed.correction_tolerance_m),
    )
    lower = seed_state
    if event_value(lower) >= 0.0:
        secant_seed = initialize_path_state(
            state=base_displacement[free],
            marker=0.0,
            next_state=seed_displacement[free],
            next_marker=seed_distance,
        )
        lower_equilibrium = PathState(
            state=base_displacement[free],
            marker=0.0,
            tangent=secant_seed.tangent,
            residual_norm=float(np.linalg.norm(residual(base_displacement[free], 0.0))),
            constraint_residual=0.0,
            correction_count=0,
            correction_norm=0.0,
        )
        try:
            critical = locate_path_event(
                lower_equilibrium,
                seed_state,
                event_function=event_value,
                residual=residual,
                jacobian=jacobian,
                settings=settings,
                event_tolerance=max(
                    1.0e-12, 0.1 * damage_input.damage_event_tolerance
                ),
            ).critical
        except PathContinuationError as exc:
            raise path_failure(exc, 'event_correction', True) from exc
    else:
        try:
            for _ in range(maximum_steps):
                upper = pseudo_arclength_step(
                    lower,
                    residual=residual,
                    jacobian=jacobian,
                    settings=settings,
                )
                if event_value(upper) >= 0.0:
                    critical = locate_path_event(
                        lower,
                        upper,
                        event_function=event_value,
                        residual=residual,
                        jacobian=jacobian,
                        settings=settings,
                        event_tolerance=max(
                            1.0e-12, 0.1 * damage_input.damage_event_tolerance
                        ),
                    ).critical
                    break
                lower = upper
            else:
                raise path_failure(ContactSolverError(
                    "terminal damage was not reached within maximum_steps"
                ), 'path_step_budget', True)
        except PathContinuationError as exc:
            raise path_failure(exc, 'augmented_correction', True) from exc

    residual_free, _, body, contact = assemble(critical.state, critical.marker)
    displacement = base_displacement.copy()
    displacement[free] = critical.state
    displacement[fixed] = 0.0
    fraction = start + critical.marker / path_length
    grain = grain_at_fraction(float(fraction))
    contact_scale = float(np.linalg.norm(contact.force_N[free]))
    residual_absolute_tolerance = max(
        case.newton.residual_absolute_tolerance_N,
        1.0e-12,
        float(seed.residual_absolute_tolerance_N),
    )
    residual_tolerance = (
        residual_absolute_tolerance
        + case.newton.residual_relative_tolerance * contact_scale
    )
    if critical.residual_norm > residual_tolerance:
        raise ConstrainedSeparationEventError(
            "constrained event state does not satisfy the Newton residual gate"
        )
    equilibrium = body.internal_force_N - contact.force_N
    fixed_x = float(math.fsum(float(v) for v in equilibrium[prepared.structural.fixed_x_dofs]))
    fixed_y = float(math.fsum(float(v) for v in equilibrium[prepared.structural.fixed_y_dofs]))
    total_contact = np.array(
        [
            math.fsum(
                float(contact.force_N[dofs[component]])
                for dofs in prepared.candidate_component_dofs
            )
            for component in (0, 1)
        ],
        dtype=float,
    )
    reference = _grain_reference(grain)
    return ConvergedContactIncrement(
        marker=marker_base + fraction,
        displacement_vector_m=displacement,
        element_states=body.element_states,
        contact_states=contact.states,
        edge_contact_states=contact.edge_contact_states,
        archived_edge_contact_states=committed.archived_edge_contact_states,
        contact_kinematics=contact.kinematics,
        contact_pressure_Pa=contact.pressures_Pa,
        contact_force_vector_N=contact.force_N,
        internal_force_vector_N=body.internal_force_N,
        element_total_strain=body.element_total_strain,
        element_stress_Pa=body.element_stress_Pa,
        element_sigma_zz_Pa=body.element_sigma_zz_Pa,
        newton_iterations=critical.correction_count,
        residual_norm_N=float(np.linalg.norm(residual_free)),
        residual_tolerance_N=residual_tolerance,
        correction_norm_m=critical.correction_norm,
        correction_tolerance_m=float(seed.correction_tolerance_m),
        reference_force_N=float(seed.reference_force_N),
        reference_contact_size_m=float(seed.reference_contact_size_m),
        reference_displacement_m=float(seed.reference_displacement_m),
        residual_absolute_tolerance_N=residual_absolute_tolerance,
        displacement_absolute_tolerance_m=float(seed.displacement_absolute_tolerance_m),
        residual_norm_history_N=(critical.residual_norm,),
        total_contact_force_N=total_contact,
        rigid_grain_reaction_N=-total_contact,
        fixed_reaction_x_N=fixed_x,
        fixed_reaction_y_N=fixed_y,
        balance_residual_x_N=float(total_contact[0] + fixed_x),
        balance_residual_y_N=float(total_contact[1] + fixed_y),
        maximum_penetration_m=max(contact.integrated_maximum_penetration_m,
            max(item.penetration_m for item in contact.kinematics)),
        grain_center_m=(
            (float(grain.center_x_m), float(grain.center_y_m))
            if hasattr(grain, "center_x_m")
            else None
        ),
        candidate_keys=prepared.candidate_keys,
        grain_reference_m=(float(reference[0]), float(reference[1])),
        friction_dissipation_increment_J=contact.friction_dissipation_increment_J,
        augmented_iterations=0,
        damage_states=body.damage_states,
        background_element_ids=np.asarray(
            prepared.active_to_background_element_ids, dtype=np.int32
        ).copy(),
        path_state=critical,
        solve_mode=SolveMode.PATH_CONTINUATION,
        accepted_path_reference=getattr(committed,'accepted_path_reference',None),
    )


def advance_normal_contact_increment(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    *,
    grain: RigidAnalyticalGrain,
    penalty_Pa_per_m: object,
    target_marker: object,
    tangential_penalty_Pa_per_m: object | None = None,
    friction_coefficient: object = 0.0,
    friction_regularization_ratio: object = 1.0e-6,
    damage_input: DamageInput | None = None,
    time_step_s: float | None = None,
    velocity_m_per_s: object | None = None,
    density_kg_per_m3: float | None = None,
    initial_contact_history: CommittedContactStructureState | ConvergedContactIncrement | None = None,
) -> ConvergedContactIncrement:
    """Solve one prescribed-grain normal-contact increment and commit on success."""

    marker = float(target_marker)
    penalty = float(penalty_Pa_per_m)
    tangential_penalty = (
        None
        if tangential_penalty_Pa_per_m is None
        else float(tangential_penalty_Pa_per_m)
    )
    coefficient = float(friction_coefficient)
    regularization = float(friction_regularization_ratio)
    if not math.isfinite(marker):
        raise ValueError("target_marker must be finite")
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty_Pa_per_m must be finite and positive")
    if tangential_penalty is not None and (
        not math.isfinite(tangential_penalty) or tangential_penalty <= 0.0
    ):
        raise ValueError("tangential_penalty_Pa_per_m must be finite and positive")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("friction_coefficient must be finite and nonnegative")
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("friction_regularization_ratio must be finite and positive")
    if len(committed.element_states) != prepared.element_count:
        raise ContactSolverError("committed material-state count does not match mesh")
    aligned_contact_states = _aligned_contact_states(prepared, committed)
    if aligned_contact_states != committed.contact_states:
        committed = replace(
            committed,
            contact_states=aligned_contact_states,
            candidate_keys=prepared.candidate_keys,
        )
    displacement = np.asarray(committed.displacement_vector_m, dtype=float).copy()
    if displacement.shape != (prepared.total_degrees_of_freedom,):
        raise ContactSolverError("committed displacement vector shape is invalid")

    structural_committed = type("StructuralCommit", (), {})()
    structural_committed.displacement_vector_m = np.asarray(
        committed.displacement_vector_m, dtype=float
    )
    structural_committed.element_states = committed.element_states

    residual_history: list[float] = []
    correction_norm = 0.0
    body_trial = None
    contact_trial = None
    free = prepared.structural.free_dofs
    reference_contact_size = float(
        np.median(prepared.candidate_tributary_lengths_m)
    )
    reference_displacement = max(
        max(
            motion.penetration_m
            for motion in (
                rigid_contact_query(grain, coordinate)
                for coordinate in prepared.candidate_reference_coordinates_m
            )
        ),
        1.0e-10 * reference_contact_size,
    )
    reference_force = (
        penalty
        * reference_displacement
        * reference_contact_size
        * case.analysis.thickness
    )
    residual_absolute_tolerance = max(
        case.newton.residual_absolute_tolerance_N,
        1.0e-12,
        1.0e-10 * reference_force,
    )
    displacement_absolute_tolerance = max(
        case.newton.displacement_absolute_tolerance_m,
        1.0e-15,
        1.0e-10 * reference_contact_size,
    )
    mass = None
    dynamic_increment = None
    if time_step_s is not None:
        from .inertial_transition import assemble_consistent_mass_matrix
        dt = float(time_step_s)
        velocity0 = np.asarray(velocity_m_per_s, dtype=float)
        if (not math.isfinite(dt) or dt <= 0.0
                or velocity0.shape != displacement.shape
                or not np.all(np.isfinite(velocity0))
                or np.any(velocity0[prepared.structural.fixed_dofs] != 0.0)):
            raise ValueError('invalid physical transient step or velocity')
        mass = assemble_consistent_mass_matrix(prepared.structural,
            density_kg_per_m3=density_kg_per_m3, thickness_m=case.analysis.thickness)
        displacement0 = displacement.copy()
        old_reference = np.asarray(committed.grain_reference_m, dtype=float)
        if hasattr(grain, 'center_x_m'):
            grain0 = replace(grain, center_x_m=float(old_reference[0]),
                             center_y_m=float(old_reference[1]))
        else:
            grain0 = replace(grain, reference_x_m=float(old_reference[0]),
                             reference_y_m=float(old_reference[1]))
        if damage_input is not None:
            # The old endpoint is already committed. Re-entering the local
            # damage return at zero increment can demand a joint coordinate
            # on a flat/decreasing resistance surface even when this physical
            # step unloads. Its force follows directly from nominal stress;
            # only the new endpoint needs a constitutive trial and tangent.
            internal0 = _committed_damage_fields(case, prepared, committed).internal_force_N
        else:
            internal0 = _assemble_trial(
                case, prepared.structural, structural_committed, displacement0
            ).internal_force_N
        contact0 = _assemble_contact(case, prepared,
            committed if initial_contact_history is None else initial_contact_history, displacement0, grain0,
            penalty, tangential_penalty, coefficient, regularization)
        initial_net_force = contact0.force_N - internal0
        dynamic_increment = dt*velocity0.copy()
        displacement = displacement0+dynamic_increment
    elif velocity_m_per_s is not None or density_kg_per_m3 is not None:
        raise ValueError('transient velocity and density require physical time step')

    def equation_residual(body, contact, trial_increment):
        net_force = contact.force_N - body.internal_force_N
        if mass is None:
            return net_force
        velocity1 = 2.0 * trial_increment / dt - velocity0
        return 0.5 * (net_force+initial_net_force) - mass @ ((velocity1-velocity0)/dt)
    entry_material_side=None
    if mass is None and getattr(prepared,'circle_edge_quadrature_order',0) and damage_input is not None:
        motion=_grain_reference(grain)-np.asarray(committed.grain_reference_m)
        if np.linalg.norm(motion)>0. and any(d.damage>0. for d in committed.damage_states):
            accepted_grain=replace(grain,center_x_m=grain.center_x_m-motion[0],
                                   center_y_m=grain.center_y_m-motion[1])
            entry_material_side=close_path_entry_material_side(case,prepared,committed,
                grain=accepted_grain,grain_direction_m=motion,penalty=penalty,
                tangential=tangential_penalty,friction_coefficient=coefficient,
                regularization=regularization,damage_input=damage_input)
    accepted_dynamic_trial = None
    for correction_count in range(case.newton.maximum_iterations + 1):
        body_trial = accepted_dynamic_trial[0] if accepted_dynamic_trial is not None else (
            _assemble_damage_trial(
                case, prepared, committed, displacement, damage_input,
                active_side_direction=entry_material_side if correction_count==0 else None
            )
            if damage_input is not None
            else _assemble_trial(
                case, prepared.structural, structural_committed, displacement
            )
        )
        contact_trial = accepted_dynamic_trial[1] if accepted_dynamic_trial is not None else _assemble_contact(
            case,
            prepared,
            committed,
            displacement,
            grain,
            penalty,
            tangential_penalty,
            coefficient,
            regularization,
        )
        accepted_dynamic_trial = None
        residual = equation_residual(body_trial, contact_trial, dynamic_increment)
        residual_norm = float(np.linalg.norm(residual[free]))
        residual_history.append(residual_norm)
        contact_scale = float(np.linalg.norm(contact_trial.force_N[free]))
        residual_tolerance = (
            residual_absolute_tolerance
            + case.newton.residual_relative_tolerance * contact_scale
        )
        displacement_scale = float(np.linalg.norm(displacement[free]))
        correction_tolerance = (
            displacement_absolute_tolerance
            + case.newton.displacement_relative_tolerance * displacement_scale
        )
        if residual_norm <= residual_tolerance and correction_norm <= correction_tolerance:
            break
        if correction_count >= case.newton.maximum_iterations:
            raise ContactConvergenceError(
                "contact Newton iteration did not converge before maximum_iterations",
                attempted_marker=marker,
                residual_norm_history_N=tuple(residual_history),
            )
        tangent = (
            body_trial.tangent_stiffness_N_per_m
            + contact_trial.residual_tangent_N_per_m
        )
        if mass is not None:
            tangent = 0.5 * tangent + 2.0 * mass / dt**2
        tangent_free = tangent[free][:, free]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                correction_free = np.asarray(
                    spsolve(tangent_free, residual[free]), dtype=float
                )
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                raise ContactSolverError("contact Newton tangent solve failed") from exc
        if not np.all(np.isfinite(correction_free)):
            raise ContactSolverError("contact Newton correction is non-finite")
        accepted = False
        step_factor = 1.0
        line_search_trial_norms: list[float] = []
        line_search_contact_counts: list[tuple[int, int, int]] = []
        line_search_damage_branch_counts: list[tuple[tuple[str, int], ...]] = []
        line_search_damage_states: list[tuple[DamageState, ...]] = []
        material_domain_failure = None
        for _line_search in range(13):
            candidate_increment = None
            if mass is not None:
                candidate_increment = dynamic_increment.copy()
                candidate_increment[free] += step_factor*correction_free
                candidate = displacement0+candidate_increment
            else:
                candidate = displacement.copy()
                candidate[free] += step_factor * correction_free
            candidate[prepared.structural.fixed_dofs] = 0.0
            try:
                candidate_body = (
                    _assemble_damage_trial(case, prepared, committed, candidate, damage_input)
                    if damage_input is not None
                    else _assemble_trial(case, prepared.structural, structural_committed, candidate)
                )
            except DamagePathConstraintRequired as exc:
                # An unbalanced line-search candidate is outside the local
                # return domain. Preserve its cause; do not create a material
                # event or an artificial finite residual for this candidate.
                material_domain_failure = exc
                step_factor *= 0.5
                continue
            candidate_contact = _assemble_contact(
                case,
                prepared,
                committed,
                candidate,
                grain,
                penalty,
                tangential_penalty,
                coefficient,
                regularization,
            )
            candidate_residual = equation_residual(candidate_body, candidate_contact, candidate_increment)
            candidate_norm = float(np.linalg.norm(candidate_residual[free]))
            line_search_trial_norms.append(candidate_norm)
            candidate_statuses = [state.status for state in candidate_contact.states]
            line_search_contact_counts.append((
                candidate_statuses.count("open"),
                candidate_statuses.count("stick"),
                candidate_statuses.count("slip"),
            ))
            candidate_branches = tuple(
                getattr(candidate_body, "tangent_branches", ())
            )
            line_search_damage_branch_counts.append(tuple(
                sorted(
                    (name, candidate_branches.count(name))
                    for name in set(candidate_branches)
                )
            ))
            line_search_damage_states.append(
                tuple(getattr(candidate_body, "damage_states", ()))
            )
            transient_converged_correction = (
                mass is not None and candidate_norm <= residual_tolerance
                and float(np.linalg.norm(step_factor * correction_free)) <= correction_tolerance
            )
            if math.isfinite(candidate_norm) and (
                candidate_norm < residual_norm or transient_converged_correction
            ):
                displacement = candidate
                dynamic_increment = candidate_increment
                if mass is not None:
                    # The accepted candidate is exactly the next iterate,
                    # with the same committed material/contact histories.
                    # Retain its assemblies but recheck residual/correction
                    # at the top of the Newton loop before convergence.
                    accepted_dynamic_trial = (candidate_body, candidate_contact)
                correction_norm = float(
                    np.linalg.norm(step_factor * correction_free)
                )
                accepted = True
                break
            step_factor *= 0.5
        if not accepted:
            diagnostics = _newton_failure_diagnostics(
                prepared,
                body_trial,
                contact_trial,
                tangent_free,
                damage_input,
            )
            diagnostics["contact_open_stick_slip_counts"] = (
                *diagnostics["contact_open_stick_slip_counts"],
                *line_search_contact_counts,
            )
            diagnostics["damage_branch_counts"] = (
                *diagnostics["damage_branch_counts"],
                *line_search_damage_branch_counts,
            )
            event_trial_damage = next(
                (
                    states
                    for states in line_search_damage_states
                    if event_aware_substep_scale(
                        background_element_ids=prepared.active_to_background_element_ids,
                        committed_damage_states=committed.damage_states,
                        trial_damage_states=states,
                        damage_event_tolerance=damage_input.damage_event_tolerance,
                        fracture_energy_J_per_m2=damage_input.fracture_energy_J_per_m2,
                    ) is not None
                ),
                line_search_damage_states[0] if line_search_damage_states else (),
            ) if damage_input is not None else ()
            diagnostics["event_trial_background_element_ids"] = tuple(
                int(value) for value in prepared.active_to_background_element_ids
            )
            diagnostics["event_trial_damage_states"] = event_trial_damage
            raise ContactConvergenceError(
                "contact Newton line search could not reduce the residual",
                attempted_marker=marker,
                residual_norm_history_N=tuple(residual_history),
                line_search_trial_norms_N=tuple(line_search_trial_norms),
                **diagnostics,
            ) from material_domain_failure
    else:  # pragma: no cover
        raise AssertionError("unreachable Newton loop exit")

    assert body_trial is not None and contact_trial is not None
    equilibrium = body_trial.internal_force_N - contact_trial.force_N
    dynamic_fields = {}
    if mass is not None:
        velocity1 = 2.0 * dynamic_increment / dt - velocity0
        velocity1[prepared.structural.fixed_dofs] = 0.0
        # The velocity difference gives the interval-average acceleration.
        # Persist the endpoint acceleration from the endpoint force balance,
        # consistent with topology rebasing and physical checkpoint semantics.
        acceleration = np.zeros_like(velocity1)
        free = prepared.structural.free_dofs
        acceleration[free] = spsolve(mass[free][:, free], -equilibrium[free])
        equilibrium = equilibrium + mass @ acceleration
        dynamic_fields = dict(
            solve_mode=SolveMode.INERTIAL_TRANSITION,
            transient_step_count=int(getattr(committed,'transient_step_count',0))+1,
            transient_last_time_step_s=dt,
            transient_absolute_energy_error_J=float(getattr(committed, 'transient_absolute_energy_error_J', 0.0)),
            transient_maximum_energy_scale_J=float(getattr(committed, 'transient_maximum_energy_scale_J', 0.0)),
            physical_time_s=float(getattr(committed, 'physical_time_s', 0.0))+dt,
            velocity_vector_m_per_s=velocity1.copy(),
            acceleration_vector_m_per_s2=acceleration.copy(),
            kinetic_energy_J=0.5 * float(velocity1 @ mass @ velocity1),
        )
    fixed_x = float(
        math.fsum(float(v) for v in equilibrium[prepared.structural.fixed_x_dofs])
    )
    fixed_y = float(
        math.fsum(float(v) for v in equilibrium[prepared.structural.fixed_y_dofs])
    )
    total_contact = np.array(
        [
            math.fsum(
                float(contact_trial.force_N[dofs[component]])
                for dofs in prepared.candidate_component_dofs
            )
            for component in (0, 1)
        ],
        dtype=float,
    )
    return ConvergedContactIncrement(
        marker=marker,
        accepted_path_reference=getattr(committed,"accepted_path_reference",None),
        displacement_vector_m=displacement.copy(),
        element_states=body_trial.element_states,
        contact_states=contact_trial.states,
        edge_contact_states=contact_trial.edge_contact_states,
        archived_edge_contact_states=committed.archived_edge_contact_states,
        contact_kinematics=contact_trial.kinematics,
        contact_pressure_Pa=contact_trial.pressures_Pa,
        contact_force_vector_N=contact_trial.force_N,
        internal_force_vector_N=body_trial.internal_force_N,
        element_total_strain=body_trial.element_total_strain,
        element_stress_Pa=body_trial.element_stress_Pa,
        element_sigma_zz_Pa=body_trial.element_sigma_zz_Pa,
        newton_iterations=correction_count,
        residual_norm_N=residual_history[-1],
        residual_tolerance_N=residual_tolerance,
        correction_norm_m=correction_norm,
        correction_tolerance_m=correction_tolerance,
        reference_force_N=reference_force,
        reference_contact_size_m=reference_contact_size,
        reference_displacement_m=reference_displacement,
        residual_absolute_tolerance_N=residual_absolute_tolerance,
        displacement_absolute_tolerance_m=displacement_absolute_tolerance,
        residual_norm_history_N=tuple(residual_history),
        total_contact_force_N=total_contact,
        rigid_grain_reaction_N=-total_contact,
        fixed_reaction_x_N=fixed_x,
        fixed_reaction_y_N=fixed_y,
        balance_residual_x_N=(float(total_contact[0] + fixed_x) if mass is None else
            float(np.sum(equilibrium[prepared.structural.component_dofs[0]]) - fixed_x)),
        balance_residual_y_N=(float(total_contact[1] + fixed_y) if mass is None else
            float(np.sum(equilibrium[prepared.structural.component_dofs[1]]) - fixed_y)),
        maximum_penetration_m=max(contact_trial.integrated_maximum_penetration_m,max(
            motion.penetration_m for motion in contact_trial.kinematics
        )),
            grain_center_m=(
                (float(grain.center_x_m), float(grain.center_y_m))
                if hasattr(grain, "center_x_m") else None
            ),
            candidate_keys=prepared.candidate_keys,
            grain_reference_m=tuple(_grain_reference(grain)),
        friction_dissipation_increment_J=contact_trial.friction_dissipation_increment_J,
        augmented_iterations=0,
        damage_states=getattr(body_trial, "damage_states", ()),
        separating_states=getattr(body_trial, "separating_states", ()),
        background_element_ids=np.asarray(
            prepared.active_to_background_element_ids
            if prepared.active_to_background_element_ids.size
            else np.arange(prepared.element_count, dtype=np.int32),
            dtype=np.int32,
        ),
        **dynamic_fields,
    )


def advance_augmented_contact_increment(
    case: ElastoplasticFemCase,
    prepared: PreparedContactMesh,
    committed: CommittedContactStructureState | ConvergedContactIncrement,
    *,
    grain: RigidAnalyticalGrain,
    penalty_Pa_per_m: object,
    relaxation: object,
    maximum_outer_iterations: object,
    penetration_tolerance_m: object,
    target_marker: object,
    tangential_penalty_Pa_per_m: object | None = None,
    friction_coefficient: object = 0.0,
    friction_regularization_ratio: object = 1.0e-6,
    damage_input: DamageInput | None = None,
    time_step_s: float | None = None,
    velocity_m_per_s: object | None = None,
    density_kg_per_m3: float | None = None,
) -> ConvergedContactIncrement:
    """Run Uzawa updates outside Newton while preserving the original J2 commit."""

    penalty = float(penalty_Pa_per_m)
    factor = float(relaxation)
    tolerance = float(penetration_tolerance_m)
    if type(maximum_outer_iterations) is not int or maximum_outer_iterations < 1:
        raise ValueError("maximum_outer_iterations must be a positive strict integer")
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("relaxation must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("penetration_tolerance_m must be finite and nonnegative")

    multiplier_states = committed.contact_states
    edge_multiplier_states = committed.edge_contact_states
    for outer in range(1, maximum_outer_iterations + 1):
        trial_commit = _replace_compatible_dataclass(
            committed,
            contact_states=multiplier_states,
            edge_contact_states=edge_multiplier_states,
            candidate_keys=prepared.candidate_keys,
        )
        converged = advance_normal_contact_increment(
            case,
            prepared,
            trial_commit,
            grain=grain,
            penalty_Pa_per_m=penalty,
            target_marker=target_marker,
            tangential_penalty_Pa_per_m=tangential_penalty_Pa_per_m,
            friction_coefficient=friction_coefficient,
            friction_regularization_ratio=friction_regularization_ratio,
            damage_input=damage_input,
            **(dict(time_step_s=time_step_s,velocity_m_per_s=velocity_m_per_s,
                density_kg_per_m3=density_kg_per_m3,initial_contact_history=committed)
                if time_step_s is not None else {}),
        )
        loaded_gap=max((abs(motion.signed_gap_m)
            for motion,pressure in zip(converged.contact_kinematics,converged.contact_pressure_Pa,strict=True)
            if pressure>0.),default=0.)
        if outer > 1 and max(converged.maximum_penetration_m,loaded_gap) <= tolerance:
            return ConvergedContactIncrement(
                **{
                    name: getattr(converged, name)
                    for name in converged.__dataclass_fields__
                    if name != "augmented_iterations"
                },
                augmented_iterations=outer - 1,
            )
        multiplier_states = tuple(
            update_augmented_normal_multiplier(
                # Outer iterations refine a normal multiplier at one physical
                # target. Reusing the trial slip/traction advances friction
                # history repeatedly over that same displacement increment.
                replace(original, normal_multiplier_Pa=state.normal_multiplier_Pa),
                gap_m=motion.signed_gap_m,
                penalty_Pa_per_m=penalty,
                relaxation=factor,
            )
            for original, state, motion in zip(
                committed.contact_states, converged.contact_states,
                converged.contact_kinematics, strict=True,
            )
        )
        if prepared.edge_endpoint_keys:
            # Uzawa changes only the normal multiplier. Friction is a trial
            # over the original physical increment, never an outer-iteration
            # history increment.
            accepted_edges = dict(committed.edge_contact_states)
            node_index = {
                int(background): index
                for index, background in enumerate(prepared.candidate_background_node_ids)
            }
            edge_multiplier_states = tuple(
                (key, update_augmented_normal_multiplier(
                    ContactPointState(
                        normal_multiplier_Pa=state.normal_multiplier_Pa,
                        tangential_traction_Pa=accepted_edges.get(key, ContactPointState()).tangential_traction_Pa,
                        accumulated_slip_m=accepted_edges.get(key, ContactPointState()).accumulated_slip_m,
                        status=accepted_edges.get(key, ContactPointState()).status,
                    ),
                    gap_m=converged.contact_kinematics[node_index[key[2]]].signed_gap_m,
                    penalty_Pa_per_m=penalty,
                    relaxation=factor,
                ))
                for key, state in converged.edge_contact_states
            )
    raise ContactConvergenceError(
        "augmented contact iteration did not converge before maximum_outer_iterations",
        attempted_marker=float(target_marker),
        residual_norm_history_N=converged.residual_norm_history_N,
    )
