"""Independent solver routing and material-state contracts for separation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .damage_material import DamageState
from .damage_models import DamageInput
from .material_topology import MaterialState
from .energetic_damage import EnergeticDamageHistory


class SolveMode(str, Enum):
    STANDARD_NEWTON = "STANDARD_NEWTON"
    PATH_CONTINUATION = "PATH_CONTINUATION"
    SEPARATION_CONSTRAINED = "SEPARATION_CONSTRAINED"
    INERTIAL_TRANSITION = "INERTIAL_TRANSITION"


class SolveFailure(str, Enum):
    NEWTON = "NEWTON"
    LINE_SEARCH = "LINE_SEARCH"
    RETRY_BUDGET = "RETRY_BUDGET"
    MINIMUM_SUBSTEP = "MINIMUM_SUBSTEP"


class SeparationConstrainedError(RuntimeError):
    """Raised when the separation-augmented correction cannot be accepted."""


class TerminalSupportProjectionError(RuntimeError):
    """Raised when a proposed terminal projection is not a pure zero mode."""


@dataclass(frozen=True, slots=True)
class SolveRoute:
    solve_mode: SolveMode
    material_state: MaterialState
    failure: SolveFailure


@dataclass(frozen=True, slots=True)
class SeparationEligibility:
    converged: bool
    in_existing_softening_branch: bool
    remaining_fracture_energy_J: float
    surface_connected: bool
    in_terminal_tracking_interval: bool
    topology_precheck_passed: bool


@dataclass(frozen=True, slots=True)
class SeparatingState:
    material_state: MaterialState
    s_e: float
    entry_softening_fraction: float
    initiation_equivalent_stress_Pa: float
    effective_equivalent_stress_Pa: float
    characteristic_length_m: float
    terminal_displacement_m: float
    entry_post_initiation_plastic_displacement_m: float
    entry_strain_direction: tuple[float, float, float]
    damage: float
    nominal_equivalent_stress_Pa: float
    alpha_e: float
    entry_damage_dissipation_density_J_per_m3: float
    damage_dissipation_density_J_per_m3: float
    separation_damage_dissipation_density_J_per_m3: float
    remaining_fracture_energy_density_J_per_m3: float
    energetic_history: EnergeticDamageHistory | None = None


@dataclass(frozen=True, slots=True)
class SeparationConstrainedSettings:
    maximum_corrections: int
    residual_tolerance: float
    path_tolerance: float
    separation_tolerance: float
    displacement_correction_tolerance: float

    def __post_init__(self) -> None:
        if type(self.maximum_corrections) is not int or self.maximum_corrections < 1:
            raise ValueError("maximum_corrections must be a positive integer")
        for name in (
            "residual_tolerance",
            "path_tolerance",
            "separation_tolerance",
            "displacement_correction_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SeparationConstrainedState:
    displacement: NDArray[np.float64]
    marker: float
    separation_progress: NDArray[np.float64]
    residual_norm: float
    path_residual: float
    separation_residual_norm: float
    correction_norm: float
    correction_count: int


@dataclass(frozen=True, slots=True)
class ZeroForceCommitThresholds:
    element_internal_force_absolute_N: float
    projection_reaction_absolute_N: float
    remaining_fracture_energy_absolute_J: float
    topology_energy_jump_absolute_J: float
    balance_residual_absolute_N: float
    path_residual_absolute_m: float
    separation_residual_absolute: float
    displacement_correction_absolute_m: float


@dataclass(frozen=True, slots=True)
class ZeroForceCommitMeasures:
    constitutive_capacity_is_exactly_zero: bool
    element_internal_force_norm_N: float
    projection_reaction_norm_N: float
    remaining_fracture_energy_J: float
    topology_energy_jump_J: float
    balance_residual_N: float
    path_residual_m: float
    separation_residual: float
    displacement_correction_m: float
    damage_energy_fully_accounted: bool
    removed_dof_kinetic_energy_J: float = 0.0
    kinetic_energy_absolute_tolerance_J: float = math.inf


@dataclass(frozen=True, slots=True)
class TerminalSupportProjection:
    retained_dofs: NDArray[np.int32]
    projected_dofs: NDArray[np.int32]
    reduced_tangent: NDArray[np.float64]
    reduced_residual: NDArray[np.float64]
    projection_reaction_norm: float


def terminal_support_projection(
    *,
    tangent: object,
    residual: object,
    candidate_zero_energy_dofs: object,
    absolute_force_tolerance: object,
) -> TerminalSupportProjection:
    """Remove only algebraically uncoupled zero-force, zero-energy directions."""

    matrix = np.asarray(tangent, dtype=float)
    vector = np.asarray(residual, dtype=float)
    candidates = np.asarray(candidate_zero_energy_dofs, dtype=np.int64)
    tolerance = float(absolute_force_tolerance)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("tangent must be a square matrix")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("residual shape does not match tangent")
    if candidates.ndim != 1 or np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_zero_energy_dofs must be a unique vector")
    if np.any(candidates < 0) or np.any(candidates >= matrix.shape[0]):
        raise ValueError("candidate_zero_energy_dofs contains an invalid index")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("absolute_force_tolerance must be finite and positive")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("projection inputs must be finite")
    candidates = np.sort(candidates).astype(np.int32)
    retained = np.setdiff1d(
        np.arange(matrix.shape[0], dtype=np.int32), candidates, assume_unique=True
    )
    reaction_norm = float(np.linalg.norm(vector[candidates]))
    if reaction_norm > tolerance:
        raise TerminalSupportProjectionError(
            "candidate projection has a nonzero residual reaction"
        )
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    coupling_tolerance = 64.0 * np.finfo(float).eps * scale
    candidate_block = matrix[np.ix_(candidates, candidates)]
    coupling = matrix[np.ix_(candidates, retained)]
    reverse_coupling = matrix[np.ix_(retained, candidates)]
    if max(
        float(np.linalg.norm(candidate_block, ord=np.inf)) if candidate_block.size else 0.0,
        float(np.linalg.norm(coupling, ord=np.inf)) if coupling.size else 0.0,
        float(np.linalg.norm(reverse_coupling, ord=np.inf)) if reverse_coupling.size else 0.0,
    ) > coupling_tolerance:
        raise TerminalSupportProjectionError(
            "candidate projection is stiffness coupled to a positive-capacity degree of freedom"
        )
    return TerminalSupportProjection(
        retained_dofs=retained,
        projected_dofs=candidates,
        reduced_tangent=matrix[np.ix_(retained, retained)].copy(),
        reduced_residual=vector[retained].copy(),
        projection_reaction_norm=reaction_norm,
    )


def initialize_separation_path_direction(
    augmented_jacobian_without_path_row: object,
    *,
    reference_direction: object,
) -> NDArray[np.float64]:
    """Construct a deterministic tangent from one converged augmented state."""

    matrix = np.asarray(augmented_jacobian_without_path_row, dtype=float)
    reference = np.asarray(reference_direction, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != matrix.shape[0] + 1:
        raise ValueError("separation tangent system must have one-dimensional nullspace")
    if reference.shape != (matrix.shape[1],):
        raise ValueError("reference_direction shape is invalid")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(reference)):
        raise ValueError("separation tangent inputs must be finite")
    try:
        _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise SeparationConstrainedError(
            "separation tangent factorization failed"
        ) from exc
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    rank_tolerance = 64.0 * np.finfo(float).eps * scale
    if np.count_nonzero(singular_values > rank_tolerance) != matrix.shape[0]:
        raise SeparationConstrainedError(
            "separation tangent system does not have a unique path direction"
        )
    tangent = np.asarray(right[-1], dtype=float)
    norm = float(np.linalg.norm(tangent))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise SeparationConstrainedError("separation tangent is zero or non-finite")
    tangent /= norm
    orientation = float(np.dot(tangent, reference))
    if orientation < 0.0:
        tangent = -tangent
    elif orientation == 0.0:
        for value in tangent:
            if value != 0.0:
                if value < 0.0:
                    tangent = -tangent
                break
    return tangent


def route_after_failure(
    *,
    solve_mode: SolveMode,
    material_state: MaterialState,
    failure: SolveFailure,
) -> SolveRoute:
    """Route an algorithm failure without changing the material state."""

    if material_state is MaterialState.REMOVED:
        raise ValueError("REMOVED material cannot enter a balance solver")
    routed_mode = (
        SolveMode.PATH_CONTINUATION
        if solve_mode is SolveMode.STANDARD_NEWTON
        else solve_mode
    )
    return SolveRoute(
        solve_mode=routed_mode,
        material_state=material_state,
        failure=failure,
    )


def can_enter_separating(evidence: SeparationEligibility) -> bool:
    """Return eligibility based only on a converged constitutive/topology state."""

    return bool(
        evidence.converged
        and evidence.in_existing_softening_branch
        and math.isfinite(evidence.remaining_fracture_energy_J)
        and evidence.remaining_fracture_energy_J > 0.0
        and evidence.surface_connected
        and evidence.in_terminal_tracking_interval
        and evidence.topology_precheck_passed
    )


def _softening_dissipation_density(
    initiation_stress_Pa: float,
    terminal_displacement_m: float,
    softening_fraction: float,
    characteristic_length_m: float,
) -> float:
    return (
        initiation_stress_Pa
        * terminal_displacement_m
        * (softening_fraction - 0.5 * softening_fraction**2)
        / characteristic_length_m
    )


def begin_separating(
    *,
    damage_input: DamageInput,
    committed: DamageState,
    characteristic_length_m: float,
    effective_equivalent_stress_Pa: float,
    entry_strain_direction: object | None = None,
) -> SeparatingState:
    """Map a committed terminal-zone DAMAGED state without a response jump."""

    if committed.status != "DAMAGED":
        raise ValueError("only a committed DAMAGED state can enter SEPARATING")
    if committed.initiation_equivalent_stress_Pa is None:
        raise ValueError("SEPARATING requires an initiated softening state")
    length = float(characteristic_length_m)
    effective = float(effective_equivalent_stress_Pa)
    initiation = float(committed.initiation_equivalent_stress_Pa)
    if not all(math.isfinite(value) and value > 0.0 for value in (length, effective, initiation)):
        raise ValueError("separating constitutive scales must be finite and positive")
    entry_direction = np.zeros(3, dtype=float)
    if entry_strain_direction is not None:
        entry_direction = np.asarray(entry_strain_direction, dtype=float)
        if entry_direction.shape != (3,) or not np.all(np.isfinite(entry_direction)):
            raise ValueError("entry_strain_direction must be a finite three-vector")
    terminal_displacement = 2.0 * damage_input.fracture_energy_J_per_m2 / initiation
    entry_fraction = min(
        1.0,
        max(
            0.0,
            1.0 - committed.nominal_equivalent_stress_Pa / initiation,
        ),
    )
    terminal_density = damage_input.fracture_energy_J_per_m2 / length
    energetic_history = getattr(committed,"energetic_history",None)
    if energetic_history is not None:
        # Coordinates only: p=l*(1-D_entry)*s; this is not a plastic opening law.
        terminal_displacement = length
        entry_fraction = committed.damage
        terminal_density = energetic_history.fracture_energy_density
    remaining_density = max(
        0.0, terminal_density - committed.damage_dissipation_density_J_per_m3
    )
    return SeparatingState(
        material_state=MaterialState.SEPARATING,
        s_e=0.0,
        entry_softening_fraction=entry_fraction,
        initiation_equivalent_stress_Pa=initiation,
        effective_equivalent_stress_Pa=effective,
        characteristic_length_m=length,
        terminal_displacement_m=terminal_displacement,
        entry_post_initiation_plastic_displacement_m=(
            committed.post_initiation_plastic_displacement_m
        ),
        entry_strain_direction=tuple(float(value) for value in entry_direction),
        damage=committed.damage,
        nominal_equivalent_stress_Pa=committed.nominal_equivalent_stress_Pa,
        alpha_e=1.0,
        entry_damage_dissipation_density_J_per_m3=(
            committed.damage_dissipation_density_J_per_m3
        ),
        damage_dissipation_density_J_per_m3=(
            committed.damage_dissipation_density_J_per_m3
        ),
        separation_damage_dissipation_density_J_per_m3=0.0,
        remaining_fracture_energy_density_J_per_m3=remaining_density,
        energetic_history=energetic_history,
    )


def separation_constraint_terms(
    committed: SeparatingState,
    *,
    current_post_initiation_plastic_displacement_m: object,
    separation_progress: object,
) -> tuple[float, float]:
    """Return the existing terminal-segment constraint and its ds_e derivative."""

    current = float(current_post_initiation_plastic_displacement_m)
    progress = float(separation_progress)
    if not math.isfinite(current):
        raise ValueError("current plastic displacement must be finite")
    if not math.isfinite(progress) or progress < committed.s_e or progress > 1.0:
        raise ValueError("separation_progress must be irreversible and within [0, 1]")
    remaining_segment = committed.terminal_displacement_m * (
        1.0 - committed.entry_softening_fraction
    )
    return (
        current
        - committed.entry_post_initiation_plastic_displacement_m
        - progress * remaining_segment,
        -remaining_segment,
    )


def advance_separating(committed: SeparatingState, *, s_e: float) -> SeparatingState:
    """Evaluate the same linear softening law on its remaining terminal segment."""

    progress = float(s_e)
    if not math.isfinite(progress) or progress < committed.s_e or progress > 1.0:
        raise ValueError("s_e must be finite, irreversible, and within [0, 1]")
    history=getattr(committed,"energetic_history",None)
    if history is not None:
        from dataclasses import replace
        from .thermodynamic_energy import fracture_resistance
        damage=history.damage+progress*(1.-history.damage)
        curve=fracture_resistance(initial_driving_energy=history.initial_driving_energy,
                                 fracture_energy_density=history.fracture_energy_density)
        total=curve.cumulative(damage)
        if progress==1.:
            damage=1.; total=history.fracture_energy_density
        return replace(committed,s_e=progress,damage=damage,
            nominal_equivalent_stress_Pa=(1.-damage)*committed.effective_equivalent_stress_Pa,
            # A naturally completed dynamic entry has no remaining damage
            # interval. Its metadata carries zero capacity at the same time.
            alpha_e=0. if history.damage==1. else (1.-damage)/(1.-history.damage),
            damage_dissipation_density_J_per_m3=total,
            separation_damage_dissipation_density_J_per_m3=total-committed.entry_damage_dissipation_density_J_per_m3,
            remaining_fracture_energy_density_J_per_m3=history.fracture_energy_density-total)
    fraction = committed.entry_softening_fraction + progress * (
        1.0 - committed.entry_softening_fraction
    )
    if progress == 1.0:
        nominal = 0.0
        damage = 1.0
        alpha = 0.0
    else:
        nominal = committed.initiation_equivalent_stress_Pa * (1.0 - fraction)
        damage = min(
            1.0,
            max(0.0, 1.0 - nominal / committed.effective_equivalent_stress_Pa),
        )
        entry_nominal = committed.initiation_equivalent_stress_Pa * (
            1.0 - committed.entry_softening_fraction
        )
        alpha = nominal / entry_nominal if entry_nominal > 0.0 else 0.0
    total = _softening_dissipation_density(
        committed.initiation_equivalent_stress_Pa,
        committed.terminal_displacement_m,
        fraction,
        committed.characteristic_length_m,
    )
    terminal = _softening_dissipation_density(
        committed.initiation_equivalent_stress_Pa,
        committed.terminal_displacement_m,
        1.0,
        committed.characteristic_length_m,
    )
    if progress == 1.0:
        total = terminal
    total = max(committed.entry_damage_dissipation_density_J_per_m3, total)
    return SeparatingState(
        material_state=MaterialState.SEPARATING,
        s_e=progress,
        entry_softening_fraction=committed.entry_softening_fraction,
        initiation_equivalent_stress_Pa=committed.initiation_equivalent_stress_Pa,
        effective_equivalent_stress_Pa=committed.effective_equivalent_stress_Pa,
        characteristic_length_m=committed.characteristic_length_m,
        terminal_displacement_m=committed.terminal_displacement_m,
        entry_post_initiation_plastic_displacement_m=(
            committed.entry_post_initiation_plastic_displacement_m
        ),
        entry_strain_direction=committed.entry_strain_direction,
        damage=damage,
        nominal_equivalent_stress_Pa=nominal,
        alpha_e=alpha,
        entry_damage_dissipation_density_J_per_m3=(
            committed.entry_damage_dissipation_density_J_per_m3
        ),
        damage_dissipation_density_J_per_m3=total,
        separation_damage_dissipation_density_J_per_m3=(
            total - committed.entry_damage_dissipation_density_J_per_m3
        ),
        remaining_fracture_energy_density_J_per_m3=max(0.0, terminal - total),
    )


def separation_constrained_correct(
    *,
    displacement: object,
    marker: object,
    separation_progress: object,
    equations: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    settings: SeparationConstrainedSettings,
    minimum_separation_progress: object | None = None,
) -> SeparationConstrainedState:
    """Correct displacement, path marker, and existing-softening progress jointly."""

    trial_displacement = np.asarray(displacement, dtype=float).copy()
    trial_progress = np.asarray(separation_progress, dtype=float).copy()
    trial_marker = float(marker)
    if trial_displacement.ndim != 1 or trial_progress.ndim != 1:
        raise ValueError("separation constrained unknowns must be vectors")
    if not (
        np.all(np.isfinite(trial_displacement))
        and np.all(np.isfinite(trial_progress))
        and math.isfinite(trial_marker)
    ):
        raise ValueError("separation constrained unknowns must be finite")
    minimum = (
        trial_progress.copy()
        if minimum_separation_progress is None
        else np.asarray(minimum_separation_progress, dtype=float).copy()
    )
    if minimum.shape != trial_progress.shape or not np.all(np.isfinite(minimum)):
        raise ValueError("minimum_separation_progress shape is invalid")
    combined = np.concatenate((trial_displacement, [trial_marker], trial_progress))
    displacement_size = trial_displacement.size
    correction_norm = 0.0
    for correction_count in range(settings.maximum_corrections + 1):
        values = np.asarray(equations(combined), dtype=float)
        expected_size = combined.size
        if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
            raise SeparationConstrainedError("augmented equations are invalid")
        residual = values[:displacement_size]
        path_residual = float(values[displacement_size])
        separation_residual = values[displacement_size + 1 :]
        residual_norm = float(np.linalg.norm(residual))
        separation_norm = float(np.linalg.norm(separation_residual))
        if (
            residual_norm <= settings.residual_tolerance
            and abs(path_residual) <= settings.path_tolerance
            and separation_norm <= settings.separation_tolerance
            and correction_norm <= settings.displacement_correction_tolerance
        ):
            progress = combined[displacement_size + 1 :].copy()
            if np.any(progress + 64.0 * np.finfo(float).eps < minimum):
                raise SeparationConstrainedError(
                    "separation progress must remain irreversible"
                )
            return SeparationConstrainedState(
                displacement=combined[:displacement_size].copy(),
                marker=float(combined[displacement_size]),
                separation_progress=progress,
                residual_norm=residual_norm,
                path_residual=path_residual,
                separation_residual_norm=separation_norm,
                correction_norm=correction_norm,
                correction_count=correction_count,
            )
        if correction_count >= settings.maximum_corrections:
            break
        matrix = np.asarray(jacobian(combined), dtype=float)
        if matrix.shape != (expected_size, expected_size) or not np.all(
            np.isfinite(matrix)
        ):
            raise SeparationConstrainedError("augmented Jacobian is invalid")
        try:
            correction = np.linalg.solve(matrix, -values)
        except np.linalg.LinAlgError as exc:
            raise SeparationConstrainedError(
                "separation augmented correction system is singular"
            ) from exc
        candidate = combined + correction
        candidate_progress = candidate[displacement_size + 1 :]
        if np.any(candidate_progress + 64.0 * np.finfo(float).eps < minimum):
            raise SeparationConstrainedError(
                "separation progress must remain irreversible"
            )
        correction_norm = float(np.linalg.norm(correction[:displacement_size]))
        combined = candidate
    raise SeparationConstrainedError(
        "separation augmented correction did not converge"
    )


def registered_zero_force_thresholds(
    *,
    residual_absolute_tolerance_N: float,
    displacement_absolute_tolerance_m: float,
    damage_event_tolerance: float,
    minimum_substep_fraction: float,
    reference_path_length_m: float,
    fracture_energy_J_per_m2: float,
    element_area_m2: float,
    thickness_m: float,
    characteristic_length_m: float,
) -> ZeroForceCommitThresholds:
    """Derive the frozen zero-force gates from existing inputs and precision."""

    values = tuple(
        float(value)
        for value in (
            residual_absolute_tolerance_N,
            displacement_absolute_tolerance_m,
            damage_event_tolerance,
            minimum_substep_fraction,
            reference_path_length_m,
            fracture_energy_J_per_m2,
            element_area_m2,
            thickness_m,
            characteristic_length_m,
        )
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("zero-force threshold scales must be finite and positive")
    prescribed_energy = (
        fracture_energy_J_per_m2
        * element_area_m2
        * thickness_m
        / characteristic_length_m
    )
    energy_absolute = max(
        1.0e-18, 64.0 * np.finfo(float).eps * prescribed_energy
    )
    return ZeroForceCommitThresholds(
        element_internal_force_absolute_N=residual_absolute_tolerance_N,
        projection_reaction_absolute_N=residual_absolute_tolerance_N,
        remaining_fracture_energy_absolute_J=energy_absolute,
        topology_energy_jump_absolute_J=energy_absolute,
        balance_residual_absolute_N=residual_absolute_tolerance_N,
        path_residual_absolute_m=max(
            1.0e-15,
            1.0e-6 * minimum_substep_fraction * reference_path_length_m,
        ),
        separation_residual_absolute=damage_event_tolerance,
        displacement_correction_absolute_m=displacement_absolute_tolerance_m,
    )


def qualifies_for_removed_commit(
    measures: ZeroForceCommitMeasures,
    thresholds: ZeroForceCommitThresholds,
) -> bool:
    """Require strict constitutive zero plus every registered numerical gate."""

    checks = (
        (measures.element_internal_force_norm_N, thresholds.element_internal_force_absolute_N),
        (measures.projection_reaction_norm_N, thresholds.projection_reaction_absolute_N),
        (measures.remaining_fracture_energy_J, thresholds.remaining_fracture_energy_absolute_J),
        (abs(measures.topology_energy_jump_J), thresholds.topology_energy_jump_absolute_J),
        (measures.balance_residual_N, thresholds.balance_residual_absolute_N),
        (abs(measures.path_residual_m), thresholds.path_residual_absolute_m),
        (measures.separation_residual, thresholds.separation_residual_absolute),
        (measures.displacement_correction_m, thresholds.displacement_correction_absolute_m),
    )
    return bool(
        measures.constitutive_capacity_is_exactly_zero
        and measures.damage_energy_fully_accounted
        and math.isfinite(measures.removed_dof_kinetic_energy_J)
        and measures.removed_dof_kinetic_energy_J >= 0.0
        and measures.removed_dof_kinetic_energy_J
        <= measures.kinetic_energy_absolute_tolerance_J
        and all(
            math.isfinite(value) and value >= 0.0 and value <= limit
            for value, limit in checks
        )
    )


__all__ = [
    "MaterialState",
    "SeparationEligibility",
    "SeparationConstrainedError",
    "SeparationConstrainedSettings",
    "SeparationConstrainedState",
    "SeparatingState",
    "SolveFailure",
    "SolveMode",
    "SolveRoute",
    "TerminalSupportProjection",
    "TerminalSupportProjectionError",
    "ZeroForceCommitMeasures",
    "ZeroForceCommitThresholds",
    "advance_separating",
    "begin_separating",
    "separation_constraint_terms",
    "can_enter_separating",
    "qualifies_for_removed_commit",
    "initialize_separation_path_direction",
    "registered_zero_force_thresholds",
    "route_after_failure",
    "separation_constrained_correct",
    "terminal_support_projection",
]
