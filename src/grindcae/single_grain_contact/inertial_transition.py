"""Real-mass transient primitives for the Schema 3 hybrid solver.

This module contains no damping, mass scaling, residual stiffness, topology
commit, or branch preference.  Histories beyond displacement and velocity are
owned by the caller and are committed only after a converged step is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable
import re

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True, slots=True)
class DynamicState:
    time_s: float
    displacement_m: NDArray[np.float64]
    velocity_m_per_s: NDArray[np.float64]

    def __post_init__(self) -> None:
        displacement = np.asarray(self.displacement_m, dtype=float)
        velocity = np.asarray(self.velocity_m_per_s, dtype=float)
        if (
            not math.isfinite(self.time_s)
            or self.time_s < 0.0
            or displacement.ndim != 1
            or velocity.shape != displacement.shape
            or not np.all(np.isfinite(displacement))
            or not np.all(np.isfinite(velocity))
        ):
            raise ValueError("invalid dynamic state")
        object.__setattr__(self, "displacement_m", displacement.copy())
        object.__setattr__(self, "velocity_m_per_s", velocity.copy())


@dataclass(frozen=True, slots=True)
class DynamicEnergyAudit:
    external_work_J: float
    kinetic_energy_change_J: float
    stored_energy_change_J: float
    energy_error_J: float
    residual_norm_N: float
    correction_norm_m: float
    iteration_count: int


@dataclass(frozen=True, slots=True)
class TopologyKineticFluxAudit:
    original_kinetic_energy_J: float
    retained_kinetic_energy_J: float
    exported_kinetic_energy_J: float
    original_momentum_kg_m_per_s: NDArray[np.float64]
    retained_momentum_kg_m_per_s: NDArray[np.float64]
    exported_momentum_kg_m_per_s: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FrictionlessImpactResult:
    velocity_after_m_per_s: NDArray[np.float64]
    impulses_N_s: NDArray[np.float64]
    gap_gradients: NDArray[np.float64]
    kinetic_energy_change_J: float
    boundary_work_J: float
    energy_error_J: float


@dataclass(frozen=True, slots=True)
class IdealEdgeContactEventStep:
    dynamic_state: DynamicState
    center_m: NDArray[np.float64]
    active_edges: NDArray[np.int32]
    normal_forces_N: NDArray[np.float64]
    body: object
    activated_edge: int | None
    impulse_N_s: float
    impact_work_J: float
    impact_energy_error_J: float


@dataclass(frozen=True, slots=True)
class IdealEdgeActiveSetUpdate:
    active_edges: NDArray[np.int32]
    normal_forces_N: NDArray[np.float64]
    entered_edges: tuple[int, ...]
    released_edges: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InertialTransitionState:
    """Committed state and physical ledger for one transient window."""

    committed_state: object
    dynamic_state: DynamicState
    center_m: NDArray[np.float64]
    grain_motion_direction: NDArray[np.float64]
    active_edges: NDArray[np.int32]
    normal_forces_N: NDArray[np.float64]
    constraint_edges: NDArray[np.int32]
    steps: tuple[object, ...] = ()
    cumulative_external_contact_work_J: float = 0.0
    cumulative_material_storage_change_J: float = 0.0
    cumulative_plastic_dissipation_J: float = 0.0
    cumulative_damage_dissipation_J: float = 0.0
    cumulative_energy_residual_J: float = 0.0
    terminal_damage_background_ids: tuple[int, ...] = ()
    cumulative_absolute_energy_error_J: float = 0.0
    maximum_energy_scale_J: float = 0.0


def build_inertial_transition_state(
    case: object,
    prepared: object,
    committed: object,
    *,
    center_m: object,
    grain_motion_direction: object,
    accepted_tangent: object,
    active_edges: object,
    normal_forces_N: object,
    constraint_edges: object,
) -> InertialTransitionState:
    """Create the first transient state from an accepted quasi-static tangent."""

    if case.transient is None:
        raise ValueError("Schema 3 transient controls are required")
    center = np.asarray(center_m, dtype=float)
    direction = np.asarray(grain_motion_direction, dtype=float)
    tangent = np.asarray(accepted_tangent, dtype=float)
    free = np.asarray(prepared.structural.free_dofs, dtype=np.int32)
    active = np.asarray(active_edges, dtype=np.int32)
    expected_tangent_size = free.size + active.size + 1
    if (
        center.shape != (2,)
        or direction.shape != (2,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(direction))
        or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1.0e-12)
        or tangent.shape != (expected_tangent_size,)
        or not np.all(np.isfinite(tangent))
        or abs(float(tangent[-1])) <= np.finfo(float).tiny
    ):
        raise ValueError("invalid inertial transition direction")
    velocity = np.zeros(prepared.total_degrees_of_freedom)
    velocity[free] = (
        tangent[: free.size] / tangent[-1] * float(case.transient.grain_speed_m_per_s)
    )
    dynamic = DynamicState(
        float(getattr(committed, "physical_time_s", 0.0)),
        np.asarray(committed.displacement_vector_m, dtype=float),
        velocity,
    )
    return InertialTransitionState(
        replace(
            committed,
            solve_mode=_inertial_solve_mode(),
            physical_time_s=dynamic.time_s,
            velocity_vector_m_per_s=velocity.copy(),
            kinetic_energy_J=_kinetic_energy(
                assemble_consistent_mass_matrix(
                    prepared.structural,
                    density_kg_per_m3=case.material.density_kg_per_m3,
                    thickness_m=case.analysis.thickness,
                ).toarray(),
                velocity,
            ),
        ),
        dynamic,
        center.copy(),
        direction.copy(),
        active.copy(),
        np.asarray(normal_forces_N, dtype=float).copy(),
        np.asarray(constraint_edges, dtype=np.int32).copy(),
    )


def _inertial_solve_mode():
    from .continuous_separation import SolveMode

    return SolveMode.INERTIAL_TRANSITION


def advance_inertial_transition_window(
    case: object,
    prepared: object,
    committed: object,
    transition: InertialTransitionState,
    *,
    maximum_steps: int,
    stop_at_terminal_damage: bool = False,
) -> InertialTransitionState:
    """Advance a bounded real-mass window and commit only converged midpoint steps."""

    if case.transient is None or type(maximum_steps) is not int or maximum_steps < 1:
        raise ValueError("invalid inertial transition window")
    state = transition
    accepted = transition.committed_state
    steps = list(transition.steps)
    external = transition.cumulative_external_contact_work_J
    storage = transition.cumulative_material_storage_change_J
    plastic = transition.cumulative_plastic_dissipation_J
    damage = transition.cumulative_damage_dissipation_J
    residual = transition.cumulative_energy_residual_J
    absolute_error = getattr(transition, 'cumulative_absolute_energy_error_J', 0.0)
    energy_scale = getattr(transition, 'maximum_energy_scale_J', 0.0)
    # Old diagnostic windows do not contain impulse-by-impulse absolute
    # accounting. Their signed residual cannot reconstruct that information.
    if transition.steps and not hasattr(transition, 'cumulative_absolute_energy_error_J'):
        raise ValueError('legacy ideal window lacks absolute energy history')
    if not all(math.isfinite(value) and value >= 0.0 for value in (absolute_error, energy_scale)):
        raise ValueError('invalid ideal window absolute energy history')
    if type(stop_at_terminal_damage) is not bool:
        raise ValueError("stop_at_terminal_damage must be a strict bool")

    def advance_once(
        source: InertialTransitionState, time_step_s: float
    ) -> IdealEdgeContactEventStep:
        return advance_ideal_edge_until_contact_event(
            case.to_elastoplastic_fem_case(),
            prepared,
            source.committed_state,
            case.damage,
            dynamic_state=source.dynamic_state,
            center_m=source.center_m,
            radius_m=case.grain.radius_m,
            grain_motion_direction=source.grain_motion_direction,
            grain_speed_m_per_s=case.transient.grain_speed_m_per_s,
            maximum_time_step_s=time_step_s,
            minimum_time_step_s=case.transient.minimum_time_step_s,
            active_edges=source.active_edges,
            normal_forces_N=source.normal_forces_N,
            constraint_edges=source.constraint_edges,
            density_kg_per_m3=case.material.density_kg_per_m3,
            gap_tolerance_m=1.0e-15,
        )

    def advance_with_registered_reduction(
        source: InertialTransitionState, time_step_s: float
    ) -> IdealEdgeContactEventStep:
        candidate = float(time_step_s)
        while True:
            try:
                event = advance_once(source, candidate)
                checked_energy(event)
                return event
            except RuntimeError as error:
                if ("ideal-edge inertial midpoint did not converge" not in str(error)
                        and 'ideal window physical energy budget' not in str(error)):
                    raise
                reduced = 0.5 * candidate
                if reduced < float(case.transient.minimum_time_step_s):
                    raise
                candidate = reduced

    def checked_energy(event):
        step = event.body
        terms = (step.external_contact_work_J, step.material_storage_change_J,
            step.plastic_dissipation_J, step.damage_dissipation_J,
            step.kinetic_energy_J, step.energy_residual_J,
            event.impact_work_J, event.impact_energy_error_J,
            getattr(step, 'maximum_energy_scale_J', 0.0))
        scale = max(energy_scale, *(abs(float(value)) for value in terms))
        error = absolute_error + abs(float(step.energy_residual_J)) + abs(float(event.impact_energy_error_J))
        if not all(math.isfinite(float(value)) for value in terms) or error > max(1e-10, 1e-5*scale):
            raise RuntimeError('ideal window physical energy budget exceeded')
        return error, scale

    for _ in range(maximum_steps):
        already_terminal = tuple(
            int(background)
            for background, damage_state in zip(
                accepted.background_element_ids, accepted.damage_states, strict=True
            )
            if damage_state.damage >= 1.0
        )
        if stop_at_terminal_damage and already_terminal:
            state = replace(state, terminal_damage_background_ids=already_terminal)
            break
        maximum_dt = float(case.transient.initial_time_step_s)
        candidate_dt = maximum_dt
        event = advance_with_registered_reduction(state, candidate_dt)
        terminal_ids = tuple(
            int(background)
            for background, damage_state in zip(
                accepted.background_element_ids,
                event.body.body.damage_states,
                strict=True,
            )
            if damage_state.damage >= 1.0
        )
        if stop_at_terminal_damage and terminal_ids:
            lower = 0.0
            upper = event.dynamic_state.time_s-state.dynamic_state.time_s
            located = event
            for _ in range(64):
                if upper - lower <= max(
                    np.finfo(float).eps * candidate_dt, case.transient.minimum_time_step_s
                ):
                    break
                midpoint = 0.5 * (lower + upper)
                if midpoint < case.transient.minimum_time_step_s:
                    break
                # Only a converged trial is evidence for either event bracket.
                trial = advance_once(state, midpoint)
                actual_time = trial.dynamic_state.time_s-state.dynamic_state.time_s
                if any(item.damage >= 1.0 for item in trial.body.body.damage_states):
                    upper = actual_time
                    located = trial
                else:
                    if actual_time <= lower:
                        raise RuntimeError('terminal locator made no accepted time progress')
                    lower = actual_time
            event = located
            terminal_ids = tuple(
                int(background)
                for background, damage_state in zip(
                    accepted.background_element_ids,
                    event.body.body.damage_states,
                    strict=True,
                )
                if damage_state.damage >= 1.0
            )
        step = event.body
        # Terminal localization replaces the initially audited trial. Audit
        # that exact event again before committing any history or topology.
        next_absolute_error, next_energy_scale = checked_energy(event)
        kinetic_energy = float(step.kinetic_energy_J)
        if event.activated_edge is not None:
            mass = assemble_consistent_mass_matrix(prepared.structural,
                density_kg_per_m3=case.material.density_kg_per_m3,
                thickness_m=case.analysis.thickness)
            velocity = event.dynamic_state.velocity_m_per_s
            kinetic_energy = .5*float(velocity@mass@velocity)
        accepted = replace(
            accepted,
            displacement_vector_m=np.asarray(step.displacement).copy(),
            element_states=step.body.element_states,
            damage_states=step.body.damage_states,
            grain_center_m=tuple(float(value) for value in event.center_m),
            grain_reference_m=tuple(float(value) for value in event.center_m),
            solve_mode=_inertial_solve_mode(),
            solve_failure=None,
            physical_time_s=event.dynamic_state.time_s,
            velocity_vector_m_per_s=event.dynamic_state.velocity_m_per_s.copy(),
            kinetic_energy_J=kinetic_energy,
        )
        steps.append(step)
        external += float(step.external_contact_work_J) + float(event.impact_work_J)
        storage += float(step.material_storage_change_J)
        plastic += float(step.plastic_dissipation_J)
        damage += float(step.damage_dissipation_J)
        residual += float(step.energy_residual_J) + float(event.impact_energy_error_J)
        absolute_error, energy_scale = next_absolute_error, next_energy_scale
        state = replace(
            state,
            committed_state=accepted,
            dynamic_state=event.dynamic_state,
            center_m=event.center_m.copy(),
            active_edges=event.active_edges.copy(),
            normal_forces_N=event.normal_forces_N.copy(),
            steps=tuple(steps),
            cumulative_external_contact_work_J=external,
            cumulative_material_storage_change_J=storage,
            cumulative_plastic_dissipation_J=plastic,
            cumulative_damage_dissipation_J=damage,
            cumulative_energy_residual_J=residual,
            terminal_damage_background_ids=terminal_ids,
            cumulative_absolute_energy_error_J=absolute_error,
            maximum_energy_scale_J=energy_scale,
        )
        if stop_at_terminal_damage and terminal_ids:
            break
    return state


def ideal_edge_active_state_from_history(
    *,
    edge_endpoint_keys: object,
    edge_contact_states: object,
    constraint_edge_keys: object,
    edge_lengths_m: object,
    thickness_m: float,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Convert stored endpoint pressures to total ideal-edge normal forces."""

    endpoint_keys = tuple(tuple(int(value) for value in key) for key in edge_endpoint_keys)
    history = dict(edge_contact_states)
    edges = tuple(tuple(int(value) for value in key) for key in constraint_edge_keys)
    lengths = np.asarray(edge_lengths_m, dtype=float)
    if (
        len(set(endpoint_keys)) != len(endpoint_keys)
        or set(history) != set(endpoint_keys)
        or lengths.shape != (len(edges),)
        or not np.all(np.isfinite(lengths))
        or np.any(lengths <= 0.0)
        or not math.isfinite(thickness_m)
        or thickness_m <= 0.0
    ):
        raise ValueError("invalid ideal-edge contact history")
    forces = []
    active = []
    for index, ((a, b), length) in enumerate(zip(edges, lengths, strict=True)):
        canonical = tuple(sorted((a, b)))
        pressures = [
            float(history[(canonical[0], canonical[1], node)].normal_multiplier_Pa)
            for node in canonical
        ]
        force = 0.5 * float(length) * thickness_m * math.fsum(pressures)
        if force > 0.0:
            active.append(index)
            forces.append(force)
    return np.asarray(active, dtype=np.int32), np.asarray(forces, dtype=float)


def locate_first_inactive_edge_violation(
    gap_function: Callable[[float], object],
    *,
    inactive_edges: object,
    maximum_time_step_s: float,
    gap_tolerance_m: float,
    return_admissible_side: bool = False,
) -> tuple[float, int] | None:
    """Locate the first transition from admissible gap to g < -tolerance.

    Diagnostic callers receive the violating bracket side by default. A
    physical event uses the admissible side of the same tight bracket so its
    next step satisfies the unchanged initial-geometry gate.
    """

    inactive = np.asarray(inactive_edges, dtype=np.int32)
    if (
        inactive.ndim != 1
        or np.unique(inactive).size != inactive.size
        or np.any(inactive < 0)
        or not math.isfinite(maximum_time_step_s)
        or maximum_time_step_s <= 0.0
        or not math.isfinite(gap_tolerance_m)
        or gap_tolerance_m < 0.0
    ):
        raise ValueError("invalid inactive-edge event search")
    if inactive.size == 0:
        return None

    def selected(time_s: float) -> NDArray[np.float64]:
        values = np.asarray(gap_function(float(time_s)), dtype=float)
        if values.ndim != 1 or np.any(inactive >= values.size) or not np.all(np.isfinite(values)):
            raise ValueError("invalid gap-function result")
        return values[inactive]

    start = selected(0.0)
    if np.any(start < -gap_tolerance_m):
        local = int(np.argmin(start))
        return 0.0, int(inactive[local])
    sample_count = 32
    previous_time = 0.0
    previous = start
    bracket = None
    for sample in range(1, sample_count + 1):
        current_time = maximum_time_step_s * sample / sample_count
        current = selected(current_time)
        crossed = np.flatnonzero(
            (previous >= -gap_tolerance_m) & (current < -gap_tolerance_m)
        )
        if crossed.size:
            bracket = (previous_time, current_time)
            break
        previous_time = current_time
        previous = current
    if bracket is None:
        return None
    lower, upper = bracket
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        # Bisect the union of violations. A secant estimate cannot order
        # nonlinear crossings of different edges inside one sample interval.
        if np.any(selected(midpoint) < -gap_tolerance_m):
            upper = midpoint
        else:
            lower = midpoint
        if upper - lower <= max(
            np.finfo(float).eps * maximum_time_step_s, 1.0e-18
        ):
            break
    first = np.flatnonzero(selected(upper) < -gap_tolerance_m)
    return (lower if return_admissible_side else upper), int(np.min(inactive[first]))


def retry_ideal_edge_inertial_step_after_release(
    solve: Callable[[float, NDArray[np.int32], NDArray[np.float64]], object],
    *,
    time_step_s: float,
    active_edges: object,
    normal_forces_N: object,
) -> tuple[object, NDArray[np.int32], NDArray[np.float64], tuple[int, ...]]:
    """Repeat one unchanged time interval after exact KKT multiplier releases."""

    active = np.asarray(active_edges, dtype=np.int32).copy()
    forces = np.asarray(normal_forces_N, dtype=float).copy()
    if active.ndim != 1 or forces.shape != active.shape:
        raise ValueError("invalid release retry state")
    released: list[int] = []
    while True:
        try:
            result = solve(float(time_step_s), active, forces)
            accepted_forces = np.asarray(
                getattr(result, "normal_forces_N", forces), dtype=float
            )
            if accepted_forces.shape != active.shape or not np.all(
                np.isfinite(accepted_forces)
            ):
                raise ValueError("invalid converged contact multipliers")
            return result, active, accepted_forces.copy(), tuple(released)
        except RuntimeError as error:
            match = re.fullmatch(
                r"ideal-edge inertial step requires contact release: ([0-9,]+)",
                str(error),
            )
            if match is None:
                raise
            identities = tuple(int(value) for value in match.group(1).split(","))
            local = [int(np.flatnonzero(active == identity)[0]) for identity in identities]
            keep = np.ones(active.size, dtype=bool)
            keep[local] = False
            active = active[keep]
            forces = forces[keep]
            released.extend(identities)


def update_ideal_edge_contact_active_set(
    *,
    gaps_m: object,
    gap_rates_m_per_s: object,
    active_edges: object,
    normal_forces_N: object,
    gap_tolerance_m: float,
    force_tolerance_N: float,
) -> IdealEdgeActiveSetUpdate:
    """Apply the normal-contact KKT entry and release rules without mutation."""

    gaps = np.asarray(gaps_m, dtype=float)
    rates = np.asarray(gap_rates_m_per_s, dtype=float)
    active = np.asarray(active_edges, dtype=np.int32)
    forces = np.asarray(normal_forces_N, dtype=float)
    if (
        gaps.ndim != 1
        or rates.shape != gaps.shape
        or active.ndim != 1
        or forces.shape != active.shape
        or np.unique(active).size != active.size
        or np.any(active < 0)
        or np.any(active >= gaps.size)
        or not all(np.all(np.isfinite(value)) for value in (gaps, rates, forces))
        or not math.isfinite(gap_tolerance_m)
        or gap_tolerance_m < 0.0
        or not math.isfinite(force_tolerance_N)
        or force_tolerance_N < 0.0
    ):
        raise ValueError("invalid ideal-edge active-set state")
    force_by_edge = {
        int(edge): float(force)
        for edge, force in zip(active, forces, strict=True)
    }
    released = tuple(
        int(edge)
        for edge, force in zip(active, forces, strict=True)
        if force <= force_tolerance_N and rates[int(edge)] > 0.0
    )
    retained = tuple(int(edge) for edge in active if int(edge) not in released)
    retained_set = set(retained)
    previously_active = set(int(edge) for edge in active)
    entered = tuple(
        int(edge)
        for edge in np.flatnonzero(gaps < -gap_tolerance_m)
        if int(edge) not in retained_set and int(edge) not in previously_active
    )
    updated = np.asarray(sorted((*retained, *entered)), dtype=np.int32)
    updated_forces = np.asarray(
        [force_by_edge.get(int(edge), 0.0) for edge in updated], dtype=float
    )
    return IdealEdgeActiveSetUpdate(updated, updated_forces, entered, released)


def _coalesce_identical_ideal_edge_constraints(
    active_edges: object,
    normal_forces_N: object,
    *,
    all_gaps_m: object,
    all_gap_displacement: object,
) -> tuple[NDArray[np.int32], NDArray[np.float64], tuple[int, ...]]:
    """Merge exact duplicate algebraic constraints at a shared edge endpoint.

    Adjacent background edges can have the same closest point at their common
    node.  They then impose one identical gap equation, so retaining both rows
    makes the multiplier representation non-unique.  The total normal force is
    assigned to the lowest stable edge identity; geometrically distinct rows
    are never merged.
    """

    active = np.asarray(active_edges, dtype=np.int32)
    forces = np.asarray(normal_forces_N, dtype=float)
    gaps = np.asarray(all_gaps_m, dtype=float)
    gradients = np.asarray(all_gap_displacement, dtype=float)
    if (
        active.ndim != 1
        or forces.shape != active.shape
        or gaps.ndim != 1
        or gradients.ndim != 2
        or gradients.shape[0] != gaps.size
        or np.any(active < 0)
        or np.any(active >= gaps.size)
    ):
        raise ValueError("invalid ideal-edge constraint coalescence state")
    retained: list[int] = []
    retained_forces: list[float] = []
    redundant: list[int] = []
    for edge, force in sorted(zip(active.tolist(), forces.tolist(), strict=True)):
        duplicate = next(
            (
                index
                for index, retained_edge in enumerate(retained)
                if gaps[edge] == gaps[retained_edge]
                and np.array_equal(gradients[edge], gradients[retained_edge])
            ),
            None,
        )
        if duplicate is None:
            retained.append(int(edge))
            retained_forces.append(float(force))
        else:
            retained_forces[duplicate] += float(force)
            redundant.append(int(edge))
    return (
        np.asarray(retained, dtype=np.int32),
        np.asarray(retained_forces, dtype=float),
        tuple(redundant),
    )


def triangle_consistent_mass_matrix(
    coordinates_m: object, *, density_kg_per_m3: float, thickness_m: float
) -> NDArray[np.float64]:
    """Return the 6x6 consistent translational mass of one linear triangle."""

    coordinates = np.asarray(coordinates_m, dtype=float)
    if (
        coordinates.shape != (3, 2)
        or not np.all(np.isfinite(coordinates))
        or not math.isfinite(density_kg_per_m3)
        or density_kg_per_m3 <= 0.0
        or not math.isfinite(thickness_m)
        or thickness_m <= 0.0
    ):
        raise ValueError("finite triangle coordinates, density, and thickness required")
    first = coordinates[1] - coordinates[0]
    second = coordinates[2] - coordinates[0]
    signed_twice_area = float(first[0] * second[1] - first[1] * second[0])
    area = 0.5 * abs(signed_twice_area)
    if area <= 0.0:
        raise ValueError("triangle area must be positive")
    scalar = density_kg_per_m3 * thickness_m * area / 12.0
    return scalar * np.kron(
        np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]),
        np.eye(2),
    )


def assemble_consistent_mass_matrix(
    prepared_structural: object,
    *,
    density_kg_per_m3: float,
    thickness_m: float,
) -> csr_matrix:
    """Assemble the physical consistent mass on the current active mesh."""

    total = int(prepared_structural.total_degrees_of_freedom)
    connectivity = np.asarray(prepared_structural.element_connectivity)
    element_dofs = np.asarray(prepared_structural.element_dofs)
    coordinates = np.asarray(
        prepared_structural.imported_mesh.mesh.p.T, dtype=float
    )
    if connectivity.shape != (len(element_dofs), 3) or element_dofs.shape[1:] != (6,):
        raise ValueError("prepared structural triangle mapping is invalid")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for nodes, dofs in zip(connectivity, element_dofs, strict=True):
        local = triangle_consistent_mass_matrix(
            coordinates[nodes],
            density_kg_per_m3=density_kg_per_m3,
            thickness_m=thickness_m,
        )
        rows.extend(np.repeat(dofs, 6).tolist())
        columns.extend(np.tile(dofs, 6).tolist())
        values.extend(local.reshape(-1).tolist())
    mass = coo_matrix((values, (rows, columns)), shape=(total, total)).tocsr()
    if not np.all(np.isfinite(mass.data)):
        raise ValueError("assembled mass matrix is non-finite")
    return mass


def material_kinetic_energy(prepared, state, background_element_ids, *, density_kg_per_m3, thickness_m):
    """Integrate selected material mass, including mass on shared nodes."""
    selected = set(int(value) for value in background_element_ids)
    available = set(int(value) for value in prepared.active_to_background_element_ids)
    if not selected <= available:
        raise ValueError('kinetic audit references absent material')
    velocity = np.asarray(getattr(state, 'velocity_vector_m_per_s', ()), dtype=float)
    if velocity.size == 0:
        if float(getattr(state, 'kinetic_energy_J', 0.0)) != 0.0:
            raise ValueError('kinetic energy requires velocity history')
        return 0.0
    if velocity.shape != (prepared.total_degrees_of_freedom,) or not np.all(np.isfinite(velocity)):
        raise ValueError('invalid velocity for material mass audit')
    energies = []
    for local, background in enumerate(prepared.active_to_background_element_ids):
        if int(background) not in selected:
            continue
        nodes = prepared.structural.element_connectivity[local]
        dofs = prepared.structural.element_dofs[local]
        mass = triangle_consistent_mass_matrix(
            prepared.structural.imported_mesh.mesh.p.T[nodes],
            density_kg_per_m3=density_kg_per_m3, thickness_m=thickness_m)
        energies.append(0.5 * float(velocity[dofs] @ mass @ velocity[dofs]))
    return math.fsum(energies)


def _physical_linear_momentum(
    mass: NDArray[np.float64], component_dofs: NDArray[np.int32], velocity: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(
        [float(np.sum((mass @ velocity)[component_dofs[component]])) for component in (0, 1)]
    )


def rebase_dynamic_state_after_topology(
    committed: DynamicState,
    old_prepared: object,
    new_prepared: object,
    *,
    density_kg_per_m3: float,
    thickness_m: float,
) -> tuple[DynamicState, TopologyKineticFluxAudit]:
    """Map retained background nodes and audit mass/momentum leaving the domain.

    Exported quantities are boundary fluxes owned by removed material; they are
    not dissipation.  This function does not decide whether removal is legal.
    """

    old_mass = assemble_consistent_mass_matrix(
        old_prepared.structural,
        density_kg_per_m3=density_kg_per_m3,
        thickness_m=thickness_m,
    ).toarray()
    new_mass = assemble_consistent_mass_matrix(
        new_prepared.structural,
        density_kg_per_m3=density_kg_per_m3,
        thickness_m=thickness_m,
    ).toarray()
    if committed.displacement_m.shape != (old_prepared.total_degrees_of_freedom,):
        raise ValueError("dynamic state does not match old topology")
    old_lookup = {
        int(background): active
        for active, background in enumerate(old_prepared.active_to_background_node_ids)
    }
    displacement = np.zeros(new_prepared.total_degrees_of_freedom)
    velocity = np.zeros(new_prepared.total_degrees_of_freedom)
    for new_node, background in enumerate(new_prepared.active_to_background_node_ids):
        try:
            old_node = old_lookup[int(background)]
        except KeyError as exc:
            raise ValueError("new topology contains an unmapped background node") from exc
        for component in (0, 1):
            old_dof = old_prepared.structural.component_dofs[component, old_node]
            new_dof = new_prepared.structural.component_dofs[component, new_node]
            displacement[new_dof] = committed.displacement_m[old_dof]
            velocity[new_dof] = committed.velocity_m_per_s[old_dof]
    velocity[new_prepared.structural.fixed_dofs] = 0.0
    displacement[new_prepared.structural.fixed_dofs] = 0.0
    original_energy = _kinetic_energy(old_mass, committed.velocity_m_per_s)
    retained_energy = _kinetic_energy(new_mass, velocity)
    original_momentum = _physical_linear_momentum(
        old_mass, old_prepared.structural.component_dofs, committed.velocity_m_per_s
    )
    retained_momentum = _physical_linear_momentum(
        new_mass, new_prepared.structural.component_dofs, velocity
    )
    energy_tolerance = 64.0 * np.finfo(float).eps * max(original_energy, retained_energy, 1.0)
    if retained_energy > original_energy + energy_tolerance:
        raise ValueError("topology mapping created kinetic energy")
    rebased = DynamicState(committed.time_s, displacement, velocity)
    audit = TopologyKineticFluxAudit(
        original_energy,
        retained_energy,
        max(0.0, original_energy - retained_energy),
        original_momentum,
        retained_momentum,
        original_momentum - retained_momentum,
    )
    return rebased, audit


def resolve_frictionless_elastic_impact(
    mass_matrix_kg: object,
    velocity_before_m_per_s: object,
    *,
    gap_gradients: object,
    free_dofs: object,
    prescribed_gap_velocity_m_per_s: object | None = None,
) -> FrictionlessImpactResult:
    """Apply the energy-conserving zero-restitution-loss normal impulse.

    The active gradients must be independent.  This function neither merges
    contact identities nor invents tangential or material dissipation.
    """

    mass = np.asarray(mass_matrix_kg, dtype=float)
    velocity = np.asarray(velocity_before_m_per_s, dtype=float)
    gradients = np.asarray(gap_gradients, dtype=float)
    free_raw = np.asarray(free_dofs)
    count = velocity.size
    if (
        velocity.ndim != 1
        or mass.shape != (count, count)
        or gradients.ndim != 2
        or gradients.shape[1] != count
        or gradients.shape[0] < 1
        or free_raw.ndim != 1
        or not np.issubdtype(free_raw.dtype, np.integer)
        or not all(np.all(np.isfinite(value)) for value in (mass, velocity, gradients))
    ):
        raise ValueError("invalid impact system")
    free = np.asarray(free_raw, dtype=np.int32)
    prescribed = (
        np.zeros(gradients.shape[0])
        if prescribed_gap_velocity_m_per_s is None
        else np.asarray(prescribed_gap_velocity_m_per_s, dtype=float)
    )
    if prescribed.shape != (gradients.shape[0],) or not np.all(np.isfinite(prescribed)):
        raise ValueError("invalid prescribed gap velocity")
    mass_free = mass[np.ix_(free, free)]
    gradients_free = gradients[:, free]
    if np.linalg.matrix_rank(gradients_free) != gradients.shape[0]:
        raise RuntimeError("impact constraint gradients are rank deficient")
    try:
        inverse_action = np.linalg.solve(mass_free, gradients_free.T)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("impact mass solve is rank deficient") from exc
    effective = gradients_free @ inverse_action
    if np.linalg.matrix_rank(effective) != gradients.shape[0]:
        raise RuntimeError("impact effective mass is rank deficient")
    normal_velocity = gradients @ velocity + prescribed
    try:
        impulses = np.linalg.solve(effective, -2.0 * normal_velocity)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded above
        raise RuntimeError("impact impulse solve is rank deficient") from exc
    if np.any(impulses < -64.0 * np.finfo(float).eps * max(np.linalg.norm(impulses), 1.0)):
        raise RuntimeError("impact active set requires a negative normal impulse")
    impulses = np.maximum(impulses, 0.0)
    velocity_after = velocity.copy()
    velocity_after[free] += inverse_action @ impulses
    fixed = np.setdiff1d(np.arange(count), free)
    velocity_after[fixed] = 0.0
    energy_change = _kinetic_energy(mass, velocity_after) - _kinetic_energy(mass, velocity)
    boundary_work = float(-impulses @ prescribed)
    return FrictionlessImpactResult(
        velocity_after, impulses, gradients.copy(), float(energy_change),
        boundary_work, float(energy_change - boundary_work)
    )


def advance_ideal_edge_until_contact_event(
    case: object,
    prepared: object,
    committed: object,
    damage_input: object,
    *,
    dynamic_state: DynamicState,
    center_m: object,
    radius_m: float,
    grain_motion_direction: object,
    grain_speed_m_per_s: float,
    maximum_time_step_s: float,
    active_edges: object,
    normal_forces_N: object,
    constraint_edges: object,
    density_kg_per_m3: float,
    gap_tolerance_m: float,
    minimum_time_step_s: float | None = None,
) -> IdealEdgeContactEventStep:
    """Advance or stop at the earliest inactive-edge impact and apply its impulse."""

    from .normal_constraint_path import (
        advance_ideal_edge_inertial_step,
        circle_edge_normal_constraints,
    )

    center0 = np.asarray(center_m, dtype=float)
    minimum_dt = float(minimum_time_step_s if minimum_time_step_s is not None else
        getattr(getattr(case,'transient',None),'minimum_time_step_s',maximum_time_step_s))
    if not math.isfinite(minimum_dt) or minimum_dt<=0. or minimum_dt>maximum_time_step_s:
        raise ValueError('invalid physical minimum time step')
    direction = np.asarray(grain_motion_direction, dtype=float)
    edges = np.asarray(constraint_edges, dtype=int)
    active = np.asarray(active_edges, dtype=np.int32)
    forces = np.asarray(normal_forces_N, dtype=float)
    if forces.shape != active.shape:
        raise ValueError("active edge force shape is invalid")
    geometry = circle_edge_normal_constraints(
        prepared.candidate_reference_coordinates_m,
        prepared.candidate_component_dofs,
        edges,
        dynamic_state.displacement_m,
        center0,
        radius_m,
        active,
        forces,
        direction,
    )
    all_gap_rates = np.empty(len(edges), dtype=float)
    all_geometry = circle_edge_normal_constraints(
        prepared.candidate_reference_coordinates_m,
        prepared.candidate_component_dofs,
        edges,
        dynamic_state.displacement_m,
        center0,
        radius_m,
        np.arange(len(edges), dtype=np.int32),
        np.zeros(len(edges)),
        direction,
    )
    all_gap_rates[:] = (
        all_geometry.gap_displacement @ dynamic_state.velocity_m_per_s
        + all_geometry.gap_marker * grain_speed_m_per_s
    )
    if np.any(all_geometry.all_gaps < -gap_tolerance_m):
        raise ValueError('ideal transition initial geometry already violates penetration tolerance')
    active_update = update_ideal_edge_contact_active_set(
        gaps_m=all_geometry.all_gaps,
        gap_rates_m_per_s=all_gap_rates,
        active_edges=active,
        normal_forces_N=forces,
        gap_tolerance_m=gap_tolerance_m,
        force_tolerance_N=case.newton.residual_absolute_tolerance_N,
    )
    active = active_update.active_edges
    forces = active_update.normal_forces_N
    active, forces, redundant_edges = _coalesce_identical_ideal_edge_constraints(
        active,
        forces,
        all_gaps_m=all_geometry.all_gaps,
        all_gap_displacement=all_geometry.gap_displacement,
    )
    active_at_step_start = active.copy()
    forces_at_step_start = forces.copy()
    all_ids = np.arange(len(edges), dtype=np.int32)
    inactive = np.setdiff1d(
        all_ids,
        np.asarray(
            sorted(active.tolist()),
            dtype=np.int32,
        ),
    )
    # Algebraic equality at the starting endpoint does not persist through
    # motion. Keep coalesced background edges in the geometric event scan so
    # a closest-point split cannot hide penetration of the omitted edge.

    def solve_with_active(
        dt: float,
        step_active: NDArray[np.int32],
        step_forces: NDArray[np.float64],
    ):
        return advance_ideal_edge_inertial_step(
            case,
            prepared,
            committed,
            damage_input,
            displacement_velocity_m_per_s=dynamic_state.velocity_m_per_s,
            center_m=center0,
            radius_m=radius_m,
            grain_motion_direction=direction,
            grain_speed_m_per_s=grain_speed_m_per_s,
            time_step_s=dt,
            active_edges=step_active,
            normal_forces_N=step_forces,
            constraint_edges=edges,
            density_kg_per_m3=density_kg_per_m3,
        )

    initial_active = active_at_step_start
    initial_forces = forces_at_step_start
    released_during_solve: set[int] = set()

    def solve(dt: float):
        result, solved_active, solved_forces, released = retry_ideal_edge_inertial_step_after_release(
            solve_with_active,
            time_step_s=dt,
            active_edges=initial_active,
            normal_forces_N=initial_forces,
        )
        released_during_solve.update(released)
        return result, solved_active, solved_forces

    full_dt = float(maximum_time_step_s)
    while True:
        try:
            full, full_active, full_forces = solve(full_dt)
            full_active = np.asarray(full_active, dtype=np.int32).copy()
            full_forces = np.asarray(full_forces, dtype=float).copy()
            initial_gaps = all_geometry.all_gaps.copy()
            scan_edges=set(int(edge) for edge in inactive)|released_during_solve
            while True:
                event = locate_first_inactive_edge_violation(
                    lambda dt: (
                        initial_gaps if dt == 0.0
                        else full.contact.all_gaps if dt == full_dt
                        else solve(dt)[0].contact.all_gaps),
                    inactive_edges=np.asarray(sorted(scan_edges),dtype=np.int32),
                    maximum_time_step_s=full_dt,gap_tolerance_m=gap_tolerance_m,
                    return_admissible_side=True)
                # A multiplier release can first appear at an intermediate
                # search time. Rescan from the same history until every such
                # background edge has been included; no trial is committed.
                additional=released_during_solve-scan_edges
                if not additional:break
                scan_edges.update(additional)
            break
        except RuntimeError as error:
            if "ideal-edge inertial midpoint did not converge" not in str(error):
                raise
            reduced = 0.5 * full_dt
            if reduced < minimum_dt:
                raise
            full_dt = reduced
    if event is None:
        return IdealEdgeContactEventStep(
            DynamicState(dynamic_state.time_s + full_dt,
                full.displacement, full.velocity_m_per_s),
            center0 + full_dt * grain_speed_m_per_s * direction,
            full_active, full_forces, full, None, 0.0, 0.0, 0.0,
        )
    upper, event_edge = event
    if upper <= max(np.finfo(float).eps * full_dt, 1.0e-18):
        gradient = np.zeros((1, prepared.total_degrees_of_freedom))
        gradient[0] = all_geometry.gap_displacement[event_edge]
        mass = assemble_consistent_mass_matrix(
            prepared.structural,
            density_kg_per_m3=density_kg_per_m3,
            thickness_m=case.analysis.thickness,
        ).toarray()
        impact = resolve_frictionless_elastic_impact(
            mass,
            dynamic_state.velocity_m_per_s,
            gap_gradients=gradient,
            free_dofs=prepared.structural.free_dofs,
            prescribed_gap_velocity_m_per_s=np.array([
                all_geometry.gap_marker[event_edge] * grain_speed_m_per_s
            ]),
        )
        new_active = np.asarray(
            sorted((*initial_active.tolist(), event_edge)), dtype=np.int32
        )
        force_by_edge = dict(
            zip(initial_active.tolist(), initial_forces.tolist(), strict=True)
        )
        force_by_edge[event_edge] = 0.0
        new_forces = np.asarray([force_by_edge[int(index)] for index in new_active])
        post_rates = (
            all_geometry.gap_displacement @ impact.velocity_after_m_per_s
            + all_geometry.gap_marker * grain_speed_m_per_s
        )
        post_update = update_ideal_edge_contact_active_set(
            gaps_m=all_geometry.all_gaps,
            gap_rates_m_per_s=post_rates,
            active_edges=new_active,
            normal_forces_N=new_forces,
            gap_tolerance_m=gap_tolerance_m,
            force_tolerance_N=case.newton.residual_absolute_tolerance_N,
        )
        # This event has no elapsed time. The full trial was only a search
        # probe: none of its displacement, material evolution or work belongs
        # to the instantaneous event.
        from .solver import _assemble_damage_trial
        from .normal_constraint_path import IdealEdgeInertialIncrement
        body0 = _assemble_damage_trial(case, prepared, committed,
            dynamic_state.displacement_m, damage_input)
        body0 = replace(body0, element_states=committed.element_states,
            damage_states=committed.damage_states)
        zero_step = IdealEdgeInertialIncrement(
            time_s=0., grain_increment_m=0., displacement=dynamic_state.displacement_m.copy(),
            velocity_m_per_s=dynamic_state.velocity_m_per_s.copy(), normal_forces_N=initial_forces.copy(),
            body=body0, contact=geometry, residual_norm_N=0.,
            residual_tolerance_N=case.newton.residual_absolute_tolerance_N,
            correction_norm_m=0., kinetic_energy_J=.5*float(
                dynamic_state.velocity_m_per_s@mass@dynamic_state.velocity_m_per_s),
            external_contact_work_J=0., material_storage_change_J=0.,
            plastic_dissipation_J=0., damage_dissipation_J=0., energy_residual_J=0.,
            maximum_energy_scale_J=abs(impact.boundary_work_J))
        return IdealEdgeContactEventStep(
            DynamicState(
                dynamic_state.time_s,
                dynamic_state.displacement_m.copy(),
                impact.velocity_after_m_per_s,
            ),
            center0.copy(),
            post_update.active_edges,
            post_update.normal_forces_N,
            zero_step,
            event_edge,
            float(impact.impulses_N_s[0]),
            impact.boundary_work_J,
            impact.energy_error_J,
        )
    located, active, forces = solve(upper)
    event_center = center0 + upper * grain_speed_m_per_s * direction
    geometry = circle_edge_normal_constraints(
        prepared.candidate_reference_coordinates_m,
        prepared.candidate_component_dofs,
        edges,
        located.displacement,
        event_center,
        radius_m,
        np.array([event_edge]),
        np.zeros(1),
        direction,
    )
    gradient = np.zeros((1, prepared.total_degrees_of_freedom))
    gradient[0] = geometry.gap_displacement[0]
    mass = assemble_consistent_mass_matrix(
        prepared.structural,
        density_kg_per_m3=density_kg_per_m3,
        thickness_m=case.analysis.thickness,
    ).toarray()
    impact = resolve_frictionless_elastic_impact(
        mass,
        located.velocity_m_per_s,
        gap_gradients=gradient,
        free_dofs=prepared.structural.free_dofs,
        prescribed_gap_velocity_m_per_s=geometry.gap_marker * grain_speed_m_per_s,
    )
    new_active = np.asarray(sorted((*active.tolist(), event_edge)), dtype=np.int32)
    force_by_edge = dict(zip(active.tolist(), forces.tolist(), strict=True))
    force_by_edge[event_edge] = 0.0
    new_forces = np.asarray([force_by_edge[int(index)] for index in new_active])
    post_geometry = circle_edge_normal_constraints(
        prepared.candidate_reference_coordinates_m,
        prepared.candidate_component_dofs,
        edges,
        located.displacement,
        event_center,
        radius_m,
        new_active,
        new_forces,
        direction,
    )
    post_rates = np.empty(len(edges), dtype=float)
    post_all = circle_edge_normal_constraints(
        prepared.candidate_reference_coordinates_m,
        prepared.candidate_component_dofs,
        edges,
        located.displacement,
        event_center,
        radius_m,
        np.arange(len(edges), dtype=np.int32),
        np.zeros(len(edges)),
        direction,
    )
    post_rates[:] = (
        post_all.gap_displacement @ impact.velocity_after_m_per_s
        + post_all.gap_marker * grain_speed_m_per_s
    )
    post_update = update_ideal_edge_contact_active_set(
        gaps_m=post_all.all_gaps,
        gap_rates_m_per_s=post_rates,
        active_edges=new_active,
        normal_forces_N=new_forces,
        gap_tolerance_m=gap_tolerance_m,
        force_tolerance_N=case.newton.residual_absolute_tolerance_N,
    )
    return IdealEdgeContactEventStep(
        DynamicState(dynamic_state.time_s + upper, located.displacement,
            impact.velocity_after_m_per_s),
        event_center,
        post_update.active_edges,
        post_update.normal_forces_N,
        located,
        event_edge,
        float(impact.impulses_N_s[0]),
        impact.boundary_work_J,
        impact.energy_error_J,
    )


def _kinetic_energy(mass: NDArray[np.float64], velocity: NDArray[np.float64]) -> float:
    return 0.5 * float(velocity @ mass @ velocity)


def advance_implicit_midpoint(
    committed: DynamicState,
    *,
    time_step_s: float,
    mass_matrix_kg: object,
    internal_force_and_tangent: Callable[
        [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
    external_force_N: object,
    free_dofs: object,
    stored_energy_J: Callable[[NDArray[np.float64]], float],
    residual_absolute_tolerance_N: float,
    displacement_absolute_tolerance_m: float,
    maximum_iterations: int,
) -> tuple[DynamicState, DynamicEnergyAudit]:
    """Advance one undamped implicit-midpoint step without mutating input state."""

    displacement0 = committed.displacement_m
    velocity0 = committed.velocity_m_per_s
    count = displacement0.size
    mass = np.asarray(mass_matrix_kg, dtype=float)
    external = np.asarray(external_force_N, dtype=float)
    free_raw = np.asarray(free_dofs)
    if (
        not math.isfinite(time_step_s)
        or time_step_s <= 0.0
        or mass.shape != (count, count)
        or external.shape != (count,)
        or free_raw.ndim != 1
        or not np.issubdtype(free_raw.dtype, np.integer)
        or not np.all(np.isfinite(mass))
        or not np.all(np.isfinite(external))
        or not math.isfinite(residual_absolute_tolerance_N)
        or residual_absolute_tolerance_N <= 0.0
        or not math.isfinite(displacement_absolute_tolerance_m)
        or displacement_absolute_tolerance_m <= 0.0
        or type(maximum_iterations) is not int
        or maximum_iterations < 0
    ):
        raise ValueError("invalid implicit-midpoint controls")
    free = np.asarray(free_raw, dtype=np.int32)
    if free.size == 0 or np.unique(free).size != free.size or np.any(free < 0) or np.any(free >= count):
        raise ValueError("free_dofs must contain unique in-range indices")
    mass_free = mass[np.ix_(free, free)]
    if np.min(np.linalg.eigvalsh(0.5 * (mass_free + mass_free.T))) <= 0.0:
        raise ValueError("free mass matrix must be symmetric positive definite")

    displacement1 = displacement0 + time_step_s * velocity0
    correction_norm = 0.0
    residual_norm = math.inf
    iteration = 0
    for iteration in range(maximum_iterations + 1):
        midpoint = 0.5 * (displacement0 + displacement1)
        velocity1 = 2.0 * (displacement1 - displacement0) / time_step_s - velocity0
        internal, tangent = internal_force_and_tangent(midpoint.copy())
        internal = np.asarray(internal, dtype=float)
        tangent = np.asarray(tangent, dtype=float)
        if internal.shape != (count,) or tangent.shape != (count, count):
            raise ValueError("internal force or tangent shape is invalid")
        residual = mass @ ((velocity1 - velocity0) / time_step_s) + internal - external
        residual_norm = float(np.linalg.norm(residual[free]))
        if residual_norm <= residual_absolute_tolerance_N and correction_norm <= displacement_absolute_tolerance_m:
            break
        if iteration >= maximum_iterations:
            raise RuntimeError("implicit midpoint did not converge")
        jacobian = 2.0 * mass / time_step_s**2 + 0.5 * tangent
        try:
            correction_free = np.linalg.solve(
                jacobian[np.ix_(free, free)], -residual[free]
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("implicit midpoint tangent solve failed") from exc
        if not np.all(np.isfinite(correction_free)):
            raise RuntimeError("implicit midpoint correction is non-finite")
        correction_norm = float(np.linalg.norm(correction_free))
        displacement1 = displacement1.copy()
        displacement1[free] += correction_free
        fixed = np.setdiff1d(np.arange(count), free)
        displacement1[fixed] = displacement0[fixed]
    else:  # pragma: no cover
        raise AssertionError("unreachable midpoint loop exit")

    velocity1 = 2.0 * (displacement1 - displacement0) / time_step_s - velocity0
    fixed = np.setdiff1d(np.arange(count), free)
    velocity1[fixed] = 0.0
    external_work = float(external @ (displacement1 - displacement0))
    kinetic_change = _kinetic_energy(mass, velocity1) - _kinetic_energy(mass, velocity0)
    stored_change = float(stored_energy_J(displacement1)) - float(stored_energy_J(displacement0))
    values = (external_work, kinetic_change, stored_change)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dynamic energy audit is non-finite")
    state = DynamicState(
        committed.time_s + time_step_s, displacement1, velocity1
    )
    audit = DynamicEnergyAudit(
        external_work,
        kinetic_change,
        stored_change,
        kinetic_change + stored_change - external_work,
        residual_norm,
        correction_norm,
        iteration,
    )
    return state, audit


__all__ = [
    "DynamicEnergyAudit",
    "DynamicState",
    "FrictionlessImpactResult",
    "IdealEdgeContactEventStep",
    "IdealEdgeActiveSetUpdate",
    "InertialTransitionState",
    "build_inertial_transition_state",
    "advance_inertial_transition_window",
    "ideal_edge_active_state_from_history",
    "locate_first_inactive_edge_violation",
    "retry_ideal_edge_inertial_step_after_release",
    "advance_ideal_edge_until_contact_event",
    "TopologyKineticFluxAudit",
    "advance_implicit_midpoint",
    "assemble_consistent_mass_matrix",
    "rebase_dynamic_state_after_topology",
    "resolve_frictionless_elastic_impact",
    "triangle_consistent_mass_matrix",
    "update_ideal_edge_contact_active_set",
]
