"""Deterministic location and grouping of damage separation events."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import TypeVar

from .damage_material import DamageState


class SeparationEventError(RuntimeError):
    """Raised when a separation event cannot be located or accepted."""


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LocatedSeparationEvent:
    fraction: float
    background_element_ids: tuple[int, ...]
    maximum_damage: float
    solved_state: object


@dataclass(frozen=True, slots=True)
class EventAwareSubstepProposal:
    scale: float
    event_kind: str
    background_element_ids: tuple[int, ...]


def event_aware_substep_scale(
    *,
    background_element_ids: Sequence[int],
    committed_damage_states: Sequence[DamageState],
    trial_damage_states: Sequence[DamageState],
    damage_event_tolerance: float,
    fracture_energy_J_per_m2: float,
) -> EventAwareSubstepProposal | None:
    """Return a conservative scale to the earliest material event lower side."""

    if not (
        len(background_element_ids)
        == len(committed_damage_states)
        == len(trial_damage_states)
    ):
        raise SeparationEventError("damage states do not match background ids")
    tolerance = float(damage_event_tolerance)
    fracture_energy = float(fracture_energy_J_per_m2)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise SeparationEventError("damage event tolerance must be positive")
    if not math.isfinite(fracture_energy) or fracture_energy <= 0.0:
        raise SeparationEventError("fracture energy must be positive")

    candidates: list[tuple[float, str, int]] = []
    for background_id, committed, trial in zip(
        background_element_ids,
        committed_damage_states,
        trial_damage_states,
        strict=True,
    ):
        if committed.omega < 1.0 <= trial.omega and trial.omega > committed.omega:
            event_fraction = (1.0 - committed.omega) / (
                trial.omega - committed.omega
            )
            candidates.append(
                (event_fraction, "damage_initiation", int(background_id))
            )
        initiation_stress = (
            committed.initiation_equivalent_stress_Pa
            if committed.initiation_equivalent_stress_Pa is not None
            else trial.initiation_equivalent_stress_Pa
        )
        if initiation_stress is None or initiation_stress <= 0.0:
            continue
        terminal_displacement = 2.0 * fracture_energy / initiation_stress
        lower = committed.post_initiation_plastic_displacement_m
        upper = trial.post_initiation_plastic_displacement_m
        if lower < terminal_displacement <= upper and upper > lower:
            event_fraction = (terminal_displacement - lower) / (upper - lower)
            candidates.append(
                (event_fraction, "damage_separation", int(background_id))
            )
    if not candidates:
        return None
    earliest = min(item[0] for item in candidates)
    event_kind = min(
        item[1] for item in candidates if abs(item[0] - earliest) <= tolerance
    )
    ids = tuple(
        sorted(
            item[2]
            for item in candidates
            if item[1] == event_kind and abs(item[0] - earliest) <= tolerance
        )
    )
    lower_side_scale = max(0.0, min(0.5, earliest * (1.0 - 0.1)))
    return EventAwareSubstepProposal(
        scale=lower_side_scale,
        event_kind=event_kind,
        background_element_ids=ids,
    )


def event_candidate_background_ids(
    *,
    background_element_ids: Sequence[int],
    damage_states: Sequence[DamageState],
    damage_event_tolerance: float,
    accepted_events_at_position: int = 0,
    maximum_events_per_position: int | None = None,
) -> tuple[int, ...]:
    tolerance = float(damage_event_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise SeparationEventError("damage event tolerance must be positive")
    if len(background_element_ids) != len(damage_states):
        raise SeparationEventError("damage states do not match background ids")
    if (
        maximum_events_per_position is not None
        and accepted_events_at_position >= maximum_events_per_position
    ):
        raise SeparationEventError("maximum separation events per position exceeded")
    return tuple(
        sorted(
            int(background_id)
            for background_id, state in zip(
                background_element_ids, damage_states, strict=True
            )
            if state.damage >= 1.0 - tolerance
        )
    )


def locate_earliest_separation_event(
    *,
    lower_fraction: float,
    upper_fraction: float,
    solve_at_fraction: Callable[[float], tuple[T, Sequence[int], Sequence[DamageState]]],
    damage_event_tolerance: float,
    fraction_tolerance: float = 1.0e-6,
) -> LocatedSeparationEvent:
    lower = float(lower_fraction)
    upper = float(upper_fraction)
    fraction_tol = float(fraction_tolerance)
    if not 0.0 <= lower < upper <= 1.0:
        raise SeparationEventError("event fraction bracket is invalid")
    if not math.isfinite(fraction_tol) or fraction_tol <= 0.0:
        raise SeparationEventError("event fraction tolerance must be positive")

    upper_state, upper_ids, upper_damage = solve_at_fraction(upper)
    upper_candidates = event_candidate_background_ids(
        background_element_ids=upper_ids,
        damage_states=upper_damage,
        damage_event_tolerance=damage_event_tolerance,
    )
    if not upper_candidates:
        raise SeparationEventError("upper state does not contain a separation crossing")

    accepted_state = upper_state
    accepted_ids = upper_ids
    accepted_damage = upper_damage
    for _ in range(80):
        if upper - lower <= fraction_tol:
            break
        midpoint = 0.5 * (lower + upper)
        state, ids, damage = solve_at_fraction(midpoint)
        candidates = event_candidate_background_ids(
            background_element_ids=ids,
            damage_states=damage,
            damage_event_tolerance=damage_event_tolerance,
        )
        if candidates:
            upper = midpoint
            accepted_state = state
            accepted_ids = ids
            accepted_damage = damage
        else:
            lower = midpoint
    candidates = event_candidate_background_ids(
        background_element_ids=accepted_ids,
        damage_states=accepted_damage,
        damage_event_tolerance=damage_event_tolerance,
    )
    return LocatedSeparationEvent(
        fraction=upper,
        background_element_ids=candidates,
        maximum_damage=max(state.damage for state in accepted_damage),
        solved_state=accepted_state,
    )


__all__ = [
    "EventAwareSubstepProposal",
    "LocatedSeparationEvent",
    "SeparationEventError",
    "event_candidate_background_ids",
    "event_aware_substep_scale",
    "locate_earliest_separation_event",
]
