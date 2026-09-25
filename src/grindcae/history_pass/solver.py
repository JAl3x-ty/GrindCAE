"""Deterministic external-load interpolation and failure reduction for Phase 6B.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from grindcae.elastoplastic_fem import ElastoplasticConvergenceError

from .models import TransitionSettings


StateT = TypeVar("StateT")


class HistoryTransitionError(RuntimeError):
    """Raised when a required load transition cannot reach its exact target."""


@dataclass(frozen=True, slots=True)
class ExternalLoadTransitionResult(Generic[StateT]):
    states: tuple[StateT, ...]
    achieved_fractions: tuple[float, ...]
    requested_substep_count: int
    achieved_substep_count: int
    retry_count: int
    maximum_newton_iterations: int


def solve_external_load_transition(
    *,
    committed: StateT,
    start_load_vector_N: NDArray[np.float64],
    target_load_vector_N: NDArray[np.float64],
    requested_substep_count: int,
    settings: TransitionSettings,
    advance: Callable[[StateT, NDArray[np.float64], float], StateT],
    marker_start: float,
) -> ExternalLoadTransitionResult[StateT]:
    """Interpolate one required target, reducing failed substeps without skipping it."""

    if type(requested_substep_count) is not int or requested_substep_count < 1:
        raise ValueError("requested_substep_count must be a positive strict integer")
    start = np.asarray(start_load_vector_N, dtype=float)
    target = np.asarray(target_load_vector_N, dtype=float)
    if start.ndim != 1 or target.shape != start.shape:
        raise ValueError("load transition vectors must be matching one-dimensional arrays")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("load transition vectors must be finite")
    pending = [index / requested_substep_count for index in range(1, requested_substep_count + 1)]
    current_fraction = 0.0
    current = committed
    states: list[StateT] = []
    fractions: list[float] = []
    retry_count = 0
    while pending:
        requested_fraction = pending.pop(0)
        attempted_fraction = requested_fraction
        while True:
            load = start + attempted_fraction * (target - start)
            try:
                converged = advance(current, load, marker_start + attempted_fraction)
                break
            except ElastoplasticConvergenceError as exc:
                half_increment = 0.5 * (attempted_fraction - current_fraction)
                if not settings.allow_reduction:
                    raise HistoryTransitionError(
                        "required target load failed and transition reduction is disabled"
                    ) from exc
                if retry_count >= settings.maximum_retries:
                    raise HistoryTransitionError(
                        "required target load exceeded maximum transition retries"
                    ) from exc
                if half_increment < settings.minimum_substep_fraction:
                    raise HistoryTransitionError(
                        "required target load failed below the minimum substep fraction"
                    ) from exc
                attempted_fraction = current_fraction + half_increment
                retry_count += 1
        states.append(converged)
        fractions.append(attempted_fraction)
        current = converged
        current_fraction = attempted_fraction
        if attempted_fraction < requested_fraction:
            pending.insert(0, requested_fraction)
    if not fractions or fractions[-1] != 1.0:
        raise HistoryTransitionError("load transition did not reach the exact target")
    return ExternalLoadTransitionResult(
        states=tuple(states),
        achieved_fractions=tuple(fractions),
        requested_substep_count=requested_substep_count,
        achieved_substep_count=len(states),
        retry_count=retry_count,
        maximum_newton_iterations=max(
            (int(getattr(state, "newton_iterations", 0)) for state in states),
            default=0,
        ),
    )
