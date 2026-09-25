from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from grindcae.elastoplastic_fem import ElastoplasticConvergenceError
from grindcae.history_pass.solver import (
    HistoryTransitionError,
    solve_external_load_transition,
)
from grindcae.history_pass.models import TransitionSettings


@dataclass(frozen=True)
class FakeState:
    marker: float
    displacement_vector_m: np.ndarray
    element_states: tuple[object, ...]
    external_load_vector_N: np.ndarray
    newton_iterations: int = 1


def _settings(**changes: object) -> TransitionSettings:
    values = {
        "base_substep_count": 2,
        "final_unloading_substep_count": 4,
        "allow_reduction": True,
        "minimum_substep_fraction": 0.125,
        "maximum_retries": 4,
    }
    values.update(changes)
    return TransitionSettings(**values)


def test_transition_interpolates_from_committed_load_and_reaches_exact_target() -> None:
    initial = FakeState(0.0, np.asarray([0.0]), ("initial",), np.asarray([2.0, -4.0]))
    calls: list[tuple[FakeState, np.ndarray]] = []

    def advance(committed: FakeState, target: np.ndarray, marker: float) -> FakeState:
        calls.append((committed, target.copy()))
        return FakeState(marker, np.asarray([marker]), (marker,), target.copy())

    result = solve_external_load_transition(
        committed=initial,
        start_load_vector_N=initial.external_load_vector_N,
        target_load_vector_N=np.asarray([10.0, -20.0]),
        requested_substep_count=2,
        settings=_settings(),
        advance=advance,
        marker_start=10.0,
    )

    assert [target.tolist() for _, target in calls] == [[6.0, -12.0], [10.0, -20.0]]
    assert calls[0][0] is initial
    assert calls[1][0] is result.states[0]
    assert result.states[-1].external_load_vector_N == pytest.approx([10.0, -20.0])
    assert result.achieved_substep_count == 2
    assert result.retry_count == 0


def test_failed_substep_is_retried_from_unchanged_committed_state_with_half_fraction() -> None:
    initial = FakeState(0.0, np.asarray([0.0]), ("initial",), np.asarray([0.0]))
    calls: list[tuple[FakeState, float]] = []
    failed_once = False

    def advance(committed: FakeState, target: np.ndarray, marker: float) -> FakeState:
        nonlocal failed_once
        calls.append((committed, float(target[0])))
        if not failed_once and target[0] == pytest.approx(0.5):
            failed_once = True
            raise ElastoplasticConvergenceError(
                "injected",
                attempted_load_factor=marker,
                residual_norm_history_N=(1.0,),
            )
        return FakeState(marker, np.asarray([marker]), (marker,), target.copy())

    result = solve_external_load_transition(
        committed=initial,
        start_load_vector_N=np.asarray([0.0]),
        target_load_vector_N=np.asarray([1.0]),
        requested_substep_count=2,
        settings=_settings(),
        advance=advance,
        marker_start=0.0,
    )

    assert calls[0][0] is initial
    assert calls[1][0] is initial
    assert [value for _, value in calls[:3]] == pytest.approx([0.5, 0.25, 0.5])
    assert result.retry_count == 1
    assert result.states[-1].external_load_vector_N == pytest.approx([1.0])


def test_transition_fails_without_skipping_when_reduction_reaches_minimum() -> None:
    initial = FakeState(0.0, np.asarray([0.0]), ("initial",), np.asarray([0.0]))
    committed_ids: list[int] = []

    def always_fail(committed: FakeState, target: np.ndarray, marker: float) -> FakeState:
        committed_ids.append(id(committed))
        raise ElastoplasticConvergenceError(
            "injected",
            attempted_load_factor=marker,
            residual_norm_history_N=(1.0,),
        )

    with pytest.raises(HistoryTransitionError, match="minimum substep"):
        solve_external_load_transition(
            committed=initial,
            start_load_vector_N=np.asarray([0.0]),
            target_load_vector_N=np.asarray([1.0]),
            requested_substep_count=2,
            settings=_settings(minimum_substep_fraction=0.25),
            advance=always_fail,
            marker_start=0.0,
        )

    assert committed_ids and set(committed_ids) == {id(initial)}
