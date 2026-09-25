"""Load-unload history orchestration for Phase 6A.2."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .models import ElastoplasticFemCase
from .solver import (
    CommittedStructureState,
    ConvergedIncrementState,
    ElastoplasticConvergenceError,
    PreparedElastoplasticMesh,
    advance_load_increment,
    initial_structure_state,
)


@dataclass(frozen=True, slots=True)
class IncrementRecord:
    increment_id: int
    segment: str
    requested_load_factor: float
    achieved_load_factor: float
    load_factor_increment: float
    retry_count: int
    step_reduction_occurred: bool
    state: CommittedStructureState | ConvergedIncrementState
    equivalent_plastic_strain: NDArray[np.float64]

    @property
    def load_factor(self) -> float:
        return self.achieved_load_factor

    def __getattr__(self, name: str) -> object:
        return getattr(self.state, name)


@dataclass(frozen=True, slots=True)
class LoadingUnloadingResult:
    case: ElastoplasticFemCase
    prepared_mesh: PreparedElastoplasticMesh
    increments: tuple[IncrementRecord, ...]
    peak: IncrementRecord
    residual: IncrementRecord

    @property
    def maximum_newton_iterations(self) -> int:
        return max(int(getattr(item.state, "newton_iterations", 0)) for item in self.increments)

    @property
    def any_step_reduction(self) -> bool:
        return any(item.step_reduction_occurred for item in self.increments)

    @property
    def maximum_balance_residual_norm_N(self) -> float:
        return max(float(getattr(item.state, "balance_residual_norm_N", 0.0)) for item in self.increments)


def _plastic_array(
    state: CommittedStructureState | ConvergedIncrementState,
) -> NDArray[np.float64]:
    return np.array(
        [point.equivalent_plastic_strain for point in state.element_states],
        dtype=float,
    )


def _targets(case: ElastoplasticFemCase) -> tuple[tuple[str, float], ...]:
    loading = tuple(
        ("loading", index / case.increments.loading_step_count)
        for index in range(1, case.increments.loading_step_count + 1)
    )
    unloading = tuple(
        ("unloading", 1.0 - index / case.increments.unloading_step_count)
        for index in range(1, case.increments.unloading_step_count + 1)
    )
    return loading + unloading


def solve_loading_unloading(
    case: ElastoplasticFemCase,
    prepared: PreparedElastoplasticMesh,
) -> LoadingUnloadingResult:
    """Solve the deterministic ``0 -> 1 -> 0`` load-factor history."""

    committed: CommittedStructureState | ConvergedIncrementState = initial_structure_state(
        case, prepared
    )
    zero_plastic = _plastic_array(committed)
    records: list[IncrementRecord] = [
        IncrementRecord(
            increment_id=0,
            segment="initial",
            requested_load_factor=0.0,
            achieved_load_factor=0.0,
            load_factor_increment=0.0,
            retry_count=0,
            step_reduction_occurred=False,
            state=committed,
            equivalent_plastic_strain=zero_plastic,
        )
    ]
    pending = list(_targets(case))
    while pending:
        segment, requested = pending.pop(0)
        start_factor = committed.load_factor
        attempted = requested
        retry_count = 0
        reduced = False
        while True:
            try:
                converged = advance_load_increment(
                    case,
                    prepared,
                    committed,
                    target_load_factor=attempted,
                )
                break
            except ElastoplasticConvergenceError:
                step = attempted - start_factor
                if (
                    not case.step_control.allow_reduction
                    or retry_count >= case.step_control.maximum_retries
                    or 0.5 * abs(step)
                    < case.step_control.minimum_load_factor_increment
                ):
                    raise
                attempted = start_factor + 0.5 * step
                retry_count += 1
                reduced = True
        record = IncrementRecord(
            increment_id=len(records),
            segment=segment,
            requested_load_factor=requested,
            achieved_load_factor=attempted,
            load_factor_increment=attempted - start_factor,
            retry_count=retry_count,
            step_reduction_occurred=reduced,
            state=converged,
            equivalent_plastic_strain=_plastic_array(converged),
        )
        records.append(record)
        committed = converged
        if reduced and not math.isclose(attempted, requested, rel_tol=0.0, abs_tol=1.0e-15):
            pending.insert(0, (segment, requested))

    peak = next(record for record in records if record.load_factor == 1.0)
    residual = records[-1]
    if residual.segment != "unloading" or not math.isclose(
        residual.load_factor, 0.0, rel_tol=0.0, abs_tol=1.0e-14
    ):
        raise RuntimeError("loading-unloading history did not finish at zero load")
    return LoadingUnloadingResult(
        case=case,
        prepared_mesh=prepared,
        increments=tuple(records),
        peak=peak,
        residual=residual,
    )
