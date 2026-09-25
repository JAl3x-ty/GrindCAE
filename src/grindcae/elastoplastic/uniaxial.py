"""Uniaxial-stress loading and zero-stress unloading material-point driver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .j2 import (
    MaterialPointError,
    MaterialPointState,
    initial_material_point_state,
    update_j2_material_point,
)
from .models import BilinearIsotropicMaterial, MaterialPointCase


Segment = Literal["initial", "loading", "unloading"]
_TRANSVERSE_STRESS_ABS_TOLERANCE_PA = 5.0e-4
_AXIAL_STRESS_ABS_TOLERANCE_PA = 5.0e-4


class UniaxialDriverError(RuntimeError):
    """Raised when the controlled uniaxial-stress local solve does not converge."""


def _finite_symmetric_total_strain(value: object) -> np.ndarray:
    try:
        tensor = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UniaxialDriverError(
            "total_strain [symmetric 3 x 3 tensor]: contains a non-numeric value"
        ) from exc
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise UniaxialDriverError(
            "total_strain [symmetric 3 x 3 tensor]: shape must be 3 x 3 and values finite"
        )
    if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1.0e-15):
        raise UniaxialDriverError(
            "total_strain [symmetric 3 x 3 tensor]: must be symmetric"
        )
    return 0.5 * (tensor + tensor.T)


@dataclass(frozen=True, slots=True)
class UniaxialStep:
    """One converged state in the uniaxial loading-unloading history."""

    step_id: int
    segment: Segment
    total_strain_tensor: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    state: MaterialPointState

    @property
    def axial_total_strain(self) -> float:
        return self.total_strain_tensor[0][0]

    @property
    def transverse_total_strain_y(self) -> float:
        return self.total_strain_tensor[1][1]

    @property
    def transverse_total_strain_z(self) -> float:
        return self.total_strain_tensor[2][2]

    @property
    def axial_stress_Pa(self) -> float:
        return self.state.stress_tensor_Pa[0][0]

    @property
    def transverse_stress_y_Pa(self) -> float:
        return self.state.stress_tensor_Pa[1][1]

    @property
    def transverse_stress_z_Pa(self) -> float:
        return self.state.stress_tensor_Pa[2][2]

    @property
    def axial_plastic_strain(self) -> float:
        return self.state.plastic_strain_tensor[0][0]

    @property
    def increment_class(self) -> str:
        return self.state.increment_class

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "step_id": self.step_id,
            "segment": self.segment,
            "axial_total_strain": self.axial_total_strain,
            "transverse_total_strain_y": self.transverse_total_strain_y,
            "transverse_total_strain_z": self.transverse_total_strain_z,
            "axial_stress_Pa": self.axial_stress_Pa,
            "transverse_stress_y_Pa": self.transverse_stress_y_Pa,
            "transverse_stress_z_Pa": self.transverse_stress_z_Pa,
            "axial_plastic_strain": self.axial_plastic_strain,
            "equivalent_plastic_strain": self.state.equivalent_plastic_strain,
            "current_yield_strength_Pa": self.state.current_yield_strength_Pa,
            "yield_function_Pa": self.state.yield_function_Pa,
            "increment_class": self.increment_class,
        }


@dataclass(frozen=True, slots=True)
class UniaxialHistoryResult:
    """Complete independent phase 6A.1 material-point history."""

    case: MaterialPointCase
    steps: tuple[UniaxialStep, ...]
    first_yield_step_id: int | None
    yield_strain: float

    @property
    def peak_step(self) -> UniaxialStep:
        return self.steps[self.case.history.loading_step_count]

    @property
    def final_step(self) -> UniaxialStep:
        return self.steps[-1]

    @property
    def maximum_plastic_yield_function_residual_Pa(self) -> float:
        residuals = [
            abs(step.state.yield_function_Pa)
            for step in self.steps
            if step.increment_class == "plastic"
        ]
        return max(residuals, default=0.0)

    @property
    def maximum_absolute_transverse_stress_Pa(self) -> float:
        return max(
            max(
                abs(step.transverse_stress_y_Pa),
                abs(step.transverse_stress_z_Pa),
            )
            for step in self.steps
        )


def _strain_tensor(axial: float, transverse: float) -> np.ndarray:
    return np.diag([axial, transverse, transverse]).astype(float)


def _immutable_tensor(tensor: np.ndarray) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return tuple(tuple(float(value) for value in row) for row in tensor)  # type: ignore[return-value]


def _trial_from_components(
    material: BilinearIsotropicMaterial,
    state: MaterialPointState,
    axial_increment: float,
    transverse_increment: float,
) -> MaterialPointState:
    return update_j2_material_point(
        material,
        state,
        _strain_tensor(axial_increment, transverse_increment),
    )


def advance_uniaxial_axial_strain(
    material: BilinearIsotropicMaterial,
    state: MaterialPointState,
    total_strain: object,
    target_axial_total_strain: float,
) -> tuple[MaterialPointState, np.ndarray]:
    """Advance prescribed axial strain while locally enforcing zero lateral stress."""

    current_total = _finite_symmetric_total_strain(total_strain)
    target = float(target_axial_total_strain)
    if not math.isfinite(target):
        raise UniaxialDriverError(
            "target_axial_total_strain [strain]: must be finite"
        )
    axial_increment = target - float(current_total[0, 0])
    scale = max(abs(axial_increment), np.finfo(float).eps)
    lower = -4.0 * scale
    upper = 4.0 * scale

    def evaluate(transverse_increment: float) -> tuple[float, MaterialPointState]:
        candidate = _trial_from_components(
            material, state, axial_increment, transverse_increment
        )
        return candidate.stress_tensor_Pa[1][1], candidate

    lower_value, lower_state = evaluate(lower)
    upper_value, upper_state = evaluate(upper)
    for _ in range(20):
        if lower_value <= 0.0 <= upper_value:
            break
        lower *= 2.0
        upper *= 2.0
        lower_value, lower_state = evaluate(lower)
        upper_value, upper_state = evaluate(upper)
    else:
        raise UniaxialDriverError(
            "transverse_strain_increment [strain]: could not bracket the zero-lateral-stress solution"
        )

    chosen_increment = lower
    chosen_state = lower_state
    for _ in range(120):
        middle = 0.5 * (lower + upper)
        middle_value, middle_state = evaluate(middle)
        chosen_increment = middle
        chosen_state = middle_state
        if abs(middle_value) <= _TRANSVERSE_STRESS_ABS_TOLERANCE_PA:
            break
        if middle_value < 0.0:
            lower = middle
            lower_value = middle_value
        else:
            upper = middle
            upper_value = middle_value
    else:
        raise UniaxialDriverError(
            "transverse_strain_increment [strain]: zero-lateral-stress solve did not converge"
        )

    new_total = current_total + _strain_tensor(axial_increment, chosen_increment)
    return chosen_state, new_total


def advance_uniaxial_axial_stress(
    material: BilinearIsotropicMaterial,
    state: MaterialPointState,
    total_strain: object,
    target_axial_stress_Pa: float,
) -> tuple[MaterialPointState, np.ndarray]:
    """Advance to a target axial stress while enforcing both lateral stresses zero."""

    current_total = _finite_symmetric_total_strain(total_strain)
    target = float(target_axial_stress_Pa)
    if not math.isfinite(target):
        raise UniaxialDriverError("target_axial_stress_Pa [Pa]: must be finite")
    current_axial_stress = state.stress_tensor_Pa[0][0]
    stress_change = target - current_axial_stress
    unknown = np.array(
        [stress_change / material.E, -material.nu * stress_change / material.E],
        dtype=float,
    )

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, MaterialPointState]:
        candidate = _trial_from_components(material, state, values[0], values[1])
        residual = np.array(
            [
                candidate.stress_tensor_Pa[0][0] - target,
                candidate.stress_tensor_Pa[1][1],
            ],
            dtype=float,
        )
        return residual, candidate

    chosen_state: MaterialPointState | None = None
    for _ in range(12):
        residual, candidate = evaluate(unknown)
        chosen_state = candidate
        if (
            abs(residual[0]) <= _AXIAL_STRESS_ABS_TOLERANCE_PA
            and abs(residual[1]) <= _TRANSVERSE_STRESS_ABS_TOLERANCE_PA
        ):
            break
        jacobian = np.empty((2, 2), dtype=float)
        for index in range(2):
            step = max(1.0e-12, abs(unknown[index]) * 1.0e-6)
            plus = unknown.copy()
            minus = unknown.copy()
            plus[index] += step
            minus[index] -= step
            residual_plus, _ = evaluate(plus)
            residual_minus, _ = evaluate(minus)
            jacobian[:, index] = (residual_plus - residual_minus) / (2.0 * step)
        try:
            correction = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError as exc:
            raise UniaxialDriverError(
                "stress_control [local solve]: singular numerical Jacobian"
            ) from exc
        unknown -= correction
    else:
        raise UniaxialDriverError(
            "stress_control [local solve]: target uniaxial stress did not converge"
        )

    if chosen_state is None:
        raise UniaxialDriverError("stress_control [local solve]: no state was evaluated")
    new_total = current_total + _strain_tensor(unknown[0], unknown[1])
    return chosen_state, new_total


def _make_step(
    step_id: int,
    segment: Segment,
    total_strain: np.ndarray,
    state: MaterialPointState,
) -> UniaxialStep:
    return UniaxialStep(
        step_id=step_id,
        segment=segment,
        total_strain_tensor=_immutable_tensor(total_strain),
        state=state,
    )


def run_uniaxial_loading_unloading(case: MaterialPointCase) -> UniaxialHistoryResult:
    """Run x-tension to the prescribed peak and unload to zero axial stress."""

    if not isinstance(case, MaterialPointCase):
        raise UniaxialDriverError(
            "case [MaterialPointCase]: has the wrong object type"
        )
    material = case.material
    state = initial_material_point_state(material)
    total_strain = np.zeros((3, 3), dtype=float)
    steps = [_make_step(0, "initial", total_strain, state)]
    first_yield_step_id: int | None = None

    for index in range(1, case.history.loading_step_count + 1):
        target_strain = (
            case.history.peak_axial_strain
            * index
            / case.history.loading_step_count
        )
        state, total_strain = advance_uniaxial_axial_strain(
            material, state, total_strain, target_strain
        )
        step = _make_step(index, "loading", total_strain, state)
        steps.append(step)
        if first_yield_step_id is None and step.increment_class == "plastic":
            first_yield_step_id = step.step_id

    peak_stress = steps[-1].axial_stress_Pa
    for index in range(1, case.history.unloading_step_count + 1):
        target_stress = peak_stress * (
            1.0 - index / case.history.unloading_step_count
        )
        state, total_strain = advance_uniaxial_axial_stress(
            material, state, total_strain, target_stress
        )
        steps.append(
            _make_step(
                case.history.loading_step_count + index,
                "unloading",
                total_strain,
                state,
            )
        )

    return UniaxialHistoryResult(
        case=case,
        steps=tuple(steps),
        first_yield_step_id=first_yield_step_id,
        yield_strain=material.yield_strength / material.E,
    )
