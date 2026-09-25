"""Three-dimensional small-strain J2 radial-return material-point kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .models import BilinearIsotropicMaterial


Tensor3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
IncrementClass = Literal["initial", "elastic", "plastic"]


class MaterialPointError(ValueError):
    """Raised when a material state or strain increment is invalid."""


def _tensor(value: object, field: str) -> np.ndarray:
    try:
        tensor = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MaterialPointError(
            f"{field} [symmetric 3 x 3 tensor]: contains a non-numeric value"
        ) from exc
    if tensor.shape != (3, 3):
        raise MaterialPointError(
            f"{field} [symmetric 3 x 3 tensor]: has shape {tensor.shape}"
        )
    if not np.all(np.isfinite(tensor)):
        raise MaterialPointError(
            f"{field} [symmetric 3 x 3 tensor]: all values must be finite"
        )
    scale = max(1.0, float(np.max(np.abs(tensor))))
    if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1.0e-14 * scale):
        raise MaterialPointError(
            f"{field} [symmetric 3 x 3 tensor]: tensor must be symmetric"
        )
    return 0.5 * (tensor + tensor.T)


def _immutable_tensor(tensor: np.ndarray) -> Tensor3:
    return tuple(tuple(float(value) for value in row) for row in tensor)  # type: ignore[return-value]


def _finite_scalar(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialPointError(f"{field} [finite scalar]: has the wrong type")
    number = float(value)
    if not math.isfinite(number):
        raise MaterialPointError(f"{field} [finite scalar]: must be finite")
    return number


@dataclass(frozen=True, slots=True)
class MaterialPointState:
    """One converged immutable J2 material-point state."""

    stress_tensor_Pa: Tensor3
    plastic_strain_tensor: Tensor3
    equivalent_plastic_strain: float
    current_yield_strength_Pa: float
    yield_function_Pa: float
    increment_class: IncrementClass

    def __post_init__(self) -> None:
        stress = _tensor(self.stress_tensor_Pa, "state.stress_tensor_Pa")
        plastic_strain = _tensor(
            self.plastic_strain_tensor, "state.plastic_strain_tensor"
        )
        equivalent_plastic_strain = _finite_scalar(
            self.equivalent_plastic_strain,
            "state.equivalent_plastic_strain",
        )
        if equivalent_plastic_strain < 0.0:
            raise MaterialPointError(
                "state.equivalent_plastic_strain [strain]: must not be negative"
            )
        current_yield = _finite_scalar(
            self.current_yield_strength_Pa,
            "state.current_yield_strength_Pa",
        )
        if current_yield <= 0.0:
            raise MaterialPointError(
                "state.current_yield_strength_Pa [Pa]: must be positive"
            )
        yield_function = _finite_scalar(
            self.yield_function_Pa, "state.yield_function_Pa"
        )
        if self.increment_class not in ("initial", "elastic", "plastic"):
            raise MaterialPointError(
                "state.increment_class [literal]: must be initial, elastic, or plastic"
            )
        object.__setattr__(self, "stress_tensor_Pa", _immutable_tensor(stress))
        object.__setattr__(
            self, "plastic_strain_tensor", _immutable_tensor(plastic_strain)
        )
        object.__setattr__(
            self, "equivalent_plastic_strain", equivalent_plastic_strain
        )
        object.__setattr__(self, "current_yield_strength_Pa", current_yield)
        object.__setattr__(self, "yield_function_Pa", yield_function)

    def to_dict(self) -> dict[str, object]:
        return {
            "stress_tensor_Pa": [list(row) for row in self.stress_tensor_Pa],
            "plastic_strain_tensor": [
                list(row) for row in self.plastic_strain_tensor
            ],
            "equivalent_plastic_strain": self.equivalent_plastic_strain,
            "current_yield_strength_Pa": self.current_yield_strength_Pa,
            "yield_function_Pa": self.yield_function_Pa,
            "increment_class": self.increment_class,
        }


@dataclass(frozen=True, slots=True)
class PlaneStrainJ2Update:
    """One plane-strain trial update and its analytic algorithmic tangent.

    The strain vector convention is ``[epsilon_xx, epsilon_yy, gamma_xy]``;
    ``gamma_xy`` is engineering shear strain and is converted to the symmetric
    tensor component ``epsilon_xy = gamma_xy / 2`` exactly once here.
    """

    state: MaterialPointState
    in_plane_stress_Pa: tuple[float, float, float]
    sigma_zz_Pa: float
    algorithmic_tangent_Pa: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    strain_increment_tensor: Tensor3
    stress_tensor_tangent_Pa: tuple[Tensor3, Tensor3, Tensor3]
    equivalent_plastic_strain_gradient: tuple[float, float, float]


def _plane_strain_tensor(value: object) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MaterialPointError(
            "plane_strain_increment [epsilon_xx, epsilon_yy, gamma_xy]: "
            "contains a non-numeric value"
        ) from exc
    if vector.shape != (3,):
        raise MaterialPointError(
            "plane_strain_increment [epsilon_xx, epsilon_yy, gamma_xy]: "
            f"has shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise MaterialPointError(
            "plane_strain_increment [epsilon_xx, epsilon_yy, gamma_xy]: "
            "all values must be finite"
        )
    tensor = np.zeros((3, 3), dtype=float)
    tensor[0, 0] = vector[0]
    tensor[1, 1] = vector[1]
    tensor[0, 1] = tensor[1, 0] = 0.5 * vector[2]
    return tensor


def _elastic_tangent_action(
    material: BilinearIsotropicMaterial,
    strain: np.ndarray,
) -> np.ndarray:
    trace = float(np.trace(strain))
    deviatoric = strain - trace / 3.0 * np.eye(3)
    return (
        material.bulk_modulus * trace * np.eye(3)
        + 2.0 * material.shear_modulus * deviatoric
    )


def _algorithmic_tangent_action(
    material: BilinearIsotropicMaterial,
    state: MaterialPointState,
    increment: np.ndarray,
    direction: np.ndarray,
    increment_class: IncrementClass,
    *,
    allow_zero_plastic_limit: bool = False,
) -> np.ndarray:
    if increment_class != "plastic":
        return _elastic_tangent_action(material, direction)

    stress_n = _tensor(state.stress_tensor_Pa, "state.stress_tensor_Pa")
    trial_stress = stress_n + _elastic_tangent_action(material, increment)
    trial_pressure = float(np.trace(trial_stress)) / 3.0
    trial_deviatoric = trial_stress - trial_pressure * np.eye(3)
    q_trial = math.sqrt(
        max(1.5 * float(np.sum(trial_deviatoric * trial_deviatoric)), 0.0)
    )
    hardening = material.internal_hardening_modulus
    yield_n = material.yield_strength + hardening * state.equivalent_plastic_strain
    denominator = 3.0 * material.shear_modulus + hardening
    trial_yield_function = q_trial - yield_n
    plastic_limit_tolerance = max(1.0e-7, 1.0e-12 * max(yield_n, q_trial, 1.0))
    delta_gamma = trial_yield_function / denominator
    if allow_zero_plastic_limit and trial_yield_function >= -plastic_limit_tolerance:
        delta_gamma = max(0.0, delta_gamma)
    if q_trial <= 0.0 or delta_gamma < 0.0 or (
        delta_gamma == 0.0 and not allow_zero_plastic_limit
    ):
        raise MaterialPointError(
            "algorithmic_tangent [Pa]: plastic trial variables are invalid"
        )

    shear = material.shear_modulus
    radial_scale = 1.0 - 3.0 * shear * delta_gamma / q_trial
    trace_direction = float(np.trace(direction))
    deviatoric_direction = direction - trace_direction / 3.0 * np.eye(3)
    trial_deviatoric_direction = 2.0 * shear * deviatoric_direction
    q_direction = (
        1.5
        * float(np.sum(trial_deviatoric * trial_deviatoric_direction))
        / q_trial
    )
    radial_scale_direction = (
        -3.0
        * shear
        * q_direction
        * yield_n
        / (denominator * q_trial**2)
    )
    return (
        material.bulk_modulus * trace_direction * np.eye(3)
        + radial_scale * trial_deviatoric_direction
        + radial_scale_direction * trial_deviatoric
    )


def equivalent_von_mises_stress(stress_tensor_Pa: object) -> float:
    """Return ``sqrt(3/2 s:s)`` for one finite symmetric stress tensor."""

    stress = _tensor(stress_tensor_Pa, "stress_tensor_Pa")
    deviatoric = stress - np.trace(stress) / 3.0 * np.eye(3)
    squared = 1.5 * float(np.sum(deviatoric * deviatoric))
    return math.sqrt(max(squared, 0.0))


def initial_material_point_state(
    material: BilinearIsotropicMaterial,
) -> MaterialPointState:
    """Create the prescribed initially unstressed and unstrained state."""

    if not isinstance(material, BilinearIsotropicMaterial):
        raise MaterialPointError(
            "material [BilinearIsotropicMaterial]: has the wrong object type"
        )
    zero = np.zeros((3, 3), dtype=float)
    return MaterialPointState(
        stress_tensor_Pa=_immutable_tensor(zero),
        plastic_strain_tensor=_immutable_tensor(zero),
        equivalent_plastic_strain=0.0,
        current_yield_strength_Pa=material.yield_strength,
        yield_function_Pa=-material.yield_strength,
        increment_class="initial",
    )


def update_j2_material_point(
    material: BilinearIsotropicMaterial,
    state: MaterialPointState,
    total_strain_increment: object,
) -> MaterialPointState:
    """Apply one strain increment using an associative J2 radial return."""

    if not isinstance(material, BilinearIsotropicMaterial):
        raise MaterialPointError(
            "material [BilinearIsotropicMaterial]: has the wrong object type"
        )
    if not isinstance(state, MaterialPointState):
        raise MaterialPointError(
            "state [MaterialPointState]: has the wrong object type"
        )
    increment = _tensor(total_strain_increment, "strain_increment")
    stress_n = _tensor(state.stress_tensor_Pa, "state.stress_tensor_Pa")
    plastic_strain_n = _tensor(
        state.plastic_strain_tensor, "state.plastic_strain_tensor"
    )
    alpha_n = state.equivalent_plastic_strain
    hardening = material.internal_hardening_modulus
    expected_yield_n = material.yield_strength + hardening * alpha_n
    state_tolerance = max(1.0e-6, 1.0e-12 * expected_yield_n)
    if not math.isclose(
        state.current_yield_strength_Pa,
        expected_yield_n,
        rel_tol=0.0,
        abs_tol=state_tolerance,
    ):
        raise MaterialPointError(
            "state.current_yield_strength_Pa [Pa]: inconsistent with material and equivalent plastic strain"
        )

    trace_increment = float(np.trace(increment))
    deviatoric_increment = increment - trace_increment / 3.0 * np.eye(3)
    trial_stress = (
        stress_n
        + material.bulk_modulus * trace_increment * np.eye(3)
        + 2.0 * material.shear_modulus * deviatoric_increment
    )
    pressure = float(np.trace(trial_stress)) / 3.0
    trial_deviatoric = trial_stress - pressure * np.eye(3)
    q_trial = math.sqrt(
        max(1.5 * float(np.sum(trial_deviatoric * trial_deviatoric)), 0.0)
    )
    yield_n = material.yield_strength + hardening * alpha_n
    trial_yield_function = q_trial - yield_n
    elastic_tolerance = max(1.0e-7, 1.0e-12 * max(yield_n, q_trial, 1.0))

    if trial_yield_function <= elastic_tolerance:
        new_stress = trial_stress
        new_plastic_strain = plastic_strain_n
        new_alpha = alpha_n
        increment_class: IncrementClass = "elastic"
    else:
        denominator = 3.0 * material.shear_modulus + hardening
        delta_gamma = trial_yield_function / denominator
        if not math.isfinite(delta_gamma) or delta_gamma <= 0.0 or q_trial <= 0.0:
            raise MaterialPointError(
                "plastic_correction [strain]: radial-return increment is invalid"
            )
        flow_direction = 1.5 * trial_deviatoric / q_trial
        new_plastic_strain = plastic_strain_n + delta_gamma * flow_direction
        returned_deviatoric = (
            1.0 - 3.0 * material.shear_modulus * delta_gamma / q_trial
        ) * trial_deviatoric
        new_stress = pressure * np.eye(3) + returned_deviatoric
        new_alpha = alpha_n + delta_gamma
        increment_class = "plastic"

    new_yield = material.yield_strength + hardening * new_alpha
    new_q = equivalent_von_mises_stress(new_stress)
    new_yield_function = new_q - new_yield
    if not np.all(np.isfinite(new_stress)) or not np.all(
        np.isfinite(new_plastic_strain)
    ):
        raise MaterialPointError("material_update [state]: produced non-finite tensors")
    if new_alpha + 1.0e-16 < alpha_n:
        raise MaterialPointError(
            "equivalent_plastic_strain [strain]: material history decreased"
        )

    return MaterialPointState(
        stress_tensor_Pa=_immutable_tensor(new_stress),
        plastic_strain_tensor=_immutable_tensor(new_plastic_strain),
        equivalent_plastic_strain=new_alpha,
        current_yield_strength_Pa=new_yield,
        yield_function_Pa=new_yield_function,
        increment_class=increment_class,
    )


def update_plane_strain_j2(
    material: BilinearIsotropicMaterial,
    state: MaterialPointState,
    plane_strain_increment: object,
    *,
    active_side_direction: object | None = None,
) -> PlaneStrainJ2Update:
    """Apply one ``plane_strain`` increment using the shared 3D J2 update.

    The returned 3 x 3 matrix maps engineering strain increments ordered as
    ``[epsilon_xx, epsilon_yy, gamma_xy]`` to stresses ordered as
    ``[sigma_xx, sigma_yy, tau_xy]``.  The production tangent is analytic; a
    finite-difference approximation is intentionally left to tests.
    """

    increment = _plane_strain_tensor(plane_strain_increment)
    updated_state = update_j2_material_point(material, state, increment)
    stress = _tensor(updated_state.stress_tensor_Pa, "updated_state.stress_tensor_Pa")
    basis_directions = (
        _plane_strain_tensor((1.0, 0.0, 0.0)),
        _plane_strain_tensor((0.0, 1.0, 0.0)),
        _plane_strain_tensor((0.0, 0.0, 1.0)),
    )
    tangent_columns: list[np.ndarray] = []
    tensor_tangent_columns: list[Tensor3] = []
    alpha_gradient: list[float] = []
    stress_n = _tensor(state.stress_tensor_Pa, "state.stress_tensor_Pa")
    trial_stress = stress_n + _elastic_tangent_action(material, increment)
    trial_pressure = float(np.trace(trial_stress)) / 3.0
    trial_deviatoric = trial_stress - trial_pressure * np.eye(3)
    q_trial = math.sqrt(
        max(1.5 * float(np.sum(trial_deviatoric * trial_deviatoric)), 0.0)
    )
    denominator = 3.0 * material.shear_modulus + material.internal_hardening_modulus
    active_direction = (
        None
        if active_side_direction is None
        else _plane_strain_tensor(active_side_direction)
    )
    use_plastic_limit = False
    if active_direction is not None and updated_state.increment_class == "elastic":
        increment_scale = float(np.linalg.norm(increment))
        yield_scale = max(
            1.0,
            material.yield_strength
            + material.internal_hardening_modulus * state.equivalent_plastic_strain,
        )
        yield_distance=q_trial-(material.yield_strength
            +material.internal_hardening_modulus*state.equivalent_plastic_strain)
        on_yield=abs(yield_distance)<=max(1.0e-7,1.0e-12*max(yield_scale,q_trial))
        if increment_scale <= 64.0 * np.finfo(float).eps and on_yield:
            active_trial_direction = _elastic_tangent_action(material, active_direction)
            active_q_direction = (
                1.5
                * float(np.sum(trial_deviatoric * active_trial_direction))
                / q_trial
                if q_trial > 0.0
                else 0.0
            )
            use_plastic_limit = active_q_direction > 1.0e-12 * yield_scale
    for direction in basis_directions:
        tangent_class: IncrementClass = (
            "plastic" if use_plastic_limit else updated_state.increment_class
        )
        stress_direction = _algorithmic_tangent_action(
            material,
            state,
            increment,
            direction,
            tangent_class,
            allow_zero_plastic_limit=use_plastic_limit,
        )
        tensor_tangent_columns.append(_immutable_tensor(stress_direction))
        tangent_columns.append(
            np.array(
                [
                    stress_direction[0, 0],
                    stress_direction[1, 1],
                    stress_direction[0, 1],
                ],
                dtype=float,
            )
        )
        if tangent_class == "plastic":
            trial_direction = _elastic_tangent_action(material, direction)
            q_direction = (
                1.5
                * float(np.sum(trial_deviatoric * trial_direction))
                / q_trial
            )
            alpha_gradient.append(q_direction / denominator)
        else:
            alpha_gradient.append(0.0)
    tangent = np.column_stack(tangent_columns)
    if not np.all(np.isfinite(tangent)):
        raise MaterialPointError(
            "algorithmic_tangent [Pa]: produced a non-finite value"
        )
    return PlaneStrainJ2Update(
        state=updated_state,
        in_plane_stress_Pa=(
            float(stress[0, 0]),
            float(stress[1, 1]),
            float(stress[0, 1]),
        ),
        sigma_zz_Pa=float(stress[2, 2]),
        algorithmic_tangent_Pa=tuple(
            tuple(float(value) for value in row) for row in tangent
        ),  # type: ignore[arg-type]
        strain_increment_tensor=_immutable_tensor(increment),
        stress_tensor_tangent_Pa=tuple(tensor_tangent_columns),  # type: ignore[arg-type]
        equivalent_plastic_strain_gradient=tuple(alpha_gradient),  # type: ignore[arg-type]
    )
