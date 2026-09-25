"""Normal penalty and augmented-Lagrangian laws for rigid-grain contact."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import NDArray

from .geometry import ContactGeometryError


@dataclass(frozen=True, slots=True)
class ContactPointState:
    normal_multiplier_Pa: float = 0.0
    tangential_traction_Pa: float = 0.0
    accumulated_slip_m: float = 0.0
    status: str = "open"

    def __post_init__(self) -> None:
        values = (
            self.normal_multiplier_Pa,
            self.tangential_traction_Pa,
            self.accumulated_slip_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ContactGeometryError("contact state values must be finite")
        if self.normal_multiplier_Pa < 0.0:
            raise ContactGeometryError("normal multiplier must be nonnegative")
        if self.accumulated_slip_m < 0.0:
            raise ContactGeometryError("accumulated slip must be nonnegative")
        if self.status not in {"open", "stick", "slip"}:
            raise ContactGeometryError("contact status must be open, stick, or slip")


@dataclass(frozen=True, slots=True)
class NormalContactResponse:
    status: str
    pressure_Pa: float
    penetration_m: float
    force_per_thickness_N_per_m: NDArray[np.float64] = field(repr=False)
    total_force_N: NDArray[np.float64] = field(repr=False)
    tangent_total_N_per_m: NDArray[np.float64] = field(repr=False)


@dataclass(frozen=True, slots=True)
class FrictionContactResponse:
    status: str
    tangential_traction_Pa: float
    tangent_Pa_per_m: float
    accumulated_slip_m: float
    dissipation_increment_J_per_m2: float


def _positive_finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContactGeometryError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ContactGeometryError(f"{name} must be finite and positive")
    return result


def normal_contact_potential_density(*, gap_m, penalty_Pa_per_m, committed_multiplier_Pa):
    """Numerical normal potential (J/m2) for fixed AL multiplier, zero when open.

    This is a primitive of the registered pressure law, not material storage
    or dissipation. Changing the multiplier requires a separate algorithm audit.
    """
    gap = float(gap_m)
    multiplier = float(committed_multiplier_Pa)
    penalty = _positive_finite(penalty_Pa_per_m, 'penalty_Pa_per_m')
    if not math.isfinite(gap) or not math.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError('invalid normal potential gap or multiplier')
    pressure = max(0.0, multiplier-penalty*gap)
    return 0.5*pressure**2/penalty


def normal_contact_response(
    *,
    gap_m: object,
    normal: object,
    tributary_length_m: object,
    thickness_m: object,
    penalty_Pa_per_m: object,
    committed_multiplier_Pa: object,
) -> NormalContactResponse:
    """Return a nodal normal contact contribution in unit and total thickness form."""

    gap = float(gap_m)
    multiplier = float(committed_multiplier_Pa)
    if not math.isfinite(gap) or not math.isfinite(multiplier) or multiplier < 0.0:
        raise ContactGeometryError("gap and committed multiplier must be finite")
    length = _positive_finite(tributary_length_m, "tributary_length_m")
    thickness = _positive_finite(thickness_m, "thickness_m")
    penalty = _positive_finite(penalty_Pa_per_m, "penalty_Pa_per_m")
    direction = np.asarray(normal, dtype=float)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ContactGeometryError("normal must contain two finite values")
    norm = float(np.linalg.norm(direction))
    if not math.isclose(norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ContactGeometryError("normal must be a unit vector")

    pressure = max(0.0, multiplier - penalty * gap)
    penetration = max(0.0, -gap)
    if pressure == 0.0:
        return NormalContactResponse(
            status="open",
            pressure_Pa=0.0,
            penetration_m=penetration,
            force_per_thickness_N_per_m=np.zeros(2, dtype=float),
            total_force_N=np.zeros(2, dtype=float),
            tangent_total_N_per_m=np.zeros((2, 2), dtype=float),
        )
    force_per_thickness = pressure * length * direction
    total_force = thickness * force_per_thickness
    # ``-d(f_contact)/du`` is the contribution required by the equilibrium
    # Newton matrix for ``r = f_contact - f_internal``.  Besides the penalty
    # term it includes the analytical circular-surface normal derivative.
    tangent = thickness * length * penalty * np.outer(direction, direction)
    return NormalContactResponse(
        status="stick",
        pressure_Pa=pressure,
        penetration_m=penetration,
        force_per_thickness_N_per_m=force_per_thickness,
        total_force_N=total_force,
        tangent_total_N_per_m=tangent,
    )


def update_augmented_normal_multiplier(
    committed: ContactPointState,
    *,
    gap_m: object,
    penalty_Pa_per_m: object,
    relaxation: object,
) -> ContactPointState:
    """Return a new Uzawa multiplier without mutating the committed state."""

    if not isinstance(committed, ContactPointState):
        raise TypeError("committed must be a ContactPointState")
    gap = float(gap_m)
    penalty = _positive_finite(penalty_Pa_per_m, "penalty_Pa_per_m")
    factor = float(relaxation)
    if not math.isfinite(gap) or not math.isfinite(factor) or factor <= 0.0:
        raise ContactGeometryError("gap and relaxation must be finite; relaxation must be positive")
    multiplier = max(
        0.0, committed.normal_multiplier_Pa - factor * penalty * gap
    )
    return ContactPointState(
        normal_multiplier_Pa=multiplier,
        tangential_traction_Pa=committed.tangential_traction_Pa,
        accumulated_slip_m=committed.accumulated_slip_m,
        status="stick" if multiplier > 0.0 else "open",
    )


def friction_contact_response(
    committed: ContactPointState,
    *,
    pressure_Pa: object,
    tangential_relative_increment_m: object,
    tangential_penalty_Pa_per_m: object,
    friction_coefficient: object,
    friction_regularization_ratio: object,
) -> FrictionContactResponse:
    """Return one regularized 2D Coulomb trial state without committing it."""

    if not isinstance(committed, ContactPointState):
        raise TypeError("committed must be a ContactPointState")
    pressure = float(pressure_Pa)
    increment = float(tangential_relative_increment_m)
    penalty = _positive_finite(
        tangential_penalty_Pa_per_m, "tangential_penalty_Pa_per_m"
    )
    coefficient = float(friction_coefficient)
    regularization = _positive_finite(
        friction_regularization_ratio, "friction_regularization_ratio"
    )
    if (
        not math.isfinite(pressure)
        or pressure < 0.0
        or not math.isfinite(increment)
        or not math.isfinite(coefficient)
        or coefficient < 0.0
    ):
        raise ContactGeometryError(
            "pressure, tangential increment, and friction coefficient are invalid"
        )
    if pressure == 0.0:
        return FrictionContactResponse(
            status="open",
            tangential_traction_Pa=0.0,
            tangent_Pa_per_m=0.0,
            accumulated_slip_m=committed.accumulated_slip_m,
            dissipation_increment_J_per_m2=0.0,
        )
    if coefficient == 0.0:
        return FrictionContactResponse(
            status="stick",
            tangential_traction_Pa=0.0,
            tangent_Pa_per_m=0.0,
            accumulated_slip_m=committed.accumulated_slip_m,
            dissipation_increment_J_per_m2=0.0,
        )
    trial = committed.tangential_traction_Pa + penalty * increment
    limit = coefficient * pressure
    tolerance = regularization * limit
    if abs(trial) <= limit + tolerance:
        return FrictionContactResponse(
            status="stick",
            tangential_traction_Pa=trial,
            tangent_Pa_per_m=penalty,
            accumulated_slip_m=committed.accumulated_slip_m,
            dissipation_increment_J_per_m2=0.0,
        )
    direction = math.copysign(1.0, trial)
    traction = direction * limit
    elastic_increment = (
        traction - committed.tangential_traction_Pa
    ) / penalty
    plastic_slip_increment = increment - elastic_increment
    accumulated = committed.accumulated_slip_m + abs(plastic_slip_increment)
    dissipation = abs(traction * plastic_slip_increment)
    return FrictionContactResponse(
        status="slip",
        tangential_traction_Pa=traction,
        tangent_Pa_per_m=0.0,
        accumulated_slip_m=accumulated,
        dissipation_increment_J_per_m2=dissipation,
    )
