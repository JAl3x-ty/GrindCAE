"""Deterministic scalar ductile-damage evolution for GrindCAE 4.0.0."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from grindcae.elastoplastic.j2 import equivalent_von_mises_stress

from .damage_models import DamageInput
from .energetic_damage import EnergeticDamageHistory


class DamageMaterialError(ValueError):
    """Raised when a damage-state update is not physically well formed."""


@dataclass(frozen=True, slots=True)
class DamageState:
    equivalent_plastic_strain: float
    omega: float
    damage: float
    status: str
    initiation_equivalent_plastic_strain: float | None
    initiation_equivalent_stress_Pa: float | None
    post_initiation_plastic_displacement_m: float
    nominal_equivalent_stress_Pa: float
    damage_dissipation_density_J_per_m3: float
    energetic_history: EnergeticDamageHistory | None = None


def initial_damage_state() -> DamageState:
    return DamageState(
        equivalent_plastic_strain=0.0,
        omega=0.0,
        damage=0.0,
        status="ACTIVE",
        initiation_equivalent_plastic_strain=None,
        initiation_equivalent_stress_Pa=None,
        post_initiation_plastic_displacement_m=0.0,
        nominal_equivalent_stress_Pa=0.0,
        damage_dissipation_density_J_per_m3=0.0,
    )


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DamageMaterialError(f"{field} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise DamageMaterialError(f"{field} must be finite and nonnegative")
    return result


def _stress_tensor(value: object) -> np.ndarray:
    try:
        stress = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DamageMaterialError("effective_stress_tensor_Pa must be numeric") from exc
    if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
        raise DamageMaterialError(
            "effective_stress_tensor_Pa must be a finite 3 x 3 tensor"
        )
    scale = max(1.0, float(np.max(np.abs(stress))))
    if not np.allclose(stress, stress.T, rtol=0.0, atol=1.0e-14 * scale):
        raise DamageMaterialError("effective_stress_tensor_Pa must be symmetric")
    return 0.5 * (stress + stress.T)


def _triaxiality(stress: np.ndarray, equivalent_stress_Pa: float) -> float | None:
    tolerance = max(1.0e-7, 1.0e-12 * max(float(np.max(np.abs(stress))), 1.0))
    if equivalent_stress_Pa <= tolerance:
        return None
    return float(np.trace(stress) / 3.0) / equivalent_stress_Pa


def evolve_damage_state(
    damage_input: DamageInput,
    committed: DamageState,
    *,
    equivalent_plastic_strain: object,
    effective_stress_tensor_Pa: object,
    characteristic_length_m: object,
) -> DamageState:
    """Return one immutable trial state from total J2 plastic strain.

    The J2 stress supplied here is the effective (undamaged) stress.  The
    returned nominal equivalent stress follows the frozen linear
    traction--plastic-displacement softening law after initiation.
    """

    if not isinstance(damage_input, DamageInput):
        raise DamageMaterialError("damage_input must be DamageInput")
    if not isinstance(committed, DamageState):
        raise DamageMaterialError("committed must be DamageState")
    alpha = _finite_nonnegative(
        equivalent_plastic_strain, "equivalent_plastic_strain"
    )
    length = _finite_nonnegative(characteristic_length_m, "characteristic_length_m")
    if length <= 0.0:
        raise DamageMaterialError("characteristic_length_m must be positive")
    if alpha + 1.0e-15 < committed.equivalent_plastic_strain:
        raise DamageMaterialError("equivalent_plastic_strain must not decrease")
    stress = _stress_tensor(effective_stress_tensor_Pa)
    q_effective = equivalent_von_mises_stress(stress)
    eta = _triaxiality(stress, q_effective)
    delta_alpha = max(0.0, alpha - committed.equivalent_plastic_strain)

    omega = committed.omega
    initiation_alpha = committed.initiation_equivalent_plastic_strain
    initiation_stress = committed.initiation_equivalent_stress_Pa
    if initiation_alpha is None and delta_alpha > 0.0 and eta is not None:
        if eta >= damage_input.triaxiality_cutoff:
            failure_strain = damage_input.failure_strain_at(eta)
            delta_omega = delta_alpha / failure_strain
            if omega + delta_omega >= 1.0:
                fraction = (1.0 - omega) / delta_omega
                initiation_alpha = committed.equivalent_plastic_strain + fraction * delta_alpha
                initiation_stress = q_effective
                omega = max(1.0, omega + delta_omega)
            else:
                omega += delta_omega

    if initiation_alpha is None or initiation_stress is None:
        return DamageState(
            equivalent_plastic_strain=alpha,
            omega=max(committed.omega, omega),
            damage=committed.damage,
            status=committed.status,
            initiation_equivalent_plastic_strain=None,
            initiation_equivalent_stress_Pa=None,
            post_initiation_plastic_displacement_m=0.0,
            nominal_equivalent_stress_Pa=q_effective,
            damage_dissipation_density_J_per_m3=committed.damage_dissipation_density_J_per_m3,
        )

    terminal_displacement = (
        2.0 * damage_input.fracture_energy_J_per_m2 / initiation_stress
    )
    if not math.isfinite(terminal_displacement) or terminal_displacement <= 0.0:
        raise DamageMaterialError("terminal softening displacement is invalid")
    plastic_displacement = max(0.0, length * (alpha - initiation_alpha))
    softening_fraction = min(
        1.0 - damage_input.damage_event_tolerance,
        plastic_displacement / terminal_displacement,
    )
    target_nominal_stress = initiation_stress * (1.0 - softening_fraction)
    scalar_damage = max(
        committed.damage,
        min(
            1.0 - damage_input.damage_event_tolerance,
            1.0 - target_nominal_stress / max(q_effective, initiation_stress),
        ),
    )
    nominal_stress = (1.0 - scalar_damage) * q_effective
    dissipation = (
        initiation_stress
        * terminal_displacement
        * (softening_fraction - 0.5 * softening_fraction**2)
        / length
    )
    dissipation = max(committed.damage_dissipation_density_J_per_m3, dissipation)
    # Reaching the terminal softening point creates a pending separation
    # event.  Only the topology transaction may commit REMOVED and exclude the
    # background element from the effective system.
    status = "DAMAGED"
    return DamageState(
        equivalent_plastic_strain=alpha,
        omega=max(1.0, committed.omega, omega),
        damage=scalar_damage,
        status=status,
        initiation_equivalent_plastic_strain=initiation_alpha,
        initiation_equivalent_stress_Pa=initiation_stress,
        post_initiation_plastic_displacement_m=plastic_displacement,
        nominal_equivalent_stress_Pa=nominal_stress,
        damage_dissipation_density_J_per_m3=dissipation,
    )
