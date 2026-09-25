"""Plane-strain J2 effective response coupled to scalar ductile damage."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from grindcae.elastoplastic import (
    BilinearIsotropicMaterial,
    MaterialPointState,
    PlaneStrainJ2Update,
    update_plane_strain_j2,
)
from grindcae.elastoplastic.j2 import equivalent_von_mises_stress

from .continuous_separation import SeparatingState, advance_separating
from .damage_material import DamageState, evolve_damage_state
from .damage_models import DamageInput


@dataclass(frozen=True, slots=True)
class PlaneStrainJ2DamageUpdate:
    material_update: PlaneStrainJ2Update
    damage_state: DamageState
    nominal_stress_tensor_Pa: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    in_plane_stress_Pa: tuple[float, float, float]
    sigma_zz_Pa: float
    algorithmic_tangent_Pa: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    tangent_branch: str = "undamaged"
    finite_difference_branch_crossings: tuple[bool, bool, bool] = (
        False,
        False,
        False,
    )


@dataclass(frozen=True, slots=True)
class DamageElementTrialResponse:
    total_strain_increment: np.ndarray
    material_update: PlaneStrainJ2DamageUpdate
    internal_force_N: np.ndarray
    tangent_stiffness_N_per_m: np.ndarray
    characteristic_length_m: float


@dataclass(frozen=True, slots=True)
class ConstrainedDamageElementTrialResponse(DamageElementTrialResponse):
    internal_force_damage_derivative_N: np.ndarray
    constraint_residual: float
    constraint_displacement_derivative: np.ndarray
    constraint_damage_derivative: float


@dataclass(frozen=True, slots=True)
class SeparatingElementTrialResponse:
    total_strain_increment: np.ndarray
    material_update: PlaneStrainJ2Update
    damage_state: DamageState
    internal_force_N: np.ndarray
    tangent_stiffness_N_per_m: np.ndarray
    internal_force_progress_derivative_N: np.ndarray
    separation_state: SeparatingState
    characteristic_length_m: float
    energetic_constraint: object | None = None


def _in_plane_nominal_stress(
    material: BilinearIsotropicMaterial,
    damage_input: DamageInput,
    committed_material: MaterialPointState,
    committed_damage: DamageState,
    strain_increment: np.ndarray,
    characteristic_length_m: float,
    active_side_direction: object | None = None,
    *,
    effective_response: PlaneStrainJ2Update | None = None,
) -> tuple[np.ndarray, PlaneStrainJ2Update, DamageState, np.ndarray]:
    effective = effective_response if effective_response is not None else update_plane_strain_j2(
        material, committed_material, strain_increment,
        active_side_direction=active_side_direction,
    )
    effective_tensor = np.asarray(effective.state.stress_tensor_Pa, dtype=float)
    damage = evolve_damage_state(
        damage_input,
        committed_damage,
        equivalent_plastic_strain=effective.state.equivalent_plastic_strain,
        effective_stress_tensor_Pa=effective_tensor,
        characteristic_length_m=characteristic_length_m,
    )
    nominal_tensor = (1.0 - damage.damage) * effective_tensor
    in_plane = np.asarray(
        [nominal_tensor[0, 0], nominal_tensor[1, 1], nominal_tensor[0, 1]],
        dtype=float,
    )
    return in_plane, effective, damage, nominal_tensor


def _failure_strain_and_slope(
    damage_input: DamageInput, triaxiality: float
) -> tuple[float, float]:
    table = damage_input.failure_strain_table
    if triaxiality <= table[0][0]:
        return table[0][1], 0.0
    if triaxiality >= table[-1][0]:
        return table[-1][1], 0.0
    for (eta_0, strain_0), (eta_1, strain_1) in zip(table, table[1:]):
        if triaxiality <= eta_1:
            slope = (strain_1 - strain_0) / (eta_1 - eta_0)
            return strain_0 + slope * (triaxiality - eta_0), slope
    raise RuntimeError("failure-strain interpolation interval was not found")


def _effective_invariants_and_gradients(
    effective: PlaneStrainJ2Update,
) -> tuple[float, float | None, np.ndarray, np.ndarray]:
    stress = np.asarray(effective.state.stress_tensor_Pa, dtype=float)
    q_effective = equivalent_von_mises_stress(stress)
    stress_gradients = tuple(
        np.asarray(column, dtype=float)
        for column in effective.stress_tensor_tangent_Pa
    )
    pressure = float(np.trace(stress)) / 3.0
    pressure_gradient = np.asarray(
        [float(np.trace(column)) / 3.0 for column in stress_gradients],
        dtype=float,
    )
    if q_effective <= max(1.0e-7, 1.0e-12 * max(np.max(np.abs(stress)), 1.0)):
        return q_effective, None, np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    deviatoric = stress - pressure * np.eye(3)
    q_gradient = np.asarray(
        [1.5 * float(np.sum(deviatoric * column)) / q_effective for column in stress_gradients],
        dtype=float,
    )
    eta = pressure / q_effective
    eta_gradient = (
        pressure_gradient * q_effective - pressure * q_gradient
    ) / q_effective**2
    return q_effective, eta, q_gradient, eta_gradient


def _active_damage_tangent(
    damage_input: DamageInput,
    committed: DamageState,
    effective: PlaneStrainJ2Update,
    damage: DamageState,
    characteristic_length_m: float,
) -> tuple[np.ndarray, str]:
    effective_tangent = np.asarray(effective.algorithmic_tangent_Pa, dtype=float)
    effective_stress = np.asarray(effective.in_plane_stress_Pa, dtype=float)
    alpha_gradient = np.asarray(
        effective.equivalent_plastic_strain_gradient, dtype=float
    )
    q_effective, eta, q_gradient, eta_gradient = (
        _effective_invariants_and_gradients(effective)
    )
    cap = 1.0 - damage_input.damage_event_tolerance
    fixed_tangent = (1.0 - damage.damage) * effective_tangent

    if damage.initiation_equivalent_plastic_strain is None:
        branch = "undamaged"
        if damage.omega >= 1.0 - damage_input.damage_event_tolerance:
            branch = "initiation_lower_side"
        return fixed_tangent, branch

    if damage.damage >= cap:
        return fixed_tangent, "separation_clamped"

    initiation_alpha = damage.initiation_equivalent_plastic_strain
    initiation_stress = damage.initiation_equivalent_stress_Pa
    assert initiation_alpha is not None and initiation_stress is not None

    if committed.initiation_equivalent_plastic_strain is None:
        if eta is None:
            return fixed_tangent, "softening_frozen"
        failure_strain, failure_slope = _failure_strain_and_slope(
            damage_input, eta
        )
        initiation_alpha_gradient = (
            (1.0 - committed.omega) * failure_slope * eta_gradient
        )
        post_initiation_alpha = (
            effective.state.equivalent_plastic_strain - initiation_alpha
        )
        softening_gradient = characteristic_length_m / (
            2.0 * damage_input.fracture_energy_J_per_m2
        ) * (
            q_effective * (alpha_gradient - initiation_alpha_gradient)
            + post_initiation_alpha * q_gradient
        )
        unconstrained_damage = (
            characteristic_length_m
            * q_effective
            * post_initiation_alpha
            / (2.0 * damage_input.fracture_energy_J_per_m2)
        )
        if unconstrained_damage <= committed.damage:
            return fixed_tangent, "softening_frozen"
        tangent = (
            (1.0 - damage.damage) * effective_tangent
            - np.outer(effective_stress, softening_gradient)
        )
        return tangent, "initiation_upper_side"

    terminal_displacement = (
        2.0 * damage_input.fracture_energy_J_per_m2 / initiation_stress
    )
    softening_fraction = (
        characteristic_length_m
        * (effective.state.equivalent_plastic_strain - initiation_alpha)
        / terminal_displacement
    )
    target_nominal_stress = initiation_stress * (1.0 - softening_fraction)
    target_gradient = (
        -initiation_stress
        * characteristic_length_m
        / terminal_displacement
        * alpha_gradient
    )
    if q_effective >= initiation_stress:
        denominator = q_effective
        denominator_gradient = q_gradient
    else:
        denominator = initiation_stress
        denominator_gradient = np.zeros(3, dtype=float)
    unconstrained_damage = 1.0 - target_nominal_stress / denominator
    if unconstrained_damage <= committed.damage:
        return fixed_tangent, "softening_frozen"
    degradation = target_nominal_stress / denominator
    degradation_gradient = (
        target_gradient * denominator
        - target_nominal_stress * denominator_gradient
    ) / denominator**2
    tangent = degradation * effective_tangent + np.outer(
        effective_stress, degradation_gradient
    )
    if cap - damage.damage <= damage_input.damage_event_tolerance:
        return tangent, "separation_lower_side"
    return tangent, "softening_evolving"


def _damage_branch(
    damage_input: DamageInput,
    committed: DamageState,
    damage: DamageState,
) -> str:
    if damage.initiation_equivalent_plastic_strain is None:
        return "pre_initiation"
    if damage.damage >= 1.0 - damage_input.damage_event_tolerance:
        return "terminal"
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(committed.damage))
    if damage.damage <= committed.damage + tolerance:
        return "softening_frozen"
    return "softening_evolving"


def update_plane_strain_j2_damage(
    material: BilinearIsotropicMaterial,
    damage_input: DamageInput,
    committed_material: MaterialPointState,
    committed_damage: DamageState,
    plane_strain_increment: object,
    *,
    characteristic_length_m: object,
    active_side_direction: object | None = None,
) -> PlaneStrainJ2DamageUpdate:
    """Return the nominal stress and consistent local tangent.

    The retained J2 kernel remains the effective-stress model.  The nominal
    tensor is degraded isotropically. Analytical branch derivatives supply the
    local tangent; finite differences are reserved for independent verification.
    """

    increment = np.asarray(plane_strain_increment, dtype=float)
    if increment.shape != (3,) or not np.all(np.isfinite(increment)):
        raise ValueError("plane_strain_increment must be a finite length-3 vector")
    length = float(characteristic_length_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("characteristic_length_m must be finite and positive")
    if committed_damage.status == "REMOVED":
        retained = update_plane_strain_j2(
            material, committed_material, np.zeros(3, dtype=float)
        )
        zero = ((0.0, 0.0, 0.0),) * 3
        return PlaneStrainJ2DamageUpdate(
            material_update=retained,
            damage_state=committed_damage,
            nominal_stress_tensor_Pa=zero,
            in_plane_stress_Pa=(0.0, 0.0, 0.0),
            sigma_zz_Pa=0.0,
            algorithmic_tangent_Pa=zero,
            tangent_branch="removed",
        )

    onset = None
    effective_endpoint = None
    if (damage_input.evolution == "energetic_rational_fracture"
            and getattr(committed_damage, "energetic_history", None) is None):
        from dataclasses import replace
        from .energetic_damage import locate_plastic_initiation, EnergeticDamageHistory
        # Reuse only this exact no-side endpoint within the current trial.
        # Fractional onset roots and a requested one-sided tangent still use
        # their own material returns; nothing is cached across Newton trials.
        if active_side_direction is None:
            effective_endpoint = update_plane_strain_j2(material, committed_material, increment)
        onset = locate_plastic_initiation(material, committed_material, damage_input,
                                         increment, omega_before=committed_damage.omega,
                                         _endpoint_response=effective_endpoint)
        if onset is not None:
            committed_damage = replace(committed_damage,omega=1.0,
                initiation_equivalent_plastic_strain=onset['material_state'].equivalent_plastic_strain,
                initiation_equivalent_stress_Pa=equivalent_von_mises_stress(
                    np.asarray(onset['material_state'].stress_tensor_Pa)),
                energetic_history=EnergeticDamageHistory(0.,onset['Y0'],
                    damage_input.fracture_energy_J_per_m2/length))
        elif active_side_direction is not None:
            from .thermodynamic_energy import DamagePathConstraintRequired
            effective_trial=update_plane_strain_j2(
                material,committed_material,increment,
                active_side_direction=active_side_direction)
            if (effective_trial.state.equivalent_plastic_strain
                    > committed_material.equivalent_plastic_strain
                    and committed_damage.omega>=1.-damage_input.damage_event_tolerance):
                raise DamagePathConstraintRequired(
                    'energetic onset requires joint event localization')

    if getattr(committed_damage, "energetic_history", None) is not None:
        if committed_damage.energetic_history.damage != committed_damage.damage:
            raise ValueError("energetic history disagrees with committed material damage")
        from dataclasses import replace
        from .energetic_damage import update as energetic_update
        response = energetic_update(material, committed_material,
                                    committed_damage.energetic_history, increment,
                                    active_side_direction=active_side_direction)
        tangent = response.tangent
        if onset is not None and 0. < response.history.damage < 1.:
            # D=(1-sqrt(Y0/Y))/(1-Y0/G); Y0 varies with the
            # localized onset position during this uncommitted increment.
            from .thermodynamic_energy import stored_energy
            Y = stored_energy(response.material_update.state.stress_tensor_Pa,
                response.material_update.state.equivalent_plastic_strain,0.,
                young_modulus=material.E,poisson_ratio=material.nu,
                hardening_modulus=material.internal_hardening_modulus)['effective_total']
            Y0=onset['Y0']; G=response.history.fracture_energy_density
            root=math.sqrt(Y0/Y); c=1.-Y0/G
            derivative=(-root*c/(2.*Y0)+(1.-root)/G)/c**2
            tangent=tangent-np.outer(response.material_update.in_plane_stress_Pa,
                                    derivative*onset['Y0_gradient'])
        effective = response.material_update
        tensor = (1.0-response.history.damage)*np.asarray(effective.state.stress_tensor_Pa)
        damage = replace(committed_damage,
            equivalent_plastic_strain=effective.state.equivalent_plastic_strain,
            damage=response.history.damage,
            status="DAMAGED",
            nominal_equivalent_stress_Pa=equivalent_von_mises_stress(tensor),
            damage_dissipation_density_J_per_m3=response.damage_dissipation_density,
            energetic_history=response.history)
        return PlaneStrainJ2DamageUpdate(
            material_update=effective,damage_state=damage,
            nominal_stress_tensor_Pa=tuple(tuple(float(v) for v in row) for row in tensor),
            in_plane_stress_Pa=tuple(float(v) for v in response.stress),
            sigma_zz_Pa=float(tensor[2,2]),
            algorithmic_tangent_Pa=tuple(tuple(float(v) for v in row) for row in tangent),
            tangent_branch="energetic_damage")

    in_plane, effective, damage, nominal_tensor = _in_plane_nominal_stress(
        material,
        damage_input,
        committed_material,
        committed_damage,
        increment,
        length,
        active_side_direction=active_side_direction,
        effective_response=effective_endpoint,
    )
    tangent, tangent_branch = _active_damage_tangent(
        damage_input,
        committed_damage,
        effective,
        damage,
        length,
    )
    branch_crossings = (False, False, False)
    return PlaneStrainJ2DamageUpdate(
        material_update=effective,
        damage_state=damage,
        nominal_stress_tensor_Pa=tuple(
            tuple(float(value) for value in row) for row in nominal_tensor
        ),
        in_plane_stress_Pa=tuple(float(value) for value in in_plane),
        sigma_zz_Pa=float(nominal_tensor[2, 2]),
        algorithmic_tangent_Pa=tuple(
            tuple(float(value) for value in row) for row in tangent
        ),
        tangent_branch=tangent_branch,
        finite_difference_branch_crossings=branch_crossings,
    )


def damage_element_trial_response(
    material: BilinearIsotropicMaterial,
    damage_input: DamageInput,
    committed_material: MaterialPointState,
    committed_damage: DamageState,
    B: object,
    area_m2: object,
    *,
    thickness_m: object,
    displacement_increment: object,
    active_side_direction: object | None = None,
) -> DamageElementTrialResponse:
    """Evaluate one constant-strain triangle with frozen damage regularization."""

    matrix = np.asarray(B, dtype=float)
    displacement = np.asarray(displacement_increment, dtype=float)
    area = float(area_m2)
    thickness = float(thickness_m)
    if matrix.shape != (3, 6) or displacement.shape != (6,):
        raise ValueError("element B matrix and displacement increment shapes are invalid")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("area_m2 must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness_m must be finite and positive")
    characteristic_length = math.sqrt(4.0 * area / math.pi)
    strain_increment = matrix @ displacement
    update = update_plane_strain_j2_damage(
        material,
        damage_input,
        committed_material,
        committed_damage,
        strain_increment,
        characteristic_length_m=characteristic_length,
        active_side_direction=(None if active_side_direction is None else matrix@np.asarray(active_side_direction)),
    )
    stress = np.asarray(update.in_plane_stress_Pa, dtype=float)
    tangent = np.asarray(update.algorithmic_tangent_Pa, dtype=float)
    weight = area * thickness
    return DamageElementTrialResponse(
        total_strain_increment=strain_increment,
        material_update=update,
        internal_force_N=matrix.T @ stress * weight,
        tangent_stiffness_N_per_m=matrix.T @ tangent @ matrix * weight,
        characteristic_length_m=characteristic_length,
    )


def constrained_damage_element_trial_response(
    material: BilinearIsotropicMaterial,
    damage_input: DamageInput,
    committed_material: MaterialPointState,
    committed_damage: DamageState,
    B: object,
    area_m2: object,
    *,
    thickness_m: object,
    displacement_increment: object,
    damage: object,
    active_side_direction: object | None = None,
) -> ConstrainedDamageElementTrialResponse:
    """Evaluate one energetic element with damage as a joint unknown."""

    from dataclasses import replace
    from .energetic_damage import constrained_update

    matrix = np.asarray(B, dtype=float)
    displacement = np.asarray(displacement_increment, dtype=float)
    area = float(area_m2)
    thickness = float(thickness_m)
    damage_value = float(damage)
    history = getattr(committed_damage, "energetic_history", None)
    if matrix.shape != (3, 6) or displacement.shape != (6,):
        raise ValueError("element B matrix and displacement increment shapes are invalid")
    if not math.isfinite(area) or area <= 0.0 or not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("area and thickness must be finite and positive")
    if history is None:
        raise ValueError("joint damage coordinates require energetic history")
    characteristic_length = math.sqrt(4.0 * area / math.pi)
    strain_increment = matrix @ displacement
    joint = constrained_update(
        material,
        committed_material,
        history,
        strain_increment,
        damage_value,
        active_side_direction=(
            None
            if active_side_direction is None
            else matrix @ np.asarray(active_side_direction, dtype=float)
        ),
    )
    tensor = (1.0 - damage_value) * np.asarray(
        joint.material_update.state.stress_tensor_Pa, dtype=float
    )
    damage_state = replace(
        committed_damage,
        equivalent_plastic_strain=joint.material_update.state.equivalent_plastic_strain,
        damage=damage_value,
        status="DAMAGED",
        nominal_equivalent_stress_Pa=equivalent_von_mises_stress(tensor),
        damage_dissipation_density_J_per_m3=joint.damage_dissipation_density,
        energetic_history=joint.history,
    )
    weight = area * thickness
    return ConstrainedDamageElementTrialResponse(
        total_strain_increment=strain_increment,
        material_update=PlaneStrainJ2DamageUpdate(
            material_update=joint.material_update,
            damage_state=damage_state,
            nominal_stress_tensor_Pa=tuple(
                tuple(float(value) for value in row) for row in tensor
            ),
            in_plane_stress_Pa=tuple(float(value) for value in joint.stress),
            sigma_zz_Pa=float(tensor[2, 2]),
            algorithmic_tangent_Pa=tuple(
                tuple(float(value) for value in row) for row in joint.tangent
            ),
            tangent_branch="energetic_damage_constrained",
        ),
        internal_force_N=matrix.T @ joint.stress * weight,
        tangent_stiffness_N_per_m=matrix.T @ joint.tangent @ matrix * weight,
        characteristic_length_m=characteristic_length,
        internal_force_damage_derivative_N=(
            matrix.T @ joint.stress_damage_derivative * weight
        ),
        constraint_residual=float(joint.constraint_residual),
        constraint_displacement_derivative=(
            joint.constraint_strain_gradient @ matrix
        ),
        constraint_damage_derivative=float(joint.constraint_damage_derivative),
    )


def constrained_damage_onset_element_trial_response(
    material: BilinearIsotropicMaterial,
    damage_input: DamageInput,
    committed_material: MaterialPointState,
    committed_damage: DamageState,
    B: object,
    area_m2: object,
    *,
    thickness_m: object,
    displacement_increment: object,
    damage: object,
    active_side_direction: object | None = None,
) -> ConstrainedDamageElementTrialResponse:
    """Evaluate a joint energetic coordinate whose onset is inside the step."""

    from dataclasses import replace
    from .energetic_damage import constrained_onset_update

    matrix=np.asarray(B,dtype=float); displacement=np.asarray(displacement_increment,dtype=float)
    area=float(area_m2); thickness=float(thickness_m); damage_value=float(damage)
    if matrix.shape!=(3,6) or displacement.shape!=(6,) or area<=0. or thickness<=0.:
        raise ValueError('invalid joint onset element input')
    length=math.sqrt(4.*area/math.pi); strain_increment=matrix@displacement
    joint=constrained_onset_update(
        material,committed_material,committed_damage,damage_input,strain_increment,damage_value,
        fracture_energy_density=damage_input.fracture_energy_J_per_m2/length,
        active_side_direction=(None if active_side_direction is None
            else matrix@np.asarray(active_side_direction,dtype=float)))
    tensor=(1.-damage_value)*np.asarray(joint.material_update.state.stress_tensor_Pa)
    damage_state=replace(committed_damage,
        equivalent_plastic_strain=joint.material_update.state.equivalent_plastic_strain,
        omega=1.,damage=damage_value,status='DAMAGED',
        initiation_equivalent_plastic_strain=joint.onset['material_state'].equivalent_plastic_strain,
        initiation_equivalent_stress_Pa=equivalent_von_mises_stress(
            np.asarray(joint.onset['material_state'].stress_tensor_Pa)),
        nominal_equivalent_stress_Pa=equivalent_von_mises_stress(tensor),
        damage_dissipation_density_J_per_m3=joint.damage_dissipation_density,
        energetic_history=joint.history)
    weight=area*thickness
    return ConstrainedDamageElementTrialResponse(
        strain_increment,PlaneStrainJ2DamageUpdate(
            joint.material_update,damage_state,
            tuple(tuple(float(value) for value in row) for row in tensor),
            tuple(float(value) for value in joint.stress),float(tensor[2,2]),
            tuple(tuple(float(value) for value in row) for row in joint.tangent),
            tangent_branch='energetic_damage_onset_constrained'),
        matrix.T@joint.stress*weight,matrix.T@joint.tangent@matrix*weight,length,
        matrix.T@joint.stress_damage_derivative*weight,float(joint.constraint_residual),
        joint.constraint_strain_gradient@matrix,float(joint.constraint_damage_derivative))


def separating_element_trial_response(
    material: BilinearIsotropicMaterial,
    committed_material: MaterialPointState,
    committed_damage: DamageState,
    committed_separation: SeparatingState,
    B: object,
    area_m2: object,
    *,
    thickness_m: object,
    displacement_increment: object,
    separation_progress: object,
    active_side_direction: object | None = None,
) -> SeparatingElementTrialResponse:
    """Evaluate one triangle on the existing softening law's terminal segment."""

    matrix = np.asarray(B, dtype=float)
    displacement = np.asarray(displacement_increment, dtype=float)
    area = float(area_m2)
    thickness = float(thickness_m)
    progress = float(separation_progress)
    if matrix.shape != (3, 6) or displacement.shape != (6,):
        raise ValueError("element B matrix and displacement increment shapes are invalid")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("area_m2 must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness_m must be finite and positive")
    separation = advance_separating(committed_separation, s_e=progress)
    strain_increment = matrix @ displacement
    current_side = (committed_separation.entry_strain_direction
        if active_side_direction is None else matrix @ np.asarray(active_side_direction,dtype=float))
    if getattr(committed_separation,"energetic_history",None) is not None:
        from dataclasses import replace
        if committed_separation.energetic_history.damage==1.:
            # Natural inertial completion has already frozen effective J2
            # history. A zero-capacity metadata label cannot reopen it.
            from .energetic_damage import update
            retained=update(material,committed_material,committed_damage.energetic_history,
                            strain_increment)
            return SeparatingElementTrialResponse(
                total_strain_increment=strain_increment,material_update=retained.material_update,
                damage_state=committed_damage,internal_force_N=np.zeros(6),
                tangent_stiffness_N_per_m=np.zeros((6,6)),
                internal_force_progress_derivative_N=np.zeros(6),separation_state=separation,
                characteristic_length_m=committed_separation.characteristic_length_m)
        from .energetic_damage import constrained_update
        joint=constrained_update(material,committed_material,committed_damage.energetic_history,
            strain_increment,separation.damage,active_side_direction=current_side)
        remaining_damage=1.-committed_separation.energetic_history.damage
        weight=area*thickness
        nominal_q=equivalent_von_mises_stress(
            (1.-separation.damage)*np.asarray(joint.material_update.state.stress_tensor_Pa))
        separation=replace(separation,nominal_equivalent_stress_Pa=nominal_q,
            effective_equivalent_stress_Pa=equivalent_von_mises_stress(
                np.asarray(joint.material_update.state.stress_tensor_Pa)))
        damage_state=replace(committed_damage,damage=separation.damage,status="SEPARATING",
            equivalent_plastic_strain=joint.material_update.state.equivalent_plastic_strain,
            nominal_equivalent_stress_Pa=nominal_q,energetic_history=joint.history,
            damage_dissipation_density_J_per_m3=joint.damage_dissipation_density)
        return SeparatingElementTrialResponse(
            total_strain_increment=strain_increment,material_update=joint.material_update,
            damage_state=damage_state,internal_force_N=matrix.T@joint.stress*weight,
            tangent_stiffness_N_per_m=matrix.T@joint.tangent@matrix*weight,
            internal_force_progress_derivative_N=matrix.T@joint.stress_damage_derivative*remaining_damage*weight,
            separation_state=separation,characteristic_length_m=committed_separation.characteristic_length_m,
            energetic_constraint=joint)
    effective = update_plane_strain_j2(
        material,
        committed_material,
        strain_increment,
        active_side_direction=current_side,
    )
    effective_tensor = np.asarray(effective.state.stress_tensor_Pa, dtype=float)
    effective_in_plane = np.asarray(effective.in_plane_stress_Pa, dtype=float)
    effective_tangent = np.asarray(effective.algorithmic_tangent_Pa, dtype=float)
    q_effective, _, q_gradient, _ = _effective_invariants_and_gradients(effective)
    denominator = max(q_effective, separation.initiation_equivalent_stress_Pa)
    weight = area * thickness
    if progress == 1.0:
        nominal_in_plane = np.zeros(3, dtype=float)
        tangent = np.zeros((3, 3), dtype=float)
        progress_derivative = np.zeros(3, dtype=float)
        scalar_damage = 1.0
    else:
        target = separation.nominal_equivalent_stress_Pa
        degradation = target / denominator
        nominal_in_plane = degradation * effective_in_plane
        if q_effective >= separation.initiation_equivalent_stress_Pa:
            degradation_gradient = -target * q_gradient / denominator**2
        else:
            degradation_gradient = np.zeros(3, dtype=float)
        tangent = degradation * effective_tangent + np.outer(
            effective_in_plane, degradation_gradient
        )
        target_progress_derivative = (
            -separation.initiation_equivalent_stress_Pa
            * (1.0 - separation.entry_softening_fraction)
        )
        progress_derivative = (
            target_progress_derivative / denominator * effective_in_plane
        )
        scalar_damage = 1.0 - degradation
    damage_state = DamageState(
        equivalent_plastic_strain=effective.state.equivalent_plastic_strain,
        omega=max(1.0, committed_damage.omega),
        damage=scalar_damage,
        status="SEPARATING",
        initiation_equivalent_plastic_strain=(
            committed_damage.initiation_equivalent_plastic_strain
        ),
        initiation_equivalent_stress_Pa=separation.initiation_equivalent_stress_Pa,
        post_initiation_plastic_displacement_m=(
            separation.terminal_displacement_m
            * (
                separation.entry_softening_fraction
                + progress * (1.0 - separation.entry_softening_fraction)
            )
        ),
        nominal_equivalent_stress_Pa=separation.nominal_equivalent_stress_Pa,
        damage_dissipation_density_J_per_m3=(
            separation.damage_dissipation_density_J_per_m3
        ),
    )
    return SeparatingElementTrialResponse(
        total_strain_increment=strain_increment,
        material_update=effective,
        damage_state=damage_state,
        internal_force_N=matrix.T @ nominal_in_plane * weight,
        tangent_stiffness_N_per_m=matrix.T @ tangent @ matrix * weight,
        internal_force_progress_derivative_N=(
            matrix.T @ progress_derivative * weight
        ),
        separation_state=separation,
        characteristic_length_m=committed_separation.characteristic_length_m,
    )


__all__ = [
    "DamageElementTrialResponse",
    "PlaneStrainJ2DamageUpdate",
    "SeparatingElementTrialResponse",
    "damage_element_trial_response",
    "separating_element_trial_response",
    "update_plane_strain_j2_damage",
]
