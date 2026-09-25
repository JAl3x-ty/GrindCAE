"""Transparent phase 4A specific-energy grinding-force calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae import __version__

from .models import GrindingForceCase


class ForceModelError(RuntimeError):
    """Raised when valid finite inputs cannot produce finite model results."""


def _checked_result(value: object, field: str, unit: str, operation: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForceModelError(
            f"{field} [{unit}]: {operation} produced a non-numeric result"
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ForceModelError(
            f"{field} [{unit}]: {operation} produced an unrepresentable result"
        ) from exc
    if not math.isfinite(number):
        raise ForceModelError(
            f"{field} [{unit}]: finite inputs overflowed during {operation}"
        )
    if number < 0.0:
        raise ForceModelError(
            f"{field} [{unit}]: {operation} produced an invalid negative result"
        )
    return number


@dataclass(frozen=True, slots=True)
class GrindingForcePrediction:
    """Calculated non-negative force magnitudes and supporting quantities."""

    force_case: GrindingForceCase
    material_removal_rate_m3_per_s: float
    estimated_grinding_power_W: float
    tangential_force_magnitude_N: float
    normal_force_magnitude_N: float
    resultant_force_magnitude_N: float
    geometric_contact_length_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.force_case, GrindingForceCase):
            raise ForceModelError(
                "force_case [GrindingForceCase]: prediction has the wrong input type"
            )
        quantities = (
            ("material_removal_rate_m3_per_s", "m^3/s"),
            ("estimated_grinding_power_W", "W"),
            ("tangential_force_magnitude_N", "N"),
            ("normal_force_magnitude_N", "N"),
            ("resultant_force_magnitude_N", "N"),
            ("geometric_contact_length_m", "m"),
        )
        for name, unit in quantities:
            value = _checked_result(
                getattr(self, name), name, unit, "prediction validation"
            )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic phase 4A JSON payload."""

        return {
            "result_format": "grindcae_phase_4a_force_prediction",
            "package_version": __version__,
            "force_model_schema_version": (
                self.force_case.force_model_schema_version
            ),
            "unit_system": self.force_case.unit_system,
            "model_type": self.force_case.model_type,
            "normalized_input": self.force_case.to_dict(),
            "model_equations": {
                "material_removal_rate": "Qw = vw * ae * b",
                "estimated_grinding_power": "P = us * Qw",
                "tangential_force_magnitude": "Ft = P / vs",
                "normal_force_magnitude": "Fn = Rnt * Ft",
                "resultant_force_magnitude": "Fr = hypot(Ft, Fn)",
                "geometric_contact_length": "lg = sqrt(Ds * ae)",
            },
            "derived_quantities": {
                "material_removal_rate_m3_per_s": (
                    self.material_removal_rate_m3_per_s
                ),
                "estimated_grinding_power_W": self.estimated_grinding_power_W,
                "geometric_contact_length_m": self.geometric_contact_length_m,
            },
            "predicted_force_magnitudes": {
                "tangential_force_magnitude_N": (
                    self.tangential_force_magnitude_N
                ),
                "normal_force_magnitude_N": self.normal_force_magnitude_N,
                "resultant_force_magnitude_N": self.resultant_force_magnitude_N,
            },
            "calibration_provenance": (
                self.force_case.calibration.provenance_dict()
            ),
            "assumptions_and_cautions": [
                "This is a basic empirical estimate using specific grinding energy and a force ratio.",
                "The user-supplied us and Rnt values require calibration for the applicable process using experiments or reliable literature.",
                "Calibration can depend on workpiece material; wheel abrasive, grit, bond, and structure; dressing state; coolant and lubrication; wheel and workpiece speeds; grinding depth and width; machine behavior; and force or power measurement conditions.",
                "The current model returns non-negative force magnitudes only.",
                "Finite-element coordinate directions, force signs, and a loaded boundary are not defined in phase 4A.",
                "The geometric contact length lg = sqrt(Ds * ae) is a shallow-cut approximation intended for ae much smaller than Ds.",
                "This command does not run meshing, finite-element solving, or post-processing.",
                "The result is not experimentally validated and must not be presented directly as the true grinding force.",
                "Thermal effects, wheel wear, abrasive-grain randomness, ploughing, chatter, transient behavior, and true contact are not represented.",
            ],
        }


def predict_grinding_force(force_case: GrindingForceCase) -> GrindingForcePrediction:
    """Evaluate the phase 4A formulas without invoking the FEM workflow."""

    if not isinstance(force_case, GrindingForceCase):
        raise ForceModelError(
            "force_case [GrindingForceCase]: predict_grinding_force requires a validated force-model case"
        )
    process = force_case.process
    calibration = force_case.calibration

    # 中文导读：沿“材料去除率 -> 功率 -> 切向力 -> 法向力”逐级计算。
    feed_depth = _checked_result(
        process.workpiece_feed_speed_m_per_s * process.depth_of_cut_m,
        "material_removal_rate_m3_per_s",
        "m^3/s",
        "vw * ae",
    )
    material_removal_rate = _checked_result(
        feed_depth * process.grinding_width_m,
        "material_removal_rate_m3_per_s",
        "m^3/s",
        "vw * ae * b",
    )
    estimated_power = _checked_result(
        calibration.specific_grinding_energy_J_per_m3 * material_removal_rate,
        "estimated_grinding_power_W",
        "W",
        "us * Qw",
    )
    tangential_force = _checked_result(
        estimated_power / process.wheel_surface_speed_m_per_s,
        "tangential_force_magnitude_N",
        "N",
        "P / vs",
    )
    normal_force = _checked_result(
        calibration.normal_to_tangential_force_ratio * tangential_force,
        "normal_force_magnitude_N",
        "N",
        "Rnt * Ft",
    )
    resultant_force = _checked_result(
        math.hypot(tangential_force, normal_force),
        "resultant_force_magnitude_N",
        "N",
        "hypot(Ft, Fn)",
    )
    wheel_depth = _checked_result(
        process.wheel_diameter_m * process.depth_of_cut_m,
        "geometric_contact_length_m",
        "m",
        "Ds * ae",
    )
    contact_length = _checked_result(
        math.sqrt(wheel_depth),
        "geometric_contact_length_m",
        "m",
        "sqrt(Ds * ae)",
    )

    return GrindingForcePrediction(
        force_case=force_case,
        material_removal_rate_m3_per_s=material_removal_rate,
        estimated_grinding_power_W=estimated_power,
        tangential_force_magnitude_N=tangential_force,
        normal_force_magnitude_N=normal_force,
        resultant_force_magnitude_N=resultant_force,
        geometric_contact_length_m=contact_length,
    )
