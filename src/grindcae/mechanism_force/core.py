"""Deterministic Phase 7A.1 interpolation and force decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae import __version__
from grindcae.force_model import GrindingForcePrediction, predict_grinding_force

from .models import MechanismForceCase, MechanismFractions
from .presets import ResolvedMechanismParameters, resolve_mechanism_parameters


RESULT_FORMAT = "grindcae_phase_7a1_mechanism_force_decomposition"


class MechanismForceError(RuntimeError):
    """Raised when valid inputs cannot produce finite mechanism results."""


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismForceError(f"{field}: produced a non-numeric result")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise MechanismForceError(f"{field}: must be finite and non-negative")
    return number


def interpolate_mechanism_fractions(
    eta: float, parameters: ResolvedMechanismParameters
) -> MechanismFractions:
    """Interpolate fractions in log(eta) with componentwise smoothstep."""

    eta_value = _finite_nonnegative(eta, "eta")
    if eta_value <= 0.0:
        raise MechanismForceError("eta: must be strictly greater than zero")
    if not isinstance(parameters, ResolvedMechanismParameters):
        raise MechanismForceError("parameters: wrong resolved parameter type")
    ratios = parameters.anchor_ratios
    fractions = parameters.anchor_fractions
    if eta_value <= ratios[0]:
        return fractions[0]
    if eta_value >= ratios[-1]:
        return fractions[-1]
    upper = next(index for index, ratio in enumerate(ratios) if eta_value < ratio)
    lower = upper - 1
    log_low = math.log(ratios[lower])
    log_high = math.log(ratios[upper])
    t = (math.log(eta_value) - log_low) / (log_high - log_low)
    smooth = 3.0 * t * t - 2.0 * t * t * t
    raw = tuple(
        left + smooth * (right - left)
        for left, right in zip(
            fractions[lower].as_tuple(), fractions[upper].as_tuple(), strict=True
        )
    )
    total = math.fsum(raw)
    normalized = tuple(max(0.0, value / total) for value in raw)
    normalized_total = math.fsum(normalized)
    return MechanismFractions(
        rubbing=normalized[0] / normalized_total,
        ploughing=normalized[1] / normalized_total,
        cutting=normalized[2] / normalized_total,
    )


@dataclass(frozen=True, slots=True)
class MechanismForceComponent:
    fraction: float
    tangential_force_N: float
    normal_force_N: float

    def to_dict(self) -> dict[str, float]:
        return {
            "fraction": self.fraction,
            "tangential_force_N": self.tangential_force_N,
            "normal_force_N": self.normal_force_N,
        }


@dataclass(frozen=True, slots=True)
class MechanismForcePrediction:
    case: MechanismForceCase
    phase4a_prediction: GrindingForcePrediction
    parameters: ResolvedMechanismParameters
    equivalent_process_thickness_m: float
    eta: float
    fractions: MechanismFractions
    rubbing: MechanismForceComponent
    ploughing: MechanismForceComponent
    cutting: MechanismForceComponent
    tangential_force_conservation_residual_N: float
    normal_force_conservation_residual_N: float

    def to_dict(self, artifact_paths: dict[str, str] | None = None) -> dict[str, object]:
        artifacts = artifact_paths or {
            "summary_json": "summary.json",
            "mechanism_fractions_csv": "mechanism_fractions.csv",
            "mechanism_fractions_png": "mechanism_fractions.png",
        }
        return {
            "result_format": RESULT_FORMAT,
            "package_version": __version__,
            "mechanism_force_schema_version": self.case.mechanism_force_schema_version,
            "unit_system": self.case.unit_system,
            "model_type": self.case.model_type,
            "normalized_input": self.case.to_dict(),
            "material": {
                "parameter_source": self.parameters.parameter_source,
                "material_behavior_family": self.parameters.material_behavior_family,
                "material_label": self.parameters.material_label,
                "preset_id": self.parameters.preset_id,
                "calibration_status": self.parameters.calibration_status,
                "provenance": dict(self.parameters.provenance),
            },
            "phase4a_total_force_prediction": self.phase4a_prediction.to_dict(),
            "equivalent_process_thickness": {
                "name": "equivalent_process_thickness_m",
                "display_name_zh": "等效过程厚度指标",
                "value_m": self.equivalent_process_thickness_m,
                "eta": self.eta,
                "equation": "h_eq = vw * ae / vs; eta = h_eq / h_ref",
            },
            "mechanism_parameters": self.parameters.to_dict(),
            "current_mechanism_fractions": self.fractions.to_dict(),
            "mechanism_force_components": {
                "rubbing": self.rubbing.to_dict(),
                "ploughing": self.ploughing.to_dict(),
                "cutting": self.cutting.to_dict(),
            },
            "force_conservation": {
                "tangential_force_conservation_residual_N": self.tangential_force_conservation_residual_N,
                "normal_force_conservation_residual_N": self.normal_force_conservation_residual_N,
            },
            "closure_assumption": {
                "description": "The same mechanism fractions are used for tangential and normal force in this first demonstration closure.",
                "equations": ["Ft_i = alpha_i * Ft_total", "Fn_i = alpha_i * Fn_total"],
            },
            "assumptions_and_cautions": [
                "equivalent_process_thickness_m is a length-scale process indicator used to drive the Phase 7A.1 fraction transition.",
                "It is not a real single-grain undeformed chip thickness and is not experimentally validated as one.",
                "Phase 7A.2 may later introduce abrasive size and effective-grain statistics for a grain-scale treatment.",
                "The same fractions are applied to tangential and normal forces as a transparent first demonstration closure; later experiments may calibrate them separately.",
                "Total Ft, Fn, estimated power, and material removal rate come unchanged from the public Phase 4A force prediction.",
                "The built-in ductile alloy steel values are engineering demonstration assumptions, not experimental facts for a named alloy grade.",
                "Only ductile_metal is supported. SiC and other brittle ceramic behavior are outside Phase 7A.1.",
                "Phase 7A.3 connection to moving-load or elastoplastic-history workflows is not implemented.",
                "No FEM, J2, Newton, GUI, portable-package, thermal coupling, nonuniform load, or statistical abrasive population is produced here.",
            ],
            "artifacts": {
                "summary_json": {"filename": "summary.json", "path": artifacts["summary_json"]},
                "mechanism_fractions_csv": {"filename": "mechanism_fractions.csv", "path": artifacts["mechanism_fractions_csv"]},
                "mechanism_fractions_png": {"filename": "mechanism_fractions.png", "path": artifacts["mechanism_fractions_png"]},
            },
        }


def predict_mechanism_force(case: MechanismForceCase) -> MechanismForcePrediction:
    """Reuse Phase 4A totals and split them without changing the totals."""

    if not isinstance(case, MechanismForceCase):
        raise MechanismForceError("case: predict_mechanism_force requires a validated MechanismForceCase")
    phase4a = predict_grinding_force(case.force_model)
    process = case.force_model.process
    equivalent_thickness = _finite_nonnegative(
        process.workpiece_feed_speed_m_per_s
        * process.depth_of_cut_m
        / process.wheel_surface_speed_m_per_s,
        "equivalent_process_thickness_m",
    )
    parameters = resolve_mechanism_parameters(case.mechanism_model)
    eta = equivalent_thickness / parameters.reference_equivalent_process_thickness_m
    fractions = interpolate_mechanism_fractions(eta, parameters)
    total_tangential = phase4a.tangential_force_magnitude_N
    total_normal = phase4a.normal_force_magnitude_N
    values = fractions.as_tuple()
    tangential = [value * total_tangential for value in values]
    normal = [value * total_normal for value in values]
    components = tuple(
        MechanismForceComponent(fraction, tangential_force, normal_force)
        for fraction, tangential_force, normal_force in zip(values, tangential, normal, strict=True)
    )
    tangential_residual = math.fsum(tangential) - total_tangential
    normal_residual = math.fsum(normal) - total_normal
    return MechanismForcePrediction(
        case=case,
        phase4a_prediction=phase4a,
        parameters=parameters,
        equivalent_process_thickness_m=equivalent_thickness,
        eta=eta,
        fractions=fractions,
        rubbing=components[0],
        ploughing=components[1],
        cutting=components[2],
        tangential_force_conservation_residual_N=tangential_residual,
        normal_force_conservation_residual_N=normal_residual,
    )
