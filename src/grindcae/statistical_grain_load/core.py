"""Public Phase 7A.2 statistical equivalent-grain load prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae import __version__
from grindcae.mechanism_force import MechanismForcePrediction, predict_mechanism_force

from .grain_population import EquivalentGrainGroup, build_equivalent_grain_groups
from .grit_resolution import ResolvedGritSpecification, resolve_grit_specification
from .load_distribution import MechanismLoadPoint, build_load_distribution, trapezoidal_integral
from .models import StatisticalGrainLoadCase


RESULT_FORMAT = "grindcae_phase_7a2_statistical_equivalent_grain_mechanism_load"


@dataclass(frozen=True, slots=True)
class StatisticalGrainLoadPrediction:
    case: StatisticalGrainLoadCase
    mechanism_prediction: MechanismForcePrediction
    resolved_grit_specification: ResolvedGritSpecification
    contact_length_m: float
    grinding_width_m: float
    active_grain_surface_density_per_m2: float
    estimated_physical_active_grain_count: float
    equivalent_group_count: int
    compression_ratio: float
    groups: tuple[EquivalentGrainGroup, ...]
    distribution: tuple[MechanismLoadPoint, ...]
    mechanism_force_residuals_N: dict[str, float]
    total_tangential_force_residual_N: float
    total_normal_force_residual_N: float

    def to_dict(self, artifact_paths: dict[str, str] | None = None) -> dict[str, object]:
        artifacts = artifact_paths or {
            "summary_json": "summary.json",
            "resolved_grit_specification_json": "resolved_grit_specification.json",
            "equivalent_grain_groups_csv": "equivalent_grain_groups.csv",
            "mechanism_load_distribution_csv": "mechanism_load_distribution.csv",
            "mechanism_load_distribution_png": "mechanism_load_distribution.png",
        }
        return {
            "result_format": RESULT_FORMAT,
            "package_version": __version__,
            "statistical_grain_load_schema_version": self.case.statistical_grain_load_schema_version,
            "unit_system": self.case.unit_system,
            "model_type": self.case.model_type,
            "normalized_input": self.case.to_dict(),
            "phase7a1_mechanism_force_prediction": self.mechanism_prediction.to_dict(),
            "resolved_grit_specification": self.resolved_grit_specification.to_dict(),
            "grain_population": {
                "contact_length_m": self.contact_length_m,
                "grinding_width_m": self.grinding_width_m,
                "active_density_factor": self.case.grain_population.active_density_factor,
                "active_density_factor_status": "not_experimentally_calibrated",
                "active_grain_surface_density_per_m2": self.active_grain_surface_density_per_m2,
                "estimated_physical_active_grain_count": self.estimated_physical_active_grain_count,
                "equivalent_group_count": self.equivalent_group_count,
                "compression_ratio": self.compression_ratio,
                "random_seed": self.case.grain_population.random_seed,
            },
            "statistical_closure": {
                "status": "demonstration_statistical_closure",
                "mechanism_diameter_exponents": {"rubbing": 0.0, "ploughing": 1.0, "cutting": 2.0},
                "kernel": "truncated-to-contact-domain Gaussian-like sampled kernels, normalized by deterministic trapezoidal integration",
            },
            "force_conservation": {
                "mechanism_force_residuals_N": dict(self.mechanism_force_residuals_N),
                "total_tangential_force_residual_N": self.total_tangential_force_residual_N,
                "total_normal_force_residual_N": self.total_normal_force_residual_N,
            },
            "provenance": {
                "grit_data_status": self.resolved_grit_specification.data_status,
                "grit_source": self.resolved_grit_specification.source,
                "population_model": "C_active = k_active / d50^2; N_active = C_active * b * lg",
            },
            "assumptions_and_cautions": [
                "N_active is a statistical estimate, not an observed physical grain count.",
                "The active_density_factor is an uncalibrated engineering demonstration value.",
                "Equivalent groups compress the estimated physical population and are not explicit real grain geometries.",
                "The line loads are non-negative local magnitudes in N/m; the separate Phase 7A.3 route applies global signs and conservative FEM boundary projection.",
                "Grinding width is used only in N_active. Load integration does not multiply width or analysis thickness again.",
                "GUI updates and Windows portable-package updates remain outside the Phase 7A.2 and Phase 7A.3 calculation-kernel routes.",
            ],
            "artifacts": {
                key: {"filename": value.split("\\")[-1].split("/")[-1], "path": value}
                for key, value in artifacts.items()
            },
        }


def _force_residuals(
    mechanism: MechanismForcePrediction,
    distribution: tuple[MechanismLoadPoint, ...],
) -> tuple[dict[str, float], float, float]:
    x = tuple(point.x_local_m for point in distribution)
    residuals: dict[str, float] = {}
    for direction, force_name in (("t", "tangential_force_N"), ("n", "normal_force_N")):
        for name in ("rubbing", "ploughing", "cutting"):
            integral = trapezoidal_integral(
                tuple(getattr(point, f"q_{direction}_{name}_N_per_m") for point in distribution), x
            )
            residuals[f"{direction}_{name}"] = integral - getattr(getattr(mechanism, name), force_name)
    total_t = trapezoidal_integral(tuple(point.q_t_total_N_per_m for point in distribution), x)
    total_n = trapezoidal_integral(tuple(point.q_n_total_N_per_m for point in distribution), x)
    return (
        residuals,
        total_t - mechanism.phase4a_prediction.tangential_force_magnitude_N,
        total_n - mechanism.phase4a_prediction.normal_force_magnitude_N,
    )


def predict_statistical_grain_load(case: StatisticalGrainLoadCase) -> StatisticalGrainLoadPrediction:
    if not isinstance(case, StatisticalGrainLoadCase):
        raise TypeError("predict_statistical_grain_load requires a validated StatisticalGrainLoadCase")
    mechanism = predict_mechanism_force(case.mechanism_force)
    resolved = resolve_grit_specification(case.wheel_specification)
    contact_length = mechanism.phase4a_prediction.geometric_contact_length_m
    width = case.mechanism_force.force_model.process.grinding_width_m
    density = case.grain_population.active_density_factor / resolved.representative_diameter_d50_m**2
    estimated = density * width * contact_length
    groups = build_equivalent_grain_groups(resolved, case.grain_population, case.load_distribution, contact_length, estimated)
    distribution = build_load_distribution(mechanism, groups, case.load_distribution, contact_length)
    residuals, total_t, total_n = _force_residuals(mechanism, distribution)
    return StatisticalGrainLoadPrediction(
        case=case,
        mechanism_prediction=mechanism,
        resolved_grit_specification=resolved,
        contact_length_m=contact_length,
        grinding_width_m=width,
        active_grain_surface_density_per_m2=density,
        estimated_physical_active_grain_count=estimated,
        equivalent_group_count=len(groups),
        compression_ratio=estimated / len(groups),
        groups=groups,
        distribution=distribution,
        mechanism_force_residuals_N=residuals,
        total_tangential_force_residual_N=total_t,
        total_normal_force_residual_N=total_n,
    )
