"""Phase 4C2 contact-ratio scaling of an empirical steady grinding load."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae import __version__
from grindcae.force_model import GrindingForcePrediction, predict_grinding_force
from grindcae.trajectory import TrajectoryResult, build_surface_profile

from .models import PassLoadCase


RESULT_FORMAT = "grindcae_phase_4c2_single_pass_empirical_load"
SEGMENT_TYPES = ("unloaded", "active_projected_contact")


class PassLoadError(RuntimeError):
    """Raised when valid inputs cannot produce a consistent pass-load result."""


def _pass_load_error(field: str, unit: str, reason: str) -> None:
    raise PassLoadError(f"{field} [{unit}]: {reason}")


def _finite(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _pass_load_error(field, unit, "must be numeric")
    try:
        number = float(value)
    except OverflowError:
        _pass_load_error(field, unit, "must be finite and representable")
    if not math.isfinite(number):
        _pass_load_error(field, unit, "must be finite")
    return number


def _close(left: float, right: float, scale: float = 1.0) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12 * max(abs(scale), 1.0),
    )


@dataclass(frozen=True, slots=True)
class PassLoadSegment:
    """One positive-length interval of the ideal workpiece top boundary."""

    segment_type: str
    x_start_m: float
    x_end_m: float
    qx_N_per_m: float
    qy_N_per_m: float

    def __post_init__(self) -> None:
        if self.segment_type not in SEGMENT_TYPES:
            _pass_load_error("load_segments.segment_type", "literal", "is invalid")
        for field, unit in (
            ("x_start_m", "m"),
            ("x_end_m", "m"),
            ("qx_N_per_m", "N/m"),
            ("qy_N_per_m", "N/m"),
        ):
            object.__setattr__(
                self, field, _finite(getattr(self, field), f"load_segments.{field}", unit)
            )
        if not self.x_start_m < self.x_end_m:
            _pass_load_error("load_segments.length_m", "m", "must be positive")
        if self.segment_type == "unloaded" and (
            self.qx_N_per_m != 0.0 or self.qy_N_per_m != 0.0
        ):
            _pass_load_error("load_segments.unloaded", "N/m", "must have zero load")
        if self.qy_N_per_m > 0.0:
            _pass_load_error("load_segments.qy_N_per_m", "N/m", "must be non-positive")

    @property
    def length_m(self) -> float:
        return self.x_end_m - self.x_start_m

    @property
    def integrated_Fx_N(self) -> float:
        return self.qx_N_per_m * self.length_m

    @property
    def integrated_Fy_N(self) -> float:
        return self.qy_N_per_m * self.length_m

    def to_dict(self, segment_index: int) -> dict[str, float | int | str]:
        return {
            "segment_index": segment_index,
            "segment_type": self.segment_type,
            "x_start_m": self.x_start_m,
            "x_end_m": self.x_end_m,
            "length_m": self.length_m,
            "qx_N_per_m": self.qx_N_per_m,
            "qy_N_per_m": self.qy_N_per_m,
            "integrated_Fx_N": self.integrated_Fx_N,
            "integrated_Fy_N": self.integrated_Fy_N,
        }


@dataclass(frozen=True, slots=True)
class PassLoadResult:
    """Validated 4C1 geometry, 4A steady force, and current 4C2 load snapshot."""

    case: PassLoadCase
    trajectory_result: TrajectoryResult
    force_prediction: GrindingForcePrediction
    pass_state: str
    exact_arc_projected_length_m: float
    shallow_cut_approximation_length_m: float
    effective_contact_interval_m: tuple[float, float] | None
    effective_contact_length_m: float
    contact_ratio: float
    steady_tangential_force_magnitude_N: float
    steady_normal_force_magnitude_N: float
    steady_resultant_force_magnitude_N: float
    current_tangential_force_magnitude_N: float
    current_normal_force_magnitude_N: float
    current_resultant_force_magnitude_N: float
    current_Fx_N: float
    current_Fy_N: float
    current_qx_N_per_m: float
    current_qy_N_per_m: float
    grinding_width_m: float
    current_uniform_normal_pressure_conversion_Pa: float
    current_uniform_tangential_traction_conversion_Pa: float
    force_integral_residual_Fx_N: float
    force_integral_residual_Fy_N: float
    load_segments: tuple[PassLoadSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, PassLoadCase):
            _pass_load_error("case", "PassLoadCase", "has the wrong object type")
        if not isinstance(self.trajectory_result, TrajectoryResult):
            _pass_load_error(
                "trajectory_result", "TrajectoryResult", "has the wrong object type"
            )
        if not isinstance(self.force_prediction, GrindingForcePrediction):
            _pass_load_error(
                "force_prediction", "GrindingForcePrediction", "has the wrong object type"
            )
        if self.pass_state != self.trajectory_result.pass_state:
            _pass_load_error("pass_state", "state literal", "must match phase 4C1")

        numeric_fields = (
            ("exact_arc_projected_length_m", "m"),
            ("shallow_cut_approximation_length_m", "m"),
            ("effective_contact_length_m", "m"),
            ("contact_ratio", "dimensionless"),
            ("steady_tangential_force_magnitude_N", "N"),
            ("steady_normal_force_magnitude_N", "N"),
            ("steady_resultant_force_magnitude_N", "N"),
            ("current_tangential_force_magnitude_N", "N"),
            ("current_normal_force_magnitude_N", "N"),
            ("current_resultant_force_magnitude_N", "N"),
            ("current_Fx_N", "N"),
            ("current_Fy_N", "N"),
            ("current_qx_N_per_m", "N/m"),
            ("current_qy_N_per_m", "N/m"),
            ("grinding_width_m", "m"),
            ("current_uniform_normal_pressure_conversion_Pa", "Pa"),
            ("current_uniform_tangential_traction_conversion_Pa", "Pa"),
            ("force_integral_residual_Fx_N", "N"),
            ("force_integral_residual_Fy_N", "N"),
        )
        for field, unit in numeric_fields:
            object.__setattr__(self, field, _finite(getattr(self, field), field, unit))

        if self.exact_arc_projected_length_m <= 0.0 or self.grinding_width_m <= 0.0:
            _pass_load_error("length_basis", "m", "must be positive")
        if not 0.0 <= self.contact_ratio <= 1.0:
            _pass_load_error("contact_ratio", "dimensionless", "must lie in [0, 1]")
        if not _close(
            self.effective_contact_length_m,
            self.contact_ratio * self.exact_arc_projected_length_m,
            self.exact_arc_projected_length_m,
        ):
            _pass_load_error("contact_ratio", "dimensionless", "must equal Leff / Larc")

        steady_current_pairs = (
            (self.steady_tangential_force_magnitude_N, self.current_tangential_force_magnitude_N),
            (self.steady_normal_force_magnitude_N, self.current_normal_force_magnitude_N),
            (self.steady_resultant_force_magnitude_N, self.current_resultant_force_magnitude_N),
        )
        for steady, current in steady_current_pairs:
            if steady < 0.0 or current < 0.0 or not _close(
                current, steady * self.contact_ratio, steady
            ):
                _pass_load_error("current_force", "N", "must equal steady force times contact_ratio")
        if self.current_Fy_N > 0.0 or self.current_qy_N_per_m > 0.0:
            _pass_load_error("normal_load_direction", "N or N/m", "must be non-positive")
        if not _close(
            self.current_Fy_N,
            -self.current_normal_force_magnitude_N,
            self.steady_normal_force_magnitude_N,
        ):
            _pass_load_error("current_Fy_N", "N", "must equal -current normal force")
        expected_fx = (
            self.current_tangential_force_magnitude_N
            if self.case.force_mapping.tangential_force_direction == "positive_x"
            else -self.current_tangential_force_magnitude_N
        )
        if not _close(
            self.current_Fx_N,
            expected_fx,
            self.steady_tangential_force_magnitude_N,
        ):
            _pass_load_error(
                "current_Fx_N",
                "N",
                "must match the explicit tangential_force_direction",
            )

        interval = self.effective_contact_interval_m
        if self.effective_contact_length_m == 0.0:
            if interval is not None:
                _pass_load_error("effective_contact_interval_m", "m", "must be null at zero contact")
            zero_fields = (
                self.current_tangential_force_magnitude_N,
                self.current_normal_force_magnitude_N,
                self.current_resultant_force_magnitude_N,
                self.current_Fx_N,
                self.current_Fy_N,
                self.current_qx_N_per_m,
                self.current_qy_N_per_m,
                self.current_uniform_normal_pressure_conversion_Pa,
                self.current_uniform_tangential_traction_conversion_Pa,
            )
            if any(value != 0.0 for value in zero_fields):
                _pass_load_error("zero_contact", "finite zero load", "all current loads must be zero")
        else:
            if interval is None:
                _pass_load_error("effective_contact_interval_m", "m", "is required for nonzero contact")
            start, end = interval
            if not start < end or not _close(end - start, self.effective_contact_length_m, end - start):
                _pass_load_error("effective_contact_interval_m", "m", "must have length Leff")
            if not _close(
                self.current_qx_N_per_m * self.effective_contact_length_m,
                self.current_Fx_N,
                self.current_Fx_N,
            ) or not _close(
                self.current_qy_N_per_m * self.effective_contact_length_m,
                self.current_Fy_N,
                self.current_Fy_N,
            ):
                _pass_load_error("line_load_integral", "N", "q times Leff must reproduce current force")
            projected_area = self.grinding_width_m * self.effective_contact_length_m
            if not _close(
                self.current_uniform_normal_pressure_conversion_Pa * projected_area,
                self.current_normal_force_magnitude_N,
                self.current_normal_force_magnitude_N,
            ) or not _close(
                self.current_uniform_tangential_traction_conversion_Pa
                * projected_area,
                self.current_tangential_force_magnitude_N,
                self.current_tangential_force_magnitude_N,
            ):
                _pass_load_error(
                    "surface_traction_conversion",
                    "Pa",
                    "must reproduce the current force over grinding_width_m * Leff",
                )

        segments = tuple(self.load_segments)
        width = self.case.trajectory.workpiece.length_m
        if not segments or segments[0].x_start_m != 0.0 or segments[-1].x_end_m != width:
            _pass_load_error("load_segments", "m", "must cover exact [0, W]")
        active_count = 0
        for index, segment in enumerate(segments):
            if not isinstance(segment, PassLoadSegment):
                _pass_load_error("load_segments", "PassLoadSegment", "has wrong object type")
            if index and not _close(segments[index - 1].x_end_m, segment.x_start_m, width):
                _pass_load_error("load_segments", "m", "must be contiguous and non-overlapping")
            active_count += segment.segment_type == "active_projected_contact"
        expected_active_count = 0 if interval is None else 1
        if active_count != expected_active_count:
            _pass_load_error("load_segments", "active segment count", "does not match effective contact")
        if interval is not None:
            active = next(
                segment for segment in segments if segment.segment_type == "active_projected_contact"
            )
            if active.x_start_m != interval[0] or active.x_end_m != interval[1]:
                _pass_load_error("load_segments", "m", "active endpoints must exactly match phase 4C1")
        total_length = math.fsum(segment.length_m for segment in segments)
        integrated_fx = math.fsum(segment.integrated_Fx_N for segment in segments)
        integrated_fy = math.fsum(segment.integrated_Fy_N for segment in segments)
        if not _close(total_length, width, width):
            _pass_load_error("load_segments", "m", "lengths must sum to W")
        if not _close(integrated_fx, self.current_Fx_N, self.current_Fx_N) or not _close(
            integrated_fy, self.current_Fy_N, self.current_Fy_N
        ):
            _pass_load_error("load_segments", "N", "integrals must equal current Fx and Fy")
        object.__setattr__(self, "load_segments", segments)

    @property
    def phase_4a_geometric_contact_length_m(self) -> float:
        return self.force_prediction.geometric_contact_length_m

    @property
    def length_absolute_difference_m(self) -> float:
        return abs(
            self.phase_4a_geometric_contact_length_m
            - self.exact_arc_projected_length_m
        )

    @property
    def length_relative_difference(self) -> float:
        return self.length_absolute_difference_m / self.exact_arc_projected_length_m

    def to_summary(self, artifacts: dict[str, str]) -> dict[str, object]:
        """Return the deterministic finite phase 4C2 JSON payload."""

        segment_fx = math.fsum(segment.integrated_Fx_N for segment in self.load_segments)
        segment_fy = math.fsum(segment.integrated_Fy_N for segment in self.load_segments)
        segment_length = math.fsum(segment.length_m for segment in self.load_segments)
        return {
            "result_format": RESULT_FORMAT,
            "package_version": __version__,
            "pass_load_schema_version": self.case.pass_load_schema_version,
            "trajectory_schema_version": self.case.trajectory.trajectory_schema_version,
            "force_model_schema_version": self.case.force_model.force_model_schema_version,
            "unit_system": self.case.unit_system,
            "normalized_input": self.case.to_dict(),
            "units": {
                "length": "m",
                "force": "N",
                "line_load": "N/m",
                "surface_traction_conversion": "Pa",
                "contact_ratio": "dimensionless",
            },
            "direction_conventions": {
                "relative_feed_direction": self.case.trajectory.single_pass.relative_feed_direction,
                "relative_feed_direction_meaning": "Wheel-lowest-point trajectory direction in workpiece coordinates.",
                "tangential_force_direction": self.case.force_mapping.tangential_force_direction,
                "tangential_force_direction_meaning": "Independent sign choice for current Fx; it is not inferred from relative feed direction.",
                "normal_force_direction": "Fy and qy are non-positive, directed into the ideal workpiece top surface.",
            },
            "trajectory_contact": {
                "pass_state": self.pass_state,
                "theoretical_arc_interval_m": list(self.trajectory_result.theoretical_arc_interval_m),
                "effective_projected_interval_m": (
                    None
                    if self.effective_contact_interval_m is None
                    else list(self.effective_contact_interval_m)
                ),
                "effective_contact_length_m": self.effective_contact_length_m,
                "contact_ratio": self.contact_ratio,
            },
            "length_basis_comparison": {
                "phase_4c2_exact_arc_projected_length_m": self.exact_arc_projected_length_m,
                "phase_4a_shallow_cut_geometric_contact_length_m": self.phase_4a_geometric_contact_length_m,
                "absolute_difference_m": self.length_absolute_difference_m,
                "relative_difference": self.length_relative_difference,
                "relative_difference_reference": "phase_4c2_exact_arc_projected_length_m",
                "phase_4c2_scaling_and_integration_basis": "sqrt(Ds * ae - ae**2)",
                "phase_4a_comparison_only": "sqrt(Ds * ae)",
            },
            "steady_full_contact_empirical_force": {
                "tangential_force_magnitude_N": self.steady_tangential_force_magnitude_N,
                "normal_force_magnitude_N": self.steady_normal_force_magnitude_N,
                "resultant_force_magnitude_N": self.steady_resultant_force_magnitude_N,
                "meaning": "Phase 4A empirical steady reference for complete projected contact, not the current applied force when contact_ratio is below one.",
            },
            "current_empirical_load_snapshot": {
                "scaling_rule": "current force = steady full-contact force * contact_ratio",
                "tangential_force_magnitude_N": self.current_tangential_force_magnitude_N,
                "normal_force_magnitude_N": self.current_normal_force_magnitude_N,
                "resultant_force_magnitude_N": self.current_resultant_force_magnitude_N,
                "signed_Fx_N": self.current_Fx_N,
                "signed_Fy_N": self.current_Fy_N,
                "uniform_qx_N_per_m": self.current_qx_N_per_m,
                "uniform_qy_N_per_m": self.current_qy_N_per_m,
                "grinding_width_m": self.grinding_width_m,
                "uniform_normal_pressure_conversion_Pa": self.current_uniform_normal_pressure_conversion_Pa,
                "uniform_tangential_traction_conversion_Pa": self.current_uniform_tangential_traction_conversion_Pa,
                "effective_loading_interval_m": (
                    None
                    if self.effective_contact_interval_m is None
                    else list(self.effective_contact_interval_m)
                ),
            },
            "load_segments": {
                "indexing": "0-based contiguous",
                "boundary": self.case.force_mapping.boundary,
                "distribution": self.case.force_mapping.distribution,
                "segments": [
                    segment.to_dict(index) for index, segment in enumerate(self.load_segments)
                ],
                "partition_total_m": segment_length,
                "partition_residual_m": self.case.trajectory.workpiece.length_m - segment_length,
                "integrated_Fx_N": segment_fx,
                "integrated_Fy_N": segment_fy,
            },
            "load_integral_validation": {
                "qx_times_effective_length_N": self.current_qx_N_per_m
                * self.effective_contact_length_m,
                "qy_times_effective_length_N": self.current_qy_N_per_m
                * self.effective_contact_length_m,
                "target_current_Fx_N": self.current_Fx_N,
                "target_current_Fy_N": self.current_Fy_N,
                "residual_Fx_N": self.force_integral_residual_Fx_N,
                "residual_Fy_N": self.force_integral_residual_Fy_N,
                "csv_segment_integrated_Fx_N": segment_fx,
                "csv_segment_integrated_Fy_N": segment_fy,
                "passed": _close(segment_fx, self.current_Fx_N, self.current_Fx_N)
                and _close(segment_fy, self.current_Fy_N, self.current_Fy_N),
            },
            "calibration_provenance": self.case.force_model.calibration.provenance_dict(),
            "assumptions_and_cautions": [
                "This is a basic empirical contact-length-ratio scaling approximation.",
                "Phase 4A supplies the empirical steady force for complete projected contact.",
                "For nonzero contact, phase 4C2 assumes the load per unit projected length remains constant.",
                "The current total force builds and decays linearly with effective projected contact length.",
                "This scaling law and its user-supplied calibration are unvalidated by user experiments.",
                "The effective projected interval is not a true Hertz contact region.",
                "The uniform line load is not a real abrasive-grain pressure distribution.",
                "The result must not be presented as a true or industrially validated grinding force.",
                "No Gmsh, meshio, scikit-fem, FEM solve, stress recovery, or true contact iteration is run.",
                "Impact, acceleration, vibration, chatter, thermal fields, wheel wear, abrasive-grain randomness, ploughing, plasticity, edge chipping, and multiple passes are omitted.",
            ],
            "artifacts": artifacts,
        }


def _build_segments(
    width_m: float,
    interval_m: tuple[float, float] | None,
    qx_N_per_m: float,
    qy_N_per_m: float,
) -> tuple[PassLoadSegment, ...]:
    if interval_m is None:
        return (PassLoadSegment("unloaded", 0.0, width_m, 0.0, 0.0),)
    start, end = interval_m
    values: list[PassLoadSegment] = []
    if start > 0.0:
        values.append(PassLoadSegment("unloaded", 0.0, start, 0.0, 0.0))
    values.append(
        PassLoadSegment(
            "active_projected_contact", start, end, qx_N_per_m, qy_N_per_m
        )
    )
    if end < width_m:
        values.append(PassLoadSegment("unloaded", end, width_m, 0.0, 0.0))
    return tuple(values)


def build_pass_load(case: PassLoadCase) -> PassLoadResult:
    """Combine the existing 4C1 geometry and 4A force model without FEM."""

    if not isinstance(case, PassLoadCase):
        _pass_load_error(
            "case", "PassLoadCase", "build_pass_load requires a validated case"
        )
    trajectory = build_surface_profile(case.trajectory)
    prediction = predict_grinding_force(case.force_model)
    arc = _finite(
        trajectory.exact_arc_projected_length_m,
        "exact_arc_projected_length_m",
        "m",
    )
    effective_length = _finite(
        trajectory.effective_contact_length_m,
        "effective_contact_length_m",
        "m",
    )
    ratio = _finite(trajectory.contact_ratio, "contact_ratio", "dimensionless")

    steady_ft = prediction.tangential_force_magnitude_N
    steady_fn = prediction.normal_force_magnitude_N
    steady_fr = prediction.resultant_force_magnitude_N
    # 中文导读：入口和出口处于部分接触时，当前力按接触比低于稳态力。
    current_ft = _finite(ratio * steady_ft, "current_tangential_force_magnitude_N", "N")
    current_fn = _finite(ratio * steady_fn, "current_normal_force_magnitude_N", "N")
    current_fr = _finite(ratio * steady_fr, "current_resultant_force_magnitude_N", "N")
    sign = 1.0 if case.force_mapping.tangential_force_direction == "positive_x" else -1.0
    current_fx = _finite(sign * current_ft, "current_Fx_N", "N")
    current_fy = _finite(-current_fn, "current_Fy_N", "N")

    if effective_length == 0.0:
        current_ft = current_fn = current_fr = 0.0
        current_fx = current_fy = 0.0
        qx = qy = 0.0
        normal_pressure = tangential_traction = 0.0
        effective_interval = None
    else:
        effective_interval = trajectory.effective_arc_interval_m
        # 中文导读：将当前 Fx/Fy 分量均匀分配到有效投影长度，得到 qx/qy [N/m]。
        qx = _finite(current_fx / effective_length, "current_qx_N_per_m", "N/m")
        qy = _finite(current_fy / effective_length, "current_qy_N_per_m", "N/m")
        width = case.force_model.process.grinding_width_m
        normal_pressure = _finite(
            current_fn / (width * effective_length),
            "current_uniform_normal_pressure_conversion_Pa",
            "Pa",
        )
        tangential_traction = _finite(
            current_ft / (width * effective_length),
            "current_uniform_tangential_traction_conversion_Pa",
            "Pa",
        )

    residual_fx = _finite(qx * effective_length - current_fx, "force_integral_residual_Fx_N", "N")
    residual_fy = _finite(qy * effective_length - current_fy, "force_integral_residual_Fy_N", "N")
    segments = _build_segments(
        case.trajectory.workpiece.length_m, effective_interval, qx, qy
    )
    return PassLoadResult(
        case=case,
        trajectory_result=trajectory,
        force_prediction=prediction,
        pass_state=trajectory.pass_state,
        exact_arc_projected_length_m=arc,
        shallow_cut_approximation_length_m=prediction.geometric_contact_length_m,
        effective_contact_interval_m=effective_interval,
        effective_contact_length_m=effective_length,
        contact_ratio=ratio,
        steady_tangential_force_magnitude_N=steady_ft,
        steady_normal_force_magnitude_N=steady_fn,
        steady_resultant_force_magnitude_N=steady_fr,
        current_tangential_force_magnitude_N=current_ft,
        current_normal_force_magnitude_N=current_fn,
        current_resultant_force_magnitude_N=current_fr,
        current_Fx_N=current_fx,
        current_Fy_N=current_fy,
        current_qx_N_per_m=qx,
        current_qy_N_per_m=qy,
        grinding_width_m=case.force_model.process.grinding_width_m,
        current_uniform_normal_pressure_conversion_Pa=normal_pressure,
        current_uniform_tangential_traction_conversion_Pa=tangential_traction,
        force_integral_residual_Fx_N=residual_fx,
        force_integral_residual_Fy_N=residual_fy,
        load_segments=segments,
    )
