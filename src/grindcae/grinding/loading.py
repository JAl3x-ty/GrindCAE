"""Map phase 4A grinding-force magnitudes to a local top-edge load."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae.domain import BoundarySide
from grindcae.force_model import (
    ForceModelError,
    GrindingForcePrediction,
    predict_grinding_force,
)

from .models import GrindingSimulationCase


LOAD_SOURCE = "empirical_specific_grinding_energy_force_ratio"


class GrindingLoadError(RuntimeError):
    """Raised when a validated schema version 4 case cannot map to a load."""


def _load_error(field: str, unit: str, reason: str) -> None:
    raise GrindingLoadError(f"{field} [{unit}]: {reason}")


def _finite(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _load_error(field, unit, "must be numeric")
    try:
        number = float(value)
    except OverflowError:
        _load_error(field, unit, "must be finite and representable")
    if not math.isfinite(number):
        _load_error(field, unit, "must be finite")
    return number


def _require_close(
    actual: float,
    expected: float,
    field: str,
    unit: str,
    reason: str,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
        _load_error(field, unit, reason)


@dataclass(frozen=True, slots=True)
class ResolvedGrindingLoad:
    """Traceable force prediction, contact interval, and signed local load."""

    force_prediction: GrindingForcePrediction
    load_source: str
    calibration_id: str
    boundary: BoundarySide
    distribution: str
    tangential_direction: str
    center_x_m: float
    x_start_m: float
    x_end_m: float
    active_length_m: float
    signed_force_x_N: float
    signed_force_y_N: float
    line_load_x_N_per_m: float
    line_load_y_N_per_m: float
    average_normal_pressure_Pa: float
    average_tangential_traction_Pa: float

    def __post_init__(self) -> None:
        if not isinstance(self.force_prediction, GrindingForcePrediction):
            _load_error(
                "force_prediction",
                "GrindingForcePrediction",
                "has the wrong object type",
            )
        if self.load_source != LOAD_SOURCE:
            _load_error("load_source", LOAD_SOURCE, "has the wrong provenance")
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            _load_error("calibration_id", "calibration identifier", "must be non-empty")
        if self.boundary is not BoundarySide.TOP:
            _load_error("boundary", "rectangle side", "must be top")
        if self.distribution != "uniform":
            _load_error("distribution", "uniform", "must be uniform")
        if self.tangential_direction not in ("positive_x", "negative_x"):
            _load_error(
                "tangential_direction",
                "direction literal",
                "must be positive_x or negative_x",
            )

        quantity_units = {
            "center_x_m": "m",
            "x_start_m": "m",
            "x_end_m": "m",
            "active_length_m": "m",
            "signed_force_x_N": "N",
            "signed_force_y_N": "N",
            "line_load_x_N_per_m": "N/m",
            "line_load_y_N_per_m": "N/m",
            "average_normal_pressure_Pa": "Pa",
            "average_tangential_traction_Pa": "Pa",
        }
        for field, unit in quantity_units.items():
            object.__setattr__(self, field, _finite(getattr(self, field), field, unit))
        if not self.x_start_m < self.x_end_m:
            _load_error("contact_interval", "m", "must satisfy x_start_m < x_end_m")
        if self.active_length_m <= 0.0:
            _load_error("active_length_m", "m", "must be greater than zero")
        _require_close(
            self.active_length_m,
            self.force_prediction.geometric_contact_length_m,
            "active_length_m",
            "m",
            "must equal force_prediction.geometric_contact_length_m",
        )
        if not math.isclose(
            self.x_end_m - self.x_start_m,
            self.active_length_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            _load_error(
                "active_length_m", "m", "must equal x_end_m - x_start_m"
            )
        _require_close(
            self.center_x_m,
            (self.x_start_m + self.x_end_m) / 2.0,
            "center_x_m",
            "m",
            "must be the midpoint of x_start_m and x_end_m",
        )
        if self.signed_force_y_N > 0.0:
            _load_error("signed_force_y_N", "N", "must be compressive or zero")
        if self.average_normal_pressure_Pa < 0.0:
            _load_error(
                "average_normal_pressure_Pa", "Pa", "must be non-negative"
            )
        if self.average_tangential_traction_Pa < 0.0:
            _load_error(
                "average_tangential_traction_Pa", "Pa", "must be non-negative"
            )
        calibration = self.force_prediction.force_case.calibration
        process = self.force_prediction.force_case.process
        if self.calibration_id != calibration.calibration_id:
            _load_error(
                "calibration_id",
                "calibration identifier",
                "must match force_prediction provenance",
            )
        expected_x_sign = 1.0 if self.tangential_direction == "positive_x" else -1.0
        _require_close(
            self.signed_force_x_N,
            expected_x_sign * self.force_prediction.tangential_force_magnitude_N,
            "signed_force_x_N",
            "N",
            "must map the predicted tangential-force magnitude and direction",
        )
        _require_close(
            self.signed_force_y_N,
            -self.force_prediction.normal_force_magnitude_N,
            "signed_force_y_N",
            "N",
            "must be the negative predicted normal-force magnitude",
        )
        _require_close(
            self.line_load_x_N_per_m * self.active_length_m,
            self.signed_force_x_N,
            "line_load_x_N_per_m",
            "N/m",
            "must integrate to signed_force_x_N over active_length_m",
        )
        _require_close(
            self.line_load_y_N_per_m * self.active_length_m,
            self.signed_force_y_N,
            "line_load_y_N_per_m",
            "N/m",
            "must integrate to signed_force_y_N over active_length_m",
        )
        loaded_area = process.grinding_width_m * self.active_length_m
        _require_close(
            self.average_normal_pressure_Pa * loaded_area,
            self.force_prediction.normal_force_magnitude_N,
            "average_normal_pressure_Pa",
            "Pa",
            "must convert the predicted normal-force magnitude over the loaded area",
        )
        _require_close(
            self.average_tangential_traction_Pa * loaded_area,
            self.force_prediction.tangential_force_magnitude_N,
            "average_tangential_traction_Pa",
            "Pa",
            "must convert the predicted tangential-force magnitude over the loaded area",
        )

    def to_dict(self) -> dict[str, object]:
        """Return the prediction and traceable local-load mapping."""

        return {
            "force_prediction": self.force_prediction.to_dict(),
            "load_source": self.load_source,
            "calibration_id": self.calibration_id,
            "boundary": self.boundary.value,
            "distribution": self.distribution,
            "tangential_direction": self.tangential_direction,
            "center_x_m": self.center_x_m,
            "x_start_m": self.x_start_m,
            "x_end_m": self.x_end_m,
            "active_length_m": self.active_length_m,
            "signed_force_x_N": self.signed_force_x_N,
            "signed_force_y_N": self.signed_force_y_N,
            "line_load_x_N_per_m": self.line_load_x_N_per_m,
            "line_load_y_N_per_m": self.line_load_y_N_per_m,
            "average_normal_pressure_Pa": self.average_normal_pressure_Pa,
            "average_tangential_traction_Pa": (
                self.average_tangential_traction_Pa
            ),
        }


def resolve_grinding_load(case: GrindingSimulationCase) -> ResolvedGrindingLoad:
    """Predict force magnitudes and map them to a signed uniform top-edge load."""

    if not isinstance(case, GrindingSimulationCase):
        _load_error(
            "case",
            "GrindingSimulationCase",
            "resolve_grinding_load requires a validated schema version 4 case",
        )
    try:
        prediction = predict_grinding_force(case.force_model)
    except ForceModelError as exc:
        raise GrindingLoadError(f"grinding.force_model.prediction [SI]: {exc}") from exc

    active_length = _finite(
        prediction.geometric_contact_length_m,
        "force_prediction.geometric_contact_length_m",
        "m",
    )
    if active_length <= 0.0:
        _load_error(
            "force_prediction.geometric_contact_length_m",
            "m",
            "must be greater than zero",
        )
    center = case.contact_zone.center_x_m
    x_start = _finite(center - active_length / 2.0, "contact_zone.x_start_m", "m")
    x_end = _finite(center + active_length / 2.0, "contact_zone.x_end_m", "m")
    if not (0.0 < x_start < x_end < case.geometry.width):
        _load_error(
            "grinding.contact_zone",
            "m",
            "computed interval must satisfy 0 < x_start_m < x_end_m < geometry.width so it remains strictly inside the top edge",
        )

    tangential_force = prediction.tangential_force_magnitude_N
    normal_force = prediction.normal_force_magnitude_N
    direction_sign = 1.0 if case.contact_zone.tangential_direction == "positive_x" else -1.0
    signed_force_x = _finite(
        direction_sign * tangential_force, "signed_force_x_N", "N"
    )
    signed_force_y = _finite(-normal_force, "signed_force_y_N", "N")
    line_load_x = _finite(
        signed_force_x / active_length, "line_load_x_N_per_m", "N/m"
    )
    line_load_y = _finite(
        signed_force_y / active_length, "line_load_y_N_per_m", "N/m"
    )
    loaded_area = _finite(
        case.force_model.process.grinding_width_m * active_length,
        "loaded_area_m2",
        "m^2",
    )
    if loaded_area <= 0.0:
        _load_error("loaded_area_m2", "m^2", "must be greater than zero")
    normal_pressure = _finite(
        normal_force / loaded_area, "average_normal_pressure_Pa", "Pa"
    )
    tangential_traction = _finite(
        tangential_force / loaded_area,
        "average_tangential_traction_Pa",
        "Pa",
    )

    return ResolvedGrindingLoad(
        force_prediction=prediction,
        load_source=LOAD_SOURCE,
        calibration_id=case.force_model.calibration.calibration_id,
        boundary=case.contact_zone.boundary,
        distribution=case.contact_zone.distribution,
        tangential_direction=case.contact_zone.tangential_direction,
        center_x_m=center,
        x_start_m=x_start,
        x_end_m=x_end,
        active_length_m=active_length,
        signed_force_x_N=signed_force_x,
        signed_force_y_N=signed_force_y,
        line_load_x_N_per_m=line_load_x,
        line_load_y_N_per_m=line_load_y,
        average_normal_pressure_Pa=normal_pressure,
        average_tangential_traction_Pa=tangential_traction,
    )
