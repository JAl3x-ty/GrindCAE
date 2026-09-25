"""Strict damage-input contract for the GrindCAE 4.0.0 route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


class SingleGrainContactValidationError(ValueError):
    """Raised when a single-grain contact case violates its strict contract."""


def _fail(field: str, reason: str) -> None:
    raise SingleGrainContactValidationError(f"{field}: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        _fail(field, f"missing fields {missing}; unexpected fields {unexpected}")


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, "must be numeric and finite")
    result = float(value)
    if not math.isfinite(result):
        _fail(field, "must be finite")
    if positive and result <= 0.0:
        _fail(field, "must be strictly greater than zero")
    return result


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        _fail(field, "must be a strict integer")
    if not minimum <= value <= maximum:
        _fail(field, f"must be between {minimum} and {maximum}")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(field, "must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class DamageInput:
    model: str
    failure_strain_table: tuple[tuple[float, float], ...]
    triaxiality_cutoff: float
    evolution: str
    fracture_energy_J_per_m2: float
    characteristic_length: str
    damage_event_tolerance: float
    maximum_separation_events_per_position: int
    area_absolute_tolerance_m2: float
    parameter_source: str
    calibration_status: str
    applicability_notes: str

    @classmethod
    def from_mapping(cls, value: object) -> DamageInput:
        data = _object(value, "damage")
        fields = {
            "model",
            "failure_strain_table",
            "triaxiality_cutoff",
            "evolution",
            "fracture_energy_J_per_m2",
            "characteristic_length",
            "damage_event_tolerance",
            "maximum_separation_events_per_position",
            "area_absolute_tolerance_m2",
            "parameter_source",
            "calibration_status",
            "applicability_notes",
        }
        _exact(data, fields, "damage")
        if data["model"] != "ductile_triaxiality_table":
            _fail("damage.model", "must be ductile_triaxiality_table")
        if data["evolution"] not in ("linear_fracture_energy_softening", "energetic_rational_fracture"):
            _fail(
                "damage.evolution",
                "must be linear_fracture_energy_softening or energetic_rational_fracture",
            )
        if data["characteristic_length"] != "equivalent_circle_diameter_reference_area":
            _fail(
                "damage.characteristic_length",
                "must be equivalent_circle_diameter_reference_area",
            )
        raw_table = data["failure_strain_table"]
        if not isinstance(raw_table, list) or len(raw_table) < 2:
            _fail("damage.failure_strain_table", "must contain at least two rows")
        table: list[tuple[float, float]] = []
        for index, row in enumerate(raw_table):
            field = f"damage.failure_strain_table[{index}]"
            if not isinstance(row, list) or len(row) != 2:
                _fail(field, "must contain triaxiality and positive failure strain")
            eta = _number(row[0], f"{field}[0]")
            failure_strain = _number(row[1], f"{field}[1]", positive=True)
            if table and eta <= table[-1][0]:
                _fail("damage.failure_strain_table", "triaxiality must be strictly increasing")
            table.append((eta, failure_strain))
        tolerance = _number(
            data["damage_event_tolerance"],
            "damage.damage_event_tolerance",
            positive=True,
        )
        if tolerance > 1.0e-3:
            _fail("damage.damage_event_tolerance", "must not exceed 1e-3")
        calibration = _text(data["calibration_status"], "damage.calibration_status")
        if calibration not in {"uncalibrated", "preliminary", "calibrated"}:
            _fail(
                "damage.calibration_status",
                "must be uncalibrated, preliminary, or calibrated",
            )
        return cls(
            model="ductile_triaxiality_table",
            failure_strain_table=tuple(table),
            triaxiality_cutoff=_number(
                data["triaxiality_cutoff"], "damage.triaxiality_cutoff"
            ),
            evolution=data["evolution"],
            fracture_energy_J_per_m2=_number(
                data["fracture_energy_J_per_m2"],
                "damage.fracture_energy_J_per_m2",
                positive=True,
            ),
            characteristic_length="equivalent_circle_diameter_reference_area",
            damage_event_tolerance=tolerance,
            maximum_separation_events_per_position=_integer(
                data["maximum_separation_events_per_position"],
                "damage.maximum_separation_events_per_position",
                1,
                1000,
            ),
            area_absolute_tolerance_m2=_number(
                data["area_absolute_tolerance_m2"],
                "damage.area_absolute_tolerance_m2",
                positive=True,
            ),
            parameter_source=_text(
                data["parameter_source"], "damage.parameter_source"
            ),
            calibration_status=calibration,
            applicability_notes=_text(
                data["applicability_notes"], "damage.applicability_notes"
            ),
        )

    def failure_strain_at(self, triaxiality: float) -> float:
        eta = _number(triaxiality, "triaxiality")
        if eta <= self.failure_strain_table[0][0]:
            return self.failure_strain_table[0][1]
        if eta >= self.failure_strain_table[-1][0]:
            return self.failure_strain_table[-1][1]
        for (eta_0, strain_0), (eta_1, strain_1) in zip(
            self.failure_strain_table, self.failure_strain_table[1:]
        ):
            if eta <= eta_1:
                fraction = (eta - eta_0) / (eta_1 - eta_0)
                return strain_0 + fraction * (strain_1 - strain_0)
        raise RuntimeError("failure-strain interpolation interval was not found")

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "failure_strain_table": [list(row) for row in self.failure_strain_table],
            "triaxiality_cutoff": self.triaxiality_cutoff,
            "evolution": self.evolution,
            "fracture_energy_J_per_m2": self.fracture_energy_J_per_m2,
            "characteristic_length": self.characteristic_length,
            "damage_event_tolerance": self.damage_event_tolerance,
            "maximum_separation_events_per_position": self.maximum_separation_events_per_position,
            "area_absolute_tolerance_m2": self.area_absolute_tolerance_m2,
            "parameter_source": self.parameter_source,
            "calibration_status": self.calibration_status,
            "applicability_notes": self.applicability_notes,
        }
