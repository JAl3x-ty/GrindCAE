"""Strict SI input contract for the independent phase 4A force model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


FORCE_MODEL_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "specific_grinding_energy_force_ratio"


class ForceModelValidationError(ValueError):
    """Raised when a phase 4A force-model input violates its contract."""


def _validation_error(field: str, unit: str, reason: str) -> None:
    raise ForceModelValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _validation_error(field, "JSON object", "must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing fields {missing}")
    if unexpected:
        details.append(f"unexpected fields {unexpected}")
    if details:
        _validation_error(field, "exact field set", " and ".join(details))


def _positive_number(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _validation_error(field, unit, "must be a numeric value, not bool or text")
    try:
        number = float(value)
    except OverflowError:
        _validation_error(
            field, unit, "must be finite and representable as a floating-point value"
        )
    if not math.isfinite(number):
        _validation_error(field, unit, "must be finite; NaN and infinity are invalid")
    if number <= 0.0:
        _validation_error(field, unit, "must be strictly greater than zero")
    return number


def _nonempty_string(value: object, field: str, meaning: str) -> str:
    if not isinstance(value, str):
        _validation_error(field, meaning, "must be a string")
    normalized = value.strip()
    if not normalized:
        _validation_error(field, meaning, "must not be empty or whitespace")
    return normalized


@dataclass(frozen=True, slots=True)
class GrindingProcessParameters:
    """Grinding process inputs, all expressed in persisted SI units."""

    wheel_surface_speed_m_per_s: float
    workpiece_feed_speed_m_per_s: float
    depth_of_cut_m: float
    grinding_width_m: float
    wheel_diameter_m: float

    def __post_init__(self) -> None:
        fields = (
            ("wheel_surface_speed_m_per_s", "m/s"),
            ("workpiece_feed_speed_m_per_s", "m/s"),
            ("depth_of_cut_m", "m"),
            ("grinding_width_m", "m"),
            ("wheel_diameter_m", "m"),
        )
        for name, unit in fields:
            value = _positive_number(getattr(self, name), f"process.{name}", unit)
            object.__setattr__(self, name, value)
        if self.depth_of_cut_m >= self.wheel_diameter_m / 2.0:
            _validation_error(
                "process.depth_of_cut_m",
                "m",
                "must be less than process.wheel_diameter_m / 2",
            )

    @classmethod
    def from_dict(cls, payload: object) -> GrindingProcessParameters:
        data = _object(payload, "process")
        _exact_keys(
            data,
            {
                "wheel_surface_speed_m_per_s",
                "workpiece_feed_speed_m_per_s",
                "depth_of_cut_m",
                "grinding_width_m",
                "wheel_diameter_m",
            },
            "process",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return {
            "wheel_surface_speed_m_per_s": self.wheel_surface_speed_m_per_s,
            "workpiece_feed_speed_m_per_s": self.workpiece_feed_speed_m_per_s,
            "depth_of_cut_m": self.depth_of_cut_m,
            "grinding_width_m": self.grinding_width_m,
            "wheel_diameter_m": self.wheel_diameter_m,
        }


@dataclass(frozen=True, slots=True)
class GrindingForceCalibration:
    """User-supplied empirical calibration and its engineering provenance."""

    specific_grinding_energy_J_per_m3: float
    normal_to_tangential_force_ratio: float
    calibration_id: str
    specific_grinding_energy_source: str
    force_ratio_source: str
    applicability_notes: str

    def __post_init__(self) -> None:
        specific_energy = _positive_number(
            self.specific_grinding_energy_J_per_m3,
            "calibration.specific_grinding_energy_J_per_m3",
            "J/m^3",
        )
        force_ratio = _positive_number(
            self.normal_to_tangential_force_ratio,
            "calibration.normal_to_tangential_force_ratio",
            "dimensionless Fn/Ft",
        )
        object.__setattr__(
            self, "specific_grinding_energy_J_per_m3", specific_energy
        )
        object.__setattr__(
            self, "normal_to_tangential_force_ratio", force_ratio
        )
        text_fields = (
            ("calibration_id", "calibration identifier"),
            ("specific_grinding_energy_source", "calibration source"),
            ("force_ratio_source", "calibration source"),
            ("applicability_notes", "applicability statement"),
        )
        for name, meaning in text_fields:
            value = _nonempty_string(
                getattr(self, name), f"calibration.{name}", meaning
            )
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, payload: object) -> GrindingForceCalibration:
        data = _object(payload, "calibration")
        _exact_keys(
            data,
            {
                "specific_grinding_energy_J_per_m3",
                "normal_to_tangential_force_ratio",
                "calibration_id",
                "specific_grinding_energy_source",
                "force_ratio_source",
                "applicability_notes",
            },
            "calibration",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "specific_grinding_energy_J_per_m3": (
                self.specific_grinding_energy_J_per_m3
            ),
            "normal_to_tangential_force_ratio": (
                self.normal_to_tangential_force_ratio
            ),
            "calibration_id": self.calibration_id,
            "specific_grinding_energy_source": (
                self.specific_grinding_energy_source
            ),
            "force_ratio_source": self.force_ratio_source,
            "applicability_notes": self.applicability_notes,
        }

    def provenance_dict(self) -> dict[str, float | str]:
        return {
            "calibration_id": self.calibration_id,
            "specific_grinding_energy_J_per_m3": (
                self.specific_grinding_energy_J_per_m3
            ),
            "specific_grinding_energy_source": (
                self.specific_grinding_energy_source
            ),
            "normal_to_tangential_force_ratio": (
                self.normal_to_tangential_force_ratio
            ),
            "force_ratio_source": self.force_ratio_source,
            "applicability_notes": self.applicability_notes,
        }


@dataclass(frozen=True, slots=True)
class GrindingForceCase:
    """Independent phase 4A input envelope; it is not a SimulationCase."""

    force_model_schema_version: int
    unit_system: str
    model_type: str
    process: GrindingProcessParameters
    calibration: GrindingForceCalibration

    def __post_init__(self) -> None:
        if (
            type(self.force_model_schema_version) is not int
            or self.force_model_schema_version != FORCE_MODEL_SCHEMA_VERSION
        ):
            _validation_error(
                "force_model_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        unit_system = _nonempty_string(
            self.unit_system, "unit_system", "literal SI unit-system name"
        )
        if unit_system != UNIT_SYSTEM:
            _validation_error(
                "unit_system", "SI", "only the SI unit system is supported"
            )
        model_type = _nonempty_string(
            self.model_type, "model_type", "supported model identifier"
        )
        if model_type != MODEL_TYPE:
            _validation_error(
                "model_type",
                MODEL_TYPE,
                "unsupported force-model type",
            )
        if not isinstance(self.process, GrindingProcessParameters):
            _validation_error(
                "process", "GrindingProcessParameters", "has the wrong object type"
            )
        if not isinstance(self.calibration, GrindingForceCalibration):
            _validation_error(
                "calibration",
                "GrindingForceCalibration",
                "has the wrong object type",
            )
        object.__setattr__(self, "unit_system", unit_system)
        object.__setattr__(self, "model_type", model_type)

    @classmethod
    def from_dict(cls, payload: object) -> GrindingForceCase:
        root = _object(payload, "force_model_case")
        _exact_keys(
            root,
            {
                "force_model_schema_version",
                "unit_system",
                "model_type",
                "process",
                "calibration",
            },
            "force_model_case",
        )
        return cls(
            force_model_schema_version=root["force_model_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            process=GrindingProcessParameters.from_dict(root["process"]),
            calibration=GrindingForceCalibration.from_dict(root["calibration"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> GrindingForceCase:
        input_path = Path(path).expanduser().resolve()
        try:
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError as exc:
            _validation_error(
                "input_file", "existing UTF-8 JSON file", f"does not exist: {input_path}"
            )
        except json.JSONDecodeError as exc:
            _validation_error(
                "input_file",
                "valid UTF-8 JSON",
                f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            )
        except UnicodeDecodeError as exc:
            _validation_error(
                "input_file", "UTF-8 JSON", f"cannot be decoded as UTF-8: {exc.reason}"
            )
        except OSError as exc:
            _validation_error(
                "input_file", "readable UTF-8 JSON file", f"cannot be read: {exc}"
            )
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "force_model_schema_version": self.force_model_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "process": self.process.to_dict(),
            "calibration": self.calibration.to_dict(),
        }
