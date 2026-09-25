"""Strict SI input contract for Phase 7A.2 statistical grain loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, TypeAlias

from grindcae.mechanism_force import MechanismForceCase

from .grit_tables import GRIT_TABLES


STATISTICAL_GRAIN_LOAD_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "statistical_equivalent_grain_mechanism_load"
SUPPORTED_ABRASIVE_MATERIAL = "aluminum_oxide"
RESOLUTION_PATHS = frozenset(
    {"GB_FEPA_F", "GB_W_micropowder", "JIS_hash", "manufacturer_override"}
)


class StatisticalGrainLoadValidationError(ValueError):
    """Raised when a Phase 7A.2 input violates its strict contract."""


def _error(field: str, unit: str, reason: str) -> None:
    raise StatisticalGrainLoadValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(field, "JSON object", "must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing fields {missing}")
    if unexpected:
        details.append(f"unexpected fields {unexpected}")
    if details:
        _error(field, "exact field set", " and ".join(details))


def _string(value: object, field: str, meaning: str) -> str:
    if not isinstance(value, str):
        _error(field, meaning, "must be a string")
    normalized = value.strip()
    if not normalized:
        _error(field, meaning, "must not be empty or whitespace")
    return normalized


def _positive(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(field, unit, "must be numeric, not bool or text")
    number = float(value)
    if not math.isfinite(number):
        _error(field, unit, "must be finite; NaN and infinity are invalid")
    if number <= 0.0:
        _error(field, unit, "must be strictly greater than zero")
    return number


def _strict_int(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        _error(field, "integer", "must be an integer; bool, float, and text are invalid")
    if value < minimum or value > maximum:
        _error(field, "integer", f"must lie within [{minimum}, {maximum}]")
    return value


def _common_wheel_fields(data: Mapping[str, Any], field: str) -> tuple[str, str, str]:
    raw_path = data["resolution_path"]
    path = _string(raw_path, f"{field}.resolution_path", "resolution path")
    if path != raw_path:
        _error(
            f"{field}.resolution_path",
            "exact resolution path",
            "leading or trailing whitespace is invalid; silent normalization is not allowed",
        )
    if path not in RESOLUTION_PATHS:
        _error(f"{field}.resolution_path", "supported path", f"unsupported path {path!r}")
    abrasive = _string(data["abrasive_material"], f"{field}.abrasive_material", "abrasive material")
    if abrasive != SUPPORTED_ABRASIVE_MATERIAL:
        _error(
            f"{field}.abrasive_material",
            SUPPORTED_ABRASIVE_MATERIAL,
            f"unsupported abrasive {abrasive!r}; only aluminum_oxide is implemented",
        )
    raw_designation = data["grit_designation"]
    designation = _string(raw_designation, f"{field}.grit_designation", "grit designation")
    if designation != raw_designation:
        _error(
            f"{field}.grit_designation",
            "exact grit designation",
            "leading or trailing whitespace is invalid; silent normalization is not allowed",
        )
    return path, abrasive, designation


@dataclass(frozen=True, slots=True)
class TableWheelSpecification:
    resolution_path: str
    abrasive_material: str
    grit_designation: str

    @classmethod
    def from_dict(cls, payload: object) -> TableWheelSpecification:
        data = _object(payload, "wheel_specification")
        _exact_keys(data, {"resolution_path", "abrasive_material", "grit_designation"}, "wheel_specification")
        path, abrasive, designation = _common_wheel_fields(data, "wheel_specification")
        if path == "manufacturer_override":
            _error("wheel_specification.resolution_path", "table path", "manufacturer_override requires its complete exclusive field set")
        if designation not in GRIT_TABLES.get(path, {}):
            _error(
                "wheel_specification.grit_designation",
                "known table entry",
                f"unknown grit designation {designation!r} for {path}; no guessing or extrapolation is allowed",
            )
        return cls(path, abrasive, designation)

    def to_dict(self) -> dict[str, str]:
        return {
            "resolution_path": self.resolution_path,
            "abrasive_material": self.abrasive_material,
            "grit_designation": self.grit_designation,
        }


@dataclass(frozen=True, slots=True)
class ManufacturerWheelSpecification:
    resolution_path: str
    abrasive_material: str
    grit_designation: str
    manufacturer: str
    product_model: str
    diameter_lower_m: float
    representative_diameter_d50_m: float
    diameter_upper_m: float
    range_definition: str
    data_status: str
    source: str

    @classmethod
    def from_dict(cls, payload: object) -> ManufacturerWheelSpecification:
        data = _object(payload, "wheel_specification")
        expected = {
            "resolution_path", "abrasive_material", "grit_designation", "manufacturer",
            "product_model", "diameter_lower_m", "representative_diameter_d50_m",
            "diameter_upper_m", "range_definition", "data_status", "source",
        }
        _exact_keys(data, expected, "wheel_specification")
        path, abrasive, designation = _common_wheel_fields(data, "wheel_specification")
        if path != "manufacturer_override":
            _error("wheel_specification.resolution_path", "manufacturer_override", "manufacturer fields cannot be mixed with a built-in table path")
        lower = _positive(data["diameter_lower_m"], "wheel_specification.diameter_lower_m", "m")
        d50 = _positive(data["representative_diameter_d50_m"], "wheel_specification.representative_diameter_d50_m", "m")
        upper = _positive(data["diameter_upper_m"], "wheel_specification.diameter_upper_m", "m")
        if not lower <= d50 <= upper:
            _error("wheel_specification.diameters", "m", "must satisfy lower <= d50 <= upper")
        return cls(
            resolution_path=path,
            abrasive_material=abrasive,
            grit_designation=designation,
            manufacturer=_string(data["manufacturer"], "wheel_specification.manufacturer", "manufacturer"),
            product_model=_string(data["product_model"], "wheel_specification.product_model", "product model"),
            diameter_lower_m=lower,
            representative_diameter_d50_m=d50,
            diameter_upper_m=upper,
            range_definition=_string(data["range_definition"], "wheel_specification.range_definition", "range definition"),
            data_status=_string(data["data_status"], "wheel_specification.data_status", "source status"),
            source=_string(data["source"], "wheel_specification.source", "source description"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "resolution_path": self.resolution_path,
            "abrasive_material": self.abrasive_material,
            "grit_designation": self.grit_designation,
            "manufacturer": self.manufacturer,
            "product_model": self.product_model,
            "diameter_lower_m": self.diameter_lower_m,
            "representative_diameter_d50_m": self.representative_diameter_d50_m,
            "diameter_upper_m": self.diameter_upper_m,
            "range_definition": self.range_definition,
            "data_status": self.data_status,
            "source": self.source,
        }


WheelSpecification: TypeAlias = TableWheelSpecification | ManufacturerWheelSpecification


def _wheel_specification(payload: object) -> WheelSpecification:
    data = _object(payload, "wheel_specification")
    if "resolution_path" not in data:
        _error("wheel_specification", "exact field set", "missing fields ['resolution_path']")
    return (
        ManufacturerWheelSpecification.from_dict(data)
        if data["resolution_path"] == "manufacturer_override"
        else TableWheelSpecification.from_dict(data)
    )


@dataclass(frozen=True, slots=True)
class GrainPopulationSettings:
    active_density_factor: float
    minimum_equivalent_group_count: int
    maximum_equivalent_group_count: int
    random_seed: int

    @classmethod
    def from_dict(cls, payload: object) -> GrainPopulationSettings:
        data = _object(payload, "grain_population")
        _exact_keys(data, {"active_density_factor", "minimum_equivalent_group_count", "maximum_equivalent_group_count", "random_seed"}, "grain_population")
        minimum = _strict_int(data["minimum_equivalent_group_count"], "grain_population.minimum_equivalent_group_count", minimum=1, maximum=100000)
        maximum = _strict_int(data["maximum_equivalent_group_count"], "grain_population.maximum_equivalent_group_count", minimum=1, maximum=100000)
        if minimum > maximum:
            _error("grain_population.minimum_equivalent_group_count", "integer", "minimum must not exceed maximum")
        return cls(
            active_density_factor=_positive(data["active_density_factor"], "grain_population.active_density_factor", "dimensionless"),
            minimum_equivalent_group_count=minimum,
            maximum_equivalent_group_count=maximum,
            random_seed=_strict_int(data["random_seed"], "grain_population.random_seed", minimum=0, maximum=2**63 - 1),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "active_density_factor": self.active_density_factor,
            "minimum_equivalent_group_count": self.minimum_equivalent_group_count,
            "maximum_equivalent_group_count": self.maximum_equivalent_group_count,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True, slots=True)
class LoadDistributionSettings:
    point_count: int
    kernel_width_factor: float
    rubbing_kernel_scale: float
    ploughing_kernel_scale: float
    cutting_kernel_scale: float

    @classmethod
    def from_dict(cls, payload: object) -> LoadDistributionSettings:
        data = _object(payload, "load_distribution")
        _exact_keys(data, {"point_count", "kernel_width_factor", "rubbing_kernel_scale", "ploughing_kernel_scale", "cutting_kernel_scale"}, "load_distribution")
        count = _strict_int(data["point_count"], "load_distribution.point_count", minimum=201, maximum=5001)
        if count % 2 == 0:
            _error("load_distribution.point_count", "odd integer", "must be odd")
        return cls(
            point_count=count,
            kernel_width_factor=_positive(data["kernel_width_factor"], "load_distribution.kernel_width_factor", "dimensionless"),
            rubbing_kernel_scale=_positive(data["rubbing_kernel_scale"], "load_distribution.rubbing_kernel_scale", "dimensionless"),
            ploughing_kernel_scale=_positive(data["ploughing_kernel_scale"], "load_distribution.ploughing_kernel_scale", "dimensionless"),
            cutting_kernel_scale=_positive(data["cutting_kernel_scale"], "load_distribution.cutting_kernel_scale", "dimensionless"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "kernel_width_factor": self.kernel_width_factor,
            "rubbing_kernel_scale": self.rubbing_kernel_scale,
            "ploughing_kernel_scale": self.ploughing_kernel_scale,
            "cutting_kernel_scale": self.cutting_kernel_scale,
        }


@dataclass(frozen=True, slots=True)
class StatisticalGrainLoadCase:
    statistical_grain_load_schema_version: int
    unit_system: str
    model_type: str
    mechanism_force: MechanismForceCase
    wheel_specification: WheelSpecification
    grain_population: GrainPopulationSettings
    load_distribution: LoadDistributionSettings

    @classmethod
    def from_dict(cls, payload: object) -> StatisticalGrainLoadCase:
        root = _object(payload, "statistical_grain_load_case")
        _exact_keys(root, {"statistical_grain_load_schema_version", "unit_system", "model_type", "mechanism_force", "wheel_specification", "grain_population", "load_distribution"}, "statistical_grain_load_case")
        version = root["statistical_grain_load_schema_version"]
        if type(version) is not int or version != STATISTICAL_GRAIN_LOAD_SCHEMA_VERSION:
            _error("statistical_grain_load_schema_version", "integer 1", "must be the strict integer 1")
        unit = _string(root["unit_system"], "unit_system", "unit system")
        if unit != UNIT_SYSTEM:
            _error("unit_system", UNIT_SYSTEM, "only SI is supported")
        model = _string(root["model_type"], "model_type", "model identifier")
        if model != MODEL_TYPE:
            _error("model_type", MODEL_TYPE, "unsupported model type")
        return cls(
            statistical_grain_load_schema_version=version,
            unit_system=unit,
            model_type=model,
            mechanism_force=MechanismForceCase.from_dict(root["mechanism_force"]),
            wheel_specification=_wheel_specification(root["wheel_specification"]),
            grain_population=GrainPopulationSettings.from_dict(root["grain_population"]),
            load_distribution=LoadDistributionSettings.from_dict(root["load_distribution"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> StatisticalGrainLoadCase:
        input_path = Path(path).expanduser().resolve()
        try:
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _error("input_file", "readable UTF-8 JSON", str(exc))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "statistical_grain_load_schema_version": self.statistical_grain_load_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "mechanism_force": self.mechanism_force.to_dict(),
            "wheel_specification": self.wheel_specification.to_dict(),
            "grain_population": self.grain_population.to_dict(),
            "load_distribution": self.load_distribution.to_dict(),
        }
