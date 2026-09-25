"""Strict SI input contract for the independent phase 6A.1 material point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


MATERIAL_POINT_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "small_strain_j2_bilinear_isotropic_hardening"
MIN_STEP_COUNT = 2
MAX_STEP_COUNT = 10_000


class MaterialPointValidationError(ValueError):
    """Raised when a phase 6A.1 material-point input violates its contract."""


def _validation_error(field: str, unit: str, reason: str) -> None:
    raise MaterialPointValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _validation_error(field, "JSON object", "must be an object")
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
        _validation_error(field, "exact field set", " and ".join(details))


def _finite_number(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _validation_error(field, unit, "must be a numeric value, not bool or text")
    try:
        number = float(value)
    except OverflowError:
        _validation_error(
            field,
            unit,
            "must be finite and representable as a floating-point value",
        )
    if not math.isfinite(number):
        _validation_error(field, unit, "must be finite; NaN and infinity are invalid")
    return number


def _positive_number(value: object, field: str, unit: str) -> float:
    number = _finite_number(value, field, unit)
    if number <= 0.0:
        _validation_error(field, unit, "must be strictly greater than zero")
    return number


def _step_count(value: object, field: str) -> int:
    if type(value) is not int:
        _validation_error(
            field,
            "integer step count",
            "must be a strict integer; bool, floating point, and text are invalid",
        )
    if not MIN_STEP_COUNT <= value <= MAX_STEP_COUNT:
        _validation_error(
            field,
            "integer step count",
            f"must be between {MIN_STEP_COUNT} and {MAX_STEP_COUNT} inclusive",
        )
    return value


@dataclass(frozen=True, slots=True)
class BilinearIsotropicMaterial:
    """Small-strain J2 material parameters expressed in SI units."""

    E: float
    nu: float
    yield_strength: float
    tangent_modulus: float

    def __post_init__(self) -> None:
        elastic_modulus = _positive_number(self.E, "material.E", "Pa")
        poisson_ratio = _finite_number(self.nu, "material.nu", "dimensionless")
        if poisson_ratio < 0.0 or poisson_ratio >= 0.5:
            _validation_error(
                "material.nu",
                "dimensionless",
                "must be greater than or equal to zero and less than 0.5",
            )
        yield_strength = _positive_number(
            self.yield_strength, "material.yield_strength", "Pa"
        )
        tangent_modulus = _finite_number(
            self.tangent_modulus, "material.tangent_modulus", "Pa"
        )
        if tangent_modulus < 0.0:
            _validation_error(
                "material.tangent_modulus",
                "Pa",
                "must be greater than or equal to zero",
            )
        if tangent_modulus >= elastic_modulus:
            _validation_error(
                "material.tangent_modulus", "Pa", "must be less than material.E"
            )
        object.__setattr__(self, "E", elastic_modulus)
        object.__setattr__(self, "nu", poisson_ratio)
        object.__setattr__(self, "yield_strength", yield_strength)
        object.__setattr__(self, "tangent_modulus", tangent_modulus)

    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def bulk_modulus(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def internal_hardening_modulus(self) -> float:
        if self.tangent_modulus == 0.0:
            return 0.0
        return self.E * self.tangent_modulus / (self.E - self.tangent_modulus)

    @classmethod
    def from_mapping(cls, payload: object) -> BilinearIsotropicMaterial:
        data = _object(payload, "material")
        _exact_keys(
            data, {"E", "nu", "yield_strength", "tangent_modulus"}, "material"
        )
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return {
            "E": self.E,
            "nu": self.nu,
            "yield_strength": self.yield_strength,
            "tangent_modulus": self.tangent_modulus,
        }


@dataclass(frozen=True, slots=True)
class MaterialPointHistory:
    """Prescribed axial-strain loading and zero-stress unloading controls."""

    peak_axial_strain: float
    loading_step_count: int
    unloading_step_count: int

    def __post_init__(self) -> None:
        peak = _positive_number(
            self.peak_axial_strain, "history.peak_axial_strain", "strain"
        )
        loading = _step_count(
            self.loading_step_count, "history.loading_step_count"
        )
        unloading = _step_count(
            self.unloading_step_count, "history.unloading_step_count"
        )
        object.__setattr__(self, "peak_axial_strain", peak)
        object.__setattr__(self, "loading_step_count", loading)
        object.__setattr__(self, "unloading_step_count", unloading)

    @classmethod
    def from_mapping(cls, payload: object) -> MaterialPointHistory:
        data = _object(payload, "history")
        _exact_keys(
            data,
            {"peak_axial_strain", "loading_step_count", "unloading_step_count"},
            "history",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "peak_axial_strain": self.peak_axial_strain,
            "loading_step_count": self.loading_step_count,
            "unloading_step_count": self.unloading_step_count,
        }


@dataclass(frozen=True, slots=True)
class MaterialPointCase:
    """Independent phase 6A.1 input envelope; it is not a FEM case."""

    material_point_schema_version: int
    unit_system: str
    model_type: str
    material: BilinearIsotropicMaterial
    history: MaterialPointHistory

    def __post_init__(self) -> None:
        if (
            type(self.material_point_schema_version) is not int
            or self.material_point_schema_version != MATERIAL_POINT_SCHEMA_VERSION
        ):
            _validation_error(
                "material_point_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _validation_error(
                "unit_system", "SI", "only the SI unit system is supported"
            )
        if self.model_type != MODEL_TYPE:
            _validation_error(
                "model_type", MODEL_TYPE, "unsupported material-point model"
            )
        if not isinstance(self.material, BilinearIsotropicMaterial):
            _validation_error(
                "material", "BilinearIsotropicMaterial", "has the wrong object type"
            )
        if not isinstance(self.history, MaterialPointHistory):
            _validation_error(
                "history", "MaterialPointHistory", "has the wrong object type"
            )

    @classmethod
    def from_mapping(cls, payload: object) -> MaterialPointCase:
        root = _object(payload, "material_point_case")
        _exact_keys(
            root,
            {
                "material_point_schema_version",
                "unit_system",
                "model_type",
                "material",
                "history",
            },
            "material_point_case",
        )
        return cls(
            material_point_schema_version=root["material_point_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            material=BilinearIsotropicMaterial.from_mapping(root["material"]),
            history=MaterialPointHistory.from_mapping(root["history"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> MaterialPointCase:
        input_path = Path(path).expanduser().resolve()
        try:
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            _validation_error(
                "input_file",
                "existing UTF-8 JSON file",
                f"does not exist: {input_path}",
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
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "material_point_schema_version": self.material_point_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "material": self.material.to_dict(),
            "history": self.history.to_dict(),
        }
