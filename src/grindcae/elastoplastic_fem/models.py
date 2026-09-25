"""Strict SI input contract for Phase 6A.2 fixed-mesh elastoplastic FEM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from grindcae.domain.models import MeshSettings, Rectangle
from grindcae.elastoplastic.models import BilinearIsotropicMaterial


ELASTOPLASTIC_FEM_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "small_strain_plane_strain_j2_fixed_mesh"
MIN_INCREMENT_COUNT = 2
MAX_INCREMENT_COUNT = 10_000
MAX_NEWTON_ITERATIONS = 200
MAX_STEP_RETRIES = 20


class ElastoplasticFemValidationError(ValueError):
    """Raised when a Phase 6A.2 case violates its strict contract."""


def _fail(field: str, unit: str, reason: str) -> None:
    raise ElastoplasticFemValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "JSON object", "must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing fields {missing}")
    if unexpected:
        details.append(f"unexpected fields {unexpected}")
    if details:
        _fail(field, "exact field set", " and ".join(details))


def _number(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, unit, "must be numeric; bool and text are invalid")
    try:
        result = float(value)
    except OverflowError:
        _fail(field, unit, "must be finite and representable")
    if not math.isfinite(result):
        _fail(field, unit, "must be finite; NaN and infinity are invalid")
    return result


def _positive(value: object, field: str, unit: str) -> float:
    result = _number(value, field, unit)
    if result <= 0.0:
        _fail(field, unit, "must be strictly greater than zero")
    return result


def _strict_int(
    value: object, field: str, minimum: int, maximum: int
) -> int:
    if type(value) is not int:
        _fail(field, "integer", "must be a strict integer")
    if not minimum <= value <= maximum:
        _fail(field, "integer", f"must be between {minimum} and {maximum} inclusive")
    return value


@dataclass(frozen=True, slots=True)
class PlaneStrainAnalysis:
    type: str
    thickness: float

    def __post_init__(self) -> None:
        if self.type != "plane_strain":
            _fail("analysis.type", "plane_strain", "only plane_strain is supported")
        object.__setattr__(
            self,
            "thickness",
            _positive(self.thickness, "analysis.thickness", "m"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> PlaneStrainAnalysis:
        data = _object(value, "analysis")
        _exact(data, {"type", "thickness"}, "analysis")
        return cls(type=data["type"], thickness=data["thickness"])

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "thickness": self.thickness}


@dataclass(frozen=True, slots=True)
class FixedBoundary:
    side: str
    components: tuple[str, str]

    def __post_init__(self) -> None:
        if self.side != "bottom":
            _fail("fixed_boundary.side", "bottom", "must be bottom")
        if tuple(self.components) != ("x", "y"):
            _fail(
                "fixed_boundary.components",
                "ordered array [x, y]",
                "must constrain both x and y components",
            )
        object.__setattr__(self, "components", tuple(self.components))

    @classmethod
    def from_mapping(cls, value: object) -> FixedBoundary:
        data = _object(value, "fixed_boundary")
        _exact(data, {"side", "components"}, "fixed_boundary")
        components = data["components"]
        if not isinstance(components, list):
            _fail("fixed_boundary.components", "array", "must be an array")
        return cls(side=data["side"], components=tuple(components))

    def to_dict(self) -> dict[str, object]:
        return {"side": self.side, "components": list(self.components)}


@dataclass(frozen=True, slots=True)
class TopLineLoad:
    boundary: str
    distribution: str
    x_start_m: float
    x_end_m: float
    peak_force_x_N: float
    peak_force_y_N: float

    def __post_init__(self) -> None:
        if self.boundary != "top":
            _fail("load.boundary", "top", "must be the top boundary")
        if self.distribution != "uniform_over_interval":
            _fail(
                "load.distribution",
                "uniform_over_interval",
                "unsupported line-load distribution",
            )
        start = _number(self.x_start_m, "load.x_start_m", "m")
        end = _number(self.x_end_m, "load.x_end_m", "m")
        if end <= start:
            _fail("load.interval", "m", "must have x_end_m greater than x_start_m")
        force_x = _number(self.peak_force_x_N, "load.peak_force_x_N", "N")
        force_y = _number(self.peak_force_y_N, "load.peak_force_y_N", "N")
        if force_y > 0.0:
            _fail(
                "load.peak_force_y_N",
                "N",
                "must be non-positive for compression into the top boundary",
            )
        object.__setattr__(self, "x_start_m", start)
        object.__setattr__(self, "x_end_m", end)
        object.__setattr__(self, "peak_force_x_N", force_x)
        object.__setattr__(self, "peak_force_y_N", force_y)

    @property
    def interval_length_m(self) -> float:
        return self.x_end_m - self.x_start_m

    @property
    def peak_line_load_x_N_per_m(self) -> float:
        return self.peak_force_x_N / self.interval_length_m

    @property
    def peak_line_load_y_N_per_m(self) -> float:
        return self.peak_force_y_N / self.interval_length_m

    @classmethod
    def from_mapping(cls, value: object) -> TopLineLoad:
        data = _object(value, "load")
        _exact(
            data,
            {
                "boundary",
                "distribution",
                "x_start_m",
                "x_end_m",
                "peak_force_x_N",
                "peak_force_y_N",
            },
            "load",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "distribution": self.distribution,
            "x_start_m": self.x_start_m,
            "x_end_m": self.x_end_m,
            "peak_force_x_N": self.peak_force_x_N,
            "peak_force_y_N": self.peak_force_y_N,
        }


@dataclass(frozen=True, slots=True)
class IncrementSettings:
    loading_step_count: int
    unloading_step_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loading_step_count",
            _strict_int(
                self.loading_step_count,
                "increments.loading_step_count",
                MIN_INCREMENT_COUNT,
                MAX_INCREMENT_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "unloading_step_count",
            _strict_int(
                self.unloading_step_count,
                "increments.unloading_step_count",
                MIN_INCREMENT_COUNT,
                MAX_INCREMENT_COUNT,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> IncrementSettings:
        data = _object(value, "increments")
        _exact(data, {"loading_step_count", "unloading_step_count"}, "increments")
        return cls(**data)

    def to_dict(self) -> dict[str, int]:
        return {
            "loading_step_count": self.loading_step_count,
            "unloading_step_count": self.unloading_step_count,
        }


@dataclass(frozen=True, slots=True)
class NewtonSettings:
    residual_relative_tolerance: float
    residual_absolute_tolerance_N: float
    displacement_relative_tolerance: float
    displacement_absolute_tolerance_m: float
    maximum_iterations: int

    def __post_init__(self) -> None:
        for name, unit in (
            ("residual_relative_tolerance", "dimensionless"),
            ("residual_absolute_tolerance_N", "N"),
            ("displacement_relative_tolerance", "dimensionless"),
            ("displacement_absolute_tolerance_m", "m"),
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), f"newton.{name}", unit))
        object.__setattr__(
            self,
            "maximum_iterations",
            _strict_int(
                self.maximum_iterations,
                "newton.maximum_iterations",
                1,
                MAX_NEWTON_ITERATIONS,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> NewtonSettings:
        data = _object(value, "newton")
        _exact(
            data,
            {
                "residual_relative_tolerance",
                "residual_absolute_tolerance_N",
                "displacement_relative_tolerance",
                "displacement_absolute_tolerance_m",
                "maximum_iterations",
            },
            "newton",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "residual_relative_tolerance": self.residual_relative_tolerance,
            "residual_absolute_tolerance_N": self.residual_absolute_tolerance_N,
            "displacement_relative_tolerance": self.displacement_relative_tolerance,
            "displacement_absolute_tolerance_m": self.displacement_absolute_tolerance_m,
            "maximum_iterations": self.maximum_iterations,
        }


@dataclass(frozen=True, slots=True)
class StepControlSettings:
    allow_reduction: bool
    minimum_load_factor_increment: float
    maximum_retries: int

    def __post_init__(self) -> None:
        if type(self.allow_reduction) is not bool:
            _fail("step_control.allow_reduction", "bool", "must be a strict bool")
        minimum = _positive(
            self.minimum_load_factor_increment,
            "step_control.minimum_load_factor_increment",
            "load factor",
        )
        if minimum > 1.0:
            _fail(
                "step_control.minimum_load_factor_increment",
                "load factor",
                "must not exceed 1",
            )
        object.__setattr__(self, "minimum_load_factor_increment", minimum)
        object.__setattr__(
            self,
            "maximum_retries",
            _strict_int(
                self.maximum_retries,
                "step_control.maximum_retries",
                0,
                MAX_STEP_RETRIES,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> StepControlSettings:
        data = _object(value, "step_control")
        _exact(
            data,
            {"allow_reduction", "minimum_load_factor_increment", "maximum_retries"},
            "step_control",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_reduction": self.allow_reduction,
            "minimum_load_factor_increment": self.minimum_load_factor_increment,
            "maximum_retries": self.maximum_retries,
        }


@dataclass(frozen=True, slots=True)
class ElastoplasticFemCase:
    elastoplastic_fem_schema_version: int
    unit_system: str
    model_type: str
    geometry: Rectangle
    material: BilinearIsotropicMaterial
    mesh: MeshSettings
    analysis: PlaneStrainAnalysis
    fixed_boundary: FixedBoundary
    load: TopLineLoad
    increments: IncrementSettings
    newton: NewtonSettings
    step_control: StepControlSettings

    def __post_init__(self) -> None:
        if (
            type(self.elastoplastic_fem_schema_version) is not int
            or self.elastoplastic_fem_schema_version != ELASTOPLASTIC_FEM_SCHEMA_VERSION
        ):
            _fail(
                "elastoplastic_fem_schema_version",
                "integer version",
                "must be the strict integer 1",
            )
        if self.unit_system != UNIT_SYSTEM:
            _fail("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _fail("model_type", MODEL_TYPE, "unsupported model type")
        if not 0.0 <= self.load.x_start_m < self.load.x_end_m <= self.geometry.width:
            _fail(
                "load.interval",
                "m",
                "must be a positive interval inside the top boundary [0, geometry.width]",
            )

    @classmethod
    def from_mapping(cls, value: object) -> ElastoplasticFemCase:
        root = _object(value, "elastoplastic_fem_case")
        _exact(
            root,
            {
                "elastoplastic_fem_schema_version",
                "unit_system",
                "model_type",
                "geometry",
                "material",
                "mesh",
                "analysis",
                "fixed_boundary",
                "load",
                "increments",
                "newton",
                "step_control",
            },
            "elastoplastic_fem_case",
        )
        geometry = _object(root["geometry"], "geometry")
        _exact(geometry, {"type", "width", "height"}, "geometry")
        if geometry["type"] != "rectangle":
            _fail("geometry.type", "rectangle", "must be rectangle")
        mesh = _object(root["mesh"], "mesh")
        _exact(mesh, {"target_size"}, "mesh")
        try:
            rectangle = Rectangle(
                width=_positive(geometry["width"], "geometry.width", "m"),
                height=_positive(geometry["height"], "geometry.height", "m"),
            )
            mesh_settings = MeshSettings(
                target_size=_positive(mesh["target_size"], "mesh.target_size", "m")
            )
            material = BilinearIsotropicMaterial.from_mapping(root["material"])
        except ValueError as exc:
            raise ElastoplasticFemValidationError(str(exc)) from exc
        return cls(
            elastoplastic_fem_schema_version=root["elastoplastic_fem_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            geometry=rectangle,
            material=material,
            mesh=mesh_settings,
            analysis=PlaneStrainAnalysis.from_mapping(root["analysis"]),
            fixed_boundary=FixedBoundary.from_mapping(root["fixed_boundary"]),
            load=TopLineLoad.from_mapping(root["load"]),
            increments=IncrementSettings.from_mapping(root["increments"]),
            newton=NewtonSettings.from_mapping(root["newton"]),
            step_control=StepControlSettings.from_mapping(root["step_control"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ElastoplasticFemCase:
        source = Path(path).expanduser().resolve()
        try:
            with source.open(encoding="utf-8") as stream:
                value = json.load(stream)
        except FileNotFoundError:
            _fail("input_file", "existing UTF-8 JSON", f"does not exist: {source}")
        except json.JSONDecodeError as exc:
            _fail(
                "input_file",
                "valid UTF-8 JSON",
                f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            )
        except (UnicodeDecodeError, OSError) as exc:
            _fail("input_file", "readable UTF-8 JSON", f"cannot be read: {exc}")
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "elastoplastic_fem_schema_version": self.elastoplastic_fem_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "geometry": {
                "type": "rectangle",
                "width": self.geometry.width,
                "height": self.geometry.height,
            },
            "material": self.material.to_dict(),
            "mesh": {"target_size": self.mesh.target_size},
            "analysis": self.analysis.to_dict(),
            "fixed_boundary": self.fixed_boundary.to_dict(),
            "load": self.load.to_dict(),
            "increments": self.increments.to_dict(),
            "newton": self.newton.to_dict(),
            "step_control": self.step_control.to_dict(),
        }
