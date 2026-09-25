"""Strict schema version 4 input models for local grinding-load solves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from grindcae.domain import (
    AnalysisSettings,
    Boundary,
    BoundaryRole,
    BoundarySide,
    LinearElasticMaterial,
    MeshSettings,
    ModelValidationError,
    Rectangle,
)
from grindcae.force_model import ForceModelValidationError, GrindingForceCase


SCHEMA_VERSION = 4
UNIT_SYSTEM = "SI"
CONTACT_LENGTH_MODEL = "geometric_sqrt_wheel_diameter_depth"
CONTACT_DISTRIBUTION = "uniform"
TANGENTIAL_DIRECTIONS = ("positive_x", "negative_x")


class GrindingCaseValidationError(ValueError):
    """Raised when a schema version 4 grinding case violates its contract."""


def _validation_error(field: str, unit: str, reason: str) -> None:
    raise GrindingCaseValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _validation_error(field, "JSON object", "must be an object")
    return value


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        _validation_error(field, "JSON array", "must be an array")
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


def _finite_number(value: object, field: str, unit: str) -> float:
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
    return number


def _literal(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        _validation_error(field, "string literal", "must be a string")
    if value not in allowed:
        _validation_error(field, "string literal", f"must be one of: {', '.join(allowed)}")
    return value


def _wrap_domain_error(
    field: str,
    unit: str,
    constructor: Any,
    field_units: Mapping[str, str] | None = None,
) -> Any:
    try:
        return constructor()
    except OverflowError:
        if field_units is not None and len(field_units) == 1:
            reported_field, reported_unit = next(iter(field_units.items()))
            _validation_error(
                reported_field,
                reported_unit,
                "must be finite and representable as a floating-point value",
            )
        _validation_error(
            field,
            unit,
            "must contain finite values representable as floating-point numbers",
        )
    except ModelValidationError as exc:
        message = str(exc)
        reported_field, separator, reason = message.partition(" ")
        if separator and field_units is not None and reported_field in field_units:
            _validation_error(reported_field, field_units[reported_field], reason)
        _validation_error(field, unit, message)


def _parse_boundary_side(value: object, field: str) -> BoundarySide:
    if not isinstance(value, str):
        _validation_error(field, "rectangle side", "must be a string")
    try:
        return BoundarySide(value)
    except ValueError:
        allowed = ", ".join(side.value for side in BoundarySide)
        _validation_error(field, "rectangle side", f"must be one of: {allowed}")


def _parse_boundary_role(value: object, field: str) -> BoundaryRole:
    if not isinstance(value, str):
        _validation_error(field, "boundary role", "must be a string")
    try:
        return BoundaryRole(value)
    except ValueError:
        allowed = ", ".join(role.value for role in BoundaryRole)
        _validation_error(field, "boundary role", f"must be one of: {allowed}")


def _validate_boundary_layout(boundaries: tuple[Boundary, ...]) -> None:
    by_side: dict[BoundarySide, Boundary] = {}
    for boundary in boundaries:
        if boundary.side in by_side:
            _validation_error(
                "boundaries",
                "rectangle boundary assignments",
                f"boundary side {boundary.side.value} is duplicated",
            )
        by_side[boundary.side] = boundary

    expected_sides = set(BoundarySide)
    if set(by_side) != expected_sides:
        missing = sorted(side.value for side in expected_sides - set(by_side))
        _validation_error(
            "boundaries",
            "rectangle boundary assignments",
            f"must define every rectangle side; missing {missing}",
        )
    expected_roles = {
        BoundarySide.BOTTOM: BoundaryRole.FIXED,
        BoundarySide.TOP: BoundaryRole.CONTACT,
        BoundarySide.LEFT: BoundaryRole.FREE,
        BoundarySide.RIGHT: BoundaryRole.FREE,
    }
    for side, expected_role in expected_roles.items():
        actual_role = by_side[side].role
        if actual_role is not expected_role:
            _validation_error(
                f"boundaries.{side.value}.role",
                "boundary role",
                f"must be {expected_role.value}",
            )


def _force_model_path(message: str) -> str:
    prefix, separator, suffix = message.partition(" [")
    if not separator:
        return f"grinding.force_model [{message}]"
    if prefix == "force_model_case":
        mapped = "grinding.force_model"
    else:
        mapped = f"grinding.force_model.{prefix}"
    return f"{mapped} [{suffix}"


def _force_model_root_unit_system(payload: object) -> object:
    if isinstance(payload, Mapping):
        return payload.get("unit_system")
    return None


@dataclass(frozen=True, slots=True)
class GrindingContactZone:
    """Requested local interval and force-direction convention on the top edge."""

    boundary: BoundarySide
    center_x_m: float
    length_model: str
    distribution: str
    tangential_direction: str

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, BoundarySide):
            _validation_error(
                "grinding.contact_zone.boundary",
                "rectangle side",
                "must be a BoundarySide",
            )
        if self.boundary is not BoundarySide.TOP:
            _validation_error(
                "grinding.contact_zone.boundary",
                "rectangle side",
                "must be top",
            )
        center = _finite_number(
            self.center_x_m, "grinding.contact_zone.center_x_m", "m"
        )
        length_model = _literal(
            self.length_model,
            "grinding.contact_zone.length_model",
            (CONTACT_LENGTH_MODEL,),
        )
        distribution = _literal(
            self.distribution,
            "grinding.contact_zone.distribution",
            (CONTACT_DISTRIBUTION,),
        )
        direction = _literal(
            self.tangential_direction,
            "grinding.contact_zone.tangential_direction",
            TANGENTIAL_DIRECTIONS,
        )
        object.__setattr__(self, "center_x_m", center)
        object.__setattr__(self, "length_model", length_model)
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "tangential_direction", direction)

    @classmethod
    def from_mapping(cls, payload: object) -> GrindingContactZone:
        field = "grinding.contact_zone"
        data = _object(payload, field)
        _exact_keys(
            data,
            {
                "boundary",
                "center_x_m",
                "length_model",
                "distribution",
                "tangential_direction",
            },
            field,
        )
        return cls(
            boundary=_parse_boundary_side(data["boundary"], f"{field}.boundary"),
            center_x_m=data["center_x_m"],
            length_model=data["length_model"],
            distribution=data["distribution"],
            tangential_direction=data["tangential_direction"],
        )

    def to_dict(self) -> dict[str, object]:
        """Return the normalized persisted contact-zone object."""

        return {
            "boundary": self.boundary.value,
            "center_x_m": self.center_x_m,
            "length_model": self.length_model,
            "distribution": self.distribution,
            "tangential_direction": self.tangential_direction,
        }


@dataclass(frozen=True, slots=True)
class GrindingSimulationCase:
    """Validated schema version 4 case coupling a force model to FEM inputs."""

    schema_version: int
    unit_system: str
    geometry: Rectangle
    material: LinearElasticMaterial
    mesh: MeshSettings
    analysis: AnalysisSettings
    boundaries: tuple[Boundary, ...]
    force_model: GrindingForceCase
    contact_zone: GrindingContactZone

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            _validation_error(
                "schema_version",
                "integer version",
                "must be the strict integer 4; bool, 4.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _validation_error("unit_system", "SI", "only SI is supported")
        typed_fields = (
            ("geometry", self.geometry, Rectangle),
            ("material", self.material, LinearElasticMaterial),
            ("mesh", self.mesh, MeshSettings),
            ("analysis", self.analysis, AnalysisSettings),
            ("grinding.force_model", self.force_model, GrindingForceCase),
            ("grinding.contact_zone", self.contact_zone, GrindingContactZone),
        )
        for field, value, expected_type in typed_fields:
            if not isinstance(value, expected_type):
                _validation_error(
                    field, expected_type.__name__, "has the wrong object type"
                )

        boundaries = tuple(self.boundaries)
        if not all(isinstance(boundary, Boundary) for boundary in boundaries):
            _validation_error(
                "boundaries", "Boundary array", "must contain Boundary values"
            )
        _validate_boundary_layout(boundaries)

        if self.force_model.unit_system != self.unit_system:
            _validation_error(
                "grinding.force_model.unit_system",
                "SI",
                "must match the root unit_system",
            )
        grinding_width = self.force_model.process.grinding_width_m
        thickness = self.analysis.thickness
        if not math.isclose(grinding_width, thickness, rel_tol=1.0e-10, abs_tol=0.0):
            _validation_error(
                "grinding.force_model.process.grinding_width_m",
                "m",
                "must match analysis.thickness within rel_tol=1e-10 and abs_tol=0; silent force rescaling is not allowed",
            )
        object.__setattr__(self, "boundaries", boundaries)

    def boundary_role(self, side: BoundarySide | str) -> BoundaryRole:
        """Return the configured role for a rectangle side."""

        try:
            boundary_side = side if isinstance(side, BoundarySide) else BoundarySide(side)
        except ValueError as exc:
            _validation_error("boundary.side", "rectangle side", f"unknown side: {side}")
            raise AssertionError from exc
        return next(
            boundary.role
            for boundary in self.boundaries
            if boundary.side is boundary_side
        )

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic normalized schema version 4 payload."""

        return {
            "schema_version": self.schema_version,
            "unit_system": self.unit_system,
            "geometry": {
                "type": "rectangle",
                "width": self.geometry.width,
                "height": self.geometry.height,
            },
            "material": {"E": self.material.E, "nu": self.material.nu},
            "mesh": {"target_size": self.mesh.target_size},
            "analysis": {
                "type": self.analysis.type,
                "thickness": self.analysis.thickness,
            },
            "boundaries": [
                {"side": boundary.side.value, "role": boundary.role.value}
                for boundary in self.boundaries
            ],
            "grinding": {
                "force_model": self.force_model.to_dict(),
                "contact_zone": self.contact_zone.to_dict(),
            },
        }

    @classmethod
    def from_json(cls, path: str | Path) -> GrindingSimulationCase:
        """Load and validate a schema version 4 case from UTF-8 JSON."""

        input_path = Path(path).expanduser().resolve()
        try:
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
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
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: object) -> GrindingSimulationCase:
        """Build and validate a schema version 4 case from decoded JSON data."""

        root = _object(payload, "case")
        _exact_keys(
            root,
            {
                "schema_version",
                "unit_system",
                "geometry",
                "material",
                "mesh",
                "analysis",
                "boundaries",
                "grinding",
            },
            "case",
        )
        if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
            _validation_error(
                "schema_version",
                "integer version",
                "must be the strict integer 4; bool, 4.0, and text are invalid",
            )
        if root["unit_system"] != UNIT_SYSTEM:
            _validation_error("unit_system", "SI", "only SI is supported")

        geometry_data = _object(root["geometry"], "geometry")
        _exact_keys(geometry_data, {"type", "width", "height"}, "geometry")
        if geometry_data["type"] != "rectangle":
            _validation_error("geometry.type", "rectangle", "must be rectangle")
        geometry_width = _finite_number(
            geometry_data["width"], "geometry.width", "m"
        )
        geometry_height = _finite_number(
            geometry_data["height"], "geometry.height", "m"
        )
        geometry = _wrap_domain_error(
            "geometry",
            "m",
            lambda: Rectangle(
                width=geometry_width, height=geometry_height
            ),
            {"geometry.width": "m", "geometry.height": "m"},
        )

        material_data = _object(root["material"], "material")
        _exact_keys(material_data, {"E", "nu"}, "material")
        elastic_modulus = _finite_number(material_data["E"], "material.E", "Pa")
        poisson_ratio = _finite_number(
            material_data["nu"], "material.nu", "dimensionless"
        )
        material = _wrap_domain_error(
            "material",
            "E in Pa; nu dimensionless",
            lambda: LinearElasticMaterial(E=elastic_modulus, nu=poisson_ratio),
            {"material.E": "Pa", "material.nu": "dimensionless"},
        )

        mesh_data = _object(root["mesh"], "mesh")
        _exact_keys(mesh_data, {"target_size"}, "mesh")
        target_size = _finite_number(
            mesh_data["target_size"], "mesh.target_size", "m"
        )
        mesh = _wrap_domain_error(
            "mesh.target_size",
            "m",
            lambda: MeshSettings(target_size=target_size),
            {"mesh.target_size": "m"},
        )

        analysis_data = _object(root["analysis"], "analysis")
        _exact_keys(analysis_data, {"type", "thickness"}, "analysis")
        analysis_thickness = _finite_number(
            analysis_data["thickness"], "analysis.thickness", "m"
        )
        analysis = _wrap_domain_error(
            "analysis",
            "type plane_stress; thickness in m",
            lambda: AnalysisSettings(
                type=analysis_data["type"], thickness=analysis_thickness
            ),
            {"analysis.type": "plane_stress", "analysis.thickness": "m"},
        )

        boundaries: list[Boundary] = []
        for index, raw_boundary in enumerate(_array(root["boundaries"], "boundaries")):
            field = f"boundaries[{index}]"
            boundary_data = _object(raw_boundary, field)
            _exact_keys(boundary_data, {"side", "role"}, field)
            boundaries.append(
                Boundary(
                    side=_parse_boundary_side(boundary_data["side"], f"{field}.side"),
                    role=_parse_boundary_role(boundary_data["role"], f"{field}.role"),
                )
            )

        grinding_data = _object(root["grinding"], "grinding")
        _exact_keys(grinding_data, {"force_model", "contact_zone"}, "grinding")
        force_model_data = _object(
            grinding_data["force_model"], "grinding.force_model"
        )
        _exact_keys(
            force_model_data,
            {
                "force_model_schema_version",
                "unit_system",
                "model_type",
                "process",
                "calibration",
            },
            "grinding.force_model",
        )
        if force_model_data["unit_system"] != root["unit_system"]:
            _validation_error(
                "grinding.force_model.unit_system",
                "SI",
                "must match the root unit_system",
            )
        try:
            force_model = GrindingForceCase.from_dict(force_model_data)
        except ForceModelValidationError as exc:
            raise GrindingCaseValidationError(_force_model_path(str(exc))) from exc

        return cls(
            schema_version=root["schema_version"],
            unit_system=root["unit_system"],
            geometry=geometry,
            material=material,
            mesh=mesh,
            analysis=analysis,
            boundaries=tuple(boundaries),
            force_model=force_model,
            contact_zone=GrindingContactZone.from_mapping(
                grinding_data["contact_zone"]
            ),
        )
