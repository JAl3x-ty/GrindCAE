"""Phase 2A input models.

The models validate an SI-only rectangular contact case and aggregate supplied
equivalent loads. Analysis settings select the supported 2D plane-stress model
and give its out-of-plane thickness in metres.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, TypeVar


class ModelValidationError(ValueError):
    """Raised when an input does not satisfy the phase 2A data contract."""


class BoundarySide(str, Enum):
    """Named sides of the rectangular workpiece."""

    BOTTOM = "bottom"
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"


class BoundaryRole(str, Enum):
    """Roles assigned to rectangle sides."""

    FIXED = "fixed"
    CONTACT = "contact"
    FREE = "free"


EnumValue = TypeVar("EnumValue", bound=Enum)


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelValidationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ModelValidationError(f"{field} must be finite")
    return number


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelValidationError(f"{field} must be an object")
    return value


def _list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelValidationError(f"{field} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    if details:
        raise ModelValidationError(f"{field} has " + " and ".join(details))


def _enum_member(
    enum_type: type[EnumValue], value: object, field: str
) -> EnumValue:
    if not isinstance(value, str):
        raise ModelValidationError(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ModelValidationError(
            f"{field} must be one of: {allowed}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Rectangle:
    """A 2D rectangular workpiece with dimensions in metres."""

    width: float
    height: float

    def __post_init__(self) -> None:
        width = _finite_number(self.width, "geometry.width")
        height = _finite_number(self.height, "geometry.height")
        if width <= 0.0:
            raise ModelValidationError("geometry.width must be greater than zero")
        if height <= 0.0:
            raise ModelValidationError("geometry.height must be greater than zero")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


@dataclass(frozen=True, slots=True)
class LinearElasticMaterial:
    """Isotropic linear-elastic inputs; E is in pascals and nu is unitless."""

    E: float
    nu: float

    def __post_init__(self) -> None:
        elastic_modulus = _finite_number(self.E, "material.E")
        poisson_ratio = _finite_number(self.nu, "material.nu")
        if elastic_modulus <= 0.0:
            raise ModelValidationError("material.E must be greater than zero")
        if not -1.0 < poisson_ratio < 0.5:
            raise ModelValidationError("material.nu must satisfy -1 < nu < 0.5")
        object.__setattr__(self, "E", elastic_modulus)
        object.__setattr__(self, "nu", poisson_ratio)


@dataclass(frozen=True, slots=True)
class MeshSettings:
    """Settings for a 2D mesh; target_size is measured in metres."""

    target_size: float

    def __post_init__(self) -> None:
        target_size = _finite_number(self.target_size, "mesh.target_size")
        if target_size <= 0.0:
            raise ModelValidationError("mesh.target_size must be greater than zero")
        object.__setattr__(self, "target_size", target_size)


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Plane-stress settings; thickness is the out-of-plane size in metres."""

    type: str
    thickness: float

    def __post_init__(self) -> None:
        if self.type != "plane_stress":
            raise ModelValidationError("analysis.type must be plane_stress")
        thickness = _finite_number(self.thickness, "analysis.thickness")
        if thickness <= 0.0:
            raise ModelValidationError(
                "analysis.thickness must be greater than zero"
            )
        object.__setattr__(self, "thickness", thickness)


@dataclass(frozen=True, slots=True)
class Boundary:
    """A role assigned to one rectangle side."""

    side: BoundarySide
    role: BoundaryRole

    def __post_init__(self) -> None:
        if not isinstance(self.side, BoundarySide):
            raise ModelValidationError("boundary.side must be a BoundarySide")
        if not isinstance(self.role, BoundaryRole):
            raise ModelValidationError("boundary.role must be a BoundaryRole")


@dataclass(frozen=True, slots=True)
class EquivalentContactLoad:
    """A supplied contact-load contribution measured in newtons."""

    boundary: BoundarySide
    normal_force: float
    tangential_force: float

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, BoundarySide):
            raise ModelValidationError("load.boundary must be a BoundarySide")
        normal_force = _finite_number(self.normal_force, "load.normal_force")
        tangential_force = _finite_number(
            self.tangential_force, "load.tangential_force"
        )
        if normal_force < 0.0:
            raise ModelValidationError(
                "load.normal_force must be a non-negative compressive magnitude"
            )
        object.__setattr__(self, "normal_force", normal_force)
        object.__setattr__(self, "tangential_force", tangential_force)


@dataclass(frozen=True, slots=True)
class LoadTotals:
    """Sum of supplied equivalent loads, not finite element reactions."""

    normal_force: float
    tangential_force: float

    @property
    def resultant_force(self) -> float:
        """Return the Euclidean magnitude of the two summed components."""

        return math.hypot(self.normal_force, self.tangential_force)


@dataclass(frozen=True, slots=True)
class SimulationCase:
    """Validated phase 2A simulation input contract."""

    schema_version: int
    unit_system: str
    geometry: Rectangle
    material: LinearElasticMaterial
    mesh: MeshSettings
    analysis: AnalysisSettings
    boundaries: tuple[Boundary, ...]
    loads: tuple[EquivalentContactLoad, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ModelValidationError("schema_version must be 3")
        if self.unit_system != "SI":
            raise ModelValidationError("unit_system must be SI")
        if not isinstance(self.geometry, Rectangle):
            raise ModelValidationError("geometry must be a Rectangle")
        if not isinstance(self.material, LinearElasticMaterial):
            raise ModelValidationError("material must be LinearElasticMaterial")
        if not isinstance(self.mesh, MeshSettings):
            raise ModelValidationError("mesh must be MeshSettings")
        if not isinstance(self.analysis, AnalysisSettings):
            raise ModelValidationError("analysis must be AnalysisSettings")

        boundaries = tuple(self.boundaries)
        loads = tuple(self.loads)
        if not all(isinstance(boundary, Boundary) for boundary in boundaries):
            raise ModelValidationError("boundaries must contain Boundary values")
        if not all(isinstance(load, EquivalentContactLoad) for load in loads):
            raise ModelValidationError(
                "loads must contain EquivalentContactLoad values"
            )

        by_side: dict[BoundarySide, Boundary] = {}
        for boundary in boundaries:
            if boundary.side in by_side:
                raise ModelValidationError(
                    f"boundary side {boundary.side.value} is duplicated"
                )
            by_side[boundary.side] = boundary

        expected_sides = set(BoundarySide)
        if set(by_side) != expected_sides:
            missing = sorted(side.value for side in expected_sides - set(by_side))
            raise ModelValidationError(
                f"boundaries must define every rectangle side; missing {missing}"
            )
        if by_side[BoundarySide.BOTTOM].role is not BoundaryRole.FIXED:
            raise ModelValidationError("bottom boundary must have role fixed")
        if by_side[BoundarySide.TOP].role is not BoundaryRole.CONTACT:
            raise ModelValidationError("top boundary must have role contact")
        for side in (BoundarySide.LEFT, BoundarySide.RIGHT):
            if by_side[side].role is not BoundaryRole.FREE:
                raise ModelValidationError(
                    f"{side.value} boundary must have role free"
                )
        if not loads:
            raise ModelValidationError("loads must contain at least one entry")
        for load in loads:
            if by_side[load.boundary].role is not BoundaryRole.CONTACT:
                raise ModelValidationError(
                    f"load boundary {load.boundary.value} must have role contact"
                )

        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "loads", loads)

    def boundary_role(self, side: BoundarySide | str) -> BoundaryRole:
        """Return the configured role for a rectangle side."""

        try:
            boundary_side = side if isinstance(side, BoundarySide) else BoundarySide(side)
        except ValueError as exc:
            raise ModelValidationError(f"unknown boundary side: {side}") from exc
        return next(
            boundary.role
            for boundary in self.boundaries
            if boundary.side is boundary_side
        )

    def load_totals(self) -> LoadTotals:
        """Sum the supplied normal and tangential equivalent load inputs."""

        return LoadTotals(
            normal_force=math.fsum(load.normal_force for load in self.loads),
            tangential_force=math.fsum(
                load.tangential_force for load in self.loads
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationCase:
        """Load and validate a simulation case from a UTF-8 JSON file."""

        input_path = Path(path)
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelValidationError(
                f"could not read valid JSON from {input_path}"
            ) from exc
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: object) -> SimulationCase:
        """Build and validate a simulation case from decoded JSON data."""

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
                "loads",
            },
            "case",
        )

        geometry_data = _object(root["geometry"], "geometry")
        _exact_keys(geometry_data, {"type", "width", "height"}, "geometry")
        if geometry_data["type"] != "rectangle":
            raise ModelValidationError("geometry.type must be rectangle")
        geometry = Rectangle(
            width=geometry_data["width"],
            height=geometry_data["height"],
        )

        material_data = _object(root["material"], "material")
        _exact_keys(material_data, {"E", "nu"}, "material")
        material = LinearElasticMaterial(
            E=material_data["E"],
            nu=material_data["nu"],
        )

        mesh_data = _object(root["mesh"], "mesh")
        _exact_keys(mesh_data, {"target_size"}, "mesh")
        mesh = MeshSettings(target_size=mesh_data["target_size"])

        analysis_data = _object(root["analysis"], "analysis")
        _exact_keys(analysis_data, {"type", "thickness"}, "analysis")
        analysis = AnalysisSettings(
            type=analysis_data["type"],
            thickness=analysis_data["thickness"],
        )

        boundaries: list[Boundary] = []
        for index, raw_boundary in enumerate(
            _list(root["boundaries"], "boundaries")
        ):
            field = f"boundaries[{index}]"
            boundary_data = _object(raw_boundary, field)
            _exact_keys(boundary_data, {"side", "role"}, field)
            boundaries.append(
                Boundary(
                    side=_enum_member(
                        BoundarySide, boundary_data["side"], f"{field}.side"
                    ),
                    role=_enum_member(
                        BoundaryRole, boundary_data["role"], f"{field}.role"
                    ),
                )
            )

        loads: list[EquivalentContactLoad] = []
        for index, raw_load in enumerate(_list(root["loads"], "loads")):
            field = f"loads[{index}]"
            load_data = _object(raw_load, field)
            _exact_keys(
                load_data,
                {"boundary", "normal_force", "tangential_force"},
                field,
            )
            loads.append(
                EquivalentContactLoad(
                    boundary=_enum_member(
                        BoundarySide, load_data["boundary"], f"{field}.boundary"
                    ),
                    normal_force=load_data["normal_force"],
                    tangential_force=load_data["tangential_force"],
                )
            )

        return cls(
            schema_version=root["schema_version"],
            unit_system=root["unit_system"],
            geometry=geometry,
            material=material,
            mesh=mesh,
            analysis=analysis,
            boundaries=tuple(boundaries),
            loads=tuple(loads),
        )
