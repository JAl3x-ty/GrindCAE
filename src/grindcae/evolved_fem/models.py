"""Strict phase 4C3B input contract for evolved-surface empirical FEM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
from grindcae.pass_loading import PassLoadCase, PassLoadValidationError


EVOLVED_FEM_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_pass_evolved_surface_empirical_load_plane_stress"


class EvolvedFemValidationError(ValueError):
    """Raised when a phase 4C3B input violates its independent schema."""


def _error(field: str, unit: str, reason: str) -> None:
    raise EvolvedFemValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(field, "JSON object", "must be an object")
    return value


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        _error(field, "JSON array", "must be an array")
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


def _boundary_side(value: object, field: str) -> BoundarySide:
    if not isinstance(value, str):
        _error(field, "boundary side", "must be a string")
    try:
        return BoundarySide(value)
    except ValueError as exc:
        _error(field, "bottom/top/left/right", "has an unsupported value")
        raise AssertionError from exc


def _boundary_role(value: object, field: str) -> BoundaryRole:
    if not isinstance(value, str):
        _error(field, "boundary role", "must be a string")
    try:
        return BoundaryRole(value)
    except ValueError as exc:
        _error(field, "fixed/contact/free", "has an unsupported value")
        raise AssertionError from exc


@dataclass(frozen=True, slots=True)
class EvolvedFemCase:
    """One pass-load input plus the structural settings for phase 4C3B."""

    evolved_fem_schema_version: int
    unit_system: str
    model_type: str
    pass_load: PassLoadCase
    material: LinearElasticMaterial
    mesh: MeshSettings
    analysis: AnalysisSettings
    boundaries: tuple[Boundary, ...]
    geometry: Rectangle = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.evolved_fem_schema_version) is not int
            or self.evolved_fem_schema_version != EVOLVED_FEM_SCHEMA_VERSION
        ):
            _error(
                "evolved_fem_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _error("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _error("model_type", MODEL_TYPE, "unsupported evolved-FEM model")
        if not isinstance(self.pass_load, PassLoadCase):
            _error("pass_load", "PassLoadCase", "has the wrong object type")
        if self.pass_load.unit_system != self.unit_system:
            _error("pass_load.unit_system", "SI", "must match the root unit system")
        if not isinstance(self.material, LinearElasticMaterial):
            _error("material", "LinearElasticMaterial", "has the wrong object type")
        if not isinstance(self.mesh, MeshSettings):
            _error("mesh", "MeshSettings", "has the wrong object type")
        if not isinstance(self.analysis, AnalysisSettings):
            _error("analysis", "AnalysisSettings", "has the wrong object type")

        boundaries = tuple(self.boundaries)
        by_side: dict[BoundarySide, BoundaryRole] = {}
        for boundary in boundaries:
            if not isinstance(boundary, Boundary):
                _error("boundaries", "Boundary array", "contains an invalid value")
            if boundary.side in by_side:
                _error(
                    f"boundaries.{boundary.side.value}",
                    "unique side",
                    "is duplicated",
                )
            by_side[boundary.side] = boundary.role
        required = {
            BoundarySide.BOTTOM: BoundaryRole.FIXED,
            BoundarySide.TOP: BoundaryRole.CONTACT,
            BoundarySide.LEFT: BoundaryRole.FREE,
            BoundarySide.RIGHT: BoundaryRole.FREE,
        }
        if set(by_side) != set(required):
            missing = sorted(side.value for side in set(required) - set(by_side))
            _error("boundaries", "four complete sides", f"missing sides {missing}")
        for side, role in required.items():
            if by_side[side] is not role:
                _error(
                    f"boundaries.{side.value}.role",
                    role.value,
                    f"must be {role.value}",
                )

        grinding_width = self.pass_load.force_model.process.grinding_width_m
        if not math.isclose(
            grinding_width,
            self.analysis.thickness,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            _error(
                "pass_load.force_model.process.grinding_width_m",
                "m",
                "must match analysis.thickness without averaging or correction",
            )
        trajectory = self.pass_load.trajectory
        object.__setattr__(
            self,
            "geometry",
            Rectangle(
                width=trajectory.workpiece.length_m,
                height=trajectory.workpiece.original_surface_height_m,
            ),
        )
        object.__setattr__(self, "boundaries", boundaries)

    @classmethod
    def from_mapping(cls, payload: object) -> EvolvedFemCase:
        root = _object(payload, "case")
        # 中文导读：先严格核对 JSON 字段集合，再构造各层计算模型。
        _exact_keys(
            root,
            {
                "evolved_fem_schema_version",
                "unit_system",
                "model_type",
                "pass_load",
                "material",
                "mesh",
                "analysis",
                "boundaries",
            },
            "case",
        )
        material_data = _object(root["material"], "material")
        _exact_keys(material_data, {"E", "nu"}, "material")
        mesh_data = _object(root["mesh"], "mesh")
        _exact_keys(mesh_data, {"target_size"}, "mesh")
        analysis_data = _object(root["analysis"], "analysis")
        _exact_keys(analysis_data, {"type", "thickness"}, "analysis")
        boundaries: list[Boundary] = []
        for index, raw_boundary in enumerate(_array(root["boundaries"], "boundaries")):
            field_name = f"boundaries[{index}]"
            boundary_data = _object(raw_boundary, field_name)
            _exact_keys(boundary_data, {"side", "role"}, field_name)
            boundaries.append(
                Boundary(
                    side=_boundary_side(boundary_data["side"], f"{field_name}.side"),
                    role=_boundary_role(boundary_data["role"], f"{field_name}.role"),
                )
            )
        try:
            return cls(
                evolved_fem_schema_version=root["evolved_fem_schema_version"],
                unit_system=root["unit_system"],
                model_type=root["model_type"],
                pass_load=PassLoadCase.from_mapping(root["pass_load"]),
                material=LinearElasticMaterial(**material_data),
                mesh=MeshSettings(**mesh_data),
                analysis=AnalysisSettings(**analysis_data),
                boundaries=tuple(boundaries),
            )
        except (PassLoadValidationError, ModelValidationError) as exc:
            raise EvolvedFemValidationError(str(exc)) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> EvolvedFemCase:
        input_path = Path(path).expanduser().resolve()
        try:
            # 中文导读：持久化输入统一按 UTF-8 读取，解析后仍走同一套 Schema 校验。
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            _error("input_file", "existing UTF-8 JSON file", f"does not exist: {input_path}")
        except json.JSONDecodeError as exc:
            _error(
                "input_file",
                "valid UTF-8 JSON",
                f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            )
        except UnicodeDecodeError as exc:
            _error("input_file", "UTF-8 JSON", f"cannot be decoded: {exc.reason}")
        except OSError as exc:
            _error("input_file", "readable UTF-8 JSON", f"cannot be read: {exc}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "evolved_fem_schema_version": self.evolved_fem_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "pass_load": self.pass_load.to_dict(),
            "material": {"E": self.material.E, "nu": self.material.nu},
            "mesh": {"target_size": self.mesh.target_size},
            "analysis": {
                "type": self.analysis.type,
                "thickness": self.analysis.thickness,
            },
            "boundaries": [
                {"side": item.side.value, "role": item.role.value}
                for item in self.boundaries
            ],
        }
