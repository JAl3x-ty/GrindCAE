"""Strict phase 4C3A input contract for an evolved-surface mesh."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from grindcae.domain.models import MeshSettings, ModelValidationError
from grindcae.trajectory import TrajectoryCase, TrajectoryValidationError


EVOLVED_MESH_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_pass_evolved_surface_gmsh"


class EvolvedMeshValidationError(ValueError):
    """Raised when a phase 4C3A input violates its independent schema."""


def _error(field: str, unit: str, reason: str) -> None:
    raise EvolvedMeshValidationError(f"{field} [{unit}]: {reason}")


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
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    if details:
        _error(field, "exact fields", " and ".join(details))


@dataclass(frozen=True, slots=True)
class EvolvedMeshCase:
    """Validated geometry-and-mesh-only phase 4C3A input."""

    evolved_mesh_schema_version: int
    unit_system: str
    model_type: str
    trajectory: TrajectoryCase
    mesh: MeshSettings

    def __post_init__(self) -> None:
        if (
            type(self.evolved_mesh_schema_version) is not int
            or self.evolved_mesh_schema_version != EVOLVED_MESH_SCHEMA_VERSION
        ):
            _error(
                "evolved_mesh_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _error("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _error("model_type", MODEL_TYPE, "unsupported evolved-mesh model")
        if not isinstance(self.trajectory, TrajectoryCase):
            _error("trajectory", "TrajectoryCase", "has the wrong object type")
        if not isinstance(self.mesh, MeshSettings):
            _error("mesh", "MeshSettings", "has the wrong object type")

    @classmethod
    def from_mapping(cls, payload: object) -> EvolvedMeshCase:
        root = _object(payload, "case")
        _exact_keys(
            root,
            {
                "evolved_mesh_schema_version",
                "unit_system",
                "model_type",
                "trajectory",
                "mesh",
            },
            "case",
        )
        mesh_data = _object(root["mesh"], "mesh")
        _exact_keys(mesh_data, {"target_size"}, "mesh")
        try:
            trajectory = TrajectoryCase.from_mapping(root["trajectory"])
            mesh = MeshSettings(target_size=mesh_data["target_size"])
        except (TrajectoryValidationError, ModelValidationError) as exc:
            raise EvolvedMeshValidationError(str(exc)) from exc
        return cls(
            evolved_mesh_schema_version=root["evolved_mesh_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            trajectory=trajectory,
            mesh=mesh,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> EvolvedMeshCase:
        input_path = Path(path).expanduser()
        if not input_path.is_file():
            _error("input_file", "readable JSON file", f"does not exist: {input_path}")
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _error("input_file", "UTF-8 JSON", f"could not be read: {exc}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "evolved_mesh_schema_version": self.evolved_mesh_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "trajectory": self.trajectory.to_dict(),
            "mesh": {"target_size": self.mesh.target_size},
        }
