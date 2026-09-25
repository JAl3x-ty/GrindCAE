"""Strict Phase 6B.1 input contract for one fixed-mesh moving-load pass."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from grindcae.single_position_elastoplastic import (
    SinglePositionElastoplasticCase,
    SinglePositionElastoplasticValidationError,
)


MOVING_LOAD_PASS_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "fixed_mesh_complete_single_pass_moving_load"
PASS_RANGE = "complete_single_pass"
MIN_BASE_POSITION_COUNT = 5
MAX_BASE_POSITION_COUNT = 501


class MovingLoadPassValidationError(ValueError):
    """Raised when a Phase 6B.1 input violates its independent schema."""


def _fail(field: str, unit: str, reason: str) -> None:
    raise MovingLoadPassValidationError(f"{field} [{unit}]: {reason}")


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


@dataclass(frozen=True, slots=True)
class MovingPassSettings:
    range: str
    base_position_count: int
    retain_key_position_output: bool

    def __post_init__(self) -> None:
        if self.range != PASS_RANGE:
            _fail("pass.range", PASS_RANGE, "unsupported pass range")
        if type(self.base_position_count) is not int:
            _fail("pass.base_position_count", "integer", "must be a strict integer")
        if not MIN_BASE_POSITION_COUNT <= self.base_position_count <= MAX_BASE_POSITION_COUNT:
            _fail(
                "pass.base_position_count",
                "integer",
                f"must be between {MIN_BASE_POSITION_COUNT} and {MAX_BASE_POSITION_COUNT} inclusive",
            )
        if type(self.retain_key_position_output) is not bool:
            _fail(
                "pass.retain_key_position_output",
                "bool",
                "must be a strict bool",
            )

    @classmethod
    def from_mapping(cls, value: object) -> "MovingPassSettings":
        data = _object(value, "pass")
        _exact(
            data,
            {"range", "base_position_count", "retain_key_position_output"},
            "pass",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "range": self.range,
            "base_position_count": self.base_position_count,
            "retain_key_position_output": self.retain_key_position_output,
        }


@dataclass(frozen=True, slots=True)
class FixedMeshMovingLoadPassCase:
    moving_load_pass_schema_version: int
    unit_system: str
    model_type: str
    reference_case: SinglePositionElastoplasticCase
    pass_settings: MovingPassSettings

    def __post_init__(self) -> None:
        if (
            type(self.moving_load_pass_schema_version) is not int
            or self.moving_load_pass_schema_version != MOVING_LOAD_PASS_SCHEMA_VERSION
        ):
            _fail(
                "moving_load_pass_schema_version",
                "integer version",
                "must be the strict integer 1",
            )
        if self.unit_system != UNIT_SYSTEM:
            _fail("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _fail("model_type", MODEL_TYPE, "unsupported moving-load-pass model")
        if not isinstance(self.reference_case, SinglePositionElastoplasticCase):
            _fail("reference_case", "SinglePositionElastoplasticCase", "has the wrong object type")
        if not isinstance(self.pass_settings, MovingPassSettings):
            _fail("pass", "MovingPassSettings", "has the wrong object type")

    @property
    def geometry(self):
        return self.reference_case.geometry

    @property
    def material(self):
        return self.reference_case.material

    @property
    def mesh(self):
        return self.reference_case.mesh

    @property
    def analysis(self):
        return self.reference_case.analysis

    @property
    def motion_direction(self) -> str:
        return self.reference_case.pass_load.trajectory.single_pass.relative_feed_direction

    @classmethod
    def from_mapping(cls, value: object) -> "FixedMeshMovingLoadPassCase":
        root = _object(value, "moving_load_pass_case")
        _exact(
            root,
            {
                "moving_load_pass_schema_version",
                "unit_system",
                "model_type",
                "reference_case",
                "pass",
            },
            "moving_load_pass_case",
        )
        try:
            reference = SinglePositionElastoplasticCase.from_mapping(root["reference_case"])
        except (SinglePositionElastoplasticValidationError, ValueError) as exc:
            raise MovingLoadPassValidationError(str(exc)) from exc
        return cls(
            moving_load_pass_schema_version=root["moving_load_pass_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            reference_case=reference,
            pass_settings=MovingPassSettings.from_mapping(root["pass"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FixedMeshMovingLoadPassCase":
        source = Path(path).expanduser().resolve()
        try:
            with source.open(encoding="utf-8") as stream:
                payload = json.load(stream)
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
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "moving_load_pass_schema_version": self.moving_load_pass_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "reference_case": self.reference_case.to_dict(),
            "pass": self.pass_settings.to_dict(),
        }
