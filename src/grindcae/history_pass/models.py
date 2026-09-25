"""Strict Phase 6B.2 fixed-mesh elastoplastic history-pass input contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from grindcae.moving_load_pass import (
    FixedMeshMovingLoadPassCase,
    MovingLoadPassValidationError,
)


HISTORY_PASS_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "fixed_mesh_complete_single_pass_elastoplastic_history"


class HistoryPassValidationError(ValueError):
    """Raised when a Phase 6B.2 input violates its strict schema."""


def _fail(field: str, unit: str, reason: str) -> None:
    raise HistoryPassValidationError(f"{field} [{unit}]: {reason}")


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


def _strict_int(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        _fail(field, "integer", "must be a strict integer")
    if not minimum <= value <= maximum:
        _fail(field, "integer", f"must be between {minimum} and {maximum} inclusive")
    return value


def _fraction(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, "dimensionless", "must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        _fail(field, "dimensionless", "must be in (0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class TransitionSettings:
    base_substep_count: int
    final_unloading_substep_count: int
    allow_reduction: bool
    minimum_substep_fraction: float
    maximum_retries: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_substep_count", _strict_int(self.base_substep_count, "transition.base_substep_count", 1, 100))
        object.__setattr__(self, "final_unloading_substep_count", _strict_int(self.final_unloading_substep_count, "transition.final_unloading_substep_count", 1, 100))
        if type(self.allow_reduction) is not bool:
            _fail("transition.allow_reduction", "bool", "must be a strict bool")
        object.__setattr__(self, "minimum_substep_fraction", _fraction(self.minimum_substep_fraction, "transition.minimum_substep_fraction"))
        object.__setattr__(self, "maximum_retries", _strict_int(self.maximum_retries, "transition.maximum_retries", 0, 50))

    @classmethod
    def from_mapping(cls, value: object) -> "TransitionSettings":
        data = _object(value, "transition")
        _exact(data, {"base_substep_count", "final_unloading_substep_count", "allow_reduction", "minimum_substep_fraction", "maximum_retries"}, "transition")
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "base_substep_count": self.base_substep_count,
            "final_unloading_substep_count": self.final_unloading_substep_count,
            "allow_reduction": self.allow_reduction,
            "minimum_substep_fraction": self.minimum_substep_fraction,
            "maximum_retries": self.maximum_retries,
        }


@dataclass(frozen=True, slots=True)
class SnapshotRetentionSettings:
    retain_key_positions: bool
    retain_maximum_plastic_state: bool

    def __post_init__(self) -> None:
        for name in ("retain_key_positions", "retain_maximum_plastic_state"):
            if type(getattr(self, name)) is not bool:
                _fail(f"snapshot_retention.{name}", "bool", "must be a strict bool")

    @classmethod
    def from_mapping(cls, value: object) -> "SnapshotRetentionSettings":
        data = _object(value, "snapshot_retention")
        _exact(data, {"retain_key_positions", "retain_maximum_plastic_state"}, "snapshot_retention")
        return cls(**data)

    def to_dict(self) -> dict[str, bool]:
        return {
            "retain_key_positions": self.retain_key_positions,
            "retain_maximum_plastic_state": self.retain_maximum_plastic_state,
        }


@dataclass(frozen=True, slots=True)
class FixedMeshElastoplasticHistoryPassCase:
    history_pass_schema_version: int
    unit_system: str
    model_type: str
    moving_load_pass: FixedMeshMovingLoadPassCase
    transition: TransitionSettings
    snapshot_retention: SnapshotRetentionSettings

    def __post_init__(self) -> None:
        if type(self.history_pass_schema_version) is not int or self.history_pass_schema_version != HISTORY_PASS_SCHEMA_VERSION:
            _fail("history_pass_schema_version", "integer version", "must be the strict integer 1")
        if self.unit_system != UNIT_SYSTEM:
            _fail("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _fail("model_type", MODEL_TYPE, "unsupported history-pass model")
        if not isinstance(self.moving_load_pass, FixedMeshMovingLoadPassCase):
            _fail("moving_load_pass", "FixedMeshMovingLoadPassCase", "has the wrong object type")

    @property
    def reference_case(self):
        return self.moving_load_pass.reference_case

    @classmethod
    def from_mapping(cls, value: object) -> "FixedMeshElastoplasticHistoryPassCase":
        root = _object(value, "history_pass_case")
        _exact(root, {"history_pass_schema_version", "unit_system", "model_type", "moving_load_pass", "transition", "snapshot_retention"}, "history_pass_case")
        try:
            moving = FixedMeshMovingLoadPassCase.from_mapping(root["moving_load_pass"])
        except MovingLoadPassValidationError as exc:
            raise HistoryPassValidationError(str(exc)) from exc
        return cls(
            history_pass_schema_version=root["history_pass_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            moving_load_pass=moving,
            transition=TransitionSettings.from_mapping(root["transition"]),
            snapshot_retention=SnapshotRetentionSettings.from_mapping(root["snapshot_retention"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FixedMeshElastoplasticHistoryPassCase":
        source = Path(path).expanduser().resolve()
        try:
            with source.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            _fail("input_file", "existing UTF-8 JSON", f"does not exist: {source}")
        except json.JSONDecodeError as exc:
            _fail("input_file", "valid UTF-8 JSON", f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")
        except (UnicodeDecodeError, OSError) as exc:
            _fail("input_file", "readable UTF-8 JSON", f"cannot be read: {exc}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "history_pass_schema_version": self.history_pass_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "moving_load_pass": self.moving_load_pass.to_dict(),
            "transition": self.transition.to_dict(),
            "snapshot_retention": self.snapshot_retention.to_dict(),
        }
