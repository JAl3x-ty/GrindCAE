"""Strict schema version 1 inputs for phase 4C2 pass-load snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from grindcae.force_model import GrindingForceCase
from grindcae.trajectory import TrajectoryCase


PASS_LOAD_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_pass_contact_ratio_empirical_load"
BOUNDARY = "ideal_top_surface"
DISTRIBUTION = "uniform_over_effective_projected_interval"
TANGENTIAL_FORCE_DIRECTIONS = ("positive_x", "negative_x")


class PassLoadValidationError(ValueError):
    """Raised when a phase 4C2 input violates its independent contract."""


def _validation_error(field: str, unit: str, reason: str) -> None:
    raise PassLoadValidationError(f"{field} [{unit}]: {reason}")


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


def _literal(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        _validation_error(field, "string literal", "must be a string")
    if value not in allowed:
        _validation_error(field, "string literal", f"must be one of: {', '.join(allowed)}")
    return value


@dataclass(frozen=True, slots=True)
class PassForceMapping:
    """Signed force convention for the ideal projected top-surface interval."""

    boundary: str
    distribution: str
    tangential_force_direction: str

    def __post_init__(self) -> None:
        boundary = _literal(self.boundary, "force_mapping.boundary", (BOUNDARY,))
        distribution = _literal(
            self.distribution,
            "force_mapping.distribution",
            (DISTRIBUTION,),
        )
        direction = _literal(
            self.tangential_force_direction,
            "force_mapping.tangential_force_direction",
            TANGENTIAL_FORCE_DIRECTIONS,
        )
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "tangential_force_direction", direction)

    @classmethod
    def from_mapping(cls, payload: object) -> PassForceMapping:
        data = _object(payload, "force_mapping")
        _exact_keys(
            data,
            {"boundary", "distribution", "tangential_force_direction"},
            "force_mapping",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, str]:
        return {
            "boundary": self.boundary,
            "distribution": self.distribution,
            "tangential_force_direction": self.tangential_force_direction,
        }


@dataclass(frozen=True, slots=True)
class PassLoadCase:
    """Independent phase 4C2 envelope combining validated 4C1 and 4A inputs."""

    pass_load_schema_version: int
    unit_system: str
    model_type: str
    trajectory: TrajectoryCase
    force_model: GrindingForceCase
    force_mapping: PassForceMapping

    def __post_init__(self) -> None:
        if (
            type(self.pass_load_schema_version) is not int
            or self.pass_load_schema_version != PASS_LOAD_SCHEMA_VERSION
        ):
            _validation_error(
                "pass_load_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _validation_error("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _validation_error("model_type", MODEL_TYPE, "unsupported pass-load model")
        if not isinstance(self.trajectory, TrajectoryCase):
            _validation_error("trajectory", "TrajectoryCase", "has the wrong object type")
        if not isinstance(self.force_model, GrindingForceCase):
            _validation_error(
                "force_model", "GrindingForceCase", "has the wrong object type"
            )
        if not isinstance(self.force_mapping, PassForceMapping):
            _validation_error(
                "force_mapping", "PassForceMapping", "has the wrong object type"
            )

        trajectory_diameter = self.trajectory.wheel.diameter_m
        force_diameter = self.force_model.process.wheel_diameter_m
        if not math.isclose(
            trajectory_diameter, force_diameter, rel_tol=1.0e-12, abs_tol=0.0
        ):
            _validation_error(
                "trajectory.wheel.diameter_m",
                "m",
                "must match force_model.process.wheel_diameter_m without averaging or correction",
            )
        trajectory_depth = self.trajectory.single_pass.depth_of_cut_m
        force_depth = self.force_model.process.depth_of_cut_m
        if not math.isclose(
            trajectory_depth, force_depth, rel_tol=1.0e-12, abs_tol=0.0
        ):
            _validation_error(
                "trajectory.single_pass.depth_of_cut_m",
                "m",
                "must match force_model.process.depth_of_cut_m without averaging or correction",
            )

    @classmethod
    def from_mapping(cls, payload: object) -> PassLoadCase:
        root = _object(payload, "pass_load_case")
        _exact_keys(
            root,
            {
                "pass_load_schema_version",
                "unit_system",
                "model_type",
                "trajectory",
                "force_model",
                "force_mapping",
            },
            "pass_load_case",
        )
        return cls(
            pass_load_schema_version=root["pass_load_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            trajectory=TrajectoryCase.from_mapping(root["trajectory"]),
            force_model=GrindingForceCase.from_dict(root["force_model"]),
            force_mapping=PassForceMapping.from_mapping(root["force_mapping"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PassLoadCase:
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

    def to_dict(self) -> dict[str, object]:
        return {
            "pass_load_schema_version": self.pass_load_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "trajectory": self.trajectory.to_dict(),
            "force_model": self.force_model.to_dict(),
            "force_mapping": self.force_mapping.to_dict(),
        }
