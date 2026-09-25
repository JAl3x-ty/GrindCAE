"""Strict schema version 1 input models for ideal single-pass trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


TRAJECTORY_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_pass_ideal_wheel_profile"
RELATIVE_FEED_DIRECTIONS = ("positive_x", "negative_x")
MIN_BASE_POINT_COUNT = 101
MAX_BASE_POINT_COUNT = 10001


class TrajectoryValidationError(ValueError):
    """Raised when a phase 4C1 trajectory input violates its contract."""


def _validation_error(field: str, unit: str, reason: str) -> None:
    raise TrajectoryValidationError(f"{field} [{unit}]: {reason}")


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


def _positive_number(value: object, field: str, unit: str) -> float:
    number = _finite_number(value, field, unit)
    if number <= 0.0:
        _validation_error(field, unit, "must be strictly greater than zero")
    return number


def _literal(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        _validation_error(field, "string literal", "must be a string")
    if value not in allowed:
        _validation_error(field, "string literal", f"must be one of: {', '.join(allowed)}")
    return value


def _sqrt_product(left: float, right: float, field: str) -> float:
    try:
        result = math.sqrt(left) * math.sqrt(right)
    except (OverflowError, ValueError):
        _validation_error(field, "m", "derived projected length is invalid")
    if not math.isfinite(result):
        _validation_error(field, "m", "derived projected length must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TrajectoryWorkpiece:
    """Workpiece dimensions used only by the phase 4C1 surface model."""

    length_m: float
    original_surface_height_m: float

    def __post_init__(self) -> None:
        length = _positive_number(self.length_m, "workpiece.length_m", "m")
        height = _positive_number(
            self.original_surface_height_m,
            "workpiece.original_surface_height_m",
            "m",
        )
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "original_surface_height_m", height)

    @classmethod
    def from_mapping(cls, payload: object) -> TrajectoryWorkpiece:
        data = _object(payload, "workpiece")
        _exact_keys(data, {"length_m", "original_surface_height_m"}, "workpiece")
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "original_surface_height_m": self.original_surface_height_m,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryWheel:
    """Ideal rigid circular wheel geometry."""

    diameter_m: float

    def __post_init__(self) -> None:
        diameter = _positive_number(self.diameter_m, "wheel.diameter_m", "m")
        object.__setattr__(self, "diameter_m", diameter)

    @classmethod
    def from_mapping(cls, payload: object) -> TrajectoryWheel:
        data = _object(payload, "wheel")
        _exact_keys(data, {"diameter_m"}, "wheel")
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return {"diameter_m": self.diameter_m}


@dataclass(frozen=True, slots=True)
class SinglePassSettings:
    """One prescribed depth and current wheel-bottom trajectory position."""

    depth_of_cut_m: float
    relative_feed_direction: str
    wheel_lowest_point_x_m: float

    def __post_init__(self) -> None:
        depth = _positive_number(
            self.depth_of_cut_m, "single_pass.depth_of_cut_m", "m"
        )
        direction = _literal(
            self.relative_feed_direction,
            "single_pass.relative_feed_direction",
            RELATIVE_FEED_DIRECTIONS,
        )
        position = _finite_number(
            self.wheel_lowest_point_x_m,
            "single_pass.wheel_lowest_point_x_m",
            "m",
        )
        object.__setattr__(self, "depth_of_cut_m", depth)
        object.__setattr__(self, "relative_feed_direction", direction)
        object.__setattr__(self, "wheel_lowest_point_x_m", position)

    @classmethod
    def from_mapping(cls, payload: object) -> SinglePassSettings:
        data = _object(payload, "single_pass")
        _exact_keys(
            data,
            {
                "depth_of_cut_m",
                "relative_feed_direction",
                "wheel_lowest_point_x_m",
            },
            "single_pass",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "depth_of_cut_m": self.depth_of_cut_m,
            "relative_feed_direction": self.relative_feed_direction,
            "wheel_lowest_point_x_m": self.wheel_lowest_point_x_m,
        }


@dataclass(frozen=True, slots=True)
class TrajectorySampling:
    """Deterministic base sampling request for the workpiece length."""

    base_point_count: int

    def __post_init__(self) -> None:
        if type(self.base_point_count) is not int:
            _validation_error(
                "sampling.base_point_count",
                "integer point count",
                "must be a strict integer; bool, floating point, and text are invalid",
            )
        if not MIN_BASE_POINT_COUNT <= self.base_point_count <= MAX_BASE_POINT_COUNT:
            _validation_error(
                "sampling.base_point_count",
                "integer point count",
                f"must be between {MIN_BASE_POINT_COUNT} and {MAX_BASE_POINT_COUNT} inclusive",
            )

    @classmethod
    def from_mapping(cls, payload: object) -> TrajectorySampling:
        data = _object(payload, "sampling")
        _exact_keys(data, {"base_point_count"}, "sampling")
        return cls(**data)

    def to_dict(self) -> dict[str, int]:
        return {"base_point_count": self.base_point_count}


@dataclass(frozen=True, slots=True)
class TrajectoryCase:
    """Independent phase 4C1 input envelope; it is not a FEM case."""

    trajectory_schema_version: int
    unit_system: str
    model_type: str
    workpiece: TrajectoryWorkpiece
    wheel: TrajectoryWheel
    single_pass: SinglePassSettings
    sampling: TrajectorySampling

    def __post_init__(self) -> None:
        if (
            type(self.trajectory_schema_version) is not int
            or self.trajectory_schema_version != TRAJECTORY_SCHEMA_VERSION
        ):
            _validation_error(
                "trajectory_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _validation_error("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _validation_error("model_type", MODEL_TYPE, "unsupported trajectory model")
        typed_fields = (
            ("workpiece", self.workpiece, TrajectoryWorkpiece),
            ("wheel", self.wheel, TrajectoryWheel),
            ("single_pass", self.single_pass, SinglePassSettings),
            ("sampling", self.sampling, TrajectorySampling),
        )
        for field, value, expected_type in typed_fields:
            if not isinstance(value, expected_type):
                _validation_error(field, expected_type.__name__, "has the wrong object type")

        depth = self.single_pass.depth_of_cut_m
        height = self.workpiece.original_surface_height_m
        diameter = self.wheel.diameter_m
        if depth >= height:
            _validation_error(
                "single_pass.depth_of_cut_m",
                "m",
                "must be less than workpiece.original_surface_height_m",
            )
        if depth >= diameter / 2.0:
            _validation_error(
                "single_pass.depth_of_cut_m",
                "m",
                "must be less than wheel.diameter_m / 2",
            )

        exact_length = _sqrt_product(
            depth,
            diameter - depth,
            "single_pass.depth_of_cut_m",
        )
        _sqrt_product(depth, diameter, "single_pass.depth_of_cut_m")
        if exact_length >= self.workpiece.length_m:
            _validation_error(
                "single_pass.depth_of_cut_m",
                "m",
                "the strict circular-arc projected length must be less than workpiece.length_m",
            )
        center_y = height - depth + diameter / 2.0
        if not math.isfinite(center_y):
            _validation_error(
                "wheel.diameter_m",
                "m",
                "the derived wheel-center height must be finite",
            )

    @classmethod
    def from_mapping(cls, payload: object) -> TrajectoryCase:
        root = _object(payload, "trajectory_case")
        _exact_keys(
            root,
            {
                "trajectory_schema_version",
                "unit_system",
                "model_type",
                "workpiece",
                "wheel",
                "single_pass",
                "sampling",
            },
            "trajectory_case",
        )
        return cls(
            trajectory_schema_version=root["trajectory_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            workpiece=TrajectoryWorkpiece.from_mapping(root["workpiece"]),
            wheel=TrajectoryWheel.from_mapping(root["wheel"]),
            single_pass=SinglePassSettings.from_mapping(root["single_pass"]),
            sampling=TrajectorySampling.from_mapping(root["sampling"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> TrajectoryCase:
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
            "trajectory_schema_version": self.trajectory_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "workpiece": self.workpiece.to_dict(),
            "wheel": self.wheel.to_dict(),
            "single_pass": self.single_pass.to_dict(),
            "sampling": self.sampling.to_dict(),
        }
