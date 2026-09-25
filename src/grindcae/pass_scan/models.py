"""Strict schema version 1 input for phase 4C4 quasi-static pass scans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from grindcae.evolved_fem import EvolvedFemCase, EvolvedFemValidationError


SCAN_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_pass_quasi_static_evolved_fem_scan"
SCAN_RANGE = "complete_single_pass"
MIN_BASE_POSITION_COUNT = 5
MAX_BASE_POSITION_COUNT = 101


class PassScanValidationError(ValueError):
    """Raised when a phase 4C4 scan input violates its independent schema."""


def _error(field: str, unit: str, reason: str) -> None:
    raise PassScanValidationError(f"{field} [{unit}]: {reason}")


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
        details.append(f"missing fields {missing}")
    if unexpected:
        details.append(f"unexpected fields {unexpected}")
    if details:
        _error(field, "exact field set", " and ".join(details))


@dataclass(frozen=True, slots=True)
class ScanSettings:
    """Complete-pass position generation request."""

    range: str
    base_position_count: int

    def __post_init__(self) -> None:
        if self.range != SCAN_RANGE:
            _error("scan.range", SCAN_RANGE, "unsupported scan range")
        if type(self.base_position_count) is not int:
            _error(
                "scan.base_position_count",
                "integer position count",
                "must be a strict integer; bool, floating point, and text are invalid",
            )
        if not MIN_BASE_POSITION_COUNT <= self.base_position_count <= MAX_BASE_POSITION_COUNT:
            _error(
                "scan.base_position_count",
                "integer position count",
                f"must be between {MIN_BASE_POSITION_COUNT} and {MAX_BASE_POSITION_COUNT} inclusive",
            )

    @classmethod
    def from_mapping(cls, payload: object) -> ScanSettings:
        data = _object(payload, "scan")
        _exact_keys(data, {"range", "base_position_count"}, "scan")
        return cls(**data)

    def to_dict(self) -> dict[str, int | str]:
        return {"range": self.range, "base_position_count": self.base_position_count}


@dataclass(frozen=True, slots=True)
class PassScanCase:
    """One validated reference 4C3B case plus a complete-pass scan request."""

    scan_schema_version: int
    unit_system: str
    model_type: str
    reference_case: EvolvedFemCase
    scan: ScanSettings

    def __post_init__(self) -> None:
        if type(self.scan_schema_version) is not int or self.scan_schema_version != 1:
            _error(
                "scan_schema_version",
                "integer version",
                "must be the strict integer 1; bool, 1.0, and text are invalid",
            )
        if self.unit_system != UNIT_SYSTEM:
            _error("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _error("model_type", MODEL_TYPE, "unsupported pass-scan model")
        if not isinstance(self.reference_case, EvolvedFemCase):
            _error("reference_case", "EvolvedFemCase", "has the wrong object type")
        if self.reference_case.unit_system != self.unit_system:
            _error("reference_case.unit_system", "SI", "must match the root unit system")
        if not isinstance(self.scan, ScanSettings):
            _error("scan", "ScanSettings", "has the wrong object type")

    @classmethod
    def from_mapping(cls, payload: object) -> PassScanCase:
        root = _object(payload, "pass_scan_case")
        _exact_keys(
            root,
            {
                "scan_schema_version",
                "unit_system",
                "model_type",
                "reference_case",
                "scan",
            },
            "pass_scan_case",
        )
        try:
            reference = EvolvedFemCase.from_mapping(root["reference_case"])
        except EvolvedFemValidationError as exc:
            raise PassScanValidationError(str(exc)) from exc
        return cls(
            scan_schema_version=root["scan_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            reference_case=reference,
            scan=ScanSettings.from_mapping(root["scan"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PassScanCase:
        input_path = Path(path).expanduser().resolve()
        try:
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
            "scan_schema_version": self.scan_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "reference_case": self.reference_case.to_dict(),
            "scan": self.scan.to_dict(),
        }
