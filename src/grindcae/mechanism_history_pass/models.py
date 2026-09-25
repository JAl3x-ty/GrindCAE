"""Strict input contract for the Phase 7A.3 comparison route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from grindcae.history_pass import FixedMeshElastoplasticHistoryPassCase
from grindcae.statistical_grain_load import StatisticalGrainLoadCase


SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "fixed_mesh_complete_single_pass_statistical_mechanism_elastoplastic_history"


class MechanismHistoryPassValidationError(ValueError):
    """Raised when a Phase 7A.3 input violates the composed contract."""


def _fail(field: str, reason: str) -> None:
    raise MechanismHistoryPassValidationError(f"{field}: {reason}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value)); unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        parts = []
        if missing: parts.append(f"missing fields {missing}")
        if unexpected: parts.append(f"unexpected fields {unexpected}")
        _fail(field, " and ".join(parts))


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    mode: str
    require_identical_total_force_history: bool
    retain_both_final_fields: bool

    @classmethod
    def from_mapping(cls, value: object) -> "ComparisonSettings":
        data = _mapping(value, "comparison")
        _exact(data, {"mode", "require_identical_total_force_history", "retain_both_final_fields"}, "comparison")
        if data["mode"] != "uniform_vs_statistical_mechanism":
            _fail("comparison.mode", "must be 'uniform_vs_statistical_mechanism'")
        if data["require_identical_total_force_history"] is not True:
            _fail("comparison.require_identical_total_force_history", "must be true")
        if data["retain_both_final_fields"] is not True:
            _fail("comparison.retain_both_final_fields", "must be true")
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "require_identical_total_force_history": self.require_identical_total_force_history,
            "retain_both_final_fields": self.retain_both_final_fields,
        }


@dataclass(frozen=True, slots=True)
class MechanismHistoryPassCase:
    mechanism_history_pass_schema_version: int
    unit_system: str
    model_type: str
    baseline_history_pass: FixedMeshElastoplasticHistoryPassCase
    statistical_grain_load: StatisticalGrainLoadCase
    comparison: ComparisonSettings

    @classmethod
    def from_mapping(cls, value: object) -> "MechanismHistoryPassCase":
        root = _mapping(value, "mechanism_history_pass_case")
        _exact(root, {"mechanism_history_pass_schema_version", "unit_system", "model_type", "baseline_history_pass", "statistical_grain_load", "comparison"}, "mechanism_history_pass_case")
        if type(root["mechanism_history_pass_schema_version"]) is not int or root["mechanism_history_pass_schema_version"] != SCHEMA_VERSION:
            _fail("mechanism_history_pass_schema_version", "must be strict integer 1")
        if root["unit_system"] != UNIT_SYSTEM:
            _fail("unit_system", "only SI is supported")
        if root["model_type"] != MODEL_TYPE:
            _fail("model_type", "unsupported model type")
        try:
            baseline = FixedMeshElastoplasticHistoryPassCase.from_mapping(root["baseline_history_pass"])
        except ValueError as exc:
            _fail("baseline_history_pass", str(exc))
        try:
            statistical = StatisticalGrainLoadCase.from_dict(root["statistical_grain_load"])
        except ValueError as exc:
            _fail("statistical_grain_load", str(exc))
        _validate_cross_phase_consistency(baseline, statistical)
        return cls(root["mechanism_history_pass_schema_version"], root["unit_system"], root["model_type"], baseline, statistical, ComparisonSettings.from_mapping(root["comparison"]))

    @classmethod
    def from_json(cls, path: str | Path) -> "MechanismHistoryPassCase":
        try:
            return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail("input_file", f"must be readable UTF-8 JSON: {exc}")

    def to_dict(self) -> dict[str, object]:
        return {
            "mechanism_history_pass_schema_version": self.mechanism_history_pass_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "baseline_history_pass": self.baseline_history_pass.to_dict(),
            "statistical_grain_load": self.statistical_grain_load.to_dict(),
            "comparison": self.comparison.to_dict(),
        }


def _validate_cross_phase_consistency(
    baseline: FixedMeshElastoplasticHistoryPassCase,
    statistical: StatisticalGrainLoadCase,
) -> None:
    """Reject mismatched process/force provenance before any solve is started."""

    pass_load = baseline.reference_case.pass_load
    reference_force = pass_load.force_model.to_dict()
    statistical_force = statistical.mechanism_force.force_model.to_dict()
    if reference_force != statistical_force:
        _fail("statistical_grain_load.mechanism_force.force_model", "must exactly match baseline_history_pass force_model")
    trajectory = pass_load.trajectory
    process = statistical.mechanism_force.force_model.process
    for label, actual, expected in (
        ("wheel_diameter_m", trajectory.wheel.diameter_m, process.wheel_diameter_m),
        ("depth_of_cut_m", trajectory.single_pass.depth_of_cut_m, process.depth_of_cut_m),
    ):
        if actual != expected:
            _fail(f"cross_phase.{label}", "must exactly match both composed inputs")
    if pass_load.unit_system != UNIT_SYSTEM:
        _fail("baseline_history_pass.unit_system", "must be SI")
