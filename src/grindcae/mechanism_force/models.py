"""Strict SI input contract for Phase 7A.1 mechanism force decomposition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from grindcae.force_model import GrindingForceCase


MECHANISM_FORCE_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "ductile_metal_three_mechanism_force_decomposition"
BUILTIN_PRESET_ID = "ductile_alloy_steel_demo"
SUPPORTED_MATERIAL_BEHAVIOR_FAMILY = "ductile_metal"
CUSTOM_CALIBRATION_STATUSES = frozenset({"demonstration", "literature", "experiment"})
FRACTION_SUM_ABSOLUTE_TOLERANCE = 1.0e-12


class MechanismForceValidationError(ValueError):
    """Raised when a Phase 7A.1 input violates its strict contract."""


def _error(field: str, unit: str, reason: str) -> None:
    raise MechanismForceValidationError(f"{field} [{unit}]: {reason}")


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


def _nonempty_string(value: object, field: str, meaning: str) -> str:
    if not isinstance(value, str):
        _error(field, meaning, "must be a string")
    normalized = value.strip()
    if not normalized:
        _error(field, meaning, "must not be empty or whitespace")
    return normalized


def _positive_number(value: object, field: str, unit: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(field, unit, "must be a numeric value, not bool or text")
    try:
        number = float(value)
    except OverflowError:
        _error(field, unit, "must be finite and representable")
    if not math.isfinite(number):
        _error(field, unit, "must be finite; NaN and infinity are invalid")
    if number <= 0.0:
        _error(field, unit, "must be strictly greater than zero")
    return number


def _fraction(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(field, "[0, 1]", "must be numeric, not bool or text")
    try:
        number = float(value)
    except OverflowError:
        _error(field, "[0, 1]", "must be finite and representable")
    if not math.isfinite(number):
        _error(field, "[0, 1]", "must be finite; NaN and infinity are invalid")
    if number < 0.0 or number > 1.0:
        _error(field, "[0, 1]", "must lie within the closed interval [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class MechanismFractions:
    rubbing: float
    ploughing: float
    cutting: float

    def __post_init__(self) -> None:
        values = tuple(
            _fraction(getattr(self, name), f"mechanism_fraction.{name}")
            for name in ("rubbing", "ploughing", "cutting")
        )
        if not math.isclose(
            sum(values), 1.0, rel_tol=0.0, abs_tol=FRACTION_SUM_ABSOLUTE_TOLERANCE
        ):
            _error(
                "mechanism_fraction",
                "dimensionless fractions",
                "rubbing, ploughing, and cutting must sum to 1 within strict tolerance",
            )
        object.__setattr__(self, "rubbing", values[0])
        object.__setattr__(self, "ploughing", values[1])
        object.__setattr__(self, "cutting", values[2])

    @classmethod
    def from_dict(cls, payload: object, field: str) -> MechanismFractions:
        data = _object(payload, field)
        _exact_keys(data, {"rubbing", "ploughing", "cutting"}, field)
        return cls(
            rubbing=_fraction(data["rubbing"], f"{field}.rubbing"),
            ploughing=_fraction(data["ploughing"], f"{field}.ploughing"),
            cutting=_fraction(data["cutting"], f"{field}.cutting"),
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.rubbing, self.ploughing, self.cutting)

    def to_dict(self) -> dict[str, float]:
        return {
            "rubbing": self.rubbing,
            "ploughing": self.ploughing,
            "cutting": self.cutting,
        }


@dataclass(frozen=True, slots=True)
class MechanismProvenance:
    calibration_id: str
    calibration_status: str
    source: str
    applicability_notes: str

    def __post_init__(self) -> None:
        for name, meaning in (
            ("calibration_id", "calibration identifier"),
            ("calibration_status", "calibration status"),
            ("source", "parameter source"),
            ("applicability_notes", "applicability notes"),
        ):
            object.__setattr__(
                self, name, _nonempty_string(getattr(self, name), f"provenance.{name}", meaning)
            )
        if self.calibration_status not in CUSTOM_CALIBRATION_STATUSES:
            _error(
                "provenance.calibration_status",
                "demonstration|literature|experiment",
                "must be one of demonstration, literature, or experiment",
            )

    @classmethod
    def from_dict(cls, payload: object) -> MechanismProvenance:
        data = _object(payload, "mechanism_model.provenance")
        _exact_keys(
            data,
            {"calibration_id", "calibration_status", "source", "applicability_notes"},
            "mechanism_model.provenance",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, str]:
        return {
            "calibration_id": self.calibration_id,
            "calibration_status": self.calibration_status,
            "source": self.source,
            "applicability_notes": self.applicability_notes,
        }


@dataclass(frozen=True, slots=True)
class BuiltinPresetMechanismModel:
    parameter_source: str
    preset_id: str

    def __post_init__(self) -> None:
        source = _nonempty_string(
            self.parameter_source, "mechanism_model.parameter_source", "parameter source"
        )
        preset_id = _nonempty_string(
            self.preset_id, "mechanism_model.preset_id", "built-in preset identifier"
        )
        if source != "builtin_preset":
            _error(
                "mechanism_model.parameter_source",
                "builtin_preset",
                "must identify the built-in preset branch",
            )
        if preset_id != BUILTIN_PRESET_ID:
            _error(
                "mechanism_model.preset_id",
                BUILTIN_PRESET_ID,
                "unsupported built-in preset",
            )
        object.__setattr__(self, "parameter_source", source)
        object.__setattr__(self, "preset_id", preset_id)

    def to_dict(self) -> dict[str, str]:
        return {"parameter_source": self.parameter_source, "preset_id": self.preset_id}


@dataclass(frozen=True, slots=True)
class CustomMechanismModel:
    parameter_source: str
    material_behavior_family: str
    material_label: str
    reference_equivalent_process_thickness_m: float
    anchor_ratios: tuple[float, ...]
    anchor_fractions: tuple[MechanismFractions, ...]
    provenance: MechanismProvenance

    def __post_init__(self) -> None:
        source = _nonempty_string(
            self.parameter_source, "mechanism_model.parameter_source", "parameter source"
        )
        if source != "custom":
            _error(
                "mechanism_model.parameter_source", "custom", "must identify the custom branch"
            )
        family = _nonempty_string(
            self.material_behavior_family,
            "mechanism_model.material_behavior_family",
            "material behavior family",
        )
        if family != SUPPORTED_MATERIAL_BEHAVIOR_FAMILY:
            _error(
                "mechanism_model.material_behavior_family",
                SUPPORTED_MATERIAL_BEHAVIOR_FAMILY,
                f"unsupported material family {family!r}; brittle materials including SiC are unsupported in Phase 7A.1",
            )
        label = _nonempty_string(
            self.material_label, "mechanism_model.material_label", "material label"
        )
        reference = _positive_number(
            self.reference_equivalent_process_thickness_m,
            "mechanism_model.reference_equivalent_process_thickness_m",
            "m",
        )
        ratios = tuple(
            _positive_number(value, f"mechanism_model.anchor_ratios[{index}]", "dimensionless")
            for index, value in enumerate(self.anchor_ratios)
        )
        if len(ratios) < 2:
            _error("mechanism_model.anchor_ratios", "ordered positive sequence", "must contain at least two anchors")
        if any(right <= left for left, right in zip(ratios, ratios[1:])):
            _error("mechanism_model.anchor_ratios", "ordered positive sequence", "must be strictly increasing")
        fractions = tuple(self.anchor_fractions)
        if len(ratios) != len(fractions):
            _error(
                "mechanism_model.anchor_fractions",
                "same length as anchor_ratios",
                "anchor_ratios and anchor_fractions must have the same length",
            )
        if not all(isinstance(value, MechanismFractions) for value in fractions):
            _error("mechanism_model.anchor_fractions", "mechanism fraction objects", "contains the wrong object type")
        if not isinstance(self.provenance, MechanismProvenance):
            _error("mechanism_model.provenance", "MechanismProvenance", "has the wrong object type")
        object.__setattr__(self, "parameter_source", source)
        object.__setattr__(self, "material_behavior_family", family)
        object.__setattr__(self, "material_label", label)
        object.__setattr__(self, "reference_equivalent_process_thickness_m", reference)
        object.__setattr__(self, "anchor_ratios", ratios)
        object.__setattr__(self, "anchor_fractions", fractions)

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_source": self.parameter_source,
            "material_behavior_family": self.material_behavior_family,
            "material_label": self.material_label,
            "reference_equivalent_process_thickness_m": self.reference_equivalent_process_thickness_m,
            "anchor_ratios": list(self.anchor_ratios),
            "anchor_fractions": [value.to_dict() for value in self.anchor_fractions],
            "provenance": self.provenance.to_dict(),
        }


MechanismModel = BuiltinPresetMechanismModel | CustomMechanismModel


def _mechanism_model(payload: object) -> MechanismModel:
    data = _object(payload, "mechanism_model")
    parameter_source = data.get("parameter_source")
    if parameter_source == "builtin_preset":
        _exact_keys(data, {"parameter_source", "preset_id"}, "mechanism_model")
        return BuiltinPresetMechanismModel(**data)
    if parameter_source == "custom":
        _exact_keys(
            data,
            {
                "parameter_source",
                "material_behavior_family",
                "material_label",
                "reference_equivalent_process_thickness_m",
                "anchor_ratios",
                "anchor_fractions",
                "provenance",
            },
            "mechanism_model",
        )
        raw_ratios = data["anchor_ratios"]
        raw_fractions = data["anchor_fractions"]
        if isinstance(raw_ratios, (str, bytes)) or not isinstance(raw_ratios, Sequence):
            _error("mechanism_model.anchor_ratios", "JSON array", "must be an array")
        if isinstance(raw_fractions, (str, bytes)) or not isinstance(raw_fractions, Sequence):
            _error("mechanism_model.anchor_fractions", "JSON array", "must be an array")
        return CustomMechanismModel(
            parameter_source=data["parameter_source"],
            material_behavior_family=data["material_behavior_family"],
            material_label=data["material_label"],
            reference_equivalent_process_thickness_m=data[
                "reference_equivalent_process_thickness_m"
            ],
            anchor_ratios=tuple(raw_ratios),
            anchor_fractions=tuple(
                MechanismFractions.from_dict(value, f"mechanism_model.anchor_fractions[{index}]")
                for index, value in enumerate(raw_fractions)
            ),
            provenance=MechanismProvenance.from_dict(data["provenance"]),
        )
    if isinstance(parameter_source, str):
        _error(
            "mechanism_model.parameter_source",
            "builtin_preset|custom",
            "must be exactly builtin_preset or custom",
        )
    _error(
        "mechanism_model.parameter_source",
        "builtin_preset|custom",
        "missing or invalid parameter source",
    )


@dataclass(frozen=True, slots=True)
class MechanismForceCase:
    mechanism_force_schema_version: int
    unit_system: str
    model_type: str
    force_model: GrindingForceCase
    mechanism_model: MechanismModel

    def __post_init__(self) -> None:
        if type(self.mechanism_force_schema_version) is not int or self.mechanism_force_schema_version != 1:
            _error(
                "mechanism_force_schema_version",
                "strict integer 1",
                "must be the strict integer 1; bool, float, and text are invalid",
            )
        unit_system = _nonempty_string(self.unit_system, "unit_system", "SI unit-system name")
        if unit_system != UNIT_SYSTEM:
            _error("unit_system", UNIT_SYSTEM, "only SI is supported")
        model_type = _nonempty_string(self.model_type, "model_type", "model identifier")
        if model_type != MODEL_TYPE:
            _error("model_type", MODEL_TYPE, "unsupported model type")
        if not isinstance(self.force_model, GrindingForceCase):
            _error("force_model", "GrindingForceCase", "has the wrong object type")
        if not isinstance(self.mechanism_model, (BuiltinPresetMechanismModel, CustomMechanismModel)):
            _error("mechanism_model", "validated mechanism model", "has the wrong object type")
        object.__setattr__(self, "unit_system", unit_system)
        object.__setattr__(self, "model_type", model_type)

    @classmethod
    def from_dict(cls, payload: object) -> MechanismForceCase:
        root = _object(payload, "mechanism_force_case")
        _exact_keys(
            root,
            {
                "mechanism_force_schema_version",
                "unit_system",
                "model_type",
                "force_model",
                "mechanism_model",
            },
            "mechanism_force_case",
        )
        return cls(
            mechanism_force_schema_version=root["mechanism_force_schema_version"],
            unit_system=root["unit_system"],
            model_type=root["model_type"],
            force_model=GrindingForceCase.from_dict(root["force_model"]),
            mechanism_model=_mechanism_model(root["mechanism_model"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> MechanismForceCase:
        input_path = Path(path).expanduser().resolve()
        try:
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            _error("input_file", "existing UTF-8 JSON", f"does not exist: {input_path}")
        except json.JSONDecodeError as exc:
            _error("input_file", "valid UTF-8 JSON", f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")
        except UnicodeDecodeError as exc:
            _error("input_file", "UTF-8 JSON", f"cannot be decoded as UTF-8: {exc.reason}")
        except OSError as exc:
            _error("input_file", "readable UTF-8 JSON", f"cannot be read: {exc}")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "mechanism_force_schema_version": self.mechanism_force_schema_version,
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "force_model": self.force_model.to_dict(),
            "mechanism_model": self.mechanism_model.to_dict(),
        }
