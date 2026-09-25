"""Centralized replaceable demonstration presets for Phase 7A.1."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    BUILTIN_PRESET_ID,
    BuiltinPresetMechanismModel,
    CustomMechanismModel,
    MechanismFractions,
    MechanismModel,
)


@dataclass(frozen=True, slots=True)
class ResolvedMechanismParameters:
    parameter_source: str
    material_behavior_family: str
    material_label: str
    preset_id: str | None
    reference_equivalent_process_thickness_m: float
    anchor_ratios: tuple[float, ...]
    anchor_fractions: tuple[MechanismFractions, ...]
    calibration_status: str
    provenance: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_source": self.parameter_source,
            "material_behavior_family": self.material_behavior_family,
            "material_label": self.material_label,
            "preset_id": self.preset_id,
            "reference_equivalent_process_thickness_m": self.reference_equivalent_process_thickness_m,
            "anchor_ratios": list(self.anchor_ratios),
            "anchor_fractions": [value.to_dict() for value in self.anchor_fractions],
            "calibration_status": self.calibration_status,
            "provenance": dict(self.provenance),
        }


DUCTILE_ALLOY_STEEL_DEMO = ResolvedMechanismParameters(
    parameter_source="builtin_preset",
    material_behavior_family="ductile_metal",
    material_label="Generic ductile alloy steel demonstration preset",
    preset_id=BUILTIN_PRESET_ID,
    reference_equivalent_process_thickness_m=6.666666666666667e-8,
    anchor_ratios=(0.25, 1.0, 4.0),
    anchor_fractions=(
        MechanismFractions(rubbing=0.65, ploughing=0.30, cutting=0.05),
        MechanismFractions(rubbing=0.25, ploughing=0.35, cutting=0.40),
        MechanismFractions(rubbing=0.05, ploughing=0.20, cutting=0.75),
    ),
    calibration_status="demonstration_not_experimentally_calibrated",
    provenance={
        "calibration_id": "ductile_alloy_steel_demo_v1",
        "calibration_status": "demonstration_not_experimentally_calibrated",
        "source": "Phase 7A.1 engineering demonstration assumptions; not experimentally calibrated; replace with experiment or reliable literature.",
        "applicability_notes": "Generic ductile alloy steel demonstration only; it does not represent a named grade such as 40Cr or bearing steel.",
    },
)


def resolve_mechanism_parameters(model: MechanismModel) -> ResolvedMechanismParameters:
    """Expand a strict source choice into one reusable parameter object."""

    if isinstance(model, BuiltinPresetMechanismModel):
        return DUCTILE_ALLOY_STEEL_DEMO
    if isinstance(model, CustomMechanismModel):
        return ResolvedMechanismParameters(
            parameter_source=model.parameter_source,
            material_behavior_family=model.material_behavior_family,
            material_label=model.material_label,
            preset_id=None,
            reference_equivalent_process_thickness_m=model.reference_equivalent_process_thickness_m,
            anchor_ratios=model.anchor_ratios,
            anchor_fractions=model.anchor_fractions,
            calibration_status=model.provenance.calibration_status,
            provenance=model.provenance.to_dict(),
        )
    raise TypeError("model must be a validated Phase 7A.1 mechanism model")
