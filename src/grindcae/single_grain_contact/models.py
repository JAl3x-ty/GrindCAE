"""Strict SI schemas for the GrindCAE single-grain contact route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from grindcae.domain.models import MeshSettings, Rectangle
from grindcae.elastoplastic.models import BilinearIsotropicMaterial
from grindcae.elastoplastic_fem.models import ElastoplasticFemCase
from grindcae.single_grain_contact.damage_models import (
    DamageInput,
    SingleGrainContactValidationError,
)


SCHEMA_VERSION = 3
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_grain_real_contact"


def _fail(field: str, reason: str) -> None:
    raise SingleGrainContactValidationError(f"{field}: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        _fail(field, f"missing fields {missing}; unexpected fields {unexpected}")


def _number(value: object, field: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, "must be numeric and finite")
    result = float(value)
    if not math.isfinite(result):
        _fail(field, "must be finite")
    if positive and result <= 0.0:
        _fail(field, "must be strictly greater than zero")
    if nonnegative and result < 0.0:
        _fail(field, "must be nonnegative")
    return result


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        _fail(field, "must be a strict integer")
    if not minimum <= value <= maximum:
        _fail(field, f"must be between {minimum} and {maximum}")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(field, "must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ContactMaterial:
    E: float
    nu: float
    yield_strength: float
    tangent_modulus: float
    parameter_source: str
    material_state: str
    calibration_status: str
    density_kg_per_m3: float | None = None

    @classmethod
    def from_mapping(cls, value: object, *, schema_version: int = 1) -> ContactMaterial:
        data = _object(value, "material")
        fields = {"E", "nu", "yield_strength", "tangent_modulus", "parameter_source", "material_state", "calibration_status"}
        if schema_version == 3:
            fields.add("density_kg_per_m3")
        _exact(data, fields, "material")
        try:
            base = BilinearIsotropicMaterial.from_mapping(
                {key: data[key] for key in ("E", "nu", "yield_strength", "tangent_modulus")}
            )
        except ValueError as exc:
            raise SingleGrainContactValidationError(str(exc)) from exc
        return cls(
            E=base.E,
            nu=base.nu,
            yield_strength=base.yield_strength,
            tangent_modulus=base.tangent_modulus,
            parameter_source=_text(data["parameter_source"], "material.parameter_source"),
            material_state=_text(data["material_state"], "material.material_state"),
            calibration_status=_text(data["calibration_status"], "material.calibration_status"),
            density_kg_per_m3=(
                _number(data["density_kg_per_m3"], "material.density_kg_per_m3", positive=True)
                if schema_version == 3
                else None
            ),
        )

    @property
    def base(self) -> BilinearIsotropicMaterial:
        return BilinearIsotropicMaterial(self.E, self.nu, self.yield_strength, self.tangent_modulus)

    def to_dict(self) -> dict[str, object]:
        result = {"E": self.E, "nu": self.nu, "yield_strength": self.yield_strength, "tangent_modulus": self.tangent_modulus, "parameter_source": self.parameter_source, "material_state": self.material_state, "calibration_status": self.calibration_status}
        if self.density_kg_per_m3 is not None:
            result["density_kg_per_m3"] = self.density_kg_per_m3
        return result


@dataclass(frozen=True, slots=True)
class ContactAnalysis:
    type: str
    thickness: float

    @classmethod
    def from_mapping(cls, value: object) -> ContactAnalysis:
        data = _object(value, "analysis"); _exact(data, {"type", "thickness"}, "analysis")
        if data["type"] != "plane_strain": _fail("analysis.type", "only plane_strain is supported")
        return cls("plane_strain", _number(data["thickness"], "analysis.thickness", positive=True))

    def to_dict(self) -> dict[str, object]: return {"type": self.type, "thickness": self.thickness}


@dataclass(frozen=True, slots=True)
class ContactGrain:
    type: str
    radius_m: float | None = None
    initial_center_x_m: float | None = None
    initial_center_y_m: float | None = None
    start_angle_rad: float | None = None
    end_angle_rad: float | None = None
    tip_radius_m: float | None = None
    wedge_height_m: float | None = None
    intrinsic_rake_angle_rad: float | None = None
    intrinsic_clearance_angle_rad: float | None = None
    pose_angle_rad: float | None = None
    initial_reference_x_m: float | None = None
    initial_reference_y_m: float | None = None

    @classmethod
    def from_mapping(cls, value: object) -> ContactGrain:
        data = _object(value, "grain")
        grain_type = data.get("type")
        common = {"type", "radius_m", "initial_center_x_m", "initial_center_y_m"}
        if grain_type == "rigid_circle":
            _exact(data, common, "grain")
            return cls(
                "rigid_circle",
                _number(data["radius_m"], "grain.radius_m", positive=True),
                _number(data["initial_center_x_m"], "grain.initial_center_x_m"),
                _number(data["initial_center_y_m"], "grain.initial_center_y_m"),
            )
        if grain_type == "rigid_circular_arc":
            _exact(data, common | {"start_angle_rad", "end_angle_rad"}, "grain")
            start = _number(data["start_angle_rad"], "grain.start_angle_rad")
            end = _number(data["end_angle_rad"], "grain.end_angle_rad")
            if start < 0.0 or end <= start or end - start >= 2.0 * math.pi:
                _fail("grain arc span", "must be positive, counterclockwise, and below 2*pi")
            return cls(
                "rigid_circular_arc",
                _number(data["radius_m"], "grain.radius_m", positive=True),
                _number(data["initial_center_x_m"], "grain.initial_center_x_m"),
                _number(data["initial_center_y_m"], "grain.initial_center_y_m"),
                start,
                end,
            )
        if grain_type == "rounded_circle":
            fields = {"type", "radius_m", "initial_reference_x_m", "initial_reference_y_m"}
            _exact(data, fields, "grain")
            return cls(
                "rounded_circle",
                radius_m=_number(data["radius_m"], "grain.radius_m", positive=True),
                initial_reference_x_m=_number(data["initial_reference_x_m"], "grain.initial_reference_x_m"),
                initial_reference_y_m=_number(data["initial_reference_y_m"], "grain.initial_reference_y_m"),
            )
        if grain_type == "rounded_wedge":
            fields = {
                "type", "tip_radius_m", "wedge_height_m",
                "intrinsic_rake_angle_rad", "intrinsic_clearance_angle_rad",
                "pose_angle_rad", "initial_reference_x_m", "initial_reference_y_m",
            }
            _exact(data, fields, "grain")
            tip = _number(data["tip_radius_m"], "grain.tip_radius_m", positive=True)
            height = _number(data["wedge_height_m"], "grain.wedge_height_m", positive=True)
            rake = _number(data["intrinsic_rake_angle_rad"], "grain.intrinsic_rake_angle_rad")
            clearance = _number(data["intrinsic_clearance_angle_rad"], "grain.intrinsic_clearance_angle_rad")
            if height <= tip:
                _fail("grain.wedge_height_m", "must be strictly greater than tip_radius_m")
            if not -0.5 * math.pi < rake < 0.5 * math.pi:
                _fail("grain.intrinsic_rake_angle_rad", "must lie strictly between -pi/2 and pi/2")
            if not 0.0 < clearance < 0.5 * math.pi:
                _fail("grain.intrinsic_clearance_angle_rad", "must lie strictly between 0 and pi/2")
            return cls(
                "rounded_wedge",
                tip_radius_m=tip,
                wedge_height_m=height,
                intrinsic_rake_angle_rad=rake,
                intrinsic_clearance_angle_rad=clearance,
                pose_angle_rad=_number(data["pose_angle_rad"], "grain.pose_angle_rad"),
                initial_reference_x_m=_number(data["initial_reference_x_m"], "grain.initial_reference_x_m"),
                initial_reference_y_m=_number(data["initial_reference_y_m"], "grain.initial_reference_y_m"),
            )
        _fail("grain.type", "must be rigid_circle, rigid_circular_arc, rounded_circle, or rounded_wedge")

    def to_dict(self) -> dict[str, object]:
        if self.type == "rounded_circle":
            return {
                "type": self.type, "radius_m": self.radius_m,
                "initial_reference_x_m": self.initial_reference_x_m,
                "initial_reference_y_m": self.initial_reference_y_m,
            }
        if self.type == "rounded_wedge":
            return {
                "type": self.type,
                "tip_radius_m": self.tip_radius_m,
                "wedge_height_m": self.wedge_height_m,
                "intrinsic_rake_angle_rad": self.intrinsic_rake_angle_rad,
                "intrinsic_clearance_angle_rad": self.intrinsic_clearance_angle_rad,
                "pose_angle_rad": self.pose_angle_rad,
                "initial_reference_x_m": self.initial_reference_x_m,
                "initial_reference_y_m": self.initial_reference_y_m,
            }
        result: dict[str, object] = {"type": self.type, "radius_m": self.radius_m, "initial_center_x_m": self.initial_center_x_m, "initial_center_y_m": self.initial_center_y_m}
        if self.type == "rigid_circular_arc":
            result["start_angle_rad"] = self.start_angle_rad
            result["end_angle_rad"] = self.end_angle_rad
        return result


@dataclass(frozen=True, slots=True)
class TrajectoryTarget:
    segment: str
    center_x_m: float | None = None
    center_y_m: float | None = None
    reference_x_m: float | None = None
    reference_y_m: float | None = None

    @classmethod
    def from_mapping(cls, value: object, index: int, *, schema_version: int = 1) -> TrajectoryTarget:
        data = _object(value, f"trajectory.targets[{index}]")
        segment = _text(data["segment"], f"trajectory.targets[{index}].segment")
        if segment not in {"initial", "indentation", "hold", "scratch", "unloading"}: _fail(f"trajectory.targets[{index}].segment", "unsupported segment")
        if schema_version == 1:
            _exact(data, {"segment", "center_x_m", "center_y_m"}, f"trajectory.targets[{index}]")
            return cls(segment, _number(data["center_x_m"], f"trajectory.targets[{index}].center_x_m"), _number(data["center_y_m"], f"trajectory.targets[{index}].center_y_m"))
        _exact(data, {"segment", "reference_x_m", "reference_y_m"}, f"trajectory.targets[{index}]")
        return cls(
            segment,
            reference_x_m=_number(data["reference_x_m"], f"trajectory.targets[{index}].reference_x_m"),
            reference_y_m=_number(data["reference_y_m"], f"trajectory.targets[{index}].reference_y_m"),
        )

    def to_dict(self) -> dict[str, object]:
        if self.reference_x_m is not None:
            return {"segment": self.segment, "reference_x_m": self.reference_x_m, "reference_y_m": self.reference_y_m}
        return {"segment": self.segment, "center_x_m": self.center_x_m, "center_y_m": self.center_y_m}


@dataclass(frozen=True, slots=True)
class ContactTrajectory:
    targets: tuple[TrajectoryTarget, ...]
    nominal_scratch_direction: str | None = None

    @classmethod
    def from_mapping(cls, value: object, *, schema_version: int = 1) -> ContactTrajectory:
        data = _object(value, "trajectory")
        if schema_version == 1:
            _exact(data, {"targets"}, "trajectory")
            direction = None
        else:
            _exact(data, {"nominal_scratch_direction", "targets"}, "trajectory")
            direction = data["nominal_scratch_direction"]
            if direction not in {"positive_x", "negative_x"}:
                _fail("trajectory.nominal_scratch_direction", "must be positive_x or negative_x")
        raw = data["targets"]
        if not isinstance(raw, list) or len(raw) < 2: _fail("trajectory.targets", "must contain at least two targets")
        return cls(tuple(TrajectoryTarget.from_mapping(item, i, schema_version=schema_version) for i, item in enumerate(raw)), direction)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"targets": [item.to_dict() for item in self.targets]}
        if self.nominal_scratch_direction is not None:
            return {"nominal_scratch_direction": self.nominal_scratch_direction, **result}
        return result


@dataclass(frozen=True, slots=True)
class PresetRemoval:
    mode: str
    element_ids: tuple[int, ...] = ()
    region: Mapping[str, object] | None = None

    @classmethod
    def from_mapping(cls, value: object) -> PresetRemoval:
        data = _object(value, "material_topology.preset_removal")
        mode = data.get("mode")
        if mode == "element_ids":
            _exact(data, {"mode", "element_ids"}, "material_topology.preset_removal")
            raw = data["element_ids"]
            if not isinstance(raw, list):
                _fail("material_topology.preset_removal.element_ids", "must be a list")
            ids = tuple(_integer(item, f"material_topology.preset_removal.element_ids[{i}]", 0, 2_000_000_000) for i, item in enumerate(raw))
            if len(ids) != len(set(ids)):
                _fail("material_topology.preset_removal.element_ids", "must contain unique ids")
            if tuple(sorted(ids)) != ids:
                _fail("material_topology.preset_removal.element_ids", "must be sorted")
            return cls("element_ids", ids)
        if mode == "geometric_region":
            _exact(data, {"mode", "region"}, "material_topology.preset_removal")
            region = _object(data["region"], "material_topology.preset_removal.region")
            _exact(region, {"type", "vertices_m", "selection_rule"}, "material_topology.preset_removal.region")
            if region["type"] != "polygon":
                _fail("material_topology.preset_removal.region.type", "must be polygon")
            if region["selection_rule"] != "centroid_inside_or_on_boundary":
                _fail("material_topology.preset_removal.region.selection_rule", "must be centroid_inside_or_on_boundary")
            vertices = region["vertices_m"]
            if not isinstance(vertices, list) or len(vertices) < 3:
                _fail("material_topology.preset_removal.region.vertices_m", "must contain at least three vertices")
            normalized = []
            for index, item in enumerate(vertices):
                if not isinstance(item, list) or len(item) != 2:
                    _fail(f"material_topology.preset_removal.region.vertices_m[{index}]", "must contain two coordinates")
                normalized.append([_number(item[0], f"material_topology.preset_removal.region.vertices_m[{index}][0]"), _number(item[1], f"material_topology.preset_removal.region.vertices_m[{index}][1]")])
            return cls("geometric_region", region={"type": "polygon", "vertices_m": normalized, "selection_rule": "centroid_inside_or_on_boundary"})
        _fail("material_topology.preset_removal.mode", "must be geometric_region or element_ids")

    def to_dict(self) -> dict[str, object]:
        if self.mode == "element_ids":
            return {"mode": self.mode, "element_ids": list(self.element_ids)}
        return {"mode": self.mode, "region": dict(self.region or {})}


@dataclass(frozen=True, slots=True)
class MaterialTopologyInput:
    preset_removal: PresetRemoval

    @classmethod
    def from_mapping(cls, value: object) -> MaterialTopologyInput:
        data = _object(value, "material_topology")
        _exact(data, {"preset_removal"}, "material_topology")
        return cls(PresetRemoval.from_mapping(data["preset_removal"]))

    def to_dict(self) -> dict[str, object]:
        return {"preset_removal": self.preset_removal.to_dict()}


@dataclass(frozen=True, slots=True)
class ContactControls:
    normal_algorithm: str
    normal_penalty_factor: float
    tangential_penalty_ratio: float
    augmented_relaxation: float
    augmented_maximum_iterations: int
    friction_coefficient: float
    friction_regularization_ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> ContactControls:
        data = _object(value, "contact"); fields = {"normal_algorithm", "normal_penalty_factor", "tangential_penalty_ratio", "augmented_relaxation", "augmented_maximum_iterations", "friction_coefficient", "friction_regularization_ratio"}; _exact(data, fields, "contact")
        if data["normal_algorithm"] not in {"penalty", "augmented_lagrangian"}: _fail("contact.normal_algorithm", "must be penalty or augmented_lagrangian")
        return cls(str(data["normal_algorithm"]), _number(data["normal_penalty_factor"], "contact.normal_penalty_factor", positive=True), _number(data["tangential_penalty_ratio"], "contact.tangential_penalty_ratio", positive=True), _number(data["augmented_relaxation"], "contact.augmented_relaxation", positive=True), _integer(data["augmented_maximum_iterations"], "contact.augmented_maximum_iterations", 1, 100), _number(data["friction_coefficient"], "contact.friction_coefficient", nonnegative=True), _number(data["friction_regularization_ratio"], "contact.friction_regularization_ratio", positive=True))

    def to_dict(self) -> dict[str, object]: return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ContactNewton:
    residual_relative_tolerance: float
    residual_absolute_tolerance_N: float
    displacement_relative_tolerance: float
    displacement_absolute_tolerance_m: float
    maximum_iterations: int

    @classmethod
    def from_mapping(cls, value: object) -> ContactNewton:
        data = _object(value, "newton"); fields = {"residual_relative_tolerance", "residual_absolute_tolerance_N", "displacement_relative_tolerance", "displacement_absolute_tolerance_m", "maximum_iterations"}; _exact(data, fields, "newton")
        return cls(
            residual_relative_tolerance=_number(data["residual_relative_tolerance"], "newton.residual_relative_tolerance", positive=True),
            residual_absolute_tolerance_N=_number(data["residual_absolute_tolerance_N"], "newton.residual_absolute_tolerance_N", positive=True),
            displacement_relative_tolerance=_number(data["displacement_relative_tolerance"], "newton.displacement_relative_tolerance", positive=True),
            displacement_absolute_tolerance_m=_number(data["displacement_absolute_tolerance_m"], "newton.displacement_absolute_tolerance_m", positive=True),
            maximum_iterations=_integer(data["maximum_iterations"], "newton.maximum_iterations", 1, 200),
        )

    def to_dict(self) -> dict[str, object]: return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ContactStepControl:
    allow_reduction: bool
    minimum_substep_fraction: float
    maximum_retries: int

    @classmethod
    def from_mapping(cls, value: object) -> ContactStepControl:
        data = _object(value, "step_control"); _exact(data, {"allow_reduction", "minimum_substep_fraction", "maximum_retries"}, "step_control")
        if type(data["allow_reduction"]) is not bool: _fail("step_control.allow_reduction", "must be a strict bool")
        fraction = _number(data["minimum_substep_fraction"], "step_control.minimum_substep_fraction", positive=True)
        if fraction > 1.0: _fail("step_control.minimum_substep_fraction", "must not exceed one")
        return cls(data["allow_reduction"], fraction, _integer(data["maximum_retries"], "step_control.maximum_retries", 0, 20))

    def to_dict(self) -> dict[str, object]: return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ContactOutput:
    save_field_evolution: bool
    snapshot_stride: int

    @classmethod
    def from_mapping(cls, value: object) -> ContactOutput:
        data = _object(value, "output"); _exact(data, {"save_field_evolution", "snapshot_stride"}, "output")
        if type(data["save_field_evolution"]) is not bool: _fail("output.save_field_evolution", "must be a strict bool")
        if data["save_field_evolution"]:
            _fail("output.save_field_evolution", "field evolution is optional and not implemented in 3.0.0")
        return cls(data["save_field_evolution"], _integer(data["snapshot_stride"], "output.snapshot_stride", 1, 10000))

    def to_dict(self) -> dict[str, object]: return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ContactSafety:
    maximum_degrees_of_freedom: int
    warn_displacement_over_local_size: float
    stop_displacement_over_local_size: float
    warn_displacement_over_radius: float
    stop_displacement_over_radius: float
    warn_strain: float
    stop_strain: float

    @classmethod
    def from_mapping(cls, value: object) -> ContactSafety:
        data = _object(value, "safety"); fields = {"maximum_degrees_of_freedom", "warn_displacement_over_local_size", "stop_displacement_over_local_size", "warn_displacement_over_radius", "stop_displacement_over_radius", "warn_strain", "stop_strain"}; _exact(data, fields, "safety")
        result = cls(_integer(data["maximum_degrees_of_freedom"], "safety.maximum_degrees_of_freedom", 2, 200000), *[_number(data[name], f"safety.{name}", positive=True) for name in ("warn_displacement_over_local_size", "stop_displacement_over_local_size", "warn_displacement_over_radius", "stop_displacement_over_radius", "warn_strain", "stop_strain")])
        if result.warn_displacement_over_local_size >= result.stop_displacement_over_local_size or result.warn_displacement_over_radius >= result.stop_displacement_over_radius or result.warn_strain >= result.stop_strain: _fail("safety", "warning thresholds must be below stop thresholds")
        return result

    def to_dict(self) -> dict[str, object]: return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ContactTransient:
    integration_method: str
    grain_speed_m_per_s: float
    initial_time_step_s: float
    minimum_time_step_s: float
    maximum_steps: int

    @classmethod
    def from_mapping(cls, value: object) -> ContactTransient:
        data = _object(value, "transient")
        _exact(data, {"integration_method", "grain_speed_m_per_s", "initial_time_step_s", "minimum_time_step_s", "maximum_steps"}, "transient")
        if data["integration_method"] != "implicit_midpoint":
            _fail("transient.integration_method", "must be implicit_midpoint")
        result = cls(
            "implicit_midpoint",
            _number(data["grain_speed_m_per_s"], "transient.grain_speed_m_per_s", positive=True),
            _number(data["initial_time_step_s"], "transient.initial_time_step_s", positive=True),
            _number(data["minimum_time_step_s"], "transient.minimum_time_step_s", positive=True),
            _integer(data["maximum_steps"], "transient.maximum_steps", 1, 10_000_000),
        )
        if result.minimum_time_step_s > result.initial_time_step_s:
            _fail("transient.minimum_time_step_s", "must not exceed initial_time_step_s")
        return result

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class SingleGrainContactCase:
    single_grain_contact_schema_version: int
    unit_system: str
    model_type: str
    geometry: Rectangle
    mesh: MeshSettings
    analysis: ContactAnalysis
    fixed_boundary: Mapping[str, object]
    material: ContactMaterial
    grain: ContactGrain
    trajectory: ContactTrajectory
    contact: ContactControls
    newton: ContactNewton
    step_control: ContactStepControl
    output: ContactOutput
    safety: ContactSafety
    material_topology: MaterialTopologyInput | None = None
    damage: DamageInput | None = None
    circle_edge_quadrature_order: int = 0
    transient: ContactTransient | None = None

    @property
    def effective_modulus_Pa(self) -> float: return self.material.E / (1.0 - self.material.nu**2)

    @classmethod
    def from_mapping(cls, value: object) -> SingleGrainContactCase:
        root = _object(value, "single_grain_contact_case")
        version = root.get("single_grain_contact_schema_version")
        if type(version) is not int or version not in {1, 2, 3}: _fail("single_grain_contact_schema_version", "must be strict integer 1, 2, or 3")
        fields = {"single_grain_contact_schema_version", "unit_system", "model_type", "geometry", "mesh", "analysis", "fixed_boundary", "material", "grain", "trajectory", "contact", "newton", "step_control", "output", "safety"}
        if version >= 2:
            fields.add("material_topology")
        if version == 3:
            fields.add("damage")
            fields.add("transient")
            if 'contact_integration' in root:
                fields.add('contact_integration')
        _exact(root, fields, "single_grain_contact_case")
        if root["unit_system"] != UNIT_SYSTEM: _fail("unit_system", "only SI is supported")
        if root["model_type"] != MODEL_TYPE: _fail("model_type", f"must be {MODEL_TYPE}")
        geometry = _object(root["geometry"], "geometry"); _exact(geometry, {"type", "width", "height"}, "geometry")
        if geometry["type"] != "rectangle": _fail("geometry.type", "must be rectangle")
        mesh = _object(root["mesh"], "mesh"); _exact(mesh, {"target_size"}, "mesh")
        fixed = _object(root["fixed_boundary"], "fixed_boundary"); _exact(fixed, {"side", "components"}, "fixed_boundary")
        if fixed["side"] != "bottom" or fixed["components"] != ["x", "y"]: _fail("fixed_boundary", "must fix bottom x and y")
        try:
            rectangle = Rectangle(_number(geometry["width"], "geometry.width", positive=True), _number(geometry["height"], "geometry.height", positive=True))
            mesh_settings = MeshSettings(_number(mesh["target_size"], "mesh.target_size", positive=True))
        except ValueError as exc: raise SingleGrainContactValidationError(str(exc)) from exc
        case = cls(version, UNIT_SYSTEM, MODEL_TYPE, rectangle, mesh_settings, ContactAnalysis.from_mapping(root["analysis"]), dict(fixed), ContactMaterial.from_mapping(root["material"], schema_version=version), ContactGrain.from_mapping(root["grain"]), ContactTrajectory.from_mapping(root["trajectory"], schema_version=version), ContactControls.from_mapping(root["contact"]), ContactNewton.from_mapping(root["newton"]), ContactStepControl.from_mapping(root["step_control"]), ContactOutput.from_mapping(root["output"]), ContactSafety.from_mapping(root["safety"]), MaterialTopologyInput.from_mapping(root["material_topology"]) if version >= 2 else None, DamageInput.from_mapping(root["damage"]) if version == 3 else None, 0, ContactTransient.from_mapping(root["transient"]) if version == 3 else None)
        if 'contact_integration' in root:
            integration=_object(root['contact_integration'],'contact_integration')
            _exact(integration,{'rule','order'},'contact_integration')
            if (integration['rule']!='clipped_circle_penalty' or case.grain.type!='rounded_circle'
                or case.contact.normal_algorithm!='penalty' or case.contact.friction_coefficient!=0.):
                _fail('contact_integration','requires full rounded_circle, pure penalty and zero friction')
            from dataclasses import replace
            case=replace(case,circle_edge_quadrature_order=_integer(integration['order'],'contact_integration.order',1,32))
        first = case.trajectory.targets[0]
        if version == 1:
            if first.segment != "initial" or first.center_x_m != case.grain.initial_center_x_m or first.center_y_m != case.grain.initial_center_y_m: _fail("trajectory initial target", "must exactly match the grain initial center")
        elif first.segment != "initial" or first.reference_x_m != case.grain.initial_reference_x_m or first.reference_y_m != case.grain.initial_reference_y_m:
            _fail("trajectory initial target", "must exactly match the grain initial reference point")
        return case

    @classmethod
    def from_json(cls, path: str | Path) -> SingleGrainContactCase:
        try: return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc: raise SingleGrainContactValidationError(f"input_file: {exc}") from exc

    def to_elastoplastic_fem_case(self) -> ElastoplasticFemCase:
        return ElastoplasticFemCase.from_mapping({"elastoplastic_fem_schema_version": 1, "unit_system": "SI", "model_type": "small_strain_plane_strain_j2_fixed_mesh", "geometry": {"type": "rectangle", "width": self.geometry.width, "height": self.geometry.height}, "material": self.material.base.to_dict(), "mesh": {"target_size": self.mesh.target_size}, "analysis": self.analysis.to_dict(), "fixed_boundary": dict(self.fixed_boundary), "load": {"boundary": "top", "distribution": "uniform_over_interval", "x_start_m": 0.0, "x_end_m": self.geometry.width, "peak_force_x_N": 0.0, "peak_force_y_N": 0.0}, "increments": {"loading_step_count": 2, "unloading_step_count": 2}, "newton": self.newton.to_dict(), "step_control": {"allow_reduction": self.step_control.allow_reduction, "minimum_load_factor_increment": self.step_control.minimum_substep_fraction, "maximum_retries": self.step_control.maximum_retries}})

    def to_dict(self) -> dict[str, object]:
        result = {"single_grain_contact_schema_version": self.single_grain_contact_schema_version, "unit_system": UNIT_SYSTEM, "model_type": MODEL_TYPE, "geometry": {"type": "rectangle", "width": self.geometry.width, "height": self.geometry.height}, "mesh": {"target_size": self.mesh.target_size}, "analysis": self.analysis.to_dict(), "fixed_boundary": dict(self.fixed_boundary), "material": self.material.to_dict(), "grain": self.grain.to_dict(), "trajectory": self.trajectory.to_dict(), "contact": self.contact.to_dict(), "newton": self.newton.to_dict(), "step_control": self.step_control.to_dict(), "output": self.output.to_dict(), "safety": self.safety.to_dict()}
        if self.material_topology is not None:
            result["material_topology"] = self.material_topology.to_dict()
        if self.damage is not None:
            result["damage"] = self.damage.to_dict()
        if self.transient is not None:
            result["transient"] = self.transient.to_dict()
        if getattr(self,'circle_edge_quadrature_order',0):
            result['contact_integration']={'rule':'clipped_circle_penalty','order':self.circle_edge_quadrature_order}
        return result

    def input_fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
