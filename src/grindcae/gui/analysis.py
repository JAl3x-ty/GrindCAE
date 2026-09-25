"""Unified analysis contracts and strict input adapters for workbench 2.8."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from grindcae.evolved_fem import EvolvedFemCase
from grindcae.history_pass import FixedMeshElastoplasticHistoryPassCase
from grindcae.mechanism_history_pass import MechanismHistoryPassCase
from grindcae.pass_scan import PassScanCase
from grindcae.single_position_elastoplastic import SinglePositionElastoplasticCase
from grindcae.single_grain_contact import SingleGrainContactCase
from grindcae.statistical_grain_load import (
    GrainPopulationSettings,
    LoadDistributionSettings,
    StatisticalGrainLoadCase,
    recommended_grain_population_settings,
    recommended_load_distribution_settings,
)
from grindcae.presentation_labels import ANALYSIS_MODE_LABELS_ZH

from .geometry import ParametricGeometryCase
from .material_workspace import WorkbenchMaterialCase
from .mesh_workspace import WorkbenchMeshCase
from .process_workspace import WorkbenchProcessCase, WorkbenchWheelCase
from .project import GrindCaeProject, ProjectNodeStatus


ANALYSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AnalysisModeDefinition:
    mode_id: str
    display_name: str
    material_requirement: str
    process_scope: str
    analysis_assumption: str
    load_type: str
    case_type_name: str
    result_format: str
    summary_filename: str
    representative_image_filename: str
    model_note: str
    enabled: bool = True


def _mode(
    mode_id: str,
    display_name: str,
    material: str,
    scope: str,
    assumption: str,
    load_type: str,
    case_type: str,
    result_format: str,
    summary: str,
    image: str,
    note: str,
    *,
    enabled: bool = True,
) -> AnalysisModeDefinition:
    return AnalysisModeDefinition(
        mode_id, display_name, material, scope, assumption, load_type,
        case_type, result_format, summary, image, note, enabled
    )


ANALYSIS_MODE_DEFINITIONS = {
    item.mode_id: item
    for item in (
        _mode(
            "literature_grinding_force", ANALYSIS_MODE_LABELS_ZH["literature_grinding_force"],
            "literature_440C", "force_prediction", "analytical_statistical", "literature_grain_force",
            "ReconstructionCase", "grindcae_zhang2017_reconstruction_v2", "summary.json", "force_components.png",
            "440C文献磨削力重建；使用本模式独立参数，可显式同步工程工艺。无需FE网格，不输出应力场或真实材料分离。",
        ),
        _mode(
            "linear_elastic_single_position", ANALYSIS_MODE_LABELS_ZH["linear_elastic_single_position"], "elastic",
            "single_position", "plane_stress", "uniform_empirical_load",
            "EvolvedFemCase",
            "grindcae_phase_4c3b_evolved_surface_empirical_load_postprocess",
            "summary.json", "stress_contour.png",
            "当前任务只使用材料的弹性模量和泊松比，不考虑屈服、塑性累积和残余状态。",
        ),
        _mode(
            "linear_elastic_single_pass", ANALYSIS_MODE_LABELS_ZH["linear_elastic_single_pass"], "elastic",
            "complete_single_pass", "plane_stress", "uniform_empirical_load",
            "PassScanCase", "grindcae_phase_4c4_single_pass_quasi_static_scan",
            "scan_summary.json", "scan_overview.png",
            "每个位置是相互独立的线弹性准静态计算，不传递塑性历史。",
        ),
        _mode(
            "elastoplastic_single_position", ANALYSIS_MODE_LABELS_ZH["elastoplastic_single_position"], "elastoplastic",
            "single_position_loading_unloading", "plane_strain",
            "uniform_empirical_load", "SinglePositionElastoplasticCase",
            "grindcae_phase_6a3_single_position_elastoplastic", "summary.json",
            "surface_recovery.png",
            "当前任务采用小应变平面应变 J2 弹塑性模型，执行单位置加载—卸载。",
        ),
        _mode(
            "elastoplastic_single_pass", ANALYSIS_MODE_LABELS_ZH["elastoplastic_single_pass"], "elastoplastic",
            "complete_single_pass", "plane_strain", "uniform_equivalent_moving_load",
            "FixedMeshElastoplasticHistoryPassCase",
            "grindcae_phase_6b2_fixed_mesh_elastoplastic_history_pass",
            "pass_summary.json", "pass_residual_state.png",
            "当前任务沿完整单程传递塑性历史，最终外载荷为零，但残余状态可以非零。",
        ),
        _mode(
            "mechanism_elastoplastic_single_pass", ANALYSIS_MODE_LABELS_ZH["mechanism_elastoplastic_single_pass"],
            "elastoplastic_ductile_metal_aluminum_oxide", "complete_single_pass",
            "plane_strain", "statistical_mechanism_nonuniform_load",
            "MechanismHistoryPassCase",
            "grindcae_phase_7a3_statistical_mechanism_elastoplastic_history_comparison",
            "summary.json", "mechanism_history_comparison.png",
            "当前任务使用统计等效磨粒群和宏观尺度非均匀载荷投影，不是逐颗磨粒高保真解析，也不预测真实粗糙度。",
        ),
        _mode(
            "single_grain_high_fidelity", ANALYSIS_MODE_LABELS_ZH["single_grain_high_fidelity"],
            "elastoplastic", "single_grain_trajectory", "plane_strain",
            "prescribed_rigid_grain_contact", "SingleGrainContactCase",
            "grindcae_single_grain_real_contact_v1", "summary.json",
            "reaction_history.png",
            "二维小应变固定网格单磨粒真实接触；残余沟槽是卸载后表面位移轮廓，不代表材料删除或切屑形成。",
        ),
    )
}

# Keep the six legacy modes and their default order stable.
ANALYSIS_MODE_DEFINITIONS["literature_grinding_force"] = ANALYSIS_MODE_DEFINITIONS.pop("literature_grinding_force")
ANALYSIS_MODE_DEFINITIONS["literature_elastoplastic_single_pass"] = _mode(
    "literature_elastoplastic_single_pass", ANALYSIS_MODE_LABELS_ZH["literature_elastoplastic_single_pass"],
    "elastoplastic", "complete_single_pass", "plane_strain", "literature_strip_load",
    "LiteratureHistoryCase", "grindcae_literature_elastoplastic_history_v1", "summary.json",
    "literature_history_comparison.png",
    "3.8.1候选功能。工程工艺与屈服强度驱动论文力重算；砂轮粒度/角度/修整高度来自文献参数页。"
    "同网格比较均匀与条带载荷的J2历史；其他材料使用440C经验关系属于未标定迁移。",
)

FUTURE_ANALYSIS_MODE_DEFINITIONS = {
    item.mode_id: item
    for item in (
        _mode("thermo_mechanical_single_position", "热力耦合·单位置", "future", "single_position", "future", "future", "", "", "", "", "当前版本暂未启用。", enabled=False),
        _mode("thermo_mechanical_single_pass", "热力耦合·单程", "future", "single_pass", "future", "future", "", "", "", "", "当前版本暂未启用。", enabled=False),
        _mode("multi_pass_elastoplastic", "多道次磨削", "future", "multi_pass", "future", "future", "", "", "", "", "当前版本暂未启用。", enabled=False),
        _mode("roughness_prediction", "粗糙度预测", "future", "roughness", "future", "future", "", "", "", "", "当前版本暂未启用。", enabled=False),
    )
}


def analysis_mode_definition(
    mode_id: str, *, require_enabled: bool = False
) -> AnalysisModeDefinition:
    item = ANALYSIS_MODE_DEFINITIONS.get(mode_id) or FUTURE_ANALYSIS_MODE_DEFINITIONS.get(mode_id)
    if item is None:
        raise ValueError(f"未知分析任务：{mode_id}")
    if require_enabled and not item.enabled:
        raise ValueError(f"分析任务“{item.display_name}”暂未启用。")
    return item


def analysis_mode_by_display_name(display_name: str) -> AnalysisModeDefinition:
    for item in (*ANALYSIS_MODE_DEFINITIONS.values(), *FUTURE_ANALYSIS_MODE_DEFINITIONS.values()):
        if item.display_name == display_name:
            return item
    raise ValueError(f"未知分析任务名称：{display_name}")


def _strict_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label}必须是 {minimum} 到 {maximum} 之间的严格整数。")
    return value


def _positive(value: object, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是有限数值。")
    number = float(value)
    if not math.isfinite(number) or (number < 0.0 if allow_zero else number <= 0.0):
        raise ValueError(f"{label}必须是{'非负' if allow_zero else '有限正'}数值。")
    return number


@dataclass(frozen=True)
class NewtonAnalysisSettings:
    residual_relative_tolerance: float = 1.0e-8
    residual_absolute_tolerance_N: float = 1.0e-6
    displacement_relative_tolerance: float = 1.0e-8
    displacement_absolute_tolerance_m: float = 1.0e-14
    maximum_iterations: int = 40

    @classmethod
    def from_mapping(cls, value: object) -> "NewtonAnalysisSettings":
        if not isinstance(value, Mapping): raise ValueError("Newton 设置必须是对象。")
        result = cls(**value)
        _positive(result.residual_relative_tolerance, "残差相对容差")
        _positive(result.residual_absolute_tolerance_N, "残差绝对容差")
        _positive(result.displacement_relative_tolerance, "位移相对容差")
        _positive(result.displacement_absolute_tolerance_m, "位移绝对容差")
        _strict_int(result.maximum_iterations, "最大 Newton 迭代次数", 1, 1000)
        return result

    def to_dict(self) -> dict[str, object]: return dict(self.__dict__)


@dataclass(frozen=True)
class StepControlAnalysisSettings:
    allow_reduction: bool = True
    minimum_load_factor_increment: float = 1.0e-4
    maximum_retries: int = 10

    @classmethod
    def from_mapping(cls, value: object) -> "StepControlAnalysisSettings":
        if not isinstance(value, Mapping): raise ValueError("步长控制设置必须是对象。")
        result = cls(**value)
        if type(result.allow_reduction) is not bool: raise ValueError("允许步长缩减必须是布尔值。")
        _positive(result.minimum_load_factor_increment, "最小载荷因子增量")
        _strict_int(result.maximum_retries, "最大步长重试次数", 0, 1000)
        return result

    def to_dict(self) -> dict[str, object]: return dict(self.__dict__)


@dataclass(frozen=True)
class HistoryTransitionAnalysisSettings:
    base_substep_count: int = 2
    final_unloading_substep_count: int = 4
    allow_reduction: bool = True
    minimum_substep_fraction: float = 0.03125
    maximum_retries: int = 8

    @classmethod
    def from_mapping(cls, value: object) -> "HistoryTransitionAnalysisSettings":
        if not isinstance(value, Mapping): raise ValueError("历史转换设置必须是对象。")
        result = cls(**value)
        _strict_int(result.base_substep_count, "基础子步数", 1, 1000)
        _strict_int(result.final_unloading_substep_count, "最终卸载子步数", 1, 1000)
        if type(result.allow_reduction) is not bool: raise ValueError("允许历史子步缩减必须是布尔值。")
        fraction = _positive(result.minimum_substep_fraction, "最小子步比例")
        if fraction > 1.0: raise ValueError("最小子步比例不能大于 1。")
        _strict_int(result.maximum_retries, "最大历史重试次数", 0, 1000)
        return result

    def to_dict(self) -> dict[str, object]: return dict(self.__dict__)


@dataclass(frozen=True)
class GrainPopulationAnalysisSettings:
    active_density_factor: float = recommended_grain_population_settings().active_density_factor
    minimum_equivalent_group_count: int = recommended_grain_population_settings().minimum_equivalent_group_count
    maximum_equivalent_group_count: int = recommended_grain_population_settings().maximum_equivalent_group_count
    random_seed: int = recommended_grain_population_settings().random_seed

    @classmethod
    def from_mapping(cls, value: object) -> "GrainPopulationAnalysisSettings":
        if not isinstance(value, Mapping): raise ValueError("统计磨粒设置必须是对象。")
        result = cls(**value)
        _positive(result.active_density_factor, "有效密度因子")
        minimum = _strict_int(result.minimum_equivalent_group_count, "最小等效磨粒组数", 1, 100000)
        maximum = _strict_int(result.maximum_equivalent_group_count, "最大等效磨粒组数", 1, 100000)
        if minimum > maximum: raise ValueError("最小等效磨粒组数不能大于最大值。")
        _strict_int(result.random_seed, "随机种子", 0, 2**63 - 1)
        return result

    def to_dict(self) -> dict[str, object]: return dict(self.__dict__)

    @classmethod
    def from_shared(cls, value: GrainPopulationSettings) -> "GrainPopulationAnalysisSettings":
        return cls(**value.to_dict())


@dataclass(frozen=True)
class LoadDistributionAnalysisSettings:
    point_count: int = recommended_load_distribution_settings().point_count
    kernel_width_factor: float = recommended_load_distribution_settings().kernel_width_factor
    rubbing_kernel_scale: float = recommended_load_distribution_settings().rubbing_kernel_scale
    ploughing_kernel_scale: float = recommended_load_distribution_settings().ploughing_kernel_scale
    cutting_kernel_scale: float = recommended_load_distribution_settings().cutting_kernel_scale

    @classmethod
    def from_mapping(cls, value: object) -> "LoadDistributionAnalysisSettings":
        if not isinstance(value, Mapping): raise ValueError("载荷分布设置必须是对象。")
        result = cls(**value)
        count = _strict_int(result.point_count, "载荷分布采样点数", 201, 5001)
        if count % 2 == 0: raise ValueError("载荷分布采样点数必须为奇数。")
        for field_name, label in (("kernel_width_factor", "公共核宽度系数"), ("rubbing_kernel_scale", "滑擦核尺度"), ("ploughing_kernel_scale", "耕犁核尺度"), ("cutting_kernel_scale", "切削核尺度")):
            _positive(getattr(result, field_name), label)
        return result

    def to_dict(self) -> dict[str, object]: return dict(self.__dict__)

    @classmethod
    def from_shared(cls, value: LoadDistributionSettings) -> "LoadDistributionAnalysisSettings":
        return cls(**value.to_dict())


@dataclass(frozen=True)
class SingleGrainAnalysisSettings:
    workpiece_width_m: float = 400.0e-6
    workpiece_height_m: float = 200.0e-6
    mesh_target_size_m: float = 20.0e-6
    workpiece_thickness_m: float = 1.0e-3
    grain_type: str = "rigid_circle"
    grain_radius_m: float = 200.0e-6
    grain_arc_start_angle_rad: float = 1.25 * math.pi
    grain_arc_end_angle_rad: float = 1.75 * math.pi
    grain_tip_radius_m: float = 20.0e-6
    grain_wedge_height_m: float = 80.0e-6
    intrinsic_rake_angle_rad: float = 0.0
    intrinsic_clearance_angle_rad: float = math.pi / 4.0
    grain_pose_angle_rad: float = 0.0
    nominal_scratch_direction: str = "positive_x"
    preset_removal_mode: str = "element_ids"
    preset_removal_element_ids: tuple[int, ...] = ()
    preset_removal_polygon_vertices_m: tuple[tuple[float, float], ...] = ()
    initial_clearance_m: float = 1.0e-6
    indentation_depth_m: float = 1.0e-6
    scratch_distance_m: float = 20.0e-6
    normal_algorithm: str = "penalty"
    normal_penalty_factor: float = 10.0
    tangential_penalty_ratio: float = 0.5
    augmented_relaxation: float = 1.0
    augmented_maximum_iterations: int = 8
    friction_coefficient: float = 0.15
    friction_regularization_ratio: float = 1.0e-6
    damage_separation_enabled: bool = False
    damage_evolution: str = "linear_fracture_energy_softening"
    damage_failure_strain_table: tuple[tuple[float, float], ...] = (
        (-1.0, 1.0),
        (1.0, 1.0),
    )
    damage_triaxiality_cutoff: float = -1.0
    damage_fracture_energy_J_per_m2: float = 1000.0
    damage_event_tolerance: float = 1.0e-6
    damage_maximum_separation_events_per_position: int = 25
    damage_area_absolute_tolerance_m2: float = 1.0e-16
    damage_parameter_source: str = "synthetic development baseline"
    damage_calibration_status: str = "uncalibrated"
    damage_applicability_notes: str = "2D small-strain fixed-background-mesh development model."
    density_kg_per_m3: float = 7850.0
    grain_speed_m_per_s: float = 0.01
    transient_initial_time_step_s: float = 1.25e-8
    transient_minimum_time_step_s: float = 1.5625e-9
    transient_maximum_steps: int = 100000

    @classmethod
    def from_mapping(cls, value: object) -> "SingleGrainAnalysisSettings":
        if not isinstance(value, Mapping):
            raise ValueError("单磨粒接触设置必须是对象。")
        expected = set(cls.__dataclass_fields__)
        legacy_arc_fields = {
            "grain_type",
            "grain_arc_start_angle_rad",
            "grain_arc_end_angle_rad",
        }
        schema_2_fields = {
            "grain_tip_radius_m", "grain_wedge_height_m", "intrinsic_rake_angle_rad",
            "intrinsic_clearance_angle_rad", "grain_pose_angle_rad", "nominal_scratch_direction",
            "preset_removal_mode", "preset_removal_element_ids", "preset_removal_polygon_vertices_m",
        }
        schema_3_fields = {
            "damage_separation_enabled", "damage_evolution", "damage_failure_strain_table",
            "damage_triaxiality_cutoff", "damage_fracture_energy_J_per_m2",
            "damage_event_tolerance",
            "damage_maximum_separation_events_per_position",
            "damage_area_absolute_tolerance_m2", "damage_parameter_source",
            "damage_calibration_status", "damage_applicability_notes",
            "density_kg_per_m3", "grain_speed_m_per_s",
            "transient_initial_time_step_s", "transient_minimum_time_step_s",
            "transient_maximum_steps",
        }
        actual = set(value)
        # Historical workbench settings have no model selector. Their saved
        # meaning remains linear; choosing the rational route is explicit.
        if actual in (expected - {"damage_evolution"}, expected - legacy_arc_fields - {"damage_evolution"}):
            value = dict(value, damage_evolution="linear_fracture_energy_softening")
            actual = set(value)
        legacy = expected - schema_2_fields - schema_3_fields
        schema_2_only = expected - schema_3_fields
        legacy_without_arc = legacy - legacy_arc_fields
        expected_without_arc = expected - legacy_arc_fields
        schema_2_without_arc = schema_2_only - legacy_arc_fields
        if actual not in (
            expected, expected_without_arc, schema_2_only, schema_2_without_arc,
            legacy, legacy_without_arc,
        ):
            raise ValueError("单磨粒接触设置字段不完整或包含未知字段。")
        normalized = dict(value)
        defaults = cls()
        if actual in (legacy, legacy_without_arc):
            for name in schema_2_fields | schema_3_fields:
                normalized[name] = getattr(defaults, name)
        if actual in (schema_2_only, schema_2_without_arc):
            for name in schema_3_fields:
                normalized[name] = getattr(defaults, name)
        if actual in (expected_without_arc, schema_2_without_arc, legacy_without_arc):
            defaults = cls()
            normalized.update(
                grain_type=defaults.grain_type,
                grain_arc_start_angle_rad=defaults.grain_arc_start_angle_rad,
                grain_arc_end_angle_rad=defaults.grain_arc_end_angle_rad,
            )
        result = cls(**normalized)
        if result.damage_evolution not in ("linear_fracture_energy_softening", "energetic_rational_fracture"):
            raise ValueError("损伤演化 damage_evolution 必须是已登记的模型。")
        for name, label in (
            ("workpiece_width_m", "单磨粒工件宽度"),
            ("workpiece_height_m", "单磨粒工件高度"),
            ("mesh_target_size_m", "单磨粒网格尺寸"),
            ("workpiece_thickness_m", "单磨粒显式厚度"),
            ("grain_radius_m", "磨粒圆角半径"),
            ("grain_tip_radius_m", "楔形磨粒尖端半径"),
            ("grain_wedge_height_m", "楔形磨粒高度"),
            ("initial_clearance_m", "初始间隙"),
            ("indentation_depth_m", "压入深度"),
            ("normal_penalty_factor", "法向罚因子"),
            ("tangential_penalty_ratio", "切向罚刚度比例"),
            ("augmented_relaxation", "增广松弛系数"),
            ("friction_regularization_ratio", "摩擦正则化比例"),
        ):
            _positive(getattr(result, name), label)
        for name in (
            "density_kg_per_m3", "grain_speed_m_per_s",
            "transient_initial_time_step_s", "transient_minimum_time_step_s",
        ):
            _positive(getattr(result, name), name)
        _positive(result.scratch_distance_m, "划擦距离", allow_zero=True)
        _positive(result.friction_coefficient, "摩擦系数", allow_zero=True)
        _strict_int(result.augmented_maximum_iterations, "增广最大迭代", 1, 100)
        _strict_int(result.transient_maximum_steps, "transient_maximum_steps", 1, 10_000_000)
        if result.transient_minimum_time_step_s > result.transient_initial_time_step_s:
            raise ValueError("transient minimum time step must not exceed initial time step")
        if result.normal_algorithm not in {"penalty", "augmented_lagrangian"}:
            raise ValueError("法向接触算法必须为 penalty 或 augmented_lagrangian。")
        if result.grain_type not in {"rigid_circle", "rigid_circular_arc", "rounded_circle", "rounded_wedge"}:
            raise ValueError("磨粒类型必须为 rigid_circle、rigid_circular_arc、rounded_circle 或 rounded_wedge。")
        if (
            result.grain_arc_start_angle_rad < 0.0
            or result.grain_arc_end_angle_rad <= result.grain_arc_start_angle_rad
            or result.grain_arc_end_angle_rad - result.grain_arc_start_angle_rad
            >= 2.0 * math.pi
        ):
            raise ValueError("圆弧角度必须按逆时针给出正跨度，且跨度小于 2π。")
        if result.mesh_target_size_m >= min(result.workpiece_width_m, result.workpiece_height_m):
            raise ValueError("单磨粒网格尺寸必须小于工件宽度和高度。")
        if result.indentation_depth_m >= result.grain_radius_m:
            raise ValueError("压入深度必须小于磨粒圆角半径。")
        if result.grain_wedge_height_m <= result.grain_tip_radius_m:
            raise ValueError("楔形磨粒高度必须大于尖端半径。")
        if not -0.5 * math.pi < result.intrinsic_rake_angle_rad < 0.5 * math.pi:
            raise ValueError("本征前角必须位于 -π/2 与 π/2 之间。")
        if not 0.0 < result.intrinsic_clearance_angle_rad < 0.5 * math.pi:
            raise ValueError("本征后角必须位于 0 与 π/2 之间。")
        if result.nominal_scratch_direction not in {"positive_x", "negative_x"}:
            raise ValueError("名义划擦方向必须为 positive_x 或 negative_x。")
        if result.preset_removal_mode not in {"element_ids", "geometric_region"}:
            raise ValueError("预设去除模式必须为 element_ids 或 geometric_region。")
        ids = tuple(result.preset_removal_element_ids)
        if any(type(item) is not int or item < 0 for item in ids) or tuple(sorted(set(ids))) != ids:
            raise ValueError("预设去除单元 ID 必须是排序后的唯一非负整数。")
        vertices = tuple(tuple(point) for point in result.preset_removal_polygon_vertices_m)
        if result.preset_removal_mode == "geometric_region":
            if len(vertices) < 3 or any(len(point) != 2 or not all(math.isfinite(float(v)) for v in point) for point in vertices):
                raise ValueError("预设去除多边形必须包含至少三个有限二维顶点。")
        object.__setattr__(result, "preset_removal_element_ids", ids)
        object.__setattr__(result, "preset_removal_polygon_vertices_m", vertices)
        if type(result.damage_separation_enabled) is not bool:
            raise ValueError("损伤分离开关必须为布尔值。")
        damage_table = tuple(tuple(row) for row in result.damage_failure_strain_table)
        if result.damage_separation_enabled:
            if result.grain_type not in {"rounded_circle", "rounded_wedge"}:
                raise ValueError("Schema 3 损伤分离只支持 rounded_circle 或 rounded_wedge。")
            from grindcae.single_grain_contact.damage_models import DamageInput

            DamageInput.from_mapping({
                "model": "ductile_triaxiality_table",
                "failure_strain_table": [list(row) for row in damage_table],
                "triaxiality_cutoff": result.damage_triaxiality_cutoff,
                "evolution": result.damage_evolution,
                "fracture_energy_J_per_m2": result.damage_fracture_energy_J_per_m2,
                "characteristic_length": "equivalent_circle_diameter_reference_area",
                "damage_event_tolerance": result.damage_event_tolerance,
                "maximum_separation_events_per_position": result.damage_maximum_separation_events_per_position,
                "area_absolute_tolerance_m2": result.damage_area_absolute_tolerance_m2,
                "parameter_source": result.damage_parameter_source,
                "calibration_status": result.damage_calibration_status,
                "applicability_notes": result.damage_applicability_notes,
            })
        object.__setattr__(result, "damage_failure_strain_table", damage_table)
        if result.scratch_distance_m >= result.workpiece_width_m / 2.0:
            raise ValueError("划擦距离必须小于单磨粒工件半宽。")
        return result

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AnalysisSettings:
    literature_case: dict[str, Any] | None = None
    scan_base_position_count: int = 21
    trajectory_base_point_count: int = 401
    loading_step_count: int = 8
    unloading_step_count: int = 8
    newton: NewtonAnalysisSettings = field(default_factory=NewtonAnalysisSettings)
    step_control: StepControlAnalysisSettings = field(default_factory=StepControlAnalysisSettings)
    history_transition: HistoryTransitionAnalysisSettings = field(default_factory=HistoryTransitionAnalysisSettings)
    grain_population: GrainPopulationAnalysisSettings = field(default_factory=GrainPopulationAnalysisSettings)
    load_distribution: LoadDistributionAnalysisSettings = field(default_factory=LoadDistributionAnalysisSettings)
    single_grain: SingleGrainAnalysisSettings = field(default_factory=SingleGrainAnalysisSettings)

    @classmethod
    def recommended(cls) -> "AnalysisSettings":
        return cls(
            grain_population=GrainPopulationAnalysisSettings.from_shared(
                recommended_grain_population_settings()
            ),
            load_distribution=LoadDistributionAnalysisSettings.from_shared(
                recommended_load_distribution_settings()
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "AnalysisSettings":
        if not isinstance(value, Mapping): raise ValueError("分析高级设置必须是对象。")
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        if actual not in (expected, expected - {"single_grain"}, expected - {"literature_case"}, expected - {"single_grain", "literature_case"}):
            raise ValueError("分析高级设置字段不完整或包含未知字段。")
        literature = value.get("literature_case")
        if literature is not None:
            from grindcae380.reconstruction import ReconstructionCase
            literature = ReconstructionCase.from_mapping(literature).to_dict()
        result = cls(
            literature_case=literature,
            scan_base_position_count=value["scan_base_position_count"],
            trajectory_base_point_count=value["trajectory_base_point_count"],
            loading_step_count=value["loading_step_count"],
            unloading_step_count=value["unloading_step_count"],
            newton=NewtonAnalysisSettings.from_mapping(value["newton"]),
            step_control=StepControlAnalysisSettings.from_mapping(value["step_control"]),
            history_transition=HistoryTransitionAnalysisSettings.from_mapping(value["history_transition"]),
            grain_population=GrainPopulationAnalysisSettings.from_mapping(value["grain_population"]),
            load_distribution=LoadDistributionAnalysisSettings.from_mapping(value["load_distribution"]),
            single_grain=SingleGrainAnalysisSettings.from_mapping(
                value.get("single_grain", SingleGrainAnalysisSettings().to_dict())
            ),
        )
        _strict_int(result.scan_base_position_count, "基础扫描位置数", 5, 101)
        _strict_int(result.trajectory_base_point_count, "轨迹基础采样点数", 101, 10001)
        _strict_int(result.loading_step_count, "加载步数", 1, 10000)
        _strict_int(result.unloading_step_count, "卸载步数", 1, 10000)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "scan_base_position_count": self.scan_base_position_count,
            "trajectory_base_point_count": self.trajectory_base_point_count,
            "loading_step_count": self.loading_step_count,
            "unloading_step_count": self.unloading_step_count,
            "newton": self.newton.to_dict(), "step_control": self.step_control.to_dict(),
            "history_transition": self.history_transition.to_dict(),
            "grain_population": self.grain_population.to_dict(),
            "load_distribution": self.load_distribution.to_dict(),
            "single_grain": self.single_grain.to_dict(),
            **({"literature_case": self.literature_case} if self.literature_case is not None else {}),
        }


@dataclass(frozen=True)
class AnalysisProjectData:
    mode: str
    display_name: str
    status: str
    output_directory: str
    settings: AnalysisSettings
    input_fingerprint: str = ""
    input_snapshot: dict[str, Any] = field(default_factory=dict)
    last_run: dict[str, Any] = field(default_factory=lambda: {"status": "not_run", "result_id": None, "started_at": None, "finished_at": None, "error_message": None})
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: object) -> "AnalysisProjectData":
        if not isinstance(value, Mapping): raise ValueError("工程 analysis 必须是对象。")
        known = {"analysis_schema_version", "mode", "display_name", "status", "output_directory", "settings", "input_fingerprint", "input_snapshot", "last_run"}
        missing = known - set(value)
        if missing: raise ValueError(f"工程 analysis 缺少字段：{', '.join(sorted(missing))}")
        if type(value["analysis_schema_version"]) is not int or value["analysis_schema_version"] != 1: raise ValueError("不支持的分析设置版本。")
        definition = analysis_mode_definition(str(value["mode"]), require_enabled=True)
        if value["display_name"] != definition.display_name: raise ValueError("分析任务中文名称与内部任务不一致。")
        return cls(
            mode=definition.mode_id, display_name=definition.display_name,
            status=str(value["status"]), output_directory=str(value["output_directory"]),
            settings=AnalysisSettings.from_mapping(value["settings"]),
            input_fingerprint=str(value["input_fingerprint"]),
            input_snapshot=dict(value["input_snapshot"]), last_run=dict(value["last_run"]),
            extra_fields={key: item for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION, "mode": self.mode,
            "display_name": self.display_name, "status": self.status,
            "output_directory": self.output_directory,
            "input_fingerprint": self.input_fingerprint,
            "input_snapshot": self.input_snapshot, "settings": self.settings.to_dict(),
            "last_run": self.last_run, **self.extra_fields,
        }


@dataclass(frozen=True)
class BuiltAnalysisInput:
    definition: AnalysisModeDefinition
    case: Any
    normalized_input: dict[str, Any]
    input_fingerprint: str


class AnalysisInputBuilder:
    """Build only existing strict solver cases from persisted workbench data."""

    def build(self, project: GrindCaeProject, mode_id: str, settings: AnalysisSettings) -> BuiltAnalysisInput:
        definition = analysis_mode_definition(mode_id, require_enabled=True)
        if mode_id == "literature_grinding_force":
            from .literature_integration import configured_case
            case = configured_case(settings)
            normalized = case.to_dict()
            serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            return BuiltAnalysisInput(definition, case, normalized, hashlib.sha256(serialized.encode("utf-8")).hexdigest())
        geometry, mesh, material, wheel, process = self._validated_sources(project)
        if definition.material_requirement.startswith("elastoplastic"):
            if material.behavior != "elastoplastic" or material.yield_strength_pa is None or material.tangent_modulus_pa is None:
                raise ValueError("弹塑性任务需要完整的屈服强度和切线模量。")
        if mode_id == "literature_elastoplastic_single_pass":
            from .literature_integration import configured_case
            from grindcae.literature_history.models import compose_case
            history = self._plastic_pass(geometry, mesh, material, process, settings)
            provenance = json.dumps(material.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)
            case = compose_case(configured_case(settings), history, provenance)
            return BuiltAnalysisInput(definition, case, case.to_dict(), case.input_fingerprint)
        if mode_id == "mechanism_elastoplastic_single_pass":
            if material.family != "ductile_metal" or wheel.abrasive_material != "aluminum_oxide":
                raise ValueError("机制化磨削当前只支持刚玉砂轮和延性金属。")
            if not isinstance(wheel.resolved_specification, Mapping):
                raise ValueError("请先完成砂轮粒度解析。")
        if mode_id == "linear_elastic_single_position": case = self._linear_single(geometry, mesh, material, process, settings)
        elif mode_id == "linear_elastic_single_pass": case = self._linear_pass(geometry, mesh, material, process, settings)
        elif mode_id == "elastoplastic_single_position": case = self._plastic_single(geometry, mesh, material, process, settings)
        elif mode_id == "elastoplastic_single_pass": case = self._plastic_pass(geometry, mesh, material, process, settings)
        elif mode_id == "single_grain_high_fidelity": case = self._single_grain(material, settings)
        else: case = self._mechanism_pass(geometry, mesh, material, wheel, process, settings)
        normalized = case.to_dict()
        fingerprint_payload = {"mode": mode_id, "settings": settings.to_dict(), "strict_input": normalized}
        serialized = json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return BuiltAnalysisInput(definition, case, normalized, hashlib.sha256(serialized.encode("utf-8")).hexdigest())

    @staticmethod
    def _validated_sources(project: GrindCaeProject):
        for node in ("几何", "网格", "材料", "工艺"):
            if project.node_status(node) is not ProjectNodeStatus.COMPLETED:
                raise ValueError(f"请先完成并确认“{node}”节点。")
        try:
            return (
                ParametricGeometryCase.from_mapping(project.geometry),
                WorkbenchMeshCase.from_mapping(project.mesh),
                WorkbenchMaterialCase.from_mapping(project.material),
                WorkbenchWheelCase.from_mapping(project.wheel),
                WorkbenchProcessCase.from_mapping(project.process),
            )
        except ValueError as exc:
            raise ValueError(f"工程参数无效：{exc}") from exc

    @staticmethod
    def _pass_load(g, p, settings):
        source = p.source_description or "当前工程记录的用户参数，真实性需由用户确认。"
        return {
            "pass_load_schema_version": 1, "unit_system": "SI",
            "model_type": "single_pass_contact_ratio_empirical_load",
            "trajectory": {
                "trajectory_schema_version": 1, "unit_system": "SI",
                "model_type": "single_pass_ideal_wheel_profile",
                "workpiece": {"length_m": g.workpiece_length_m, "original_surface_height_m": g.workpiece_height_m},
                "wheel": {"diameter_m": g.wheel_diameter_m},
                "single_pass": {"depth_of_cut_m": g.depth_of_cut_m, "relative_feed_direction": g.relative_feed_direction, "wheel_lowest_point_x_m": g.wheel_lowest_point_x_m},
                "sampling": {"base_point_count": settings.trajectory_base_point_count},
            },
            "force_model": {
                "force_model_schema_version": 1, "unit_system": "SI",
                "model_type": "specific_grinding_energy_force_ratio",
                "process": {"wheel_surface_speed_m_per_s": p.wheel_surface_speed_m_per_s, "workpiece_feed_speed_m_per_s": p.workpiece_feed_speed_m_per_s, "depth_of_cut_m": g.depth_of_cut_m, "grinding_width_m": p.grinding_width_m, "wheel_diameter_m": g.wheel_diameter_m},
                "calibration": {"specific_grinding_energy_J_per_m3": p.specific_grinding_energy_j_per_m3, "normal_to_tangential_force_ratio": p.normal_to_tangential_force_ratio, "calibration_id": p.calibration_id, "specific_grinding_energy_source": source, "force_ratio_source": source, "applicability_notes": p.applicability_notes or source},
            },
            "force_mapping": {"boundary": "ideal_top_surface", "distribution": "uniform_over_effective_projected_interval", "tangential_force_direction": g.relative_feed_direction},
        }

    def _linear_single(self, g, mesh, material, process, settings):
        return EvolvedFemCase.from_mapping({
            "evolved_fem_schema_version": 1, "unit_system": "SI",
            "model_type": "single_pass_evolved_surface_empirical_load_plane_stress",
            "pass_load": self._pass_load(g, process, settings),
            "material": {"E": material.elastic_modulus_pa, "nu": material.poisson_ratio},
            "mesh": {"target_size": mesh.target_size_m},
            "analysis": {"type": "plane_stress", "thickness": process.grinding_width_m},
            "boundaries": [{"side": "bottom", "role": "fixed"}, {"side": "top", "role": "contact"}, {"side": "left", "role": "free"}, {"side": "right", "role": "free"}],
        })

    def _linear_pass(self, g, mesh, material, process, settings):
        return PassScanCase.from_mapping({"scan_schema_version": 1, "unit_system": "SI", "model_type": "single_pass_quasi_static_evolved_fem_scan", "reference_case": self._linear_single(g, mesh, material, process, settings).to_dict(), "scan": {"range": "complete_single_pass", "base_position_count": settings.scan_base_position_count}})

    def _plastic_single(self, g, mesh, material, process, settings):
        return SinglePositionElastoplasticCase.from_mapping({
            "single_position_elastoplastic_schema_version": 1, "unit_system": "SI", "model_type": "single_position_evolved_surface_elastoplastic_loading_unloading",
            "pass_load": self._pass_load(g, process, settings),
            "material": {"E": material.elastic_modulus_pa, "nu": material.poisson_ratio, "yield_strength": material.yield_strength_pa, "tangent_modulus": material.tangent_modulus_pa},
            "mesh": {"target_size": mesh.target_size_m}, "analysis": {"type": "plane_strain", "thickness": process.grinding_width_m},
            "fixed_boundary": {"side": "bottom", "components": ["x", "y"]},
            "increments": {"loading_step_count": settings.loading_step_count, "unloading_step_count": settings.unloading_step_count},
            "newton": settings.newton.to_dict(), "step_control": settings.step_control.to_dict(),
        })

    def _plastic_pass(self, g, mesh, material, process, settings):
        reference = self._plastic_single(g, mesh, material, process, settings).to_dict()
        return FixedMeshElastoplasticHistoryPassCase.from_mapping({
            "history_pass_schema_version": 1, "unit_system": "SI", "model_type": "fixed_mesh_complete_single_pass_elastoplastic_history",
            "moving_load_pass": {"moving_load_pass_schema_version": 1, "unit_system": "SI", "model_type": "fixed_mesh_complete_single_pass_moving_load", "reference_case": reference, "pass": {"range": "complete_single_pass", "base_position_count": settings.scan_base_position_count, "retain_key_position_output": True}},
            "transition": settings.history_transition.to_dict(),
            "snapshot_retention": {"retain_key_positions": True, "retain_maximum_plastic_state": True},
        })

    def _mechanism_pass(self, g, mesh, material, wheel, process, settings):
        baseline = self._plastic_pass(g, mesh, material, process, settings)
        statistical = self._statistical_grain_load(
            baseline.reference_case.pass_load.force_model.to_dict(), wheel, settings
        )
        return MechanismHistoryPassCase.from_mapping({
            "mechanism_history_pass_schema_version": 1, "unit_system": "SI", "model_type": "fixed_mesh_complete_single_pass_statistical_mechanism_elastoplastic_history",
            "baseline_history_pass": baseline.to_dict(),
            "statistical_grain_load": statistical.to_dict(),
            "comparison": {"mode": "uniform_vs_statistical_mechanism", "require_identical_total_force_history": True, "retain_both_final_fields": True},
        })

    @staticmethod
    def _single_grain(material, settings):
        selected = settings.single_grain
        center_x = 0.5 * selected.workpiece_width_m
        if selected.grain_type == "rounded_wedge":
            initial_y = selected.workpiece_height_m + selected.initial_clearance_m
            indentation_y = selected.workpiece_height_m - selected.indentation_depth_m
        else:
            vertical_radius = selected.grain_radius_m
            initial_y = (
                selected.workpiece_height_m
                + vertical_radius
                + selected.initial_clearance_m
            )
            indentation_y = (
                selected.workpiece_height_m
                + vertical_radius
                - selected.indentation_depth_m
            )
        is_v2 = selected.grain_type in {"rounded_circle", "rounded_wedge"}
        schema_version = 3 if selected.damage_separation_enabled else (2 if is_v2 else 1)
        coordinate_fields = ("reference_x_m", "reference_y_m") if is_v2 else ("center_x_m", "center_y_m")
        targets = [
            {"segment": "initial", coordinate_fields[0]: center_x, coordinate_fields[1]: initial_y},
            {"segment": "indentation", coordinate_fields[0]: center_x, coordinate_fields[1]: indentation_y},
        ]
        if selected.scratch_distance_m > 0.0:
            targets.append(
                {
                    "segment": "scratch",
                    coordinate_fields[0]: center_x + selected.scratch_distance_m,
                    coordinate_fields[1]: indentation_y,
                }
            )
        targets.append(
            {
                "segment": "unloading",
                coordinate_fields[0]: center_x + selected.scratch_distance_m,
                coordinate_fields[1]: initial_y,
            }
        )
        return SingleGrainContactCase.from_mapping(
            {
                "single_grain_contact_schema_version": schema_version,
                "unit_system": "SI",
                "model_type": "single_grain_real_contact",
                "geometry": {"type": "rectangle", "width": selected.workpiece_width_m, "height": selected.workpiece_height_m},
                "mesh": {"target_size": selected.mesh_target_size_m},
                "analysis": {"type": "plane_strain", "thickness": selected.workpiece_thickness_m},
                "fixed_boundary": {"side": "bottom", "components": ["x", "y"]},
                "material": {
                    "E": material.elastic_modulus_pa,
                    "nu": material.poisson_ratio,
                    "yield_strength": material.yield_strength_pa,
                    "tangent_modulus": material.tangent_modulus_pa,
                    "parameter_source": material.parameter_source,
                    "material_state": material.label,
                    "calibration_status": material.calibration_status,
                    **({"density_kg_per_m3": selected.density_kg_per_m3} if schema_version == 3 else {}),
                },
                "grain": ({
                    "type": selected.grain_type,
                    "radius_m": selected.grain_radius_m,
                    "initial_reference_x_m": center_x,
                    "initial_reference_y_m": initial_y,
                } if selected.grain_type == "rounded_circle" else {
                    "type": "rounded_wedge",
                    "tip_radius_m": selected.grain_tip_radius_m,
                    "wedge_height_m": selected.grain_wedge_height_m,
                    "intrinsic_rake_angle_rad": selected.intrinsic_rake_angle_rad,
                    "intrinsic_clearance_angle_rad": selected.intrinsic_clearance_angle_rad,
                    "pose_angle_rad": selected.grain_pose_angle_rad,
                    "initial_reference_x_m": center_x,
                    "initial_reference_y_m": initial_y,
                }) if is_v2 else {
                    "type": selected.grain_type,
                    "radius_m": selected.grain_radius_m,
                    "initial_center_x_m": center_x,
                    "initial_center_y_m": initial_y,
                    **(
                        {
                            "start_angle_rad": selected.grain_arc_start_angle_rad,
                            "end_angle_rad": selected.grain_arc_end_angle_rad,
                        }
                        if selected.grain_type == "rigid_circular_arc"
                        else {}
                    ),
                },
                "trajectory": ({"nominal_scratch_direction": selected.nominal_scratch_direction, "targets": targets} if is_v2 else {"targets": targets}),
                **({"material_topology": {"preset_removal": (
                    {"mode": "element_ids", "element_ids": list(selected.preset_removal_element_ids)}
                    if selected.preset_removal_mode == "element_ids"
                    else {"mode": "geometric_region", "region": {"type": "polygon", "vertices_m": [list(point) for point in selected.preset_removal_polygon_vertices_m], "selection_rule": "centroid_inside_or_on_boundary"}}
                )}} if is_v2 else {}),
                **({"damage": {
                    "model": "ductile_triaxiality_table",
                    "failure_strain_table": [list(row) for row in selected.damage_failure_strain_table],
                    "triaxiality_cutoff": selected.damage_triaxiality_cutoff,
                    "evolution": selected.damage_evolution,
                    "fracture_energy_J_per_m2": selected.damage_fracture_energy_J_per_m2,
                    "characteristic_length": "equivalent_circle_diameter_reference_area",
                    "damage_event_tolerance": selected.damage_event_tolerance,
                    "maximum_separation_events_per_position": selected.damage_maximum_separation_events_per_position,
                    "area_absolute_tolerance_m2": selected.damage_area_absolute_tolerance_m2,
                    "parameter_source": selected.damage_parameter_source,
                    "calibration_status": selected.damage_calibration_status,
                    "applicability_notes": selected.damage_applicability_notes,
                }} if schema_version == 3 else {}),
                **({"transient": {
                    "integration_method": "implicit_midpoint",
                    "grain_speed_m_per_s": selected.grain_speed_m_per_s,
                    "initial_time_step_s": selected.transient_initial_time_step_s,
                    "minimum_time_step_s": selected.transient_minimum_time_step_s,
                    "maximum_steps": selected.transient_maximum_steps,
                }} if schema_version == 3 else {}),
                "contact": {
                    "normal_algorithm": selected.normal_algorithm,
                    "normal_penalty_factor": selected.normal_penalty_factor,
                    "tangential_penalty_ratio": selected.tangential_penalty_ratio,
                    "augmented_relaxation": selected.augmented_relaxation,
                    "augmented_maximum_iterations": selected.augmented_maximum_iterations,
                    "friction_coefficient": selected.friction_coefficient,
                    "friction_regularization_ratio": selected.friction_regularization_ratio,
                },
                "newton": settings.newton.to_dict(),
                "step_control": {"allow_reduction": True, "minimum_substep_fraction": 1.0 / 1024.0, "maximum_retries": 10},
                "output": {"save_field_evolution": False, "snapshot_stride": 1},
                "safety": {"maximum_degrees_of_freedom": 50000, "warn_displacement_over_local_size": 0.10, "stop_displacement_over_local_size": 0.50, "warn_displacement_over_radius": 0.05, "stop_displacement_over_radius": 0.20, "warn_strain": 0.05, "stop_strain": 0.15},
            }
        )

    @staticmethod
    def _statistical_grain_load(force_model, wheel, settings) -> StatisticalGrainLoadCase:
        resolved = wheel.resolved_specification
        if wheel.resolution_path == "manufacturer_override":
            wheel_spec = {
                "resolution_path": "manufacturer_override",
                "abrasive_material": wheel.abrasive_material,
                "grit_designation": wheel.grit_designation,
                "manufacturer": resolved["manufacturer"],
                "product_model": resolved["product_model"],
                "diameter_lower_m": resolved["diameter_lower_m"],
                "representative_diameter_d50_m": resolved["representative_diameter_d50_m"],
                "diameter_upper_m": resolved["diameter_upper_m"],
                "range_definition": resolved["range_definition"],
                "data_status": resolved["data_status"],
                "source": resolved["source"],
            }
        else:
            wheel_spec = {
                "resolution_path": wheel.resolution_path,
                "abrasive_material": wheel.abrasive_material,
                "grit_designation": wheel.grit_designation,
            }
        return StatisticalGrainLoadCase.from_dict({
            "statistical_grain_load_schema_version": 1,
            "unit_system": "SI",
            "model_type": "statistical_equivalent_grain_mechanism_load",
            "mechanism_force": {
                "mechanism_force_schema_version": 1,
                "unit_system": "SI",
                "model_type": "ductile_metal_three_mechanism_force_decomposition",
                "force_model": force_model,
                "mechanism_model": {
                    "parameter_source": "builtin_preset",
                    "preset_id": "ductile_alloy_steel_demo",
                },
            },
            "wheel_specification": wheel_spec,
            "grain_population": settings.grain_population.to_dict(),
            "load_distribution": settings.load_distribution.to_dict(),
        })


def build_process_grain_preview_case(
    project: GrindCaeProject, settings: AnalysisSettings | None = None
) -> StatisticalGrainLoadCase:
    """Build the same statistical-load input consumed by mechanism analysis."""
    selected = settings or AnalysisSettings.recommended()
    geometry, _mesh, material, wheel, process = AnalysisInputBuilder._validated_sources(project)
    if material.family != "ductile_metal" or wheel.abrasive_material != "aluminum_oxide":
        raise ValueError("统计磨粒预览当前只支持刚玉砂轮和延性金属。")
    force_model = AnalysisInputBuilder._pass_load(
        geometry, process, selected
    )["force_model"]
    return AnalysisInputBuilder._statistical_grain_load(force_model, wheel, selected)
