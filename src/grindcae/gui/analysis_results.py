"""Strict minimal result adapters for the five workbench analysis modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
from grindcae.single_grain_contact.exporters import read_contact_summary
from grindcae.single_grain_contact.exporters_v2 import read_contact_summary_v2
from grindcae.single_grain_contact.exporters_v3 import read_contact_summary_v3

from .analysis import analysis_mode_definition


class AnalysisResultError(ValueError):
    """Raised when a published solver result cannot be trusted by the GUI."""


@dataclass(frozen=True, slots=True)
class AnalysisPublishedResult:
    mode: str
    display_name: str
    result_format: str
    output_directory: Path
    summary_path: Path
    representative_image: Path
    summary: dict[str, Any]
    summary_text: str


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisResultError(f"结果读取失败：{field} 必须是对象。")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisResultError(f"结果读取失败：{field} 必须是有限数值。")
    number = float(value)
    if not math.isfinite(number):
        raise AnalysisResultError(f"结果读取失败：{field} 包含非有限数值。")
    return number


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise AnalysisResultError(f"结果读取失败：{field} 必须是非负整数。")
    return value


def _force_is_zero(value: object, field: str) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        raise AnalysisResultError(f"结果读取失败：{field} 必须包含两个分量。")
    return all(abs(_finite(item, f"{field}[{index}]")) <= 1.0e-10 for index, item in enumerate(value))


def _artifact_path(summary: Mapping[str, Any], filename: str, output: Path) -> Path:
    artifacts = _mapping(summary.get("artifacts"), "artifacts")
    candidates: list[Path] = []
    for value in artifacts.values():
        raw = value.get("path") if isinstance(value, Mapping) else value
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = output / candidate
            if candidate.name == filename:
                candidates.append(candidate.resolve())
    if len(candidates) != 1:
        raise AnalysisResultError(f"结果读取失败：摘要未唯一登记代表图 {filename}。")
    image = candidates[0]
    if not image.is_file() or image.stat().st_size <= 0:
        raise AnalysisResultError(f"结果读取失败：代表图缺失或为空：{image}")
    try:
        if image.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("PNG signature")
        pixels = mpimg.imread(image)
        if pixels.size <= 0 or pixels.ndim < 2:
            raise ValueError("empty pixels")
    except (OSError, ValueError, SyntaxError) as exc:
        raise AnalysisResultError(f"结果读取失败：代表图 PNG 无法解码：{image.name}") from exc
    return image


def _linear_single(summary: Mapping[str, Any]) -> str:
    load = _mapping(_mapping(summary.get("pass_load"), "pass_load").get("current_empirical_force"), "current_empirical_force")
    displacement = _mapping(summary.get("maximum_displacement"), "maximum_displacement")
    stress = _mapping(summary.get("stress_statistics"), "stress_statistics")
    residual = _mapping(_mapping(summary.get("force_tracking"), "force_tracking").get("force_balance_residual"), "force_balance_residual")
    return "\n".join((
        "任务：线弹性·单位置（二维平面应力）",
        f"当前磨削力：Ft={_finite(load.get('tangential_force_magnitude_N'), 'Ft'):.6g} N，Fn={_finite(load.get('normal_force_magnitude_N'), 'Fn'):.6g} N",
        f"最大位移：{_finite(displacement.get('magnitude_m'), 'maximum_displacement') * 1e6:.6g} μm",
        f"面积加权等效应力：p95={_finite(stress.get('von_mises_area_weighted_p95_Pa'), 'p95') * 1e-6:.6g} MPa，p99={_finite(stress.get('von_mises_area_weighted_p99_Pa'), 'p99') * 1e-6:.6g} MPa",
        f"力平衡残差：{_finite(residual.get('norm_N'), 'force_balance_residual'):.3e} N",
        "只使用弹性模量和泊松比，不包含屈服和残余塑性。",
    ))


def _linear_pass(summary: Mapping[str, Any]) -> str:
    counts = _mapping(summary.get("scan_counts"), "scan_counts")
    force = _mapping(_mapping(summary.get("force_extrema"), "force_extrema").get("maximum_current_resultant_force"), "maximum_current_resultant_force")
    response = _mapping(summary.get("response_extrema"), "response_extrema")
    displacement = _mapping(response.get("maximum_displacement"), "maximum_displacement")
    stress = _mapping(response.get("maximum_von_mises_p99"), "maximum_von_mises_p99")
    validations = _mapping(summary.get("validations"), "validations")
    return "\n".join((
        "任务：线弹性·单程（独立准静态位置）",
        f"扫描位置数：{_integer(counts.get('total_positions'), 'total_positions')}",
        f"最大磨削力位置：{_finite(force.get('wheel_lowest_point_x_m'), 'maximum_force_position') * 1e3:.6g} mm",
        f"最大位移：{_finite(displacement.get('value'), 'maximum_displacement') * 1e6:.6g} μm",
        f"最大面积加权等效应力 p99：{_finite(stress.get('value'), 'maximum_von_mises_p99') * 1e-6:.6g} MPa",
        f"最大力平衡残差：{_finite(validations.get('maximum_force_balance_residual_norm_N'), 'maximum_force_balance_residual'):.3e} N",
        "每个位置相互独立，不传递塑性历史。",
    ))


def _plastic_single(summary: Mapping[str, Any]) -> str:
    snapshots = _mapping(summary.get("snapshots"), "snapshots")
    peak = _mapping(_mapping(snapshots.get("peak_loaded"), "peak_loaded").get("statistics"), "peak_loaded.statistics")
    residual_state = _mapping(snapshots.get("residual_unloaded"), "residual_unloaded")
    residual = _mapping(residual_state.get("statistics"), "residual_unloaded.statistics")
    zero = _force_is_zero(residual_state.get("external_force_N"), "residual_unloaded.external_force_N")
    newton = _mapping(summary.get("newton_history"), "newton_history")
    return "\n".join((
        "任务：弹塑性·单位置（小应变平面应变 J2）",
        f"峰值最大位移：{_finite(peak.get('maximum_displacement_m'), 'peak.maximum_displacement') * 1e6:.6g} μm",
        f"最大等效塑性应变：{_finite(peak.get('maximum_equivalent_plastic_strain'), 'peak.maximum_plastic_strain'):.6g}",
        f"塑性单元：{_integer(peak.get('plastic_element_count'), 'peak.plastic_element_count')}（{_finite(peak.get('plastic_element_fraction'), 'peak.plastic_element_fraction'):.2%}）",
        f"最终残余位移：{_finite(residual.get('maximum_displacement_m'), 'residual.maximum_displacement') * 1e6:.6g} μm",
        f"最终残余 von Mises 应力 p99：{_finite(residual.get('von_mises_area_weighted_p99_Pa'), 'residual.von_mises_p99') * 1e-6:.6g} MPa",
        f"最终外载荷为零：{'是' if zero else '否'}；最大 Newton 迭代：{_integer(newton.get('maximum_iterations_used'), 'maximum_iterations_used')}",
    ))


def _plastic_pass(summary: Mapping[str, Any]) -> str:
    pass_data = _mapping(summary.get("pass"), "pass")
    maximum = _mapping(summary.get("maximum_response"), "maximum_response")
    final = _mapping(summary.get("final_unloaded"), "final_unloaded")
    zero = _force_is_zero(final.get("external_force_N"), "final_unloaded.external_force_N")
    return "\n".join((
        "任务：弹塑性·单程（固定网格塑性历史传递）",
        f"单程位置数：{_integer(pass_data.get('target_position_count'), 'target_position_count')}",
        f"最大等效塑性应变：{_finite(maximum.get('maximum_equivalent_plastic_strain'), 'maximum_equivalent_plastic_strain'):.6g}",
        f"塑性单元：{_integer(final.get('plastic_element_count'), 'plastic_element_count')}（{_finite(final.get('plastic_element_fraction'), 'plastic_element_fraction'):.2%}）",
        f"最终残余位移：{_finite(final.get('maximum_residual_displacement_m'), 'maximum_residual_displacement') * 1e6:.6g} μm",
        f"最终残余应力 p99：{_finite(final.get('von_mises_area_weighted_p99_Pa'), 'residual_von_mises_p99') * 1e-6:.6g} MPa",
        f"最终外载荷为零：{'是' if zero else '否'}；最大 Newton 迭代：{_integer(pass_data.get('maximum_newton_iterations'), 'maximum_newton_iterations')}",
    ))


def _mechanism_pass(summary: Mapping[str, Any]) -> str:
    prediction = _mapping(summary.get("phase7a2_prediction"), "phase7a2_prediction")
    grit = _mapping(prediction.get("resolved_grit_specification"), "resolved_grit_specification")
    population = _mapping(prediction.get("grain_population"), "grain_population")
    comparison = _mapping(summary.get("comparison"), "comparison")
    final = _mapping(summary.get("mechanism_final_unloaded"), "mechanism_final_unloaded")
    resolution = _mapping(summary.get("resolution"), "resolution")
    if resolution.get("resolution_status") != "macro_scale_conservative_projection_not_grain_resolved":
        raise AnalysisResultError("结果读取失败：机制化结果的宏观投影标识无效。")
    zero = _force_is_zero(final.get("external_force_N"), "mechanism_final_unloaded.external_force_N")
    identical = comparison.get("total_force_history_identical")
    if type(identical) is not bool:
        raise AnalysisResultError("结果读取失败：总力历史一致性必须为布尔值。")
    return "\n".join((
        "任务：机制化磨削·单程（宏观统计保守投影）",
        f"粒度路径：{grit.get('resolution_path')}；代表粒径：{_finite(grit.get('representative_diameter_d50_m'), 'representative_diameter_d50_m') * 1e6:.6g} μm",
        f"等效磨粒组数：{_integer(population.get('equivalent_group_count'), 'equivalent_group_count')}",
        f"基准与机制路线总力历史一致：{'是' if identical else '否'}",
        f"最大节点载荷分布差：{_finite(comparison.get('maximum_target_nodal_load_vector_l2_difference_N'), 'maximum_target_nodal_load_difference'):.6g} N",
        f"机制路线最大等效塑性应变：{_finite(final.get('maximum_equivalent_plastic_strain'), 'maximum_equivalent_plastic_strain'):.6g}",
        f"最终残余位移：{_finite(final.get('maximum_residual_displacement_m'), 'maximum_residual_displacement') * 1e6:.6g} μm；最终残余应力：{_finite(final.get('maximum_residual_von_mises_stress_Pa'), 'maximum_residual_von_mises_stress') * 1e-6:.6g} MPa",
        f"最终外载荷为零：{'是' if zero else '否'}；该结果不代表逐颗磨粒高保真解析或真实粗糙度。",
    ))


def _single_grain(summary: Mapping[str, Any]) -> str:
    if summary.get("result_format") == "grindcae_single_grain_damage_separation_v3":
        topology = _mapping(summary.get("topology"), "topology")
        separation = _mapping(summary.get("separation"), "separation")
        damage = _mapping(summary.get("damage_model"), "damage_model")
        energy = _mapping(summary.get("energy"), "energy")
        scope = _mapping(summary.get("scope"), "scope")
        if not (
            summary.get("automatic_damage_evolution") is True
            and summary.get("automatic_material_separation") is True
            and summary.get("chip_formation_prediction") is False
            and scope.get("free_chip_motion") is False
        ):
            raise AnalysisResultError("结果读取失败：Schema 3 损伤分离范围声明无效。")
        interpretation = summary.get("material_removal_interpretation")
        interpretation_text = {
            "model_predicted_unvalidated": "模型预测、尚未独立验证",
            "calibrated": "已标定模型预测、尚未独立验证",
            "independently_validated": "已登记独立验证",
        }.get(interpretation)
        if interpretation_text is None:
            raise AnalysisResultError("结果读取失败：Schema 3 证据状态无效。")
        return "\n".join((
            "任务：单磨粒损伤演化与材料分离（二维小应变固定背景网格）",
            f"自动分离事件：{_integer(separation.get('event_count'), 'event_count')}；自动移除参考面积：{_finite(separation.get('automatic_removed_area_m2'), 'automatic_removed_area_m2') * 1e12:.6g} μm²",
            f"最终有效/移除单元：{_integer(topology.get('active_element_count'), 'active_element_count')}/{_integer(topology.get('removed_element_count'), 'removed_element_count')}；有效自由度：{_integer(topology.get('active_degrees_of_freedom'), 'active_degrees_of_freedom')}；自由表面边：{_integer(topology.get('free_surface_edge_count'), 'free_surface_edge_count')}",
            f"损伤耗散：{_finite(energy.get('damage_dissipation_J'), 'damage_dissipation_J'):.6g} J；分离事件能量：{_finite(energy.get('separation_event_energy_J'), 'separation_event_energy_J'):.6g} J；总能量残差：{_finite(energy.get('balance_residual_J'), 'balance_residual_J'):.6g} J",
            f"参数来源：{damage.get('parameter_source')}；标定状态：{damage.get('calibration_status')}；证据状态：{interpretation_text}。",
            "当前结果表示模型预测的单磨粒材料分离和加工表面；不预测游离切屑，不替代真实磨削力标定或独立物理验证。",
        ))
    peak = _mapping(summary.get("peak"), "peak")
    final = _mapping(summary.get("final_state"), "final_state")
    energy = _mapping(summary.get("energy"), "energy")
    scope = _mapping(summary.get("scope"), "scope")
    applicability = _mapping(summary.get("applicability"), "applicability")
    if scope.get("physical_material_removal") is not False:
        raise AnalysisResultError("结果读取失败：单磨粒结果的材料去除边界无效。")
    warning_metrics = applicability.get("warning_metrics")
    if not isinstance(warning_metrics, list) or not all(
        isinstance(item, str) for item in warning_metrics
    ):
        raise AnalysisResultError("结果读取失败：单磨粒适用性警告字段无效。")
    if applicability.get("hard_stop_reasons") not in ([], ()):
        raise AnalysisResultError("结果读取失败：单磨粒正式结果包含硬停止状态。")
    status = summary.get("result_status")
    if status not in {"completed", "completed_with_applicability_warning"}:
        raise AnalysisResultError("结果读取失败：单磨粒结果状态无效。")
    labels = {
        "maximum_displacement_to_contact_size": "位移/局部接触尺寸",
        "maximum_displacement_to_grain_radius": "位移/磨粒半径",
        "maximum_displacement_gradient": "位移梯度",
        "maximum_absolute_principal_strain": "最大主小应变",
        "maximum_local_rotation_rad": "局部转动",
        "minimum_area_ratio": "单元面积比",
        "minimum_normalized_jacobian_ratio": "归一化 Jacobian",
    }
    applicability_text = (
        "适用性警告：" + "、".join(labels.get(item, item) for item in warning_metrics)
        if status == "completed_with_applicability_warning"
        else "小应变适用性诊断：在首版工程建议范围内"
    )
    lines = [
        "任务：单磨粒真实接触（二维小应变固定网格）",
        f"峰值总法向反力：{_finite(peak.get('normal_reaction_N'), 'normal_reaction_N'):.6g} N；单位厚度：{_finite(peak.get('normal_reaction_per_thickness_N_per_m'), 'normal_reaction_per_thickness_N_per_m'):.6g} N/m",
        f"峰值总切向反力：{_finite(peak.get('tangential_reaction_N'), 'tangential_reaction_N'):.6g} N；最大接触压力：{_finite(peak.get('maximum_pressure_Pa'), 'maximum_pressure_Pa') * 1e-6:.6g} MPa",
        f"最大数值穿透：{_finite(peak.get('maximum_penetration_m'), 'maximum_penetration_m') * 1e6:.6g} μm；最大平衡残差：{_finite(summary.get('maximum_balance_residual_N'), 'maximum_balance_residual_N'):.3e} N",
        f"最终接触点数：{_integer(final.get('contact_count'), 'final.contact_count')}；摩擦耗散：{_finite(energy.get('friction_dissipation_J'), 'friction_dissipation_J'):.6g} J；塑性耗散：{_finite(energy.get('plastic_dissipation_J'), 'plastic_dissipation_J'):.6g} J",
        f"能量平衡残差：{_finite(energy.get('energy_balance_residual_J'), 'energy_balance_residual_J'):.6g} J；相对残差：{_finite(energy.get('energy_balance_relative_residual'), 'energy_balance_relative_residual'):.3e}",
        f"{applicability_text}；最大位移梯度：{_finite(applicability.get('maximum_displacement_gradient'), 'maximum_displacement_gradient'):.6g}；最大主小应变：{_finite(applicability.get('maximum_absolute_principal_strain'), 'maximum_absolute_principal_strain'):.6g}",
        "残余沟槽仅表示卸载后固定网格表面位移，不代表真实材料去除、切屑形成或独立实验验证。",
    ]
    if summary.get("result_format") == "grindcae_single_grain_real_contact_v2":
        topology = _mapping(summary.get("topology"), "topology")
        geometry = _mapping(summary.get("grain_geometry"), "grain_geometry")
        lines[0] = "任务：单磨粒真实接触（二维小应变有效材料网格）"
        lines.insert(1, f"磨粒形貌：{geometry.get('type')}；预设去除单元：{_integer(topology.get('removed_element_count'), 'removed_element_count')}；有效单元：{_integer(topology.get('active_element_count'), 'active_element_count')}")
        lines.insert(2, f"有效节点/自由度：{_integer(topology.get('active_node_count'), 'active_node_count')}/{_integer(topology.get('active_degrees_of_freedom'), 'active_degrees_of_freedom')}；自由表面边：{_integer(topology.get('free_surface_edge_count'), 'free_surface_edge_count')}；长度：{_finite(topology.get('free_surface_length_m'), 'free_surface_length_m') * 1e6:.6g} μm")
        lines[-1] = "预设 REMOVED 用于验证有效网格、自由表面和接触候选更新；当前不包含自动损伤、自动材料分离或真实材料去除预测。"
    return "\n".join(lines)


_SUMMARY_READERS = {
    "linear_elastic_single_position": _linear_single,
    "linear_elastic_single_pass": _linear_pass,
    "elastoplastic_single_position": _plastic_single,
    "elastoplastic_single_pass": _plastic_pass,
    "mechanism_elastoplastic_single_pass": _mechanism_pass,
    "single_grain_high_fidelity": _single_grain,
}


class AnalysisResultAdapter:
    def read(self, mode: str, output_directory: str | Path) -> AnalysisPublishedResult:
        if mode == "literature_elastoplastic_single_pass":
            from .literature_integration import read_history_published
            return read_history_published(output_directory)
        if mode == "literature_grinding_force":
            from .literature_integration import read_published
            return read_published(output_directory)
        definition = analysis_mode_definition(mode, require_enabled=True)
        output = Path(output_directory).expanduser().resolve()
        summary_path = output / definition.summary_filename
        try:
            if mode == "single_grain_high_fidelity":
                raw = json.loads(summary_path.read_text(encoding="utf-8"))
                if raw.get("result_format") == "grindcae_single_grain_real_contact_v2":
                    summary = read_contact_summary_v2(summary_path)
                elif raw.get("result_format") == "grindcae_single_grain_damage_separation_v3":
                    summary = read_contact_summary_v3(summary_path)
                else:
                    summary = read_contact_summary(summary_path)
            else:
                raw = summary_path.read_text(encoding="utf-8")
                summary = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise AnalysisResultError(f"结果读取失败：无法读取摘要 {definition.summary_filename}：{exc}") from exc
        if not isinstance(summary, dict):
            raise AnalysisResultError("结果读取失败：摘要根对象无效。")
        valid_formats = {definition.result_format}
        if mode == "single_grain_high_fidelity":
            valid_formats.add("grindcae_single_grain_real_contact_v2")
            valid_formats.add("grindcae_single_grain_damage_separation_v3")
        if summary.get("result_format") not in valid_formats:
            raise AnalysisResultError(f"结果读取失败：{definition.display_name} result_format 无效。")
        image = _artifact_path(summary, definition.representative_image_filename, output)
        text = _SUMMARY_READERS[mode](summary)
        return AnalysisPublishedResult(mode, definition.display_name, str(summary["result_format"]), output, summary_path, image, summary, text)
