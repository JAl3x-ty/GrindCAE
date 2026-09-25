"""Read published strict analysis summaries for GUI presentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .form import MECHANISM_HISTORY_MODE, GuiInputError, SCAN_MODE, SINGLE_MODE
from .labels import calibration_set_display_text


LIMITATION_TEXT = (
    "当前结果来自单程独立准静态、二维平面应力、线弹性和经验磨削力模型。\n"
    "不包含真实接触、温度、塑性、粗糙度、砂轮磨损、多行程和工业验证。"
)
MECHANISM_HISTORY_LIMITATION_TEXT = (
    "当前结果来自固定网格、二维平面应变 J2 弹塑性完整单程历史和统计等效磨粒群投影。\n"
    "不包含真实接触、温度、热耦合、单磨粒高保真解析、真实粗糙度、多道次和工业验证。"
)

SINGLE_PNG_KEYS = (
    "mesh_png",
    "displacement_magnitude_png",
    "von_mises_stress_png",
    "principal_stress_1_in_plane_png",
    "deformed_shape_png",
    "stress_contour_png",
    "strain_contour_png",
)
PNG_EXPLANATIONS = {
    "mesh.png": "显示完整计算网格、边界条件和实际受载接触圆弧。",
    "displacement_magnitude.png": "颜色表示位移幅值，单位为 um。",
    "von_mises_stress.png": "单元等效应力为未平滑结果，局部峰值具有网格敏感性。",
    "principal_stress_1_in_plane.png": "显示第一面内主应力，正值偏拉、负值偏压。",
    "deformed_shape.png": "变形比例仅用于显示，真实位移以摘要、JSON 和 CSV 为准。",
    "stress_contour.png": "原生 P1 三角形单元等效应力分块云图，单位 MPa；未做节点平均或平滑。",
    "strain_contour.png": "原生 P1 三角形单元第一面内主小应变分块云图，单位 με；未做节点平均或平滑。",
    "scan_overview.png": "从加工进度 0% 到 100% 显示单程扫描变化。",
}
MECHANISM_PNG_EXPLANATION = (
    "对比统一载荷基准路线与统计机制载荷路线的载荷分布、总力历史、"
    "塑性历史和最终卸载场。"
)

ROLE_LABELS = {
    "representative_entry": "代表性入口",
    "representative_full_contact": "代表性满接触",
    "maximum_displacement": "最大位移",
    "maximum_von_mises_p95": "最大 p95",
    "maximum_von_mises_p99": "最大 p99",
    "representative_exit": "代表性出口",
}


def limitation_text(mode: str) -> str:
    if mode == MECHANISM_HISTORY_MODE:
        return MECHANISM_HISTORY_LIMITATION_TEXT
    return LIMITATION_TEXT


@dataclass(frozen=True, slots=True)
class GuiImage:
    label: str
    path: Path
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class GuiPublishedResult:
    mode: str
    output_directory: Path
    summary_path: Path
    summary: dict[str, object]
    summary_text: str
    images: tuple[GuiImage, ...]


def _read_summary(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="ascii") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuiInputError(f"无法读取已发布结果摘要 {path.name}：{exc}") from exc
    if not isinstance(value, dict):
        raise GuiInputError(f"结果摘要 {path.name} 的根对象无效。")
    return value


def _single_summary(summary: dict[str, object]) -> str:
    load = summary["pass_load"]
    force = load["current_empirical_force"]
    tracking = summary["force_tracking"]
    reaction = tracking["fixed_support_reaction"]
    residual = tracking["force_balance_residual"]
    stress = summary["stress_statistics"]
    calibration = load["calibration_provenance"]
    tangential = float(force["tangential_force_magnitude_N"])
    normal = float(force["normal_force_magnitude_N"])
    fx = float(force["signed_Fx_N"])
    fy = float(force["signed_Fy_N"])
    resultant = float(force["resultant_force_magnitude_N"])
    return "\n".join(
        (
            f"加工状态：{load['pass_state']}    接触比：{float(load['contact_ratio']):.6g}",
            "当前等效磨削力（经验估算）："
            f"Ft={tangential:.6g} N，Fn={normal:.6g} N，Fr={resultant:.6g} N",
            f"全局分量：Fx={fx:.6g} N，Fy={fy:.6g} N（Fy<0 表示压向工件）",
            f"最大位移：{float(summary['maximum_displacement']['magnitude_m']) * 1.0e6:.6g} um",
            "面积加权等效应力："
            f"p95={float(stress['von_mises_area_weighted_p95_Pa']) * 1.0e-6:.6g} MPa，"
            f"p99={float(stress['von_mises_area_weighted_p99_Pa']) * 1.0e-6:.6g} MPa",
            f"固定端反力：Rx={float(reaction['Rx_N']):.6g} N，Ry={float(reaction['Ry_N']):.6g} N",
            "力平衡残差："
            f"x={float(residual['x_N']):.3e} N，y={float(residual['y_N']):.3e} N，"
            f"norm={float(residual['norm_N']):.3e} N",
            f"经验参数集：{calibration_set_display_text(calibration['calibration_id'])}",
            "",
            LIMITATION_TEXT,
        )
    )


def _scan_summary(summary: dict[str, object]) -> str:
    counts = summary["scan_counts"]
    states = counts["states"]
    plateau = summary["steady_full_contact_empirical_force"]
    extrema = summary["response_extrema"]
    maximum_force = summary["force_extrema"]["maximum_current_resultant_force"]

    def response(role: str, multiplier: float, unit: str) -> str:
        item = extrema[role]
        return (
            f"{float(item['value']) * multiplier:.6g} {unit}，"
            f"x={float(item['wheel_lowest_point_x_m']) * 1.0e3:.6g} mm，"
            f"scan_index={int(item['scan_index'])}"
        )

    state_text = "，".join(f"{name}={int(states[name])}" for name in (
        "before_entry", "entry", "full_contact", "exit", "after_exit"
    ))
    fx_magnitude = float(plateau["Fx_magnitude_N"])
    fy = float(plateau["Fy_N"])
    return "\n".join(
        (
            f"实际扫描位置：{int(counts['total_positions'])}    唯一快照目录：{int(counts['unique_snapshot_directories'])}",
            f"状态计数：{state_text}",
            "满接触稳态参考力（经验估算）："
            f"切向力幅值 |Fx|={fx_magnitude:.6g} N，全局分量 Fy={fy:.6g} N",
            f"最大位移：{response('maximum_displacement', 1.0e6, 'um')}",
            f"最大面积加权等效应力 p95：{response('maximum_von_mises_p95', 1.0e-6, 'MPa')}",
            f"最大面积加权等效应力 p99：{response('maximum_von_mises_p99', 1.0e-6, 'MPa')}",
            "最大力平衡残差："
            f"{float(summary['validations']['maximum_force_balance_residual_norm_N']):.3e} N",
            "最大当前合力位置："
            f"x={float(maximum_force['wheel_lowest_point_x_m']) * 1.0e3:.6g} mm，"
            f"scan_index={int(maximum_force['scan_index'])}",
            "",
            LIMITATION_TEXT,
        )
    )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GuiInputError(f"结果读取失败：{field} 必须为 JSON 对象。")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuiInputError(f"结果读取失败：{field} 必须为有限数值。")
    number = float(value)
    if not math.isfinite(number):
        raise GuiInputError(f"结果读取失败：{field} 包含非有限数值。")
    return number


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise GuiInputError(f"结果读取失败：{field} 必须为非负整数。")
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise GuiInputError(f"结果读取失败：{field} 必须为布尔值。")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GuiInputError(f"结果读取失败：{field} 必须为非空文本。")
    return value.strip()


def _force_vector(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise GuiInputError(f"结果读取失败：{field} 必须为两个分量的数组。")
    return _finite(value[0], f"{field}[0]"), _finite(value[1], f"{field}[1]")


def _mechanism_summary(summary: dict[str, object]) -> str:
    prediction = _mapping(summary.get("phase7a2_prediction"), "phase7a2_prediction")
    grit = _mapping(
        prediction.get("resolved_grit_specification"),
        "phase7a2_prediction.resolved_grit_specification",
    )
    population = _mapping(
        prediction.get("grain_population"),
        "phase7a2_prediction.grain_population",
    )
    comparison = _mapping(summary.get("comparison"), "comparison")
    resolution = _mapping(summary.get("resolution"), "resolution")
    baseline = _mapping(summary.get("baseline_final_unloaded"), "baseline_final_unloaded")
    mechanism = _mapping(summary.get("mechanism_final_unloaded"), "mechanism_final_unloaded")

    abrasive = _text(grit.get("abrasive_material"), "resolved_grit_specification.abrasive_material")
    grit_path = _text(grit.get("resolution_path"), "resolved_grit_specification.resolution_path")
    designation = _text(grit.get("grit_designation"), "resolved_grit_specification.grit_designation")
    d50_m = _finite(grit.get("representative_diameter_d50_m"), "resolved_grit_specification.representative_diameter_d50_m")
    data_status = _text(grit.get("data_status"), "resolved_grit_specification.data_status")
    estimated_count = _finite(population.get("estimated_physical_active_grain_count"), "grain_population.estimated_physical_active_grain_count")
    group_count = _integer(population.get("equivalent_group_count"), "grain_population.equivalent_group_count")
    histories_identical = _boolean(comparison.get("total_force_history_identical"), "comparison.total_force_history_identical")
    force_difference = _finite(comparison.get("maximum_total_force_history_difference_N"), "comparison.maximum_total_force_history_difference_N")
    nodal_difference = _finite(comparison.get("maximum_target_nodal_load_vector_l2_difference_N"), "comparison.maximum_target_nodal_load_vector_l2_difference_N")
    balance_residual = _finite(comparison.get("maximum_force_balance_residual_N"), "comparison.maximum_force_balance_residual_N")
    resolution_status = _text(resolution.get("resolution_status"), "resolution.resolution_status")
    if resolution_status != "macro_scale_conservative_projection_not_grain_resolved":
        raise GuiInputError("结果读取失败：机制化单程结果的宏观投影标识无效。")

    baseline_force = _force_vector(baseline.get("external_force_N"), "baseline_final_unloaded.external_force_N")
    mechanism_force = _force_vector(mechanism.get("external_force_N"), "mechanism_final_unloaded.external_force_N")
    final_zero = all(abs(value) <= 1.0e-10 for value in (*baseline_force, *mechanism_force))
    if not final_zero:
        raise GuiInputError("结果读取失败：机制化单程结果的最终外载荷不是零。")

    baseline_displacement = _finite(baseline.get("maximum_residual_displacement_m"), "baseline_final_unloaded.maximum_residual_displacement_m")
    mechanism_displacement = _finite(mechanism.get("maximum_residual_displacement_m"), "mechanism_final_unloaded.maximum_residual_displacement_m")
    baseline_stress = _finite(baseline.get("maximum_residual_von_mises_stress_Pa"), "baseline_final_unloaded.maximum_residual_von_mises_stress_Pa")
    mechanism_stress = _finite(mechanism.get("maximum_residual_von_mises_stress_Pa"), "mechanism_final_unloaded.maximum_residual_von_mises_stress_Pa")
    baseline_plastic = _finite(baseline.get("maximum_equivalent_plastic_strain"), "baseline_final_unloaded.maximum_equivalent_plastic_strain")
    mechanism_plastic = _finite(mechanism.get("maximum_equivalent_plastic_strain"), "mechanism_final_unloaded.maximum_equivalent_plastic_strain")
    baseline_plastic_count = _integer(baseline.get("plastic_element_count"), "baseline_final_unloaded.plastic_element_count")
    mechanism_plastic_count = _integer(mechanism.get("plastic_element_count"), "mechanism_final_unloaded.plastic_element_count")
    baseline_plastic_fraction = _finite(baseline.get("plastic_element_fraction"), "baseline_final_unloaded.plastic_element_fraction")
    mechanism_plastic_fraction = _finite(mechanism.get("plastic_element_fraction"), "mechanism_final_unloaded.plastic_element_fraction")
    baseline_dissipation = _finite(baseline.get("accumulated_plastic_dissipation_J"), "baseline_final_unloaded.accumulated_plastic_dissipation_J")
    mechanism_dissipation = _finite(mechanism.get("accumulated_plastic_dissipation_J"), "mechanism_final_unloaded.accumulated_plastic_dissipation_J")

    return "\n".join(
        (
            "计算模式：机制化弹塑性完整单程",
            "对比路线：统一载荷基准路线 / 统计机制载荷路线（统计等效磨粒群）",
            f"总力历史一致：{'是' if histories_identical else '否'}；最大总力差={force_difference:.3e} N",
            f"最大节点载荷向量差：{nodal_difference:.6g} N",
            f"砂轮磨料：{abrasive}；粒度路径：{grit_path}；粒度标记：{designation}",
            f"代表粒径 d50：{d50_m * 1.0e6:.6g} μm；粒度数据状态：{data_status}",
            f"估算有效磨粒数：{estimated_count:.6g}；等效磨粒组数：{group_count}",
            "统一载荷基准路线最终卸载："
            f"最大等效塑性应变={baseline_plastic:.6g}，"
            f"最大残余位移={baseline_displacement * 1.0e6:.6g} um，"
            f"最大残余 von Mises 应力={baseline_stress * 1.0e-6:.6g} MPa，"
            f"塑性单元={baseline_plastic_count}（{baseline_plastic_fraction * 100.0:.6g}%），"
            f"累计塑性耗散={baseline_dissipation:.6g} J",
            "统计机制载荷路线最终卸载："
            f"机制路线最终最大等效塑性应变：{mechanism_plastic:.6g}；"
            f"机制路线最终最大残余位移：{mechanism_displacement * 1.0e6:.6g} um；"
            f"机制路线最终最大残余 von Mises 应力：{mechanism_stress * 1.0e-6:.6g} MPa；"
            f"塑性单元：{mechanism_plastic_count}（{mechanism_plastic_fraction * 100.0:.6g}%）；"
            f"累计塑性耗散={mechanism_dissipation:.6g} J",
            "最终外载荷为零："
            f"基准路线 Fx={baseline_force[0]:.6g} N、Fy={baseline_force[1]:.6g} N；"
            f"机制路线 Fx={mechanism_force[0]:.6g} N、Fy={mechanism_force[1]:.6g} N。"
            "残余应力、残余位移和塑性应变仍可非零。",
            f"最大力平衡残差：{balance_residual:.3e} N",
            "模型限制：当前结果属于宏观尺度保守投影，使用统计等效磨粒群；"
            f"没有解析真实 {d50_m * 1.0e6:.6g} μm 磨粒应力集中，"
            "不代表单磨粒高保真解析或真实粗糙度。",
            "参数状态：当前机制、磨粒群和 J2 材料参数仍为演示性且尚未实验标定；"
            + (
                "#5000 数据仍待厂家资料核实。"
                if designation == "#5000"
                else "当前粒度数据状态以已发布摘要为准。"
            )
            + "当前阶段没有温度场和热耦合。",
        )
    )


def _single_images(summary: dict[str, object]) -> tuple[GuiImage, ...]:
    artifacts = summary["artifacts"]
    return tuple(
        GuiImage(
            Path(str(artifacts[key])).stem,
            Path(str(artifacts[key])).resolve(),
            PNG_EXPLANATIONS[Path(str(artifacts[key])).name],
        )
        for key in SINGLE_PNG_KEYS
    )


def _scan_images(summary: dict[str, object]) -> tuple[GuiImage, ...]:
    images = [
        GuiImage(
            "扫描总览 scan_overview.png",
            Path(str(summary["artifacts"]["scan_overview_png"])).resolve(),
            PNG_EXPLANATIONS["scan_overview.png"],
        )
    ]
    roles_by_directory: dict[Path, list[str]] = {}
    for role in ROLE_LABELS:
        value = summary["snapshot_roles"][role]
        directory = Path(str(value["snapshot_directory"])).resolve()
        roles_by_directory.setdefault(directory, []).append(ROLE_LABELS[role])
    filenames = (
        "mesh.png",
        "displacement_magnitude.png",
        "von_mises_stress.png",
        "principal_stress_1_in_plane.png",
        "deformed_shape.png",
        "stress_contour.png",
        "strain_contour.png",
    )
    for directory, roles in roles_by_directory.items():
        role_label = "/".join(roles)
        for filename in filenames:
            images.append(
                GuiImage(
                    f"{role_label} - {filename}",
                    directory / filename,
                    PNG_EXPLANATIONS[filename],
                )
            )
    return tuple(images)


def _mechanism_images(summary: dict[str, object]) -> tuple[GuiImage, ...]:
    artifacts = _mapping(summary.get("artifacts"), "artifacts")
    raw_path = _text(
        artifacts.get("mechanism_history_comparison_png"),
        "artifacts.mechanism_history_comparison_png",
    )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise GuiInputError(
            "结果读取失败：mechanism_history_comparison.png 缺失或为空。"
        )
    try:
        signature = path.read_bytes()[:8]
    except OSError as exc:
        raise GuiInputError(
            f"结果读取失败：无法读取 mechanism_history_comparison.png：{exc}"
        ) from exc
    if signature != b"\x89PNG\r\n\x1a\n":
        raise GuiInputError(
            "结果读取失败：mechanism_history_comparison.png 不是有效 PNG 文件。"
        )
    return (
        GuiImage(
            "机制历史对比 mechanism_history_comparison.png",
            path,
            MECHANISM_PNG_EXPLANATION,
        ),
    )


def _append_image_explanations(summary_text: str, images: tuple[GuiImage, ...]) -> str:
    seen: set[str] = set()
    lines = ["", "PNG 阅读说明："]
    for image in images:
        filename = image.path.name
        if filename not in seen:
            lines.append(f"- {filename}：{image.explanation}")
            seen.add(filename)
    return summary_text + "\n".join(lines)


def read_published_result(mode: str, output_directory: str | Path) -> GuiPublishedResult:
    output = Path(output_directory).expanduser().resolve()
    if mode == SINGLE_MODE:
        summary_path = output / "summary.json"
        summary = _read_summary(summary_path)
        if summary.get("result_format") != "grindcae_phase_4c3b_evolved_surface_empirical_load_postprocess":
            raise GuiInputError("单位置结果摘要的 result_format 无效。")
        text = _single_summary(summary)
        images = _single_images(summary)
    elif mode == SCAN_MODE:
        summary_path = output / "scan_summary.json"
        summary = _read_summary(summary_path)
        if summary.get("result_format") != "grindcae_phase_4c4_single_pass_quasi_static_scan":
            raise GuiInputError("扫描结果摘要的 result_format 无效。")
        text = _scan_summary(summary)
        images = _scan_images(summary)
    elif mode == MECHANISM_HISTORY_MODE:
        summary_path = output / "summary.json"
        summary = _read_summary(summary_path)
        if summary.get("result_format") != "grindcae_phase_7a3_statistical_mechanism_elastoplastic_history_comparison":
            raise GuiInputError("机制化弹塑性结果摘要的 result_format 无效。")
        text = _mechanism_summary(summary)
        images = _mechanism_images(summary)
    else:
        raise GuiInputError("计算模式无效。")
    text = _append_image_explanations(text, images)
    return GuiPublishedResult(mode, output, summary_path, summary, text, images)
