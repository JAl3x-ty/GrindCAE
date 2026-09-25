"""Presentation-neutral catalog for persisted formal analysis results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .analysis import analysis_mode_definition
from .analysis_results import AnalysisPublishedResult, AnalysisResultAdapter
from .field_evolution_result import (
    FieldEvolutionAvailability,
    read_field_evolution_availability,
)
from .mechanism_field_evolution_result import (
    MechanismFieldEvolutionAvailability,
    read_mechanism_field_evolution_availability,
)
from .project import FORMAL_ANALYSIS_TYPES, GrindCaeProject, ResultRecord


RESULT_CATEGORY_IDS = ("summary", "history", "fields", "mesh", "data")

CATEGORY_TITLES = {
    "summary": "结果摘要",
    "history": "曲线与历程",
    "fields": "云图与场量",
    "mesh": "网格与变形",
    "data": "数据文件",
}


class ResultCatalogError(ValueError):
    """Raised when a persisted result cannot be safely categorized."""


@dataclass(frozen=True, slots=True)
class ResultArtifactItem:
    display_name: str
    path: Path
    kind: str
    reading_note: str = ""

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def file_type(self) -> str:
        return self.path.suffix.lstrip(".").upper() or "文件"

    @property
    def size_bytes(self) -> int | None:
        try:
            return self.path.stat().st_size if self.path.is_file() else None
        except OSError:
            return None

    @property
    def status(self) -> str:
        size = self.size_bytes
        if size is None:
            return "缺失"
        if size == 0:
            return "文件为空"
        return "可用"


def format_file_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "—"
    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.1f} {unit}"


@dataclass(frozen=True, slots=True)
class ResultCategory:
    category_id: str
    title: str
    items: tuple[ResultArtifactItem, ...] = ()
    empty_message: str = ""


@dataclass(frozen=True, slots=True)
class CategorizedAnalysisResult:
    record: ResultRecord
    published: AnalysisPublishedResult
    categories: tuple[ResultCategory, ...]
    field_evolution: FieldEvolutionAvailability = FieldEvolutionAvailability.not_applicable()
    mechanism_field_evolution: MechanismFieldEvolutionAvailability = (
        MechanismFieldEvolutionAvailability.not_applicable()
    )

    def category(self, category_id: str) -> ResultCategory:
        for category in self.categories:
            if category.category_id == category_id:
                return category
        raise KeyError(category_id)


_IMAGE_NOTES = {
    "field": "颜色表示计算场量，实际数值和单位以摘要及数据文件为准；局部峰值可能具有网格敏感性。",
    "mesh": "图中网格或变形比例用于工程观察，实际坐标与位移以摘要和数据文件为准。",
    "history": "曲线用于查看载荷或响应随位置和增量的变化，精确数值以摘要和数据文件为准。",
    "mechanism": "该结果属于宏观统计等效磨粒载荷投影，不代表真实砂轮微观组织或实际粗糙度。",
}


_MODE_ITEMS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "literature_elastoplastic_single_pass": {
        "history": (("literature_history_comparison.png", "均匀/条带历史响应对比"), ("force_components.png", "论文力四分量")),
        "fields": (("baseline_pass_residual_state.png", "均匀载荷卸载残余场"), ("spatial_pass_residual_state.png", "条带载荷卸载残余场")),
        "mesh": (("baseline_final_surface_profile.png", "均匀载荷残余表面位移"), ("spatial_final_surface_profile.png", "条带载荷残余表面位移")),
        "data": (("summary.json", "文献联算摘要"), ("input.json", "联算输入与来源"),
                 ("resolved_history_input.json", "重算力后的历史输入"), ("grains.csv", "逐磨粒力"),
                 ("strip_template.json", "周向条带四分量"), ("projection_history.csv", "分项投影守恒记录"),
                 ("baseline_pass_history.csv", "均匀载荷历史"), ("spatial_pass_history.csv", "条带载荷历史"),
                 ("baseline_final_results.vtu", "均匀载荷最终场"), ("spatial_final_results.vtu", "条带载荷最终场"),
                 ("baseline_final_nodes.csv", "均匀载荷节点"), ("spatial_final_nodes.csv", "条带载荷节点"),
                 ("baseline_final_elements.csv", "均匀载荷单元"), ("spatial_final_elements.csv", "条带载荷单元"),
                 ("reference_mesh.msh", "共享参考网格")),
    },
    "literature_grinding_force": {
        "history": (("force_components.png", "磨削力四分量"),),
        "data": (("summary.json", "文献力结果摘要"), ("input.json", "完整文献算例"), ("grains.csv", "逐磨粒力与切深")),
    },
    "linear_elastic_single_position": {
        "fields": (
            ("displacement_magnitude.png", "位移幅值"),
            ("von_mises_stress.png", "von Mises 等效应力"),
            ("principal_stress_1_in_plane.png", "第一面内主应力"),
            ("stress_contour.png", "原生单元应力云图"),
            ("strain_contour.png", "原生单元应变云图"),
        ),
        "mesh": (("mesh.png", "网格"), ("deformed_shape.png", "变形形状")),
        "data": (
            ("summary.json", "结果摘要"), ("evolved_surface.msh", "演化表面网格"),
            ("results.vtu", "有限元场数据"), ("nodes.csv", "节点数据"),
            ("elements.csv", "单元数据"), ("active_contact_facets.csv", "有效接触边数据"),
        ),
    },
    "elastoplastic_single_position": {
        "history": (("surface_recovery.png", "表面加载与弹性恢复"),),
        "fields": (
            ("peak_displacement.png", "峰值位移"), ("peak_von_mises_stress.png", "峰值等效应力"),
            ("equivalent_plastic_strain.png", "等效塑性应变"), ("residual_displacement.png", "卸载后残余位移"),
            ("residual_von_mises_stress.png", "卸载后残余应力"),
        ),
        "mesh": (("mesh.png", "网格"),),
        "data": (
            ("summary.json", "结果摘要"), ("increment_history.csv", "增量与 Newton 历程"),
            ("solve_input.msh", "求解网格"), ("results.vtu", "峰值与残余场数据"),
            ("nodes.csv", "节点数据"), ("elements.csv", "单元数据"),
            ("active_contact_facets.csv", "有效接触边数据"), ("surface_recovery.csv", "表面恢复数据"),
        ),
    },
    "elastoplastic_single_pass": {
        "history": (
            ("pass_force_history.png", "单程磨削力历程"), ("pass_plastic_history.png", "单程塑性响应历程"),
            ("final_surface_profile.png", "最终表面轮廓"),
        ),
        "fields": (("pass_residual_state.png", "最终卸载残余状态"),),
        "mesh": (("moving_load_overview.png", "固定网格与移动载荷总览"),),
        "data": (
            ("pass_summary.json", "结果摘要"), ("pass_history.csv", "单程历史数据"),
            ("final_results.vtu", "最终残余场数据"), ("final_nodes.csv", "最终节点数据"),
            ("final_elements.csv", "最终单元数据"), ("reference_mesh.msh", "固定参考网格"),
        ),
    },
    "mechanism_elastoplastic_single_pass": {
        "history": (("mechanism_history_comparison.png", "基准载荷与机制载荷历史对比"),),
        "data": (
            ("summary.json", "结果摘要"), ("comparison_history.csv", "基准与机制响应对比历史"),
            ("mechanism_load_history.csv", "机制载荷历史"), ("baseline_final_results.vtu", "基准路线最终场数据"),
            ("mechanism_final_results.vtu", "机制路线最终场数据"), ("final_nodes.csv", "路线对比节点数据"),
            ("baseline_final_elements.csv", "基准路线单元数据"), ("mechanism_final_elements.csv", "机制路线单元数据"),
            ("reference_mesh.msh", "固定参考网格"),
        ),
    },
    "single_grain_high_fidelity": {
        "history": (
            ("reaction_history.png", "单磨粒法向/切向反力历程"),
            ("energy_history.png", "接触功、储能、耗散与平衡历程"),
            ("residual_groove.png", "卸载后固定网格残余沟槽"),
        ),
        "fields": (
            ("contact_pressure.png", "峰值接触压力"),
            ("contact_state.png", "峰值接触 open/stick/slip 状态"),
            ("final_fields.png", "最终残余应力与塑性区"),
        ),
        "mesh": (("reference_mesh.msh", "固定参考网格"),),
        "data": (
            ("summary.json", "结果摘要"),
            ("contact_history.csv", "接触轨迹历史"),
            ("contact_points.csv", "代表位置接触点"),
            ("final_results.vtu", "最终残余场数据"),
            ("final_nodes.csv", "最终节点数据"),
            ("final_elements.csv", "最终单元数据"),
            ("residual_surface_profile.csv", "残余表面位移轮廓"),
        ),
    },
}

_SINGLE_GRAIN_V2_ITEMS = {
    "history": (
        ("reaction_history.png", "单磨粒法向/切向反力历程"),
        ("energy_history.png", "接触功、储能、耗散与平衡历程"),
        ("unloaded_surface_profile.png", "卸载后动态自由表面轮廓"),
    ),
    "fields": (
        ("contact_pressure.png", "峰值接触压力"),
        ("contact_state.png", "峰值接触状态与磨粒特征"),
        ("final_fields.png", "最终有效材料残余场"),
    ),
    "mesh": (
        ("reference_mesh.msh", "背景参考网格"),
        ("grain_geometry.png", "磨粒形貌、圆角与姿态"),
        ("effective_mesh_surface.png", "有效网格与动态自由表面"),
    ),
    "data": (
        ("summary.json", "结果摘要"), ("contact_history.csv", "接触轨迹历史"),
        ("contact_points.csv", "代表位置接触点与磨粒特征"),
        ("material_topology.csv", "背景单元材料状态"),
        ("topology_history.csv", "材料拓扑提交历史"),
        ("free_surface.csv", "动态自由表面历史"),
        ("final_results.vtu", "最终有效材料物理场"),
        ("effective_mesh.vtu", "完整背景拓扑与材料状态"),
        ("final_nodes.csv", "最终有效节点与背景映射"),
        ("final_elements.csv", "最终有效单元与背景映射"),
        ("unloaded_surface_profile.csv", "卸载后自由表面数据"),
    ),
}

_SINGLE_GRAIN_V3_ITEMS = {
    "history": (
        ("reaction_history.png", "单磨粒法向/切向反力历程"),
        ("energy_history.png", "接触功、储能、塑性、损伤与分离能量"),
        ("damage_evolution.png", "损伤演化历程"),
        ("removal_area_history.png", "自动分离面积历程"),
        ("unloaded_surface_profile.png", "卸载后动态自由表面轮廓"),
        ("processed_surface.png", "模型预测的加工表面"),
    ),
    "fields": (
        ("contact_pressure.png", "接触压力"),
        ("contact_state.png", "接触状态与磨粒特征"),
        ("final_fields.png", "最终有效材料物理场"),
        ("final_damage_field.png", "最终损伤场"),
    ),
    "mesh": (
        ("reference_mesh.msh", "背景参考网格"),
        ("grain_geometry.png", "磨粒形貌、圆角与姿态"),
        ("effective_mesh_surface.png", "有效网格与动态自由表面"),
    ),
    "data": (
        ("summary.json", "结果摘要"), ("contact_history.csv", "接触与能量历史"),
        ("contact_points.csv", "最终接触点、背景 ID 与特征"),
        ("material_topology.csv", "最终材料拓扑与移除来源"),
        ("topology_history.csv", "拓扑提交历史"), ("free_surface.csv", "自由表面历史"),
        ("damage_history.csv", "损伤与移除面积历史"),
        ("damage_element_history.csv", "稀疏背景单元损伤历史"),
        ("separation_events.csv", "分离事件历史"),
        ("final_damage_state.csv", "完整背景单元最终损伤状态"),
        ("final_results.vtu", "最终有效材料物理场"),
        ("effective_mesh.vtu", "完整背景拓扑与状态"),
        ("final_nodes.csv", "最终有效节点与背景映射"),
        ("final_elements.csv", "最终有效单元与背景映射"),
        ("unloaded_surface_profile.csv", "卸载后自由表面数据"),
    ),
}


_PASS_SNAPSHOT_ITEMS = {
    "fields": _MODE_ITEMS["linear_elastic_single_position"]["fields"],
    "mesh": _MODE_ITEMS["linear_elastic_single_position"]["mesh"],
    "data": tuple(
        item for item in _MODE_ITEMS["linear_elastic_single_position"]["data"]
        if item[0] != "summary.json"
    ),
}


_SNAPSHOT_ROLE_NAMES = {
    "representative_entry": "磨入代表位置",
    "representative_full_contact": "满接触代表位置",
    "maximum_displacement": "最大位移位置",
    "maximum_von_mises_p95": "最大等效应力 p95 位置",
    "maximum_von_mises_p99": "最大响应位置",
    "representative_exit": "磨出代表位置",
}


def _created_at_key(record: ResultRecord) -> tuple[datetime, str]:
    try:
        created_at = datetime.fromisoformat(record.created_at)
    except ValueError:
        created_at = datetime.min
    if created_at.tzinfo is not None:
        created_at = created_at.replace(tzinfo=None)
    return created_at, record.result_id


def formal_result_records(project: GrindCaeProject) -> tuple[ResultRecord, ...]:
    """Return completed formal-analysis records in deterministic newest-first order."""

    records = (
        record
        for record in project.results
        if record.analysis_type in FORMAL_ANALYSIS_TYPES and record.status == "completed"
    )
    return tuple(sorted(records, key=_created_at_key, reverse=True))


def _safe_path(raw: str, output: Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = output / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(output)
    except ValueError as exc:
        raise ResultCatalogError(f"结果产物路径超出结果目录：{raw}") from exc
    return resolved


def _artifact_paths(summary: Mapping[str, Any], output: Path) -> tuple[Path, ...]:
    if summary.get("result_format") == "grindcae_literature_elastoplastic_history_v1":
        from grindcae.literature_history import read_result
        read_result(output)
        # The root contains dedicated display copies. Nested route bundles keep
        # their native filenames and are checked by the composed reader.
        return tuple(_safe_path(name, output) for name in summary["artifacts"] if len(Path(name).parts)==1)
    if summary.get("result_format") == "grindcae_zhang2017_reconstruction_v2":
        from grindcae380.workflow import read_result
        read_result(output)
        return tuple(_safe_path(name, output) for name in ("input.json", "grains.csv", "force_components.png"))
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ResultCatalogError("结果摘要缺少有效的 artifacts 登记。")
    paths: list[Path] = []
    for value in artifacts.values():
        raw = value.get("path") if isinstance(value, Mapping) else value
        if not isinstance(raw, str) or not raw.strip():
            raise ResultCatalogError("结果摘要包含无效的产物路径。")
        paths.append(_safe_path(raw, output))
    return tuple(paths)


def _find_filename(paths: tuple[Path, ...], filename: str) -> Path | None:
    matches = tuple(path for path in paths if path.name == filename)
    if len(matches) > 1:
        raise ResultCatalogError(f"结果摘要重复登记文件：{filename}")
    return matches[0] if matches else None


def _item(display_name: str, path: Path, category_id: str, mode: str) -> ResultArtifactItem:
    note_key = "mechanism" if mode == "mechanism_elastoplastic_single_pass" else (
        "field" if category_id == "fields" else "mesh" if category_id == "mesh" else "history"
    )
    return ResultArtifactItem(display_name, path, "image" if path.suffix.lower() == ".png" else "data", _IMAGE_NOTES.get(note_key, ""))


def _standard_categories(
    mode: str,
    summary_path: Path,
    artifact_paths: tuple[Path, ...],
) -> dict[str, list[ResultArtifactItem]]:
    categories = {category_id: [] for category_id in RESULT_CATEGORY_IDS}
    if mode == "single_grain_high_fidelity" and _find_filename(
        artifact_paths, "damage_history.csv"
    ) is not None:
        mapping = _SINGLE_GRAIN_V3_ITEMS
    elif mode == "single_grain_high_fidelity" and _find_filename(
        artifact_paths, "material_topology.csv"
    ) is not None:
        mapping = _SINGLE_GRAIN_V2_ITEMS
    else:
        mapping = _MODE_ITEMS.get(mode, {})
    for category_id, definitions in mapping.items():
        for filename, display_name in definitions:
            path = summary_path if filename == summary_path.name else _find_filename(artifact_paths, filename)
            if path is not None:
                categories[category_id].append(_item(display_name, path, category_id, mode))
    return categories


def _linear_pass_categories(
    summary: Mapping[str, Any], output: Path, summary_path: Path, artifact_paths: tuple[Path, ...]
) -> dict[str, list[ResultArtifactItem]]:
    categories = {category_id: [] for category_id in RESULT_CATEGORY_IDS}
    overview = _find_filename(artifact_paths, "scan_overview.png")
    if overview is not None:
        categories["history"].append(_item("单程扫描总览", overview, "history", "linear_elastic_single_pass"))
    categories["data"].append(_item("结果摘要", summary_path, "data", "linear_elastic_single_pass"))
    history = _find_filename(artifact_paths, "scan_history.csv")
    if history is not None:
        categories["data"].append(_item("扫描历史数据", history, "data", "linear_elastic_single_pass"))

    roles = summary.get("snapshot_roles", {})
    if not isinstance(roles, Mapping):
        raise ResultCatalogError("单程结果的代表快照登记无效。")
    directories: dict[Path, list[str]] = {}
    for role, details in roles.items():
        if not isinstance(role, str) or not isinstance(details, Mapping):
            continue
        raw_directory = details.get("snapshot_directory")
        if not isinstance(raw_directory, str):
            continue
        directory = _safe_path(raw_directory, output)
        directories.setdefault(directory, []).append(_SNAPSHOT_ROLE_NAMES.get(role, role))
    for directory, role_names in directories.items():
        role_label = "／".join(dict.fromkeys(role_names))
        for category_id, definitions in _PASS_SNAPSHOT_ITEMS.items():
            for filename, display_name in definitions:
                path = (directory / filename).resolve()
                if category_id == "data" or path.is_file():
                    categories[category_id].append(
                        _item(f"{role_label}－{display_name}", path, category_id, "linear_elastic_single_pass")
                    )
    return categories


class ResultCatalog:
    """Read one already-published formal result without invoking a solver."""

    def __init__(self, adapter: AnalysisResultAdapter | None = None) -> None:
        self._adapter = adapter or AnalysisResultAdapter()

    def read(self, record: ResultRecord) -> CategorizedAnalysisResult:
        if record.analysis_type not in FORMAL_ANALYSIS_TYPES:
            raise ResultCatalogError("结果中心只接受已注册的正式分析记录。")
        output = Path(record.output_directory).expanduser().resolve()
        definition = analysis_mode_definition(record.analysis_type, require_enabled=True)
        summary_path = Path(record.summary_path).expanduser().resolve()
        expected_summary = (output / definition.summary_filename).resolve()
        if summary_path != expected_summary:
            raise ResultCatalogError("结果摘要路径与登记输出目录不一致。")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultCatalogError(f"结果摘要无法读取：{exc}") from exc
        if not isinstance(summary, Mapping):
            raise ResultCatalogError("结果摘要根对象无效。")
        artifact_paths = _artifact_paths(summary, output)
        try:
            published = self._adapter.read(record.analysis_type, output)
        except ValueError as exc:
            raise ResultCatalogError(str(exc)) from exc
        if record.analysis_type == "linear_elastic_single_pass":
            item_map = _linear_pass_categories(summary, output, summary_path, artifact_paths)
        else:
            item_map = _standard_categories(record.analysis_type, summary_path, artifact_paths)
        field_evolution = FieldEvolutionAvailability.not_applicable()
        mechanism_field_evolution = MechanismFieldEvolutionAvailability.not_applicable()
        if record.analysis_type == "elastoplastic_single_pass":
            field_evolution = read_field_evolution_availability(summary, output)
            if field_evolution.sequence is not None:
                item_map["data"].extend((
                    _item("场演化清单", field_evolution.sequence.manifest_path, "data", record.analysis_type),
                    _item("场演化位置历史", field_evolution.sequence.position_history_csv_path, "data", record.analysis_type),
                    _item("ParaView 场序列", field_evolution.sequence.pvd_path, "data", record.analysis_type),
                ))
        elif record.analysis_type == "mechanism_elastoplastic_single_pass":
            mechanism_field_evolution = read_mechanism_field_evolution_availability(
                summary, output
            )
            if mechanism_field_evolution.sequence is not None:
                sequence = mechanism_field_evolution.sequence
                item_map["data"].extend((
                    _item("机制化场演化清单", sequence.manifest_path, "data", record.analysis_type),
                    _item("机制化场演化位置历史", sequence.position_history_csv_path, "data", record.analysis_type),
                    _item("基准路线 ParaView 场序列", sequence.pvd_paths["baseline"], "data", record.analysis_type),
                    _item("机制路线 ParaView 场序列", sequence.pvd_paths["mechanism"], "data", record.analysis_type),
                    _item("差值路线 ParaView 场序列", sequence.pvd_paths["difference"], "data", record.analysis_type),
                ))
        categories = []
        for category_id in RESULT_CATEGORY_IDS:
            empty_message = ""
            if category_id == "summary":
                empty_message = published.summary_text
            elif record.analysis_type == "mechanism_elastoplastic_single_pass" and category_id == "fields":
                empty_message = "当前结果没有独立云图产物。可在历史对比图和 VTU 数据中查看机制路线与基准路线结果。"
            elif not item_map[category_id]:
                empty_message = "当前结果没有该分类的已登记产物。"
            categories.append(ResultCategory(category_id, CATEGORY_TITLES[category_id], tuple(item_map[category_id]), empty_message))
        return CategorizedAnalysisResult(
            record,
            published,
            tuple(categories),
            field_evolution,
            mechanism_field_evolution,
        )


__all__ = [
    "CATEGORY_TITLES", "RESULT_CATEGORY_IDS", "CategorizedAnalysisResult",
    "ResultArtifactItem", "ResultCatalog", "ResultCatalogError", "ResultCategory",
    "formal_result_records", "format_file_size",
]
