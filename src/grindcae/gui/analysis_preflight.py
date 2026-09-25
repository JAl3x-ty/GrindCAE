"""Fast read-only readiness audit before a formal analysis is started."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path
import tempfile
from typing import Callable

from .analysis import BuiltAnalysisInput


@dataclass(frozen=True, slots=True)
class AnalysisAuditItem:
    level: str
    title: str
    message: str


@dataclass(frozen=True, slots=True)
class AnalysisAuditReport:
    can_run: bool
    items: tuple[AnalysisAuditItem, ...]
    position_count: int
    route_count: int
    summary_text: str


def probe_directory_publication(parent: Path) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grindcae-audit-", dir=parent) as root_text:
        root = Path(root_text)
        source = root / "source"
        target = root / "renamed"
        source.mkdir()
        (source / "probe.txt").write_text("ok", encoding="ascii")
        source.replace(target)
        if (target / "probe.txt").read_text(encoding="ascii") != "ok":
            raise OSError("目录级改名后的测试文件无法回读")


def _nested(value: object, *keys: str, default=None):
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _history_root(mode: str, normalized: dict[str, object]) -> dict[str, object]:
    if mode == "literature_elastoplastic_single_pass":
        return normalized.get("history_template", {})
    if mode == "mechanism_elastoplastic_single_pass":
        value = normalized.get("baseline_history_pass", {})
        return value if isinstance(value, dict) else {}
    return normalized


def quick_analysis_audit(
    built: BuiltAnalysisInput,
    output_directory: str | Path,
    *,
    available_bytes: int | None = None,
    publish_probe: Callable[[Path], None] = probe_directory_publication,
) -> AnalysisAuditReport:
    output = Path(output_directory).expanduser().resolve()
    parent = output.parent
    mode = built.definition.mode_id
    normalized = built.normalized_input
    history = _history_root(mode, normalized)
    if mode == "single_grain_high_fidelity":
        targets = _nested(normalized, "trajectory", "targets", default=())
        position_count = len(targets) if isinstance(targets, (list, tuple)) else 1
    else:
        position_count = int(
            _nested(history, "moving_load_pass", "pass", "base_position_count", default=0)
            or _nested(normalized, "scan", "base_position_count", default=1)
            or 1
        )
    route_count = 2 if mode in {"mechanism_elastoplastic_single_pass", "literature_elastoplastic_single_pass"} else 1
    items: list[AnalysisAuditItem] = [
        AnalysisAuditItem("pass", "严格输入", "规范化求解输入已构造并通过模型校验。")
    ]
    if mode == "literature_elastoplastic_single_pass":
        items.append(AnalysisAuditItem("warning", "模型适用性", "材料迁移尚未标定；条带载荷采用明确的宏观分布假设。"))
        if output.exists():
            items.append(AnalysisAuditItem("block", "独占输出", "文献联算必须使用尚不存在的新目录。"))
    try:
        publish_probe(parent)
    except OSError as exc:
        items.append(AnalysisAuditItem("block", "结果发布", f"输出父目录无法完成目录级改名测试：{exc}"))
    else:
        items.append(AnalysisAuditItem("pass", "结果发布", "输出父目录已通过创建、写入、目录级改名和回读测试。"))
    existing_sequence = output / "field_evolution"
    if existing_sequence.exists():
        items.append(AnalysisAuditItem(
            "warning",
            "已有场序列",
            "已有场序列：目标目录包含 field_evolution；正式运行会事务替换该受管目录，请关闭正在查看其中图片或文件的程序。",
        ))
    free = available_bytes
    if free is None:
        try:
            free = shutil.disk_usage(parent if parent.exists() else parent.parent).free
        except OSError as exc:
            items.append(AnalysisAuditItem("warning", "磁盘空间", f"无法读取剩余空间：{exc}"))
    if free is not None:
        if free < 512 * 1024**2:
            items.append(AnalysisAuditItem("block", "磁盘空间", "可用空间不足 512 MB。"))
        elif free < 2 * 1024**3:
            items.append(AnalysisAuditItem("warning", "磁盘空间", "可用空间不足 2 GB，场序列发布存在风险。"))
        else:
            items.append(AnalysisAuditItem("pass", "磁盘空间", f"可用空间约 {free / 1024**3:.1f} GB。"))
    newton = _nested(history, "moving_load_pass", "reference_case", "newton", default={})
    if not isinstance(newton, dict) or not newton:
        newton = _nested(history, "reference_case", "newton", default={})
    if (not isinstance(newton, dict) or not newton) and mode == "single_grain_high_fidelity":
        newton = _nested(normalized, "solver", "newton", default={})
    if isinstance(newton, dict):
        tolerance = float(newton.get("residual_relative_tolerance", 1.0e-8))
        maximum_iterations = int(newton.get("maximum_iterations", 40))
        if tolerance < 1.0e-10:
            items.append(AnalysisAuditItem("warning", "收敛设置", "Newton 相对残差容差非常严格，可能明显增加迭代和失败风险。"))
        if maximum_iterations < 20:
            items.append(AnalysisAuditItem("warning", "收敛设置", "Newton 最大迭代次数低于 20，复杂塑性增量可能提前终止。"))
    transition = history.get("transition", {}) if isinstance(history, dict) else {}
    if isinstance(transition, dict) and int(transition.get("base_substep_count", 2)) < 2:
        items.append(AnalysisAuditItem("warning", "历史步长", "历史基础子步少于 2，位置间载荷变化较大时收敛余量较小。"))
    workload = position_count * route_count
    symbols = {"pass": "✓", "warning": "⚠", "block": "✕"}
    lines = [
        f"运行前快速审查：{built.definition.display_name}",
        f"预计正式位置工作量：{workload}（{position_count} 个位置 × {route_count} 条路线）",
        "",
        *(f"{symbols[item.level]} {item.title}：{item.message}" for item in items),
    ]
    return AnalysisAuditReport(
        not any(item.level == "block" for item in items),
        tuple(items), position_count, route_count, "\n".join(lines)
    )


__all__ = ["AnalysisAuditItem", "AnalysisAuditReport", "probe_directory_publication", "quick_analysis_audit"]
