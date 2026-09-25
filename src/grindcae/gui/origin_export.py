"""Read-only Origin-friendly table export for one registered formal result."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .result_catalog import CategorizedAnalysisResult


class OriginExportError(ValueError):
    """Raised when a read-only Origin export cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class OriginExportResult:
    directory: Path
    table_paths: tuple[Path, ...]


_MODE_CSV_FILENAMES = {
    "literature_elastoplastic_single_pass": ("grains.csv", "projection_history.csv", "baseline_pass_history.csv", "spatial_pass_history.csv",
        "baseline_final_nodes.csv", "spatial_final_nodes.csv", "baseline_final_elements.csv", "spatial_final_elements.csv"),
    "literature_grinding_force": ("grains.csv",),
    "linear_elastic_single_position": (
        "nodes.csv",
        "elements.csv",
        "active_contact_facets.csv",
    ),
    "linear_elastic_single_pass": (
        "scan_history.csv",
        "nodes.csv",
        "elements.csv",
        "active_contact_facets.csv",
    ),
    "elastoplastic_single_position": (
        "increment_history.csv",
        "surface_recovery.csv",
        "nodes.csv",
        "elements.csv",
        "active_contact_facets.csv",
    ),
    "elastoplastic_single_pass": (
        "pass_history.csv",
        "position_history.csv",
        "final_nodes.csv",
        "final_elements.csv",
    ),
    "mechanism_elastoplastic_single_pass": (
        "comparison_history.csv",
        "mechanism_load_history.csv",
        "position_history.csv",
        "final_nodes.csv",
        "baseline_final_elements.csv",
        "mechanism_final_elements.csv",
    ),
}

_CONVERSIONS = {
    "depth_m": ("depth_um", 1.0e6),
    "static_depth_m": ("static_depth_um", 1.0e6),
    "protrusion_m": ("protrusion_um", 1.0e6),
    "diameter_m": ("diameter_um", 1.0e6),
    "motion_coordinate_m": ("motion_coordinate_mm", 1.0e3),
    "wheel_lowest_point_x_m": ("wheel_lowest_point_x_mm", 1.0e3),
    "x_m": ("x_mm", 1.0e3),
    "y_m": ("y_mm", 1.0e3),
    "maximum_displacement_m": ("maximum_displacement_um", 1.0e6),
    "baseline_maximum_displacement_m": (
        "baseline_maximum_displacement_um",
        1.0e6,
    ),
    "mechanism_maximum_displacement_m": (
        "mechanism_maximum_displacement_um",
        1.0e6,
    ),
    "maximum_von_mises_stress_Pa": ("maximum_von_mises_stress_MPa", 1.0e-6),
    "baseline_maximum_von_mises_stress_Pa": (
        "baseline_maximum_von_mises_stress_MPa",
        1.0e-6,
    ),
    "mechanism_maximum_von_mises_stress_Pa": (
        "mechanism_maximum_von_mises_stress_MPa",
        1.0e-6,
    ),
}


def _safe_registered_csvs(result: CategorizedAnalysisResult) -> tuple[Path, ...]:
    output = Path(result.record.output_directory).expanduser().resolve()
    allowed = set(_MODE_CSV_FILENAMES.get(result.record.analysis_type, ()))
    paths: list[Path] = []
    for item in result.category("data").items:
        path = item.path.expanduser().resolve()
        if path.name not in allowed:
            continue
        try:
            path.relative_to(output)
        except ValueError as exc:
            raise OriginExportError(f"数据文件超出正式结果目录：{path}") from exc
        if not path.is_file() or path.stat().st_size <= 0:
            raise OriginExportError(f"数据文件不存在或为空：{path}")
        paths.append(path)
    if not paths:
        raise OriginExportError("当前结果没有可导出的已登记 CSV 数据。")
    return tuple(paths)


def _format_number(value: float) -> str:
    return f"{value:.15g}"


def _write_converted_csv(source: Path, target: Path) -> None:
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise OriginExportError(f"无法读取 CSV：{source}：{exc}") from exc
    if not fieldnames:
        raise OriginExportError(f"CSV 缺少表头：{source}")
    derived = tuple(
        converted for field in fieldnames if (converted := _CONVERSIONS.get(field))
    )
    output_fields = fieldnames + tuple(name for name, _factor in derived)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for source_row in rows:
            row = dict(source_row)
            for field in fieldnames:
                conversion = _CONVERSIONS.get(field)
                if conversion is None or source_row.get(field, "") == "":
                    continue
                name, factor = conversion
                try:
                    row[name] = _format_number(float(source_row[field]) * factor)
                except ValueError as exc:
                    raise OriginExportError(
                        f"CSV 工程单位换算失败：{source.name} 的 {field} 不是数值。"
                    ) from exc
            writer.writerow(row)


def _flatten_metrics(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"normalized_input", "artifacts", "field_evolution"}:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_metrics(child, path))
    elif isinstance(value, (int, float, bool, str)) and prefix:
        rows.append((prefix, str(value)))
    return rows


def _write_key_metrics(summary_path: Path, target: Path) -> None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OriginExportError(f"无法读取结果摘要：{exc}") from exc
    if not isinstance(summary, dict):
        raise OriginExportError("结果摘要根对象无效。")
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(_flatten_metrics(summary))


def _write_guide(target: Path, table_names: tuple[str, ...]) -> None:
    lines = [
        "GrindCAE Origin 数据导入说明",
        "",
        "正式结果文件不会被修改；本目录仅包含从已登记结果生成的只读副本。",
        "CSV 使用 UTF-8 BOM 和英文逗号，Origin 可通过 数据 -> 导入 -> 单个 ASCII/CSV 打开。",
        "原始 SI 列完整保留，新增 mm、um、MPa 工程单位列用于直接作图。",
        "",
        "建议作图：",
        "X：motion_coordinate_mm（若当前表包含该列）",
        "Y：Fx_N、Fy_N、maximum_displacement_um、maximum_von_mises_stress_MPa 等。",
        "机制路线可将 baseline_* 与 mechanism_* 设置为两条 Y 曲线。",
        "",
        "导出的数据表：",
        *(f"- {name}" for name in table_names),
        "- key_metrics.csv",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def export_origin_data(
    result: CategorizedAnalysisResult,
    destination: str | Path,
) -> OriginExportResult:
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise OriginExportError(f"导出目录已存在，请选择新目录：{target}")
    sources = _safe_registered_csvs(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        table_names: list[str] = []
        duplicate_names = {
            source.name
            for source in sources
            if sum(item.name == source.name for item in sources) > 1
        }
        for source in sources:
            prefix = f"{source.parent.name}_" if source.name in duplicate_names else ""
            name = f"{prefix}{source.stem}_origin.csv"
            _write_converted_csv(source, temporary / name)
            table_names.append(name)
        summary_path = Path(result.record.summary_path).expanduser().resolve()
        _write_key_metrics(summary_path, temporary / "key_metrics.csv")
        _write_guide(temporary / "Origin导入说明.txt", tuple(table_names))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return OriginExportResult(
        target,
        tuple(target / name for name in table_names),
    )


__all__ = ["OriginExportError", "OriginExportResult", "export_origin_data"]
