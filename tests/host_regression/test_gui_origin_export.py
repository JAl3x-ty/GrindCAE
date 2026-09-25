from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from grindcae.gui.analysis_results import AnalysisPublishedResult
from grindcae.gui.origin_export import OriginExportError, export_origin_data
from grindcae.gui.project import ResultRecord
from grindcae.gui.result_catalog import (
    CategorizedAnalysisResult,
    ResultArtifactItem,
    ResultCategory,
)


def _history_result(tmp_path: Path) -> CategorizedAnalysisResult:
    summary = tmp_path / "pass_summary.json"
    summary.write_text(
        json.dumps(
            {
                "result_format": "grindcae_phase_6b2_fixed_mesh_elastoplastic_history_pass",
                "package_version": "2.13.6",
                "unit_system": "SI",
                "maximum_response": {
                    "maximum_displacement_m": 2.5e-6,
                    "maximum_von_mises_stress_Pa": 4.6e6,
                    "maximum_equivalent_plastic_strain": 0.0012,
                },
                "artifacts": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    history = tmp_path / "pass_history.csv"
    with history.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "position_id",
                "motion_coordinate_m",
                "Fx_N",
                "Fy_N",
                "maximum_displacement_m",
                "maximum_von_mises_stress_Pa",
                "maximum_equivalent_plastic_strain",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "position_id": 1,
                "motion_coordinate_m": 0.012,
                "Fx_N": 80.0,
                "Fy_N": -200.0,
                "maximum_displacement_m": 2.5e-6,
                "maximum_von_mises_stress_Pa": 4.6e6,
                "maximum_equivalent_plastic_strain": 0.0012,
            }
        )
    record = ResultRecord(
        "result-1",
        "elastoplastic_single_pass",
        "format",
        str(tmp_path),
        str(summary),
        {},
        "2026-08-28T10:00:00+08:00",
        input_fingerprint="abcdef123456",
    )
    published = AnalysisPublishedResult(
        record.analysis_type,
        "弹塑性·完整单程",
        "format",
        tmp_path,
        summary,
        tmp_path / "unused.png",
        {},
        "最大位移：2.5 μm",
    )
    return CategorizedAnalysisResult(
        record,
        published,
        (
            ResultCategory("summary", "结果摘要", empty_message=published.summary_text),
            ResultCategory("history", "曲线与历程"),
            ResultCategory("fields", "云图与场量"),
            ResultCategory("mesh", "网格与变形"),
            ResultCategory(
                "data",
                "数据文件",
                (
                    ResultArtifactItem("结果摘要", summary, "data"),
                    ResultArtifactItem("单程历史数据", history, "data"),
                ),
            ),
        ),
    )


def test_export_origin_data_creates_engineering_unit_tables_without_touching_source(
    tmp_path: Path,
) -> None:
    result = _history_result(tmp_path)
    source = tmp_path / "pass_history.csv"
    source_before = source.read_bytes()
    destination = tmp_path / "origin_export"

    exported = export_origin_data(result, destination)

    assert source.read_bytes() == source_before
    assert exported.directory == destination.resolve()
    assert exported.table_paths == (destination / "pass_history_origin.csv",)
    with exported.table_paths[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["motion_coordinate_mm"] == "12"
    assert rows[0]["maximum_displacement_um"] == "2.5"
    assert rows[0]["maximum_von_mises_stress_MPa"] == "4.6"
    assert rows[0]["Fx_N"] == "80.0"
    assert (destination / "key_metrics.csv").is_file()
    guide = (destination / "Origin导入说明.txt").read_text(encoding="utf-8-sig")
    assert "X：motion_coordinate_mm" in guide
    assert "正式结果文件不会被修改" in guide


def test_export_origin_data_rejects_existing_destination_and_missing_csv(
    tmp_path: Path,
) -> None:
    result = _history_result(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(OriginExportError, match="已存在"):
        export_origin_data(result, destination)

    (tmp_path / "pass_history.csv").unlink()
    with pytest.raises(OriginExportError, match="不存在"):
        export_origin_data(result, tmp_path / "missing_source_export")


def test_export_origin_data_preserves_same_named_tables_from_multiple_snapshots(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "scan_summary.json"
    summary.write_text(
        '{"result_format":"scan","package_version":"2.13.7","artifacts":{}}',
        encoding="utf-8",
    )
    items = []
    for directory_name, x_value in (("point_0001_entry", "0.001"), ("point_0004_full", "0.004")):
        directory = tmp_path / "snapshots" / directory_name
        directory.mkdir(parents=True)
        nodes = directory / "nodes.csv"
        nodes.write_text(f"node_id,x_m,y_m\n0,{x_value},0.02\n", encoding="utf-8")
        items.append(ResultArtifactItem(directory_name, nodes, "data"))
    record = ResultRecord(
        "scan-result",
        "linear_elastic_single_pass",
        "format",
        str(tmp_path),
        str(summary),
        {},
        "2026-08-28T10:00:00+08:00",
        input_fingerprint="abcdef123456",
    )
    published = AnalysisPublishedResult(
        record.analysis_type,
        "线弹性·完整单程",
        "format",
        tmp_path,
        summary,
        tmp_path / "unused.png",
        {},
        "扫描结果",
    )
    result = CategorizedAnalysisResult(
        record,
        published,
        tuple(
            ResultCategory(category_id, category_id, tuple(items) if category_id == "data" else ())
            for category_id in ("summary", "history", "fields", "mesh", "data")
        ),
    )

    exported = export_origin_data(result, tmp_path / "origin-snapshots")

    assert tuple(path.name for path in exported.table_paths) == (
        "point_0001_entry_nodes_origin.csv",
        "point_0004_full_nodes_origin.csv",
    )
