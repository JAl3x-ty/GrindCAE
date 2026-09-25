from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS
from grindcae.gui.analysis_results import AnalysisPublishedResult
from grindcae.gui.project import GrindCaeProject, ResultRecord
from grindcae.gui.result_catalog import (
    RESULT_CATEGORY_IDS,
    ResultCatalog,
    ResultCatalogError,
    formal_result_records,
    format_file_size,
)


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _record(tmp_path: Path, result_id: str, analysis_type: str, created_at: str) -> ResultRecord:
    output = tmp_path / result_id
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    return ResultRecord(
        result_id=result_id,
        analysis_type=analysis_type,
        result_format="test",
        output_directory=str(output),
        summary_path=str(summary),
        artifact_paths={},
        created_at=created_at,
        input_fingerprint=f"fingerprint-{result_id}",
    )


def test_formal_result_records_exclude_previews_and_sort_newest_first(tmp_path: Path) -> None:
    project = GrindCaeProject("结果目录")
    project.results = [
        _record(tmp_path, "old", "linear_elastic_single_position", "2026-08-23T09:00:00+08:00"),
        _record(tmp_path, "geometry", "geometry_preview", "2026-08-23T12:00:00+08:00"),
        _record(tmp_path, "new", "elastoplastic_single_position", "2026-08-23T11:00:00+08:00"),
        _record(tmp_path, "mesh", "mesh_preview", "2026-08-23T13:00:00+08:00"),
        _record(tmp_path, "grain", "process_grain_preview", "2026-08-23T14:00:00+08:00"),
    ]

    assert [record.result_id for record in formal_result_records(project)] == ["new", "old"]
    assert RESULT_CATEGORY_IDS == (
        "summary",
        "history",
        "fields",
        "mesh",
        "data",
    )


def _valid_summary(mode: str, output: Path, artifacts: dict[str, object]) -> dict[str, object]:
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    common = {"result_format": definition.result_format, "artifacts": artifacts}
    if mode == "linear_elastic_single_position":
        return {**common, "pass_load": {"current_empirical_force": {"tangential_force_magnitude_N": 1.0, "normal_force_magnitude_N": 2.0}}, "maximum_displacement": {"magnitude_m": 3e-6}, "stress_statistics": {"von_mises_area_weighted_p95_Pa": 4e6, "von_mises_area_weighted_p99_Pa": 5e6}, "force_tracking": {"force_balance_residual": {"norm_N": 1e-9}}}
    if mode == "linear_elastic_single_pass":
        return {**common, "scan_counts": {"total_positions": 5}, "force_extrema": {"maximum_current_resultant_force": {"wheel_lowest_point_x_m": 0.01}}, "response_extrema": {"maximum_displacement": {"value": 3e-6}, "maximum_von_mises_p99": {"value": 5e6}}, "validations": {"maximum_force_balance_residual_norm_N": 1e-9}}
    if mode == "elastoplastic_single_position":
        state = {"external_force_N": [0.0, 0.0], "statistics": {"maximum_displacement_m": 3e-6, "von_mises_area_weighted_p99_Pa": 5e6, "maximum_equivalent_plastic_strain": 0.01, "plastic_element_count": 2, "plastic_element_fraction": 0.2}}
        return {**common, "snapshots": {"peak_loaded": state, "residual_unloaded": state}, "newton_history": {"maximum_iterations_used": 4}}
    if mode == "elastoplastic_single_pass":
        return {**common, "pass": {"target_position_count": 5, "maximum_newton_iterations": 4}, "maximum_response": {"maximum_equivalent_plastic_strain": 0.01}, "final_unloaded": {"external_force_N": [0.0, 0.0], "maximum_residual_displacement_m": 3e-6, "von_mises_area_weighted_p99_Pa": 5e6, "plastic_element_count": 2, "plastic_element_fraction": 0.2}}
    if mode == "single_grain_high_fidelity":
        return {**common, "result_status": "completed", "workpiece_thickness_m": 1e-3, "scope": {"dimension": "2D", "kinematics": "small_strain_fixed_mesh", "physical_material_removal": False}, "material": {"calibration_status": "demonstration_not_calibrated"}, "peak": {"normal_reaction_N": 10.0, "normal_reaction_per_thickness_N_per_m": 10000.0, "tangential_reaction_N": -1.0, "tangential_reaction_per_thickness_N_per_m": -1000.0, "maximum_pressure_Pa": 2e9, "maximum_penetration_m": 1e-9}, "final_state": {"segment": "unloading", "contact_count": 0, "normal_reaction_N": 0.0, "tangential_reaction_N": 0.0}, "energy": {"contact_work_J": 1e-6, "friction_dissipation_J": 2e-7, "plastic_dissipation_J": 3e-7}, "applicability": {"status": "within_range", "maximum_displacement_gradient": 0.01, "maximum_absolute_principal_strain": 0.01, "warning_metrics": [], "hard_stop_reasons": []}, "maximum_balance_residual_N": 1e-9}
    return {**common, "phase7a2_prediction": {"resolved_grit_specification": {"resolution_path": "JIS_hash", "representative_diameter_d50_m": 3.4e-6}, "grain_population": {"equivalent_group_count": 16}}, "comparison": {"total_force_history_identical": True, "maximum_target_nodal_load_vector_l2_difference_N": 0.2}, "mechanism_final_unloaded": {"external_force_N": [0.0, 0.0], "maximum_equivalent_plastic_strain": 0.01, "maximum_residual_displacement_m": 3e-6, "maximum_residual_von_mises_stress_Pa": 5e6}, "resolution": {"resolution_status": "macro_scale_conservative_projection_not_grain_resolved"}}


def _published_record(tmp_path: Path, mode: str, filenames: list[str], *, summary_extra: dict[str, object] | None = None) -> ResultRecord:
    output = tmp_path / mode
    output.mkdir()
    artifacts: dict[str, object] = {}
    for filename in filenames:
        path = output / filename
        path.write_bytes(PNG if path.suffix == ".png" else b"data\n")
        artifacts[filename.replace(".", "_")] = str(path)
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    payload = _valid_summary(mode, output, artifacts)
    if summary_extra:
        payload.update(summary_extra)
    summary = output / definition.summary_filename
    summary.write_text(json.dumps(payload), encoding="utf-8")
    artifacts.setdefault("summary", str(summary))
    return ResultRecord("result", mode, definition.result_format, str(output), str(summary), {}, "2026-08-23T10:00:00+08:00", input_fingerprint="fp")


def test_catalog_classifies_linear_single_position_artifacts(tmp_path: Path) -> None:
    record = _published_record(tmp_path, "linear_elastic_single_position", [
        "stress_contour.png", "displacement_magnitude.png", "von_mises_stress.png",
        "principal_stress_1_in_plane.png", "strain_contour.png", "mesh.png",
        "deformed_shape.png", "evolved_surface.msh", "results.vtu", "nodes.csv",
        "elements.csv", "active_contact_facets.csv",
    ])

    result = ResultCatalog().read(record)

    assert [category.category_id for category in result.categories] == list(RESULT_CATEGORY_IDS)
    assert [item.display_name for item in result.category("fields").items] == [
        "位移幅值", "von Mises 等效应力", "第一面内主应力", "原生单元应力云图", "原生单元应变云图",
    ]
    assert [item.display_name for item in result.category("mesh").items] == ["网格", "变形形状"]
    assert {item.path.name for item in result.category("data").items} >= {
        "summary.json", "evolved_surface.msh", "results.vtu", "nodes.csv", "elements.csv", "active_contact_facets.csv",
    }
    assert "4C3B" not in result.published.summary_text


def test_catalog_mechanism_result_has_explicit_no_field_figure_message(tmp_path: Path) -> None:
    record = _published_record(tmp_path, "mechanism_elastoplastic_single_pass", [
        "mechanism_history_comparison.png", "comparison_history.csv", "mechanism_load_history.csv",
        "baseline_final_results.vtu", "mechanism_final_results.vtu", "final_nodes.csv",
        "baseline_final_elements.csv", "mechanism_final_elements.csv", "reference_mesh.msh",
    ])

    result = ResultCatalog().read(record)

    assert [item.display_name for item in result.category("history").items] == ["基准载荷与机制载荷历史对比"]
    assert result.category("fields").items == ()
    assert result.category("fields").empty_message.startswith("当前结果没有独立云图产物")
    assert len(result.category("data").items) == 9


@pytest.mark.parametrize(
    ("mode", "filenames", "expected"),
    (
        (
            "elastoplastic_single_position",
            ["surface_recovery.png", "peak_displacement.png", "peak_von_mises_stress.png", "equivalent_plastic_strain.png", "residual_displacement.png", "residual_von_mises_stress.png", "mesh.png", "increment_history.csv", "solve_input.msh", "results.vtu", "nodes.csv", "elements.csv", "active_contact_facets.csv", "surface_recovery.csv"],
            {"history": 1, "fields": 5, "mesh": 1, "data": 8},
        ),
        (
            "elastoplastic_single_pass",
            ["pass_force_history.png", "pass_plastic_history.png", "final_surface_profile.png", "pass_residual_state.png", "moving_load_overview.png", "pass_history.csv", "final_results.vtu", "final_nodes.csv", "final_elements.csv", "reference_mesh.msh"],
            {"history": 3, "fields": 1, "mesh": 1, "data": 6},
        ),
    ),
)
def test_catalog_classifies_elastoplastic_modes(tmp_path: Path, mode: str, filenames: list[str], expected: dict[str, int]) -> None:
    record = _published_record(tmp_path, mode, filenames)

    result = ResultCatalog().read(record)

    assert {category_id: len(result.category(category_id).items) for category_id in expected} == expected


def test_catalog_classifies_single_grain_contact_result(tmp_path: Path) -> None:
    filenames = [
        "reaction_history.png", "contact_pressure.png", "contact_state.png",
        "energy_history.png", "final_fields.png",
        "residual_groove.png", "contact_history.csv", "contact_points.csv",
        "final_results.vtu", "final_nodes.csv", "final_elements.csv",
        "residual_surface_profile.csv", "reference_mesh.msh",
    ]
    record = _published_record(tmp_path, "single_grain_high_fidelity", filenames)

    class ClassificationOnlyAdapter:
        def read(self, mode: str, output_directory: str | Path) -> AnalysisPublishedResult:
            output = Path(output_directory).resolve()
            summary_path = output / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return AnalysisPublishedResult(
                mode=mode,
                display_name=ANALYSIS_MODE_DEFINITIONS[mode].display_name,
                result_format=ANALYSIS_MODE_DEFINITIONS[mode].result_format,
                output_directory=output,
                summary_path=summary_path,
                representative_image=output / "reaction_history.png",
                summary=summary,
                summary_text="固定网格残余沟槽，不代表材料去除。",
            )

    result = ResultCatalog(adapter=ClassificationOnlyAdapter()).read(record)

    assert {item.path.name for item in result.category("history").items} == {
        "reaction_history.png", "energy_history.png", "residual_groove.png"
    }
    assert {item.path.name for item in result.category("fields").items} == {
        "contact_pressure.png", "contact_state.png", "final_fields.png"
    }
    assert "材料去除" in result.published.summary_text


def test_catalog_classifies_v2_topology_surface_and_grain_artifacts(tmp_path: Path) -> None:
    filenames = [
        "reaction_history.png", "contact_pressure.png", "contact_state.png",
        "energy_history.png", "final_fields.png", "unloaded_surface_profile.png",
        "grain_geometry.png", "effective_mesh_surface.png", "contact_history.csv",
        "contact_points.csv", "material_topology.csv", "topology_history.csv",
        "free_surface.csv", "final_results.vtu", "effective_mesh.vtu",
        "final_nodes.csv", "final_elements.csv", "unloaded_surface_profile.csv",
        "reference_mesh.msh",
    ]
    record = _published_record(tmp_path, "single_grain_high_fidelity", filenames)
    summary_path = Path(record.summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["result_format"] = "grindcae_single_grain_real_contact_v2"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    class ClassificationOnlyAdapter:
        def read(self, mode: str, output_directory: str | Path) -> AnalysisPublishedResult:
            output = Path(output_directory).resolve()
            return AnalysisPublishedResult(
                mode, ANALYSIS_MODE_DEFINITIONS[mode].display_name,
                "grindcae_single_grain_real_contact_v2", output, summary_path,
                output / "reaction_history.png", payload, "预设去除与动态自由表面。",
            )

    result = ResultCatalog(adapter=ClassificationOnlyAdapter()).read(record)

    assert {item.path.name for item in result.category("history").items} == {
        "reaction_history.png", "energy_history.png", "unloaded_surface_profile.png"
    }
    assert {item.path.name for item in result.category("mesh").items} == {
        "reference_mesh.msh", "grain_geometry.png", "effective_mesh_surface.png"
    }
    assert {item.path.name for item in result.category("data").items}.issuperset({
        "material_topology.csv", "topology_history.csv", "free_surface.csv", "effective_mesh.vtu"
    })


def test_catalog_classifies_v3_damage_separation_artifacts(tmp_path: Path) -> None:
    filenames = [
        "reaction_history.png", "contact_pressure.png", "contact_state.png",
        "energy_history.png", "final_fields.png", "unloaded_surface_profile.png",
        "grain_geometry.png", "effective_mesh_surface.png", "damage_evolution.png",
        "final_damage_field.png", "removal_area_history.png", "processed_surface.png",
        "contact_history.csv", "contact_points.csv", "material_topology.csv",
        "topology_history.csv", "free_surface.csv", "damage_history.csv",
        "damage_element_history.csv", "separation_events.csv", "final_damage_state.csv",
        "final_results.vtu", "effective_mesh.vtu", "final_nodes.csv",
        "final_elements.csv", "unloaded_surface_profile.csv", "reference_mesh.msh",
    ]
    record = _published_record(tmp_path, "single_grain_high_fidelity", filenames)
    summary_path = Path(record.summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["result_format"] = "grindcae_single_grain_damage_separation_v3"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    class ClassificationOnlyAdapter:
        def read(self, mode: str, output_directory: str | Path) -> AnalysisPublishedResult:
            output = Path(output_directory).resolve()
            return AnalysisPublishedResult(
                mode, ANALYSIS_MODE_DEFINITIONS[mode].display_name,
                "grindcae_single_grain_damage_separation_v3", output, summary_path,
                output / "reaction_history.png", payload,
                "模型预测的单磨粒损伤与材料分离。",
            )

    result = ResultCatalog(adapter=ClassificationOnlyAdapter()).read(record)

    assert {item.path.name for item in result.category("history").items} >= {
        "reaction_history.png", "energy_history.png", "damage_evolution.png",
        "removal_area_history.png", "processed_surface.png",
    }
    assert {item.path.name for item in result.category("fields").items} >= {
        "contact_pressure.png", "contact_state.png", "final_fields.png",
        "final_damage_field.png",
    }
    assert {item.path.name for item in result.category("data").items} >= {
        "damage_history.csv", "damage_element_history.csv", "separation_events.csv",
        "final_damage_state.csv", "effective_mesh.vtu",
    }


def test_catalog_rejects_artifact_path_that_escapes_output_directory(tmp_path: Path) -> None:
    record = _published_record(tmp_path, "linear_elastic_single_position", ["stress_contour.png"])
    summary = Path(record.summary_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["artifacts"]["stress_contour"] = "../outside.png"
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResultCatalogError, match="超出结果目录"):
        ResultCatalog().read(record)


def test_linear_pass_snapshot_roles_are_merged_by_real_directory(tmp_path: Path) -> None:
    output = tmp_path / "linear_elastic_single_pass"
    output.mkdir()
    snapshots = output / "snapshots" / "point_0003_full_contact"
    snapshots.mkdir(parents=True)
    for filename in ("stress_contour.png", "mesh.png", "results.vtu", "nodes.csv"):
        (snapshots / filename).write_bytes(PNG if filename.endswith(".png") else b"data\n")
    (output / "scan_overview.png").write_bytes(PNG)
    (output / "scan_history.csv").write_text("position\n1\n", encoding="utf-8")
    definition = ANALYSIS_MODE_DEFINITIONS["linear_elastic_single_pass"]
    payload = _valid_summary(
        "linear_elastic_single_pass",
        output,
        {
            "scan_overview_png": "scan_overview.png",
            "scan_history_csv": "scan_history.csv",
            "snapshots": "snapshots",
        },
    )
    payload["snapshot_roles"] = {
        "representative_full_contact": {"snapshot_directory": str(snapshots)},
        "maximum_von_mises_p99": {"snapshot_directory": str(snapshots)},
    }
    summary = output / definition.summary_filename
    summary.write_text(json.dumps(payload), encoding="utf-8")
    record = ResultRecord("scan", "linear_elastic_single_pass", definition.result_format, str(output), str(summary), {}, "2026-08-23T10:00:00+08:00", input_fingerprint="fp")

    result = ResultCatalog().read(record)

    stress_items = [item for item in result.category("fields").items if item.path.name == "stress_contour.png"]
    assert len(stress_items) == 1
    assert stress_items[0].display_name == "满接触代表位置／最大响应位置－原生单元应力云图"
    data_by_name = {item.path.name: item for item in result.category("data").items}
    assert set(data_by_name) == {
        "scan_summary.json", "scan_history.csv", "evolved_surface.msh",
        "results.vtu", "nodes.csv", "elements.csv", "active_contact_facets.csv",
    }
    assert data_by_name["results.vtu"].status == "可用"
    assert data_by_name["elements.csv"].status == "缺失"


def test_data_file_status_and_size_are_metadata_only(tmp_path: Path) -> None:
    record = _published_record(tmp_path, "mechanism_elastoplastic_single_pass", [
        "mechanism_history_comparison.png", "comparison_history.csv", "mechanism_load_history.csv",
        "baseline_final_results.vtu", "mechanism_final_results.vtu", "final_nodes.csv",
        "baseline_final_elements.csv", "mechanism_final_elements.csv", "reference_mesh.msh",
    ])
    result = ResultCatalog().read(record)
    data = result.category("data").items
    empty = next(item for item in data if item.path.name == "comparison_history.csv")
    empty.path.write_bytes(b"")
    missing = next(item for item in data if item.path.name == "reference_mesh.msh")
    missing.path.unlink()

    assert empty.status == "文件为空"
    assert missing.status == "缺失"
    assert format_file_size(1536) == "1.5 KB"
