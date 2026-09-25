from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS
import grindcae.gui.analysis_results as analysis_results
from grindcae.gui.analysis_results import AnalysisResultError, AnalysisResultAdapter


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def summary_for(mode: str, output: Path) -> dict[str, object]:
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    image = output / definition.representative_image_filename
    common = {"result_format": definition.result_format}
    if mode == "linear_elastic_single_position":
        return {**common, "pass_load": {"current_empirical_force": {"tangential_force_magnitude_N": 1.0, "normal_force_magnitude_N": 2.0}}, "maximum_displacement": {"magnitude_m": 3e-6}, "stress_statistics": {"von_mises_area_weighted_p95_Pa": 4e6, "von_mises_area_weighted_p99_Pa": 5e6}, "force_tracking": {"force_balance_residual": {"norm_N": 1e-9}}, "artifacts": {"stress_contour_png": str(image)}}
    if mode == "linear_elastic_single_pass":
        return {**common, "scan_counts": {"total_positions": 5}, "force_extrema": {"maximum_current_resultant_force": {"wheel_lowest_point_x_m": 0.01}}, "response_extrema": {"maximum_displacement": {"value": 3e-6}, "maximum_von_mises_p99": {"value": 5e6}}, "validations": {"maximum_force_balance_residual_norm_N": 1e-9}, "artifacts": {"scan_overview_png": str(image)}}
    if mode == "elastoplastic_single_position":
        state = {"external_force_N": [0.0, 0.0], "statistics": {"maximum_displacement_m": 3e-6, "von_mises_area_weighted_p99_Pa": 5e6, "maximum_equivalent_plastic_strain": 0.01, "plastic_element_count": 2, "plastic_element_fraction": 0.2}}
        return {**common, "snapshots": {"peak_loaded": state, "residual_unloaded": state}, "newton_history": {"maximum_iterations_used": 4}, "artifacts": {"surface_recovery_png": {"path": str(image)}}}
    if mode == "elastoplastic_single_pass":
        return {**common, "pass": {"target_position_count": 5, "maximum_newton_iterations": 4}, "maximum_response": {"maximum_equivalent_plastic_strain": 0.01, "maximum_plastic_element_count": 2}, "final_unloaded": {"external_force_N": [0.0, 0.0], "maximum_residual_displacement_m": 3e-6, "von_mises_area_weighted_p99_Pa": 5e6, "maximum_equivalent_plastic_strain": 0.01, "plastic_element_count": 2, "plastic_element_fraction": 0.2}, "artifacts": {"pass_residual_state_png": str(image)}}
    if mode == "single_grain_high_fidelity":
        return {**common, "result_status": "completed_with_applicability_warning", "workpiece_thickness_m": 1e-3, "scope": {"dimension": "2D", "kinematics": "small_strain_fixed_mesh", "physical_material_removal": False}, "material": {"calibration_status": "demonstration_not_calibrated"}, "peak": {"normal_reaction_N": 10.0, "normal_reaction_per_thickness_N_per_m": 10000.0, "tangential_reaction_N": -1.0, "tangential_reaction_per_thickness_N_per_m": -1000.0, "maximum_pressure_Pa": 2e9, "maximum_penetration_m": 1e-9}, "final_state": {"segment": "unloading", "contact_count": 0, "normal_reaction_N": 0.0, "tangential_reaction_N": 0.0}, "energy": {"contact_work_J": 1e-6, "friction_dissipation_J": 2e-7, "plastic_dissipation_J": 3e-7, "energy_balance_residual_J": 5e-9, "energy_balance_relative_residual": 0.005}, "applicability": {"status": "warning", "maximum_displacement_to_contact_size": 0.12, "maximum_displacement_to_grain_radius": 0.03, "maximum_displacement_gradient": 0.11, "maximum_absolute_principal_strain": 0.06, "maximum_equivalent_total_strain": 0.05, "maximum_local_rotation_rad": 0.01, "minimum_area_ratio": 0.95, "minimum_normalized_jacobian_ratio": 0.95, "surface_self_intersection": False, "closest_point_unique": True, "candidate_boundary_order_preserved": True, "warning_metrics": ["maximum_displacement_gradient"], "hard_stop_reasons": []}, "maximum_balance_residual_N": 1e-9, "artifacts": {"reaction_history_png": str(image)}}
    return {**common, "phase7a2_prediction": {"resolved_grit_specification": {"resolution_path": "JIS_hash", "representative_diameter_d50_m": 3.4e-6}, "grain_population": {"equivalent_group_count": 16}}, "comparison": {"total_force_history_identical": True, "maximum_target_nodal_load_vector_l2_difference_N": 0.2}, "mechanism_final_unloaded": {"external_force_N": [0.0, 0.0], "maximum_equivalent_plastic_strain": 0.01, "maximum_residual_displacement_m": 3e-6, "maximum_residual_von_mises_stress_Pa": 5e6}, "resolution": {"resolution_status": "macro_scale_conservative_projection_not_grain_resolved"}, "artifacts": {"mechanism_history_comparison_png": str(image)}}


@pytest.mark.parametrize("mode", tuple(m for m in ANALYSIS_MODE_DEFINITIONS if m not in {"literature_grinding_force", "literature_elastoplastic_single_pass"}))
def test_adapter_reads_each_registered_real_format_and_representative_png(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / f"结果 空格 {mode}"
    output.mkdir()
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    (output / definition.representative_image_filename).write_bytes(PNG)
    (output / definition.summary_filename).write_text(json.dumps(summary_for(mode, output), allow_nan=False), encoding="utf-8")
    if mode == "single_grain_high_fidelity":
        monkeypatch.setattr(
            analysis_results,
            "read_contact_summary",
            lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
            raising=False,
        )
    result = AnalysisResultAdapter().read(mode, output)
    assert result.result_format == definition.result_format
    assert result.representative_image.name == definition.representative_image_filename
    assert result.summary_text
    if mode == "single_grain_high_fidelity":
        assert "适用性警告" in result.summary_text
        assert "位移梯度" in result.summary_text
        assert "能量平衡残差" in result.summary_text


def test_single_grain_adapter_delegates_to_strict_formal_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = "single_grain_high_fidelity"
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    output = tmp_path / "strict-single-grain"
    output.mkdir()
    image = output / definition.representative_image_filename
    image.write_bytes(PNG)
    summary = output / definition.summary_filename
    summary.write_text(json.dumps(summary_for(mode, output)), encoding="utf-8")
    calls: list[Path] = []

    def rejecting_reader(path: str | Path):
        calls.append(Path(path))
        raise ValueError("strict artifact validation failed")

    monkeypatch.setattr(
        analysis_results, "read_contact_summary", rejecting_reader, raising=False
    )

    with pytest.raises(AnalysisResultError, match="strict artifact validation failed"):
        AnalysisResultAdapter().read(mode, output)
    assert calls == [summary]


def test_single_grain_adapter_dispatches_v2_to_the_independent_strict_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = "single_grain_high_fidelity"
    output = tmp_path / "strict-single-grain-v2"; output.mkdir()
    image = output / "reaction_history.png"; image.write_bytes(PNG)
    payload = summary_for(mode, output)
    payload["result_format"] = "grindcae_single_grain_real_contact_v2"
    payload["topology"] = {
        "active_element_count": 12, "removed_element_count": 4,
        "active_node_count": 14, "active_degrees_of_freedom": 28,
        "free_surface_edge_count": 6, "free_surface_length_m": 0.0015,
    }
    payload["grain_geometry"] = {"type": "rounded_wedge"}
    summary = output / "summary.json"; summary.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[Path] = []

    def reader(path: str | Path):
        calls.append(Path(path)); return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(analysis_results, "read_contact_summary_v2", reader, raising=False)
    monkeypatch.setattr(analysis_results, "read_contact_summary", lambda path: (_ for _ in ()).throw(AssertionError("v1 reader called")), raising=False)

    result = AnalysisResultAdapter().read(mode, output)

    assert calls == [summary]
    assert result.result_format == "grindcae_single_grain_real_contact_v2"
    assert "预设去除" in result.summary_text
    assert "rounded_wedge" in result.summary_text


def test_single_grain_adapter_dispatches_v3_and_reports_damage_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = "single_grain_high_fidelity"
    output = tmp_path / "strict-single-grain-v3"; output.mkdir()
    image = output / "reaction_history.png"; image.write_bytes(PNG)
    payload = summary_for(mode, output)
    payload.update({
        "result_format": "grindcae_single_grain_damage_separation_v3",
        "automatic_damage_evolution": True,
        "automatic_material_separation": True,
        "material_removal_interpretation": "model_predicted_unvalidated",
        "chip_formation_prediction": False,
        "scope": {
            "dimension": "2D longitudinal section", "kinematics": "small_strain",
            "background_mesh": "fixed", "free_chip_motion": False,
            "independent_physical_validation": False,
        },
        "topology": {"active_element_count": 11, "removed_element_count": 1,
                     "active_degrees_of_freedom": 28, "free_surface_edge_count": 5},
        "separation": {"event_count": 1, "automatic_removed_area_m2": 3.125e-8},
        "damage_model": {"parameter_source": "synthetic development baseline",
                         "calibration_status": "uncalibrated"},
        "energy": {"contact_work_J": 1e-6, "friction_dissipation_J": 0.0,
                   "plastic_dissipation_J": 2e-7, "damage_dissipation_J": 3e-7,
                   "separation_event_energy_J": 2e-7, "balance_residual_J": 1e-9,
                   "balance_relative_residual": 1e-3},
    })
    summary = output / "summary.json"; summary.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[Path] = []

    def reader(path: str | Path):
        calls.append(Path(path)); return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(analysis_results, "read_contact_summary_v3", reader, raising=False)

    result = AnalysisResultAdapter().read(mode, output)

    assert calls == [summary]
    assert "自动分离事件：1" in result.summary_text
    assert "模型预测、尚未独立验证" in result.summary_text
    assert "不预测游离切屑" in result.summary_text


def test_adapter_rejects_wrong_format_missing_or_corrupt_png(tmp_path: Path) -> None:
    mode = "elastoplastic_single_position"
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    output = tmp_path / "bad"
    output.mkdir()
    payload = summary_for(mode, output)
    payload["result_format"] = "wrong"
    (output / definition.summary_filename).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnalysisResultError, match="result_format"):
        AnalysisResultAdapter().read(mode, output)
    payload["result_format"] = definition.result_format
    (output / definition.summary_filename).write_text(json.dumps(payload), encoding="utf-8")
    (output / definition.representative_image_filename).write_bytes(b"not-png")
    with pytest.raises(AnalysisResultError, match="PNG"):
        AnalysisResultAdapter().read(mode, output)
