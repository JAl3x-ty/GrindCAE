from __future__ import annotations

import copy
from dataclasses import replace
import math

import pytest

from grindcae.evolved_fem import EvolvedFemCase
from grindcae.history_pass import FixedMeshElastoplasticHistoryPassCase
from grindcae.mechanism_history_pass import MechanismHistoryPassCase
from grindcae.pass_scan import PassScanCase
from grindcae.single_position_elastoplastic import SinglePositionElastoplasticCase
from grindcae.single_grain_contact import SingleGrainContactCase
from grindcae.gui.analysis import AnalysisInputBuilder, AnalysisSettings
from grindcae.gui.geometry import GeometryDisplayInput, Parametric2DGeometryProvider
from grindcae.gui.material_workspace import MaterialDisplayInput
from grindcae.gui.mesh_workspace import MeshDisplayInput
from grindcae.gui.process_workspace import ProcessDisplayInput, WheelDisplayInput
from grindcae.gui.project import ProjectNodeStatus, create_project


def configured_project(*, behavior: str = "elastoplastic", grit_path: str = "JIS_hash", grit: str = "#5000"):
    project = create_project("分析适配测试")
    geometry = Parametric2DGeometryProvider(GeometryDisplayInput().build_case()).to_geometry_schema()
    mesh = MeshDisplayInput(target_size_mm="4").build_case().to_dict()
    mesh["status"] = "completed"
    material = MaterialDisplayInput(behavior=behavior).build_case().to_dict()
    wheel_case = WheelDisplayInput(resolution_path=grit_path, grit_designation=grit).build_case()
    process = ProcessDisplayInput(
        wheel_surface_speed_m_per_s="31",
        workpiece_feed_speed_m_per_s="0.12",
        grinding_width_mm="9",
        specific_grinding_energy_j_per_mm3="4321",
        normal_to_tangential_force_ratio="2.7",
    ).build_case(geometry, wheel_case).to_dict()
    project.geometry = geometry
    project.mesh = mesh
    project.material = material
    project.wheel = wheel_case.to_dict()
    project.process = process
    project.set_node_status("几何", ProjectNodeStatus.COMPLETED)
    project.set_node_status("网格", ProjectNodeStatus.READY)
    project.set_node_status("网格", ProjectNodeStatus.COMPLETED)
    project.set_node_status("材料", ProjectNodeStatus.READY)
    project.set_node_status("材料", ProjectNodeStatus.COMPLETED)
    project.set_node_status("工艺", ProjectNodeStatus.READY)
    project.set_node_status("工艺", ProjectNodeStatus.COMPLETED)
    return project


@pytest.mark.parametrize(
    ("mode_id", "expected_type"),
    [
        ("linear_elastic_single_position", EvolvedFemCase),
        ("linear_elastic_single_pass", PassScanCase),
        ("elastoplastic_single_position", SinglePositionElastoplasticCase),
        ("elastoplastic_single_pass", FixedMeshElastoplasticHistoryPassCase),
        ("mechanism_elastoplastic_single_pass", MechanismHistoryPassCase),
        ("single_grain_high_fidelity", SingleGrainContactCase),
    ],
)
def test_each_mode_builds_its_existing_strict_schema(mode_id: str, expected_type: type) -> None:
    built = AnalysisInputBuilder().build(configured_project(), mode_id, AnalysisSettings.recommended())
    assert isinstance(built.case, expected_type)
    assert built.normalized_input == built.case.to_dict()
    assert len(built.input_fingerprint) == 64


def test_solver_mapping_uses_current_engineering_sources_and_not_example_constants() -> None:
    built = AnalysisInputBuilder().build(
        configured_project(), "elastoplastic_single_position", AnalysisSettings.recommended()
    )
    data = built.normalized_input
    trajectory = data["pass_load"]["trajectory"]
    force_model = data["pass_load"]["force_model"]
    force_process = force_model["process"]
    force_calibration = force_model["calibration"]
    assert trajectory["wheel"]["diameter_m"] == pytest.approx(0.2)
    assert trajectory["single_pass"]["depth_of_cut_m"] == pytest.approx(20e-6)
    assert trajectory["single_pass"]["wheel_lowest_point_x_m"] == pytest.approx(0.05)
    assert force_process["wheel_surface_speed_m_per_s"] == pytest.approx(31)
    assert force_process["workpiece_feed_speed_m_per_s"] == pytest.approx(0.12)
    assert force_process["grinding_width_m"] == pytest.approx(0.009)
    assert force_calibration["specific_grinding_energy_J_per_m3"] == pytest.approx(4.321e12)
    assert force_calibration["normal_to_tangential_force_ratio"] == pytest.approx(2.7)
    assert data["mesh"]["target_size"] == pytest.approx(0.004)
    assert data["analysis"]["thickness"] == pytest.approx(0.009)


def test_single_grain_mode_uses_independent_contact_geometry_and_current_material() -> None:
    settings = AnalysisSettings.recommended()
    built = AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", settings
    )
    data = built.normalized_input

    assert data["geometry"]["width"] == pytest.approx(400e-6)
    assert data["geometry"]["height"] == pytest.approx(200e-6)
    assert data["grain"]["radius_m"] == pytest.approx(200e-6)
    assert data["analysis"]["thickness"] == pytest.approx(1e-3)
    assert data["material"]["E"] == pytest.approx(210e9)
    assert data["material"]["calibration_status"] == "demonstration_not_calibrated"
    assert any(item["segment"] == "scratch" for item in data["trajectory"]["targets"])


def test_single_grain_mode_builds_parameterized_arc_from_gui_settings() -> None:
    defaults = AnalysisSettings.recommended()
    settings = replace(
        defaults,
        single_grain=replace(
            defaults.single_grain,
            grain_type="rigid_circular_arc",
            grain_arc_start_angle_rad=1.20 * math.pi,
            grain_arc_end_angle_rad=1.80 * math.pi,
        ),
    )

    built = AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", settings
    )

    assert built.normalized_input["grain"]["type"] == "rigid_circular_arc"
    assert built.normalized_input["grain"]["start_angle_rad"] == pytest.approx(1.20 * math.pi)
    assert built.normalized_input["grain"]["end_angle_rad"] == pytest.approx(1.80 * math.pi)


def test_single_grain_gui_settings_build_schema_2_rounded_wedge_and_preset_removal() -> None:
    defaults = AnalysisSettings.recommended()
    settings = replace(
        defaults,
        single_grain=replace(
            defaults.single_grain,
            grain_type="rounded_wedge",
            grain_tip_radius_m=20e-6,
            grain_wedge_height_m=80e-6,
            intrinsic_rake_angle_rad=math.radians(10.0),
            intrinsic_clearance_angle_rad=math.radians(35.0),
            grain_pose_angle_rad=math.radians(5.0),
            initial_clearance_m=1e-6,
            indentation_depth_m=2e-6,
            nominal_scratch_direction="negative_x",
            preset_removal_mode="element_ids",
            preset_removal_element_ids=(2, 5, 9),
        ),
    )

    built = AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", settings
    )
    data = built.normalized_input

    assert data["single_grain_contact_schema_version"] == 2
    assert data["grain"]["type"] == "rounded_wedge"
    assert data["grain"]["tip_radius_m"] == pytest.approx(20e-6)
    assert data["grain"]["wedge_height_m"] == pytest.approx(80e-6)
    assert data["trajectory"]["nominal_scratch_direction"] == "negative_x"
    assert data["trajectory"]["targets"][0]["reference_x_m"] == pytest.approx(200e-6)
    assert data["trajectory"]["targets"][0]["reference_y_m"] == pytest.approx(201e-6)
    assert data["trajectory"]["targets"][1]["reference_y_m"] == pytest.approx(198e-6)
    assert data["material_topology"] == {
        "preset_removal": {"mode": "element_ids", "element_ids": [2, 5, 9]}
    }


def test_single_grain_gui_rounded_circle_keeps_center_based_vertical_trajectory() -> None:
    defaults = AnalysisSettings.recommended()
    settings = replace(
        defaults,
        single_grain=replace(
            defaults.single_grain,
            grain_type="rounded_circle",
            grain_radius_m=20e-6,
            initial_clearance_m=1e-6,
            indentation_depth_m=2e-6,
            preset_removal_mode="element_ids",
            preset_removal_element_ids=(),
        ),
    )

    built = AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", settings
    )
    targets = built.normalized_input["trajectory"]["targets"]

    assert targets[0]["reference_y_m"] == pytest.approx(221e-6)
    assert targets[1]["reference_y_m"] == pytest.approx(218e-6)


def test_single_grain_gui_settings_round_trip_preset_polygon() -> None:
    defaults = AnalysisSettings.recommended().single_grain.to_dict()
    defaults.update(
        grain_type="rounded_circle",
        preset_removal_mode="geometric_region",
        preset_removal_polygon_vertices_m=((100e-6, 200e-6), (300e-6, 200e-6), (250e-6, 150e-6)),
    )

    restored = type(AnalysisSettings.recommended().single_grain).from_mapping(defaults)

    assert restored.preset_removal_polygon_vertices_m == (
        (100e-6, 200e-6), (300e-6, 200e-6), (250e-6, 150e-6)
    )
    assert restored.to_dict() == defaults


def test_single_grain_gui_explicit_damage_settings_build_schema_3_and_change_fingerprint() -> None:
    defaults = AnalysisSettings.recommended()
    damage_settings = replace(
        defaults,
        single_grain=replace(
            defaults.single_grain,
            grain_type="rounded_wedge",
            damage_separation_enabled=True,
            damage_failure_strain_table=((-1.0, 0.25), (0.0, 0.18), (1.0, 0.12)),
            damage_triaxiality_cutoff=-0.5,
            damage_fracture_energy_J_per_m2=1200.0,
            damage_event_tolerance=1.0e-6,
            damage_maximum_separation_events_per_position=25,
            damage_area_absolute_tolerance_m2=1.0e-16,
            damage_parameter_source="用户试验开发参数",
            damage_calibration_status="preliminary",
            damage_applicability_notes="二维小应变固定背景网格开发范围。",
        ),
    )

    built = AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", damage_settings
    )
    data = built.normalized_input

    assert data["single_grain_contact_schema_version"] == 3
    assert data["damage"] == {
        "model": "ductile_triaxiality_table",
        "failure_strain_table": [[-1.0, 0.25], [0.0, 0.18], [1.0, 0.12]],
        "triaxiality_cutoff": -0.5,
        "evolution": "linear_fracture_energy_softening",
        "fracture_energy_J_per_m2": 1200.0,
        "characteristic_length": "equivalent_circle_diameter_reference_area",
        "damage_event_tolerance": 1.0e-6,
        "maximum_separation_events_per_position": 25,
        "area_absolute_tolerance_m2": 1.0e-16,
        "parameter_source": "用户试验开发参数",
        "calibration_status": "preliminary",
        "applicability_notes": "二维小应变固定背景网格开发范围。",
    }
    changed = replace(
        damage_settings,
        single_grain=replace(
            damage_settings.single_grain,
            damage_fracture_energy_J_per_m2=1300.0,
        ),
    )
    assert built.input_fingerprint != AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", changed
    ).input_fingerprint


def test_single_grain_gui_does_not_enable_schema_3_implicitly() -> None:
    defaults = AnalysisSettings.recommended()
    rounded = replace(
        defaults,
        single_grain=replace(defaults.single_grain, grain_type="rounded_circle"),
    )

    built = AnalysisInputBuilder().build(
        configured_project(), "single_grain_high_fidelity", rounded
    )

    assert built.normalized_input["single_grain_contact_schema_version"] == 2
    assert "damage" not in built.normalized_input


def test_linear_mode_uses_only_elastic_material_fields_even_for_plastic_material() -> None:
    built = AnalysisInputBuilder().build(
        configured_project(), "linear_elastic_single_position", AnalysisSettings.recommended()
    )
    assert set(built.normalized_input["material"]) == {"E", "nu"}


def test_elastoplastic_and_mechanism_eligibility_reports_specific_chinese_reasons() -> None:
    builder = AnalysisInputBuilder()
    linear_project = configured_project(behavior="linear_elastic")
    with pytest.raises(ValueError, match="屈服强度和切线模量"):
        builder.build(linear_project, "elastoplastic_single_position", AnalysisSettings.recommended())

    no_grit = configured_project()
    no_grit.wheel = copy.deepcopy(no_grit.wheel)
    no_grit.wheel["resolved_specification"] = None
    with pytest.raises(ValueError, match="粒度解析"):
        builder.build(no_grit, "mechanism_elastoplastic_single_pass", AnalysisSettings.recommended())


def test_input_fingerprint_is_stable_and_changes_with_physical_input() -> None:
    builder = AnalysisInputBuilder()
    first_project = configured_project()
    first = builder.build(first_project, "linear_elastic_single_pass", AnalysisSettings.recommended())
    same = builder.build(first_project, "linear_elastic_single_pass", AnalysisSettings.recommended())
    changed_project = configured_project()
    changed_project.process["workpiece_feed_speed_m_per_s"] = 0.2
    changed = builder.build(changed_project, "linear_elastic_single_pass", AnalysisSettings.recommended())
    assert first.input_fingerprint == same.input_fingerprint
    assert first.input_fingerprint != changed.input_fingerprint


@pytest.mark.parametrize(("path", "designation"), (("JIS_hash", "#5000"), ("GB_W_micropowder", "W3.5")))
def test_mechanism_mode_accepts_supported_current_workbench_grit_paths(path: str, designation: str) -> None:
    built = AnalysisInputBuilder().build(
        configured_project(grit_path=path, grit=designation),
        "mechanism_elastoplastic_single_pass", AnalysisSettings.recommended(),
    )
    wheel = built.normalized_input["statistical_grain_load"]["wheel_specification"]
    assert wheel["resolution_path"] == path
    assert wheel["grit_designation"] == designation


def test_mechanism_mode_accepts_complete_manufacturer_override() -> None:
    project = configured_project()
    project.wheel = copy.deepcopy(project.wheel)
    project.wheel.update({
        "resolution_path": "manufacturer_override", "grit_designation": "CUSTOM-5",
        "resolved_specification": {
            "manufacturer": "用户厂家", "product_model": "CUSTOM-5",
            "diameter_lower_m": 4e-6, "representative_diameter_d50_m": 5e-6,
            "diameter_upper_m": 6e-6, "range_definition": "manufacturer_reported_range",
            "data_status": "manufacturer_input_unverified", "source": "用户数据表",
        },
    })
    built = AnalysisInputBuilder().build(project, "mechanism_elastoplastic_single_pass", AnalysisSettings.recommended())
    assert built.normalized_input["statistical_grain_load"]["wheel_specification"]["manufacturer"] == "用户厂家"


def test_direction_mapping_uses_geometry_for_trajectory_and_tangential_force() -> None:
    project = configured_project()
    project.geometry = copy.deepcopy(project.geometry)
    project.geometry["process"]["relative_feed_direction"] = "negative_x"
    built = AnalysisInputBuilder().build(project, "linear_elastic_single_position", AnalysisSettings.recommended())
    pass_load = built.normalized_input["pass_load"]
    assert pass_load["trajectory"]["single_pass"]["relative_feed_direction"] == "negative_x"
    assert pass_load["force_mapping"]["tangential_force_direction"] == "negative_x"


def test_stale_upstream_node_is_rejected_before_solver_input_build() -> None:
    project = configured_project()
    project.set_node_status("网格", ProjectNodeStatus.STALE, "网格过期")
    with pytest.raises(ValueError, match="网格"):
        AnalysisInputBuilder().build(project, "linear_elastic_single_position", AnalysisSettings.recommended())
