"""Presentation-only names and plot titles for the stable engineering UI."""

from __future__ import annotations


ANALYSIS_MODE_LABELS_ZH = {
    "literature_elastoplastic_single_pass": "文献力驱动·弹塑性单程",
    "literature_grinding_force": "文献磨削力·材料去除与塑性堆积",
    "linear_elastic_single_position": "线弹性·单位置",
    "linear_elastic_single_pass": "线弹性·单程",
    "elastoplastic_single_position": "弹塑性·单位置",
    "elastoplastic_single_pass": "弹塑性·单程",
    "mechanism_elastoplastic_single_pass": "机制化磨削·单程",
    "single_grain_high_fidelity": "单磨粒真实接触",
}

LOAD_MODEL_LABELS_ZH = {
    "literature_strip_load": "文献逐磨粒力·周向条带保守投影",
    "literature_grain_force": "文献逐磨粒解析力模型",
    "uniform_empirical_load": "均匀经验载荷",
    "uniform_equivalent_moving_load": "均匀等效移动载荷",
    "statistical_mechanism_nonuniform_load": "统计等效磨粒非均匀载荷",
    "prescribed_rigid_grain_contact": "规定刚性磨粒真实接触",
}

MATERIAL_MODEL_LABELS_ZH = {
    "linear_elastic": "线弹性",
    "j2_elastoplastic": "J2 弹塑性延性金属",
}

RESULT_STATUS_LABELS_ZH = {
    "not_run": "尚未计算",
    "configured": "已配置，等待运行",
    "running": "正在计算",
    "completed": "已完成",
    "failed": "计算失败",
    "stale": "结果已过期",
    "unavailable": "结果记录不可用",
}

ANALYSIS_MODE_PARAMETER_KEYS = {
    "literature_elastoplastic_single_pass": (
        "scan_base_position_count", "loading_step_count", "unloading_step_count",
        "newton_residual_relative_tolerance", "newton_maximum_iterations",
        "history_base_substep_count", "history_final_unloading_substep_count",
    ),
    "literature_grinding_force": (),
    "linear_elastic_single_position": (),
    "linear_elastic_single_pass": ("scan_base_position_count",),
    "elastoplastic_single_position": (
        "loading_step_count",
        "unloading_step_count",
        "newton_residual_relative_tolerance",
        "newton_maximum_iterations",
    ),
    "elastoplastic_single_pass": (
        "scan_base_position_count",
        "loading_step_count",
        "unloading_step_count",
        "newton_residual_relative_tolerance",
        "newton_maximum_iterations",
        "history_base_substep_count",
        "history_final_unloading_substep_count",
    ),
    "mechanism_elastoplastic_single_pass": (
        "scan_base_position_count",
        "loading_step_count",
        "unloading_step_count",
        "newton_residual_relative_tolerance",
        "newton_maximum_iterations",
        "history_base_substep_count",
        "history_final_unloading_substep_count",
        "grain_minimum_group_count",
        "grain_maximum_group_count",
        "grain_random_seed",
        "load_point_count",
    ),
    "single_grain_high_fidelity": (
        "single_grain_workpiece_width_um",
        "single_grain_workpiece_height_um",
        "single_grain_mesh_size_um",
        "single_grain_thickness_mm",
        "single_grain_type",
        "single_grain_radius_um",
        "single_grain_arc_start_angle_deg",
        "single_grain_arc_end_angle_deg",
        "single_grain_tip_radius_um",
        "single_grain_wedge_height_um",
        "single_grain_intrinsic_rake_angle_deg",
        "single_grain_intrinsic_clearance_angle_deg",
        "single_grain_pose_angle_deg",
        "single_grain_nominal_direction",
        "single_grain_preset_removal_mode",
        "single_grain_preset_removal_element_ids",
        "single_grain_preset_removal_polygon_json",
        "single_grain_initial_clearance_um",
        "single_grain_indentation_depth_um",
        "single_grain_scratch_distance_um",
        "single_grain_normal_algorithm",
        "single_grain_penalty_factor",
        "single_grain_tangential_penalty_ratio",
        "single_grain_augmented_relaxation",
        "single_grain_augmented_maximum_iterations",
        "single_grain_friction_coefficient",
        "single_grain_friction_regularization_ratio",
        "single_grain_damage_enabled",
        "single_grain_damage_failure_table_json",
        "single_grain_damage_triaxiality_cutoff",
        "single_grain_damage_fracture_energy",
        "single_grain_damage_event_tolerance",
        "single_grain_damage_max_events",
        "single_grain_damage_area_tolerance_m2",
        "single_grain_damage_parameter_source",
        "single_grain_damage_calibration_status",
        "single_grain_damage_applicability_notes",
        "newton_residual_relative_tolerance",
        "newton_maximum_iterations",
    ),
}

PLOT_TITLES_EN = {
    "trajectory_surface_profile": "Ideal Single-Pass Surface Profile",
    "pass_load_snapshot": "Single-Pass Empirical Load Snapshot",
    "evolved_mesh": "Evolved Single-Pass Gmsh Mesh",
    "evolved_fem_mesh": "Evolved-Surface Empirical-Load FEM Mesh",
    "pass_scan": "Single-Pass Quasi-Static Scan",
    "j2_material_point": "J2 Material-Point Loading and Unloading",
    "fixed_mesh_j2_benchmark": "Fixed P1 Mesh with Prescribed Top Line Load",
    "fixed_mesh_j2_displacement": "Peak-Loaded Displacement Magnitude",
    "fixed_mesh_j2_von_mises": "Peak-Loaded von Mises Stress",
    "fixed_mesh_j2_plastic_strain": "Peak-Loaded Equivalent Plastic Strain",
    "single_position_surface_recovery": "Single-Position Elastoplastic Surface Response",
    "moving_load_overview": "Fixed-Mesh Complete Single-Pass Moving Load",
    "history_pass_force": "Complete-Pass Target-Force History",
    "history_pass_plastic": "Accumulated Plastic History",
    "history_pass_residual_state": "Final Fully Unloaded State | External Load = 0",
    "mechanism_fractions": "Ductile-Metal Grinding-Force Mechanism Fractions",
    "statistical_grain_distribution": "Statistical Equivalent Grain Load Distribution",
    "mechanism_history_comparison": "Statistical-Mechanism Complete-Pass Comparison",
}

PLOT_SUBTITLES_EN = {
    "fixed_mesh_j2_benchmark": "Plane-Strain J2 Structural Benchmark",
    "single_position_surface_recovery": "Relative Height for Display; CSV Coordinates Remain Unscaled",
    "moving_load_overview": "Exact Partial-Facet Overlap; Independent Load Positions",
    "history_pass_force": "Entry – Full Contact – Exit – Final Zero Load",
    "history_pass_plastic": "Committed Converged States; Final State Fully Unloaded",
    "history_pass_residual_state": "Fixed Mesh; Element-Constant Fields; Display-Only Deformation Scale",
    "mechanism_history_comparison": "Macro-Scale Conservative Projection; Final External Load = 0; Not Measured Roughness",
}

CLI_WORKFLOW_LABELS_EN = {
    "history_pass": "fixed-mesh complete-pass elastoplastic history analysis",
    "mechanism_history": "statistical-mechanism complete-pass comparison",
    "moving_load": "fixed-mesh complete single-pass moving-load analysis",
}


def analysis_mode_label(mode_id: str) -> str:
    return ANALYSIS_MODE_LABELS_ZH.get(mode_id, mode_id)


def load_model_label(load_model: str) -> str:
    return LOAD_MODEL_LABELS_ZH.get(load_model, load_model)


def analysis_mode_parameter_keys(mode_id: str) -> tuple[str, ...]:
    return ANALYSIS_MODE_PARAMETER_KEYS.get(mode_id, ())
