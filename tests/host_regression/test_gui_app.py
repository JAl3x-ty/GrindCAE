from pathlib import Path
import json
import math
import time
import tkinter as tk
from tkinter import ttk

import pytest

from grindcae.gui import app
from grindcae.gui.analysis import AnalysisSettings
from grindcae.gui.analysis_runner import AnalysisEvent
from grindcae.gui.help_text import FORCE_MODEL_HELP_TEXT, MECHANISM_HISTORY_HELP_TEXT
from grindcae.gui.results import GuiImage
from grindcae.gui.geometry import GeometryDisplayInput, Parametric2DGeometryProvider
from grindcae.gui.project import ProjectNodeStatus, create_project
from grindcae.gui.image_preview import ImagePreview
from grindcae.gui.material_workspace import MaterialDisplayInput
from grindcae.gui.mesh_workspace import MeshDisplayInput
from grindcae.gui.workbench import WorkbenchSession


APP_PATH = Path(__file__).parents[2] / "src" / "grindcae" / "gui" / "app.py"


def test_initial_window_geometry_fits_scaled_1080p_desktops() -> None:
    assert app.initial_window_geometry(1920, 1080) == (1420, 900, 1120, 720)
    assert app.initial_window_geometry(1536, 864) == (1420, 744, 1120, 720)
    assert app.initial_window_geometry(1280, 720) == (1200, 600, 1120, 600)


def test_dynamic_status_labels_use_the_cjk_gui_font() -> None:
    """Keep worker-event Chinese text out of the non-CJK default ttk font."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'GUI_FONT_FAMILY = "Microsoft YaHei UI"' in source
    assert 'style.configure("Status.TLabel", font=(GUI_FONT_FAMILY, 10)' in source
    assert 'style.configure("Stage.TLabel", font=(GUI_FONT_FAMILY, 10))' in source
    assert 'textvariable=self.stage_var, style="Stage.TLabel"' in source


def test_engineering_terms_and_read_only_formula_help_are_present() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    for label in (
        "工艺参数与经验磨削力估算",
        "砂轮圆周线速度 vs [m/s]",
        "工件相对进给速度 vw [m/s]",
        "磨削宽度 b（同时作为二维分析厚度）[mm]",
        "比磨削能 us [J/mm^3]（需实验或文献标定）",
        "磨削力比 Rnt = Fn/Ft（需标定）",
        "经验参数集 ID（内部标识）",
        "参数集名称：",
        "us 数据来源或标定依据",
        "Rnt 数据来源或标定依据",
        "参数适用工况与限制",
        "指定/参考位置：砂轮最低点 x [mm]",
        "基础扫描位置数 [5-101]（关键位置会自动插入）",
        "查看经验模型公式与参数说明",
    ):
        assert label in source
    assert 'text.configure(state="disabled")' in source

    for formula in (
        "Qw = vw * ae * b",
        "P = us * Qw",
        "Ft,ss = P / vs",
        "Fn,ss = Rnt * Ft,ss",
        "Fr = sqrt(Ft,ss^2 + Fn,ss^2)",
        "Larc = sqrt(Ds * ae - ae^2)",
        "alpha = Leff / Larc",
        "Ft = alpha * Ft,ss",
        "Fn = alpha * Fn,ss",
    ):
        assert formula in FORCE_MODEL_HELP_TEXT
    assert "本窗口只展示说明文字，不参与计算" in FORCE_MODEL_HELP_TEXT


def test_mechanism_json_mode_help_keeps_parameter_form_and_portable_work_deferred() -> None:
    for text in (
            "机制化磨削·单程",
        "MechanismHistoryPassCase",
        "run_mechanism_elastoplastic_history_pass",
        "后续版本",
        "宏观尺度保守投影",
        "Windows 便携包尚未重建",
    ):
        assert text in MECHANISM_HISTORY_HELP_TEXT


def test_user_facing_mode_text_uses_chinese_names_without_internal_phase_numbers() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    for label in (
        "单位置线弹性计算",
        "完整单程线弹性扫描",
        "机制化弹塑性完整单程",
    ):
        assert label in source
    for legacy_label in (
        "单位置计算（4C3B）",
        "完整单程扫描（4C4）",
        "机制化弹塑性完整单程（7A.3）",
        "Phase 8A.1",
        "Phase 8A.2",
    ):
        assert legacy_label not in source
        assert legacy_label not in MECHANISM_HISTORY_HELP_TEXT


def test_main_window_builds_the_chinese_work_tree_and_retains_legacy_page() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        rows = tuple(
            (
                application.work_tree.item(item, "text"),
                application.work_tree.item(item, "values")[0],
            )
            for item in application.work_tree.get_children("")
        )
        assert rows == (
            ("工程", "可执行"),
            ("几何", "未配置"),
            ("网格", "暂未启用"),
            ("材料", "暂未启用"),
            ("工艺", "未配置"),
            ("分析", "未配置"),
            ("结果", "未配置"),
        )
        assert application.main_notebook.tab(application.workbench_tab, "text") == "工程工作台"
        assert application.main_notebook.tab(application.legacy_tab, "text") == "高级兼容计算"
        assert application.project_name_var.get() == "未命名工程"
    finally:
        root.destroy()


def test_single_grain_advanced_settings_are_editable_and_round_trip_in_si() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        expected_keys = {
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
        }
        assert expected_keys <= set(application.analysis_settings_vars)
        assert expected_keys <= set(application.analysis_advanced_entries)

        application.analysis_settings_vars["single_grain_workpiece_width_um"].set("500")
        application.analysis_settings_vars["single_grain_thickness_mm"].set("2")
        application.analysis_settings_vars["single_grain_type"].set(
            "rigid_circular_arc"
        )
        application.analysis_settings_vars["single_grain_arc_start_angle_deg"].set("210")
        application.analysis_settings_vars["single_grain_arc_end_angle_deg"].set("330")
        application.analysis_settings_vars["single_grain_normal_algorithm"].set(
            "augmented_lagrangian"
        )
        application.analysis_settings_vars[
            "single_grain_augmented_maximum_iterations"
        ].set("12")
        settings = application._analysis_settings_from_widgets()

        assert settings.single_grain.workpiece_width_m == pytest.approx(500e-6)
        assert settings.single_grain.workpiece_thickness_m == pytest.approx(2e-3)
        assert settings.single_grain.grain_type == "rigid_circular_arc"
        assert settings.single_grain.grain_arc_start_angle_rad == pytest.approx(7 * math.pi / 6)
        assert settings.single_grain.grain_arc_end_angle_rad == pytest.approx(11 * math.pi / 6)
        assert settings.single_grain.normal_algorithm == "augmented_lagrangian"
        assert settings.single_grain.augmented_maximum_iterations == 12

        application.analysis_settings_vars["single_grain_type"].set("rounded_wedge")
        application.analysis_settings_vars["single_grain_tip_radius_um"].set("20")
        application.analysis_settings_vars["single_grain_wedge_height_um"].set("80")
        application.analysis_settings_vars["single_grain_intrinsic_rake_angle_deg"].set("10")
        application.analysis_settings_vars["single_grain_intrinsic_clearance_angle_deg"].set("35")
        application.analysis_settings_vars["single_grain_pose_angle_deg"].set("5")
        application.analysis_settings_vars["single_grain_nominal_direction"].set("negative_x")
        application.analysis_settings_vars["single_grain_preset_removal_mode"].set("element_ids")
        application.analysis_settings_vars["single_grain_preset_removal_element_ids"].set("2,5,9")
        settings = application._analysis_settings_from_widgets()
        assert settings.single_grain.grain_tip_radius_m == pytest.approx(20e-6)
        assert settings.single_grain.preset_removal_element_ids == (2, 5, 9)
        application.analysis_settings_vars["single_grain_damage_enabled"].set("true")
        application.analysis_settings_vars["single_grain_damage_failure_table_json"].set(
            "[[-1.0, 0.25], [1.0, 0.12]]"
        )
        application.analysis_settings_vars["single_grain_damage_fracture_energy"].set("1200")
        application.analysis_settings_vars["single_grain_damage_parameter_source"].set("用户试验开发参数")
        settings = application._analysis_settings_from_widgets()
        assert settings.single_grain.damage_separation_enabled is True
        assert settings.single_grain.damage_failure_strain_table == ((-1.0, 0.25), (1.0, 0.12))
        assert settings.single_grain.damage_fracture_energy_J_per_m2 == pytest.approx(1200.0)

        application._apply_analysis_settings(AnalysisSettings.recommended())
        assert application.analysis_settings_vars[
            "single_grain_workpiece_width_um"
        ].get() == "400.0"
        assert application.analysis_settings_vars["single_grain_thickness_mm"].get() == "1.0"
        assert application.analysis_settings_vars["single_grain_type"].get() == "rigid_circle"
        assert application.analysis_settings_vars["single_grain_arc_start_angle_deg"].get() == "225.0"
        assert application.analysis_settings_vars["single_grain_arc_end_angle_deg"].get() == "315.0"
    finally:
        root.destroy()


def test_single_grain_shape_selection_disables_irrelevant_fields() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application.analysis_mode_var.set("单磨粒真实接触")
        application.analysis_settings_vars["single_grain_type"].set("rounded_wedge")
        application._apply_analysis_advanced_parameter_state("single_grain_high_fidelity")

        assert str(application.analysis_advanced_entries["single_grain_tip_radius_um"].cget("state")) == "normal"
        assert str(application.analysis_advanced_entries["single_grain_radius_um"].cget("state")) == "disabled"
        assert str(application.analysis_advanced_entries["single_grain_arc_start_angle_deg"].cget("state")) == "disabled"
        assert str(application.analysis_advanced_entries["single_grain_preset_removal_mode"].cget("state")) == "normal"

        application.analysis_settings_vars["single_grain_type"].set("rigid_circle")
        application._apply_analysis_advanced_parameter_state("single_grain_high_fidelity")

        assert str(application.analysis_advanced_entries["single_grain_radius_um"].cget("state")) == "normal"
        assert str(application.analysis_advanced_entries["single_grain_tip_radius_um"].cget("state")) == "disabled"
        assert str(application.analysis_advanced_entries["single_grain_preset_removal_mode"].cget("state")) == "disabled"
    finally:
        root.destroy()


def test_workbench_right_pane_has_a_left_vertical_scrollbar_at_compact_height() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        root.geometry("1120x720")
        root.deiconify()
        root.update_idletasks()
        root.update()
        application._show_workbench_page(
            application.workbench.select_node("分析")
        )
        root.update_idletasks()
        root.update()

        assert application.workbench_scrollbar.winfo_ismapped()
        assert application.workbench_scrollbar.winfo_rootx() < (
            application.workbench_scroll_canvas.winfo_rootx()
        )
        first, last = application.workbench_scroll_canvas.yview()
        assert first == pytest.approx(0.0)
        assert last < 1.0
    finally:
        root.destroy()


def test_workbench_scroll_reaches_analysis_run_button_at_compact_height() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        root.geometry("1120x720")
        root.deiconify()
        root.update_idletasks()
        root.update()
        application._show_workbench_page(
            application.workbench.select_node("分析")
        )
        root.update_idletasks()
        root.update()

        application.workbench_scroll_canvas.yview_moveto(1.0)
        root.update_idletasks()
        root.update()

        viewport_top = application.workbench_scroll_canvas.winfo_rooty()
        viewport_bottom = viewport_top + application.workbench_scroll_canvas.winfo_height()
        button_top = application.analysis_run_button.winfo_rooty()
        button_bottom = button_top + application.analysis_run_button.winfo_height()
        assert viewport_top <= button_top
        assert button_bottom <= viewport_bottom
    finally:
        root.destroy()


def test_analysis_progress_details_survive_stage_refresh_and_can_be_hidden() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._handle_analysis_event(
            AnalysisEvent(
                "progress",
                "基准路线已收敛位置 6/46",
                current=6,
                total=92,
                elapsed_seconds=639.0,
                remaining_seconds=9120.0,
                phase="formal",
            )
        )
        progress_text = application.analysis_progress_detail_var.get()
        assert "6.5%" in progress_text
        assert "预计剩余" in progress_text

        update_clock = application._analysis_progress_updated_at
        assert update_clock is not None
        application._refresh_analysis_progress_clock(now=update_clock + 60.0)
        assert "预计剩余：2小时31分" in application.analysis_progress_detail_var.get()
        progress_text = application.analysis_progress_detail_var.get()

        application._handle_analysis_event(
            AnalysisEvent("stage", "正在生成场演化图片。")
        )
        application._refresh_analysis_workspace()
        assert application.analysis_progress_detail_var.get() == progress_text

        assert application.analysis_progress_details_visible is True
        application._toggle_analysis_progress_details()
        assert application.analysis_progress_details_visible is False
        assert application.analysis_progress_detail_var.get() == progress_text
        assert application.analysis_progress_detail_label.winfo_manager() == ""
        assert application.analysis_progress.winfo_manager() == ""

        application._toggle_analysis_progress_details()
        assert application.analysis_progress_details_visible is True
        assert application.analysis_progress_detail_label.winfo_manager() == "pack"
        assert application.analysis_progress.winfo_manager() == "pack"
    finally:
        root.destroy()


def test_pilot_success_resets_visible_progress_to_formal_zero() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._analysis_progress_formal_total = 92
        application._handle_analysis_event(
            AnalysisEvent(
                "progress",
                "试算机制路线位置 5/5",
                current=10,
                total=10,
                elapsed_seconds=600.0,
                remaining_seconds=0.0,
                phase="pilot",
            )
        )

        application._handle_analysis_event(
            AnalysisEvent("pilot_success", "短程试算通过，开始正式计算。")
        )

        detail = application.analysis_progress_detail_var.get()
        assert "正式计算：正在准备" in detail
        assert "0.0%（0/92）" in detail
        assert "预计剩余：正在估算" in detail
        assert application.analysis_progress_var.get() == 0.0
    finally:
        root.destroy()


def test_analysis_progress_clock_updates_elapsed_before_first_completed_position() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._analysis_progress_started_at = 100.0
        application._analysis_progress_current = 0
        application._analysis_progress_total = 92
        application._analysis_progress_message = "正式计算正在准备"
        application._analysis_progress_phase = "formal"

        application._refresh_analysis_progress_clock(now=739.0)

        text = application.analysis_progress_detail_var.get()
        assert "已用：10分39秒" in text
        assert "预计剩余：正在估算" in text
    finally:
        root.destroy()


def test_opening_a_project_restores_description_even_from_a_disabled_node() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._show_workbench_page(application.workbench.select_node("网格"))
        application._set_workbench_project(
            app.create_project("恢复工程", "重新打开后的中文备注")
        )

        assert application.project_name_var.get() == "恢复工程"
        assert application.project_description_text.get("1.0", "end-1c") == (
            "重新打开后的中文备注"
        )
    finally:
        root.destroy()


def test_geometry_node_shows_default_engineering_inputs_and_generates_preview(
    monkeypatch, tmp_path: Path
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            application,
            "_geometry_output_directory",
            lambda: tmp_path / "geometry",
        )
        application._show_workbench_page(application.workbench.select_node("几何"))

        assert application.geometry_variables["workpiece_length_mm"].get() == "100"
        assert application.geometry_variables["workpiece_height_mm"].get() == "20"
        assert application.geometry_variables["wheel_diameter_mm"].get() == "200"
        assert application.geometry_variables["depth_of_cut_um"].get() == "20"
        assert application.geometry_variables["relative_feed_direction"].get() == "正向"
        assert application.geometry_variables["wheel_lowest_point_x_mm"].get() == "50"
        assert application.geometry_editor.winfo_manager() == "pack"

        application._generate_geometry()

        preview = tmp_path / "geometry" / "geometry_preview.png"
        assert preview.is_file() and preview.stat().st_size > 0
        assert application.workbench.project.node_status("几何") is ProjectNodeStatus.COMPLETED
        assert application.workbench.project.node_status("分析") is ProjectNodeStatus.NOT_CONFIGURED
        assert application.workbench.project.geometry["source"] == "parametric_2d"
        assert application.geometry_preview_photo is not None
        report = application.geometry_report_text.get("1.0", "end-1c")
        assert "几何状态：有效" in report
        assert "工件长度：100 mm" in report
        assert "当前结果不包含应力、应变、位移或磨削力" in report
    finally:
        root.destroy()


def test_geometry_gui_failure_keeps_previous_valid_model(
    monkeypatch, tmp_path: Path
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            application,
            "_geometry_output_directory",
            lambda: tmp_path / "geometry",
        )
        errors: list[str] = []
        monkeypatch.setattr(
            app.messagebox,
            "showerror",
            lambda _title, message, **_kwargs: errors.append(message),
        )
        application._generate_geometry()
        previous = json.loads(json.dumps(application.workbench.project.geometry))
        previous_preview = (tmp_path / "geometry" / "geometry_preview.png").read_bytes()

        application.geometry_variables["workpiece_length_mm"].set("0")
        application._generate_geometry()

        assert errors and "工件长度必须大于 0" in errors[-1]
        assert application.workbench.project.geometry == previous
        assert (tmp_path / "geometry" / "geometry_preview.png").read_bytes() == previous_preview
        assert application.workbench.project.node_status("几何") is ProjectNodeStatus.FAILED
    finally:
        root.destroy()


def test_geometry_report_is_fully_visible_at_default_window_size(
    monkeypatch, tmp_path: Path
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            application,
            "_geometry_output_directory",
            lambda: tmp_path / "geometry",
        )
        application._show_workbench_page(application.workbench.select_node("几何"))
        application._generate_geometry()
        root.deiconify()
        root.geometry("1420x900")
        root.update()
        root.update_idletasks()

        assert application.geometry_report_text.yview() == (0.0, 1.0)
    finally:
        root.destroy()


def test_opening_project_restores_geometry_form_and_existing_preview(
    monkeypatch, tmp_path: Path
) -> None:
    project = create_project("重开几何")
    provider = Parametric2DGeometryProvider(
        GeometryDisplayInput(
            workpiece_length_mm="120",
            workpiece_height_mm="25",
            wheel_diameter_mm="180",
            depth_of_cut_um="15",
            relative_feed_direction="negative_x",
            wheel_lowest_point_x_mm="70",
        ).build_case()
    )
    published = provider.publish(tmp_path / "geometry")
    from grindcae.gui.workbench import WorkbenchSession

    WorkbenchSession(project).register_geometry_result(provider, published)

    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._set_workbench_project(project, tmp_path / "project.gcase.json")
        application._show_workbench_page(application.workbench.select_node("几何"))

        assert application.geometry_variables["workpiece_length_mm"].get() == "120"
        assert application.geometry_variables["workpiece_height_mm"].get() == "25"
        assert application.geometry_variables["wheel_diameter_mm"].get() == "180"
        assert application.geometry_variables["depth_of_cut_um"].get() == "15"
        assert application.geometry_variables["relative_feed_direction"].get() == "反向"
        assert application.geometry_variables["wheel_lowest_point_x_mm"].get() == "70"
        assert application.geometry_preview_photo is not None
    finally:
        root.destroy()


def test_opening_old_project_without_geometry_clears_previous_geometry_widgets(
    monkeypatch, tmp_path: Path
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            application,
            "_geometry_output_directory",
            lambda: tmp_path / "geometry",
        )
        application._generate_geometry()
        assert application.geometry_preview_photo is not None

        application._set_workbench_project(app.create_project("2.4.0 旧工程"))
        application._show_workbench_page(application.workbench.select_node("几何"))

        assert application.geometry_variables["workpiece_length_mm"].get() == "100"
        assert application.geometry_variables["workpiece_height_mm"].get() == "20"
        assert application.geometry_variables["wheel_diameter_mm"].get() == "200"
        assert application.geometry_variables["depth_of_cut_um"].get() == "20"
        assert application.geometry_variables["relative_feed_direction"].get() == "正向"
        assert application.geometry_variables["wheel_lowest_point_x_mm"].get() == "50"
        assert application.geometry_preview_photo is None
        assert "点击“生成二维模型”" in application.geometry_image_preview.canvas.itemcget(
            application.geometry_image_preview.message_item, "text"
        )
        assert application.geometry_report_text.get("1.0", "end-1c") == "几何尚未配置。"
    finally:
        root.destroy()


def test_png_preview_failure_is_isolated_from_published_calculation_result(
    monkeypatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "strain_contour.png"
    image_path.write_bytes(b"not a PNG")
    log_messages: list[str] = []
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._images = (
            GuiImage("strain contour", image_path, "explanation"),
        )
        application.image_choice_var.set("strain contour")
        monkeypatch.setattr(application, "_log", log_messages.append)

        application._load_selected_image()

        assert application._preview_photo is None
        assert application.result_image_preview.image_metadata() is None
        assert "PNG 预览失败，但计算结果仍保留" in (
            application.result_image_preview.canvas.itemcget(
                application.result_image_preview.message_item, "text"
            )
        )
        assert str(application.result_image_preview.open_original_button.cget("state")) == "disabled"
        assert log_messages and "strain_contour.png" in log_messages[-1]
    finally:
        root.destroy()


def test_geometry_mesh_and_analysis_pages_reuse_image_preview_component() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)

        assert isinstance(application.geometry_image_preview, ImagePreview)
        assert isinstance(application.mesh_image_preview, ImagePreview)
        assert isinstance(application.result_image_preview, ImagePreview)
        toolbar_text = {
            child.cget("text")
            for child in application.result_preview_toolbar.winfo_children()
            if isinstance(child, ttk.Button)
        }
        assert "系统程序打开原图" not in toolbar_text
        assert "打开输出目录" in toolbar_text
    finally:
        root.destroy()


def test_mesh_page_defaults_and_missing_geometry_error(monkeypatch) -> None:
    errors: list[str] = []
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            app.messagebox,
            "showerror",
            lambda _title, message, **_kwargs: errors.append(message),
        )
        application._show_workbench_page(application.workbench.select_node("网格"))

        assert application.mesh_variables["mesh_type"].get() == "三角形"
        assert application.mesh_variables["target_size_mm"].get() == "3"
        assert application.mesh_variables["minimum_size_mm"].get() == "1"
        assert application.mesh_variables["maximum_size_mm"].get() == "5"
        assert application.mesh_editor.winfo_manager() == "pack"

        application._generate_mesh()

        assert errors and "请先完成二维几何建模" in errors[-1]
        assert application.workbench.project.node_status("网格") is ProjectNodeStatus.DISABLED
    finally:
        root.destroy()


def test_mesh_page_generates_real_mesh_in_background_and_restores_preview(
    monkeypatch, tmp_path: Path
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            application,
            "_geometry_output_directory",
            lambda: tmp_path / "geometry",
        )
        monkeypatch.setattr(
            application,
            "_mesh_output_directory",
            lambda: tmp_path / "mesh",
        )
        application._generate_geometry()
        application._show_workbench_page(application.workbench.select_node("网格"))

        application._generate_mesh()
        deadline = time.monotonic() + 10.0
        while application.mesh_controller.is_running and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        root.update()
        application._poll_mesh_events()

        assert application.mesh_controller.is_running is False
        assert application.workbench.project.node_status("网格") is ProjectNodeStatus.COMPLETED
        assert application.workbench.project.mesh["status"] == "completed"
        assert application.workbench.project.latest_mesh_result is not None
        assert (tmp_path / "mesh" / "mesh.msh").is_file()
        assert (tmp_path / "mesh" / "mesh.png").is_file()
        assert (tmp_path / "mesh" / "mesh_summary.json").is_file()
        metadata = application.mesh_image_preview.image_metadata()
        assert metadata is not None
        assert metadata.filename == "mesh.png"
        assert metadata.source == "mesh"
        assert str(application.mesh_image_preview.open_original_button.cget("state")) == "normal"
        report = application.mesh_report_text.get("1.0", "end-1c")
        assert "网格状态：有效" in report
        assert "三角形单元数量" in report
        assert "预览策略：完整网格线与节点" in report
        assert "完整 mesh.msh：已保留" in report
        assert "预计二维向量自由度" in report
    finally:
        root.destroy()


def test_historical_mesh_summary_without_preview_metadata_stays_compatible() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._show_mesh_report(
            {
                "mesh_settings": {"target_size_m": 0.003},
                "mesh_statistics": {
                    "node_count": 100,
                    "triangle_count": 180,
                    "boundary_element_count": 20,
                    "minimum_element_area_m2": 1.0e-6,
                    "maximum_element_area_m2": 2.0e-6,
                    "actual_average_size_m": 0.003,
                },
            }
        )

        report = application.mesh_report_text.get("1.0", "end-1c")
        assert "预览策略：历史版本预览" in report
        assert "预计二维向量自由度：历史摘要未记录" in report
        assert "超过默认安全阈值" not in report
    finally:
        root.destroy()


def test_new_geometry_immediately_clears_the_old_mesh_preview(
    monkeypatch, tmp_path: Path
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        monkeypatch.setattr(
            application, "_geometry_output_directory", lambda: tmp_path / "geometry"
        )
        monkeypatch.setattr(
            application, "_mesh_output_directory", lambda: tmp_path / "mesh"
        )
        application._generate_geometry()
        application._generate_mesh()
        deadline = time.monotonic() + 10.0
        while application.mesh_controller.is_running and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        root.update()
        application._poll_mesh_events()
        assert application.mesh_image_preview.image_metadata() is not None

        application.geometry_variables["depth_of_cut_um"].set("25")
        application._generate_geometry()

        assert application.workbench.project.node_status("网格") is ProjectNodeStatus.STALE
        assert application.mesh_image_preview.image_metadata() is None
        assert "重新生成网格" in application.mesh_image_preview.canvas.itemcget(
            application.mesh_image_preview.message_item, "text"
        )
        assert "需要更新" in application.mesh_report_text.get("1.0", "end-1c")
    finally:
        root.destroy()


def _configured_parameter_project():
    project = create_project("材料工艺 GUI")
    project.geometry = Parametric2DGeometryProvider(
        GeometryDisplayInput().build_case()
    ).to_geometry_schema()
    project.mesh = MeshDisplayInput().build_case().to_dict()
    project.mesh["status"] = "completed"
    project.set_node_status("几何", ProjectNodeStatus.COMPLETED)
    project.set_node_status("网格", ProjectNodeStatus.READY)
    project.set_node_status("网格", ProjectNodeStatus.COMPLETED)
    return project


def test_material_page_edits_saves_and_disables_unused_plastic_fields() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        application = app.GrindCaeApp(root)
        application._set_workbench_project(_configured_parameter_project())
        application._show_workbench_page(application.workbench.select_node("材料"))

        assert application.material_editor.winfo_manager() == "pack"
        assert application.material_variables["elastic_modulus_gpa"].get() == "210"
        assert application.material_variables["yield_strength_mpa"].get() == "250"
        application.material_variables["behavior"].set("线弹性")
        application._update_material_behavior()
        assert str(application.material_plastic_entries["yield_strength_mpa"].cget("state")) == "disabled"
        assert str(application.material_plastic_entries["tangent_modulus_gpa"].cget("state")) == "disabled"

        application._save_material_parameters()

        assert application.workbench.project.material["behavior"] == "linear_elastic"
        assert application.workbench.project.material["yield_strength_Pa"] is None
        assert application.workbench.project.material["tangent_modulus_Pa"] is None
        assert application.workbench.project.node_status("材料") is ProjectNodeStatus.COMPLETED
        assert "不参与当前模型" in application.material_report_text.get("1.0", "end-1c")
    finally:
        root.destroy()


def test_process_page_resolves_existing_grit_and_saves_si_parameters() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        project = _configured_parameter_project()
        WorkbenchSession(project).register_material(MaterialDisplayInput().build_case())
        application = app.GrindCaeApp(root)
        application._set_workbench_project(project)
        application._show_workbench_page(application.workbench.select_node("工艺"))

        assert application.process_editor.winfo_manager() == "pack"
        assert application.process_geometry_variables["wheel_diameter"].get() == "200 mm（来自几何，只读）"
        assert application.process_geometry_variables["depth_of_cut"].get() == "20 μm（来自几何，只读）"
        assert application.wheel_variables["resolution_path"].get() == "JIS # 二手参考"
        assert tuple(application.grit_designation_combo.cget("values")) == (
            "#450", "#700", "#1000", "#1800", "#4000", "#5000", "#6000",
        )

        application._resolve_workbench_wheel()
        assert "3.4 μm" in application.process_report_text.get("1.0", "end-1c")
        assert "临时工程参考，待厂家数据核实" in application.process_report_text.get("1.0", "end-1c")
        application._save_process_parameters()

        assert application.workbench.project.wheel["grit_designation"] == "#5000"
        assert application.workbench.project.process["grinding_width_m"] == pytest.approx(0.01)
        assert application.workbench.project.process["specific_grinding_energy_J_per_m3"] == pytest.approx(16.1e9)
        assert application.workbench.project.process["calibration_status"] == "literature_reference"
        assert application.workbench.project.node_status("工艺") is ProjectNodeStatus.COMPLETED
    finally:
        root.destroy()


def test_process_page_manufacturer_override_uses_visible_strict_fields() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        project = _configured_parameter_project()
        WorkbenchSession(project).register_material(MaterialDisplayInput().build_case())
        application = app.GrindCaeApp(root)
        application._set_workbench_project(project)
        application._show_workbench_page(application.workbench.select_node("工艺"))
        application.wheel_variables["resolution_path"].set("厂家数据覆盖")
        application._update_grit_path()

        assert application.manufacturer_frame.winfo_manager() == "grid"
        for name, value in {
            "grit_designation": "AO-X",
            "manufacturer": "示例磨料厂",
            "product_model": "AO-X-01",
            "diameter_lower_um": "2",
            "representative_diameter_um": "3",
            "diameter_upper_um": "4",
            "manufacturer_source": "产品目录第 3 页",
            "manufacturer_data_status": "manufacturer_catalogue",
        }.items():
            application.wheel_variables[name].set(value)

        application._resolve_workbench_wheel()

        assert application._resolved_workbench_wheel is not None
        assert application._resolved_workbench_wheel.resolved_specification[
            "manufacturer_override_applied"
        ] is True
        assert "厂家数据覆盖" in application.process_report_text.get("1.0", "end-1c")
    finally:
        root.destroy()


def test_opening_parameter_project_restores_material_wheel_and_process_forms() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        project = _configured_parameter_project()
        session = WorkbenchSession(project)
        material = MaterialDisplayInput(
            label="恢复材料", elastic_modulus_gpa="190"
        ).build_case()
        session.register_material(material)
        wheel = app.WheelDisplayInput(
            resolution_path="GB_W_micropowder", grit_designation="W3.5"
        ).build_case()
        process = app.ProcessDisplayInput(
            grinding_width_mm="8"
        ).build_case(project.geometry, wheel)
        session.register_process(wheel, process)
        application = app.GrindCaeApp(root)

        application._set_workbench_project(project)
        application._show_workbench_page(application.workbench.select_node("材料"))
        assert application.material_variables["label"].get() == "恢复材料"
        assert application.material_variables["elastic_modulus_gpa"].get() == "190"
        application._show_workbench_page(application.workbench.select_node("工艺"))
        assert application.wheel_variables["resolution_path"].get() == "W 系列微粉参考"
        assert application.wheel_variables["grit_designation"].get() == "W3.5"
        assert application.process_variables["grinding_width_mm"].get() == "8"
    finally:
        root.destroy()


def test_failed_grit_reresolution_cannot_save_the_previous_valid_wheel(
    monkeypatch,
) -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        project = _configured_parameter_project()
        WorkbenchSession(project).register_material(MaterialDisplayInput().build_case())
        application = app.GrindCaeApp(root)
        application._set_workbench_project(project)
        application._show_workbench_page(application.workbench.select_node("工艺"))
        application._resolve_workbench_wheel()
        assert application._resolved_workbench_wheel is not None
        errors: list[str] = []
        monkeypatch.setattr(
            app.messagebox,
            "showerror",
            lambda _title, message, **_kwargs: errors.append(message),
        )
        application.wheel_variables["resolution_path"].set("厂家数据覆盖")
        application._update_grit_path()
        application.wheel_variables["grit_designation"].set("AO-X")
        application.wheel_variables["manufacturer"].set("")

        application._save_process_parameters()

        assert errors
        assert application._resolved_workbench_wheel is None
        assert project.wheel is None
        assert project.process is None
        assert project.node_status("工艺") is ProjectNodeStatus.READY
    finally:
        root.destroy()
