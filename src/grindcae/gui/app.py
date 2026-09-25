"""Tkinter widgets for the phase 5B2 desktop application."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from grindcae import __version__

from .config import load_form, save_form
from .controller import GuiController, GuiEvent
from .analysis import (
    ANALYSIS_MODE_DEFINITIONS,
    AnalysisProjectData,
    AnalysisSettings,
    GrainPopulationAnalysisSettings,
    HistoryTransitionAnalysisSettings,
    LoadDistributionAnalysisSettings,
    NewtonAnalysisSettings,
    analysis_mode_by_display_name,
)
from .analysis_results import AnalysisPublishedResult, AnalysisResultAdapter, AnalysisResultError
from .analysis_runner import AnalysisEvent, AnalysisRunError, AnalysisRunner
from .analysis_preflight import quick_analysis_audit
from .analysis_progress import estimate_progress, format_progress_status
from .analysis_workspace import (
    AnalysisWorkspace,
    analysis_input_summary,
    analysis_presentation_state,
    default_analysis_output_directory,
)
from .background_tasks import BackgroundTaskGate
from .form import (
    MECHANISM_HISTORY_MODE,
    GuiForm,
    GuiInputError,
    SCAN_MODE,
    SINGLE_MODE,
)
from .help_text import FORCE_MODEL_HELP_TEXT, MECHANISM_HISTORY_HELP_TEXT
from .geometry import (
    GeometryDisplayInput,
    GeometryValidationError,
    Parametric2DGeometryProvider,
    ParametricGeometryCase,
)
from .image_preview import ImagePreview
from .material_workspace import (
    MaterialDisplayInput,
    MaterialValidationError,
    WorkbenchMaterialCase,
)
from .mesh_controller import MeshController, MeshControllerError
from .mesh_workspace import (
    MeshDisplayInput,
    MeshPublishedResult,
    MeshValidationError,
    TriangleMeshProvider,
    WorkbenchMeshCase,
)
from .process_workspace import (
    ProcessDisplayInput,
    ProcessValidationError,
    WheelDisplayInput,
    WorkbenchProcessCase,
    WorkbenchWheelCase,
)
from .process_grain_preview import (
    BuiltProcessGrainPreview,
    ProcessGrainPreviewBuilder,
    ProcessGrainPreviewError,
    ProcessGrainPreviewEvent,
    ProcessGrainPreviewRunner,
    process_grain_preview_state,
)
from grindcae.statistical_grain_load import available_grit_designations
from grindcae.presentation_labels import analysis_mode_parameter_keys
from .labels import calibration_set_display_text, calibration_set_name
from .project import (
    ProjectError,
    ProjectNodeStatus,
    ResultRecord,
    create_project,
    load_project,
    save_project,
)
from .project_archive import export_project_archive, import_project_archive
from .project_history import (
    RecentProjectStore,
    RecentResultStore,
    RecoveryStore,
    entry_as_result_record,
    import_result_directory,
    recent_entry_from_record,
    relink_result_record,
)
from .resources import apply_window_icon
from .results import GuiImage, limitation_text
from .result_center import build_result_center_state
from .result_center_view import ResultCenterView
from .runtime_paths import (
    application_directory,
    application_state_directory,
    is_frozen,
    mechanism_history_example_path,
    recovery_directory,
)
from .workbench import WorkbenchPage, WorkbenchSession


FORM_FIELDS = tuple(GuiForm.__dataclass_fields__)
GUI_FONT_FAMILY = "Microsoft YaHei UI"

_MATERIAL_BEHAVIOR_LABELS = {"线弹性": "linear_elastic", "弹塑性": "elastoplastic"}
_MATERIAL_SOURCE_LABELS = {
    "通用延性合金钢演示参数": "builtin_demo",
    "用户自定义": "user_custom",
}
_CALIBRATION_LABELS = {
    "演示参数，未标定": "demonstration_not_calibrated",
    "用户输入，未标定": "user_input_not_calibrated",
    "文献参考": "literature_reference",
    "实验标定": "experimentally_calibrated",
}
_GRIT_PATH_LABELS = {
    "GB/FEPA 参考": "GB_FEPA_F",
    "W 系列微粉参考": "GB_W_micropowder",
    "JIS # 二手参考": "JIS_hash",
    "厂家数据覆盖": "manufacturer_override",
}


def initial_window_geometry(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    """Return a usable initial/minimum size for the current logical desktop."""

    available_width = max(1, int(screen_width) - 80)
    available_height = max(1, int(screen_height) - 120)
    width = min(1420, available_width)
    height = min(900, available_height)
    minimum_width = min(1120, width)
    minimum_height = min(720, height)
    return width, height, minimum_width, minimum_height


def configure_main_window(root: tk.Tk) -> bool:
    """Set stable window properties while treating the icon as optional."""

    root.title(f"GrindCAE {__version__} - 计算工作台")
    icon_loaded = apply_window_icon(root)
    screen_width = (
        root.winfo_screenwidth() if hasattr(root, "winfo_screenwidth") else 1920
    )
    screen_height = (
        root.winfo_screenheight() if hasattr(root, "winfo_screenheight") else 1080
    )
    width, height, minimum_width, minimum_height = initial_window_geometry(
        screen_width, screen_height
    )
    root.geometry(f"{width}x{height}")
    root.minsize(minimum_width, minimum_height)
    return icon_loaded


class GrindCaeApp:
    """Main-thread-only Tk view backed by a queue-driven controller."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window_icon_loaded = configure_main_window(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.controller = GuiController()
        self.background_task_gate = BackgroundTaskGate()
        self.analysis_runner = AnalysisRunner(gate=self.background_task_gate)
        self.process_grain_preview_runner = ProcessGrainPreviewRunner(
            gate=self.background_task_gate
        )
        self.mesh_controller = MeshController()
        self.workbench = WorkbenchSession(create_project())
        self.project_path: Path | None = None
        state_directory = application_state_directory()
        self.recent_project_store = RecentProjectStore(state_directory / "recent_projects.json")
        self.recent_result_store = RecentResultStore(state_directory / "recent_results.json")
        self.recovery_store = RecoveryStore(recovery_directory())
        self._session_marker_path = state_directory / "active_session.json"
        self._previous_session_unclean = self._initialize_session_marker()
        self._saved_project_snapshot = ""
        self._recovery_observed_snapshot = ""
        self._recovery_change_started_at: float | None = None
        self._last_recovery_snapshot = ""
        self._loaded_project_file_sha256: str | None = None
        self._external_result_records: list[ResultRecord] = []
        self.project_name_var = tk.StringVar(value=self.workbench.project.project_name)
        self.project_description_var = tk.StringVar(
            value=self.workbench.project.description
        )
        self.project_path_var = tk.StringVar(value="尚未保存")
        self.workbench_title_var = tk.StringVar(value="工程")
        self.workbench_status_var = tk.StringVar(value="可执行")
        self.workbench_description_var = tk.StringVar(value="")
        self.workbench_input_var = tk.StringVar(value="")
        self.workbench_output_var = tk.StringVar(value="")
        self.workbench_limitation_var = tk.StringVar(value="")
        self.geometry_variables = {
            "workpiece_length_mm": tk.StringVar(value="100"),
            "workpiece_height_mm": tk.StringVar(value="20"),
            "wheel_diameter_mm": tk.StringVar(value="200"),
            "depth_of_cut_um": tk.StringVar(value="20"),
            "relative_feed_direction": tk.StringVar(value="正向"),
            "wheel_lowest_point_x_mm": tk.StringVar(value="50"),
        }
        self.geometry_preview_photo: tk.PhotoImage | None = None
        self._active_mesh_provider: TriangleMeshProvider | None = None
        self.mesh_variables = {
            "mesh_type": tk.StringVar(value="三角形"),
            "target_size_mm": tk.StringVar(value="3"),
            "minimum_size_mm": tk.StringVar(value="1"),
            "maximum_size_mm": tk.StringVar(value="5"),
        }
        material_defaults = MaterialDisplayInput()
        self.material_variables = {
            "label": tk.StringVar(value=material_defaults.label),
            "behavior": tk.StringVar(value="弹塑性"),
            "family": tk.StringVar(value="延性金属"),
            "elastic_modulus_gpa": tk.StringVar(value=material_defaults.elastic_modulus_gpa),
            "poisson_ratio": tk.StringVar(value=material_defaults.poisson_ratio),
            "yield_strength_mpa": tk.StringVar(value=material_defaults.yield_strength_mpa),
            "tangent_modulus_gpa": tk.StringVar(value=material_defaults.tangent_modulus_gpa),
            "parameter_source": tk.StringVar(value="通用延性合金钢演示参数"),
            "calibration_status": tk.StringVar(value="演示参数，未标定"),
            "source_description": tk.StringVar(value=material_defaults.source_description),
            "applicability_notes": tk.StringVar(value=material_defaults.applicability_notes),
        }
        wheel_defaults = WheelDisplayInput()
        self.wheel_variables = {
            "abrasive_material": tk.StringVar(value="刚玉"),
            "resolution_path": tk.StringVar(value="JIS # 二手参考"),
            "grit_designation": tk.StringVar(value=wheel_defaults.grit_designation),
            "manufacturer": tk.StringVar(value=""),
            "product_model": tk.StringVar(value=""),
            "diameter_lower_um": tk.StringVar(value=""),
            "representative_diameter_um": tk.StringVar(value=""),
            "diameter_upper_um": tk.StringVar(value=""),
            "manufacturer_source": tk.StringVar(value=""),
            "manufacturer_data_status": tk.StringVar(value="manufacturer_input_unverified"),
        }
        process_defaults = ProcessDisplayInput()
        self.process_variables = {
            "wheel_surface_speed_m_per_s": tk.StringVar(value=process_defaults.wheel_surface_speed_m_per_s),
            "workpiece_feed_speed_m_per_s": tk.StringVar(value=process_defaults.workpiece_feed_speed_m_per_s),
            "grinding_width_mm": tk.StringVar(value=process_defaults.grinding_width_mm),
            "specific_grinding_energy_j_per_mm3": tk.StringVar(value=process_defaults.specific_grinding_energy_j_per_mm3),
            "normal_to_tangential_force_ratio": tk.StringVar(value=process_defaults.normal_to_tangential_force_ratio),
            "calibration_id": tk.StringVar(value=process_defaults.calibration_id),
            "calibration_status": tk.StringVar(
                value={value: key for key, value in _CALIBRATION_LABELS.items()}[
                    process_defaults.calibration_status
                ]
            ),
            "source_description": tk.StringVar(value=process_defaults.source_description),
            "applicability_notes": tk.StringVar(value=process_defaults.applicability_notes),
        }
        self.process_geometry_variables = {
            "wheel_diameter": tk.StringVar(value="尚未配置几何"),
            "depth_of_cut": tk.StringVar(value="尚未配置几何"),
        }
        self._resolved_workbench_wheel: WorkbenchWheelCase | None = None
        self.process_grain_preview_status_var = tk.StringVar(
            value="尚未生成统计等效磨粒与载荷预览。"
        )
        self._active_process_grain_preview: BuiltProcessGrainPreview | None = None
        self._process_grain_preview_output: Path | None = None
        analysis_defaults = AnalysisSettings.recommended()
        self.analysis_mode_var = tk.StringVar(value="线弹性·单位置")
        self.analysis_output_var = tk.StringVar(value="")
        self.analysis_task_var = tk.StringVar(value="")
        self.analysis_checks_var = tk.StringVar(value="")
        self.analysis_mapping_var = tk.StringVar(value="")
        self.analysis_status_var = tk.StringVar(value="等待配置")
        self.analysis_progress_detail_var = tk.StringVar(value="尚未开始计算。")
        self.analysis_progress_var = tk.DoubleVar(value=0.0)
        self.analysis_progress_details_visible = True
        self._analysis_progress_started_at: float | None = None
        self._analysis_progress_current = 0
        self._analysis_progress_total = 1
        self._analysis_progress_formal_total = 1
        self._analysis_progress_message = "尚未开始计算"
        self._analysis_progress_phase = "formal"
        self._analysis_progress_updated_at: float | None = None
        self._analysis_progress_remaining_at_update: float | None = None
        self.analysis_settings_vars = {
            "scan_base_position_count": tk.StringVar(value=str(analysis_defaults.scan_base_position_count)),
            "loading_step_count": tk.StringVar(value=str(analysis_defaults.loading_step_count)),
            "unloading_step_count": tk.StringVar(value=str(analysis_defaults.unloading_step_count)),
            "newton_residual_relative_tolerance": tk.StringVar(value=str(analysis_defaults.newton.residual_relative_tolerance)),
            "newton_maximum_iterations": tk.StringVar(value=str(analysis_defaults.newton.maximum_iterations)),
            "history_base_substep_count": tk.StringVar(value=str(analysis_defaults.history_transition.base_substep_count)),
            "history_final_unloading_substep_count": tk.StringVar(value=str(analysis_defaults.history_transition.final_unloading_substep_count)),
            "grain_minimum_group_count": tk.StringVar(value=str(analysis_defaults.grain_population.minimum_equivalent_group_count)),
            "grain_maximum_group_count": tk.StringVar(value=str(analysis_defaults.grain_population.maximum_equivalent_group_count)),
            "grain_random_seed": tk.StringVar(value=str(analysis_defaults.grain_population.random_seed)),
            "load_point_count": tk.StringVar(value=str(analysis_defaults.load_distribution.point_count)),
            "single_grain_workpiece_width_um": tk.StringVar(value=str(analysis_defaults.single_grain.workpiece_width_m * 1e6)),
            "single_grain_workpiece_height_um": tk.StringVar(value=str(analysis_defaults.single_grain.workpiece_height_m * 1e6)),
            "single_grain_mesh_size_um": tk.StringVar(value=str(analysis_defaults.single_grain.mesh_target_size_m * 1e6)),
            "single_grain_thickness_mm": tk.StringVar(value=str(analysis_defaults.single_grain.workpiece_thickness_m * 1e3)),
            "single_grain_type": tk.StringVar(value=analysis_defaults.single_grain.grain_type),
            "single_grain_radius_um": tk.StringVar(value=str(analysis_defaults.single_grain.grain_radius_m * 1e6)),
            "single_grain_arc_start_angle_deg": tk.StringVar(value=str(math.degrees(analysis_defaults.single_grain.grain_arc_start_angle_rad))),
            "single_grain_arc_end_angle_deg": tk.StringVar(value=str(math.degrees(analysis_defaults.single_grain.grain_arc_end_angle_rad))),
            "single_grain_tip_radius_um": tk.StringVar(value=str(analysis_defaults.single_grain.grain_tip_radius_m * 1e6)),
            "single_grain_wedge_height_um": tk.StringVar(value=str(analysis_defaults.single_grain.grain_wedge_height_m * 1e6)),
            "single_grain_intrinsic_rake_angle_deg": tk.StringVar(value=str(math.degrees(analysis_defaults.single_grain.intrinsic_rake_angle_rad))),
            "single_grain_intrinsic_clearance_angle_deg": tk.StringVar(value=str(math.degrees(analysis_defaults.single_grain.intrinsic_clearance_angle_rad))),
            "single_grain_pose_angle_deg": tk.StringVar(value=str(math.degrees(analysis_defaults.single_grain.grain_pose_angle_rad))),
            "single_grain_nominal_direction": tk.StringVar(value=analysis_defaults.single_grain.nominal_scratch_direction),
            "single_grain_preset_removal_mode": tk.StringVar(value=analysis_defaults.single_grain.preset_removal_mode),
            "single_grain_preset_removal_element_ids": tk.StringVar(value=",".join(str(item) for item in analysis_defaults.single_grain.preset_removal_element_ids)),
            "single_grain_preset_removal_polygon_json": tk.StringVar(value=json.dumps(analysis_defaults.single_grain.preset_removal_polygon_vertices_m)),
            "single_grain_initial_clearance_um": tk.StringVar(value=str(analysis_defaults.single_grain.initial_clearance_m * 1e6)),
            "single_grain_indentation_depth_um": tk.StringVar(value=str(analysis_defaults.single_grain.indentation_depth_m * 1e6)),
            "single_grain_scratch_distance_um": tk.StringVar(value=str(analysis_defaults.single_grain.scratch_distance_m * 1e6)),
            "single_grain_normal_algorithm": tk.StringVar(value=analysis_defaults.single_grain.normal_algorithm),
            "single_grain_penalty_factor": tk.StringVar(value=str(analysis_defaults.single_grain.normal_penalty_factor)),
            "single_grain_tangential_penalty_ratio": tk.StringVar(value=str(analysis_defaults.single_grain.tangential_penalty_ratio)),
            "single_grain_augmented_relaxation": tk.StringVar(value=str(analysis_defaults.single_grain.augmented_relaxation)),
            "single_grain_augmented_maximum_iterations": tk.StringVar(value=str(analysis_defaults.single_grain.augmented_maximum_iterations)),
            "single_grain_friction_coefficient": tk.StringVar(value=str(analysis_defaults.single_grain.friction_coefficient)),
            "single_grain_friction_regularization_ratio": tk.StringVar(value=str(analysis_defaults.single_grain.friction_regularization_ratio)),
            "single_grain_damage_enabled": tk.StringVar(value=str(analysis_defaults.single_grain.damage_separation_enabled).lower()),
            "single_grain_damage_evolution": tk.StringVar(value=analysis_defaults.single_grain.damage_evolution),
            "single_grain_damage_failure_table_json": tk.StringVar(value=json.dumps(analysis_defaults.single_grain.damage_failure_strain_table)),
            "single_grain_damage_triaxiality_cutoff": tk.StringVar(value=str(analysis_defaults.single_grain.damage_triaxiality_cutoff)),
            "single_grain_damage_fracture_energy": tk.StringVar(value=str(analysis_defaults.single_grain.damage_fracture_energy_J_per_m2)),
            "single_grain_damage_event_tolerance": tk.StringVar(value=str(analysis_defaults.single_grain.damage_event_tolerance)),
            "single_grain_damage_max_events": tk.StringVar(value=str(analysis_defaults.single_grain.damage_maximum_separation_events_per_position)),
            "single_grain_damage_area_tolerance_m2": tk.StringVar(value=str(analysis_defaults.single_grain.damage_area_absolute_tolerance_m2)),
            "single_grain_damage_parameter_source": tk.StringVar(value=analysis_defaults.single_grain.damage_parameter_source),
            "single_grain_damage_calibration_status": tk.StringVar(value=analysis_defaults.single_grain.damage_calibration_status),
            "single_grain_damage_applicability_notes": tk.StringVar(value=analysis_defaults.single_grain.damage_applicability_notes),
        }
        self.analysis_advanced_notice_var = tk.StringVar(value="")
        self.analysis_advanced_entries: dict[str, ttk.Entry] = {}
        self.analysis_advanced_labels: dict[str, ttk.Label] = {}
        self._active_analysis_built = None
        self._active_analysis_output: Path | None = None
        self.variables = {
            name: tk.StringVar(value=str(value))
            for name, value in asdict(GuiForm()).items()
        }
        self.validation_var = tk.StringVar(value="尚未校验")
        self.stage_var = tk.StringVar(value="等待运行")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.image_choice_var = tk.StringVar(value="")
        self.image_explanation_var = tk.StringVar(value="")
        self.limitation_var = tk.StringVar(value=limitation_text(GuiForm().mode))
        self.calibration_display_var = tk.StringVar(value="")
        self.calibration_name_var = tk.StringVar(value="")
        self._images: tuple[GuiImage, ...] = ()
        self._preview_photo: tk.PhotoImage | None = None
        self._preview_after_id: str | None = None
        self._running_widgets: list[tk.Widget] = []
        self._readonly_widgets: list[ttk.Combobox] = []
        self._parameter_widgets: list[tk.Widget] = []
        self._parameter_readonly_widgets: list[ttk.Combobox] = []
        self.variables["calibration_id"].trace_add("write", self._update_calibration_display)
        self._update_calibration_display()

        self._configure_style()
        self._build_layout()
        self._restore_analysis_widgets()
        self._apply_mode_state()
        self.root.after(100, self._poll_events)
        self.root.after(100, self._poll_mesh_events)
        self.root.after(100, self._poll_analysis_events)
        self.root.after(1000, self._tick_analysis_progress_clock)
        self.root.after(100, self._poll_process_grain_preview_events)
        self._accept_saved_snapshot()
        if is_frozen() and self._previous_session_unclean:
            self.root.after(250, self._offer_recovery_snapshot)
        self.root.after(500, self._tick_project_dirty_state)
        self.root.after(1000, self._tick_auto_recovery)

    def _sync_project_metadata_from_widgets(self) -> None:
        name = self.project_name_var.get().strip()
        if not name:
            raise ProjectError("工程名称不能为空。")
        description = self.project_description_text.get("1.0", "end-1c")
        self.workbench.project.project_name = name
        self.workbench.project.description = description

    def _project_snapshot(self) -> str:
        try:
            self._sync_project_metadata_from_widgets()
        except (ProjectError, tk.TclError):
            pass
        return json.dumps(
            self.workbench.project.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _accept_saved_snapshot(self) -> None:
        self._saved_project_snapshot = self._project_snapshot()
        self._loaded_project_file_sha256 = self._project_file_sha256(self.project_path)
        self._update_window_title()

    @staticmethod
    def _project_file_sha256(path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    def _project_is_dirty(self) -> bool:
        return self._project_snapshot() != self._saved_project_snapshot

    def _update_window_title(self) -> None:
        name = self.project_path.name if self.project_path else self.workbench.project.project_name
        marker = " *" if self._project_is_dirty() else ""
        self.root.title(f"GrindCAE {__version__} — {name}{marker}")

    def _tick_project_dirty_state(self) -> None:
        try:
            self._update_window_title()
            self.root.after(500, self._tick_project_dirty_state)
        except tk.TclError:
            return

    def _tick_auto_recovery(self, *, now: float | None = None) -> None:
        try:
            moment = time.monotonic() if now is None else now
            snapshot = self._project_snapshot()
            if snapshot != self._recovery_observed_snapshot:
                self._recovery_observed_snapshot = snapshot
                self._recovery_change_started_at = moment
            elif (
                snapshot != self._last_recovery_snapshot
                and snapshot != self._saved_project_snapshot
                and self._recovery_change_started_at is not None
                and moment - self._recovery_change_started_at >= 8.0
            ):
                self._write_recovery_snapshot()
                self._last_recovery_snapshot = snapshot
            if now is None:
                self.root.after(1000, self._tick_auto_recovery)
        except tk.TclError:
            return

    def _write_recovery_snapshot(self) -> None:
        try:
            self._sync_project_metadata_from_widgets()
            self._save_analysis_configuration()
            self.recovery_store.save(
                self.workbench.project, source_path=self.project_path
            )
        except (ProjectError, ValueError, OSError):
            pass

    def _initialize_session_marker(self) -> bool:
        previous_pid = None
        try:
            if self._session_marker_path.is_file():
                previous = json.loads(self._session_marker_path.read_text(encoding="utf-8"))
                previous_pid = previous.get("pid") if isinstance(previous, dict) else None
            self._session_marker_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_marker_path.write_text(
                json.dumps({"pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}),
                encoding="utf-8",
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
        return isinstance(previous_pid, int) and previous_pid != os.getpid()

    def _offer_recovery_snapshot(self) -> None:
        try:
            snapshots = self.recovery_store.list()
        except (ProjectError, OSError):
            return
        if not snapshots:
            return
        newest = next((item for item in snapshots if item.is_newer_than_source()), None)
        if newest is None:
            return
        try:
            recovered = load_project(newest.path)
        except ProjectError:
            return
        if not messagebox.askyesno(
            "发现自动恢复工程",
            f"发现可用的自动恢复副本：\n{recovered.project_name}\n\n是否恢复到当前窗口？",
            parent=self.root,
        ):
            return
        self._set_workbench_project(recovered, None)
        self._saved_project_snapshot = ""
        self._update_window_title()

    def _confirm_discard_or_save(self) -> bool:
        if not self._project_is_dirty():
            return True
        self._write_recovery_snapshot()
        decision = messagebox.askyesnocancel(
            "工程尚未保存",
            f"当前工程“{self.workbench.project.project_name}”包含尚未保存的修改。\n\n"
            "选择“是”保存，选择“否”放弃本次修改，选择“取消”留在当前工程。",
            parent=self.root,
        )
        if decision is None:
            return False
        if decision:
            return self._save_project()
        return True

    def _apply_project_metadata(self) -> None:
        try:
            self._sync_project_metadata_from_widgets()
        except ProjectError as exc:
            messagebox.showerror("工程信息错误", str(exc), parent=self.root)
            return
        self._refresh_work_tree("工程")

    def _set_workbench_project(
        self,
        project,
        path: Path | None = None,
    ) -> None:
        self.workbench = WorkbenchSession(project)
        self.project_path = path
        self.project_name_var.set(project.project_name)
        self.project_description_var.set(project.description)
        self.project_description_text.configure(state="normal")
        self.project_description_text.delete("1.0", "end")
        self.project_description_text.insert("1.0", project.description)
        self.project_path_var.set(str(path) if path is not None else "尚未保存")
        self._restore_geometry_widgets()
        self._restore_mesh_widgets()
        self._restore_material_widgets()
        self._restore_process_widgets()
        self._restore_analysis_widgets()
        self._refresh_work_tree("工程")
        if path is None:
            self._saved_project_snapshot = ""
            self._loaded_project_file_sha256 = None
            self._update_window_title()
        else:
            self._accept_saved_snapshot()

    def _new_project(self) -> None:
        if not self._confirm_discard_or_save():
            return
        name = simpledialog.askstring(
            "新建工程",
            "请输入工程名称：",
            initialvalue="未命名工程",
            parent=self.root,
        )
        if name is None:
            return
        try:
            project = create_project(name)
        except ProjectError as exc:
            messagebox.showerror("新建失败", str(exc), parent=self.root)
            return
        self._set_workbench_project(project)

    def _open_project(self) -> None:
        if not self._confirm_discard_or_save():
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="打开 GrindCAE 工程",
            filetypes=(
                ("GrindCAE 工程", "*.gcae"),
                ("旧版 GrindCAE 工程", "*.gcase.json"),
            ),
        )
        if not path:
            return
        try:
            source = Path(path).resolve()
            project = load_project(source)
        except ProjectError as exc:
            messagebox.showerror("打开工程失败", str(exc), parent=self.root)
            return
        self._set_workbench_project(project, source)
        try:
            self.recent_project_store.add(project, source)
        except ProjectError:
            pass

    def _save_project(self) -> bool:
        try:
            self._sync_project_metadata_from_widgets()
            self._save_analysis_configuration()
        except (ProjectError, ValueError) as exc:
            messagebox.showerror("保存工程失败", str(exc), parent=self.root)
            return False
        path = self.project_path
        if path is None or path.name.lower().endswith(".gcase.json"):
            selected = filedialog.asksaveasfilename(
                parent=self.root,
                title="保存 GrindCAE 工程",
                defaultextension=".gcae",
                initialfile="project.gcae",
                filetypes=(("GrindCAE 工程", "*.gcae"),),
            )
            if not selected:
                return False
            path = Path(selected).resolve()
        elif (
            self._loaded_project_file_sha256 is not None
            and self._project_file_sha256(path) != self._loaded_project_file_sha256
        ):
            messagebox.showerror(
                "工程文件已被修改",
                "当前工程文件在打开后已被其他程序或另一个窗口修改。为避免覆盖，请重新打开磁盘版本或另存为新工程。",
                parent=self.root,
            )
            return False
        try:
            self.project_path = save_project(self.workbench.project, path)
            load_project(self.project_path)
        except ProjectError as exc:
            messagebox.showerror("保存工程失败", str(exc), parent=self.root)
            return False
        self.project_path_var.set(str(self.project_path))
        self._accept_saved_snapshot()
        try:
            self.recent_project_store.add(self.workbench.project, self.project_path)
        except ProjectError:
            pass
        self._refresh_work_tree(self.workbench.current_node)
        messagebox.showinfo(
            "工程保存成功",
            f"工程文件已保存并通过回读校验：\n{self.project_path}",
            parent=self.root,
        )
        return True

    def _populate_recent_projects_menu(self) -> None:
        self.recent_projects_menu.delete(0, "end")
        try:
            entries = self.recent_project_store.load()
        except ProjectError as exc:
            self.recent_projects_menu.add_command(label=f"近期工程不可用：{exc}", state="disabled")
            return
        if not entries:
            self.recent_projects_menu.add_command(label="暂无近期工程", state="disabled")
            return
        for entry in entries:
            exists = Path(entry.path).is_file()
            label = f"{entry.project_name} — {entry.path}"
            self.recent_projects_menu.add_command(
                label=label,
                state="normal" if exists else "disabled",
                command=lambda path=entry.path: self._open_project_path(Path(path)),
            )

    def _open_project_path(self, source: Path) -> None:
        if not self._confirm_discard_or_save():
            return
        try:
            project = load_project(source)
        except ProjectError as exc:
            messagebox.showerror("打开工程失败", str(exc), parent=self.root)
            return
        self._external_result_records.clear()
        self._set_workbench_project(project, source)
        try:
            self.recent_project_store.add(project, source)
        except ProjectError:
            pass

    def _populate_recent_results_menu(self) -> None:
        self.recent_results_menu.delete(0, "end")
        try:
            entries = self.recent_result_store.load()
        except ProjectError as exc:
            self.recent_results_menu.add_command(label=f"近期结果不可用：{exc}", state="disabled")
            return
        for entry in entries[:10]:
            exists = Path(entry.summary_path).is_file()
            definition = ANALYSIS_MODE_DEFINITIONS.get(entry.analysis_type)
            label = f"{definition.display_name if definition else entry.analysis_type} — {entry.created_at}"
            self.recent_results_menu.add_command(
                label=label,
                state="normal" if exists else "disabled",
                command=lambda item=entry: self._show_recent_result(item),
            )
        if entries:
            self.recent_results_menu.add_separator()
        self.recent_results_menu.add_command(label="打开其他结果目录…", command=self._import_result_directory)

    def _show_recent_result(self, entry) -> None:
        try:
            record = entry_as_result_record(entry)
        except (ProjectError, AnalysisResultError, OSError, ValueError) as exc:
            messagebox.showerror("打开近期结果失败", str(exc), parent=self.root)
            return
        self._external_result_records = [record]
        self._refresh_result_center()
        self.workbench_tree.selection_set("结果")
        self._refresh_work_tree("结果")

    def _import_result_directory(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="选择一个正式分析结果目录")
        if not selected:
            return
        try:
            entry = import_result_directory(selected, store=self.recent_result_store)
        except ProjectError as exc:
            messagebox.showerror("结果目录导入失败", str(exc), parent=self.root)
            return
        self._show_recent_result(entry)

    def _relink_result(self, result_id: str) -> None:
        record = next(
            (item for item in self.workbench.project.results if item.result_id == result_id),
            None,
        )
        if record is None:
            messagebox.showerror("重新定位失败", "当前工程中找不到该结果记录。", parent=self.root)
            return
        selected = filedialog.askdirectory(parent=self.root, title="选择移动后的结果目录")
        if not selected:
            return
        try:
            replacement = relink_result_record(record, selected)
        except ProjectError as exc:
            messagebox.showerror("重新定位失败", str(exc), parent=self.root)
            return
        self.workbench.project.results = [
            replacement if item.result_id == result_id else item
            for item in self.workbench.project.results
        ]
        self._write_recovery_snapshot()
        self._refresh_result_center()

    def _export_complete_archive(self) -> None:
        if self.project_path is None or not self._save_project():
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出 GrindCAE 完整归档",
            defaultextension=".gcae-archive",
            initialfile=f"{self.project_path.stem}.gcae-archive",
            filetypes=(("GrindCAE 完整归档", "*.gcae-archive"),),
        )
        if not selected:
            return
        try:
            output = export_project_archive(self.project_path, selected)
        except ProjectError as exc:
            messagebox.showerror("完整归档导出失败", str(exc), parent=self.root)
            return
        messagebox.showinfo("完整归档导出成功", f"归档已生成并通过校验：\n{output}", parent=self.root)

    def _import_complete_archive(self) -> None:
        if not self._confirm_discard_or_save():
            return
        source = filedialog.askopenfilename(
            parent=self.root,
            title="导入 GrindCAE 完整归档",
            filetypes=(("GrindCAE 完整归档", "*.gcae-archive"),),
        )
        if not source:
            return
        target = filedialog.askdirectory(parent=self.root, title="选择空的归档导入目录", mustexist=False)
        if not target:
            return
        try:
            imported = import_project_archive(source, target)
            project = load_project(imported.project_path)
        except ProjectError as exc:
            messagebox.showerror("完整归档导入失败", str(exc), parent=self.root)
            return
        self._set_workbench_project(project, imported.project_path)
        messagebox.showinfo("完整归档导入成功", f"工程已恢复：\n{imported.project_path}", parent=self.root)

    def _analysis_settings_from_widgets(self) -> AnalysisSettings:
        base = AnalysisSettings.recommended()
        payload = base.to_dict()
        if getattr(self, "_literature_case_payload", None) is not None:
            payload["literature_case"] = self._literature_case_payload
        payload["scan_base_position_count"] = int(self.analysis_settings_vars["scan_base_position_count"].get())
        payload["loading_step_count"] = int(self.analysis_settings_vars["loading_step_count"].get())
        payload["unloading_step_count"] = int(self.analysis_settings_vars["unloading_step_count"].get())
        payload["newton"]["residual_relative_tolerance"] = float(self.analysis_settings_vars["newton_residual_relative_tolerance"].get())
        payload["newton"]["maximum_iterations"] = int(self.analysis_settings_vars["newton_maximum_iterations"].get())
        payload["history_transition"]["base_substep_count"] = int(self.analysis_settings_vars["history_base_substep_count"].get())
        payload["history_transition"]["final_unloading_substep_count"] = int(self.analysis_settings_vars["history_final_unloading_substep_count"].get())
        payload["grain_population"]["minimum_equivalent_group_count"] = int(self.analysis_settings_vars["grain_minimum_group_count"].get())
        payload["grain_population"]["maximum_equivalent_group_count"] = int(self.analysis_settings_vars["grain_maximum_group_count"].get())
        payload["grain_population"]["random_seed"] = int(self.analysis_settings_vars["grain_random_seed"].get())
        payload["load_distribution"]["point_count"] = int(self.analysis_settings_vars["load_point_count"].get())
        single_grain = payload["single_grain"]
        single_grain["workpiece_width_m"] = float(self.analysis_settings_vars["single_grain_workpiece_width_um"].get()) * 1e-6
        single_grain["workpiece_height_m"] = float(self.analysis_settings_vars["single_grain_workpiece_height_um"].get()) * 1e-6
        single_grain["mesh_target_size_m"] = float(self.analysis_settings_vars["single_grain_mesh_size_um"].get()) * 1e-6
        single_grain["workpiece_thickness_m"] = float(self.analysis_settings_vars["single_grain_thickness_mm"].get()) * 1e-3
        single_grain["grain_type"] = self.analysis_settings_vars["single_grain_type"].get().strip()
        single_grain["grain_radius_m"] = float(self.analysis_settings_vars["single_grain_radius_um"].get()) * 1e-6
        single_grain["grain_arc_start_angle_rad"] = math.radians(float(self.analysis_settings_vars["single_grain_arc_start_angle_deg"].get()))
        single_grain["grain_arc_end_angle_rad"] = math.radians(float(self.analysis_settings_vars["single_grain_arc_end_angle_deg"].get()))
        single_grain["grain_tip_radius_m"] = float(self.analysis_settings_vars["single_grain_tip_radius_um"].get()) * 1e-6
        single_grain["grain_wedge_height_m"] = float(self.analysis_settings_vars["single_grain_wedge_height_um"].get()) * 1e-6
        single_grain["intrinsic_rake_angle_rad"] = math.radians(float(self.analysis_settings_vars["single_grain_intrinsic_rake_angle_deg"].get()))
        single_grain["intrinsic_clearance_angle_rad"] = math.radians(float(self.analysis_settings_vars["single_grain_intrinsic_clearance_angle_deg"].get()))
        single_grain["grain_pose_angle_rad"] = math.radians(float(self.analysis_settings_vars["single_grain_pose_angle_deg"].get()))
        single_grain["nominal_scratch_direction"] = self.analysis_settings_vars["single_grain_nominal_direction"].get().strip()
        single_grain["preset_removal_mode"] = self.analysis_settings_vars["single_grain_preset_removal_mode"].get().strip()
        raw_ids = self.analysis_settings_vars["single_grain_preset_removal_element_ids"].get().strip()
        single_grain["preset_removal_element_ids"] = tuple(int(item.strip()) for item in raw_ids.split(",") if item.strip())
        single_grain["preset_removal_polygon_vertices_m"] = tuple(tuple(float(value) for value in point) for point in json.loads(self.analysis_settings_vars["single_grain_preset_removal_polygon_json"].get()))
        single_grain["initial_clearance_m"] = float(self.analysis_settings_vars["single_grain_initial_clearance_um"].get()) * 1e-6
        single_grain["indentation_depth_m"] = float(self.analysis_settings_vars["single_grain_indentation_depth_um"].get()) * 1e-6
        single_grain["scratch_distance_m"] = float(self.analysis_settings_vars["single_grain_scratch_distance_um"].get()) * 1e-6
        single_grain["normal_algorithm"] = self.analysis_settings_vars["single_grain_normal_algorithm"].get().strip()
        single_grain["normal_penalty_factor"] = float(self.analysis_settings_vars["single_grain_penalty_factor"].get())
        single_grain["tangential_penalty_ratio"] = float(self.analysis_settings_vars["single_grain_tangential_penalty_ratio"].get())
        single_grain["augmented_relaxation"] = float(self.analysis_settings_vars["single_grain_augmented_relaxation"].get())
        single_grain["augmented_maximum_iterations"] = int(self.analysis_settings_vars["single_grain_augmented_maximum_iterations"].get())
        single_grain["friction_coefficient"] = float(self.analysis_settings_vars["single_grain_friction_coefficient"].get())
        single_grain["friction_regularization_ratio"] = float(self.analysis_settings_vars["single_grain_friction_regularization_ratio"].get())
        enabled_text = self.analysis_settings_vars["single_grain_damage_enabled"].get().strip().lower()
        if enabled_text not in {"true", "false"}:
            raise ValueError("损伤分离开关必须填写 true 或 false。")
        single_grain["damage_separation_enabled"] = enabled_text == "true"
        single_grain["damage_evolution"] = self.analysis_settings_vars["single_grain_damage_evolution"].get().strip()
        single_grain["damage_failure_strain_table"] = tuple(
            tuple(float(value) for value in row)
            for row in json.loads(self.analysis_settings_vars["single_grain_damage_failure_table_json"].get())
        )
        single_grain["damage_triaxiality_cutoff"] = float(self.analysis_settings_vars["single_grain_damage_triaxiality_cutoff"].get())
        single_grain["damage_fracture_energy_J_per_m2"] = float(self.analysis_settings_vars["single_grain_damage_fracture_energy"].get())
        single_grain["damage_event_tolerance"] = float(self.analysis_settings_vars["single_grain_damage_event_tolerance"].get())
        single_grain["damage_maximum_separation_events_per_position"] = int(self.analysis_settings_vars["single_grain_damage_max_events"].get())
        single_grain["damage_area_absolute_tolerance_m2"] = float(self.analysis_settings_vars["single_grain_damage_area_tolerance_m2"].get())
        single_grain["damage_parameter_source"] = self.analysis_settings_vars["single_grain_damage_parameter_source"].get().strip()
        single_grain["damage_calibration_status"] = self.analysis_settings_vars["single_grain_damage_calibration_status"].get().strip()
        single_grain["damage_applicability_notes"] = self.analysis_settings_vars["single_grain_damage_applicability_notes"].get().strip()
        return AnalysisSettings.from_mapping(payload)

    def _apply_analysis_settings(self, settings: AnalysisSettings) -> None:
        self._literature_case_payload = settings.literature_case
        self.analysis_settings_vars["scan_base_position_count"].set(str(settings.scan_base_position_count))
        self.analysis_settings_vars["loading_step_count"].set(str(settings.loading_step_count))
        self.analysis_settings_vars["unloading_step_count"].set(str(settings.unloading_step_count))
        self.analysis_settings_vars["newton_residual_relative_tolerance"].set(str(settings.newton.residual_relative_tolerance))
        self.analysis_settings_vars["newton_maximum_iterations"].set(str(settings.newton.maximum_iterations))
        self.analysis_settings_vars["history_base_substep_count"].set(str(settings.history_transition.base_substep_count))
        self.analysis_settings_vars["history_final_unloading_substep_count"].set(str(settings.history_transition.final_unloading_substep_count))
        self.analysis_settings_vars["grain_minimum_group_count"].set(str(settings.grain_population.minimum_equivalent_group_count))
        self.analysis_settings_vars["grain_maximum_group_count"].set(str(settings.grain_population.maximum_equivalent_group_count))
        self.analysis_settings_vars["grain_random_seed"].set(str(settings.grain_population.random_seed))
        self.analysis_settings_vars["load_point_count"].set(str(settings.load_distribution.point_count))
        single_grain = settings.single_grain
        self.analysis_settings_vars["single_grain_workpiece_width_um"].set(str(single_grain.workpiece_width_m * 1e6))
        self.analysis_settings_vars["single_grain_workpiece_height_um"].set(str(single_grain.workpiece_height_m * 1e6))
        self.analysis_settings_vars["single_grain_mesh_size_um"].set(str(single_grain.mesh_target_size_m * 1e6))
        self.analysis_settings_vars["single_grain_thickness_mm"].set(str(single_grain.workpiece_thickness_m * 1e3))
        self.analysis_settings_vars["single_grain_type"].set(single_grain.grain_type)
        self.analysis_settings_vars["single_grain_radius_um"].set(str(single_grain.grain_radius_m * 1e6))
        self.analysis_settings_vars["single_grain_arc_start_angle_deg"].set(str(math.degrees(single_grain.grain_arc_start_angle_rad)))
        self.analysis_settings_vars["single_grain_arc_end_angle_deg"].set(str(math.degrees(single_grain.grain_arc_end_angle_rad)))
        self.analysis_settings_vars["single_grain_tip_radius_um"].set(str(single_grain.grain_tip_radius_m * 1e6))
        self.analysis_settings_vars["single_grain_wedge_height_um"].set(str(single_grain.grain_wedge_height_m * 1e6))
        self.analysis_settings_vars["single_grain_intrinsic_rake_angle_deg"].set(str(math.degrees(single_grain.intrinsic_rake_angle_rad)))
        self.analysis_settings_vars["single_grain_intrinsic_clearance_angle_deg"].set(str(math.degrees(single_grain.intrinsic_clearance_angle_rad)))
        self.analysis_settings_vars["single_grain_pose_angle_deg"].set(str(math.degrees(single_grain.grain_pose_angle_rad)))
        self.analysis_settings_vars["single_grain_nominal_direction"].set(single_grain.nominal_scratch_direction)
        self.analysis_settings_vars["single_grain_preset_removal_mode"].set(single_grain.preset_removal_mode)
        self.analysis_settings_vars["single_grain_preset_removal_element_ids"].set(",".join(str(item) for item in single_grain.preset_removal_element_ids))
        self.analysis_settings_vars["single_grain_preset_removal_polygon_json"].set(json.dumps(single_grain.preset_removal_polygon_vertices_m))
        self.analysis_settings_vars["single_grain_initial_clearance_um"].set(str(single_grain.initial_clearance_m * 1e6))
        self.analysis_settings_vars["single_grain_indentation_depth_um"].set(str(single_grain.indentation_depth_m * 1e6))
        self.analysis_settings_vars["single_grain_scratch_distance_um"].set(str(single_grain.scratch_distance_m * 1e6))
        self.analysis_settings_vars["single_grain_normal_algorithm"].set(single_grain.normal_algorithm)
        self.analysis_settings_vars["single_grain_penalty_factor"].set(str(single_grain.normal_penalty_factor))
        self.analysis_settings_vars["single_grain_tangential_penalty_ratio"].set(str(single_grain.tangential_penalty_ratio))
        self.analysis_settings_vars["single_grain_augmented_relaxation"].set(str(single_grain.augmented_relaxation))
        self.analysis_settings_vars["single_grain_augmented_maximum_iterations"].set(str(single_grain.augmented_maximum_iterations))
        self.analysis_settings_vars["single_grain_friction_coefficient"].set(str(single_grain.friction_coefficient))
        self.analysis_settings_vars["single_grain_friction_regularization_ratio"].set(str(single_grain.friction_regularization_ratio))
        self.analysis_settings_vars["single_grain_damage_enabled"].set(str(single_grain.damage_separation_enabled).lower())
        self.analysis_settings_vars["single_grain_damage_evolution"].set(single_grain.damage_evolution)
        self.analysis_settings_vars["single_grain_damage_failure_table_json"].set(json.dumps(single_grain.damage_failure_strain_table))
        self.analysis_settings_vars["single_grain_damage_triaxiality_cutoff"].set(str(single_grain.damage_triaxiality_cutoff))
        self.analysis_settings_vars["single_grain_damage_fracture_energy"].set(str(single_grain.damage_fracture_energy_J_per_m2))
        self.analysis_settings_vars["single_grain_damage_event_tolerance"].set(str(single_grain.damage_event_tolerance))
        self.analysis_settings_vars["single_grain_damage_max_events"].set(str(single_grain.damage_maximum_separation_events_per_position))
        self.analysis_settings_vars["single_grain_damage_area_tolerance_m2"].set(str(single_grain.damage_area_absolute_tolerance_m2))
        self.analysis_settings_vars["single_grain_damage_parameter_source"].set(single_grain.damage_parameter_source)
        self.analysis_settings_vars["single_grain_damage_calibration_status"].set(single_grain.damage_calibration_status)
        self.analysis_settings_vars["single_grain_damage_applicability_notes"].set(single_grain.damage_applicability_notes)

    def _restore_recommended_analysis_settings(self) -> None:
        self._apply_analysis_settings(AnalysisSettings.recommended())
        self._refresh_analysis_workspace()

    def _apply_analysis_advanced_parameter_state(self, mode_id: str) -> None:
        active = set(analysis_mode_parameter_keys(mode_id))
        if mode_id == "single_grain_high_fidelity":
            grain_type = self.analysis_settings_vars["single_grain_type"].get().strip()
            shape_fields = {
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
            }
            allowed_by_shape = {
                "rigid_circle": {"single_grain_radius_um"},
                "rigid_circular_arc": {
                    "single_grain_radius_um", "single_grain_arc_start_angle_deg",
                    "single_grain_arc_end_angle_deg",
                },
                "rounded_circle": {
                    "single_grain_radius_um", "single_grain_nominal_direction",
                    "single_grain_preset_removal_mode",
                    "single_grain_preset_removal_element_ids",
                    "single_grain_preset_removal_polygon_json",
                },
                "rounded_wedge": {
                    "single_grain_tip_radius_um", "single_grain_wedge_height_um",
                    "single_grain_intrinsic_rake_angle_deg",
                    "single_grain_intrinsic_clearance_angle_deg",
                    "single_grain_pose_angle_deg", "single_grain_nominal_direction",
                    "single_grain_preset_removal_mode",
                    "single_grain_preset_removal_element_ids",
                    "single_grain_preset_removal_polygon_json",
                },
            }.get(grain_type, set())
            active -= shape_fields - allowed_by_shape
            damage_fields = {
                "single_grain_damage_evolution",
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
            enabled = self.analysis_settings_vars["single_grain_damage_enabled"].get().strip().lower() == "true"
            if grain_type not in {"rounded_circle", "rounded_wedge"} or not enabled:
                active -= damage_fields
        for key, entry in self.analysis_advanced_entries.items():
            entry.configure(state="normal" if key in active else "disabled")
            self.analysis_advanced_labels[key].configure(
                style="Heading.TLabel" if key in active else "Disabled.TLabel"
            )
        if not active:
            message = "当前模式使用网格、经验载荷和线弹性材料参数，无额外非线性求解设置。"
        else:
            message = "仅高亮当前分析模式实际读取的高级参数；禁用项的已保存值会保留。"
        self.analysis_advanced_notice_var.set(message)
        if mode_id == "literature_grinding_force":
            self.analysis_advanced_notice_var.set("文献模型使用上方“文献参数设置”；下方FE高级参数不参与计算。")
        if mode_id == "literature_elastoplastic_single_pass":
            self.analysis_advanced_notice_var.set("文献力联算：工程工艺/材料驱动力重算；本页高级设置控制J2历史。3.8.1候选功能；独立实验验证未完成。")
        if hasattr(self, "literature_parameters_button"):
            self.literature_parameters_button.configure(state="normal" if mode_id in {"literature_grinding_force", "literature_elastoplastic_single_pass"} else "disabled")

    def _restore_analysis_widgets(self) -> None:
        try:
            workspace = AnalysisWorkspace.from_project(self.workbench.project)
        except ValueError as exc:
            self.analysis_status_var.set(f"分析设置恢复失败：{exc}")
            workspace = AnalysisWorkspace(self.workbench.project, "linear_elastic_single_position", AnalysisSettings.recommended())
        definition = ANALYSIS_MODE_DEFINITIONS[workspace.mode_id]
        self.analysis_mode_var.set(definition.display_name)
        self.analysis_output_var.set(workspace.output_directory)
        self._apply_analysis_settings(workspace.settings)
        self._refresh_analysis_workspace()

    def _save_analysis_configuration(self) -> None:
        if not hasattr(self, "analysis_mode_var"):
            return
        definition = analysis_mode_by_display_name(self.analysis_mode_var.get())
        settings = self._analysis_settings_from_widgets()
        AnalysisWorkspace(self.workbench.project, definition.mode_id, settings, self.analysis_output_var.get().strip()).save_configuration()

    def _refresh_analysis_workspace(self) -> None:
        if not hasattr(self, "analysis_mode_var"):
            return
        try:
            definition = analysis_mode_by_display_name(self.analysis_mode_var.get())
            self._apply_analysis_advanced_parameter_state(definition.mode_id)
            self._update_calibration_display()
            settings = self._analysis_settings_from_widgets()
            workspace = AnalysisWorkspace(self.workbench.project, definition.mode_id, settings, self.analysis_output_var.get().strip())
            state = workspace.evaluate()
        except (ValueError, TypeError) as exc:
            self.analysis_task_var.set("分析设置无效。")
            self.analysis_checks_var.set(f"分析设置：无效\n{exc}")
            self.analysis_mapping_var.set("尚未生成合法求解输入。")
            self.analysis_run_button.configure(state="disabled")
            return
        self.analysis_task_var.set(state.task_summary)
        self.analysis_checks_var.set("\n".join(state.input_checks))
        self.analysis_mapping_var.set(state.mapping_summary if state.can_run else state.error_message)
        presentation = analysis_presentation_state(self.workbench.project, state)
        self._apply_analysis_presentation(presentation)
        self.workbench_input_var.set("\n".join(analysis_input_summary(
            definition.mode_id, input_valid=state.can_run
        )))
        self.workbench_output_var.set("\n".join(presentation.output_summary))
        self.analysis_run_button.configure(
            state="normal"
            if state.can_run and self.background_task_gate.owner is None
            else "disabled"
        )

    def _apply_analysis_presentation(self, presentation) -> None:
        self.analysis_summary_text.configure(state="normal")
        self.analysis_summary_text.delete("1.0", "end")
        self.analysis_summary_text.insert("1.0", presentation.detail_text)
        self.analysis_summary_text.configure(state="disabled")
        self.analysis_status_var.set(presentation.output_summary[0])
        if presentation.representative_image is None:
            messages = {
                "running": "正在计算，旧结果不会作为当前结果显示。",
                "failed": "当前分析失败；历史结果仍保留用于追溯。",
                "stale": "当前结果已过期，请重新运行当前模式。",
                "unavailable": "最近结果记录不可用，请重新运行当前模式。",
            }
            self.analysis_image_preview.clear(
                messages.get(presentation.status, "当前模式尚无有效代表图。")
            )
            self._active_analysis_output = None
        elif presentation.published_result is not None:
            self.analysis_image_preview.set_image(
                presentation.representative_image, source="analysis-workbench"
            )
            self._active_analysis_output = presentation.published_result.output_directory

    def _choose_analysis_output(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="选择分析输出目录")
        if selected:
            self.analysis_output_var.set(selected)

    def _set_default_analysis_output(self) -> None:
        definition = analysis_mode_by_display_name(self.analysis_mode_var.get())
        self.analysis_output_var.set(str(default_analysis_output_directory(definition.mode_id, project_file=self.project_path)))

    def _start_workbench_analysis(self) -> None:
        if self.project_path is None:
            decision = messagebox.askyesnocancel(
                "工程尚未保存",
                "当前工程尚未保存。保存后，计算结果会默认放在工程相邻目录，便于移动、恢复和归档。\n\n"
                "选择“是”先保存工程，选择“否”继续计算，选择“取消”暂不运行。",
                parent=self.root,
            )
            if decision is None:
                return
            if decision and not self._save_project():
                return
        self._write_recovery_snapshot()
        try:
            definition = analysis_mode_by_display_name(self.analysis_mode_var.get())
            settings = self._analysis_settings_from_widgets()
            workspace = AnalysisWorkspace(self.workbench.project, definition.mode_id, settings, self.analysis_output_var.get().strip())
            state = workspace.evaluate()
            if not state.can_run or state.built is None:
                raise AnalysisRunError(state.error_message or "分析输入未通过校验。")
            if not self.analysis_output_var.get().strip():
                self._set_default_analysis_output()
            output = Path(self.analysis_output_var.get()).expanduser().resolve()
            workspace.output_directory = str(output)
            workspace.save_configuration()
            audit = quick_analysis_audit(state.built, output)
            self.analysis_status_var.set(audit.summary_text)
            if not audit.can_run:
                messagebox.showerror("运行前审查未通过", audit.summary_text, parent=self.root)
                return
            decision = messagebox.askyesnocancel(
                "运行前审查已完成",
                audit.summary_text
                + "\n\n选择“是”：先做短程试算，再正式计算。"
                + "\n选择“否”：直接正式计算。"
                + "\n选择“取消”：暂不运行。",
                parent=self.root,
            )
            if decision is None:
                return
            total = audit.position_count * audit.route_count
            initial_message = "短程试算正在准备" if decision else "正式计算正在准备"
            self._analysis_progress_started_at = time.monotonic()
            self._analysis_progress_current = 0
            self._analysis_progress_total = total
            self._analysis_progress_formal_total = total
            self._analysis_progress_message = initial_message
            self._analysis_progress_phase = "pilot" if decision else "formal"
            self._analysis_progress_updated_at = self._analysis_progress_started_at
            self._analysis_progress_remaining_at_update = None
            self.analysis_progress_detail_var.set(
                format_progress_status(
                    initial_message,
                    estimate_progress(current=0, total=total, elapsed_seconds=0.0),
                )
            )
            self.analysis_progress_var.set(0.0)
            self.analysis_progress.configure(mode="determinate", maximum=total)
            self._active_analysis_built = state.built
            self._active_analysis_output = output
            self.analysis_runner.start(state.built, output, run_pilot=decision)
            self.workbench.begin_analysis(
                mode=definition.mode_id, output_directory=output,
                input_fingerprint=state.built.input_fingerprint,
                input_snapshot=state.built.normalized_input,
            )
        except (ValueError, AnalysisRunError, OSError) as exc:
            self.analysis_status_var.set(f"输入校验失败：{exc}")
            messagebox.showerror("无法运行分析", str(exc), parent=self.root)
            return
        self._refresh_analysis_workspace()
        self.analysis_run_button.configure(state="disabled")
        self._refresh_work_tree("分析")

    def _poll_analysis_events(self) -> None:
        try:
            while True:
                self._handle_analysis_event(self.analysis_runner.events.get_nowait())
        except Empty:
            pass
        self.root.after(100, self._poll_analysis_events)

    def _handle_analysis_event(self, event: AnalysisEvent) -> None:
        self.analysis_status_var.set(event.message)
        if event.kind == "progress" and event.current is not None and event.total:
            self.analysis_progress.stop()
            self.analysis_progress.configure(mode="determinate", maximum=event.total)
            self.analysis_progress_var.set(event.current)
            estimate = estimate_progress(
                current=event.current,
                total=event.total,
                elapsed_seconds=event.elapsed_seconds or 0.0,
            )
            if event.remaining_seconds is not None:
                estimate = estimate.__class__(
                    estimate.current,
                    estimate.total,
                    estimate.fraction,
                    estimate.percent,
                    estimate.elapsed_seconds,
                    max(0.0, event.remaining_seconds),
                )
            phase = "短程试算" if event.phase == "pilot" else "正式计算"
            now = time.monotonic()
            if (
                self._analysis_progress_started_at is None
                or event.phase != self._analysis_progress_phase
            ):
                self._analysis_progress_started_at = now - estimate.elapsed_seconds
            self._analysis_progress_current = estimate.current
            self._analysis_progress_total = estimate.total
            self._analysis_progress_message = event.message
            self._analysis_progress_phase = event.phase or "formal"
            self._analysis_progress_updated_at = now
            self._analysis_progress_remaining_at_update = estimate.remaining_seconds
            self.analysis_progress_detail_var.set(
                format_progress_status(f"{phase}：{event.message}", estimate)
            )
        elif event.kind == "pilot_success":
            now = time.monotonic()
            self._analysis_progress_started_at = now
            self._analysis_progress_current = 0
            self._analysis_progress_total = self._analysis_progress_formal_total
            self._analysis_progress_message = "正在准备"
            self._analysis_progress_phase = "formal"
            self._analysis_progress_updated_at = now
            self._analysis_progress_remaining_at_update = None
            self.analysis_progress_var.set(0.0)
            self.analysis_progress.configure(
                mode="determinate", maximum=self._analysis_progress_formal_total
            )
            self.analysis_progress_detail_var.set(
                format_progress_status(
                    "正式计算：正在准备",
                    estimate_progress(
                        current=0,
                        total=self._analysis_progress_formal_total,
                        elapsed_seconds=0.0,
                    ),
                )
            )
        elif event.kind == "success" and event.result is not None and self._active_analysis_built is not None:
            if event.result.mode == "literature_grinding_force":
                self._analysis_progress_current = self._analysis_progress_total
                self._analysis_progress_message = "计算完成，结果已验证"
                self._analysis_progress_remaining_at_update = 0.0
                self.analysis_progress_var.set(self._analysis_progress_total)
                self._refresh_analysis_progress_clock()
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self.workbench.register_published_result(
                mode=event.result.mode, result_format=event.result.result_format,
                output_directory=event.result.output_directory, summary_path=event.result.summary_path,
                artifact_paths={"representative_image": event.result.representative_image},
                created_at=created_at, input_fingerprint=self._active_analysis_built.input_fingerprint,
            )
            latest = self.workbench.project.latest_analysis_result
            if latest is not None:
                try:
                    self.recent_result_store.add(
                        recent_entry_from_record(
                            latest, project_id=self.workbench.project.project_id
                        )
                    )
                except ProjectError:
                    pass
            self._write_recovery_snapshot()
            self._show_workbench_analysis_result(event.result)
            self._refresh_work_tree("分析")
        elif event.kind == "error":
            self.workbench.mark_analysis_failed(event.message)
            self._refresh_analysis_workspace()
            self._refresh_work_tree("分析")
            messagebox.showerror("GrindCAE 分析失败", event.message, parent=self.root)
        elif event.kind == "finished":
            self.analysis_progress.stop()
            self._active_analysis_built = None
            self._refresh_analysis_workspace()

    def _toggle_analysis_progress_details(self) -> None:
        self.analysis_progress_details_visible = not self.analysis_progress_details_visible
        if self.analysis_progress_details_visible:
            self.analysis_progress_detail_label.pack(fill="x", pady=(4, 0))
            self.analysis_progress.pack(fill="x", pady=(5, 0))
            self.analysis_progress_toggle_button.configure(text="隐藏进度详情")
        else:
            self.analysis_progress_detail_label.pack_forget()
            self.analysis_progress.pack_forget()
            self.analysis_progress_toggle_button.configure(text="显示进度详情")

    def _refresh_analysis_progress_clock(self, *, now: float | None = None) -> None:
        if self._analysis_progress_started_at is None:
            return
        elapsed = max(
            0.0,
            (time.monotonic() if now is None else float(now))
            - self._analysis_progress_started_at,
        )
        estimate = estimate_progress(
            current=self._analysis_progress_current,
            total=self._analysis_progress_total,
            elapsed_seconds=elapsed,
        )
        if (
            self._analysis_progress_remaining_at_update is not None
            and self._analysis_progress_updated_at is not None
        ):
            estimate = estimate.__class__(
                estimate.current,
                estimate.total,
                estimate.fraction,
                estimate.percent,
                estimate.elapsed_seconds,
                max(
                    0.0,
                    self._analysis_progress_remaining_at_update
                    - ((time.monotonic() if now is None else float(now)) - self._analysis_progress_updated_at),
                ),
            )
        phase = "短程试算" if self._analysis_progress_phase == "pilot" else "正式计算"
        self.analysis_progress_detail_var.set(
            format_progress_status(
                f"{phase}：{self._analysis_progress_message}", estimate
            )
        )

    def _tick_analysis_progress_clock(self) -> None:
        if self.analysis_runner.is_running:
            self._refresh_analysis_progress_clock()
        self.root.after(1000, self._tick_analysis_progress_clock)

    def _show_workbench_analysis_result(self, result: AnalysisPublishedResult) -> None:
        self.analysis_summary_text.configure(state="normal")
        self.analysis_summary_text.delete("1.0", "end")
        self.analysis_summary_text.insert("1.0", result.summary_text)
        self.analysis_summary_text.configure(state="disabled")
        self.analysis_image_preview.set_image(result.representative_image, source="analysis-workbench")
        self.analysis_status_var.set(f"已完成：{result.display_name}\n输出目录：{result.output_directory}")
        self._active_analysis_output = result.output_directory
        self.workbench_output_var.set("\n".join((
            "计算状态：已完成",
            f"分析模式：{result.display_name}",
            f"代表图：{result.representative_image.name}",
            f"输出目录：{result.output_directory}",
        )))

    def _open_workbench_analysis_output(self) -> None:
        raw = self._active_analysis_output or (Path(self.analysis_output_var.get()).expanduser() if self.analysis_output_var.get().strip() else None)
        if raw is not None:
            self._open_path(Path(raw))

    def _refresh_work_tree(self, selected_node: str | None = None) -> None:
        existing = self.work_tree.get_children("")
        for item in existing:
            self.work_tree.delete(item)
        selected_item = None
        for name, status in self.workbench.tree_rows():
            item = self.work_tree.insert("", "end", text=name, values=(status,))
            if name == (selected_node or self.workbench.current_node):
                selected_item = item
        if selected_item is not None:
            self.work_tree.selection_set(selected_item)
            self.work_tree.focus(selected_item)
            self.work_tree.see(selected_item)
        self._show_workbench_page(
            self.workbench.select_node(selected_node or self.workbench.current_node)
        )

    def _on_work_tree_selected(self, _event=None) -> None:
        selection = self.work_tree.selection()
        if not selection:
            return
        name = self.work_tree.item(selection[0], "text")
        self._show_workbench_page(self.workbench.select_node(name))

    def _show_workbench_page(self, page: WorkbenchPage) -> None:
        self.workbench_scroll_canvas.yview_moveto(0.0)
        self.workbench_title_var.set(page.title)
        self.workbench_status_var.set(page.status)
        self.workbench_description_var.set(page.description)
        self.workbench_input_var.set("\n".join(page.input_summary) or "暂无输入")
        self.workbench_output_var.set("\n".join(page.output_summary) or "暂无输出")
        self.workbench_limitation_var.set("\n".join(page.limitations))
        if page.title == "结果":
            self.workbench_summaries.pack_forget()
        elif not self.workbench_summaries.winfo_manager():
            self.workbench_summaries.pack(
                fill="both", expand=True, before=self.workbench_limitation_label
            )
        for child in self.workbench_action_host.winfo_children():
            child.destroy()
        if page.title == "分析":
            ttk.Button(
                self.workbench_action_host,
                text="打开高级兼容计算",
                command=lambda: self.main_notebook.select(self.legacy_tab),
            ).pack(side="left")
        elif page.title == "结果" and self.workbench.project.latest_analysis_result is not None:
            ttk.Button(
                self.workbench_action_host,
                text="打开结果目录",
                command=self.result_center_view.open_selected_result_directory,
            ).pack(side="left")
        editors = (
            self.project_editor,
            self.geometry_editor,
            self.mesh_editor,
            self.material_editor,
            self.process_editor,
            self.analysis_editor,
            self.result_center_editor,
        )
        self.result_center_editor.pack_forget()
        if page.title == "工程":
            self.geometry_editor.pack_forget()
            self.mesh_editor.pack_forget()
            self.material_editor.pack_forget()
            self.process_editor.pack_forget()
            self.analysis_editor.pack_forget()
            if not self.project_editor.winfo_manager():
                self.project_editor.pack(
                    fill="x", pady=(0, 8), before=self.workbench_summaries
                )
            self.project_name_entry.configure(state="normal")
            self.project_description_text.configure(state="normal")
        elif page.title == "几何":
            self.project_editor.pack_forget()
            self.mesh_editor.pack_forget()
            self.material_editor.pack_forget()
            self.process_editor.pack_forget()
            self.analysis_editor.pack_forget()
            if not self.geometry_editor.winfo_manager():
                self.geometry_editor.pack(
                    fill="both", expand=True, pady=(0, 8), before=self.workbench_summaries
                )
            self._restore_geometry_widgets()
        elif page.title == "网格":
            self.project_editor.pack_forget()
            self.geometry_editor.pack_forget()
            self.material_editor.pack_forget()
            self.process_editor.pack_forget()
            self.analysis_editor.pack_forget()
            if not self.mesh_editor.winfo_manager():
                self.mesh_editor.pack(
                    fill="both", expand=True, pady=(0, 8), before=self.workbench_summaries
                )
            self._restore_mesh_widgets()
        elif page.title == "材料":
            self.project_editor.pack_forget()
            self.geometry_editor.pack_forget()
            self.mesh_editor.pack_forget()
            self.process_editor.pack_forget()
            self.analysis_editor.pack_forget()
            if not self.material_editor.winfo_manager():
                self.material_editor.pack(
                    fill="both", expand=True, pady=(0, 8), before=self.workbench_summaries
                )
            self._restore_material_widgets()
        elif page.title == "工艺":
            self.project_editor.pack_forget()
            self.geometry_editor.pack_forget()
            self.mesh_editor.pack_forget()
            self.material_editor.pack_forget()
            self.analysis_editor.pack_forget()
            if not self.process_editor.winfo_manager():
                self.process_editor.pack(
                    fill="both", expand=True, pady=(0, 8), before=self.workbench_summaries
                )
            self._restore_process_widgets()
        elif page.title == "分析":
            self.project_editor.pack_forget()
            self.geometry_editor.pack_forget()
            self.mesh_editor.pack_forget()
            self.material_editor.pack_forget()
            self.process_editor.pack_forget()
            if not self.analysis_editor.winfo_manager():
                self.analysis_editor.pack(fill="both", expand=True, pady=(0, 8), before=self.workbench_summaries)
            self._refresh_analysis_workspace()
        elif page.title == "结果":
            for editor in editors:
                if editor is not self.result_center_editor:
                    editor.pack_forget()
            if not self.result_center_editor.winfo_manager():
                self.result_center_editor.pack(fill="both", expand=True, pady=(0, 8), before=self.workbench_limitation_label)
            self._refresh_result_center()
        else:
            self.project_editor.pack_forget()
            self.geometry_editor.pack_forget()
            self.mesh_editor.pack_forget()
            self.material_editor.pack_forget()
            self.process_editor.pack_forget()
            self.analysis_editor.pack_forget()
            self.result_center_editor.pack_forget()

    def _current_result_center_input(self) -> tuple[str | None, str | None]:
        try:
            definition = analysis_mode_by_display_name(self.analysis_mode_var.get())
            settings = self._analysis_settings_from_widgets()
            workspace = AnalysisWorkspace(
                self.workbench.project,
                definition.mode_id,
                settings,
                self.analysis_output_var.get().strip(),
            ).evaluate()
        except (ValueError, TypeError):
            return None, None
        return definition.mode_id, workspace.built.input_fingerprint if workspace.built is not None else None

    def _refresh_result_center(self) -> None:
        mode, fingerprint = self._current_result_center_input()
        project = self.workbench.project
        if self._external_result_records:
            project = copy.copy(project)
            project.results = [*project.results, *self._external_result_records]
        self.result_center_view.set_state(
            build_result_center_state(project, mode, fingerprint)
        )

    def _material_display_input(self) -> MaterialDisplayInput:
        return MaterialDisplayInput(
            label=self.material_variables["label"].get(),
            behavior=_MATERIAL_BEHAVIOR_LABELS.get(self.material_variables["behavior"].get(), ""),
            family="ductile_metal" if self.material_variables["family"].get() == "延性金属" else "unsupported",
            elastic_modulus_gpa=self.material_variables["elastic_modulus_gpa"].get(),
            poisson_ratio=self.material_variables["poisson_ratio"].get(),
            yield_strength_mpa=self.material_variables["yield_strength_mpa"].get(),
            tangent_modulus_gpa=self.material_variables["tangent_modulus_gpa"].get(),
            parameter_source=_MATERIAL_SOURCE_LABELS.get(self.material_variables["parameter_source"].get(), ""),
            calibration_status=_CALIBRATION_LABELS.get(self.material_variables["calibration_status"].get(), ""),
            source_description=self.material_variables["source_description"].get(),
            applicability_notes=self.material_variables["applicability_notes"].get(),
        )

    def _update_material_behavior(self, *_args: object) -> None:
        state = "disabled" if self.material_variables["behavior"].get() == "线弹性" else "normal"
        for entry in self.material_plastic_entries.values():
            entry.configure(state=state)

    def _set_material_report(self, text: str) -> None:
        self.material_report_text.configure(state="normal")
        self.material_report_text.delete("1.0", "end")
        self.material_report_text.insert("1.0", text)
        self.material_report_text.configure(state="disabled")

    def _show_material_report(self, case: WorkbenchMaterialCase) -> None:
        summary = case.material_summary()
        lines = ["材料状态：有效", *(f"{key}：{value}" for key, value in summary.items())]
        lines.extend(("", f"来源说明：{case.source_description or '未填写'}", f"适用范围：{case.applicability_notes or '未填写'}"))
        if case.parameter_source == "builtin_demo":
            lines.extend(("", "这些参数只用于软件流程演示，未针对具体钢号进行实验标定。"))
        self._set_material_report("\n".join(lines))

    def _save_material_parameters(self) -> None:
        try:
            case = self._material_display_input().build_case()
            self.workbench.register_material(case)
        except (MaterialValidationError, ProjectError) as exc:
            messagebox.showerror("材料参数保存失败", str(exc), parent=self.root)
            return
        self._show_material_report(case)
        self._refresh_work_tree("材料")

    def _restore_material_widgets(self) -> None:
        material = self.workbench.project.material
        if material is None:
            case = MaterialDisplayInput().build_case()
        else:
            try:
                case = WorkbenchMaterialCase.from_mapping(material)
            except MaterialValidationError as exc:
                self._set_material_report(str(exc))
                return
        reverse_behavior = {value: key for key, value in _MATERIAL_BEHAVIOR_LABELS.items()}
        reverse_source = {value: key for key, value in _MATERIAL_SOURCE_LABELS.items()}
        reverse_calibration = {value: key for key, value in _CALIBRATION_LABELS.items()}
        values = {
            "label": case.label,
            "behavior": reverse_behavior[case.behavior],
            "family": "延性金属",
            "elastic_modulus_gpa": f"{case.elastic_modulus_pa / 1.0e9:g}",
            "poisson_ratio": f"{case.poisson_ratio:g}",
            "yield_strength_mpa": "" if case.yield_strength_pa is None else f"{case.yield_strength_pa / 1.0e6:g}",
            "tangent_modulus_gpa": "" if case.tangent_modulus_pa is None else f"{case.tangent_modulus_pa / 1.0e9:g}",
            "parameter_source": reverse_source.get(case.parameter_source, "用户自定义"),
            "calibration_status": reverse_calibration[case.calibration_status],
            "source_description": case.source_description,
            "applicability_notes": case.applicability_notes,
        }
        for name, value in values.items():
            self.material_variables[name].set(value)
        self._update_material_behavior()
        self._show_material_report(case) if material is not None else self._set_material_report("材料尚未配置。\n演示默认值可用于工作流试用，工程应用前需要标定。")

    def _wheel_display_input(self) -> WheelDisplayInput:
        return WheelDisplayInput(
            abrasive_material="aluminum_oxide" if self.wheel_variables["abrasive_material"].get() == "刚玉" else "unsupported",
            resolution_path=_GRIT_PATH_LABELS.get(self.wheel_variables["resolution_path"].get(), ""),
            grit_designation=self.wheel_variables["grit_designation"].get(),
            manufacturer=self.wheel_variables["manufacturer"].get(),
            product_model=self.wheel_variables["product_model"].get(),
            diameter_lower_um=self.wheel_variables["diameter_lower_um"].get(),
            representative_diameter_um=self.wheel_variables["representative_diameter_um"].get(),
            diameter_upper_um=self.wheel_variables["diameter_upper_um"].get(),
            manufacturer_source=self.wheel_variables["manufacturer_source"].get(),
            manufacturer_data_status=self.wheel_variables["manufacturer_data_status"].get(),
        )

    def _process_display_input(self) -> ProcessDisplayInput:
        return ProcessDisplayInput(
            wheel_surface_speed_m_per_s=self.process_variables["wheel_surface_speed_m_per_s"].get(),
            workpiece_feed_speed_m_per_s=self.process_variables["workpiece_feed_speed_m_per_s"].get(),
            grinding_width_mm=self.process_variables["grinding_width_mm"].get(),
            specific_grinding_energy_j_per_mm3=self.process_variables["specific_grinding_energy_j_per_mm3"].get(),
            normal_to_tangential_force_ratio=self.process_variables["normal_to_tangential_force_ratio"].get(),
            calibration_id=self.process_variables["calibration_id"].get(),
            calibration_status=_CALIBRATION_LABELS.get(self.process_variables["calibration_status"].get(), ""),
            source_description=self.process_variables["source_description"].get(),
            applicability_notes=self.process_variables["applicability_notes"].get(),
        )

    def _set_process_report(self, text: str) -> None:
        self.process_report_text.configure(state="normal")
        self.process_report_text.delete("1.0", "end")
        self.process_report_text.insert("1.0", text)
        self.process_report_text.configure(state="disabled")

    def _update_grit_path(self, *_args: object) -> None:
        path = _GRIT_PATH_LABELS.get(self.wheel_variables["resolution_path"].get(), "")
        if path == "manufacturer_override":
            self.manufacturer_frame.grid()
            self.grit_designation_combo.configure(values=())
        else:
            self.manufacturer_frame.grid_remove()
            try:
                designations = available_grit_designations(path)
            except ValueError:
                designations = ()
            self.grit_designation_combo.configure(values=designations)
            if self.wheel_variables["grit_designation"].get() not in designations:
                self.wheel_variables["grit_designation"].set(designations[0] if designations else "")
        self._resolved_workbench_wheel = None

    def _resolve_workbench_wheel(self) -> None:
        try:
            wheel = self._wheel_display_input().build_case()
        except ProcessValidationError as exc:
            messagebox.showerror("粒度解析失败", str(exc), parent=self.root)
            return
        self._resolved_workbench_wheel = wheel
        summary = wheel.wheel_summary()
        self._set_process_report("\n".join(["粒度解析状态：有效", *(f"{key}：{value}" for key, value in summary.items())]))

    def _save_process_parameters(self) -> None:
        geometry = self.workbench.project.geometry
        if geometry is None:
            messagebox.showerror("工艺参数保存失败", "请先完成二维几何建模。", parent=self.root)
            return
        self._resolve_workbench_wheel()
        wheel = self._resolved_workbench_wheel
        if wheel is None:
            return
        try:
            process = self._process_display_input().build_case(geometry, wheel)
            self.workbench.register_process(wheel, process)
        except (ProcessValidationError, ProjectError) as exc:
            messagebox.showerror("工艺参数保存失败", str(exc), parent=self.root)
            return
        wheel_summary = wheel.wheel_summary()
        process_summary = process.process_summary()
        self._set_process_report("\n".join(["工艺状态：有效", *(f"{key}：{value}" for key, value in wheel_summary.items()), "", *(f"{key}：{value}" for key, value in process_summary.items()), "", "当前比磨削能和力比属于演示参数，工程应用前需要实验或可靠文献标定。"] ))
        self._refresh_process_grain_preview()
        self._refresh_work_tree("工艺")

    def _default_process_grain_preview_output(self) -> Path:
        root = (
            self.project_path.parent / "results"
            if self.project_path is not None
            else Path.home() / "Documents" / "GrindCAE" / "results"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = root / f"{timestamp}_process_grain_preview"
        candidate = base
        suffix = 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}_{suffix:02d}")
            suffix += 1
        return candidate

    def _refresh_process_grain_preview(self) -> None:
        if not hasattr(self, "process_grain_preview_button"):
            return
        try:
            built = ProcessGrainPreviewBuilder().build(self.workbench.project)
        except ProcessGrainPreviewError as exc:
            self._active_process_grain_preview = None
            state = process_grain_preview_state(self.workbench.project, None)
            message = (
                state.message if state.status == "stale"
                else f"暂不能生成预览：{exc}"
            )
            self.process_grain_preview_status_var.set(message)
            self.process_grain_preview.clear(message)
            self.process_grain_preview_button.configure(state="disabled")
            return
        self._active_process_grain_preview = built
        state = process_grain_preview_state(self.workbench.project, built)
        self.process_grain_preview_status_var.set(state.message)
        self._process_grain_preview_output = state.output_directory
        if state.image_path is not None:
            try:
                self.process_grain_preview.set_image(
                    state.image_path, source="process-grain-preview"
                )
            except (OSError, ValueError):
                self.process_grain_preview.clear("预览记录不可用，请重新生成。")
        else:
            self.process_grain_preview.clear(state.message)
        self.process_grain_preview_button.configure(
            state="normal" if self.background_task_gate.owner is None else "disabled"
        )

    def _start_process_grain_preview(self) -> None:
        try:
            built = ProcessGrainPreviewBuilder().build(self.workbench.project)
            output = self._default_process_grain_preview_output()
            self._active_process_grain_preview = built
            self._process_grain_preview_output = output
            self.process_grain_preview_runner.start(built, output)
        except (ProcessGrainPreviewError, OSError) as exc:
            self.process_grain_preview_status_var.set(f"无法生成预览：{exc}")
            messagebox.showerror("无法生成磨粒预览", str(exc), parent=self.root)
            return
        self.process_grain_preview.clear("正在生成统计等效磨粒与载荷预览。")
        self.process_grain_preview_button.configure(state="disabled")
        self.analysis_run_button.configure(state="disabled")

    def _poll_process_grain_preview_events(self) -> None:
        try:
            while True:
                self._handle_process_grain_preview_event(
                    self.process_grain_preview_runner.events.get_nowait()
                )
        except Empty:
            pass
        self.root.after(100, self._poll_process_grain_preview_events)

    def _handle_process_grain_preview_event(
        self, event: ProcessGrainPreviewEvent
    ) -> None:
        self.process_grain_preview_status_var.set(event.message)
        if (
            event.kind == "success"
            and event.result is not None
            and self._active_process_grain_preview is not None
        ):
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self.workbench.project.register_process_grain_preview(ResultRecord(
                result_id=f"process-grain-preview-{created_at}",
                analysis_type="process_grain_preview",
                result_format=str(event.result.summary.get("result_format", "statistical_equivalent_grain_mechanism_load_result")),
                output_directory=str(event.result.output_directory),
                summary_path=str(event.result.summary_path),
                artifact_paths={
                    key: str(path) for key, path in event.result.artifact_paths.items()
                    if key != "summary_json"
                },
                created_at=created_at,
                input_fingerprint=self._active_process_grain_preview.input_fingerprint,
            ))
            self._process_grain_preview_output = event.result.output_directory
            self.process_grain_preview.set_image(
                event.result.image_path, source="process-grain-preview"
            )
        elif event.kind == "error":
            self.process_grain_preview.clear(
                "预览生成失败；没有登记有效预览记录。"
            )
            messagebox.showerror("磨粒预览失败", event.message, parent=self.root)
        elif event.kind == "finished":
            self._active_process_grain_preview = None
            self._refresh_process_grain_preview()
            self._refresh_analysis_workspace()

    def _open_process_grain_preview_output(self) -> None:
        if self._process_grain_preview_output is not None:
            self._open_path(self._process_grain_preview_output)

    def _restore_process_widgets(self) -> None:
        geometry = self.workbench.project.geometry
        if isinstance(geometry, dict) and geometry.get("source") == "parametric_2d":
            try:
                geometry_case = ParametricGeometryCase.from_mapping(geometry)
                self.process_geometry_variables["wheel_diameter"].set(f"{geometry_case.wheel_diameter_m * 1.0e3:g} mm（来自几何，只读）")
                self.process_geometry_variables["depth_of_cut"].set(f"{geometry_case.depth_of_cut_m * 1.0e6:g} μm（来自几何，只读）")
            except GeometryValidationError:
                pass
        else:
            self.process_geometry_variables["wheel_diameter"].set("尚未配置几何")
            self.process_geometry_variables["depth_of_cut"].set("尚未配置几何")
        wheel_payload = self.workbench.project.wheel
        if wheel_payload is None:
            wheel = WheelDisplayInput().build_case()
            self._resolved_workbench_wheel = None
        else:
            try:
                wheel = WorkbenchWheelCase.from_mapping(wheel_payload)
                self._resolved_workbench_wheel = wheel
            except ProcessValidationError as exc:
                self._set_process_report(str(exc))
                return
        reverse_paths = {value: key for key, value in _GRIT_PATH_LABELS.items()}
        resolved = wheel.resolved_specification
        wheel_values = {
            "abrasive_material": "刚玉",
            "resolution_path": reverse_paths[wheel.resolution_path],
            "grit_designation": wheel.grit_designation,
            "manufacturer": str(resolved.get("manufacturer") or ""),
            "product_model": str(resolved.get("product_model") or ""),
            "diameter_lower_um": f"{float(resolved['diameter_lower_m']) * 1.0e6:g}",
            "representative_diameter_um": f"{float(resolved['representative_diameter_d50_m']) * 1.0e6:g}",
            "diameter_upper_um": f"{float(resolved['diameter_upper_m']) * 1.0e6:g}",
            "manufacturer_source": str(resolved.get("source") or ""),
            "manufacturer_data_status": str(resolved.get("data_status") or "manufacturer_input_unverified"),
        }
        for name, value in wheel_values.items():
            self.wheel_variables[name].set(value)
        self._update_grit_path()
        if wheel_payload is not None:
            self._resolved_workbench_wheel = wheel
        process_payload = self.workbench.project.process
        if process_payload is None:
            process = ProcessDisplayInput().build_case(geometry, wheel) if geometry is not None else None
        else:
            try:
                process = WorkbenchProcessCase.from_mapping(process_payload)
            except ProcessValidationError as exc:
                self._set_process_report(str(exc))
                return
        if process is not None:
            reverse_calibration = {value: key for key, value in _CALIBRATION_LABELS.items()}
            values = {
                "wheel_surface_speed_m_per_s": f"{process.wheel_surface_speed_m_per_s:g}",
                "workpiece_feed_speed_m_per_s": f"{process.workpiece_feed_speed_m_per_s:g}",
                "grinding_width_mm": f"{process.grinding_width_m * 1.0e3:g}",
                "specific_grinding_energy_j_per_mm3": f"{process.specific_grinding_energy_j_per_m3 / 1.0e9:g}",
                "normal_to_tangential_force_ratio": f"{process.normal_to_tangential_force_ratio:g}",
                "calibration_id": process.calibration_id,
                "calibration_status": reverse_calibration[process.calibration_status],
                "source_description": process.source_description,
                "applicability_notes": process.applicability_notes,
            }
            for name, value in values.items():
                self.process_variables[name].set(value)
        if wheel_payload is not None and process_payload is not None and process is not None:
            self._set_process_report("\n".join(["工艺状态：已从工程恢复", *(f"{key}：{value}" for key, value in wheel.wheel_summary().items()), "", *(f"{key}：{value}" for key, value in process.process_summary().items())]))
        else:
            self._set_process_report("砂轮与工艺尚未配置。\n请先确认几何和材料，再解析粒度并保存工艺参数。")
        self._refresh_process_grain_preview()

    def _geometry_display_input(self) -> GeometryDisplayInput:
        direction = {
            "正向": "positive_x",
            "反向": "negative_x",
        }.get(self.geometry_variables["relative_feed_direction"].get(), "")
        return GeometryDisplayInput(
            workpiece_length_mm=self.geometry_variables["workpiece_length_mm"].get(),
            workpiece_height_mm=self.geometry_variables["workpiece_height_mm"].get(),
            wheel_diameter_mm=self.geometry_variables["wheel_diameter_mm"].get(),
            depth_of_cut_um=self.geometry_variables["depth_of_cut_um"].get(),
            relative_feed_direction=direction,
            wheel_lowest_point_x_mm=self.geometry_variables[
                "wheel_lowest_point_x_mm"
            ].get(),
        )

    def _geometry_output_directory(self) -> Path:
        if self.project_path is not None:
            name = self.project_path.name
            stem = name[:-11] if name.lower().endswith(".gcase.json") else self.project_path.stem
            return self.project_path.parent / f"{stem}_geometry"
        return application_directory() / "outputs" / "gui_geometry"

    def _restore_geometry_widgets(self) -> None:
        geometry = self.workbench.project.geometry
        if geometry is None:
            defaults = GeometryDisplayInput()
            self.geometry_variables["workpiece_length_mm"].set(
                defaults.workpiece_length_mm
            )
            self.geometry_variables["workpiece_height_mm"].set(
                defaults.workpiece_height_mm
            )
            self.geometry_variables["wheel_diameter_mm"].set(
                defaults.wheel_diameter_mm
            )
            self.geometry_variables["depth_of_cut_um"].set(defaults.depth_of_cut_um)
            self.geometry_variables["relative_feed_direction"].set("正向")
            self.geometry_variables["wheel_lowest_point_x_mm"].set(
                defaults.wheel_lowest_point_x_mm
            )
            self.geometry_preview_photo = None
            self.geometry_image_preview.clear(
                "点击“生成二维模型”后显示 geometry_preview.png。"
            )
            self.geometry_report_text.configure(state="normal")
            self.geometry_report_text.delete("1.0", "end")
            self.geometry_report_text.insert("1.0", "几何尚未配置。")
            self.geometry_report_text.configure(state="disabled")
            return
        try:
            display = ParametricGeometryCase.from_mapping(geometry).to_display_input()
        except GeometryValidationError as exc:
            self.geometry_report_text.configure(state="normal")
            self.geometry_report_text.delete("1.0", "end")
            self.geometry_report_text.insert("1.0", str(exc))
            self.geometry_report_text.configure(state="disabled")
            return
        values = {
            "workpiece_length_mm": display.workpiece_length_mm,
            "workpiece_height_mm": display.workpiece_height_mm,
            "wheel_diameter_mm": display.wheel_diameter_mm,
            "depth_of_cut_um": display.depth_of_cut_um,
            "relative_feed_direction": (
                "正向"
                if display.relative_feed_direction == "positive_x"
                else "反向"
            ),
            "wheel_lowest_point_x_mm": display.wheel_lowest_point_x_mm,
        }
        for name, value in values.items():
            self.geometry_variables[name].set(value)
        latest = self.workbench.project.latest_geometry_result
        if latest is not None:
            preview = latest.artifact_paths.get("geometry_preview")
            if preview:
                self._load_geometry_preview(Path(preview))
            try:
                summary = json.loads(Path(latest.summary_path).read_text(encoding="utf-8"))
                self._show_geometry_report(summary)
            except (OSError, json.JSONDecodeError):
                pass

    def _show_geometry_report(self, summary: dict[str, object]) -> None:
        chinese = summary.get("geometry_summary_chinese", {})
        lines = ["几何状态：有效"]
        if isinstance(chinese, dict):
            lines.extend(f"{key}：{value}" for key, value in chinese.items())
        limitations = summary.get("limitations", [])
        if isinstance(limitations, list):
            lines.extend(("", *(str(item) for item in limitations)))
        self.geometry_report_text.configure(state="normal")
        self.geometry_report_text.delete("1.0", "end")
        self.geometry_report_text.insert("1.0", "\n".join(lines))
        self.geometry_report_text.configure(state="disabled")

    def _load_geometry_preview(self, path: Path) -> None:
        try:
            self.geometry_image_preview.set_image(path, source="geometry")
        except (OSError, ValueError) as exc:
            self.geometry_preview_photo = None
            self.geometry_image_preview.clear(f"几何预览加载失败：\n{path}\n{exc}")
            return
        self.geometry_preview_photo = self.geometry_image_preview._photo

    def _generate_geometry(self) -> None:
        try:
            provider = Parametric2DGeometryProvider(
                self._geometry_display_input().build_case()
            )
            published = provider.publish(self._geometry_output_directory())
            self.workbench.register_geometry_result(provider, published)
        except GeometryValidationError as exc:
            self.workbench.mark_geometry_failed(str(exc))
            self._refresh_work_tree("几何")
            messagebox.showerror("二维几何生成失败", str(exc), parent=self.root)
            return
        self._load_geometry_preview(published.preview_path)
        self._show_geometry_report(published.summary)
        if self.workbench.project.node_status("网格") is ProjectNodeStatus.STALE:
            self.mesh_image_preview.clear("几何已更新，请重新生成网格。")
            self._set_mesh_report(
                "网格状态：需要更新\n几何已经变化，旧网格不再作为当前有效网格。"
            )
        self._refresh_work_tree("几何")

    def _mesh_display_input(self) -> MeshDisplayInput:
        mesh_type = {
            "三角形": "triangle",
            "四边形（暂未启用）": "quadrilateral",
        }.get(self.mesh_variables["mesh_type"].get(), "")
        return MeshDisplayInput(
            mesh_type=mesh_type,
            target_size_mm=self.mesh_variables["target_size_mm"].get(),
            minimum_size_mm=self.mesh_variables["minimum_size_mm"].get(),
            maximum_size_mm=self.mesh_variables["maximum_size_mm"].get(),
        )

    def _mesh_output_directory(self) -> Path:
        if self.project_path is not None:
            name = self.project_path.name
            stem = name[:-11] if name.lower().endswith(".gcase.json") else self.project_path.stem
            return self.project_path.parent / f"{stem}_mesh"
        return application_directory() / "outputs" / "gui_mesh"

    def _restore_mesh_widgets(self) -> None:
        mesh = self.workbench.project.mesh
        if mesh is None:
            defaults = MeshDisplayInput()
            self.mesh_variables["mesh_type"].set("三角形")
            self.mesh_variables["target_size_mm"].set(defaults.target_size_mm)
            self.mesh_variables["minimum_size_mm"].set(defaults.minimum_size_mm)
            self.mesh_variables["maximum_size_mm"].set(defaults.maximum_size_mm)
            self.mesh_image_preview.clear("网格尚未生成。")
            self._set_mesh_report("网格尚未生成。")
            return
        try:
            case = WorkbenchMeshCase.from_mapping(mesh, allow_deferred=True)
        except MeshValidationError as exc:
            self.mesh_image_preview.clear(str(exc))
            self._set_mesh_report(str(exc))
            return
        display = case.to_display_input()
        self.mesh_variables["mesh_type"].set(
            "三角形" if display.mesh_type == "triangle" else "四边形（暂未启用）"
        )
        self.mesh_variables["target_size_mm"].set(display.target_size_mm)
        self.mesh_variables["minimum_size_mm"].set(display.minimum_size_mm)
        self.mesh_variables["maximum_size_mm"].set(display.maximum_size_mm)
        latest = self.workbench.project.latest_mesh_result
        if case.status == "completed" and latest is not None:
            preview = latest.artifact_paths.get("mesh_png")
            if preview:
                try:
                    self.mesh_image_preview.set_image(preview, source="mesh")
                except (OSError, ValueError):
                    pass
            try:
                summary = json.loads(Path(latest.summary_path).read_text(encoding="utf-8"))
                self._show_mesh_report(summary)
            except (OSError, json.JSONDecodeError):
                self._set_mesh_report("已登记网格，但摘要文件无法读取。")
        elif case.status == "stale":
            self.mesh_image_preview.clear("几何已更新，请重新生成网格。")
            self._set_mesh_report("网格状态：需要更新\n几何已经变化，旧网格不再作为当前有效网格。")

    def _set_mesh_report(self, text: str) -> None:
        self.mesh_report_text.configure(state="normal")
        self.mesh_report_text.delete("1.0", "end")
        self.mesh_report_text.insert("1.0", text)
        self.mesh_report_text.configure(state="disabled")

    def _show_mesh_report(self, summary: dict[str, object]) -> None:
        settings = summary.get("mesh_settings", {})
        statistics = summary.get("mesh_statistics", {})
        preview = summary.get("preview", {})
        solve_risk = summary.get("solve_size_risk", {})
        if not isinstance(settings, dict) or not isinstance(statistics, dict):
            self._set_mesh_report("网格摘要格式无效。")
            return
        preview_strategy = {
            "complete_mesh": "完整网格线与节点",
            "complete_edges_without_nodes": "完整网格线（隐藏节点）",
            "global_outline_with_local_true_mesh": "全局轮廓＋局部真实网格",
        }.get(str(preview.get("strategy", "")), "历史版本预览")
        if "estimated_vector_p1_degrees_of_freedom" in solve_risk:
            estimated_dofs = int(
                solve_risk["estimated_vector_p1_degrees_of_freedom"]
            )
            maximum_dofs = int(
                solve_risk.get("maximum_vector_p1_degrees_of_freedom", 0)
            )
            risk_text = (
                "默认安全阈值内"
                if solve_risk.get("within_default_limit") is True
                else "超过默认安全阈值，正式分析前将继续执行安全检查"
            )
            dof_line = (
                f"预计二维向量自由度：{estimated_dofs} / {maximum_dofs}（{risk_text}）"
            )
        else:
            dof_line = "预计二维向量自由度：历史摘要未记录。"
        lines = (
            "网格状态：有效",
            "网格类型：三角形",
            f"节点数量：{statistics.get('node_count', '')}",
            f"三角形单元数量：{statistics.get('triangle_count', '')}",
            f"边界单元数量：{statistics.get('boundary_element_count', '')}",
            f"最小单元面积：{float(statistics.get('minimum_element_area_m2', 0.0)) * 1.0e6:.6g} mm²",
            f"最大单元面积：{float(statistics.get('maximum_element_area_m2', 0.0)) * 1.0e6:.6g} mm²",
            f"目标尺寸：{float(settings.get('target_size_m', 0.0)) * 1.0e3:.6g} mm",
            f"实际平均尺寸：{float(statistics.get('actual_average_size_m', 0.0)) * 1.0e3:.6g} mm",
            f"预览策略：{preview_strategy}",
            "完整 mesh.msh：已保留，预览不会删减正式网格。",
            dof_line,
            "",
            "当前阶段只生成二维三角形网格。",
            "当前结果不包含载荷、材料、位移、应力、应变或温度场。",
        )
        self._set_mesh_report("\n".join(lines))

    def _generate_mesh(self) -> None:
        geometry = self.workbench.project.geometry
        if geometry is None or self.workbench.project.node_status("几何") is not ProjectNodeStatus.COMPLETED:
            messagebox.showerror("网格生成失败", "请先完成二维几何建模。", parent=self.root)
            return
        try:
            geometry_case = ParametricGeometryCase.from_mapping(geometry)
            geometry_model = Parametric2DGeometryProvider(geometry_case).build_geometry()
            provider = TriangleMeshProvider(
                geometry_model, self._mesh_display_input().build_case()
            )
            self._active_mesh_provider = provider
            self.mesh_controller.start(provider, self._mesh_output_directory())
        except (GeometryValidationError, MeshValidationError, MeshControllerError) as exc:
            messagebox.showerror("网格生成失败", str(exc), parent=self.root)
            return
        self.workbench.project.set_node_status(
            "网格", ProjectNodeStatus.RUNNING, "三角形网格正在后台生成。"
        )
        self.mesh_generate_button.configure(state="disabled")
        self._refresh_work_tree("网格")

    def _poll_mesh_events(self) -> None:
        try:
            while True:
                event = self.mesh_controller.events.get_nowait()
                if event.kind == "success" and isinstance(event.result, MeshPublishedResult):
                    provider = self._active_mesh_provider
                    if provider is None:
                        raise MeshValidationError("网格任务来源已丢失，拒绝登记结果。")
                    self.workbench.register_mesh_result(provider, event.result)
                    self.mesh_image_preview.set_image(event.result.preview_path, source="mesh")
                    self._show_mesh_report(event.result.summary)
                    self._refresh_work_tree("网格")
                elif event.kind == "error":
                    self.workbench.mark_mesh_failed(event.message)
                    self._refresh_work_tree("网格")
                    messagebox.showerror("网格生成失败", event.message, parent=self.root)
                elif event.kind == "finished":
                    self._active_mesh_provider = None
                    self.mesh_generate_button.configure(state="normal")
        except Empty:
            pass
        self.root.after(100, self._poll_mesh_events)

    def _update_calibration_display(self, *_args: object) -> None:
        calibration_id = self.variables["calibration_id"].get()
        self.calibration_display_var.set(
            f"经验参数集：{calibration_set_display_text(calibration_id)}"
        )
        self.calibration_name_var.set(
            f"参数集名称：{calibration_set_name(calibration_id)}"
        )
        if (hasattr(self, "analysis_mode_var") and self.analysis_mode_var.get()
                == ANALYSIS_MODE_DEFINITIONS["literature_grinding_force"].display_name):
            self.calibration_display_var.set(
                "当前分析来源：Zhang 2017 · 440C 文献力模型；参数与标定来源见文献参数设置"
            )
        if (hasattr(self, "analysis_mode_var") and self.analysis_mode_var.get()
                == ANALYSIS_MODE_DEFINITIONS["literature_elastoplastic_single_pass"].display_name):
            self.calibration_display_var.set("文献力 + 工程J2材料；砂轮/经验关系见文献参数页；3.8.1候选功能；独立实验验证未完成")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=(GUI_FONT_FAMILY, 16, "bold"))
        style.configure("Heading.TLabel", font=(GUI_FONT_FAMILY, 10, "bold"))
        # Dynamic worker-event text must use the same CJK-capable font as its labels.
        style.configure("Status.TLabel", font=(GUI_FONT_FAMILY, 10), foreground="#145A32")
        style.configure("Stage.TLabel", font=(GUI_FONT_FAMILY, 10))
        style.configure("Warning.TLabel", font=(GUI_FONT_FAMILY, 10), foreground="#9A5B00")

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(14, 10, 14, 6))
        header.pack(fill="x")
        ttk.Label(header, text="GrindCAE 工程工作台", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            textvariable=self.calibration_display_var,
            style="Warning.TLabel",
        ).pack(side="right", padx=8)

        project_toolbar = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        project_toolbar.pack(fill="x")
        ttk.Button(project_toolbar, text="新建工程", command=self._new_project).pack(side="left")
        ttk.Button(project_toolbar, text="打开工程", command=self._open_project).pack(side="left", padx=5)
        recent_project_button = ttk.Menubutton(project_toolbar, text="最近工程")
        self.recent_projects_menu = tk.Menu(recent_project_button, tearoff=False)
        recent_project_button.configure(menu=self.recent_projects_menu)
        recent_project_button.pack(side="left")
        recent_project_button.bind("<Button-1>", lambda _event: self._populate_recent_projects_menu())
        recent_result_button = ttk.Menubutton(project_toolbar, text="最近结果")
        self.recent_results_menu = tk.Menu(recent_result_button, tearoff=False)
        recent_result_button.configure(menu=self.recent_results_menu)
        recent_result_button.pack(side="left", padx=5)
        recent_result_button.bind("<Button-1>", lambda _event: self._populate_recent_results_menu())
        ttk.Button(project_toolbar, text="保存工程", command=self._save_project).pack(side="left")
        archive_button = ttk.Menubutton(project_toolbar, text="完整归档")
        archive_menu = tk.Menu(archive_button, tearoff=False)
        archive_menu.add_command(label="导出完整归档…", command=self._export_complete_archive)
        archive_menu.add_command(label="导入完整归档…", command=self._import_complete_archive)
        archive_button.configure(menu=archive_menu)
        archive_button.pack(side="left", padx=5)
        ttk.Label(project_toolbar, textvariable=self.project_path_var).pack(
            side="left", fill="x", expand=True, padx=12
        )

        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        self.workbench_tab = ttk.Frame(self.main_notebook)
        self.legacy_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.workbench_tab, text="工程工作台")
        self.main_notebook.add(self.legacy_tab, text="高级兼容计算")
        self._build_workbench_layout(self.workbench_tab)
        self._build_legacy_layout(self.legacy_tab)

    def _build_workbench_layout(self, parent: ttk.Frame) -> None:
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        tree_host = ttk.LabelFrame(paned, text="中文工作树", padding=8)
        workspace_host = ttk.LabelFrame(paned, text="当前节点", padding=0)
        paned.add(tree_host, weight=2)
        paned.add(workspace_host, weight=7)

        workbench_scrollbar = ttk.Scrollbar(workspace_host, orient="vertical")
        workbench_scrollbar.pack(side="left", fill="y")
        workbench_scroll_canvas = tk.Canvas(
            workspace_host,
            highlightthickness=0,
            borderwidth=0,
        )
        workbench_scroll_canvas.pack(side="right", fill="both", expand=True)
        workbench_scrollbar.configure(command=workbench_scroll_canvas.yview)
        workbench_scroll_canvas.configure(yscrollcommand=workbench_scrollbar.set)
        workspace = ttk.Frame(workbench_scroll_canvas, padding=12)
        workbench_window_id = workbench_scroll_canvas.create_window(
            (0, 0), window=workspace, anchor="nw"
        )
        workspace.bind(
            "<Configure>",
            lambda _event: workbench_scroll_canvas.configure(
                scrollregion=workbench_scroll_canvas.bbox("all")
            ),
        )
        workbench_scroll_canvas.bind(
            "<Configure>",
            lambda event: workbench_scroll_canvas.itemconfigure(
                workbench_window_id, width=event.width
            ),
        )
        self.workbench_scrollbar = workbench_scrollbar
        self.workbench_scroll_canvas = workbench_scroll_canvas
        self.workbench_scroll_content = workspace
        from .scrolling import bind_canvas_wheel
        bind_canvas_wheel(workbench_scroll_canvas)

        self.work_tree = ttk.Treeview(
            tree_host,
            columns=("status",),
            show="tree headings",
            selectmode="browse",
            height=10,
        )
        self.work_tree.heading("#0", text="工作节点")
        self.work_tree.heading("status", text="状态")
        self.work_tree.column("#0", width=120, stretch=True)
        self.work_tree.column("status", width=90, anchor="center", stretch=False)
        self.work_tree.pack(fill="both", expand=True)
        self.work_tree.bind("<<TreeviewSelect>>", self._on_work_tree_selected)

        heading = ttk.Frame(workspace)
        heading.pack(fill="x")
        ttk.Label(heading, textvariable=self.workbench_title_var, style="Title.TLabel").pack(side="left")
        ttk.Label(heading, text="状态：", style="Heading.TLabel").pack(side="left", padx=(18, 2))
        ttk.Label(heading, textvariable=self.workbench_status_var, style="Status.TLabel").pack(side="left")
        ttk.Label(
            workspace,
            textvariable=self.workbench_description_var,
            justify="left",
            wraplength=820,
        ).pack(fill="x", pady=(12, 10))

        self.workbench_action_host = ttk.Frame(workspace)
        self.workbench_action_host.pack(fill="x", pady=(0, 10))

        project_editor = ttk.LabelFrame(workspace, text="工程信息", padding=8)
        project_editor.pack(fill="x", pady=(0, 8))
        project_editor.columnconfigure(1, weight=1)
        ttk.Label(project_editor, text="工程名称").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.project_name_entry = ttk.Entry(project_editor, textvariable=self.project_name_var)
        self.project_name_entry.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(project_editor, text="工程备注").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=3)
        self.project_description_text = tk.Text(
            project_editor,
            height=4,
            wrap="word",
            font=(GUI_FONT_FAMILY, 10),
        )
        self.project_description_text.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(
            project_editor,
            text="应用工程信息",
            command=self._apply_project_metadata,
        ).grid(row=2, column=1, sticky="e", pady=(4, 0))
        self.project_editor = project_editor

        geometry_editor = ttk.LabelFrame(workspace, text="二维参数化几何", padding=8)
        geometry_editor.columnconfigure(1, weight=1)
        geometry_editor.columnconfigure(2, weight=3)
        labels = (
            ("工件长度 [mm]", "workpiece_length_mm"),
            ("工件初始高度 [mm]", "workpiece_height_mm"),
            ("砂轮直径 [mm]", "wheel_diameter_mm"),
            ("磨削深度 [μm]", "depth_of_cut_um"),
            ("砂轮最低点位置 [mm]", "wheel_lowest_point_x_mm"),
        )
        for row, (label, name) in enumerate(labels):
            ttk.Label(geometry_editor, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=3
            )
            ttk.Entry(
                geometry_editor, textvariable=self.geometry_variables[name], width=18
            ).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Label(geometry_editor, text="磨削方向").grid(
            row=5, column=0, sticky="w", padx=(0, 8), pady=3
        )
        ttk.Combobox(
            geometry_editor,
            textvariable=self.geometry_variables["relative_feed_direction"],
            values=("正向", "反向"),
            state="readonly",
            width=16,
        ).grid(row=5, column=1, sticky="ew", pady=3)
        ttk.Button(
            geometry_editor,
            text="生成二维模型",
            command=self._generate_geometry,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 3))
        ttk.Label(
            geometry_editor,
            text="输入采用 mm/μm，工程文件内部统一保存为 SI 单位。",
            style="Warning.TLabel",
            wraplength=280,
            justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

        preview_host = ttk.LabelFrame(geometry_editor, text="二维几何预览", padding=6)
        preview_host.grid(row=0, column=2, rowspan=6, sticky="nsew", padx=(12, 0))
        self.geometry_image_preview = ImagePreview(
            preview_host,
            source="geometry",
            error_handler=lambda message: messagebox.showerror(
                "几何图片错误", message, parent=self.root
            ),
        )
        self.geometry_image_preview.pack(fill="both", expand=True)
        self.geometry_image_preview.clear(
            "点击“生成二维模型”后显示 geometry_preview.png。"
        )
        self.geometry_preview_label = self.geometry_image_preview.canvas
        report_host = ttk.LabelFrame(geometry_editor, text="几何检查与说明", padding=4)
        report_host.grid(row=6, column=2, rowspan=2, sticky="nsew", padx=(12, 0), pady=(6, 0))
        self.geometry_report_text = tk.Text(
            report_host,
            height=14,
            wrap="word",
            state="disabled",
            font=(GUI_FONT_FAMILY, 9),
        )
        self.geometry_report_text.pack(fill="both", expand=True)
        self.geometry_editor = geometry_editor

        mesh_editor = ttk.LabelFrame(workspace, text="三角形网格", padding=8)
        mesh_editor.columnconfigure(1, weight=1)
        mesh_editor.columnconfigure(2, weight=3)
        ttk.Label(mesh_editor, text="网格类型").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=3
        )
        ttk.Combobox(
            mesh_editor,
            textvariable=self.mesh_variables["mesh_type"],
            values=("三角形", "四边形（暂未启用）"),
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="ew", pady=3)
        for row, (label, name) in enumerate(
            (
                ("目标尺寸 [mm]", "target_size_mm"),
                ("最小尺寸 [mm]", "minimum_size_mm"),
                ("最大尺寸 [mm]", "maximum_size_mm"),
            ),
            start=1,
        ):
            ttk.Label(mesh_editor, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=3
            )
            ttk.Entry(
                mesh_editor, textvariable=self.mesh_variables[name], width=18
            ).grid(row=row, column=1, sticky="ew", pady=3)
        self.mesh_generate_button = ttk.Button(
            mesh_editor, text="生成网格", command=self._generate_mesh
        )
        self.mesh_generate_button.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8, 3)
        )
        ttk.Label(
            mesh_editor,
            text="第一版只生成一阶三角形网格；输入采用 mm，工程文件保存为 SI。",
            style="Warning.TLabel",
            wraplength=280,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        mesh_preview_host = ttk.LabelFrame(mesh_editor, text="网格预览", padding=6)
        mesh_preview_host.grid(
            row=0, column=2, rowspan=5, sticky="nsew", padx=(12, 0)
        )
        self.mesh_image_preview = ImagePreview(
            mesh_preview_host,
            source="mesh",
            error_handler=lambda message: messagebox.showerror(
                "网格图片错误", message, parent=self.root
            ),
        )
        self.mesh_image_preview.pack(fill="both", expand=True)
        self.mesh_image_preview.clear("网格尚未生成。")
        mesh_report_host = ttk.LabelFrame(mesh_editor, text="网格报告", padding=4)
        mesh_report_host.grid(
            row=5, column=2, rowspan=2, sticky="nsew", padx=(12, 0), pady=(6, 0)
        )
        self.mesh_report_text = tk.Text(
            mesh_report_host,
            height=11,
            wrap="word",
            state="disabled",
            font=(GUI_FONT_FAMILY, 9),
        )
        self.mesh_report_text.pack(fill="both", expand=True)
        self.mesh_editor = mesh_editor

        material_editor = ttk.LabelFrame(workspace, text="材料参数", padding=8)
        material_editor.columnconfigure(1, weight=1)
        for row, (label, name) in enumerate(
            (
                ("材料名称", "label"),
                ("弹性模量 [GPa]", "elastic_modulus_gpa"),
                ("泊松比 [-]", "poisson_ratio"),
                ("屈服强度 [MPa]", "yield_strength_mpa"),
                ("切线模量 [GPa]", "tangent_modulus_gpa"),
                ("来源说明", "source_description"),
                ("适用范围", "applicability_notes"),
            )
        ):
            ttk.Label(material_editor, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(material_editor, textvariable=self.material_variables[name])
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            if name in ("yield_strength_mpa", "tangent_modulus_gpa"):
                if not hasattr(self, "material_plastic_entries"):
                    self.material_plastic_entries = {}
                self.material_plastic_entries[name] = entry
        for row, (label, name, values) in enumerate(
            (
                ("材料行为", "behavior", tuple(_MATERIAL_BEHAVIOR_LABELS)),
                ("材料类别", "family", ("延性金属", "硬脆材料（暂未启用）", "碳化硅（暂未启用）")),
                ("参数来源", "parameter_source", tuple(_MATERIAL_SOURCE_LABELS)),
                ("标定状态", "calibration_status", tuple(_CALIBRATION_LABELS)),
            ),
            start=7,
        ):
            ttk.Label(material_editor, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            combo = ttk.Combobox(material_editor, textvariable=self.material_variables[name], values=values, state="readonly")
            combo.grid(row=row, column=1, sticky="ew", pady=3)
            if name == "behavior":
                combo.bind("<<ComboboxSelected>>", self._update_material_behavior)
        ttk.Button(material_editor, text="保存材料参数", command=self._save_material_parameters).grid(row=11, column=0, columnspan=2, sticky="ew", pady=(8, 3))
        ttk.Label(material_editor, text="这些演示参数未针对具体钢号进行实验标定。输入采用 GPa/MPa，工程文件保存为 SI。", style="Warning.TLabel", wraplength=360, justify="left").grid(row=12, column=0, columnspan=2, sticky="w", pady=(4, 0))
        material_report_host = ttk.LabelFrame(material_editor, text="材料检查与说明", padding=4)
        material_report_host.grid(row=0, column=2, rowspan=13, sticky="nsew", padx=(12, 0))
        material_editor.columnconfigure(2, weight=2)
        self.material_report_text = tk.Text(material_report_host, height=22, wrap="word", state="disabled", font=(GUI_FONT_FAMILY, 9))
        self.material_report_text.pack(fill="both", expand=True)
        self.material_editor = material_editor

        process_editor = ttk.LabelFrame(workspace, text="砂轮与磨削工艺", padding=8)
        process_editor.columnconfigure(1, weight=1)
        process_editor.columnconfigure(3, weight=1)
        ttk.Label(process_editor, text="砂轮直径").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(process_editor, textvariable=self.process_geometry_variables["wheel_diameter"], state="readonly").grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(process_editor, text="磨削深度").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(process_editor, textvariable=self.process_geometry_variables["depth_of_cut"], state="readonly").grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(process_editor, text="磨料类型").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(process_editor, textvariable=self.wheel_variables["abrasive_material"], values=("刚玉", "碳化硅（暂未启用）", "CBN（暂未启用）", "金刚石（暂未启用）"), state="readonly").grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(process_editor, text="粒度路径").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        path_combo = ttk.Combobox(process_editor, textvariable=self.wheel_variables["resolution_path"], values=tuple(_GRIT_PATH_LABELS), state="readonly")
        path_combo.grid(row=3, column=1, sticky="ew", pady=3)
        path_combo.bind("<<ComboboxSelected>>", self._update_grit_path)
        ttk.Label(process_editor, text="粒度号").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=3)
        self.grit_designation_combo = ttk.Combobox(process_editor, textvariable=self.wheel_variables["grit_designation"], state="normal")
        self.grit_designation_combo.grid(row=4, column=1, sticky="ew", pady=3)
        ttk.Button(process_editor, text="解析粒度", command=self._resolve_workbench_wheel).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 3))

        manufacturer_frame = ttk.LabelFrame(process_editor, text="厂家数据覆盖", padding=6)
        manufacturer_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 3))
        manufacturer_frame.columnconfigure(1, weight=1)
        for row, (label, name) in enumerate((
            ("厂家名称", "manufacturer"), ("产品型号", "product_model"),
            ("粒径下限 [μm]", "diameter_lower_um"), ("代表粒径 d50 [μm]", "representative_diameter_um"),
            ("粒径上限 [μm]", "diameter_upper_um"), ("数据来源", "manufacturer_source"), ("数据状态", "manufacturer_data_status"),
        )):
            ttk.Label(manufacturer_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(manufacturer_frame, textvariable=self.wheel_variables[name]).grid(row=row, column=1, sticky="ew", pady=2)
        self.manufacturer_frame = manufacturer_frame

        for row, (label, name) in enumerate((
            ("砂轮线速度 [m/s]", "wheel_surface_speed_m_per_s"),
            ("工件进给速度 [m/s]", "workpiece_feed_speed_m_per_s"),
            ("磨削宽度 [mm]", "grinding_width_mm"),
            ("比磨削能 [J/mm³]", "specific_grinding_energy_j_per_mm3"),
            ("法向/切向力比 [-]", "normal_to_tangential_force_ratio"),
            ("经验参数编号", "calibration_id"),
            ("参数来源说明", "source_description"),
            ("适用范围说明", "applicability_notes"),
        )):
            ttk.Label(process_editor, text=label).grid(row=row, column=2, sticky="w", padx=(12, 8), pady=3)
            ttk.Entry(process_editor, textvariable=self.process_variables[name]).grid(row=row, column=3, sticky="ew", pady=3)
        ttk.Label(process_editor, text="标定状态").grid(row=8, column=2, sticky="w", padx=(12, 8), pady=3)
        ttk.Combobox(process_editor, textvariable=self.process_variables["calibration_status"], values=tuple(_CALIBRATION_LABELS), state="readonly").grid(row=8, column=3, sticky="ew", pady=3)
        ttk.Button(process_editor, text="保存工艺参数", command=self._save_process_parameters).grid(row=9, column=2, columnspan=2, sticky="ew", padx=(12, 0), pady=(8, 3))
        ttk.Label(process_editor, text="砂轮直径和磨削深度来自几何，只读。当前比磨削能与力比采用 Jiang 2023 指定工况的公开实验参考值；换材料、粒度或工况时仍需重新标定。", style="Warning.TLabel", wraplength=380, justify="left").grid(row=10, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(4, 0))
        process_report_host = ttk.LabelFrame(process_editor, text="工艺检查与说明", padding=4)
        process_report_host.grid(row=11, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
        self.process_report_text = tk.Text(process_report_host, height=10, wrap="word", state="disabled", font=(GUI_FONT_FAMILY, 9))
        self.process_report_text.pack(fill="both", expand=True)
        grain_preview_box = ttk.LabelFrame(
            process_editor, text="统计等效磨粒与载荷预览", padding=6
        )
        grain_preview_box.grid(
            row=12, column=0, columnspan=4, sticky="nsew", pady=(8, 0)
        )
        process_editor.rowconfigure(12, weight=1)
        preview_actions = ttk.Frame(grain_preview_box)
        preview_actions.pack(fill="x", pady=(0, 4))
        self.process_grain_preview_button = ttk.Button(
            preview_actions,
            text="生成磨粒预览",
            command=self._start_process_grain_preview,
            state="disabled",
        )
        self.process_grain_preview_button.pack(side="left")
        ttk.Button(
            preview_actions,
            text="打开预览目录",
            command=self._open_process_grain_preview_output,
        ).pack(side="left", padx=(4, 0))
        ttk.Label(
            preview_actions,
            textvariable=self.process_grain_preview_status_var,
            justify="left",
            wraplength=600,
        ).pack(side="left", padx=(10, 0), fill="x", expand=True)
        self.process_grain_preview = ImagePreview(
            grain_preview_box,
            source="process-grain-preview",
            error_handler=lambda message: messagebox.showerror(
                "磨粒预览错误", message, parent=self.root
            ),
            open_handler=self._open_path,
        )
        self.process_grain_preview.pack(fill="both", expand=True)
        self.process_grain_preview.clear(
            "保存有效砂轮、粒度和工艺参数后，可以生成统计等效磨粒载荷图。"
        )
        self.process_editor = process_editor
        self._update_material_behavior()
        self._update_grit_path()

        analysis_editor = ttk.LabelFrame(workspace, text="分析任务选择与统一计算调度", padding=8)
        analysis_editor.columnconfigure(0, weight=1, uniform="analysis")
        analysis_editor.columnconfigure(1, weight=1, uniform="analysis")
        analysis_left = ttk.Frame(analysis_editor)
        analysis_right = ttk.Frame(analysis_editor)
        analysis_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        analysis_right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        def resize_analysis(event):
            if event.width < 1000:
                analysis_left.grid_configure(row=0, column=0, columnspan=2, padx=0)
                analysis_right.grid_configure(row=1, column=0, columnspan=2, padx=0, pady=(8, 0))
            else:
                analysis_left.grid_configure(row=0, column=0, columnspan=1, padx=(0, 6))
                analysis_right.grid_configure(row=0, column=1, columnspan=1, padx=(6, 0), pady=0)
        analysis_editor.bind("<Configure>", resize_analysis)
        ttk.Label(analysis_left, text="分析任务").pack(anchor="w")
        self.analysis_mode_combo = ttk.Combobox(
            analysis_left, textvariable=self.analysis_mode_var,
            values=tuple(item.display_name for item in ANALYSIS_MODE_DEFINITIONS.values()),
            state="readonly",
        )
        self.analysis_mode_combo.pack(fill="x", pady=(2, 6))
        from .literature_integration import LiteratureParameterDialog
        self.literature_parameters_button = ttk.Button(analysis_left, text="文献参数设置 / 同步工程工艺", command=lambda: LiteratureParameterDialog(self))
        self.literature_parameters_button.pack(fill="x", pady=(0, 6))
        self.analysis_mode_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_analysis_workspace()
        )
        task_box = ttk.LabelFrame(analysis_left, text="任务说明与模型假设", padding=6)
        task_box.pack(fill="x", pady=(0, 6))
        ttk.Label(task_box, textvariable=self.analysis_task_var, justify="left", wraplength=390).pack(fill="x")
        checks_box = ttk.LabelFrame(analysis_left, text="输入状态", padding=6)
        checks_box.pack(fill="x", pady=(0, 6))
        ttk.Label(checks_box, textvariable=self.analysis_checks_var, justify="left", wraplength=390).pack(fill="x")
        output_box = ttk.LabelFrame(analysis_left, text="输出目录", padding=6)
        output_box.pack(fill="x", pady=(0, 6))
        output_box.columnconfigure(0, weight=1)
        ttk.Entry(output_box, textvariable=self.analysis_output_var).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(output_box, text="选择", command=self._choose_analysis_output).grid(row=0, column=1)
        ttk.Button(output_box, text="生成默认目录", command=self._set_default_analysis_output).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        advanced = ttk.LabelFrame(analysis_left, text="高级设置（推荐默认值可直接使用）", padding=6)
        advanced.pack(fill="x", pady=(0, 6))
        advanced.columnconfigure(1, weight=1)
        def wrap_advanced(event):
            label_width = max(100, min(260, int((event.width - 24) * .48)))
            for child in advanced.winfo_children():
                if isinstance(child, ttk.Label):
                    spanning = int(child.grid_info().get("columnspan", 1)) > 1
                    child.configure(wraplength=max(100, event.width - 24) if spanning else label_width)
        advanced.bind("<Configure>", wrap_advanced)
        ttk.Label(
            advanced,
            textvariable=self.analysis_advanced_notice_var,
            style="Warning.TLabel",
            justify="left",
            wraplength=390,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        advanced_fields = (
            ("扫描位置数", "scan_base_position_count"), ("加载步数", "loading_step_count"),
            ("卸载步数", "unloading_step_count"), ("Newton 相对残差容差", "newton_residual_relative_tolerance"),
            ("Newton 最大迭代", "newton_maximum_iterations"), ("历史基础子步", "history_base_substep_count"),
            ("最终卸载子步", "history_final_unloading_substep_count"), ("最小等效磨粒组数", "grain_minimum_group_count"),
            ("最大等效磨粒组数", "grain_maximum_group_count"), ("随机种子", "grain_random_seed"),
            ("载荷采样点数", "load_point_count"),
            ("单磨粒工件宽度 [μm]", "single_grain_workpiece_width_um"),
            ("单磨粒工件高度 [μm]", "single_grain_workpiece_height_um"),
            ("单磨粒网格尺寸 [μm]", "single_grain_mesh_size_um"),
            ("单磨粒显式厚度 [mm]", "single_grain_thickness_mm"),
            ("磨粒类型 [rigid_circle/rigid_circular_arc/rounded_circle/rounded_wedge]", "single_grain_type"),
            ("磨粒圆角半径 [μm]", "single_grain_radius_um"),
            ("圆弧起始角 [deg, CCW]", "single_grain_arc_start_angle_deg"),
            ("圆弧终止角 [deg, CCW]", "single_grain_arc_end_angle_deg"),
            ("楔形尖端半径 [μm]", "single_grain_tip_radius_um"),
            ("楔形高度 [μm]", "single_grain_wedge_height_um"),
            ("本征前角 [deg]", "single_grain_intrinsic_rake_angle_deg"),
            ("本征后角 [deg]", "single_grain_intrinsic_clearance_angle_deg"),
            ("磨粒姿态角 [deg]", "single_grain_pose_angle_deg"),
            ("名义方向 [positive_x/negative_x]", "single_grain_nominal_direction"),
            ("预设去除模式 [element_ids/geometric_region]", "single_grain_preset_removal_mode"),
            ("预设去除单元 ID [逗号分隔]", "single_grain_preset_removal_element_ids"),
            ("预设去除多边形 JSON [m]", "single_grain_preset_removal_polygon_json"),
            ("初始间隙 [μm]", "single_grain_initial_clearance_um"),
            ("压入深度 [μm]", "single_grain_indentation_depth_um"),
            ("划擦距离 [μm]", "single_grain_scratch_distance_um"),
            ("法向算法 [penalty/augmented_lagrangian]", "single_grain_normal_algorithm"),
            ("法向罚因子", "single_grain_penalty_factor"),
            ("切向罚刚度比例", "single_grain_tangential_penalty_ratio"),
            ("增广松弛系数", "single_grain_augmented_relaxation"),
            ("增广最大迭代", "single_grain_augmented_maximum_iterations"),
            ("摩擦系数", "single_grain_friction_coefficient"),
            ("摩擦正则化比例", "single_grain_friction_regularization_ratio"),
            ("启用自动损伤分离 [true/false]", "single_grain_damage_enabled"),
            ("损伤演化模型", "single_grain_damage_evolution"),
            ("三轴度－起始应变表 JSON", "single_grain_damage_failure_table_json"),
            ("三轴度截止值", "single_grain_damage_triaxiality_cutoff"),
            ("断裂能 Gf [J/m²]", "single_grain_damage_fracture_energy"),
            ("损伤事件容差", "single_grain_damage_event_tolerance"),
            ("每位置最大分离事件数", "single_grain_damage_max_events"),
            ("面积绝对容差 [m²]", "single_grain_damage_area_tolerance_m2"),
            ("损伤参数来源", "single_grain_damage_parameter_source"),
            ("损伤标定状态 [uncalibrated/preliminary/calibrated]", "single_grain_damage_calibration_status"),
            ("损伤适用性说明", "single_grain_damage_applicability_notes"),
        )
        for row, (label, key) in enumerate(advanced_fields, start=1):
            label_widget = ttk.Label(advanced, text=label, wraplength=220, justify="left")
            label_widget.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
            if key == "single_grain_damage_evolution":
                entry = ttk.Combobox(advanced, textvariable=self.analysis_settings_vars[key], width=24,
                    values=("linear_fracture_energy_softening", "energetic_rational_fracture"))
            else:
                entry = ttk.Entry(advanced, textvariable=self.analysis_settings_vars[key], width=16)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            self.analysis_advanced_labels[key] = label_widget
            self.analysis_advanced_entries[key] = entry
        ttk.Label(advanced, text=(
            "有理模型：Gf 为损伤耗散，塑性热单独计入。\n"
            "旧线性模型保留原工程含义，完整能量验收尚未通过。"),
            justify="left", wraplength=310).grid(
                row=len(advanced_fields) + 1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Button(advanced, text="恢复推荐默认值", command=self._restore_recommended_analysis_settings).grid(row=len(advanced_fields) + 2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.analysis_run_button = ttk.Button(analysis_editor, text="运行分析", command=self._start_workbench_analysis)
        self.analysis_run_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        mapping_box = ttk.LabelFrame(analysis_right, text="工程参数到求解输入映射", padding=6)
        mapping_box.pack(fill="x", pady=(0, 6))
        ttk.Label(mapping_box, textvariable=self.analysis_mapping_var, justify="left", wraplength=430).pack(fill="x")
        status_box = ttk.LabelFrame(analysis_right, text="计算状态", padding=6)
        status_box.pack(fill="x", pady=(0, 6))
        ttk.Label(status_box, textvariable=self.analysis_status_var, justify="left", wraplength=430).pack(fill="x")
        self.analysis_progress_toggle_button = ttk.Button(
            status_box,
            text="隐藏进度详情",
            command=self._toggle_analysis_progress_details,
        )
        self.analysis_progress_toggle_button.pack(anchor="e", pady=(4, 0))
        self.analysis_progress_detail_label = ttk.Label(
            status_box,
            textvariable=self.analysis_progress_detail_var,
            justify="left",
            wraplength=430,
        )
        self.analysis_progress_detail_label.pack(fill="x", pady=(4, 0))
        self.analysis_progress = ttk.Progressbar(status_box, variable=self.analysis_progress_var, mode="determinate")
        self.analysis_progress.pack(fill="x", pady=(5, 0))
        summary_box = ttk.LabelFrame(analysis_right, text="最小结果摘要", padding=6)
        summary_box.pack(fill="x", pady=(0, 6))
        self.analysis_summary_text = tk.Text(summary_box, width=1, height=10, wrap="word", state="disabled", font=(GUI_FONT_FAMILY, 9))
        self.analysis_summary_text.pack(fill="x")
        preview_box = ttk.LabelFrame(analysis_right, text="代表图预览", padding=6)
        preview_box.pack(fill="both", expand=True)
        ttk.Label(
            preview_box,
            text="历史结果图片保留生成时的标题格式。重新运行分析后，新结果将采用当前版本的显示样式。",
            style="Warning.TLabel",
            justify="left",
            wraplength=420,
        ).pack(fill="x", pady=(0, 5))
        self.analysis_image_preview = ImagePreview(
            preview_box, source="analysis-workbench",
            error_handler=lambda message: messagebox.showerror("分析图片错误", message, parent=self.root),
            open_handler=self._open_path,
        )
        self.analysis_image_preview.pack(fill="both", expand=True)
        self.analysis_image_preview.clear("计算成功后显示当前任务的一张真实代表图。")
        ttk.Button(preview_box, text="打开输出文件夹", command=self._open_workbench_analysis_output).pack(fill="x", pady=(5, 0))
        self.analysis_editor = analysis_editor

        result_center_editor = ttk.LabelFrame(workspace, text="分类结果中心", padding=8)
        self.result_center_view = ResultCenterView(
            result_center_editor,
            open_handler=lambda path: self._open_path(path),
        )
        self.result_center_view.pack(fill="both", expand=True)
        self.result_center_view.refresh_handler = self._refresh_result_center
        self.result_center_view.relink_handler = self._relink_result
        self.result_center_editor = result_center_editor

        summaries = ttk.Frame(workspace)
        summaries.pack(fill="both", expand=True)
        self.workbench_summaries = summaries
        summaries.columnconfigure(0, weight=1)
        summaries.columnconfigure(1, weight=1)
        input_box = ttk.LabelFrame(summaries, text="输入摘要", padding=8)
        output_box = ttk.LabelFrame(summaries, text="输出摘要", padding=8)
        input_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        output_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ttk.Label(
            input_box,
            textvariable=self.workbench_input_var,
            justify="left",
            wraplength=390,
        ).pack(fill="both", expand=True)
        ttk.Label(
            output_box,
            textvariable=self.workbench_output_var,
            justify="left",
            wraplength=390,
        ).pack(fill="both", expand=True)
        self.workbench_limitation_label = ttk.Label(
            workspace,
            textvariable=self.workbench_limitation_var,
            style="Warning.TLabel",
            justify="left",
            wraplength=820,
        )
        self.workbench_limitation_label.pack(fill="x", pady=(10, 0))

        self._refresh_work_tree("工程")

    def _build_legacy_layout(self, parent: ttk.Frame) -> None:

        ttk.Label(
            parent,
            text="该页面保留用于兼容旧版严格 JSON 工作流。一般工程计算建议使用“工程工作台”。",
            style="Warning.TLabel",
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 8))

        mode_frame = ttk.LabelFrame(parent, text="1. 计算模式", padding=8)
        mode_frame.pack(fill="x", padx=14, pady=(0, 8))
        for label, value in (
            ("单位置线弹性计算", SINGLE_MODE),
            ("完整单程线弹性扫描", SCAN_MODE),
            ("机制化弹塑性完整单程", MECHANISM_HISTORY_MODE),
        ):
            radio = ttk.Radiobutton(
                mode_frame,
                text=label,
                value=value,
                variable=self.variables["mode"],
                command=self._on_mode_changed,
            )
            radio.pack(side="left", padx=(0, 18))
            self._running_widgets.append(radio)
        load_button = ttk.Button(mode_frame, text="加载严格 JSON", command=self._load_config)
        save_button = ttk.Button(mode_frame, text="保存当前合法 JSON", command=self._save_config)
        example_button = ttk.Button(
            mode_frame,
            text="加载机制化默认示例",
            command=self._load_mechanism_example,
        )
        help_button = ttk.Button(
            mode_frame,
            text="机制化模式说明",
            command=self._show_mechanism_history_help,
        )
        load_button.pack(side="right", padx=4)
        save_button.pack(side="right", padx=4)
        example_button.pack(side="right", padx=4)
        help_button.pack(side="right", padx=4)
        self._running_widgets.extend(
            (load_button, save_button, example_button, help_button)
        )

        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        parameter_host = ttk.LabelFrame(paned, text="2. 参数设置", padding=0)
        result_host = ttk.LabelFrame(paned, text="3. 运行与结果", padding=8)
        paned.add(parameter_host, weight=4)
        paned.add(result_host, weight=6)
        self._build_parameter_panel(parameter_host)
        self._build_result_panel(result_host)

    def _build_parameter_panel(self, parent: ttk.LabelFrame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0, background="#f4f4f4")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        interior = ttk.Frame(canvas, padding=8)
        window_id = canvas.create_window((0, 0), window=interior, anchor="nw")
        interior.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
        from .scrolling import bind_canvas_wheel
        bind_canvas_wheel(canvas)

        ttk.Label(
            interior,
            text=(
                "机制化高级模式通过严格 JSON 接入。加载合法配置后，"
                "传统参数表单会锁定；完整中文参数表单留到后续版本。"
            ),
            style="Warning.TLabel",
            justify="left",
            wraplength=500,
        ).pack(fill="x", pady=(0, 7))

        geometry = ttk.LabelFrame(interior, text="工件与砂轮", padding=8)
        geometry.pack(fill="x", pady=(0, 7))
        self._entry(geometry, 0, "工件长度 [mm]", "workpiece_length_mm")
        self._entry(geometry, 1, "工件高度 [mm]", "workpiece_height_mm")
        self._entry(geometry, 2, "砂轮直径 [mm]", "wheel_diameter_mm")
        self._entry(geometry, 3, "磨削深度 [um]", "depth_of_cut_um")
        self._entry(
            geometry,
            4,
            "指定/参考位置：砂轮最低点 x [mm]",
            "wheel_lowest_point_x_mm",
        )
        ttk.Label(
            geometry,
            text=(
                "指定位置模式：这是实际求解位置。\n"
                "完整扫描模式：这是额外保留的参考采样位置。"
            ),
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 4))

        process = ttk.LabelFrame(interior, text="工艺参数与经验磨削力估算", padding=8)
        process.pack(fill="x", pady=(0, 7))
        self._entry(process, 0, "砂轮圆周线速度 vs [m/s]", "wheel_surface_speed_m_per_s")
        self._entry(process, 1, "工件相对进给速度 vw [m/s]", "workpiece_feed_speed_m_per_s")
        self._entry(
            process,
            2,
            "磨削宽度 b（同时作为二维分析厚度）[mm]",
            "grinding_width_and_thickness_mm",
        )
        self._entry(
            process,
            3,
            "比磨削能 us [J/mm^3]（需实验或文献标定）",
            "specific_grinding_energy_J_per_mm3",
        )
        self._entry(
            process,
            4,
            "磨削力比 Rnt = Fn/Ft（需标定）",
            "normal_to_tangential_force_ratio",
        )
        self._entry(process, 5, "经验参数集 ID（内部标识）", "calibration_id")
        ttk.Label(
            process,
            textvariable=self.calibration_name_var,
            style="Warning.TLabel",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(1, 3))
        self._entry(process, 7, "us 数据来源或标定依据", "specific_grinding_energy_source")
        self._entry(process, 8, "Rnt 数据来源或标定依据", "force_ratio_source")
        self._entry(process, 9, "参数适用工况与限制", "applicability_notes")
        help_button = ttk.Button(
            process,
            text="查看经验模型公式与参数说明",
            command=self._show_force_model_help,
        )
        help_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        self._running_widgets.append(help_button)
        self._parameter_widgets.append(help_button)

        material = ttk.LabelFrame(interior, text="材料与网格", padding=8)
        material.pack(fill="x", pady=(0, 7))
        self._entry(material, 0, "弹性模量 E [GPa]", "elastic_modulus_GPa")
        self._entry(material, 1, "泊松比 nu", "poisson_ratio")
        self._entry(material, 2, "网格目标尺寸 [mm]", "mesh_target_size_mm")


        direction = ttk.LabelFrame(interior, text="方向和扫描设置", padding=8)
        direction.pack(fill="x", pady=(0, 7))
        self._combo(direction, 0, "轨迹方向", "relative_feed_direction")
        self._combo(direction, 1, "切向力方向", "tangential_force_direction")
        self.scan_count_entry = self._entry(
            direction,
            2,
            "基础扫描位置数 [5-101]（关键位置会自动插入）",
            "scan_base_position_count",
        )

        boundary = ttk.LabelFrame(interior, text="固定边界条件（首版只读）", padding=8)
        boundary.pack(fill="x", pady=(0, 7))
        ttk.Label(
            boundary,
            text="底面固定  |  顶面候选磨削边界  |  左右侧自由",
        ).pack(anchor="w")

        output = ttk.LabelFrame(interior, text="输出目录", padding=8)
        output.pack(fill="x", pady=(0, 4))
        output.columnconfigure(0, weight=1)
        self.output_entry = ttk.Entry(output, textvariable=self.variables["output_directory"])
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        browse = ttk.Button(output, text="选择文件夹", command=self._choose_output)
        browse.grid(row=0, column=1)
        self._running_widgets.extend((self.output_entry, browse))

    def _entry(self, parent: ttk.LabelFrame, row: int, label: str, field: str) -> ttk.Entry:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        entry = ttk.Entry(parent, textvariable=self.variables[field])
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        self._running_widgets.append(entry)
        self._parameter_widgets.append(entry)
        return entry

    def _combo(self, parent: ttk.LabelFrame, row: int, label: str, field: str) -> ttk.Combobox:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        combo = ttk.Combobox(
            parent,
            textvariable=self.variables[field],
            values=("positive_x", "negative_x"),
            state="readonly",
        )
        combo.grid(row=row, column=1, sticky="ew", pady=3)
        self._running_widgets.append(combo)
        self._readonly_widgets.append(combo)
        self._parameter_widgets.append(combo)
        self._parameter_readonly_widgets.append(combo)
        return combo

    def _show_force_model_help(self) -> None:
        self._show_help_dialog("经验模型公式与参数说明", FORCE_MODEL_HELP_TEXT)

    def _show_mechanism_history_help(self) -> None:
        self._show_help_dialog("机制化弹塑性完整单程说明", MECHANISM_HISTORY_HELP_TEXT)

    def _show_help_dialog(self, title: str, content: str) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("760x680")
        dialog.minsize(620, 480)
        dialog.transient(self.root)

        body = ttk.Frame(dialog, padding=10)
        body.pack(fill="both", expand=True)
        text = tk.Text(body, wrap="word", font=(GUI_FONT_FAMILY, 10))
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.insert("1.0", content)
        text.configure(state="disabled")
        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=(0, 10))
        dialog.focus_set()

    def _build_result_panel(self, parent: ttk.LabelFrame) -> None:
        status = ttk.Frame(parent)
        status.pack(fill="x")
        ttk.Label(status, text="校验状态：", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.validation_var, style="Status.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(status, text="当前阶段：", style="Heading.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(status, textvariable=self.stage_var, style="Stage.TLabel").grid(
            row=1, column=1, sticky="w", pady=(4, 0)
        )
        status.columnconfigure(1, weight=1)

        self.progress = ttk.Progressbar(status, variable=self.progress_var, maximum=1.0)
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 4))
        self.run_button = ttk.Button(status, text="校验并开始计算", command=self._start_run)
        self.run_button.grid(row=0, column=2, rowspan=3, sticky="ns", padx=(10, 0))

        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, pady=(8, 0))
        log_tab = ttk.Frame(notebook, padding=6)
        summary_tab = ttk.Frame(notebook, padding=6)
        preview_tab = ttk.Frame(notebook, padding=6)
        notebook.add(log_tab, text="运行日志")
        notebook.add(summary_tab, text="结果摘要")
        notebook.add(preview_tab, text="PNG 预览")

        self.log_text = tk.Text(log_tab, wrap="word", state="disabled", font=("Consolas", 10))
        log_scroll = ttk.Scrollbar(log_tab, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        self.summary_text = tk.Text(summary_tab, wrap="word", state="disabled", font=("Microsoft YaHei UI", 10))
        summary_scroll = ttk.Scrollbar(summary_tab, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        summary_scroll.pack(side="right", fill="y")
        self.summary_text.pack(fill="both", expand=True)

        preview_toolbar = ttk.Frame(preview_tab)
        preview_toolbar.pack(fill="x", pady=(0, 6))
        self.result_preview_toolbar = preview_toolbar
        ttk.Button(preview_toolbar, text="上一张", command=lambda: self._step_image(-1)).pack(side="left")
        ttk.Button(preview_toolbar, text="下一张", command=lambda: self._step_image(1)).pack(side="left", padx=4)
        self.image_combo = ttk.Combobox(
            preview_toolbar,
            textvariable=self.image_choice_var,
            state="readonly",
        )
        self.image_combo.pack(side="left", fill="x", expand=True, padx=4)
        self.image_combo.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_image())
        self.open_output_button = ttk.Button(
            preview_toolbar,
            text="打开输出目录",
            command=self._open_output,
            state="disabled",
        )
        self.open_output_button.pack(side="right")

        self.image_explanation_label = ttk.Label(
            preview_tab,
            textvariable=self.image_explanation_var,
            style="Stage.TLabel",
            justify="left",
            wraplength=900,
        )
        self.image_explanation_label.pack(fill="x", pady=(0, 6))
        self.result_image_preview = ImagePreview(
            preview_tab,
            source="analysis",
            error_handler=lambda message: self._log(message),
            open_handler=self._open_path,
        )
        self.result_image_preview.pack(fill="both", expand=True)
        self.result_image_preview.clear("计算成功后，可从下拉框按需加载现有 PNG。")
        self.preview_frame = self.result_image_preview.canvas
        self.preview_label = self.result_image_preview.canvas

        limitation = ttk.Label(
            parent,
            textvariable=self.limitation_var,
            style="Warning.TLabel",
            justify="left",
        )
        limitation.pack(fill="x", pady=(8, 0))

    def _form(self) -> GuiForm:
        values = {name: self.variables[name].get() for name in FORM_FIELDS}
        return GuiForm(**values)

    def _set_form(self, form: GuiForm) -> None:
        for name, value in asdict(form).items():
            self.variables[name].set(str(value))
        self._apply_mode_state()

    def _on_mode_changed(self) -> None:
        try:
            form = self._form().with_mode(self.variables["mode"].get())
        except GuiInputError as exc:
            messagebox.showerror("模式错误", str(exc), parent=self.root)
            return
        self._set_form(form)

    def _apply_mode_state(self) -> None:
        mode = self.variables["mode"].get()
        self.limitation_var.set(limitation_text(mode))
        mechanism_mode = mode == MECHANISM_HISTORY_MODE
        for widget in self._parameter_widgets:
            widget.configure(state="disabled" if mechanism_mode else "normal")
        if not mechanism_mode:
            for widget in self._parameter_readonly_widgets:
                widget.configure(state="readonly")
        self.scan_count_entry.configure(
            state="normal" if mode == SCAN_MODE else "disabled"
        )

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="选择 GrindCAE 输出目录")
        if selected:
            self.variables["output_directory"].set(selected)

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="加载单位置、完整单程扫描或机制化 JSON",
            filetypes=(("JSON 配置", "*.json"), ("所有文件", "*.*")),
        )
        if not path:
            return
        try:
            form = load_form(path)
        except GuiInputError as exc:
            messagebox.showerror("加载失败", str(exc), parent=self.root)
            return
        self._set_form(form)
        self.validation_var.set("已加载并通过严格 Schema 校验")
        self._log(f"已加载配置：{Path(path).resolve()}")

    def _load_mechanism_example(self) -> None:
        try:
            path = mechanism_history_example_path()
            form = load_form(path)
        except (GuiInputError, OSError) as exc:
            messagebox.showerror("示例加载失败", str(exc), parent=self.root)
            return
        self._set_form(form)
        self.validation_var.set("已加载机制化默认示例并通过严格 Schema 校验")
        self._log(f"已加载机制化默认示例：{path}")

    def _save_config(self) -> None:
        try:
            form = self._form()
            form.build_case()
        except GuiInputError as exc:
            self.validation_var.set("校验失败")
            messagebox.showerror("无法保存", str(exc), parent=self.root)
            return
        initial = {
            SINGLE_MODE: "single_position.json",
            SCAN_MODE: "single_pass_scan.json",
            MECHANISM_HISTORY_MODE: "mechanism_history_pass.json",
        }[form.mode]
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存合法 UTF-8 JSON",
            defaultextension=".json",
            initialfile=initial,
            filetypes=(("JSON 配置", "*.json"),),
        )
        if not path:
            return
        try:
            output = save_form(form, path)
        except GuiInputError as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return
        self.validation_var.set("参数有效，JSON 已保存")
        self._log(f"已保存严格 Schema JSON：{output}")

    def _start_run(self) -> None:
        try:
            form = self._form()
            self.controller.start(form)
        except GuiInputError as exc:
            self.validation_var.set("校验失败")
            self.stage_var.set("尚未启动 Gmsh")
            self._log(f"输入错误：{exc}")
            messagebox.showerror("参数校验失败", str(exc), parent=self.root)
            return
        self.workbench.begin_analysis()
        self._refresh_work_tree("分析")
        self._set_running(True)
        self._clear_result()

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        for widget in self._running_widgets:
            try:
                widget.configure(state="disabled" if running else "normal")
            except tk.TclError:
                pass
        if not running:
            for widget in self._readonly_widgets:
                widget.configure(state="readonly")
            self._apply_mode_state()

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self.controller.events.get_nowait())
        except Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: GuiEvent) -> None:
        self.stage_var.set(event.message)
        self._log(event.message)
        if event.log_detail:
            self._log(event.log_detail.rstrip())
        if event.kind == "validated":
            self.validation_var.set("通过严格 Schema 校验")
        elif event.kind == "started":
            self.progress_var.set(0.0)
        elif event.kind == "progress" and event.current is not None and event.total:
            self.progress.configure(maximum=event.total)
            self.progress_var.set(event.current)
        elif event.kind == "success" and event.result is not None:
            self.validation_var.set("计算与结果发布成功")
            self._show_result(event.result.summary_text, event.result.images)
            self._current_output = event.result.output_directory
            self.open_output_button.configure(state="normal")
            result_format = str(event.result.summary.get("result_format", ""))
            self.workbench.register_published_result(
                mode=event.result.mode,
                result_format=result_format,
                output_directory=event.result.output_directory,
                summary_path=event.result.summary_path,
                artifact_paths={image.label: image.path for image in event.result.images},
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            latest = self.workbench.project.latest_analysis_result
            if latest is not None:
                try:
                    self.recent_result_store.add(
                        recent_entry_from_record(
                            latest, project_id=self.workbench.project.project_id
                        )
                    )
                except ProjectError:
                    pass
            self._write_recovery_snapshot()
            self._refresh_work_tree("结果")
        elif event.kind == "error":
            self.validation_var.set("计算失败；未报告旧结果")
            self.workbench.mark_analysis_failed(event.message)
            self._refresh_work_tree("分析")
            messagebox.showerror("GrindCAE 计算失败", event.message, parent=self.root)
        elif event.kind == "finished":
            self._set_running(False)

    def _log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_result(self) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")
        self._images = ()
        self._preview_photo = None
        self.image_combo.configure(values=())
        self.image_choice_var.set("")
        self.result_image_preview.clear("等待当前计算完成。")
        self.image_explanation_var.set("")
        self.open_output_button.configure(state="disabled")

    def _show_result(self, summary: str, images: tuple[GuiImage, ...]) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", summary)
        self.summary_text.configure(state="disabled")
        self._images = images
        labels = tuple(image.label for image in images)
        self.image_combo.configure(values=labels)
        if labels:
            self.image_choice_var.set(labels[0])
            self._load_selected_image()

    def _selected_image_index(self) -> int | None:
        label = self.image_choice_var.get()
        return next((index for index, item in enumerate(self._images) if item.label == label), None)

    def _step_image(self, offset: int) -> None:
        if not self._images:
            return
        current = self._selected_image_index()
        index = ((current or 0) + offset) % len(self._images)
        self.image_choice_var.set(self._images[index].label)
        self._load_selected_image()

    def _schedule_preview_reload(self, _event=None) -> None:
        if self._preview_after_id is not None:
            self.root.after_cancel(self._preview_after_id)
        self._preview_after_id = self.root.after(150, self._load_selected_image)

    def _load_selected_image(self) -> None:
        self._preview_after_id = None
        index = self._selected_image_index()
        if index is None:
            return
        path = self._images[index].path
        try:
            self.image_explanation_var.set(self._images[index].explanation)
            self.result_image_preview.set_image(path, source="analysis")
        except (OSError, ValueError) as exc:
            self._preview_photo = None
            self.result_image_preview.clear(
                f"PNG 预览失败，但计算结果仍保留。\n{path}\n{exc}"
            )
            self._log(f"PNG 预览失败：{path}：{exc}")
            return
        self._preview_photo = self.result_image_preview._photo

    def _open_output(self) -> None:
        path = getattr(self, "_current_output", None)
        if path is None:
            return
        self._open_path(Path(path))

    def _open_path(self, path: Path) -> None:
        try:
            os.startfile(str(path.resolve()))
        except OSError as exc:
            messagebox.showerror("无法打开", f"{path}\n{exc}", parent=self.root)

    def _on_close(self) -> None:
        if self.controller.is_running or self.background_task_gate.owner is not None:
            messagebox.showwarning(
                "计算仍在运行",
                "当前计算不能强制终止。请等待事务化发布或失败回滚完成后再关闭窗口。",
                parent=self.root,
            )
            return
        if not self._confirm_discard_or_save():
            return
        try:
            self._session_marker_path.unlink(missing_ok=True)
        except OSError:
            pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GrindCaeApp(root)
    root.mainloop()
