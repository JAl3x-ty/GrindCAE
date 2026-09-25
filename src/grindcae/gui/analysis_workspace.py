"""Presentation-neutral state for the formal 2.8 analysis workbench page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .analysis import (
    ANALYSIS_MODE_DEFINITIONS,
    AnalysisInputBuilder,
    AnalysisProjectData,
    AnalysisSettings,
    BuiltAnalysisInput,
    analysis_mode_definition,
)
from .geometry import ParametricGeometryCase
from .project import GrindCaeProject
from .project import ProjectNodeStatus
from .analysis_results import (
    AnalysisPublishedResult,
    AnalysisResultAdapter,
    AnalysisResultError,
)
from grindcae.presentation_labels import (
    LOAD_MODEL_LABELS_ZH,
    MATERIAL_MODEL_LABELS_ZH,
    load_model_label,
)


@dataclass(frozen=True, slots=True)
class AnalysisWorkspaceState:
    mode_id: str
    display_name: str
    can_run: bool
    task_summary: str
    mapping_summary: str
    input_checks: tuple[str, ...]
    error_message: str
    built: BuiltAnalysisInput | None


@dataclass(frozen=True, slots=True)
class AnalysisPresentationState:
    status: str
    detail_text: str
    output_summary: tuple[str, ...]
    representative_image: Path | None
    published_result: AnalysisPublishedResult | None = None


_MATERIAL_LABELS = {
    "literature_elastoplastic_single_pass": MATERIAL_MODEL_LABELS_ZH["j2_elastoplastic"],
    "linear_elastic_single_position": MATERIAL_MODEL_LABELS_ZH["linear_elastic"],
    "linear_elastic_single_pass": MATERIAL_MODEL_LABELS_ZH["linear_elastic"],
    "elastoplastic_single_position": MATERIAL_MODEL_LABELS_ZH["j2_elastoplastic"],
    "elastoplastic_single_pass": MATERIAL_MODEL_LABELS_ZH["j2_elastoplastic"],
    "mechanism_elastoplastic_single_pass": MATERIAL_MODEL_LABELS_ZH["j2_elastoplastic"],
    "single_grain_high_fidelity": MATERIAL_MODEL_LABELS_ZH["j2_elastoplastic"],
}


def analysis_input_summary(
    mode_id: str, *, input_valid: bool
) -> tuple[str, ...]:
    definition = analysis_mode_definition(mode_id, require_enabled=True)
    if mode_id == "literature_grinding_force":
        return (f"当前模式：{definition.display_name}", "模型：440C文献逐磨粒解析力", "输入：文献参数设置（随工程保存）",
                "无需FE网格；同步工程工艺需显式点击", f"输入校验：{'已通过' if input_valid else '未通过'}")
    scope = (
        "单磨粒轨迹"
        if definition.process_scope == "single_grain_trajectory"
        else "单位置" if "single_position" in definition.process_scope else "完整单程"
    )
    assumption = "二维平面应力" if definition.analysis_assumption == "plane_stress" else "二维平面应变"
    return (
        f"当前模式：{definition.display_name}",
        f"分析假设：{assumption}",
        f"计算范围：{scope}",
        f"材料模型：{_MATERIAL_LABELS[mode_id]}",
        f"载荷模型：{load_model_label(definition.load_type)}",
        f"输入校验：{'已通过' if input_valid else '未通过'}",
    )


def analysis_presentation_state(
    project: GrindCaeProject,
    workspace: AnalysisWorkspaceState,
    *,
    result_reader: Callable[[str, str | Path], AnalysisPublishedResult] | None = None,
) -> AnalysisPresentationState:
    definition = analysis_mode_definition(workspace.mode_id, require_enabled=True)
    prefix = f"当前分析模式：{definition.display_name}"
    saved = (
        AnalysisProjectData.from_mapping(project.analysis)
        if project.analysis is not None else None
    )
    if saved is not None and saved.mode == workspace.mode_id:
        if project.node_status("分析") is ProjectNodeStatus.STALE:
            return AnalysisPresentationState(
                "stale",
                "结果已过期：当前工程输入已经变化。\n历史结果仍保留用于追溯，请重新运行当前分析模式。",
                ("计算状态：结果已过期", f"分析模式：{definition.display_name}"),
                None,
            )
        if saved.status == "running":
            return AnalysisPresentationState(
                "running",
                f"正在计算：{definition.display_name}\n当前任务完成后将显示新的结果摘要和代表图。\n历史结果仍保留用于追溯，旧结果不会作为当前结果显示。",
                ("计算状态：正在计算", f"分析模式：{definition.display_name}"),
                None,
            )
        if saved.status == "failed":
            reason = saved.last_run.get("error_message") or "未知错误"
            return AnalysisPresentationState(
                "failed",
                f"计算失败：{definition.display_name}\n失败原因：{reason}\n历史结果仍保留用于追溯，本次计算没有发布为有效结果。",
                ("计算状态：计算失败", f"分析模式：{definition.display_name}"),
                None,
            )
    if workspace.built is not None:
        record = project.current_analysis_result(
            workspace.mode_id, workspace.built.input_fingerprint
        )
        if record is not None:
            reader = result_reader or AnalysisResultAdapter().read
            try:
                result = reader(record.analysis_type, record.output_directory)
            except (AnalysisResultError, OSError, ValueError) as exc:
                return AnalysisPresentationState(
                    "unavailable",
                    f"最近结果记录不可用：{definition.display_name}\n原因：{exc}\n请重新运行当前分析模式。",
                    ("计算状态：结果记录不可用", f"分析模式：{definition.display_name}"),
                    None,
                )
            return AnalysisPresentationState(
                "completed",
                result.summary_text,
                (
                    "计算状态：已完成",
                    f"分析模式：{definition.display_name}",
                    f"完成时间：{record.created_at}",
                    f"代表图：{result.representative_image.name}",
                    f"输出目录：{result.output_directory}",
                ),
                result.representative_image,
                result,
            )
    if workspace.mode_id in {"literature_grinding_force", "literature_elastoplastic_single_pass"} and any(record.analysis_type == workspace.mode_id for record in project.results):
        return AnalysisPresentationState(
            "stale", "文献参数已改变；已有结果保留在结果中心，请重新计算当前输入。",
            ("计算状态：结果已过期", f"分析模式：{definition.display_name}"), None,
        )
    if workspace.can_run:
        return AnalysisPresentationState(
            "configured",
            f"已配置，等待运行：{definition.display_name}\n输入校验已通过，可以运行当前分析任务。",
            ("计算状态：已配置，等待运行", f"分析模式：{definition.display_name}"),
            None,
        )
    return AnalysisPresentationState(
        "not_run",
        f"尚未计算：{definition.display_name}\n{workspace.error_message or prefix}",
        ("计算状态：尚未计算", f"分析模式：{definition.display_name}"),
        None,
    )


def default_analysis_output_directory(
    mode_id: str,
    *,
    project_file: str | Path | None = None,
    unsaved_root: str | Path | None = None,
    now: datetime | None = None,
    exists: Callable[[Path], bool] = Path.exists,
) -> Path:
    analysis_mode_definition(mode_id, require_enabled=True)
    if project_file is not None:
        project_path = Path(project_file).expanduser().resolve()
        root = project_path.parent / f"{project_path.stem}_results"
    else:
        root = Path(unsaved_root).expanduser().resolve() if unsaved_root is not None else Path.home() / "Documents" / "GrindCAE" / "results"
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = root / f"{timestamp}_{mode_id}"
    candidate = base
    suffix = 2
    while exists(candidate):
        candidate = base.with_name(f"{base.name}_{suffix:02d}")
        suffix += 1
    return candidate


class AnalysisWorkspace:
    def __init__(self, project: GrindCaeProject, mode_id: str, settings: AnalysisSettings, output_directory: str = "") -> None:
        self.project = project
        self.mode_id = mode_id
        self.settings = settings
        self.output_directory = output_directory

    @classmethod
    def from_project(cls, project: GrindCaeProject) -> "AnalysisWorkspace":
        if project.analysis is None:
            return cls(project, next(iter(ANALYSIS_MODE_DEFINITIONS)), AnalysisSettings.recommended())
        restored = AnalysisProjectData.from_mapping(project.analysis)
        return cls(project, restored.mode, restored.settings, restored.output_directory)

    def evaluate(self) -> AnalysisWorkspaceState:
        definition = analysis_mode_definition(self.mode_id, require_enabled=True)
        try:
            built = AnalysisInputBuilder().build(self.project, self.mode_id, self.settings)
        except ValueError as exc:
            return AnalysisWorkspaceState(
                self.mode_id, definition.display_name, False,
                self._task_summary(definition), "尚未生成合法求解输入。",
                self._input_checks(valid=False), str(exc), None,
            )
        return AnalysisWorkspaceState(
            self.mode_id, definition.display_name, True,
            self._task_summary(definition), self._mapping_summary(built),
            self._input_checks(valid=True), "", built,
        )

    def save_configuration(self, *, status: str = "configured") -> AnalysisProjectData:
        definition = analysis_mode_definition(self.mode_id, require_enabled=True)
        previous = AnalysisProjectData.from_mapping(self.project.analysis) if self.project.analysis is not None else None
        value = AnalysisProjectData(
            mode=self.mode_id, display_name=definition.display_name, status=status,
            output_directory=self.output_directory, settings=self.settings,
            input_fingerprint=previous.input_fingerprint if previous else "",
            input_snapshot=previous.input_snapshot if previous else {},
            last_run=previous.last_run if previous else {"status": "not_run", "result_id": None, "started_at": None, "finished_at": None, "error_message": None},
            extra_fields=previous.extra_fields if previous else {},
        )
        self.project.analysis = value.to_dict()
        return value

    def _input_checks(self, *, valid: bool) -> tuple[str, ...]:
        if self.mode_id == "literature_grinding_force":
            return ("文献材料：440C", "几何/工艺：本模式参数", "FE网格：无需", f"严格输入：{'有效' if valid else '未通过'}")
        if self.mode_id == "literature_elastoplastic_single_pass":
            return tuple(f"{node}：{self.project.node_status(node).value}" for node in ("几何", "网格", "材料", "工艺")) + (
                "论文砂轮：文献参数页的粒径/组织号/角度/高度", "屈服强度：工程材料；β/断裂应力/摩擦：文献参数页",
                "工艺修改后论文力自动重算；经验比能与力比不参与预测", f"严格输入：{'有效' if valid else '未通过'}")
        labels = []
        for node in ("几何", "网格", "材料", "工艺"):
            labels.append(f"{node}：{'有效' if self.project.node_status(node).value == '已完成' else self.project.node_status(node).value}")
        wheel_resolved = isinstance(self.project.wheel, dict) and isinstance(self.project.wheel.get("resolved_specification"), dict)
        labels.extend((
            f"砂轮粒度：{'已解析' if wheel_resolved else '未解析'}",
            "分析设置：有效",
            f"严格求解输入：{'已通过校验' if valid else '未通过校验'}",
        ))
        return tuple(labels)

    @staticmethod
    def _task_summary(definition) -> str:
        if definition.mode_id == "literature_grinding_force":
            return definition.display_name + "\n" + definition.model_note
        scope = (
            "单磨粒轨迹"
            if definition.process_scope == "single_grain_trajectory"
            else "单位置" if "single_position" in definition.process_scope else "完整单程"
        )
        history = "传递历史" if definition.mode_id in ("elastoplastic_single_pass", "mechanism_elastoplastic_single_pass", "literature_elastoplastic_single_pass") else "不传递位置间塑性历史"
        return "\n".join((
            f"任务：{definition.display_name}",
            f"计算范围：{scope}",
            f"分析假设：{'平面应力' if definition.analysis_assumption == 'plane_stress' else '平面应变'}",
            f"载荷模型：{load_model_label(definition.load_type)}",
            f"历史：{history}",
            definition.model_note,
        ))

    def _mapping_summary(self, built: BuiltAnalysisInput) -> str:
        if built.definition.mode_id == "literature_elastoplastic_single_pass":
            c = built.case.literature.reference_case
            return (f"工程轮径：{c.wheel_diameter_m*1e3:g} mm；切深：{c.depth_m*1e6:g} μm\n"
                    f"工程宽度/厚度：{c.width_m*1e3:g} mm；轮速：{c.wheel_speed_m_s:g} m/s\n"
                    f"工程屈服强度：{c.yield_stress_Pa/1e6:g} MPa；砂轮与经验关系：文献参数页\n"
                    "均匀/周向条带两路线共用网格、相同总力历史、独立塑性状态；3.8.1待核验\n"
                    f"输入指纹：{built.input_fingerprint}")
        if built.definition.mode_id == "literature_grinding_force":
            c = built.case.reference_case
            return f"砂轮直径：{c.wheel_diameter_m*1e3:g} mm；切深：{c.depth_m*1e6:g} μm\n宽度：{c.width_m*1e3:g} mm；轮速：{c.wheel_speed_m_s:g} m/s\n进给：{c.feed_speed_m_s*60:g} m/min；种子：{c.seed}\n参数来源：文献参数设置；修改后需重新计算\n输入指纹：{built.input_fingerprint}"
        if built.definition.mode_id == "single_grain_high_fidelity":
            data = built.normalized_input
            return "\n".join((
                f"单磨粒工件：{data['geometry']['width'] * 1e6:g} × {data['geometry']['height'] * 1e6:g} μm",
                f"磨粒圆角半径：{data['grain']['radius_m'] * 1e6:g} μm",
                f"显式厚度：{data['analysis']['thickness'] * 1e3:g} mm",
                f"网格目标尺寸：{data['mesh']['target_size'] * 1e6:g} μm",
                f"法向算法：{data['contact']['normal_algorithm']}；罚因子：{data['contact']['normal_penalty_factor']:g}",
                f"摩擦系数：{data['contact']['friction_coefficient']:g}",
                f"规范化输入指纹：{built.input_fingerprint}",
            ))
        geometry = ParametricGeometryCase.from_mapping(self.project.geometry)
        mesh = self.project.mesh
        material = self.project.material
        process = self.project.process
        return "\n".join((
            f"计算位置：{geometry.wheel_lowest_point_x_m * 1e3:g} mm（来自几何）",
            f"砂轮直径：{geometry.wheel_diameter_m * 1e3:g} mm（来自几何）",
            f"磨削深度：{geometry.depth_of_cut_m * 1e6:g} μm（来自几何）",
            f"求解网格目标尺寸：{float(mesh['target_size_m']) * 1e3:g} mm",
            f"材料：E={float(material['elastic_modulus_Pa']) * 1e-9:g} GPa，ν={float(material['poisson_ratio']):g}",
            f"磨削宽度/分析厚度：{float(process['grinding_width_m']) * 1e3:g} mm",
            f"规范化输入指纹：{built.input_fingerprint}",
        ))
