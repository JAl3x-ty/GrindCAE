"""Presentation-neutral engineering workbench session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from .geometry import (
    GeometryPublishedResult,
    Parametric2DGeometryProvider,
    ParametricGeometryCase,
)
from .mesh_workspace import (
    MeshPublishedResult,
    TriangleMeshProvider,
    WorkbenchMeshCase,
)
from .material_workspace import WorkbenchMaterialCase
from .process_workspace import WorkbenchProcessCase, WorkbenchWheelCase

from .project import (
    GrindCaeProject,
    PROJECT_NODE_ORDER,
    ProjectNodeStatus,
    ResultRecord,
)


@dataclass(frozen=True)
class WorkbenchPage:
    title: str
    status: str
    description: str
    actions: tuple[str, ...] = ()
    input_summary: tuple[str, ...] = ()
    output_summary: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


_PAGE_TEXT = {
    "工程": "编辑工程名称和备注，并保存或打开 UTF-8 .gcase.json 工程文件。",
    "几何": "输入工件、砂轮和单程磨削参数，生成参数化二维轮廓和预览图。",
    "网格": "在有效二维几何基础上生成一阶三角形网格，并预览真实 mesh.msh。",
    "材料": "配置线弹性或弹塑性延性金属参数，以 SI 单位保存到当前工程。",
    "工艺": "读取当前几何，配置刚玉砂轮、严格粒度解析和磨削工艺参数。",
    "分析": "选择中文分析任务，将工程或文献参数适配为严格输入并在统一后台线程中计算。",
    "结果": "浏览正式分析历史，并按摘要、曲线、云图、网格和数据文件分类查看真实计算结果。",
}


class WorkbenchSession:
    def __init__(self, project: GrindCaeProject) -> None:
        self.project = project
        self.current_node = "工程"

    def tree_rows(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, self.project.node_status(name).value) for name in PROJECT_NODE_ORDER
        )

    def begin_analysis(
        self,
        *,
        mode: str | None = None,
        output_directory: str | Path | None = None,
        input_fingerprint: str = "",
        input_snapshot: dict[str, object] | None = None,
        started_at: str | None = None,
    ) -> None:
        if mode is not None:
            from .analysis import AnalysisProjectData, AnalysisSettings, analysis_mode_definition

            definition = analysis_mode_definition(mode, require_enabled=True)
            previous = (
                AnalysisProjectData.from_mapping(self.project.analysis)
                if self.project.analysis is not None
                else AnalysisProjectData(
                    mode=definition.mode_id,
                    display_name=definition.display_name,
                    status="configured",
                    output_directory="",
                    settings=AnalysisSettings.recommended(),
                )
            )
            self.project.analysis = AnalysisProjectData(
                mode=definition.mode_id,
                display_name=definition.display_name,
                status="running",
                output_directory=str(Path(output_directory).expanduser().resolve()) if output_directory is not None else previous.output_directory,
                settings=previous.settings,
                input_fingerprint=input_fingerprint,
                input_snapshot=dict(input_snapshot or {}),
                last_run={
                    "status": "running", "result_id": None,
                    "started_at": started_at or datetime.now().astimezone().isoformat(timespec="seconds"),
                    "finished_at": None, "error_message": None,
                },
                extra_fields=previous.extra_fields,
            ).to_dict()
        self.project.set_node_status("分析", ProjectNodeStatus.RUNNING, "计算正在后台执行。")
        self.project.set_node_status("结果", ProjectNodeStatus.RUNNING, "等待当前分析发布并验证真实结果。")

    def select_node(self, name: str) -> WorkbenchPage:
        if name not in PROJECT_NODE_ORDER:
            self.project.node_status(name)
        self.current_node = name
        status = self.project.node_status(name).value
        actions: tuple[str, ...] = ()
        inputs: tuple[str, ...] = ()
        outputs: tuple[str, ...] = ()
        limitations = (self.project.node_reason(name),) if self.project.node_reason(name) else ()
        if name == "工程":
            actions = ("新建工程", "保存工程", "打开工程")
            inputs = (
                f"工程名称：{self.project.project_name}",
                f"工程备注：{self.project.description or '无'}",
            )
        elif name == "几何":
            actions = ("生成二维模型",)
            if self.project.geometry is None:
                outputs = ("尚未生成二维几何。",)
            elif self.project.geometry.get("source") == "step_cad":
                status = ProjectNodeStatus.DISABLED.value
                outputs = ("STEP/CAD 导入将在后续版本实现。",)
                limitations = ("工程中的未来 CAD 字段已保留，当前版本不会解析或修改。",)
            else:
                case = ParametricGeometryCase.from_mapping(self.project.geometry)
                inputs = (
                    f"工件长度：{case.workpiece_length_m * 1.0e3:g} mm",
                    f"工件初始高度：{case.workpiece_height_m * 1.0e3:g} mm",
                    f"砂轮直径：{case.wheel_diameter_m * 1.0e3:g} mm",
                    f"磨削深度：{case.depth_of_cut_m * 1.0e6:g} μm",
                )
                latest = self.project.latest_geometry_result
                outputs = (
                    "二维参数化几何已生成。",
                    *(f"预览图：{latest.artifact_paths['geometry_preview']}" for _ in (0,) if latest),
                )
        elif name == "网格":
            actions = ("生成网格",)
            if self.project.geometry is None:
                outputs = ("请先完成二维几何建模。",)
            elif self.project.mesh is None:
                inputs = (
                    "网格类型：三角形",
                    "目标尺寸：3 mm",
                    "最小尺寸：1 mm",
                    "最大尺寸：5 mm",
                )
                outputs = ("网格尚未生成。",)
            else:
                mesh_case = WorkbenchMeshCase.from_mapping(
                    self.project.mesh, allow_deferred=True
                )
                inputs = (
                    f"网格类型：{'三角形' if mesh_case.mesh_type == 'triangle' else '四边形'}",
                    f"目标尺寸：{mesh_case.target_size_m * 1.0e3:g} mm",
                    f"最小尺寸：{mesh_case.minimum_size_m * 1.0e3:g} mm",
                    f"最大尺寸：{mesh_case.maximum_size_m * 1.0e3:g} mm",
                )
                latest_mesh = self.project.latest_mesh_result
                if mesh_case.status == "stale":
                    outputs = ("几何已更新，请重新生成网格。",)
                elif latest_mesh is None:
                    outputs = ("网格记录存在，但真实产物不可用。",)
                else:
                    outputs = (
                        "三角形网格已生成。",
                        f"网格图：{latest_mesh.artifact_paths['mesh_png']}",
                    )
        elif name == "材料":
            actions = ("保存材料参数",)
            if self.project.material is None:
                outputs = ("尚未配置材料参数。",)
            else:
                summary = WorkbenchMaterialCase.from_mapping(
                    self.project.material
                ).material_summary()
                inputs = tuple(f"{key}：{value}" for key, value in summary.items())
                outputs = ("材料参数已保存为 SI 单位。",)
        elif name == "工艺":
            actions = ("解析粒度", "保存工艺参数")
            geometry = self.project.geometry
            geometry_inputs: tuple[str, ...] = ()
            if isinstance(geometry, dict) and geometry.get("source") == "parametric_2d":
                geometry_case = ParametricGeometryCase.from_mapping(geometry)
                geometry_inputs = (
                    f"砂轮直径：{geometry_case.wheel_diameter_m * 1.0e3:g} mm（来自几何，只读）",
                    f"磨削深度：{geometry_case.depth_of_cut_m * 1.0e6:g} μm（来自几何，只读）",
                )
            if self.project.wheel is None or self.project.process is None:
                inputs = geometry_inputs
                outputs = ("尚未完成砂轮粒度解析和工艺参数保存。",)
            else:
                wheel_summary = WorkbenchWheelCase.from_mapping(
                    self.project.wheel
                ).wheel_summary()
                process_summary = WorkbenchProcessCase.from_mapping(
                    self.project.process
                ).process_summary()
                inputs = (
                    *geometry_inputs,
                    wheel_summary["标题"],
                    *(f"{key}：{value}" for key, value in process_summary.items()),
                )
                outputs = (
                    f"粒度状态：{wheel_summary['数据状态']}",
                    "砂轮、粒度解析和工艺参数已保存为 SI 单位。",
                )
        elif name == "分析":
            actions = ("运行分析", "打开高级兼容计算")
            from .analysis import (
                ANALYSIS_MODE_DEFINITIONS,
                AnalysisProjectData,
            )
            from .analysis_workspace import analysis_input_summary

            saved = (
                AnalysisProjectData.from_mapping(self.project.analysis)
                if self.project.analysis is not None else None
            )
            mode = saved.mode if saved is not None else next(iter(ANALYSIS_MODE_DEFINITIONS))
            inputs = analysis_input_summary(
                mode,
                input_valid=self.project.node_status("分析") in (
                    ProjectNodeStatus.READY,
                    ProjectNodeStatus.RUNNING,
                    ProjectNodeStatus.COMPLETED,
                ),
            )
            status_labels = {
                ProjectNodeStatus.NOT_CONFIGURED: "尚未计算",
                ProjectNodeStatus.READY: "已配置，等待运行",
                ProjectNodeStatus.RUNNING: "正在计算",
                ProjectNodeStatus.COMPLETED: "已完成",
                ProjectNodeStatus.FAILED: "计算失败",
                ProjectNodeStatus.STALE: "结果已过期",
                ProjectNodeStatus.DISABLED: "尚未计算",
            }
            outputs = (
                f"计算状态：{status_labels[self.project.node_status('分析')]}",
                f"分析模式：{ANALYSIS_MODE_DEFINITIONS[mode].display_name}",
            )
        elif name == "结果":
            latest = self.project.latest_analysis_result
            if latest is None:
                outputs = ("暂无结果",)
            else:
                outputs = (
                    f"结果目录：{latest.output_directory}",
                    f"结果摘要：{latest.summary_path}",
                    f"分析类型：{latest.analysis_type}",
                )
        return WorkbenchPage(
            title=name,
            status=status,
            description=_PAGE_TEXT[name],
            actions=actions,
            input_summary=inputs,
            output_summary=outputs,
            limitations=limitations,
        )

    def register_published_result(
        self,
        *,
        mode: str,
        result_format: str,
        output_directory: Path,
        summary_path: Path,
        artifact_paths: dict[str, Path],
        created_at: str,
        input_fingerprint: str = "",
    ) -> None:
        result_id = f"{mode}-{created_at}-{uuid4().hex[:8]}"
        self.project.register_result(
            ResultRecord(
                result_id=result_id,
                analysis_type=mode,
                result_format=result_format,
                output_directory=str(output_directory),
                summary_path=str(summary_path),
                artifact_paths={key: str(value) for key, value in artifact_paths.items()},
                created_at=created_at,
                input_fingerprint=input_fingerprint,
            )
        )
        if self.project.analysis is not None:
            from .analysis import AnalysisProjectData

            previous = AnalysisProjectData.from_mapping(self.project.analysis)
            self.project.analysis = AnalysisProjectData(
                mode=previous.mode, display_name=previous.display_name,
                status="completed", output_directory=str(output_directory),
                settings=previous.settings, input_fingerprint=input_fingerprint or previous.input_fingerprint,
                input_snapshot=previous.input_snapshot,
                last_run={
                    "status": "completed", "result_id": result_id,
                    "started_at": previous.last_run.get("started_at"),
                    "finished_at": created_at, "error_message": None,
                }, extra_fields=previous.extra_fields,
            ).to_dict()

    def register_geometry_result(
        self,
        provider: Parametric2DGeometryProvider,
        published: GeometryPublishedResult,
    ) -> None:
        model = provider.build_geometry()
        self.project.geometry = provider.to_geometry_schema(model)
        self.project.register_geometry_result(
            ResultRecord(
                result_id=f"geometry-{datetime.now().astimezone().isoformat(timespec='seconds')}",
                analysis_type="geometry_preview",
                result_format=published.result_format,
                output_directory=str(published.output_directory),
                summary_path=str(published.summary_path),
                artifact_paths={"geometry_preview": str(published.preview_path)},
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        )

    def register_mesh_result(
        self,
        provider: TriangleMeshProvider,
        published: MeshPublishedResult,
    ) -> None:
        mesh = WorkbenchMeshCase(
            mesh_type=provider.settings.mesh_type,
            target_size_m=provider.settings.target_size_m,
            minimum_size_m=provider.settings.minimum_size_m,
            maximum_size_m=provider.settings.maximum_size_m,
            status="completed",
            artifacts={
                "mesh_msh": str(published.mesh_path),
                "mesh_png": str(published.preview_path),
                "mesh_summary_json": str(published.summary_path),
            },
            extra_fields=provider.settings.extra_fields,
        ).to_dict()
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.project.register_mesh_result(
            ResultRecord(
                result_id=f"mesh-{created_at}",
                analysis_type="mesh_preview",
                result_format=published.result_format,
                output_directory=str(published.output_directory),
                summary_path=str(published.summary_path),
                artifact_paths={
                    "mesh_msh": str(published.mesh_path),
                    "mesh_png": str(published.preview_path),
                },
                created_at=created_at,
            ),
            mesh,
        )

    def register_material(self, material: WorkbenchMaterialCase) -> None:
        self.project.register_material(material.to_material_schema())

    def register_process(
        self, wheel: WorkbenchWheelCase, process: WorkbenchProcessCase
    ) -> None:
        self.project.register_process(
            wheel.to_wheel_schema(), process.to_process_schema()
        )

    def mark_geometry_failed(self, reason: str) -> None:
        self.project.set_node_status("几何", ProjectNodeStatus.FAILED, reason)

    def mark_mesh_failed(self, reason: str) -> None:
        self.project.set_node_status("网格", ProjectNodeStatus.FAILED, reason)

    def mark_analysis_failed(self, reason: str, *, finished_at: str | None = None) -> None:
        if self.project.analysis is not None:
            from .analysis import AnalysisProjectData

            previous = AnalysisProjectData.from_mapping(self.project.analysis)
            self.project.analysis = AnalysisProjectData(
                mode=previous.mode, display_name=previous.display_name,
                status="failed", output_directory=previous.output_directory,
                settings=previous.settings, input_fingerprint=previous.input_fingerprint,
                input_snapshot=previous.input_snapshot,
                last_run={
                    "status": "failed", "result_id": None,
                    "started_at": previous.last_run.get("started_at"),
                    "finished_at": finished_at or datetime.now().astimezone().isoformat(timespec="seconds"),
                    "error_message": reason,
                }, extra_fields=previous.extra_fields,
            ).to_dict()
        self.project.mark_analysis_failed(reason)
