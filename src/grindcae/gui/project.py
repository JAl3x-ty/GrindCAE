"""Persistent project model for the GrindCAE engineering workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4
import zipfile

from grindcae import __version__


PROJECT_SCHEMA_VERSION = 2
LEGACY_PROJECT_SCHEMA_VERSION = 1
PROJECT_TYPE = "grindcae_project"
PROJECT_CONTAINER_TYPE = "grindcae_project_container"
PROJECT_CONTAINER_SCHEMA_VERSION = 1
PROJECT_NODE_ORDER = ("工程", "几何", "网格", "材料", "工艺", "分析", "结果")
_REQUIRED_FIELDS = (
    "project_schema_version",
    "project_type",
    "project_name",
    "description",
    "workflow",
    "geometry",
    "analysis",
    "results",
    "extensions",
)
_KNOWN_FIELDS = (*_REQUIRED_FIELDS, "project_id", "mesh", "material", "wheel", "process")
_DEPENDENCIES = {
    "网格": ("几何",),
    "材料": ("网格",),
    "工艺": ("几何", "材料"),
    "分析": ("几何", "网格", "材料", "工艺"),
}
PREVIEW_RESULT_TYPES = frozenset(
    {"geometry_preview", "mesh_preview", "process_grain_preview"}
)
FORMAL_ANALYSIS_TYPES = frozenset({
    "literature_elastoplastic_single_pass",
    "literature_grinding_force",
    "linear_elastic_single_position",
    "linear_elastic_single_pass",
    "elastoplastic_single_position",
    "elastoplastic_single_pass",
    "mechanism_elastoplastic_single_pass",
    "single_grain_high_fidelity",
})


class ProjectError(ValueError):
    """A project-file or workbench-state error suitable for Chinese UI display."""


class ProjectNodeStatus(str, Enum):
    NOT_CONFIGURED = "未配置"
    READY = "可执行"
    RUNNING = "计算中"
    COMPLETED = "已完成"
    FAILED = "失败"
    STALE = "需要更新"
    DISABLED = "暂未启用"


@dataclass(frozen=True)
class ResultRecord:
    result_id: str
    analysis_type: str
    result_format: str
    output_directory: str
    summary_path: str
    artifact_paths: dict[str, str]
    created_at: str
    status: str = "completed"
    input_fingerprint: str = ""
    location_kind: str = "external_absolute"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResultRecord":
        optional = {"input_fingerprint", "location_kind"}
        required = tuple(name for name in cls.__dataclass_fields__ if name not in optional)
        missing = [name for name in required if name not in data]
        if missing:
            raise ProjectError(f"结果记录缺少必需字段：{', '.join(missing)}")
        artifact_paths = data["artifact_paths"]
        if not isinstance(artifact_paths, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in artifact_paths.items()
        ):
            raise ProjectError("结果记录 artifact_paths 必须是字符串路径对象。")
        values = {name: data[name] for name in required if name != "artifact_paths"}
        if not all(isinstance(value, str) for value in values.values()):
            raise ProjectError("结果记录字段必须使用字符串。")
        fingerprint = data.get("input_fingerprint", "")
        if not isinstance(fingerprint, str):
            raise ProjectError("结果记录 input_fingerprint 必须是字符串。")
        location_kind = data.get("location_kind", "external_absolute")
        if location_kind not in {"project_relative", "external_absolute", "recovered_external"}:
            raise ProjectError("结果记录 location_kind 无效。")
        return cls(
            artifact_paths=dict(artifact_paths),
            input_fingerprint=fingerprint,
            location_kind=location_kind,
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "analysis_type": self.analysis_type,
            "result_format": self.result_format,
            "output_directory": self.output_directory,
            "summary_path": self.summary_path,
            "artifact_paths": dict(self.artifact_paths),
            "created_at": self.created_at,
            "status": self.status,
            "input_fingerprint": self.input_fingerprint,
            "location_kind": self.location_kind,
        }


@dataclass
class ProjectNode:
    status: ProjectNodeStatus
    reason: str = ""

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "ProjectNode":
        try:
            status = ProjectNodeStatus(data["status"])
        except KeyError as exc:
            raise ProjectError(f"工作节点“{name}”缺少必需字段：status") from exc
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"工作节点“{name}”包含未知状态。") from exc
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            raise ProjectError(f"工作节点“{name}”的原因说明必须是文字。")
        return cls(status, reason)

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status.value, "reason": self.reason}


def _default_nodes() -> dict[str, ProjectNode]:
    return {
        "工程": ProjectNode(ProjectNodeStatus.READY, "可编辑工程名称和备注。"),
        "几何": ProjectNode(
            ProjectNodeStatus.NOT_CONFIGURED,
            "尚未配置二维参数化几何。",
        ),
        "网格": ProjectNode(
            ProjectNodeStatus.DISABLED,
            "需要先完成二维几何；完成后可以生成三角形网格。",
        ),
        "材料": ProjectNode(
            ProjectNodeStatus.DISABLED,
            "需要先完成网格；完成后可以配置材料参数。",
        ),
        "工艺": ProjectNode(
            ProjectNodeStatus.NOT_CONFIGURED,
            "可以查看几何引用；完成材料后可保存砂轮与磨削工艺参数。",
        ),
        "分析": ProjectNode(
            ProjectNodeStatus.NOT_CONFIGURED,
            "完成几何、网格、材料和工艺后可以选择正式分析任务。",
        ),
        "结果": ProjectNode(ProjectNodeStatus.NOT_CONFIGURED, "暂无结果。"),
    }


@dataclass
class GrindCaeProject:
    project_name: str
    project_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    nodes: dict[str, ProjectNode] = field(default_factory=_default_nodes)
    geometry: Any = None
    mesh: Any = None
    material: Any = None
    wheel: Any = None
    process: Any = None
    analysis: Any = None
    results: list[ResultRecord] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)
    workflow_extra: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def node_status(self, name: str) -> ProjectNodeStatus:
        return self._node(name).status

    def node_reason(self, name: str) -> str:
        return self._node(name).reason

    def _node(self, name: str) -> ProjectNode:
        try:
            return self.nodes[name]
        except KeyError as exc:
            raise ProjectError(f"未知工作节点：{name}") from exc

    def can_execute(self, name: str) -> tuple[bool, str]:
        node = self._node(name)
        incomplete = [
            dependency
            for dependency in _DEPENDENCIES.get(name, ())
            if self.node_status(dependency) is not ProjectNodeStatus.COMPLETED
        ]
        if incomplete:
            return False, f"需要先完成：{', '.join(incomplete)}。"
        if node.status is ProjectNodeStatus.DISABLED:
            return False, node.reason or f"“{name}”暂未启用。"
        if node.status is ProjectNodeStatus.RUNNING:
            return False, f"“{name}”正在计算。"
        return True, ""

    def set_node_status(
        self,
        name: str,
        status: ProjectNodeStatus,
        reason: str = "",
    ) -> None:
        node = self._node(name)
        if node.status is ProjectNodeStatus.DISABLED and status is ProjectNodeStatus.COMPLETED:
            raise ProjectError(f"“{name}”暂未启用，不能标记为已完成。")
        self.nodes[name] = ProjectNode(status, reason)
        if name in ("几何", "网格", "材料", "工艺") and all(
            self.node_status(dependency) is ProjectNodeStatus.COMPLETED
            for dependency in _DEPENDENCIES["分析"]
        ) and self.node_status("分析") not in (
            ProjectNodeStatus.RUNNING,
            ProjectNodeStatus.COMPLETED,
        ):
            self.nodes["分析"] = ProjectNode(
                ProjectNodeStatus.READY,
                "工程参数完整，可以选择并运行分析任务。",
            )

    @property
    def latest_result(self) -> ResultRecord | None:
        return self.results[-1] if self.results else None

    @property
    def latest_geometry_result(self) -> ResultRecord | None:
        return next(
            (
                record
                for record in reversed(self.results)
                if record.analysis_type == "geometry_preview"
            ),
            None,
        )

    @property
    def latest_analysis_result(self) -> ResultRecord | None:
        return next(
            (
                record
                for record in reversed(self.results)
                if record.analysis_type in FORMAL_ANALYSIS_TYPES
                and record.status == "completed"
            ),
            None,
        )

    @property
    def latest_mesh_result(self) -> ResultRecord | None:
        return next(
            (
                record
                for record in reversed(self.results)
                if record.analysis_type == "mesh_preview"
            ),
            None,
        )

    def current_analysis_result(
        self, mode: str, input_fingerprint: str
    ) -> ResultRecord | None:
        if (
            not isinstance(mode, str)
            or not mode
            or not isinstance(input_fingerprint, str)
            or not input_fingerprint
        ):
            return None
        return next(
            (
                record
                for record in reversed(self.results)
                if record.analysis_type == mode
                and record.analysis_type in FORMAL_ANALYSIS_TYPES
                and record.status == "completed"
                and record.input_fingerprint == input_fingerprint
            ),
            None,
        )

    def current_process_grain_preview(
        self, input_fingerprint: str
    ) -> ResultRecord | None:
        if not isinstance(input_fingerprint, str) or not input_fingerprint:
            return None
        return next(
            (
                record
                for record in reversed(self.results)
                if record.analysis_type == "process_grain_preview"
                and record.input_fingerprint == input_fingerprint
            ),
            None,
        )

    def _validated_result(self, record: ResultRecord) -> ResultRecord:
        output = Path(record.output_directory).expanduser().resolve()
        summary = Path(record.summary_path).expanduser().resolve()
        if record.status != "completed" or not output.is_dir() or not summary.is_file():
            raise ProjectError("结果注册表只能记录真实计算输出和已存在的结果摘要。")
        artifacts: dict[str, str] = {}
        for name, raw_path in record.artifact_paths.items():
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise ProjectError(f"结果文件不存在，不能注册真实计算输出：{name}")
            artifacts[name] = str(path)
        return ResultRecord(
            result_id=record.result_id,
            analysis_type=record.analysis_type,
            result_format=record.result_format,
            output_directory=str(output),
            summary_path=str(summary),
            artifact_paths=artifacts,
            created_at=record.created_at,
            status=record.status,
            input_fingerprint=record.input_fingerprint,
        )

    def register_result(self, record: ResultRecord) -> None:
        if record.analysis_type in PREVIEW_RESULT_TYPES:
            raise ProjectError("预处理预览必须使用对应的独立登记接口。")
        normalized = self._validated_result(record)
        self.results.append(normalized)
        self.set_node_status("分析", ProjectNodeStatus.COMPLETED, "最近一次计算已完成。")
        self.set_node_status("结果", ProjectNodeStatus.COMPLETED, "已登记真实计算结果。")

    def register_process_grain_preview(self, record: ResultRecord) -> None:
        if record.analysis_type != "process_grain_preview":
            raise ProjectError("工艺磨粒预览必须使用 process_grain_preview 类型。")
        normalized = self._validated_result(record)
        self.results = [
            existing
            for existing in self.results
            if existing.analysis_type != "process_grain_preview"
        ]
        self.results.append(normalized)

    def _mark_analysis_inputs_stale(self, reason: str) -> None:
        if self.latest_analysis_result is None and self.analysis is None:
            return
        self.set_node_status("分析", ProjectNodeStatus.STALE, reason)
        self.set_node_status(
            "结果",
            ProjectNodeStatus.STALE,
            "已有分析结果保留用于追溯，当前工程参数已变化，不能继续视为有效结果。",
        )

    def register_material(self, material: Mapping[str, Any]) -> None:
        if self.node_status("网格") is not ProjectNodeStatus.COMPLETED:
            raise ProjectError("需要先完成有效三角形网格，才能保存材料参数。")
        if self.node_status("材料") is ProjectNodeStatus.DISABLED:
            self.set_node_status("材料", ProjectNodeStatus.READY, "网格已完成，可以配置材料参数。")
        try:
            from .material_workspace import WorkbenchMaterialCase

            validated = WorkbenchMaterialCase.from_mapping(material).to_dict()
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        changed = self.material != validated
        self.material = validated
        self.set_node_status("材料", ProjectNodeStatus.COMPLETED, "材料参数有效。")
        if self.process is None:
            self.set_node_status("工艺", ProjectNodeStatus.READY, "材料已完成，可以配置砂轮与磨削工艺。")
        if changed:
            self._mark_analysis_inputs_stale("材料参数已更新，请重新执行分析。")

    def register_process(
        self, wheel: Mapping[str, Any], process: Mapping[str, Any]
    ) -> None:
        if self.geometry is None:
            raise ProjectError("需要先完成二维几何，才能保存工艺参数。")
        if self.node_status("材料") is not ProjectNodeStatus.COMPLETED:
            raise ProjectError("需要先完成材料参数，才能保存工艺参数。")
        try:
            from .process_workspace import WorkbenchProcessCase, WorkbenchWheelCase

            validated_wheel = WorkbenchWheelCase.from_mapping(wheel).to_dict()
            validated_process = WorkbenchProcessCase.from_mapping(process).to_dict()
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        changed = self.wheel != validated_wheel or self.process != validated_process
        self.wheel = validated_wheel
        self.process = validated_process
        self.set_node_status("工艺", ProjectNodeStatus.COMPLETED, "砂轮、粒度和磨削工艺参数有效。")
        if self.node_status("几何") is ProjectNodeStatus.COMPLETED and self.node_status("网格") is ProjectNodeStatus.COMPLETED and self.node_status("材料") is ProjectNodeStatus.COMPLETED:
            self.set_node_status("分析", ProjectNodeStatus.READY, "工程参数完整，可以选择并运行分析任务。")
        if changed:
            self._mark_analysis_inputs_stale("砂轮或磨削工艺参数已更新，请重新执行分析。")

    def register_geometry_result(self, record: ResultRecord) -> None:
        if record.analysis_type != "geometry_preview":
            raise ProjectError("几何结果必须使用 geometry_preview 类型。")
        normalized = self._validated_result(record)
        self.results = [
            existing
            for existing in self.results
            if existing.analysis_type not in ("geometry_preview", "mesh_preview")
        ]
        self.results.append(normalized)
        if isinstance(self.mesh, dict):
            self.mesh = dict(self.mesh)
            self.mesh["status"] = "stale"
        self.set_node_status("几何", ProjectNodeStatus.COMPLETED, "二维参数化几何有效。")
        self.set_node_status(
            "网格",
            ProjectNodeStatus.STALE if self.mesh is not None else ProjectNodeStatus.READY,
            "二维几何已更新，请重新生成网格。"
            if self.mesh is not None
            else "二维几何已完成，可以生成三角形网格。",
        )
        if self.material is not None:
            self.set_node_status("材料", ProjectNodeStatus.STALE, "几何已更新并使网格过期，请重新确认材料工作流。")
        if self.process is not None:
            self.set_node_status("工艺", ProjectNodeStatus.STALE, "几何引用已变化，请重新确认工艺参数。")
        self._mark_analysis_inputs_stale("几何参数已更新，请重新执行分析。")
        if self.node_status("结果") is not ProjectNodeStatus.STALE:
            self.set_node_status("结果", ProjectNodeStatus.COMPLETED, "已登记二维几何预览。")

    def register_mesh_result(self, record: ResultRecord, mesh: Mapping[str, Any]) -> None:
        if record.analysis_type != "mesh_preview":
            raise ProjectError("网格结果必须使用 mesh_preview 类型。")
        normalized = self._validated_result(record)
        self.results = [
            existing
            for existing in self.results
            if existing.analysis_type != "mesh_preview"
        ]
        self.results.append(normalized)
        self.mesh = dict(mesh)
        self.set_node_status("网格", ProjectNodeStatus.COMPLETED, "三角形网格有效。")
        self.set_node_status(
            "材料",
            ProjectNodeStatus.COMPLETED if self.material is not None else ProjectNodeStatus.READY,
            "材料参数有效。" if self.material is not None else "网格已完成，可以配置材料参数。",
        )
        self.set_node_status("结果", ProjectNodeStatus.COMPLETED, "已登记真实网格结果。")

    def mark_analysis_failed(self, reason: str) -> None:
        self.set_node_status("分析", ProjectNodeStatus.FAILED, reason)
        if self.latest_analysis_result is not None:
            self.set_node_status("结果", ProjectNodeStatus.FAILED, "当前分析失败；历史结果仍保留用于追溯。")
        elif self.results:
            self.set_node_status("结果", ProjectNodeStatus.FAILED, "当前分析失败；仅保留几何或网格预览。")
        else:
            self.set_node_status("结果", ProjectNodeStatus.FAILED, "当前分析失败，尚无有效结果。")

    def to_dict(self) -> dict[str, Any]:
        workflow = dict(self.workflow_extra)
        workflow["nodes"] = {
            name: self.nodes[name].to_dict() for name in PROJECT_NODE_ORDER
        }
        data: dict[str, Any] = dict(self.extra_fields)
        data.update(
            {
                "project_schema_version": PROJECT_SCHEMA_VERSION,
                "project_type": PROJECT_TYPE,
                "project_id": self.project_id,
                "project_name": self.project_name,
                "description": self.description,
                "workflow": workflow,
                "geometry": self.geometry,
                "mesh": self.mesh,
                "material": self.material,
                "wheel": self.wheel,
                "process": self.process,
                "analysis": self.analysis,
                "results": {"records": [record.to_dict() for record in self.results]},
                "extensions": dict(self.extensions),
            }
        )
        return data


def _default_extensions() -> dict[str, Any]:
    return {
        "cad": None,
        "quadrilateral_mesh": None,
        "thermal": None,
        "single_grain": None,
        "multi_pass": None,
        "calibration": None,
        "roughness": None,
    }


def create_project(project_name: str = "未命名工程", description: str = "") -> GrindCaeProject:
    if not isinstance(project_name, str) or not project_name.strip():
        raise ProjectError("工程名称不能为空。")
    if not isinstance(description, str):
        raise ProjectError("工程备注必须是文字。")
    return GrindCaeProject(
        project_name=project_name.strip(),
        description=description,
        extensions=_default_extensions(),
    )


def project_from_mapping(data: Mapping[str, Any]) -> GrindCaeProject:
    if not isinstance(data, Mapping):
        raise ProjectError("工程文件根节点必须是 JSON 对象。")
    if "project_schema_version" in data:
        version = data["project_schema_version"]
        if isinstance(version, bool) or version not in {
            LEGACY_PROJECT_SCHEMA_VERSION,
            PROJECT_SCHEMA_VERSION,
        }:
            raise ProjectError(
                f"不支持的工程文件版本：{version}。当前支持版本 1 和 2。"
            )
    missing = [name for name in _REQUIRED_FIELDS if name not in data]
    if missing:
        raise ProjectError(f"工程文件缺少必需字段：{', '.join(missing)}")
    if data["project_type"] != PROJECT_TYPE:
        raise ProjectError("工程文件 project_type 必须是 grindcae_project。")
    project_name = data["project_name"]
    description = data["description"]
    if not isinstance(project_name, str) or not project_name.strip():
        raise ProjectError("工程名称不能为空。")
    if not isinstance(description, str):
        raise ProjectError("工程备注必须是文字。")
    workflow = data["workflow"]
    if not isinstance(workflow, Mapping):
        raise ProjectError("工程 workflow 必须是 JSON 对象。")
    raw_nodes = workflow.get("nodes")
    if not isinstance(raw_nodes, Mapping):
        raise ProjectError("工程 workflow 缺少必需字段：nodes")
    nodes: dict[str, ProjectNode] = {}
    for name in PROJECT_NODE_ORDER:
        raw_node = raw_nodes.get(name)
        if not isinstance(raw_node, Mapping):
            raise ProjectError(f"工程 workflow 缺少工作节点：{name}")
        nodes[name] = ProjectNode.from_mapping(name, raw_node)
    raw_results = data["results"]
    if not isinstance(raw_results, Mapping):
        raise ProjectError("工程 results 必须是 JSON 对象。")
    raw_records = raw_results.get("records", [])
    if not isinstance(raw_records, list):
        raise ProjectError("工程结果 records 必须是列表。")
    records = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ProjectError("工程结果记录必须是 JSON 对象。")
        records.append(ResultRecord.from_mapping(raw_record))
    extensions = data["extensions"]
    if not isinstance(extensions, Mapping):
        raise ProjectError("工程 extensions 必须是 JSON 对象。")
    geometry = data["geometry"]
    if geometry is not None:
        if not isinstance(geometry, Mapping):
            raise ProjectError("工程 geometry 必须是 JSON 对象或 null。")
        source = geometry.get("source")
        if source == "parametric_2d":
            try:
                from .geometry import ParametricGeometryCase

                ParametricGeometryCase.from_mapping(geometry)
            except ValueError as exc:
                raise ProjectError(str(exc)) from exc
        elif source != "step_cad":
            raise ProjectError(f"未知几何来源：{source}")
    mesh = data.get("mesh")
    if mesh is not None:
        try:
            from .mesh_workspace import WorkbenchMeshCase

            WorkbenchMeshCase.from_mapping(mesh, allow_deferred=True)
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
    known = set(_KNOWN_FIELDS)
    project = GrindCaeProject(
        project_name=project_name,
        project_id=str(data.get("project_id") or uuid4()),
        description=description,
        nodes=nodes,
        geometry=geometry,
        mesh=mesh,
        material=data.get("material"),
        wheel=data.get("wheel"),
        process=data.get("process"),
        analysis=data["analysis"],
        results=records,
        extensions=dict(extensions),
        workflow_extra={key: value for key, value in workflow.items() if key != "nodes"},
        extra_fields={key: value for key, value in data.items() if key not in known},
    )
    if project.material is not None:
        try:
            from .material_workspace import WorkbenchMaterialCase

            project.material = WorkbenchMaterialCase.from_mapping(project.material).to_dict()
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
    if project.wheel is not None:
        try:
            from .process_workspace import WorkbenchWheelCase

            project.wheel = WorkbenchWheelCase.from_mapping(project.wheel).to_dict()
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
    if project.process is not None:
        try:
            from .process_workspace import WorkbenchProcessCase, geometry_fingerprint

            project.process = WorkbenchProcessCase.from_mapping(project.process).to_dict()
            if project.geometry is not None and project.process["geometry_reference"]["geometry_fingerprint"] != geometry_fingerprint(project.geometry):
                project.set_node_status("工艺", ProjectNodeStatus.STALE, "几何引用已变化，请重新确认工艺参数。")
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
    if project.material is None:
        material_status = (
            ProjectNodeStatus.READY
            if project.node_status("网格") is ProjectNodeStatus.COMPLETED
            else ProjectNodeStatus.DISABLED
        )
        project.set_node_status(
            "材料",
            material_status,
            "网格已完成，可以配置材料参数。"
            if material_status is ProjectNodeStatus.READY
            else "需要先完成网格；完成后可以配置材料参数。",
        )
    if project.process is None:
        project.set_node_status(
            "工艺",
            ProjectNodeStatus.READY if project.material is not None else ProjectNodeStatus.NOT_CONFIGURED,
            "材料已完成，可以配置砂轮与磨削工艺。"
            if project.material is not None
            else "可以查看几何引用；完成材料后可保存砂轮与磨削工艺参数。",
        )
    invalid_mesh_result = False
    retained_records = []
    for record in project.results:
        output_exists = Path(record.output_directory).expanduser().resolve().is_dir()
        summary_exists = Path(record.summary_path).expanduser().resolve().is_file()
        artifacts_exist = all(
            Path(path).expanduser().resolve().is_file()
            for path in record.artifact_paths.values()
        )
        if output_exists and summary_exists and artifacts_exist:
            retained_records.append(record)
        elif record.analysis_type in FORMAL_ANALYSIS_TYPES:
            # 正式分析记录即使文件已移动或损坏也要保留，结果中心负责显示“文件异常”。
            retained_records.append(record)
        elif record.analysis_type == "mesh_preview":
            invalid_mesh_result = True
    if len(retained_records) != len(project.results):
        project.results = retained_records
        if invalid_mesh_result and isinstance(project.mesh, dict):
            project.mesh = dict(project.mesh)
            project.mesh["status"] = "stale"
            project.set_node_status(
                "网格",
                ProjectNodeStatus.STALE,
                "原登记网格产物已不存在，请重新生成网格。",
            )
        if not project.results:
            project.set_node_status(
                "结果",
                ProjectNodeStatus.NOT_CONFIGURED,
                "暂无结果。原登记结果已不存在。",
            )
    return project


def _reject_nonfinite(value: Any, path: str = "工程文件") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProjectError(f"{path} 不能包含 NaN 或 Infinity。")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def save_project(project: GrindCaeProject, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.name.lower().endswith(".gcase.json"):
        data = project.to_dict()
        _reject_nonfinite(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise ProjectError(f"工程保存失败：{exc}") from exc
        return target
    if target.suffix.lower() != ".gcae":
        raise ProjectError("工程文件必须使用 .gcae 扩展名。")
    data = _project_mapping_for_container(project, target)
    _reject_nonfinite(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        project_bytes = (
            json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        manifest = {
            "container_type": PROJECT_CONTAINER_TYPE,
            "container_schema_version": PROJECT_CONTAINER_SCHEMA_VERSION,
            "project_schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": project.project_id,
            "application_version": __version__,
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "project_sha256": hashlib.sha256(project_bytes).hexdigest(),
            "members": ["manifest.json", "project.json"],
        }
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.writestr("project.json", project_bytes)
            archive.writestr(
                "manifest.json",
                (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            )
        _load_gcae_mapping(temporary)
        if target.is_file():
            _load_gcae_mapping(target)
            shutil.copy2(target, target.with_name(f"{target.name}.bak"))
        temporary.replace(target)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        temporary.unlink(missing_ok=True)
        raise ProjectError(f"工程保存失败：{exc}") from exc
    return target


def load_project(path: str | Path) -> GrindCaeProject:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".gcae" or source.name.lower().endswith(".gcae.bak"):
        data = _load_gcae_mapping(source)
        project = project_from_mapping(data)
        project.results = [
            _resolve_record_from_container(record, source.parent)
            for record in project.results
        ]
        return _validate_loaded_result_paths(project)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ProjectError("工程文件不是有效的 UTF-8 文本。") from exc
    except json.JSONDecodeError as exc:
        raise ProjectError(f"工程文件 JSON 格式错误：第 {exc.lineno} 行。") from exc
    except OSError as exc:
        raise ProjectError(f"工程文件读取失败：{exc}") from exc
    _reject_nonfinite(data)
    return project_from_mapping(data)


def _relative_or_absolute(path: str, base: Path) -> tuple[str, bool]:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(base).as_posix(), True
    except ValueError:
        return str(resolved), False


def _project_mapping_for_container(
    project: GrindCaeProject, target: Path
) -> dict[str, Any]:
    data = project.to_dict()
    records = data["results"]["records"]
    for record in records:
        output, relative = _relative_or_absolute(record["output_directory"], target.parent)
        if not relative:
            record["location_kind"] = "external_absolute"
            continue
        record["location_kind"] = "project_relative"
        record["output_directory"] = output
        record["summary_path"] = _relative_or_absolute(record["summary_path"], target.parent)[0]
        record["artifact_paths"] = {
            key: _relative_or_absolute(value, target.parent)[0]
            for key, value in record["artifact_paths"].items()
        }
    return data


def _load_gcae_mapping(source: Path) -> Mapping[str, Any]:
    try:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ProjectError("工程容器包含重复文件。")
            if set(names) != {"manifest.json", "project.json"}:
                raise ProjectError("工程容器文件清单无效。")
            manifest = json.loads(archive.read("manifest.json"))
            project_bytes = archive.read("project.json")
    except ProjectError:
        raise
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ProjectError(f"工程容器读取失败：{exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ProjectError("工程容器 manifest 无效。")
    if manifest.get("container_type") != PROJECT_CONTAINER_TYPE:
        raise ProjectError("工程容器类型无效。")
    if manifest.get("container_schema_version") != PROJECT_CONTAINER_SCHEMA_VERSION:
        raise ProjectError("不支持的工程容器版本。")
    if hashlib.sha256(project_bytes).hexdigest() != manifest.get("project_sha256"):
        raise ProjectError("工程容器校验失败：project.json 已损坏或被修改。")
    try:
        data = json.loads(project_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"工程容器 project.json 无效：{exc}") from exc
    _reject_nonfinite(data)
    if not isinstance(data, Mapping) or data.get("project_id") != manifest.get("project_id"):
        raise ProjectError("工程容器校验失败：工程身份不一致。")
    return data


def _resolve_record_from_container(record: ResultRecord, base: Path) -> ResultRecord:
    if record.location_kind != "project_relative":
        return record
    def resolve(raw: str) -> str:
        candidate = (base / Path(raw)).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ProjectError("工程结果相对路径越界。") from exc
        return str(candidate)
    return ResultRecord(
        result_id=record.result_id,
        analysis_type=record.analysis_type,
        result_format=record.result_format,
        output_directory=resolve(record.output_directory),
        summary_path=resolve(record.summary_path),
        artifact_paths={key: resolve(value) for key, value in record.artifact_paths.items()},
        created_at=record.created_at,
        status=record.status,
        input_fingerprint=record.input_fingerprint,
        location_kind=record.location_kind,
    )


def _validate_loaded_result_paths(project: GrindCaeProject) -> GrindCaeProject:
    # Reuse the established missing-artifact policy after container paths are resolved.
    invalid_mesh_result = False
    retained: list[ResultRecord] = []
    for record in project.results:
        exists = (
            Path(record.output_directory).is_dir()
            and Path(record.summary_path).is_file()
            and all(Path(path).is_file() for path in record.artifact_paths.values())
        )
        if exists or record.analysis_type in FORMAL_ANALYSIS_TYPES:
            retained.append(record)
        elif record.analysis_type == "mesh_preview":
            invalid_mesh_result = True
    project.results = retained
    if invalid_mesh_result and isinstance(project.mesh, dict):
        project.mesh = dict(project.mesh)
        project.mesh["status"] = "stale"
    return project
