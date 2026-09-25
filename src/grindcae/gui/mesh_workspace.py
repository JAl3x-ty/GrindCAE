"""Triangle-mesh adapter used by the 2.6 engineering workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import gmsh
import meshio
import numpy as np
from skfem import MeshTri
from skfem.io import from_meshio

from grindcae.gmsh_runtime import initialize_gmsh
from grindcae.mesh_contract import DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM

from .geometry import ParametricGeometryModel


MESH_SCHEMA_VERSION = 1
TRIANGLE_MESH_TYPE = "triangle"
QUADRILATERAL_MESH_TYPE = "quadrilateral"
MESH_RESULT_FORMAT = "grindcae_triangle_mesh_workspace"
MESH_FILENAME = "mesh.msh"
MESH_PREVIEW_FILENAME = "mesh.png"
MESH_SUMMARY_FILENAME = "mesh_summary.json"
MESH_ARTIFACT_KEYS = ("mesh_msh", "mesh_png", "mesh_summary_json")
_PHYSICAL_GROUPS = (
    "domain",
    "fixed",
    "entry_transition",
    "processed",
    "exit_transition",
    "free_left",
    "free_right",
)


class MeshValidationError(ValueError):
    """Chinese validation error suitable for direct GUI display."""


@dataclass(frozen=True, slots=True)
class MeshPreviewPolicy:
    strategy: str
    show_nodes: bool
    local_target_columns: int = 40
    local_target_rows: int = 24


def select_mesh_preview_policy(
    *, node_count: int, triangle_count: int
) -> MeshPreviewPolicy:
    if node_count <= 10_000 and triangle_count <= 20_000:
        return MeshPreviewPolicy("complete_mesh", True)
    if node_count <= 50_000 and triangle_count <= 100_000:
        return MeshPreviewPolicy("complete_edges_without_nodes", False)
    return MeshPreviewPolicy("global_outline_with_local_true_mesh", False)


def _display_number(value: str, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise MeshValidationError(f"{label}：请输入有效数字。")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MeshValidationError(f"{label}：请输入有效数字。") from exc
    if not math.isfinite(number):
        raise MeshValidationError(f"{label}：请输入有效数字。")
    return number


def _finite_si(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeshValidationError(f"{label}必须是有限数字。")
    number = float(value)
    if not math.isfinite(number):
        raise MeshValidationError(f"{label}必须是有限数字。")
    return number


@dataclass(frozen=True)
class MeshDisplayInput:
    mesh_type: str = TRIANGLE_MESH_TYPE
    target_size_mm: str = "3"
    minimum_size_mm: str = "1"
    maximum_size_mm: str = "5"

    def build_case(self) -> "WorkbenchMeshCase":
        target = _display_number(self.target_size_mm, "目标网格尺寸")
        minimum = _display_number(self.minimum_size_mm, "最小网格尺寸")
        maximum = _display_number(self.maximum_size_mm, "最大网格尺寸")
        if target <= 0.0 or minimum <= 0.0 or maximum <= 0.0:
            raise MeshValidationError("网格尺寸必须大于 0。")
        if maximum <= minimum:
            raise MeshValidationError("最大网格尺寸必须大于最小网格尺寸。")
        if not minimum <= target <= maximum:
            raise MeshValidationError(
                "目标网格尺寸必须位于最小和最大网格尺寸之间。"
            )
        if self.mesh_type == QUADRILATERAL_MESH_TYPE:
            raise MeshValidationError("四边形网格将在后续版本实现。")
        if self.mesh_type != TRIANGLE_MESH_TYPE:
            raise MeshValidationError(f"未知网格类型：{self.mesh_type}")
        return WorkbenchMeshCase(
            mesh_type=self.mesh_type,
            target_size_m=target * 1.0e-3,
            minimum_size_m=minimum * 1.0e-3,
            maximum_size_m=maximum * 1.0e-3,
        )


@dataclass(frozen=True)
class WorkbenchMeshCase:
    mesh_type: str
    target_size_m: float
    minimum_size_m: float
    maximum_size_m: float
    status: str = "not_configured"
    artifacts: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    mesh_schema_version: int = MESH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.mesh_schema_version) is not int or self.mesh_schema_version != 1:
            raise MeshValidationError(
                f"不支持的网格文件版本：{self.mesh_schema_version}。当前仅支持版本 1。"
            )
        target = _finite_si(self.target_size_m, "目标网格尺寸")
        minimum = _finite_si(self.minimum_size_m, "最小网格尺寸")
        maximum = _finite_si(self.maximum_size_m, "最大网格尺寸")
        if target <= 0.0 or minimum <= 0.0 or maximum <= 0.0:
            raise MeshValidationError("网格尺寸必须大于 0。")
        if maximum <= minimum:
            raise MeshValidationError("最大网格尺寸必须大于最小网格尺寸。")
        if not minimum <= target <= maximum:
            raise MeshValidationError(
                "目标网格尺寸必须位于最小和最大网格尺寸之间。"
            )
        if self.mesh_type not in (TRIANGLE_MESH_TYPE, QUADRILATERAL_MESH_TYPE):
            raise MeshValidationError(f"未知网格类型：{self.mesh_type}")
        if not isinstance(self.status, str) or not self.status:
            raise MeshValidationError("网格状态必须是非空文字。")
        if not isinstance(self.artifacts, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.artifacts.items()
        ):
            raise MeshValidationError("网格 artifacts 必须是字符串路径对象。")

    @classmethod
    def from_mapping(
        cls, payload: object, *, allow_deferred: bool = False
    ) -> "WorkbenchMeshCase":
        if not isinstance(payload, Mapping):
            raise MeshValidationError("工程 mesh 必须是 JSON 对象。")
        required = (
            "mesh_schema_version",
            "mesh_type",
            "target_size_m",
            "minimum_size_m",
            "maximum_size_m",
            "status",
            "artifacts",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise MeshValidationError(f"工程 mesh 缺少必需字段：{', '.join(missing)}")
        known = set(required)
        case = cls(
            mesh_schema_version=payload["mesh_schema_version"],
            mesh_type=payload["mesh_type"],
            target_size_m=_finite_si(payload["target_size_m"], "目标网格尺寸"),
            minimum_size_m=_finite_si(payload["minimum_size_m"], "最小网格尺寸"),
            maximum_size_m=_finite_si(payload["maximum_size_m"], "最大网格尺寸"),
            status=payload["status"],
            artifacts=dict(payload["artifacts"]),
            extra_fields={key: value for key, value in payload.items() if key not in known},
        )
        if case.mesh_type == QUADRILATERAL_MESH_TYPE and not allow_deferred:
            raise MeshValidationError("四边形网格将在后续版本实现。")
        return case

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra_fields)
        data.update(
            {
                "mesh_schema_version": self.mesh_schema_version,
                "mesh_type": self.mesh_type,
                "target_size_m": self.target_size_m,
                "minimum_size_m": self.minimum_size_m,
                "maximum_size_m": self.maximum_size_m,
                "status": self.status,
                "artifacts": dict(self.artifacts),
            }
        )
        return data

    def to_display_input(self) -> MeshDisplayInput:
        return MeshDisplayInput(
            mesh_type=self.mesh_type,
            target_size_mm=f"{self.target_size_m * 1.0e3:.12g}",
            minimum_size_mm=f"{self.minimum_size_m * 1.0e3:.12g}",
            maximum_size_mm=f"{self.maximum_size_m * 1.0e3:.12g}",
        )


@dataclass(frozen=True)
class MeshPublishedResult:
    result_format: str
    output_directory: Path
    mesh_path: Path
    preview_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedMesh:
    mesh_data: meshio.Mesh
    node_count: int
    triangle_count: int
    boundary_element_count: int
    minimum_element_area_m2: float
    maximum_element_area_m2: float
    actual_average_size_m: float
    boundary_element_counts: dict[str, int]


def _region_for_segment(geometry: ParametricGeometryModel, x_mid: float) -> str:
    for name in ("entry_transition", "exit_transition"):
        region = geometry.regions[name]
        if float(region["x_start_m"]) <= x_mid <= float(region["x_end_m"]):
            return name
    return "processed"


def _mesh_cells(mesh_data: meshio.Mesh, cell_type: str) -> np.ndarray:
    blocks = [cell.data for cell in mesh_data.cells if cell.type == cell_type]
    if not blocks:
        return np.empty((0, 3 if cell_type == "triangle" else 2), dtype=np.int64)
    return np.vstack(blocks).astype(np.int64, copy=False)


def _line_physical_tags(mesh_data: meshio.Mesh) -> np.ndarray:
    values: list[np.ndarray] = []
    physical = mesh_data.cell_data.get("gmsh:physical", [])
    for cell, tags in zip(mesh_data.cells, physical):
        if cell.type == "line":
            values.append(np.asarray(tags, dtype=np.int64))
    return np.concatenate(values) if values else np.empty(0, dtype=np.int64)


class TriangleMeshProvider:
    """Generate and validate a geometry-only triangular Gmsh workspace mesh."""

    def __init__(
        self, geometry: ParametricGeometryModel, settings: WorkbenchMeshCase
    ) -> None:
        if not isinstance(geometry, ParametricGeometryModel):
            raise MeshValidationError("请先完成二维几何建模。")
        if not isinstance(settings, WorkbenchMeshCase):
            raise MeshValidationError("网格参数对象无效。")
        if settings.mesh_type != TRIANGLE_MESH_TYPE:
            raise MeshValidationError("四边形网格将在后续版本实现。")
        self.geometry = geometry
        self.settings = settings

    def validate_mesh_settings(self) -> tuple[bool, str]:
        return True, "三角形网格参数有效。"

    def generate_mesh(self, output_path: str | Path) -> Path:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._generate_msh(target)
        return target

    def validate_generated_mesh(self, mesh_path: str | Path) -> dict[str, Any]:
        validated = self._validate_msh(Path(mesh_path).expanduser().resolve())
        return self._mesh_statistics(validated)

    def mesh_artifacts(self, output_directory: str | Path) -> dict[str, str]:
        output = Path(output_directory).expanduser().resolve()
        return {
            "mesh_msh": str((output / MESH_FILENAME).resolve()),
            "mesh_png": str((output / MESH_PREVIEW_FILENAME).resolve()),
            "mesh_summary_json": str((output / MESH_SUMMARY_FILENAME).resolve()),
        }

    def _generate_msh(self, output_path: Path) -> None:
        if gmsh.isInitialized():
            raise MeshValidationError("Gmsh 已在其他任务中运行，请稍后重试。")
        profile = tuple(
            zip(self.geometry.top_profile_x_m, self.geometry.top_profile_y_m)
        )
        if len(profile) < 4:
            raise MeshValidationError("二维几何表面点不足，无法生成三角形网格。")
        width = self.geometry.case.workpiece_length_m
        try:
            initialize_gmsh(gmsh, ["grindcae-workbench-mesh", "-nopopup"])
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
            gmsh.option.setNumber("Mesh.Binary", 0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.settings.minimum_size_m)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.settings.maximum_size_m)
            gmsh.model.add("grindcae_workbench_complete_pass_geometry")

            bottom_left = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            bottom_right = gmsh.model.occ.addPoint(width, 0.0, 0.0)
            top_points = [
                gmsh.model.occ.addPoint(x_value, y_value, 0.0)
                for x_value, y_value in profile
            ]
            bottom = gmsh.model.occ.addLine(bottom_left, bottom_right)
            right = gmsh.model.occ.addLine(bottom_right, top_points[-1])
            top_curves: list[tuple[int, int]] = []
            for index in reversed(range(len(top_points) - 1)):
                curve = gmsh.model.occ.addLine(top_points[index + 1], top_points[index])
                top_curves.append((curve, index))
            left = gmsh.model.occ.addLine(top_points[0], bottom_left)
            loop = gmsh.model.occ.addCurveLoop(
                [bottom, right, *(curve for curve, _ in top_curves), left]
            )
            surface = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            physical: dict[str, list[int]] = {
                "fixed": [bottom],
                "free_left": [left],
                "free_right": [right],
                "entry_transition": [],
                "processed": [],
                "exit_transition": [],
            }
            for curve, index in top_curves:
                x_mid = 0.5 * (profile[index][0] + profile[index + 1][0])
                physical[_region_for_segment(self.geometry, x_mid)].append(curve)
            if any(not physical[name] for name in ("entry_transition", "processed", "exit_transition")):
                raise MeshValidationError("完整单程表面分区不完整，无法生成网格。")

            domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, domain_tag, "domain")
            for name in _PHYSICAL_GROUPS[1:]:
                tag = gmsh.model.addPhysicalGroup(1, physical[name])
                gmsh.model.setPhysicalName(1, tag, name)
            gmsh.model.mesh.setSize(
                gmsh.model.getEntities(0), self.settings.target_size_m
            )
            gmsh.model.mesh.generate(2)
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            element_types, element_blocks, _ = gmsh.model.mesh.getElements(2)
            triangle_count = 0
            for element_type, element_tags in zip(element_types, element_blocks):
                properties = gmsh.model.mesh.getElementProperties(element_type)
                if properties[1] != 2 or properties[2] != 1 or properties[3] != 3:
                    raise MeshValidationError("网格包含非一阶三角形单元。")
                triangle_count += len(element_tags)
            if len(node_tags) == 0 or triangle_count == 0:
                raise MeshValidationError("Gmsh 生成了空网格。")
            gmsh.write(str(output_path))
        except MeshValidationError:
            raise
        except Exception as exc:
            raise MeshValidationError(f"三角形网格生成失败：{exc}") from exc
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    def _validate_msh(self, mesh_path: Path) -> _ValidatedMesh:
        try:
            mesh_data = meshio.read(mesh_path)
        except Exception as exc:
            raise MeshValidationError(f"meshio 无法读取生成的 mesh.msh：{exc}") from exc
        if set(mesh_data.field_data) != set(_PHYSICAL_GROUPS):
            raise MeshValidationError("生成网格的物理边界标签不完整。")
        triangles = _mesh_cells(mesh_data, "triangle")
        lines = _mesh_cells(mesh_data, "line")
        if triangles.size == 0 or lines.size == 0:
            raise MeshValidationError("生成网格缺少三角形或边界线单元。")
        points = np.asarray(mesh_data.points[:, :2], dtype=float)
        triangle_points = points[triangles]
        cross = (
            (triangle_points[:, 1, 0] - triangle_points[:, 0, 0])
            * (triangle_points[:, 2, 1] - triangle_points[:, 0, 1])
            - (triangle_points[:, 1, 1] - triangle_points[:, 0, 1])
            * (triangle_points[:, 2, 0] - triangle_points[:, 0, 0])
        )
        areas = 0.5 * np.abs(cross)
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
            raise MeshValidationError("生成网格包含非正或非有限三角形面积。")
        try:
            imported = from_meshio(mesh_data)
        except Exception as exc:
            raise MeshValidationError(f"scikit-fem 无法导入生成网格：{exc}") from exc
        if not isinstance(imported, MeshTri):
            raise MeshValidationError("scikit-fem 导入结果不是三角形网格。")
        tags = _line_physical_tags(mesh_data)
        boundary_counts = {
            name: int(np.count_nonzero(tags == int(mesh_data.field_data[name][0])))
            for name in _PHYSICAL_GROUPS[1:]
        }
        if any(count <= 0 for count in boundary_counts.values()):
            raise MeshValidationError("一个或多个物理边界没有离散线单元。")
        characteristic_sizes = np.sqrt(4.0 * areas / math.sqrt(3.0))
        return _ValidatedMesh(
            mesh_data=mesh_data,
            node_count=int(len(points)),
            triangle_count=int(len(triangles)),
            boundary_element_count=int(len(lines)),
            minimum_element_area_m2=float(np.min(areas)),
            maximum_element_area_m2=float(np.max(areas)),
            actual_average_size_m=float(np.mean(characteristic_sizes)),
            boundary_element_counts=boundary_counts,
        )

    def _write_preview(self, validated: _ValidatedMesh, output_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        mesh_data = validated.mesh_data
        points_mm = 1.0e3 * np.asarray(mesh_data.points[:, :2], dtype=float)
        triangles = _mesh_cells(mesh_data, "triangle")
        lines = _mesh_cells(mesh_data, "line")
        tags = _line_physical_tags(mesh_data)
        triangulation = mtri.Triangulation(
            points_mm[:, 0], points_mm[:, 1], triangles
        )
        policy = select_mesh_preview_policy(
            node_count=validated.node_count,
            triangle_count=validated.triangle_count,
        )
        colors = {
            "fixed": "#C44E52",
            "entry_transition": "#F28E2B",
            "processed": "#2E8B57",
            "exit_transition": "#9467BD",
            "free_left": "#4E79A7",
            "free_right": "#4E79A7",
        }
        labels = {
            "fixed": "底部边界",
            "entry_transition": "磨入边界",
            "processed": "已加工边界",
            "exit_transition": "磨出边界",
            "free_left": "侧边界",
            "free_right": None,
        }
        with plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                "axes.unicode_minus": False,
            }
        ):
            figure, axis = plt.subplots(figsize=(12.8, 5.4), constrained_layout=True)
            try:
                if policy.strategy == "complete_mesh":
                    axis.triplot(triangulation, color="#7A7A7A", linewidth=0.45)
                elif policy.strategy == "complete_edges_without_nodes":
                    axis.triplot(triangulation, color="#8A8A8A", linewidth=0.20)
                for name in _PHYSICAL_GROUPS[1:]:
                    tag = int(mesh_data.field_data[name][0])
                    selected = lines[tags == tag]
                    for index, nodes in enumerate(selected):
                        axis.plot(
                            points_mm[nodes, 0],
                            points_mm[nodes, 1],
                            color=colors[name],
                            linewidth=2.3,
                            label=labels[name] if index == 0 else None,
                        )
                if policy.show_nodes:
                    axis.scatter(
                        points_mm[:, 0], points_mm[:, 1], s=5,
                        color="#202020", alpha=0.55,
                    )
                if policy.strategy == "global_outline_with_local_true_mesh":
                    x_min = float(np.min(points_mm[:, 0]))
                    x_max = float(np.max(points_mm[:, 0]))
                    y_min = float(np.min(points_mm[:, 1]))
                    y_max = float(np.max(points_mm[:, 1]))
                    cell_mm = max(
                        1.0e3 * validated.actual_average_size_m,
                        np.finfo(float).eps,
                    )
                    local_width = min(
                        x_max - x_min, policy.local_target_columns * cell_mm
                    )
                    local_height = min(
                        y_max - y_min, policy.local_target_rows * cell_mm
                    )
                    local_x_min = 0.5 * (x_min + x_max - local_width)
                    local_x_max = local_x_min + local_width
                    local_y_max = y_max
                    local_y_min = local_y_max - local_height
                    axis.plot(
                        [local_x_min, local_x_max, local_x_max, local_x_min, local_x_min],
                        [local_y_min, local_y_min, local_y_max, local_y_max, local_y_min],
                        color="#D62728",
                        linewidth=1.5,
                    )
                    axis.text(
                        local_x_min,
                        local_y_min,
                        "局部放大范围",
                        color="#D62728",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )
                    triangle_points = points_mm[triangles]
                    triangle_centers = np.mean(triangle_points, axis=1)
                    selected_triangles = triangles[
                        (triangle_centers[:, 0] >= local_x_min)
                        & (triangle_centers[:, 0] <= local_x_max)
                        & (triangle_centers[:, 1] >= local_y_min)
                        & (triangle_centers[:, 1] <= local_y_max)
                    ]
                    inset = axis.inset_axes([0.57, 0.10, 0.40, 0.62])
                    if len(selected_triangles):
                        local_triangulation = mtri.Triangulation(
                            points_mm[:, 0], points_mm[:, 1], selected_triangles
                        )
                        inset.triplot(
                            local_triangulation, color="#555555", linewidth=0.45
                        )
                    inset.set_xlim(local_x_min, local_x_max)
                    inset.set_ylim(local_y_min, local_y_max)
                    inset.set_aspect("equal", adjustable="box")
                    inset.set_title("局部真实网格", fontsize=9)
                    inset.tick_params(labelsize=7)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("工件 x [mm]")
                axis.set_ylabel("y [mm]")
                axis.set_title("三角形网格预览（完整单程二维几何）", fontweight="bold")
                axis.grid(alpha=0.16)
                handles, legend_labels = axis.get_legend_handles_labels()
                unique: dict[str, object] = {}
                for handle, label in zip(handles, legend_labels):
                    unique.setdefault(label, handle)
                axis.legend(
                    unique.values(), unique.keys(), loc="upper center", ncol=5, fontsize=9
                )
                axis.text(
                    0.01,
                    0.97,
                    f"{validated.node_count} 个节点 | {validated.triangle_count} 个三角形",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
                strategy_text = {
                    "complete_mesh": "完整网格线与节点",
                    "complete_edges_without_nodes": "完整网格线（隐藏节点）",
                    "global_outline_with_local_true_mesh": "全局轮廓＋局部真实网格",
                }[policy.strategy]
                axis.text(
                    0.01,
                    0.03,
                    f"显示策略：{strategy_text}",
                    transform=axis.transAxes,
                    ha="left",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
                figure.savefig(output_path, dpi=160, facecolor="white")
            finally:
                plt.close(figure)
        if output_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise MeshValidationError("网格预览不是有效 PNG。")

    @staticmethod
    def _mesh_statistics(validated: _ValidatedMesh) -> dict[str, Any]:
        return {
            "node_count": validated.node_count,
            "triangle_count": validated.triangle_count,
            "boundary_element_count": validated.boundary_element_count,
            "boundary_element_counts": validated.boundary_element_counts,
            "minimum_element_area_m2": validated.minimum_element_area_m2,
            "maximum_element_area_m2": validated.maximum_element_area_m2,
            "actual_average_size_m": validated.actual_average_size_m,
        }

    def _summary_from_validated(
        self, validated: _ValidatedMesh, artifacts: dict[str, str]
    ) -> dict[str, Any]:
        policy = select_mesh_preview_policy(
            node_count=validated.node_count,
            triangle_count=validated.triangle_count,
        )
        estimated_dofs = 2 * validated.node_count
        return {
            "result_format": MESH_RESULT_FORMAT,
            "mesh_status": "valid",
            "mesh_type": TRIANGLE_MESH_TYPE,
            "unit_system": "SI",
            "mesh_settings": {
                "target_size_m": self.settings.target_size_m,
                "minimum_size_m": self.settings.minimum_size_m,
                "maximum_size_m": self.settings.maximum_size_m,
            },
            "mesh_statistics": self._mesh_statistics(validated),
            "preview": {
                "strategy": policy.strategy,
                "show_nodes": policy.show_nodes,
                "local_target_columns": policy.local_target_columns,
                "local_target_rows": policy.local_target_rows,
                "complete_msh_preserved": True,
            },
            "solve_size_risk": {
                "estimated_vector_p1_degrees_of_freedom": estimated_dofs,
                "maximum_vector_p1_degrees_of_freedom": (
                    DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM
                ),
                "within_default_limit": (
                    estimated_dofs <= DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM
                ),
            },
            "import_validation": {
                "meshio_read": True,
                "scikit_fem_from_meshio": True,
                "same_generated_msh_reused": True,
            },
            "artifacts": artifacts,
            "limitations": [
                "当前阶段只生成二维三角形网格。",
                "当前结果不包含载荷、材料、位移、应力、应变或温度场。",
                "四边形网格和 STEP/CAD 导入将在后续版本实现。",
            ],
        }

    def mesh_summary(
        self, mesh_path: str | Path, output_directory: str | Path
    ) -> dict[str, Any]:
        validated = self._validate_msh(Path(mesh_path).expanduser().resolve())
        return self._summary_from_validated(
            validated, self.mesh_artifacts(output_directory)
        )

    def publish(self, output_directory: str | Path) -> MeshPublishedResult:
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="grindcae-workbench-mesh-", dir=str(output.parent)
        ) as temporary_directory:
            workspace = Path(temporary_directory)
            temporary_mesh = workspace / MESH_FILENAME
            temporary_preview = workspace / MESH_PREVIEW_FILENAME
            temporary_summary = workspace / MESH_SUMMARY_FILENAME
            try:
                self._generate_msh(temporary_mesh)
                validated = self._validate_msh(temporary_mesh)
                self._write_preview(validated, temporary_preview)
                final_mesh = (output / MESH_FILENAME).resolve()
                final_preview = (output / MESH_PREVIEW_FILENAME).resolve()
                final_summary = (output / MESH_SUMMARY_FILENAME).resolve()
                artifacts = self.mesh_artifacts(output)
                summary = self._summary_from_validated(validated, artifacts)
                temporary_summary.write_text(
                    json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                json.loads(temporary_summary.read_text(encoding="utf-8"))
                backups: dict[Path, Path] = {}
                published: list[Path] = []
                targets = (
                    (temporary_mesh, final_mesh),
                    (temporary_preview, final_preview),
                    (temporary_summary, final_summary),
                )
                try:
                    for _source, target in targets:
                        if target.exists():
                            backup = workspace / f"backup-{target.name}"
                            shutil.copy2(target, backup)
                            backups[target] = backup
                    for source, target in targets:
                        source.replace(target)
                        published.append(target)
                except OSError:
                    for target in published:
                        target.unlink(missing_ok=True)
                    for target, backup in backups.items():
                        shutil.copy2(backup, target)
                    raise
            except MeshValidationError:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise MeshValidationError(f"三角形网格发布失败：{exc}") from exc
        return MeshPublishedResult(
            result_format=MESH_RESULT_FORMAT,
            output_directory=output,
            mesh_path=(output / MESH_FILENAME).resolve(),
            preview_path=(output / MESH_PREVIEW_FILENAME).resolve(),
            summary_path=(output / MESH_SUMMARY_FILENAME).resolve(),
            summary=summary,
        )
