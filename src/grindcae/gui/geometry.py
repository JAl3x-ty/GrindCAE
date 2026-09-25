"""Parametric two-dimensional geometry adapter for the 2.5 workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from grindcae.trajectory import (
    MODEL_TYPE,
    TRAJECTORY_SCHEMA_VERSION,
    UNIT_SYSTEM,
    SinglePassSettings,
    TrajectoryCase,
    TrajectorySampling,
    TrajectoryValidationError,
    TrajectoryWheel,
    TrajectoryWorkpiece,
    build_surface_profile,
)


GEOMETRY_SCHEMA_VERSION = 1
GEOMETRY_SOURCE = "parametric_2d"
GEOMETRY_RESULT_FORMAT = "grindcae_2d_parametric_geometry_preview"
PASS_MODE = "single_pass"
PHYSICS_FIELDS = ("mechanical",)
PREVIEW_FILENAME = "geometry_preview.png"
SUMMARY_FILENAME = "geometry_summary.json"
_DIRECTIONS = ("positive_x", "negative_x")


class GeometryValidationError(ValueError):
    """Chinese geometry validation error suitable for direct GUI display."""


def _finite_display_number(value: str, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise GeometryValidationError(f"{label}：请输入有效数字。")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryValidationError(f"{label}：请输入有效数字。") from exc
    if not math.isfinite(number):
        raise GeometryValidationError(f"{label}：请输入有效数字。")
    return number


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryValidationError(f"{label}必须是 JSON 对象。")
    return value


def _require_fields(data: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise GeometryValidationError(f"{label}缺少必需字段：{', '.join(missing)}")


def _finite_si(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryValidationError(f"{label}必须是有限数字。")
    number = float(value)
    if not math.isfinite(number):
        raise GeometryValidationError(f"{label}必须是有限数字。")
    return number


@dataclass(frozen=True)
class GeometryDisplayInput:
    workpiece_length_mm: str = "100"
    workpiece_height_mm: str = "20"
    wheel_diameter_mm: str = "200"
    depth_of_cut_um: str = "20"
    relative_feed_direction: str = "positive_x"
    wheel_lowest_point_x_mm: str = "50"

    def build_case(self) -> "ParametricGeometryCase":
        length_mm = _finite_display_number(self.workpiece_length_mm, "工件长度")
        height_mm = _finite_display_number(self.workpiece_height_mm, "工件初始高度")
        diameter_mm = _finite_display_number(self.wheel_diameter_mm, "砂轮直径")
        depth_um = _finite_display_number(self.depth_of_cut_um, "磨削深度")
        lowest_mm = _finite_display_number(
            self.wheel_lowest_point_x_mm, "砂轮最低点位置"
        )
        if length_mm <= 0.0:
            raise GeometryValidationError("工件长度必须大于 0。")
        if height_mm <= 0.0:
            raise GeometryValidationError("工件初始高度必须大于 0。")
        if diameter_mm <= 0.0:
            raise GeometryValidationError("砂轮直径必须大于 0。")
        if depth_um <= 0.0:
            raise GeometryValidationError("磨削深度必须大于 0。")
        if depth_um * 1.0e-3 >= height_mm:
            raise GeometryValidationError("磨削深度必须小于工件初始高度。")
        if self.relative_feed_direction not in _DIRECTIONS:
            raise GeometryValidationError("磨削方向无效，请选择正向或反向。")
        reasonable_margin_mm = diameter_mm / 2.0
        if not -reasonable_margin_mm <= lowest_mm <= length_mm + reasonable_margin_mm:
            raise GeometryValidationError("砂轮最低点位置超出合理范围。")
        return ParametricGeometryCase(
            workpiece_length_m=length_mm * 1.0e-3,
            workpiece_height_m=height_mm * 1.0e-3,
            wheel_diameter_m=diameter_mm * 1.0e-3,
            depth_of_cut_m=depth_um * 1.0e-6,
            relative_feed_direction=self.relative_feed_direction,
            wheel_lowest_point_x_m=lowest_mm * 1.0e-3,
        )


@dataclass(frozen=True)
class ParametricGeometryCase:
    workpiece_length_m: float
    workpiece_height_m: float
    wheel_diameter_m: float
    depth_of_cut_m: float
    relative_feed_direction: str
    wheel_lowest_point_x_m: float
    regions: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(
        default_factory=lambda: {"status": "valid", "messages": []}
    )
    physics_fields: tuple[str, ...] = PHYSICS_FIELDS
    root_extra: dict[str, Any] = field(default_factory=dict)
    workpiece_extra: dict[str, Any] = field(default_factory=dict)
    wheel_extra: dict[str, Any] = field(default_factory=dict)
    process_extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            (self.workpiece_length_m, "工件长度"),
            (self.workpiece_height_m, "工件初始高度"),
            (self.wheel_diameter_m, "砂轮直径"),
            (self.depth_of_cut_m, "磨削深度"),
            (self.wheel_lowest_point_x_m, "砂轮最低点位置"),
        )
        for value, label in values:
            _finite_si(value, label)
        if self.workpiece_length_m <= 0.0:
            raise GeometryValidationError("工件长度必须大于 0。")
        if self.workpiece_height_m <= 0.0:
            raise GeometryValidationError("工件初始高度必须大于 0。")
        if self.wheel_diameter_m <= 0.0:
            raise GeometryValidationError("砂轮直径必须大于 0。")
        if self.depth_of_cut_m <= 0.0:
            raise GeometryValidationError("磨削深度必须大于 0。")
        if self.depth_of_cut_m >= self.workpiece_height_m:
            raise GeometryValidationError("磨削深度必须小于工件初始高度。")
        if self.relative_feed_direction not in _DIRECTIONS:
            raise GeometryValidationError("磨削方向无效，请选择正向或反向。")
        margin = self.wheel_diameter_m / 2.0
        if not -margin <= self.wheel_lowest_point_x_m <= self.workpiece_length_m + margin:
            raise GeometryValidationError("砂轮最低点位置超出合理范围。")

    @classmethod
    def from_mapping(cls, payload: object) -> "ParametricGeometryCase":
        root = _mapping(payload, "geometry")
        _require_fields(
            root,
            (
                "source",
                "geometry_schema_version",
                "unit_system",
                "workpiece",
                "wheel",
                "process",
                "regions",
                "validation",
            ),
            "geometry",
        )
        version = root["geometry_schema_version"]
        if type(version) is not int or version != GEOMETRY_SCHEMA_VERSION:
            raise GeometryValidationError(
                f"不支持的几何文件版本：{version}。当前仅支持版本 1。"
            )
        source = root["source"]
        if source == "step_cad":
            raise GeometryValidationError("STEP/CAD 导入将在后续版本实现。")
        if source != GEOMETRY_SOURCE:
            raise GeometryValidationError(f"未知几何来源：{source}")
        if root["unit_system"] != UNIT_SYSTEM:
            raise GeometryValidationError("二维参数化几何仅支持 SI 单位。")
        workpiece = _mapping(root["workpiece"], "geometry.workpiece")
        wheel = _mapping(root["wheel"], "geometry.wheel")
        process = _mapping(root["process"], "geometry.process")
        _require_fields(
            workpiece,
            ("length_m", "original_surface_height_m"),
            "geometry.workpiece",
        )
        _require_fields(
            wheel,
            ("diameter_m", "lowest_point_x_m"),
            "geometry.wheel",
        )
        _require_fields(
            process,
            ("depth_of_cut_m", "relative_feed_direction", "pass_mode"),
            "geometry.process",
        )
        if process["pass_mode"] == "multi_pass":
            raise GeometryValidationError("多道次磨削将在后续版本实现。")
        if process["pass_mode"] != PASS_MODE:
            raise GeometryValidationError(f"未知磨削过程模式：{process['pass_mode']}")
        known_root = {
            "source",
            "geometry_schema_version",
            "unit_system",
            "workpiece",
            "wheel",
            "process",
            "regions",
            "validation",
            "physics_fields",
        }
        return cls(
            workpiece_length_m=_finite_si(workpiece["length_m"], "工件长度"),
            workpiece_height_m=_finite_si(
                workpiece["original_surface_height_m"], "工件初始高度"
            ),
            wheel_diameter_m=_finite_si(wheel["diameter_m"], "砂轮直径"),
            depth_of_cut_m=_finite_si(process["depth_of_cut_m"], "磨削深度"),
            relative_feed_direction=process["relative_feed_direction"],
            wheel_lowest_point_x_m=_finite_si(
                wheel["lowest_point_x_m"], "砂轮最低点位置"
            ),
            regions=dict(_mapping(root["regions"], "geometry.regions")),
            validation=dict(_mapping(root["validation"], "geometry.validation")),
            physics_fields=tuple(root.get("physics_fields", PHYSICS_FIELDS)),
            root_extra={key: value for key, value in root.items() if key not in known_root},
            workpiece_extra={
                key: value
                for key, value in workpiece.items()
                if key not in {"length_m", "original_surface_height_m"}
            },
            wheel_extra={
                key: value
                for key, value in wheel.items()
                if key not in {"diameter_m", "lowest_point_x_m"}
            },
            process_extra={
                key: value
                for key, value in process.items()
                if key
                not in {"depth_of_cut_m", "relative_feed_direction", "pass_mode"}
            },
        )

    def to_dict(self) -> dict[str, Any]:
        workpiece = dict(self.workpiece_extra)
        workpiece.update(
            {
                "length_m": self.workpiece_length_m,
                "original_surface_height_m": self.workpiece_height_m,
            }
        )
        wheel = dict(self.wheel_extra)
        wheel.update(
            {
                "diameter_m": self.wheel_diameter_m,
                "lowest_point_x_m": self.wheel_lowest_point_x_m,
            }
        )
        process = dict(self.process_extra)
        process.update(
            {
                "depth_of_cut_m": self.depth_of_cut_m,
                "relative_feed_direction": self.relative_feed_direction,
                "pass_mode": PASS_MODE,
            }
        )
        data = dict(self.root_extra)
        data.update(
            {
                "source": GEOMETRY_SOURCE,
                "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
                "unit_system": UNIT_SYSTEM,
                "workpiece": workpiece,
                "wheel": wheel,
                "process": process,
                "regions": dict(self.regions),
                "validation": dict(self.validation),
                "physics_fields": list(self.physics_fields),
            }
        )
        return data

    def to_display_input(self) -> GeometryDisplayInput:
        direction = self.relative_feed_direction
        return GeometryDisplayInput(
            workpiece_length_mm=f"{self.workpiece_length_m * 1.0e3:.12g}",
            workpiece_height_mm=f"{self.workpiece_height_m * 1.0e3:.12g}",
            wheel_diameter_mm=f"{self.wheel_diameter_m * 1.0e3:.12g}",
            depth_of_cut_um=f"{self.depth_of_cut_m * 1.0e6:.12g}",
            relative_feed_direction=direction,
            wheel_lowest_point_x_mm=f"{self.wheel_lowest_point_x_m * 1.0e3:.12g}",
        )


@dataclass(frozen=True)
class ParametricGeometryModel:
    case: ParametricGeometryCase
    entry_boundary_x_m: float
    exit_boundary_x_m: float
    ground_surface_height_m: float
    transition_length_m: float
    regions: dict[str, dict[str, Any]]
    top_profile_x_m: tuple[float, ...]
    top_profile_y_m: tuple[float, ...]
    top_profile_region: tuple[str, ...]
    wheel_center_x_m: float
    wheel_center_y_m: float
    validation_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeometryPublishedResult:
    result_format: str
    output_directory: Path
    preview_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _trajectory_case(
    case: ParametricGeometryCase,
    wheel_lowest_point_x_m: float,
    direction: str = "positive_x",
) -> TrajectoryCase:
    try:
        return TrajectoryCase(
            trajectory_schema_version=TRAJECTORY_SCHEMA_VERSION,
            unit_system=UNIT_SYSTEM,
            model_type=MODEL_TYPE,
            workpiece=TrajectoryWorkpiece(
                case.workpiece_length_m, case.workpiece_height_m
            ),
            wheel=TrajectoryWheel(case.wheel_diameter_m),
            single_pass=SinglePassSettings(
                case.depth_of_cut_m,
                direction,
                wheel_lowest_point_x_m,
            ),
            sampling=TrajectorySampling(401),
        )
    except TrajectoryValidationError as exc:
        raise GeometryValidationError(f"二维几何参数无效：{exc}") from exc


class Parametric2DGeometryProvider:
    """Adapt the public phase 4C1 geometry without importing mesh or solver code."""

    def __init__(self, case: ParametricGeometryCase) -> None:
        if not isinstance(case, ParametricGeometryCase):
            raise GeometryValidationError("二维几何对象无效。")
        self.case = case

    def validate_geometry(self) -> tuple[bool, str]:
        self.build_geometry()
        return True, "二维参数化几何有效。"

    def _transition_template(self):
        return build_surface_profile(_trajectory_case(self.case, 0.0))

    def build_geometry(self) -> ParametricGeometryModel:
        template = self._transition_template()
        width = self.case.workpiece_length_m
        height = self.case.workpiece_height_m
        depth = self.case.depth_of_cut_m
        transition = template.exact_arc_projected_length_m
        if 2.0 * transition >= width:
            raise GeometryValidationError(
                "工件长度不足以同时容纳磨入区、已加工中部和磨出区。"
            )
        arc_points = tuple(
            point for point in template.sample_points if point.region == "contact_arc"
        )
        if len(arc_points) < 2:
            raise GeometryValidationError("无法从现有单程几何逻辑取得圆弧过渡。")
        ascending = tuple((point.x_m, point.surface_y_m) for point in arc_points)
        descending = tuple(
            sorted((transition - x_value, y_value) for x_value, y_value in ascending)
        )
        ground_height = height - depth
        middle_count = 101
        middle_x = tuple(
            transition
            + (width - 2.0 * transition) * index / (middle_count - 1)
            for index in range(middle_count)
        )
        positive_segments = {
            "entry_transition": (0.0, transition),
            "processed": (transition, width - transition),
            "exit_transition": (width - transition, width),
        }
        if self.case.relative_feed_direction == "positive_x":
            entry_points = descending
            exit_points = tuple((width - transition + x, y) for x, y in ascending)
            region_bounds = positive_segments
        else:
            entry_points = tuple((width - transition + x, y) for x, y in ascending)
            exit_points = descending
            region_bounds = {
                "entry_transition": (width - transition, width),
                "processed": (transition, width - transition),
                "exit_transition": (0.0, transition),
            }
        points = sorted(
            (*entry_points, *((x, ground_height) for x in middle_x), *exit_points),
            key=lambda item: item[0],
        )
        deduplicated: list[tuple[float, float]] = []
        for point in points:
            if deduplicated and math.isclose(
                point[0], deduplicated[-1][0], rel_tol=0.0, abs_tol=1.0e-15
            ):
                deduplicated[-1] = point
            else:
                deduplicated.append(point)
        regions = {
            "unprocessed": {
                "x_start_m": 0.0,
                "x_end_m": width,
                "surface_height_m": height,
                "role": "original_reference_surface",
            },
            **{
                name: {
                    "x_start_m": bounds[0],
                    "x_end_m": bounds[1],
                    "role": name,
                }
                for name, bounds in region_bounds.items()
            },
        }
        labels: list[str] = []
        for x_value, _y_value in deduplicated:
            if region_bounds["entry_transition"][0] <= x_value <= region_bounds["entry_transition"][1]:
                labels.append("entry_transition")
            elif region_bounds["exit_transition"][0] <= x_value <= region_bounds["exit_transition"][1]:
                labels.append("exit_transition")
            else:
                labels.append("processed")
        radius = self.case.wheel_diameter_m / 2.0
        return ParametricGeometryModel(
            case=self.case,
            entry_boundary_x_m=region_bounds["entry_transition"][1]
            if self.case.relative_feed_direction == "positive_x"
            else region_bounds["entry_transition"][0],
            exit_boundary_x_m=region_bounds["exit_transition"][0]
            if self.case.relative_feed_direction == "positive_x"
            else region_bounds["exit_transition"][1],
            ground_surface_height_m=ground_height,
            transition_length_m=transition,
            regions=regions,
            top_profile_x_m=tuple(point[0] for point in deduplicated),
            top_profile_y_m=tuple(point[1] for point in deduplicated),
            top_profile_region=tuple(labels),
            wheel_center_x_m=self.case.wheel_lowest_point_x_m,
            wheel_center_y_m=ground_height + radius,
        )

    def geometry_summary(self, model: ParametricGeometryModel | None = None) -> dict[str, Any]:
        geometry = model or self.build_geometry()
        direction = (
            "正向" if self.case.relative_feed_direction == "positive_x" else "反向"
        )
        return {
            "result_format": GEOMETRY_RESULT_FORMAT,
            "geometry_status": "valid",
            "geometry_summary_chinese": {
                "工件长度": f"{self.case.workpiece_length_m * 1.0e3:g} mm",
                "工件初始高度": f"{self.case.workpiece_height_m * 1.0e3:g} mm",
                "砂轮直径": f"{self.case.wheel_diameter_m * 1.0e3:g} mm",
                "磨削深度": f"{self.case.depth_of_cut_m * 1.0e6:g} μm",
                "磨削方向": direction,
                "砂轮最低点位置": f"{self.case.wheel_lowest_point_x_m * 1.0e3:g} mm",
                "几何来源": "参数化二维",
                "磨入过渡长度": f"{geometry.transition_length_m * 1.0e3:.6g} mm",
                "磨出过渡长度": f"{geometry.transition_length_m * 1.0e3:.6g} mm",
            },
            "regions": geometry.regions,
            "validation": {"status": "valid", "messages": []},
            "preview": {"filename": PREVIEW_FILENAME, "coordinate_unit": "mm"},
            "limitations": [
                "当前阶段只完成二维几何建模。",
                "网格生成将在后续版本实现。",
                "当前结果不包含应力、应变、位移或磨削力。",
            ],
        }

    def to_geometry_schema(
        self, model: ParametricGeometryModel | None = None
    ) -> dict[str, Any]:
        geometry = model or self.build_geometry()
        case = ParametricGeometryCase(
            workpiece_length_m=self.case.workpiece_length_m,
            workpiece_height_m=self.case.workpiece_height_m,
            wheel_diameter_m=self.case.wheel_diameter_m,
            depth_of_cut_m=self.case.depth_of_cut_m,
            relative_feed_direction=self.case.relative_feed_direction,
            wheel_lowest_point_x_m=self.case.wheel_lowest_point_x_m,
            regions=geometry.regions,
            validation={"status": "valid", "messages": []},
            physics_fields=self.case.physics_fields,
            root_extra=self.case.root_extra,
            workpiece_extra=self.case.workpiece_extra,
            wheel_extra=self.case.wheel_extra,
            process_extra=self.case.process_extra,
        )
        return case.to_dict()

    def geometry_to_mesh_input(
        self, model: ParametricGeometryModel | None = None
    ) -> dict[str, Any]:
        geometry = model or self.build_geometry()
        return {
            "source": GEOMETRY_SOURCE,
            "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
            "unit_system": UNIT_SYSTEM,
            "physics_fields": list(PHYSICS_FIELDS),
            "workpiece": {
                "length_m": self.case.workpiece_length_m,
                "original_surface_height_m": self.case.workpiece_height_m,
            },
            "top_profile": {
                "x_m": list(geometry.top_profile_x_m),
                "surface_y_m": list(geometry.top_profile_y_m),
                "region": list(geometry.top_profile_region),
            },
            "regions": geometry.regions,
        }

    def build_preview(
        self,
        output_path: str | Path,
        model: ParametricGeometryModel | None = None,
    ) -> Path:
        geometry = model or self.build_geometry()
        return self._write_preview(geometry, Path(output_path).expanduser().resolve())

    def _write_preview(
        self, geometry: ParametricGeometryModel, output_path: Path
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width_mm = self.case.workpiece_length_m * 1.0e3
        height_mm = self.case.workpiece_height_m * 1.0e3
        depth_um = self.case.depth_of_cut_m * 1.0e6
        radius_mm = self.case.wheel_diameter_m * 0.5e3
        center_x_mm = geometry.wheel_center_x_m * 1.0e3
        center_y_mm = geometry.wheel_center_y_m * 1.0e3
        x_mm = [value * 1.0e3 for value in geometry.top_profile_x_m]
        elevation_um = [
            (value - self.case.workpiece_height_m) * 1.0e6
            for value in geometry.top_profile_y_m
        ]
        colors = {
            "entry_transition": "#F28E2B",
            "processed": "#2E8B57",
            "exit_transition": "#9467BD",
        }
        labels = {
            "entry_transition": "磨入过渡区",
            "processed": "已加工面",
            "exit_transition": "磨出过渡区",
        }
        with plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                "axes.unicode_minus": False,
            }
        ):
            figure, axes = plt.subplots(2, 1, figsize=(12.8, 8.0))
            figure.subplots_adjust(left=0.07, right=0.97, bottom=0.19, top=0.91, hspace=0.36)
            try:
                full, local = axes
                full.add_patch(
                    Rectangle(
                        (0.0, 0.0),
                        width_mm,
                        height_mm,
                        facecolor="#E6E6E6",
                        edgecolor="#666666",
                        linewidth=1.3,
                        label="工件原始轮廓",
                    )
                )
                full.add_patch(
                    Circle(
                        (center_x_mm, center_y_mm),
                        radius_mm,
                        fill=False,
                        edgecolor="#222222",
                        linewidth=1.8,
                        label="砂轮圆弧",
                    )
                )
                full.axvline(
                    center_x_mm,
                    color="#444444",
                    linestyle="--",
                    linewidth=1.0,
                    label="砂轮最低点位置",
                )
                full.axhline(height_mm, color="#4E79A7", linewidth=2.2, label="待加工面")
                full.set_xlim(-0.02 * width_mm, 1.02 * width_mm)
                full.set_ylim(0.0, height_mm + max(0.2 * height_mm, 4.0))
                full.set_aspect("equal", adjustable="box")
                full.set_xlabel("工件 x [mm]")
                full.set_ylabel("y [mm]")
                full.set_title("A  工件轮廓与当前砂轮参考圆弧", loc="left", fontweight="bold")
                full.grid(alpha=0.20)

                for region in ("entry_transition", "processed", "exit_transition"):
                    mask_x: list[float] = []
                    mask_y: list[float] = []
                    for x_value, y_value, point_region in zip(
                        x_mm, elevation_um, geometry.top_profile_region
                    ):
                        if point_region == region:
                            mask_x.append(x_value)
                            mask_y.append(y_value)
                    local.plot(
                        mask_x,
                        mask_y,
                        color=colors[region],
                        linewidth=3.0,
                        label=labels[region],
                    )
                local.axhline(0.0, color="#4E79A7", linestyle="--", label="原始待加工面")
                local.axvline(center_x_mm, color="#333333", linestyle=":", label="砂轮最低点位置")
                local.set_xlim(0.0, width_mm)
                local.set_ylim(-1.2 * depth_um, 0.2 * depth_um)
                local.set_xlabel("工件 x [mm]")
                local.set_ylabel("相对原始表面高度 [μm]")
                local.set_title("B  完整单程二维表面分区（竖向放大）", loc="left", fontweight="bold")
                local.grid(axis="y", alpha=0.25)
                handles, legend_labels = [], []
                for axis in axes:
                    for handle, label in zip(*axis.get_legend_handles_labels()):
                        if label not in legend_labels:
                            handles.append(handle)
                            legend_labels.append(label)
                figure.legend(
                    handles,
                    legend_labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.065),
                    ncol=4,
                    fontsize=9,
                )
                direction = "正向" if self.case.relative_feed_direction == "positive_x" else "反向"
                figure.suptitle(
                    f"二维几何预览 | {direction}单程磨削 | ae = {depth_um:g} μm",
                    fontsize=15,
                    fontweight="bold",
                )
                figure.text(
                    0.5,
                    0.018,
                    "仅表示参数化几何与理想圆弧过渡；不包含网格、力、应力、应变、位移或温度场。",
                    ha="center",
                    color="#8A4B08",
                    fontsize=9,
                )
                figure.savefig(output_path, dpi=160, facecolor="white")
            finally:
                plt.close(figure)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise GeometryValidationError("二维几何预览 PNG 生成失败。")
        return output_path

    def publish(self, output_directory: str | Path) -> GeometryPublishedResult:
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        model = self.build_geometry()
        summary = self.geometry_summary(model)
        with tempfile.TemporaryDirectory(
            prefix="grindcae-geometry-", dir=str(output.parent)
        ) as temp_directory:
            workspace = Path(temp_directory)
            temporary_preview = workspace / PREVIEW_FILENAME
            temporary_summary = workspace / SUMMARY_FILENAME
            try:
                self._write_preview(model, temporary_preview)
                temporary_summary.write_text(
                    json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                if temporary_preview.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                    raise GeometryValidationError("二维几何预览不是有效 PNG。")
                json.loads(temporary_summary.read_text(encoding="utf-8"))
                final_preview = output / PREVIEW_FILENAME
                final_summary = output / SUMMARY_FILENAME
                backups: dict[Path, Path] = {}
                published: list[Path] = []
                try:
                    for target in (final_preview, final_summary):
                        if target.exists():
                            backup = workspace / f"backup-{target.name}"
                            shutil.copy2(target, backup)
                            backups[target] = backup
                    temporary_preview.replace(final_preview)
                    published.append(final_preview)
                    temporary_summary.replace(final_summary)
                    published.append(final_summary)
                except OSError:
                    for target in published:
                        target.unlink(missing_ok=True)
                    for target, backup in backups.items():
                        shutil.copy2(backup, target)
                    raise
            except (GeometryValidationError, OSError, ValueError, TypeError) as exc:
                if isinstance(exc, GeometryValidationError):
                    message = str(exc)
                else:
                    message = f"{type(exc).__name__}: {exc}"
                raise GeometryValidationError(f"二维几何发布失败：{message}") from exc
        return GeometryPublishedResult(
            result_format=GEOMETRY_RESULT_FORMAT,
            output_directory=output,
            preview_path=(output / PREVIEW_FILENAME).resolve(),
            summary_path=(output / SUMMARY_FILENAME).resolve(),
            summary=summary,
        )
