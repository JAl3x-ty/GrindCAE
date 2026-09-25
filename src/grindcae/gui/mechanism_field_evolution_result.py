"""Read-only metadata models for mechanism comparison field evolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


RESULT_FORMAT = "grindcae_mechanism_history_pass_target_field_evolution_comparison"
ROUTES = ("baseline", "mechanism", "difference")


class MechanismFieldEvolutionResultError(ValueError):
    """Raised when an explicitly registered comparison sequence is unsafe."""


@dataclass(frozen=True, slots=True)
class MechanismFieldEvolutionRouteFrame:
    Fx_N: float
    Fy_N: float
    accumulated_plastic_dissipation_J: float
    png_path: Path
    vtu_path: Path


@dataclass(frozen=True, slots=True)
class MechanismFieldEvolutionFrame:
    frame_index: int
    position_id: int
    motion_coordinate_m: float
    wheel_lowest_point_x_m: float
    pass_state: str
    contact_ratio: float
    is_final_unloaded_target: bool
    baseline: MechanismFieldEvolutionRouteFrame
    mechanism: MechanismFieldEvolutionRouteFrame
    difference: MechanismFieldEvolutionRouteFrame


@dataclass(frozen=True, slots=True)
class MechanismFieldEvolutionSequence:
    manifest_path: Path
    position_history_csv_path: Path
    sequence_directory: Path
    frame_count: int
    frames: tuple[MechanismFieldEvolutionFrame, ...]
    pvd_paths: Mapping[str, Path]
    shared_original_color_ranges: Mapping[str, tuple[float, float]]
    difference_color_ranges: Mapping[str, tuple[float, float]]
    sequence_coordinate: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MechanismFieldEvolutionAvailability:
    status: str
    sequence: MechanismFieldEvolutionSequence | None = None
    message: str = ""

    @classmethod
    def not_applicable(cls):
        return cls("not_applicable")

    @classmethod
    def legacy_missing(cls):
        return cls(
            "legacy_missing",
            message=(
                "该历史机制化结果生成于对比场演化功能加入之前，因此没有逐位置三路线场序列。\n"
                "仍可查看原有机制化历史对比图和数据文件。\n"
                "打开历史结果不会重新计算。"
            ),
        )

    @classmethod
    def unavailable(cls, reason: str):
        return cls("unavailable", message=f"机制化场演化序列不可用：{reason}")

    @classmethod
    def available(cls, sequence: MechanismFieldEvolutionSequence):
        return cls("available", sequence=sequence)


def read_mechanism_field_evolution_availability(
    summary: Mapping[str, Any], output_directory: Path
) -> MechanismFieldEvolutionAvailability:
    index = summary.get("field_evolution")
    if index is None:
        return MechanismFieldEvolutionAvailability.legacy_missing()
    try:
        return MechanismFieldEvolutionAvailability.available(
            _read_sequence(index, output_directory.resolve())
        )
    except (MechanismFieldEvolutionResultError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return MechanismFieldEvolutionAvailability.unavailable(str(exc))


def _read_sequence(raw_index: object, output: Path) -> MechanismFieldEvolutionSequence:
    index = _mapping(raw_index, "场演化摘要索引必须为 JSON object。")
    directory = _safe_path(_text(index, "directory"), output, output)
    if not directory.is_dir():
        raise MechanismFieldEvolutionResultError("场演化目录不存在。")
    manifest_path = _safe_path(_text(index, "manifest"), output, directory)
    history_path = _safe_path(_text(index, "position_history_csv"), output, directory)
    _require_file(manifest_path, "manifest.json")
    _require_file(history_path, "position_history.csv")
    route_index = _mapping(index.get("routes"), "摘要 routes 结构无效。")
    pvd_paths: dict[str, Path] = {}
    for route in ROUTES:
        details = _mapping(route_index.get(route), f"摘要路线 {route} 结构无效。")
        pvd = _safe_path(_text(details, "pvd"), output, directory)
        _require_file(pvd, f"{route} PVD")
        pvd_paths[route] = pvd
    if len(set(pvd_paths.values())) != len(ROUTES):
        raise MechanismFieldEvolutionResultError("三条路线的 PVD 路径存在重复。")

    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "机制化场演化 manifest 根对象无效。",
    )
    if manifest.get("result_format") != RESULT_FORMAT:
        raise MechanismFieldEvolutionResultError("manifest result_format 无效。")
    count = _positive_int(manifest.get("frame_count"), "frame_count")
    if _positive_int(index.get("frame_count"), "摘要 frame_count") != count:
        raise MechanismFieldEvolutionResultError("摘要和 manifest 的 frame_count 不一致。")
    if _strict_int(manifest.get("frame_index_start"), "frame_index_start") != 0:
        raise MechanismFieldEvolutionResultError("frame_index_start 必须为 0。")
    if _strict_int(manifest.get("frame_index_end"), "frame_index_end") != count - 1:
        raise MechanismFieldEvolutionResultError("frame_index_end 与 frame_count 不一致。")
    coordinate = _mapping(manifest.get("sequence_coordinate"), "sequence_coordinate 无效。")
    if coordinate.get("field") != "motion_coordinate_m" or coordinate.get("unit") != "m" or coordinate.get("meaning") != "wheel_motion_coordinate_not_time":
        raise MechanismFieldEvolutionResultError("sequence_coordinate 无效。")
    if coordinate.get("pvd_timestep_uses_motion_coordinate") is not True:
        raise MechanismFieldEvolutionResultError("PVD 运动坐标标识无效。")
    shared_ranges = _ranges(
        manifest.get("shared_original_color_ranges"),
        ("displacement_magnitude_m", "von_mises_stress_Pa", "equivalent_plastic_strain"),
        symmetric=False,
    )
    difference_ranges = _ranges(
        manifest.get("difference_color_ranges"),
        ("displacement_magnitude_difference_m", "von_mises_stress_difference_Pa", "equivalent_plastic_strain_difference"),
        symmetric=True,
    )
    locations = _mapping(manifest.get("field_locations"), "field_locations 结构无效。")
    for name in (
        "displacement_magnitude_m",
        "von_mises_stress_Pa",
        "equivalent_plastic_strain",
        "displacement_magnitude_difference_m",
        "von_mises_stress_difference_Pa",
        "equivalent_plastic_strain_difference",
    ):
        if not isinstance(locations.get(name), str) or not locations[name].strip():
            raise MechanismFieldEvolutionResultError(
                f"field_locations 缺少有效字段：{name}"
            )
    manifest_routes = _mapping(manifest.get("routes"), "manifest routes 结构无效。")
    for route in ROUTES:
        details = _mapping(manifest_routes.get(route), f"manifest 路线 {route} 无效。")
        expected = _safe_path(_text(details, "pvd_path"), directory, output)
        if expected != pvd_paths[route]:
            raise MechanismFieldEvolutionResultError(f"{route} PVD 索引不一致。")
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != count:
        raise MechanismFieldEvolutionResultError("frames 数量与 frame_count 不一致。")
    used_paths: set[Path] = set()
    frames = tuple(
        _frame(raw, expected, directory, output, used_paths)
        for expected, raw in enumerate(raw_frames)
    )
    if not frames[-1].is_final_unloaded_target:
        raise MechanismFieldEvolutionResultError("最后一帧未标记为最终卸载目标。")
    return MechanismFieldEvolutionSequence(
        manifest_path,
        history_path,
        directory,
        count,
        frames,
        pvd_paths,
        shared_ranges,
        difference_ranges,
        dict(coordinate),
    )


def _frame(raw: object, expected: int, directory: Path, output: Path, used: set[Path]):
    item = _mapping(raw, f"第 {expected} 帧必须为 JSON object。")
    if _strict_int(item.get("frame_index"), "frame_index") != expected:
        raise MechanismFieldEvolutionResultError("frame_index 必须从 0 连续增长。")
    contact = _finite(item.get("contact_ratio"), "contact_ratio")
    if contact < -1e-12 or contact > 1.0 + 1e-12:
        raise MechanismFieldEvolutionResultError("contact_ratio 超出 [0, 1]。")
    final = item.get("is_final_unloaded_target")
    if type(final) is not bool:
        raise MechanismFieldEvolutionResultError("最终卸载标识必须为严格布尔值。")
    routes: dict[str, MechanismFieldEvolutionRouteFrame] = {}
    for route in ROUTES:
        details = _mapping(item.get(route), f"帧路线 {route} 结构无效。")
        png = _safe_path(_text(details, "png_path"), directory, output)
        vtu = _safe_path(_text(details, "vtu_path"), directory, output)
        if png in used or vtu in used:
            raise MechanismFieldEvolutionResultError("PNG 或 VTU 路径存在重复登记。")
        used.update((png, vtu))
        _require_file(png, f"{route} PNG")
        _require_file(vtu, f"{route} VTU")
        if route == "difference":
            Fx = Fy = dissipation = 0.0
        else:
            Fx = _finite(details.get("Fx_N"), f"{route}.Fx_N")
            Fy = _finite(details.get("Fy_N"), f"{route}.Fy_N")
            dissipation = _finite(
                details.get("accumulated_plastic_dissipation_J"),
                f"{route}.accumulated_plastic_dissipation_J",
            )
        routes[route] = MechanismFieldEvolutionRouteFrame(Fx, Fy, dissipation, png, vtu)
    return MechanismFieldEvolutionFrame(
        expected,
        _strict_int(item.get("position_id"), "position_id"),
        _finite(item.get("motion_coordinate_m"), "motion_coordinate_m"),
        _finite(item.get("wheel_lowest_point_x_m"), "wheel_lowest_point_x_m"),
        _text(item, "pass_state"),
        min(1.0, max(0.0, contact)),
        final,
        routes["baseline"],
        routes["mechanism"],
        routes["difference"],
    )


def _ranges(raw: object, names: tuple[str, ...], *, symmetric: bool):
    mapping = _mapping(raw, "色标结构无效。")
    result = {}
    for name in names:
        limits = _mapping(mapping.get(name), f"色标 {name} 结构无效。")
        minimum = _finite(limits.get("minimum"), f"{name}.minimum")
        maximum = _finite(limits.get("maximum"), f"{name}.maximum")
        if minimum > maximum:
            raise MechanismFieldEvolutionResultError(f"色标 {name} 范围无效。")
        if symmetric and not math.isclose(minimum, -maximum, rel_tol=1e-12, abs_tol=1e-15):
            raise MechanismFieldEvolutionResultError(f"差值色标 {name} 必须关于零对称。")
        if not symmetric and minimum < 0.0:
            raise MechanismFieldEvolutionResultError(f"原始色标 {name} 不能为负。")
        result[name] = (minimum, maximum)
    return result


def _mapping(raw: object, message: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise MechanismFieldEvolutionResultError(message)
    return raw


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MechanismFieldEvolutionResultError(f"{key} 必须为非空字符串。")
    return value


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise MechanismFieldEvolutionResultError(f"{name} 必须为严格整数。")
    return value


def _positive_int(value: object, name: str) -> int:
    value = _strict_int(value, name)
    if value <= 0:
        raise MechanismFieldEvolutionResultError(f"{name} 必须为正整数。")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismFieldEvolutionResultError(f"{name} 必须为有限数值。")
    value = float(value)
    if not math.isfinite(value):
        raise MechanismFieldEvolutionResultError(f"{name} 必须为有限数值。")
    return value


def _safe_path(raw: str, base: Path, boundary: Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(boundary.resolve())
    except ValueError as exc:
        raise MechanismFieldEvolutionResultError(f"场演化路径超出正式结果目录：{raw}") from exc
    return resolved


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MechanismFieldEvolutionResultError(f"{label} 不存在或为空。")


__all__ = [
    "MechanismFieldEvolutionAvailability",
    "MechanismFieldEvolutionFrame",
    "MechanismFieldEvolutionResultError",
    "MechanismFieldEvolutionRouteFrame",
    "MechanismFieldEvolutionSequence",
    "read_mechanism_field_evolution_availability",
]
