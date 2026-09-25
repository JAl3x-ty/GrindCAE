"""Read-only models and validation for persisted single-pass field evolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


FIELD_EVOLUTION_RESULT_FORMAT = "grindcae_history_pass_target_field_evolution"


class FieldEvolutionResultError(ValueError):
    """Raised when an explicitly registered field-evolution sequence is unsafe."""


@dataclass(frozen=True, slots=True)
class FieldEvolutionFrame:
    frame_index: int
    position_id: int
    motion_coordinate_m: float
    wheel_lowest_point_x_m: float
    pass_state: str
    contact_ratio: float
    Fx_N: float
    Fy_N: float
    accumulated_plastic_dissipation_J: float
    is_final_unloaded_target: bool
    png_path: Path
    vtu_path: Path


@dataclass(frozen=True, slots=True)
class FieldEvolutionSequence:
    manifest_path: Path
    position_history_csv_path: Path
    pvd_path: Path
    sequence_directory: Path
    frame_count: int
    frames: tuple[FieldEvolutionFrame, ...]
    color_ranges: Mapping[str, tuple[float, float]]
    sequence_coordinate: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FieldEvolutionAvailability:
    status: str
    sequence: FieldEvolutionSequence | None = None
    message: str = ""

    @classmethod
    def not_applicable(cls) -> FieldEvolutionAvailability:
        return cls("not_applicable")

    @classmethod
    def legacy_missing(cls) -> FieldEvolutionAvailability:
        return cls(
            "legacy_missing",
            message=(
                "该历史结果生成于单程场演化功能加入之前，因此没有逐位置场序列。\n"
                "仍可查看已有的最终卸载残余状态和历史数据。\n"
                "打开历史结果不会自动重新计算。"
            ),
        )

    @classmethod
    def unavailable(cls, reason: str) -> FieldEvolutionAvailability:
        return cls("unavailable", message=f"场演化序列不可用：{reason}")

    @classmethod
    def available(cls, sequence: FieldEvolutionSequence) -> FieldEvolutionAvailability:
        return cls("available", sequence=sequence)


def read_field_evolution_availability(
    summary: Mapping[str, Any], output_directory: Path
) -> FieldEvolutionAvailability:
    """Read one optional registered sequence without decoding PNG or VTU content."""

    raw_index = summary.get("field_evolution")
    if raw_index is None:
        return FieldEvolutionAvailability.legacy_missing()
    try:
        return FieldEvolutionAvailability.available(
            _read_sequence(raw_index, output_directory.resolve())
        )
    except (FieldEvolutionResultError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return FieldEvolutionAvailability.unavailable(str(exc))


def _read_sequence(raw_index: object, output: Path) -> FieldEvolutionSequence:
    index = _mapping(raw_index, "场演化摘要索引必须为 JSON object。")
    sequence_directory = _safe_path(_text(index, "directory"), output, output)
    if not sequence_directory.is_dir():
        raise FieldEvolutionResultError("场演化目录不存在。")
    manifest_path = _safe_path(_text(index, "manifest"), output, sequence_directory)
    history_path = _safe_path(
        _text(index, "position_history_csv"), output, sequence_directory
    )
    pvd_path = _safe_path(_text(index, "pvd"), output, sequence_directory)
    for path, label in (
        (manifest_path, "manifest.json"),
        (history_path, "position_history.csv"),
        (pvd_path, "field_evolution.pvd"),
    ):
        _require_nonempty_file(path, label)

    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(manifest_raw, "场演化 manifest 根对象必须为 JSON object。")
    if manifest.get("result_format") != FIELD_EVOLUTION_RESULT_FORMAT:
        raise FieldEvolutionResultError("场演化 manifest 的 result_format 无效。")

    frame_count = _strict_positive_int(manifest.get("frame_count"), "frame_count")
    index_count = _strict_positive_int(index.get("frame_count"), "摘要 frame_count")
    if index_count != frame_count:
        raise FieldEvolutionResultError("摘要与 manifest 的 frame_count 不一致。")
    if _strict_int(manifest.get("frame_index_start"), "frame_index_start") != 0:
        raise FieldEvolutionResultError("frame_index_start 必须为 0。")
    if _strict_int(manifest.get("frame_index_end"), "frame_index_end") != frame_count - 1:
        raise FieldEvolutionResultError("frame_index_end 与 frame_count 不一致。")

    sequence_coordinate = _sequence_coordinate(manifest.get("sequence_coordinate"))
    color_ranges = _color_ranges(manifest.get("color_ranges"))
    _field_locations(manifest.get("field_locations"))
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != frame_count:
        raise FieldEvolutionResultError("frames 数量与 frame_count 不一致。")

    frames: list[FieldEvolutionFrame] = []
    used_paths: set[Path] = set()
    for expected_index, raw_frame in enumerate(raw_frames):
        frame = _frame(raw_frame, expected_index, output, sequence_directory)
        if frame.png_path in used_paths or frame.vtu_path in used_paths:
            raise FieldEvolutionResultError("PNG 或 VTU 路径存在重复登记。")
        used_paths.update((frame.png_path, frame.vtu_path))
        frames.append(frame)
    if not frames[-1].is_final_unloaded_target:
        raise FieldEvolutionResultError("最后一帧未标记为最终卸载目标。")

    return FieldEvolutionSequence(
        manifest_path=manifest_path,
        position_history_csv_path=history_path,
        pvd_path=pvd_path,
        sequence_directory=sequence_directory,
        frame_count=frame_count,
        frames=tuple(frames),
        color_ranges=color_ranges,
        sequence_coordinate=sequence_coordinate,
    )


def _frame(
    raw: object, expected_index: int, output: Path, sequence_directory: Path
) -> FieldEvolutionFrame:
    frame = _mapping(raw, f"第 {expected_index} 帧必须为 JSON object。")
    frame_index = _strict_int(frame.get("frame_index"), "frame_index")
    if frame_index != expected_index:
        raise FieldEvolutionResultError("frame_index 必须从 0 连续增长。")
    pass_state = _text(frame, "pass_state")
    final_flag = frame.get("is_final_unloaded_target")
    if type(final_flag) is not bool:
        raise FieldEvolutionResultError("is_final_unloaded_target 必须为严格布尔值。")
    contact_ratio = _finite(frame.get("contact_ratio"), "contact_ratio")
    tolerance = 1.0e-12
    if contact_ratio < -tolerance or contact_ratio > 1.0 + tolerance:
        raise FieldEvolutionResultError("contact_ratio 超出 [0, 1]。")
    png_path = _safe_path(_text(frame, "png_path"), sequence_directory, output)
    vtu_path = _safe_path(_text(frame, "vtu_path"), sequence_directory, output)
    return FieldEvolutionFrame(
        frame_index=frame_index,
        position_id=_strict_int(frame.get("position_id"), "position_id"),
        motion_coordinate_m=_finite(frame.get("motion_coordinate_m"), "motion_coordinate_m"),
        wheel_lowest_point_x_m=_finite(
            frame.get("wheel_lowest_point_x_m"), "wheel_lowest_point_x_m"
        ),
        pass_state=pass_state,
        contact_ratio=min(1.0, max(0.0, contact_ratio)),
        Fx_N=_finite(frame.get("Fx_N"), "Fx_N"),
        Fy_N=_finite(frame.get("Fy_N"), "Fy_N"),
        accumulated_plastic_dissipation_J=_finite(
            frame.get("accumulated_plastic_dissipation_J"),
            "accumulated_plastic_dissipation_J",
        ),
        is_final_unloaded_target=final_flag,
        png_path=png_path,
        vtu_path=vtu_path,
    )


def _sequence_coordinate(raw: object) -> Mapping[str, Any]:
    value = _mapping(raw, "sequence_coordinate 结构无效。")
    if value.get("field") != "motion_coordinate_m" or value.get("unit") != "m":
        raise FieldEvolutionResultError("sequence_coordinate 字段或单位无效。")
    if value.get("meaning") != "wheel_motion_coordinate_not_time":
        raise FieldEvolutionResultError("sequence_coordinate 含义无效。")
    if type(value.get("pvd_timestep_uses_motion_coordinate")) is not bool:
        raise FieldEvolutionResultError("PVD 序列坐标标识必须为严格布尔值。")
    return dict(value)


def _color_ranges(raw: object) -> Mapping[str, tuple[float, float]]:
    ranges = _mapping(raw, "color_ranges 结构无效。")
    required = (
        "displacement_magnitude_m",
        "von_mises_stress_Pa",
        "equivalent_plastic_strain",
    )
    parsed: dict[str, tuple[float, float]] = {}
    for name in required:
        limits = _mapping(ranges.get(name), f"色标 {name} 结构无效。")
        minimum = _finite(limits.get("minimum"), f"{name}.minimum")
        maximum = _finite(limits.get("maximum"), f"{name}.maximum")
        if minimum > maximum:
            raise FieldEvolutionResultError(f"色标 {name} 的最小值大于最大值。")
        parsed[name] = (minimum, maximum)
    return parsed


def _field_locations(raw: object) -> None:
    locations = _mapping(raw, "field_locations 结构无效。")
    required = (
        "displacement_magnitude_m",
        "von_mises_stress_Pa",
        "equivalent_plastic_strain",
    )
    for name in required:
        if not isinstance(locations.get(name), str) or not locations[name].strip():
            raise FieldEvolutionResultError(f"场量位置 {name} 无效。")


def _mapping(raw: object, message: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise FieldEvolutionResultError(message)
    return raw


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FieldEvolutionResultError(f"{key} 必须为非空字符串。")
    return value


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise FieldEvolutionResultError(f"{name} 必须为严格整数。")
    return value


def _strict_positive_int(value: object, name: str) -> int:
    parsed = _strict_int(value, name)
    if parsed <= 0:
        raise FieldEvolutionResultError(f"{name} 必须为正整数。")
    return parsed


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldEvolutionResultError(f"{name} 必须为有限数值。")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FieldEvolutionResultError(f"{name} 必须为有限数值。")
    return parsed


def _safe_path(raw: str, base: Path, boundary: Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(boundary.resolve())
    except ValueError as exc:
        raise FieldEvolutionResultError(f"场演化路径超出正式结果目录：{raw}") from exc
    return resolved


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FieldEvolutionResultError(f"{label} 不存在。")
    if path.stat().st_size <= 0:
        raise FieldEvolutionResultError(f"{label} 为空。")


__all__ = [
    "FIELD_EVOLUTION_RESULT_FORMAT",
    "FieldEvolutionAvailability",
    "FieldEvolutionFrame",
    "FieldEvolutionResultError",
    "FieldEvolutionSequence",
    "read_field_evolution_availability",
]
