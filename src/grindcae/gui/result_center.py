"""State evaluation for the classified result center."""

from __future__ import annotations

from dataclasses import dataclass

from .field_evolution_result import (
    FieldEvolutionAvailability,
    FieldEvolutionFrame,
)
from .mechanism_field_evolution_result import (
    MechanismFieldEvolutionAvailability,
    MechanismFieldEvolutionFrame,
)
from .project import GrindCaeProject
from .result_catalog import (
    CategorizedAnalysisResult,
    ResultCatalog,
    ResultCatalogError,
    formal_result_records,
)


@dataclass(frozen=True, slots=True)
class ResultCenterEntry:
    result_id: str
    analysis_type: str
    display_name: str
    created_at: str
    status: str
    result: CategorizedAnalysisResult | None
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class ResultCenterState:
    entries: tuple[ResultCenterEntry, ...]
    selected_result_id: str | None

    def entry(self, result_id: str) -> ResultCenterEntry:
        for entry in self.entries:
            if entry.result_id == result_id:
                return entry
        raise KeyError(result_id)


class FieldEvolutionBrowserState:
    """Presentation-neutral current-frame selection for one optional sequence."""

    _PASS_STATE_LABELS = {
        "before_entry": "磨入前",
        "entry": "磨入",
        "full_contact": "满接触",
        "exit": "磨出",
        "after_exit": "磨出后",
    }

    def __init__(self) -> None:
        self.availability = FieldEvolutionAvailability.not_applicable()
        self.frame_index = 0

    def set_availability(self, availability: FieldEvolutionAvailability) -> None:
        self.availability = availability
        self.frame_index = 0

    @property
    def current_frame(self) -> FieldEvolutionFrame | None:
        sequence = self.availability.sequence
        if sequence is None or not sequence.frames:
            return None
        return sequence.frames[self.frame_index]

    @property
    def can_previous(self) -> bool:
        return self.current_frame is not None and self.frame_index > 0

    @property
    def can_next(self) -> bool:
        sequence = self.availability.sequence
        return sequence is not None and self.frame_index < sequence.frame_count - 1

    def select(self, frame_index: int | float) -> FieldEvolutionFrame | None:
        sequence = self.availability.sequence
        if sequence is None:
            self.frame_index = 0
            return None
        nearest = int(round(float(frame_index)))
        self.frame_index = min(sequence.frame_count - 1, max(0, nearest))
        return self.current_frame

    def previous(self) -> FieldEvolutionFrame | None:
        return self.select(self.frame_index - 1)

    def next(self) -> FieldEvolutionFrame | None:
        return self.select(self.frame_index + 1)

    def current_frame_display_lines(self) -> str:
        frame = self.current_frame
        sequence = self.availability.sequence
        if frame is None or sequence is None:
            return ""
        state_label = self._PASS_STATE_LABELS.get(frame.pass_state)
        if state_label is None:
            state_label = f"{frame.pass_state}（未识别状态）"
        return "\n".join((
            f"位置：{frame.frame_index + 1} / {sequence.frame_count}",
            f"运动坐标：{frame.motion_coordinate_m * 1e3:.3f} mm",
            f"砂轮最低点 x：{frame.wheel_lowest_point_x_m * 1e3:.3f} mm",
            f"磨削状态：{state_label}",
            f"接触比例：{frame.contact_ratio * 100.0:.1f} %",
            f"Fx：{frame.Fx_N:.2f} N",
            f"Fy：{frame.Fy_N:.2f} N",
            f"累计塑性耗散：{frame.accumulated_plastic_dissipation_J:.8g} J",
            f"最终卸载目标：{'是' if frame.is_final_unloaded_target else '否'}",
        ))


class MechanismFieldEvolutionBrowserState:
    """Shared position and route selection for mechanism comparison frames."""

    _PASS_STATE_LABELS = FieldEvolutionBrowserState._PASS_STATE_LABELS
    _MODES = ("baseline", "mechanism", "difference", "side_by_side")

    def __init__(self) -> None:
        self.availability = MechanismFieldEvolutionAvailability.not_applicable()
        self.frame_index = 0
        self.view_mode = "baseline"

    def set_availability(self, availability: MechanismFieldEvolutionAvailability) -> None:
        self.availability = availability
        self.frame_index = 0
        self.view_mode = "baseline"

    @property
    def current_frame(self) -> MechanismFieldEvolutionFrame | None:
        sequence = self.availability.sequence
        return None if sequence is None else sequence.frames[self.frame_index]

    @property
    def can_previous(self) -> bool:
        return self.current_frame is not None and self.frame_index > 0

    @property
    def can_next(self) -> bool:
        sequence = self.availability.sequence
        return sequence is not None and self.frame_index < sequence.frame_count - 1

    def select(self, index: int | float):
        sequence = self.availability.sequence
        if sequence is None:
            self.frame_index = 0
            return None
        self.frame_index = min(sequence.frame_count - 1, max(0, int(round(float(index)))))
        return self.current_frame

    def previous(self):
        return self.select(self.frame_index - 1)

    def next(self):
        return self.select(self.frame_index + 1)

    def set_view_mode(self, mode: str) -> None:
        if mode not in self._MODES:
            raise ValueError(f"unknown mechanism field view mode: {mode}")
        self.view_mode = mode

    @property
    def current_image_paths(self) -> tuple[object, ...]:
        frame = self.current_frame
        if frame is None:
            return ()
        if self.view_mode == "side_by_side":
            return (frame.baseline.png_path, frame.mechanism.png_path)
        return (getattr(frame, self.view_mode).png_path,)

    def current_frame_display_lines(self) -> str:
        frame = self.current_frame
        sequence = self.availability.sequence
        if frame is None or sequence is None:
            return ""
        state = self._PASS_STATE_LABELS.get(frame.pass_state, f"{frame.pass_state}（未识别状态）")
        return "\n".join((
            f"位置：{frame.frame_index + 1} / {sequence.frame_count}",
            f"运动坐标：{frame.motion_coordinate_m * 1e3:.3f} mm",
            f"砂轮最低点 x：{frame.wheel_lowest_point_x_m * 1e3:.3f} mm",
            f"磨削状态：{state}",
            f"接触比例：{frame.contact_ratio * 100.0:.1f} %",
            f"基准路线 Fx：{frame.baseline.Fx_N:.2f} N",
            f"基准路线 Fy：{frame.baseline.Fy_N:.2f} N",
            f"机制路线 Fx：{frame.mechanism.Fx_N:.2f} N",
            f"机制路线 Fy：{frame.mechanism.Fy_N:.2f} N",
            f"基准累计塑性耗散：{frame.baseline.accumulated_plastic_dissipation_J:.8g} J",
            f"机制累计塑性耗散：{frame.mechanism.accumulated_plastic_dissipation_J:.8g} J",
            f"最终卸载目标：{'是' if frame.is_final_unloaded_target else '否'}",
        ))


def build_result_center_state(
    project: GrindCaeProject,
    current_mode: str | None,
    current_input_fingerprint: str | None,
    *,
    catalog: ResultCatalog | None = None,
) -> ResultCenterState:
    reader = catalog or ResultCatalog()
    entries: list[ResultCenterEntry] = []
    for record in formal_result_records(project):
        try:
            result = reader.read(record)
        except (ResultCatalogError, OSError, ValueError) as exc:
            entries.append(ResultCenterEntry(
                record.result_id,
                record.analysis_type,
                _display_name(record.analysis_type),
                record.created_at,
                "文件异常",
                None,
                str(exc),
            ))
            continue
        if record.location_kind == "recovered_external":
            status = "只读外部结果"
        elif record.analysis_type != current_mode:
            status = "历史结果"
        elif not current_input_fingerprint:
            status = "历史结果"
        elif record.input_fingerprint != current_input_fingerprint:
            status = "输入已变化"
        else:
            status = "当前有效"
        entries.append(ResultCenterEntry(
            record.result_id,
            record.analysis_type,
            result.published.display_name,
            record.created_at,
            status,
            result,
        ))
    selected = next((entry.result_id for entry in entries if entry.status == "当前有效"), None)
    if selected is None and entries:
        selected = entries[0].result_id
    return ResultCenterState(tuple(entries), selected)


def _display_name(mode: str) -> str:
    try:
        from .analysis import analysis_mode_definition

        return analysis_mode_definition(mode).display_name
    except ValueError:
        return mode


__all__ = [
    "FieldEvolutionBrowserState",
    "MechanismFieldEvolutionBrowserState",
    "ResultCenterEntry",
    "ResultCenterState",
    "build_result_center_state",
]
