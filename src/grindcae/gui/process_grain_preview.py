"""Presentation-neutral process-page statistical grain preview workflow."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
import traceback
from typing import Callable
from PIL import Image

from grindcae.statistical_grain_load import (
    StatisticalGrainLoadCase,
    run_statistical_grain_load,
)
from grindcae.statistical_grain_load.workflow import ARTIFACT_FILENAMES

from .analysis import build_process_grain_preview_case
from .background_tasks import BackgroundTaskGate
from .project import GrindCaeProject


class ProcessGrainPreviewError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BuiltProcessGrainPreview:
    case: StatisticalGrainLoadCase
    normalized_input: dict[str, object]
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class ProcessGrainPreviewPublishedResult:
    output_directory: Path
    summary_path: Path
    image_path: Path
    artifact_paths: dict[str, Path]
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class ProcessGrainPreviewEvent:
    kind: str
    message: str
    result: ProcessGrainPreviewPublishedResult | None = None
    log_detail: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessGrainPreviewState:
    status: str
    message: str
    image_path: Path | None
    output_directory: Path | None


def process_grain_preview_state(
    project: GrindCaeProject, built: BuiltProcessGrainPreview | None
) -> ProcessGrainPreviewState:
    current = (
        project.current_process_grain_preview(built.input_fingerprint)
        if built is not None else None
    )
    if current is None:
        old = next(
            (
                record for record in reversed(project.results)
                if record.analysis_type == "process_grain_preview"
            ),
            None,
        )
        if old is not None:
            return ProcessGrainPreviewState(
                "stale",
                "磨粒预览已过期：砂轮、粒度或工艺输入已经变化。旧文件仍保留，请重新生成。",
                None,
                Path(old.output_directory),
            )
        return ProcessGrainPreviewState(
            "not_run",
            "尚未生成统计等效磨粒与载荷预览。",
            None,
            None,
        )
    image_raw = current.artifact_paths.get("mechanism_load_distribution_png")
    if not image_raw:
        return ProcessGrainPreviewState(
            "unavailable", "预览记录不可用：记录中缺少代表图路径。", None,
            Path(current.output_directory),
        )
    expected_artifact_keys = set(ARTIFACT_FILENAMES) - {"summary_json"}
    if set(current.artifact_paths) != expected_artifact_keys:
        return ProcessGrainPreviewState(
            "unavailable",
            "预览记录不可用：没有完整登记五个产物。",
            None,
            Path(current.output_directory),
        )
    image_path = Path(image_raw).expanduser().resolve()
    required_paths = [
        Path(current.summary_path).expanduser().resolve(),
        *(Path(value).expanduser().resolve() for value in current.artifact_paths.values()),
    ]
    try:
        if not all(path.is_file() and path.stat().st_size > 0 for path in required_paths):
            raise ValueError("文件缺失或为空")
        with Path(current.summary_path).open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict):
            raise ValueError("摘要根对象无效")
        resolved_path = Path(
            current.artifact_paths["resolved_grit_specification_json"]
        )
        with resolved_path.open(encoding="utf-8") as stream:
            resolved = json.load(stream)
        if not isinstance(resolved, dict):
            raise ValueError("粒度解析产物无效")
        for key in (
            "equivalent_grain_groups_csv",
            "mechanism_load_distribution_csv",
        ):
            with Path(current.artifact_paths[key]).open(
                encoding="utf-8", newline=""
            ) as stream:
                if not list(csv.DictReader(stream)):
                    raise ValueError(f"{ARTIFACT_FILENAMES[key]} 没有数据行")
        with Image.open(image_path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError("代表图不是 PNG")
    except (OSError, ValueError) as exc:
        return ProcessGrainPreviewState(
            "unavailable", f"预览记录不可用：{exc}", None,
            Path(current.output_directory),
        )
    return ProcessGrainPreviewState(
        "completed",
        "统计等效磨粒与载荷预览已完成。该图用于预处理检查，不代表逐颗磨粒高保真解析、实测粗糙度或材料标定结果。",
        image_path,
        Path(current.output_directory),
    )


class ProcessGrainPreviewBuilder:
    def build(self, project: GrindCaeProject) -> BuiltProcessGrainPreview:
        try:
            case = build_process_grain_preview_case(project)
        except ValueError as exc:
            raise ProcessGrainPreviewError(str(exc)) from exc
        normalized = case.to_dict()
        serialized = json.dumps(
            normalized, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )
        return BuiltProcessGrainPreview(
            case, normalized, hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        )


class ProcessGrainPreviewRunner:
    def __init__(
        self,
        events: Queue[ProcessGrainPreviewEvent] | None = None,
        *,
        gate: BackgroundTaskGate | None = None,
        run_workflow: Callable[..., object] = run_statistical_grain_load,
    ) -> None:
        self.events = events or Queue()
        self._gate = gate or BackgroundTaskGate()
        self._run_workflow = run_workflow
        self._lock = Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(
        self, built: BuiltProcessGrainPreview, output_directory: str | Path
    ) -> Thread:
        output = Path(output_directory).expanduser().resolve()
        if not self._gate.try_acquire("process_grain_preview"):
            raise ProcessGrainPreviewError("已有后台任务正在运行，请等待当前任务完成。")
        with self._lock:
            if self._running:
                self._gate.release("process_grain_preview")
                raise ProcessGrainPreviewError("统计磨粒预览正在运行。")
            self._running = True
        thread = Thread(
            target=self._worker,
            args=(built, output),
            name="grindcae-process-grain-preview-worker",
            daemon=True,
        )
        thread.start()
        return thread

    def _worker(self, built: BuiltProcessGrainPreview, output: Path) -> None:
        self.events.put(ProcessGrainPreviewEvent("started", "正在生成统计等效磨粒预览。"))
        try:
            _prediction, artifacts, summary = self._run_workflow(built.case, output)
            published = ProcessGrainPreviewPublishedResult(
                output_directory=output,
                summary_path=Path(artifacts["summary_json"]),
                image_path=Path(artifacts["mechanism_load_distribution_png"]),
                artifact_paths={key: Path(value) for key, value in artifacts.items()},
                summary=dict(summary),
            )
            self.events.put(ProcessGrainPreviewEvent(
                "success", "统计等效磨粒预览已生成。", result=published
            ))
        except Exception as exc:
            self.events.put(ProcessGrainPreviewEvent(
                "error", f"预览生成失败：{type(exc).__name__}: {exc}",
                log_detail=traceback.format_exc(),
            ))
        finally:
            with self._lock:
                self._running = False
            self._gate.release("process_grain_preview")
            self.events.put(ProcessGrainPreviewEvent("finished", "后台预览任务已结束。"))
