"""One-worker dispatcher over the five existing public calculation APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
import tempfile
from threading import Lock, Thread
import time
import traceback
from typing import Callable
import uuid

from grindcae.evolved_fem import run_evolved_fem_case
from grindcae.history_pass import run_fixed_mesh_elastoplastic_history_pass
from grindcae.mechanism_history_pass import run_mechanism_elastoplastic_history_pass
from grindcae.pass_scan import aggregate_artifact_paths, export_pass_scan, publish_pass_scan_artifacts, run_pass_scan
from grindcae.single_position_elastoplastic import run_single_position_elastoplastic
from grindcae.single_grain_contact import run_single_grain_contact

from .analysis import BuiltAnalysisInput
from .analysis_results import AnalysisPublishedResult, AnalysisResultAdapter
from .background_tasks import BackgroundTaskGate


class AnalysisRunError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    kind: str
    message: str
    current: int | None = None
    total: int | None = None
    result: AnalysisPublishedResult | None = None
    log_detail: str | None = None
    elapsed_seconds: float | None = None
    remaining_seconds: float | None = None
    phase: str | None = None


def _run_linear_single(case: object, output: Path, progress=None) -> None:
    run_evolved_fem_case(case, output, "auto")


def _run_linear_pass(case: object, output: Path, progress=None) -> None:
    with tempfile.TemporaryDirectory(prefix="grindcae-analysis-scan-") as temp_dir:
        workspace = Path(temp_dir)
        result = run_pass_scan(case, workspace / "solve", progress=progress)
        temporary, _ = export_pass_scan(result, workspace / "artifacts", output, "auto")
        publish_pass_scan_artifacts(temporary, aggregate_artifact_paths(output))


def _run_plastic_single(case: object, output: Path, progress=None) -> None:
    run_single_position_elastoplastic(case, output)


def _run_plastic_pass(case: object, output: Path, progress=None) -> None:
    run_fixed_mesh_elastoplastic_history_pass(case, output, progress=progress)


def _run_mechanism_pass(case: object, output: Path, progress=None) -> None:
    run_mechanism_elastoplastic_history_pass(case, output, progress=progress)


def _run_single_grain(case: object, output: Path, progress=None) -> None:
    run_single_grain_contact(case, output, progress=progress)


def _reduced_pilot_case(built: BuiltAnalysisInput) -> object:
    mode = built.definition.mode_id
    payload = built.case.to_dict()
    if mode == "literature_elastoplastic_single_pass":
        from grindcae.literature_history import LiteratureHistoryCase
        payload["history_template"]["moving_load_pass"]["pass"]["base_position_count"] = 5
        return LiteratureHistoryCase.from_mapping(payload)
    if mode == "linear_elastic_single_pass":
        from grindcae.pass_scan import PassScanCase
        payload["scan"]["base_position_count"] = 5
        return PassScanCase.from_mapping(payload)
    if mode == "elastoplastic_single_pass":
        from grindcae.history_pass import FixedMeshElastoplasticHistoryPassCase
        payload["moving_load_pass"]["pass"]["base_position_count"] = 5
        return FixedMeshElastoplasticHistoryPassCase.from_mapping(payload)
    if mode == "mechanism_elastoplastic_single_pass":
        from grindcae.mechanism_history_pass import MechanismHistoryPassCase
        payload["baseline_history_pass"]["moving_load_pass"]["pass"]["base_position_count"] = 5
        return MechanismHistoryPassCase.from_mapping(payload)
    return built.case


def _run_pilot(
    built: BuiltAnalysisInput,
    workspace: Path,
    progress: Callable[[object, int, str | None], None],
    runner: Callable[..., None],
) -> None:
    runner(
        _reduced_pilot_case(built),
        workspace / "pilot-output",
        progress=progress,
    )


def _read_result(mode: str, output: Path) -> AnalysisPublishedResult:
    return AnalysisResultAdapter().read(mode, output)


@dataclass(frozen=True, slots=True)
class AnalysisBackend:
    run_literature_elastoplastic_single_pass: Callable[..., object] | None = None
    run_linear_elastic_single_position: Callable[..., object] | None = None
    run_linear_elastic_single_pass: Callable[..., object] | None = None
    run_elastoplastic_single_position: Callable[..., object] | None = None
    run_elastoplastic_single_pass: Callable[..., object] | None = None
    run_mechanism_elastoplastic_single_pass: Callable[..., object] | None = None
    run_single_grain_high_fidelity: Callable[..., object] | None = None
    read_result: Callable[[str, Path], AnalysisPublishedResult] = _read_result

    def runner(self, mode: str) -> Callable[..., object]:
        if mode == "literature_elastoplastic_single_pass":
            from grindcae.literature_history import run_literature_history
            return self.run_literature_elastoplastic_single_pass or run_literature_history
        if mode == "literature_grinding_force":
            from .literature_integration import run_literature
            return run_literature
        values = {
            "linear_elastic_single_position": self.run_linear_elastic_single_position or _run_linear_single,
            "linear_elastic_single_pass": self.run_linear_elastic_single_pass or _run_linear_pass,
            "elastoplastic_single_position": self.run_elastoplastic_single_position or _run_plastic_single,
            "elastoplastic_single_pass": self.run_elastoplastic_single_pass or _run_plastic_pass,
            "mechanism_elastoplastic_single_pass": self.run_mechanism_elastoplastic_single_pass or _run_mechanism_pass,
            "single_grain_high_fidelity": self.run_single_grain_high_fidelity or _run_single_grain,
        }
        try:
            return values[mode]
        except KeyError as exc:
            raise AnalysisRunError(f"未知分析任务：{mode}") from exc


class AnalysisRunner:
    def __init__(
        self,
        events: Queue[AnalysisEvent] | None = None,
        *,
        backend: AnalysisBackend | None = None,
        gate: BackgroundTaskGate | None = None,
        clock: Callable[[], float] = time.monotonic,
        pilot_runner: Callable[..., None] = _run_pilot,
    ) -> None:
        self.events = events or Queue()
        self._backend = backend or AnalysisBackend()
        self._gate = gate or BackgroundTaskGate()
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None
        self._clock = clock
        self._pilot_runner = pilot_runner

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(
        self,
        built: BuiltAnalysisInput,
        output_directory: str | Path,
        *,
        run_pilot: bool = False,
    ) -> Thread:
        if not isinstance(built, BuiltAnalysisInput):
            raise AnalysisRunError("分析输入无效。")
        output = Path(output_directory).expanduser().resolve()
        if output.exists() and not output.is_dir():
            raise AnalysisRunError(f"输出路径不是文件夹：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        probe = output.parent / f".grindcae-write-probe-{uuid.uuid4().hex}.tmp"
        try:
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError as exc:
            raise AnalysisRunError(f"输出目录不可写：{output.parent}") from exc
        if not self._gate.try_acquire("analysis"):
            raise AnalysisRunError("已有后台任务正在运行，请等待当前任务完成。")
        with self._lock:
            if self._running:
                self._gate.release("analysis")
                raise AnalysisRunError("已有分析正在运行，请等待当前任务完成。")
            self._running = True
        self.events.put(AnalysisEvent("validated", "严格求解输入已通过校验。"))
        thread = Thread(target=self._worker, args=(built, output, run_pilot), name="grindcae-analysis-worker", daemon=True)
        self._thread = thread
        thread.start()
        return thread

    def _worker(self, built: BuiltAnalysisInput, output: Path, run_pilot: bool) -> None:
        definition = built.definition
        self.events.put(AnalysisEvent("started", f"正在运行：{definition.display_name}"))
        try:
            runner = self._backend.runner(definition.mode_id)
            if run_pilot:
                pilot_started_at = self._clock()

                def pilot_progress(
                    value, total: int, message: str | None = None
                ) -> None:
                    current = (
                        int(value.scan_index) + 1
                        if hasattr(value, "scan_index")
                        else int(value)
                    )
                    elapsed = max(0.0, self._clock() - pilot_started_at)
                    remaining = (
                        elapsed / current * (total - current)
                        if current > 0
                        else None
                    )
                    self.events.put(
                        AnalysisEvent(
                            "progress",
                            message or f"短程试算已完成位置 {current}/{total}",
                            current,
                            total,
                            elapsed_seconds=elapsed,
                            remaining_seconds=remaining,
                            phase="pilot",
                        )
                    )

                with tempfile.TemporaryDirectory(prefix="grindcae-analysis-pilot-") as pilot:
                    self._pilot_runner(
                        built,
                        Path(pilot),
                        pilot_progress,
                        runner,
                    )
                self.events.put(AnalysisEvent("pilot_success", "短程试算通过，开始正式计算。"))
            started_at = self._clock()
            progress_callback = None
            if definition.process_scope in {"complete_single_pass", "single_grain_trajectory"}:
                def progress(value, total: int, message: str | None = None) -> None:
                    current = int(value.scan_index) + 1 if hasattr(value, "scan_index") else int(value)
                    elapsed = max(0.0, self._clock() - started_at)
                    remaining = elapsed / current * (total - current) if current > 0 else None
                    self.events.put(AnalysisEvent(
                        "progress",
                        message or f"已完成位置 {current}/{total}",
                        current,
                        total,
                        elapsed_seconds=elapsed,
                        remaining_seconds=remaining,
                        phase="formal",
                    ))
                progress_callback = progress
            else:
                self.events.put(AnalysisEvent("stage", "正在运行计算。"))
            runner(built.case, output, progress=progress_callback)
            self.events.put(AnalysisEvent("stage", "正在验证已发布结果。"))
            result = self._backend.read_result(definition.mode_id, output)
            self.events.put(AnalysisEvent("success", "计算成功，真实结果已验证。", result=result))
        except Exception as exc:
            self.events.put(AnalysisEvent("error", f"计算失败：{type(exc).__name__}: {exc}", log_detail=traceback.format_exc()))
        finally:
            with self._lock:
                self._running = False
            self._gate.release("analysis")
            self.events.put(AnalysisEvent("finished", "后台分析任务已结束。"))
