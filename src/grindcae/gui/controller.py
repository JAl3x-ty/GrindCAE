"""Thread-safe GUI orchestration over public 4C3B, 4C4, and 7A.3 APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
import tempfile
from threading import Lock, Thread
import traceback
from typing import Callable

from grindcae.evolved_fem import run_evolved_fem_case
from grindcae.mechanism_history_pass import run_mechanism_elastoplastic_history_pass
from grindcae.pass_scan import (
    aggregate_artifact_paths,
    export_pass_scan,
    publish_pass_scan_artifacts,
    run_pass_scan,
)

from .form import (
    MECHANISM_HISTORY_MODE,
    GuiForm,
    GuiInputError,
    SCAN_MODE,
    SINGLE_MODE,
)
from .results import GuiPublishedResult, read_published_result


@dataclass(frozen=True, slots=True)
class GuiEvent:
    kind: str
    message: str
    current: int | None = None
    total: int | None = None
    result: GuiPublishedResult | None = None
    log_detail: str | None = None


GuiRunResult = GuiPublishedResult


@dataclass(frozen=True, slots=True)
class GuiBackend:
    run_single: Callable[..., object] = run_evolved_fem_case
    run_scan: Callable[..., object] = run_pass_scan
    export_scan: Callable[..., object] = export_pass_scan
    publish_scan: Callable[..., object] = publish_pass_scan_artifacts
    run_mechanism_history: Callable[..., object] = (
        run_mechanism_elastoplastic_history_pass
    )
    read_result: Callable[..., GuiPublishedResult] = read_published_result


class GuiController:
    """Own one background worker and publish immutable events to a queue."""

    def __init__(
        self,
        events: Queue[GuiEvent] | None = None,
        *,
        backend: GuiBackend | None = None,
    ) -> None:
        self.events: Queue[GuiEvent] = events or Queue()
        self._backend = backend or GuiBackend()
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def validate(self, form: GuiForm) -> object:
        if not isinstance(form, GuiForm):
            raise GuiInputError("表单对象无效。")
        if not form.output_directory.strip():
            raise GuiInputError("请选择输出目录。")
        output = Path(form.output_directory).expanduser().resolve()
        if output.exists() and not output.is_dir():
            raise GuiInputError(f"输出路径不是文件夹：{output}")
        return form.build_case()

    def start(self, form: GuiForm) -> Thread:
        case = self.validate(form)
        output = Path(form.output_directory).expanduser().resolve()
        with self._lock:
            if self._running:
                raise GuiInputError("已有计算正在运行，请等待当前任务完成。")
            self._running = True
        self.events.put(GuiEvent("validated", "参数已通过现有严格 Schema 校验。"))
        thread = Thread(
            target=self._worker,
            args=(form.mode, case, output),
            name="grindcae-gui-worker",
            daemon=True,
        )
        # 中文导读：耗时计算在后台线程运行；后台通过队列通知主线程更新 Tk 界面。
        self._thread = thread
        thread.start()
        return thread

    def _worker(self, mode: str, case: object, output: Path) -> None:
        self.events.put(GuiEvent("started", "后台计算已启动。"))
        try:
            if mode == SINGLE_MODE:
                self.events.put(GuiEvent("progress", "运行单位置线弹性计算并事务化发布结果。", 0, 1))
                self._backend.run_single(case, output, "auto")
                self.events.put(GuiEvent("progress", "单位置线弹性计算与结果发布完成。", 1, 1))
            elif mode == SCAN_MODE:
                with tempfile.TemporaryDirectory(prefix="grindcae-gui-pass-scan-") as temp_dir:
                    workspace = Path(temp_dir)

                    def progress(row, total: int) -> None:
                        self.events.put(
                            GuiEvent(
                                "progress",
                                f"扫描位置 {row.scan_index + 1}/{total}："
                                f"state={row.pass_state}，contact_ratio={row.contact_ratio:.6g}",
                                row.scan_index + 1,
                                total,
                            )
                        )

                    result = self._backend.run_scan(case, workspace / "solve", progress=progress)
                    total = len(result.history)
                    self.events.put(GuiEvent("progress", "导出扫描总览和代表性快照。", total, total))
                    temporary, _ = self._backend.export_scan(
                        result,
                        workspace / "artifacts",
                        output,
                        "auto",
                    )
                    self._backend.publish_scan(temporary, aggregate_artifact_paths(output))
            elif mode == MECHANISM_HISTORY_MODE:
                self.events.put(
                    GuiEvent(
                        "progress",
                        "运行机制化弹塑性完整单程并事务化发布结果。",
                        0,
                        1,
                    )
                )
                self._backend.run_mechanism_history(case, output)
                self.events.put(
                    GuiEvent(
                        "progress",
                        "机制化弹塑性完整单程计算与结果发布完成。",
                        1,
                        1,
                    )
                )
            else:
                raise GuiInputError("计算模式无效。")
            published = self._backend.read_result(mode, output)
            self.events.put(GuiEvent("success", "计算成功，结果已完整发布。", result=published))
        except Exception as exc:
            self.events.put(
                GuiEvent(
                    "error",
                    f"计算失败：{type(exc).__name__}: {exc}",
                    log_detail=traceback.format_exc(),
                )
            )
        finally:
            with self._lock:
                self._running = False
            self.events.put(GuiEvent("finished", "后台任务已结束。"))
