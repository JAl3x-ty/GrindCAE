"""Background-worker orchestration for workbench mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
import traceback
from typing import Any


class MeshControllerError(ValueError):
    """Mesh worker state error suitable for direct GUI display."""


@dataclass(frozen=True)
class MeshEvent:
    kind: str
    message: str
    result: Any = None
    log_detail: str | None = None


class MeshController:
    def __init__(self, events: Queue[MeshEvent] | None = None) -> None:
        self.events = events or Queue()
        self._lock = Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, provider: object, output_directory: str | Path) -> Thread:
        if not hasattr(provider, "publish"):
            raise MeshControllerError("网格生成器对象无效。")
        output = Path(output_directory).expanduser().resolve()
        with self._lock:
            if self._running:
                raise MeshControllerError("已有网格任务正在运行，请等待完成。")
            self._running = True
        thread = Thread(
            target=self._worker,
            args=(provider, output),
            name="grindcae-mesh-worker",
            daemon=True,
        )
        thread.start()
        return thread

    def _worker(self, provider: object, output: Path) -> None:
        self.events.put(MeshEvent("started", "后台三角形网格生成已启动。"))
        try:
            result = provider.publish(output)
            self.events.put(MeshEvent("success", "三角形网格已完整发布。", result))
        except Exception as exc:
            self.events.put(
                MeshEvent(
                    "error",
                    f"网格生成失败：{type(exc).__name__}: {exc}",
                    log_detail=traceback.format_exc(),
                )
            )
        finally:
            with self._lock:
                self._running = False
            self.events.put(MeshEvent("finished", "后台网格任务已结束。"))
