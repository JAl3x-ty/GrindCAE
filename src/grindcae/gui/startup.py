"""Failure-visible launcher around the public Tkinter GUI entry."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

from .runtime_paths import configure_runtime_environment, log_directory


def _write_startup_failure() -> tuple[Path, bool]:
    log_path = log_directory() / "grindcae_startup.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] GUI startup failure\n")
            traceback.print_exc(file=stream)
    except OSError:
        return log_path, False
    return log_path, True


def _show_startup_failure(log_path: Path, log_written: bool, error: Exception) -> None:
    log_message = (
        f"启动日志：{log_path}"
        if log_written
        else "启动日志写入失败；请使用 windows\\launch_grindcae.cmd 查看终端错误。"
    )
    message = (
        "GrindCAE 启动失败。\n\n"
        f"错误：{type(error).__name__}: {error}\n"
        "请运行 windows\\check_environment.cmd 检查项目 .venv 和依赖。\n"
        f"{log_message}"
    )
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "GrindCAE 启动失败", 0x10)
    except Exception:
        pass


def preflight_gui_import() -> int:
    """Check the public GUI entry and record its traceback without opening Tk."""

    try:
        configure_runtime_environment()
        from .app import main as gui_main
    except Exception:
        _write_startup_failure()
        return 1
    return 0 if callable(gui_main) else 1


def launch_gui() -> None:
    try:
        configure_runtime_environment()
        from .app import main

        main()
    except Exception as exc:
        log_path, log_written = _write_startup_failure()
        _show_startup_failure(log_path, log_written, exc)
        raise
