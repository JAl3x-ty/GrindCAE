"""Centralized source and frozen-runtime paths for the Windows GUI."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import uuid


SINGLE_OUTPUT_RELATIVE_TEXT = "outputs/gui_single_position"
SCAN_OUTPUT_RELATIVE_TEXT = "outputs/gui_pass_scan"
MECHANISM_HISTORY_OUTPUT_RELATIVE_TEXT = "outputs/gui_mechanism_history"
SINGLE_OUTPUT_RELATIVE = Path(SINGLE_OUTPUT_RELATIVE_TEXT)
SCAN_OUTPUT_RELATIVE = Path(SCAN_OUTPUT_RELATIVE_TEXT)
MECHANISM_HISTORY_OUTPUT_RELATIVE = Path(MECHANISM_HISTORY_OUTPUT_RELATIVE_TEXT)
DOCUMENTS_FOLDER_ID = "FDD39AD0-238F-46AF-ADB4-6C85480369C7"


class _Guid(ctypes.Structure):
    _fields_ = (
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    )


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _source_project_root(start: Path | None = None) -> Path:
    location = (start or Path(__file__)).resolve()
    for candidate in (location, *location.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "grindcae"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


def application_directory() -> Path:
    """Return the executable directory when frozen and the checkout root otherwise."""

    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _source_project_root()


def _known_folder_directory(folder_id: str) -> Path:
    if os.name != "nt":
        raise OSError("Windows Known Folder API is unavailable")
    identifier = uuid.UUID(folder_id)
    raw = identifier.bytes_le
    guid = _Guid(
        int.from_bytes(raw[0:4], "little"),
        int.from_bytes(raw[4:6], "little"),
        int.from_bytes(raw[6:8], "little"),
        (ctypes.c_ubyte * 8)(*raw[8:]),
    )
    output = ctypes.c_wchar_p()
    shell32 = ctypes.windll.shell32
    shell32.SHGetKnownFolderPath.argtypes = (
        ctypes.POINTER(_Guid),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    result = shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(output))
    if result != 0 or not output.value:
        raise OSError(f"SHGetKnownFolderPath failed with HRESULT 0x{result & 0xFFFFFFFF:08X}")
    try:
        return Path(output.value).resolve()
    finally:
        ctypes.windll.ole32.CoTaskMemFree(output)


def user_documents_directory() -> Path:
    try:
        return _known_folder_directory(DOCUMENTS_FOLDER_ID)
    except (AttributeError, OSError, ValueError):
        profile = os.environ.get("USERPROFILE")
        return ((Path(profile) if profile else Path.home()) / "Documents").resolve()


def _local_app_data_directory() -> Path:
    configured = os.environ.get("LOCALAPPDATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "AppData" / "Local").resolve()


def user_data_directory() -> Path:
    return user_documents_directory() / "GrindCAE"


def application_state_directory() -> Path:
    if is_frozen():
        return _local_app_data_directory() / "GrindCAE" / "state"
    return application_directory() / ".cache" / "grindcae" / "state"


def recovery_directory() -> Path:
    if is_frozen():
        return _local_app_data_directory() / "GrindCAE" / "recovery"
    return application_directory() / ".cache" / "grindcae" / "recovery"


def default_single_output_directory() -> Path:
    if is_frozen():
        return user_data_directory() / "outputs" / "single_position"
    return application_directory() / SINGLE_OUTPUT_RELATIVE


def default_scan_output_directory() -> Path:
    if is_frozen():
        return user_data_directory() / "outputs" / "pass_scan"
    return application_directory() / SCAN_OUTPUT_RELATIVE


def default_mechanism_history_output_directory() -> Path:
    if is_frozen():
        return user_data_directory() / "outputs" / "mechanism_history"
    return application_directory() / MECHANISM_HISTORY_OUTPUT_RELATIVE


def default_single_output_text() -> str:
    return str(default_single_output_directory()) if is_frozen() else SINGLE_OUTPUT_RELATIVE_TEXT


def default_scan_output_text() -> str:
    return str(default_scan_output_directory()) if is_frozen() else SCAN_OUTPUT_RELATIVE_TEXT


def default_mechanism_history_output_text() -> str:
    if is_frozen():
        return str(default_mechanism_history_output_directory())
    return MECHANISM_HISTORY_OUTPUT_RELATIVE_TEXT


def mechanism_history_example_path() -> Path:
    """Return the single source-of-truth 7A.3 example in a source checkout."""

    path = application_directory() / "examples" / "fixed_mesh_statistical_mechanism_history_pass.json"
    if not path.is_file():
        raise FileNotFoundError(
            "当前安装未包含机制化完整单程示例；Windows 便携包需要在后续版本重建。"
        )
    return path.resolve()


def log_directory() -> Path:
    if is_frozen():
        return _local_app_data_directory() / "GrindCAE" / "logs"
    return application_directory() / "logs"


def cache_directory() -> Path:
    if is_frozen():
        return _local_app_data_directory() / "GrindCAE" / "cache"
    return application_directory() / ".cache" / "grindcae"


def matplotlib_config_directory() -> Path:
    if is_frozen():
        return _local_app_data_directory() / "GrindCAE" / "matplotlib"
    configured = os.environ.get("MPLCONFIGDIR")
    return Path(configured).expanduser().resolve() if configured else cache_directory() / "matplotlib"


def configure_runtime_environment(*, create_directories: bool = True) -> None:
    """Configure writable frozen-runtime state before Matplotlib is imported."""

    if not is_frozen():
        return
    directories = (
        user_data_directory(),
        default_single_output_directory(),
        default_scan_output_directory(),
        default_mechanism_history_output_directory(),
        log_directory(),
        cache_directory(),
        matplotlib_config_directory(),
        application_state_directory(),
        recovery_directory(),
    )
    if create_directories:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config_directory())
