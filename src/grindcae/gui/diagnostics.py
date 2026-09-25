"""Read-only diagnostics retained for the phase 5B1 source launch layer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
import platform
import sys

from grindcae import __version__


DEPENDENCIES = (
    ("gmsh", "gmsh", "gmsh"),
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("Pillow", "PIL", "Pillow"),
    ("meshio", "meshio", "meshio"),
    ("scikit-fem", "skfem", "scikit-fem"),
)


@dataclass(frozen=True, slots=True)
class DiagnosticItem:
    label: str
    value: str
    passed: bool = True


def find_project_root(start: Path | None = None) -> Path:
    """Find the source checkout used by the Windows scripts and editable install."""

    location = (start or Path(__file__)).resolve()
    for candidate in (location, *location.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "grindcae"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


def _installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "未安装 metadata"


def _dependency_item(label: str, module_name: str, distribution: str) -> DiagnosticItem:
    try:
        module = import_module(module_name)
        value = getattr(module, "__version__", None) or _installed_version(distribution)
    except Exception as exc:
        return DiagnosticItem(label, f"导入失败：{type(exc).__name__}: {exc}", False)
    return DiagnosticItem(label, str(value), value != "未安装 metadata")


def _tk_item() -> DiagnosticItem:
    try:
        import tkinter

        interpreter = tkinter.Tcl()
        tcl_patch = interpreter.eval("info patchlevel")
        value = f"Tk {tkinter.TkVersion:g} / Tcl {tcl_patch}"
    except Exception as exc:
        return DiagnosticItem("Tk/Tcl 版本", f"不可用：{type(exc).__name__}: {exc}", False)
    return DiagnosticItem("Tk/Tcl 版本", value)


def _gui_entry_item() -> DiagnosticItem:
    try:
        module = import_module("grindcae.gui.app")
        entry = getattr(module, "GrindCaeApp")
    except Exception as exc:
        return DiagnosticItem("GUI 入口可导入", f"失败：{type(exc).__name__}: {exc}", False)
    return DiagnosticItem("GUI 入口可导入", f"是（{entry.__module__}.{entry.__name__}）")


def collect_diagnostics() -> tuple[DiagnosticItem, ...]:
    """Collect import and version checks without creating a Tk main window."""

    metadata_version = _installed_version("grindcae-literature-kernel")
    items = [
        DiagnosticItem("项目根目录", str(find_project_root())),
        DiagnosticItem("当前工作目录", str(Path.cwd().resolve())),
        DiagnosticItem("Python 可执行文件", sys.executable),
        DiagnosticItem("Python 版本", platform.python_version()),
        DiagnosticItem("GrindCAE 源码版本", __version__),
        DiagnosticItem(
            "installed metadata 版本",
            metadata_version,
            metadata_version == __version__,
        ),
        _tk_item(),
    ]
    items.extend(
        _dependency_item(label, module_name, distribution)
        for label, module_name, distribution in DEPENDENCIES
    )
    items.append(_gui_entry_item())
    return tuple(items)


def main() -> int:
    print(f"GrindCAE {__version__} 环境诊断")
    print("=" * 64)
    items = collect_diagnostics()
    for item in items:
        state = "PASS" if item.passed else "FAIL"
        print(f"[{state}] {item.label}: {item.value}")
    passed = all(item.passed for item in items)
    print("=" * 64)
    print(f"最终结果: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
