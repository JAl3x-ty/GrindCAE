"""Frozen-runtime diagnostics for the phase 5B2 portable distribution."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata, resources
from pathlib import Path
import platform
import sys
import uuid

from grindcae import __version__
from grindcae.gmsh_runtime import initialize_gmsh

from .runtime_paths import (
    application_directory,
    cache_directory,
    configure_runtime_environment,
    default_scan_output_directory,
    default_single_output_directory,
    is_frozen,
    log_directory,
    matplotlib_config_directory,
    user_data_directory,
    user_documents_directory,
)


DEPENDENCIES = (
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("meshio", "meshio", "meshio"),
    ("scikit-fem", "skfem", "scikit-fem"),
)


@dataclass(frozen=True, slots=True)
class PortableDiagnosticItem:
    label: str
    value: str
    passed: bool = True


def _installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "missing metadata"


def _dependency_item(label: str, module_name: str, distribution: str) -> PortableDiagnosticItem:
    try:
        module = import_module(module_name)
        value = getattr(module, "__version__", None) or _installed_version(distribution)
    except Exception as exc:
        return PortableDiagnosticItem(label, f"import failed: {type(exc).__name__}: {exc}", False)
    return PortableDiagnosticItem(label, str(value), value != "missing metadata")


def _writable_item(label: str, directory: Path) -> PortableDiagnosticItem:
    probe = directory / f".grindcae-write-probe-{uuid.uuid4().hex}.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"GrindCAE portable write probe\n")
        probe.unlink()
    except OSError as exc:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return PortableDiagnosticItem(label, f"{directory} ({type(exc).__name__}: {exc})", False)
    return PortableDiagnosticItem(label, str(directory))


def _tk_item() -> PortableDiagnosticItem:
    root = None
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.update_idletasks()
        tcl_patch = root.tk.call("info", "patchlevel")
        value = f"Tk {tkinter.TkVersion:g} / Tcl {tcl_patch}; hidden root create/destroy PASS"
    except Exception as exc:
        return PortableDiagnosticItem("Tk/Tcl runtime", f"unavailable: {type(exc).__name__}: {exc}", False)
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
    return PortableDiagnosticItem("Tk/Tcl runtime", value)


def _icon_item() -> PortableDiagnosticItem:
    try:
        icon = resources.files("grindcae.gui.assets").joinpath("grindcae.ico")
        with resources.as_file(icon) as path:
            size = path.stat().st_size
            value = f"{path} ({size} bytes)"
    except Exception as exc:
        return PortableDiagnosticItem("Packaged icon", f"unavailable: {type(exc).__name__}: {exc}", False)
    return PortableDiagnosticItem("Packaged icon", value, size > 0)


def _gmsh_item() -> PortableDiagnosticItem:
    initialized_here = False
    try:
        gmsh = import_module("gmsh")
        library_path = Path(str(getattr(gmsh, "libpath", ""))).resolve()
        if not library_path.is_file() or library_path.stat().st_size <= 0:
            raise RuntimeError(f"Gmsh runtime DLL is missing: {library_path}")
        if not gmsh.isInitialized():
            initialize_gmsh(gmsh, ["GrindCAE-Diagnostics", "-nopopup"])
            initialized_here = True
        runtime_version = gmsh.option.getString("General.Version")
        api_version = getattr(gmsh, "__version__", "unknown")
        value = f"API {api_version}; runtime {runtime_version}; DLL {library_path}"
    except Exception as exc:
        return PortableDiagnosticItem("Gmsh initialize/finalize", f"failed: {type(exc).__name__}: {exc}", False)
    finally:
        if initialized_here:
            try:
                gmsh.finalize()
            except Exception:
                pass
    return PortableDiagnosticItem("Gmsh initialize/finalize", value)


def _matplotlib_item() -> PortableDiagnosticItem:
    try:
        matplotlib = import_module("matplotlib")
        data_path = Path(matplotlib.get_data_path()).resolve()
        config_path = Path(matplotlib.get_configdir()).resolve()
        passed = data_path.is_dir() and (not is_frozen() or config_path == matplotlib_config_directory().resolve())
        value = f"data={data_path}; config={config_path}; backend={matplotlib.get_backend()}"
    except Exception as exc:
        return PortableDiagnosticItem("Matplotlib runtime paths", f"failed: {type(exc).__name__}: {exc}", False)
    return PortableDiagnosticItem("Matplotlib runtime paths", value, passed)


def _gui_entry_item() -> PortableDiagnosticItem:
    try:
        module = import_module("grindcae.gui.app")
        entry = getattr(module, "GrindCaeApp")
    except Exception as exc:
        return PortableDiagnosticItem("GUI entry import", f"failed: {type(exc).__name__}: {exc}", False)
    return PortableDiagnosticItem("GUI entry import", f"{entry.__module__}.{entry.__name__}")


def collect_portable_diagnostics() -> tuple[PortableDiagnosticItem, ...]:
    """Check bundled resources and writable locations without running FEM."""

    configure_runtime_environment()
    metadata_version = _installed_version("grindcae-literature-kernel")
    items = [
        PortableDiagnosticItem("GrindCAE source version", __version__),
        PortableDiagnosticItem("GrindCAE installed metadata", metadata_version, metadata_version == __version__),
        PortableDiagnosticItem("Windows version", platform.platform()),
        PortableDiagnosticItem("CPU architecture", platform.machine(), platform.machine().upper() in {"AMD64", "X86_64"}),
        PortableDiagnosticItem("Python executable", sys.executable),
        PortableDiagnosticItem("Python version", platform.python_version(), sys.version_info[:2] == (3, 11)),
        PortableDiagnosticItem("Frozen runtime", str(is_frozen())),
        PortableDiagnosticItem("Application directory", str(application_directory())),
        PortableDiagnosticItem("User Documents", str(user_documents_directory())),
        _writable_item("User data writable", user_data_directory()),
        _writable_item("Single-position output writable", default_single_output_directory()),
        _writable_item("Pass-scan output writable", default_scan_output_directory()),
        _writable_item("Log directory writable", log_directory()),
        _writable_item("Cache directory writable", cache_directory()),
        _writable_item("Matplotlib directory writable", matplotlib_config_directory()),
        _tk_item(),
        _icon_item(),
        _gmsh_item(),
    ]
    items.extend(_dependency_item(*dependency) for dependency in DEPENDENCIES)
    items.extend((_matplotlib_item(), _gui_entry_item()))
    return tuple(items)


def print_portable_diagnostics(items: tuple[PortableDiagnosticItem, ...]) -> bool:
    print(f"GrindCAE {__version__} portable Windows diagnostics")
    print("=" * 72)
    for item in items:
        state = "PASS" if item.passed else "FAIL"
        print(f"[{state}] {item.label}: {item.value}")
    passed = all(item.passed for item in items)
    print("=" * 72)
    print(f"FINAL RESULT: {'PASS' if passed else 'FAIL'}")
    return passed
