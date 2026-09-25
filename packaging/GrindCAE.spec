# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import tomllib

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


repo_root = Path(SPECPATH).resolve().parent
source_root = repo_root / "src"
icon_path = source_root / "grindcae" / "gui" / "assets" / "grindcae.ico"
hook_root = repo_root / "packaging" / "hooks"
with (repo_root / "pyproject.toml").open("rb") as stream:
    version = tomllib.load(stream)["project"]["version"]

common_datas = collect_data_files("grindcae.gui.assets")
common_datas += collect_data_files("grindcae.gui.portable_examples")
common_datas += copy_metadata("grindcae-literature-kernel")
common_datas += [(str(repo_root / "examples" / "literature_history_demo.json"), "examples")]
common_hiddenimports = [
    "grindcae.literature_history",
    "grindcae380.dispatch",
    "grindcae.gui.app",
    "grindcae.gui.portable_smoke",
    "matplotlib.backends.backend_agg",
    "matplotlib.image",
    "matplotlib.pyplot",
    "matplotlib.tri",
    "meshio.gmsh",
    "meshio.vtu",
    "skfem.io.meshio",
    "skfem.models.elasticity",
]


def analysis(entry):
    return Analysis(
        [str(repo_root / "packaging" / entry)],
        pathex=[str(source_root)],
        binaries=[],
        datas=common_datas,
        hiddenimports=common_hiddenimports,
        hookspath=[str(hook_root)],
        hooksconfig={},
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
        optimize=0,
    )


gui_analysis = analysis("gui_entry.py")
gui_pyz = PYZ(gui_analysis.pure)
gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="GrindCAE",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
)

diagnostics_analysis = analysis("diagnostics_entry.py")
diagnostics_pyz = PYZ(diagnostics_analysis.pure)
diagnostics_exe = EXE(
    diagnostics_pyz,
    diagnostics_analysis.scripts,
    [],
    exclude_binaries=True,
    name="GrindCAE-Diagnostics",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path),
)

collect = COLLECT(
    gui_exe,
    diagnostics_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    diagnostics_analysis.binaries,
    diagnostics_analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=f"GrindCAE-{version}-Windows-x64",
)
