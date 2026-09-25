"""Generate portable release metadata without exposing developer-private state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys

from grindcae import __version__


RUNTIME_DISTRIBUTIONS = (
    ("gmsh", "GPL-2.0-or-later", "https://gmsh.info/"),
    ("numpy", "BSD-3-Clause", "https://numpy.org/"),
    ("scipy", "BSD-3-Clause and bundled component licenses", "https://scipy.org/"),
    ("matplotlib", "PSF-based license and bundled component licenses", "https://matplotlib.org/"),
    ("meshio", "MIT", "https://github.com/nschloe/meshio"),
    ("scikit-fem", "MIT", "https://github.com/kinnala/scikit-fem"),
    ("pillow", "HPND", "https://python-pillow.org/"),
    ("contourpy", "BSD-3-Clause", "https://contourpy.readthedocs.io/"),
    ("cycler", "BSD-3-Clause", "https://matplotlib.org/cycler/"),
    ("fonttools", "MIT", "https://fonttools.readthedocs.io/"),
    ("kiwisolver", "BSD-3-Clause", "https://github.com/nucleic/kiwi"),
    ("packaging", "Apache-2.0 or BSD-2-Clause", "https://packaging.pypa.io/"),
    ("pyparsing", "MIT", "https://pyparsing-docs.readthedocs.io/"),
    ("python-dateutil", "Apache-2.0 or BSD-3-Clause", "https://dateutil.readthedocs.io/"),
    ("six", "MIT", "https://github.com/benjaminp/six"),
)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _license_files(distribution: str) -> tuple[Path, ...]:
    try:
        dist = metadata.distribution(distribution)
    except metadata.PackageNotFoundError:
        return ()
    root = Path(dist._path)
    matches = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "license" in name or "copying" in name or "notice" in name:
            matches.append(path)
    return tuple(sorted(matches))


def prepare_distribution(destination: Path, repository_root: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "VERSION.txt").write_text(
        f"GrindCAE {__version__}\nWindows x64 portable onedir\n",
        encoding="utf-8",
        newline="\n",
    )
    dependencies = [
        f"Python=={platform.python_version()}",
        f"PyInstaller=={_version('pyinstaller')}",
    ]
    dependencies.extend(f"{name}=={_version(name)}" for name, _, _ in RUNTIME_DISTRIBUTIONS)
    (destination / "DEPENDENCIES.txt").write_text(
        "\n".join(dependencies) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    notices = [
        "GrindCAE portable third-party notices",
        "",
        "This internal release candidate redistributes the CPython runtime, Tcl/Tk,",
        "Gmsh, and scientific Python packages. Public redistribution remains pending",
        "a separate review of Gmsh GPL obligations and a declared GrindCAE license.",
        "This file is informational and is not legal advice.",
        "",
        f"Python {platform.python_version()} - Python Software Foundation License - https://www.python.org/",
    ]
    notices.extend(
        f"{name} {_version(name)} - {license_name} - {homepage}"
        for name, license_name, homepage in RUNTIME_DISTRIBUTIONS
    )
    notices.extend(
        (
            f"PyInstaller {_version('pyinstaller')} - GPL-2.0-or-later with bootloader exception - https://pyinstaller.org/",
            "",
            "GrindCAE source license: not declared in the repository at build time.",
            "Public/commercial distribution status: pending license review.",
        )
    )
    (destination / "THIRD_PARTY_NOTICES.txt").write_text(
        "\n".join(notices) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    licenses = destination / "LICENSES"
    licenses.mkdir(exist_ok=True)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copy2(python_license, licenses / "Python-LICENSE.txt")
    for distribution, _, _ in RUNTIME_DISTRIBUTIONS:
        for source in _license_files(distribution):
            target_dir = licenses / distribution
            target_dir.mkdir(exist_ok=True)
            target = target_dir / source.name
            counter = 1
            while target.exists():
                target = target_dir / f"{source.stem}-{counter}{source.suffix}"
                counter += 1
            shutil.copy2(source, target)
    for distribution in ("pyinstaller", "pyinstaller-hooks-contrib"):
        for source in _license_files(distribution):
            target_dir = licenses / distribution
            target_dir.mkdir(exist_ok=True)
            shutil.copy2(source, target_dir / source.name)
    bundled = repository_root / "packaging" / "licenses"
    if bundled.is_dir():
        for source in bundled.iterdir():
            if source.is_file():
                shutil.copy2(source, licenses / source.name)


def write_manifest(
    distribution: Path,
    zip_path: Path,
    output: Path,
    git_commit: str,
    clean_machine_status: str,
) -> None:
    files = tuple(sorted(path for path in distribution.rglob("*") if path.is_file()))
    total_size = sum(path.stat().st_size for path in files)
    gmsh_dlls = [path for path in files if path.name.lower().startswith("gmsh") and path.suffix.lower() == ".dll"]
    manifest = {
        "result_format": "grindcae_phase_5b2_portable_release_manifest",
        "application_version": __version__,
        "git_commit": git_commit,
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "build_windows_version": platform.platform(),
        "target_platform": "Windows-x86_64",
        "python_version": platform.python_version(),
        "pyinstaller_version": _version("pyinstaller"),
        "pyinstaller_mode": "onedir",
        "runtime_dependencies": {name: _version(name) for name, _, _ in RUNTIME_DISTRIBUTIONS},
        "distribution_directory_name": distribution.name,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "executables": {
            name: _hash(distribution / name)
            for name in ("GrindCAE.exe", "GrindCAE-Diagnostics.exe")
        },
        "zip_file": zip_path.name,
        "zip_sha256": _hash(zip_path),
        "gmsh": {
            "python_api_version": _version("gmsh"),
            "dll_files": [str(path.relative_to(distribution)) for path in gmsh_dlls],
        },
        "build_gate": "passed",
        "clean_machine_gate": clean_machine_status,
        "license_review": "pending; internal testing candidate only",
        "physical_model_limitations": [
            "includes small-deformation 2D linear-elastic and J2 elastoplastic workflows",
            "selected linear-elastic workflows use plane stress; elastoplastic history workflows use plane strain",
            "grinding forces remain mainly derived from empirical models that have not completed industrial experimental calibration",
            "mechanism workflows use statistical equivalent-grain groups and a macro-scale conservative projection",
            "does not resolve individual abrasive grains or represent measured roughness",
            "3.8.1 literature workflow adds four-component forces and uniform/strip fixed-mesh J2 history comparison",
            "literature workflow has no resolved chip separation, thermal coupling or roughness; independent validation remains pending",
            "retained real-contact and unfinished 4.0 code are outside the 3.8.1 literature release acceptance",
        ],
    }
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--repository-root", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--distribution", type=Path, required=True)
    manifest.add_argument("--zip", dest="zip_path", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--git-commit", required=True)
    manifest.add_argument("--clean-machine-status", default="pending")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_distribution(args.destination.resolve(), args.repository_root.resolve())
    else:
        write_manifest(
            args.distribution.resolve(),
            args.zip_path.resolve(),
            args.output.resolve(),
            args.git_commit,
            args.clean_machine_status,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
