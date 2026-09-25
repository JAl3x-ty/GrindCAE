"""CLI for GrindCAE 3.0.0 single-grain real contact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .api import run_single_grain_contact
from .exporters import ARTIFACT_FILENAMES, artifact_paths, read_contact_summary
from .models import SingleGrainContactCase
from .exporters_v2 import ARTIFACT_FILENAMES_V2, artifact_paths_v2, read_contact_summary_v2
from .exporters_v3 import ARTIFACT_FILENAMES_V3, artifact_paths_v3, read_contact_summary_v3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 2D small-strain fixed-mesh single-grain real contact."
    )
    parser.add_argument("case", help="strict single_grain_contact schema-1 JSON")
    parser.add_argument("--output-dir", required=True, help="formal result directory")
    parser.add_argument("--preflight-only", action="store_true", help="validate input without solving")
    parser.add_argument("--progress", action="store_true", help="print committed trajectory progress")
    return parser


def _preflight(case: SingleGrainContactCase, output_directory: str | Path) -> tuple[int, int]:
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("output path is an existing file")
    paths = (
        artifact_paths_v3(output)
        if case.single_grain_contact_schema_version == 3
        else artifact_paths_v2(output)
        if case.single_grain_contact_schema_version == 2
        else artifact_paths(output)
    )
    if output.exists():
        for path in paths.values():
            if path.exists() and not path.is_file():
                raise ValueError(f"managed artifact path is not a regular file: {path.name}")
    parent = output.parent
    existing_parent = parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise ValueError("output parent does not resolve to an existing directory")
    count = (
        len(ARTIFACT_FILENAMES_V3)
        if case.single_grain_contact_schema_version == 3
        else len(ARTIFACT_FILENAMES_V2)
        if case.single_grain_contact_schema_version == 2
        else len(ARTIFACT_FILENAMES)
    )
    return len(case.trajectory.targets), count


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    try:
        case = SingleGrainContactCase.from_json(args.case)
        target_count, artifact_count = _preflight(case, args.output_dir)
        if args.preflight_only:
            print("Single-grain contact preflight passed.")
            print(f"Schema: {case.single_grain_contact_schema_version}")
            print(f"Trajectory targets: {target_count}")
            print(f"Managed artifacts: {artifact_count}")
            if case.single_grain_contact_schema_version == 3:
                print("Scope: 2D small-strain fixed background mesh; automatic damage and model-predicted material separation; no free chip motion or independent physical validation.")
            return 0
        progress = None
        if args.progress:
            def progress(current: int, total: int, message: str | None) -> None:
                suffix = f" - {message}" if message else ""
                print(f"Progress: {current}/{total}{suffix}")
        run_single_grain_contact(case, args.output_dir, progress=progress)
        if case.single_grain_contact_schema_version == 3:
            summary = read_contact_summary_v3(artifact_paths_v3(args.output_dir)["summary_json"])
        elif case.single_grain_contact_schema_version == 2:
            summary = read_contact_summary_v2(artifact_paths_v2(args.output_dir)["summary_json"])
        else:
            summary = read_contact_summary(artifact_paths(args.output_dir)["summary_json"])
    except Exception as exc:
        print(f"Single-grain contact error: {exc}", file=sys.stderr)
        return 2
    print(f"Result format: {summary['result_format']}")
    print(f"Output directory: {Path(args.output_dir).expanduser().resolve()}")
    if case.single_grain_contact_schema_version == 3:
        print("Scope: 2D small-strain fixed-background-mesh damage and model-predicted material separation; no free-chip motion or independent physical validation.")
    elif case.single_grain_contact_schema_version == 2:
        print("Scope: 2D small-strain active mesh with preset removed material state; no automatic damage or material-removal prediction.")
    else:
        print("Scope: 2D small-strain fixed mesh; residual groove is unloaded surface displacement, not material removal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
