"""PyInstaller console diagnostics and smoke-test entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grindcae.gui.portable_diagnostics import collect_portable_diagnostics, print_portable_diagnostics
from grindcae.gui.runtime_paths import configure_runtime_environment, user_data_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GrindCAE portable Windows diagnostics")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run real 4C3B, 4C4, and mechanism field-evolution smoke calculations",
    )
    parser.add_argument("--output-dir", type=Path, help="smoke output directory")
    parser.add_argument("--literature-smoke", action="store_true", help="run the 3.8.1 literature/J2 workflow and real Tk project round trip")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        configure_runtime_environment()
    except OSError as exc:
        print(f"[FAIL] Runtime directory setup: {type(exc).__name__}: {exc}")
        return 1
    passed = print_portable_diagnostics(collect_portable_diagnostics())
    if not passed:
        return 1
    if args.literature_smoke:
        from literature_smoke import run_literature_smoke
        output = (args.output_dir or (user_data_directory() / "literature_smoke")).resolve()
        try:
            report = run_literature_smoke(output)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] Literature smoke: {exc}")
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if args.smoke_test:
        from grindcae.gui.portable_smoke import run_portable_smoke

        output = (args.output_dir or (user_data_directory() / "portable_smoke")).expanduser().resolve()
        try:
            report = run_portable_smoke(output)
        except Exception as exc:
            print(f"[FAIL] Portable smoke test: {type(exc).__name__}: {exc}")
            return 1
        print("[PASS] Portable smoke test")
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
