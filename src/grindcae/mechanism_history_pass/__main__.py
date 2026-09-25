"""Command-line entry for the Phase 7A.3 comparison workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .models import MechanismHistoryPassCase, MechanismHistoryPassValidationError
from .workflow import run_mechanism_elastoplastic_history_pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a statistical-mechanism complete-pass comparison.")
    parser.add_argument("case", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        input_path = args.case.expanduser().resolve()
        output_path = args.output_dir.expanduser().resolve()
        if output_path == input_path:
            raise MechanismHistoryPassValidationError("output directory must not be the input JSON file")
        case = MechanismHistoryPassCase.from_json(input_path)
        _, artifacts, summary = run_mechanism_elastoplastic_history_pass(case, output_path)
    except (MechanismHistoryPassValidationError, RuntimeError, OSError, ValueError) as exc:
        print(f"Mechanism-history error: {exc}", file=sys.stderr)
        return 2
    print(f"Published comparison artifacts: {len(artifacts)}")
    print(f"Manual acceptance figure: {artifacts['mechanism_history_comparison_png']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
