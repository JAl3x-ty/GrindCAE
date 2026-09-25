"""Command-line workflow for Phase 7A.2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from collections.abc import Sequence

from grindcae.force_model import ForceModelError, ForceModelValidationError
from grindcae.mechanism_force import MechanismForceError, MechanismForceValidationError

from .core import StatisticalGrainLoadPrediction
from .models import StatisticalGrainLoadCase, StatisticalGrainLoadValidationError
from .workflow import (
    ARTIFACT_FILENAMES,
    StatisticalGrainLoadWorkflowError,
    artifact_paths,
    generate_artifacts,
    os,
    publish_artifacts,
    run_statistical_grain_load,
    validate_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a statistical equivalent-grain mechanism load.")
    parser.add_argument("case", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _validate_paths(input_path: Path, output_dir: Path) -> None:
    if output_dir == input_path:
        raise StatisticalGrainLoadValidationError("output_directory [directory]: --output-dir must not be the input file")


def _print_result(result: StatisticalGrainLoadPrediction, artifacts: dict[str, Path]) -> None:
    print(f"Resolved grit: {result.resolved_grit_specification.resolution_path} {result.resolved_grit_specification.grit_designation}")
    print(f"Estimated physical active grains: {result.estimated_physical_active_grain_count:.12g}")
    print(f"Equivalent grain groups: {result.equivalent_group_count}")
    print(f"Compression ratio: {result.compression_ratio:.12g}")
    print(f"Manual acceptance figure: {artifacts['mechanism_load_distribution_png']}")
    print("Scope: statistical line-load magnitudes only; no FEM, moving load, elastoplastic history, GUI, portable package, or mechanism-history integration.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = args.case.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        _validate_paths(input_path, output_dir)
        result, final, _summary = run_statistical_grain_load(
            StatisticalGrainLoadCase.from_json(input_path), output_dir
        )
    except (
        StatisticalGrainLoadValidationError, StatisticalGrainLoadWorkflowError,
        MechanismForceValidationError, MechanismForceError,
        ForceModelValidationError, ForceModelError,
    ) as exc:
        print(f"Statistical grain load error: {exc}", file=sys.stderr)
        return 2
    _print_result(result, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
