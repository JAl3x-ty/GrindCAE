"""Command-line workflow for phase 4C4 quasi-static single-pass scans."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from grindcae.evolved_fem import EvolvedFemWorkflowError
from grindcae.evolved_mesh import EvolvedMeshGenerationError, EvolvedMeshValidationRuntimeError
from grindcae.force_model import ForceModelError, ForceModelValidationError
from grindcae.pass_loading import PassLoadError, PassLoadValidationError
from grindcae.postprocess import PostprocessError, deformation_scale_argument
from grindcae.solver import SolverError
from grindcae.trajectory import TrajectoryError

from .core import PassScanError, ScanHistoryRow, run_pass_scan
from .exporters import (
    PassScanExportError,
    aggregate_artifact_paths,
    export_pass_scan,
    publish_pass_scan_artifacts,
)
from .models import PassScanCase, PassScanValidationError
from .positions import PassScanPositionError


def _deformation_scale_argument(value: str) -> str | float:
    try:
        return deformation_scale_argument(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a complete single pass as serial independent quasi-static "
            "plane-stress snapshots and publish aggregate force, displacement, "
            "and stress histories."
        )
    )
    parser.add_argument("case", help="pass-scan JSON input")
    parser.add_argument("--output-dir", required=True, help="aggregate scan output directory")
    parser.add_argument(
        "--deformation-scale",
        default="auto",
        type=_deformation_scale_argument,
        help="visual-only scale for detailed snapshot deformed-shape plots",
    )
    return parser


def _progress(row: ScanHistoryRow, total: int) -> None:
    print(
        f"[{row.scan_index + 1}/{total}] "
        f"x={row.wheel_lowest_point_x_m:.12g} m, "
        f"s={row.motion_coordinate_m:.12g} m, "
        f"state={row.pass_state}, contact_ratio={row.contact_ratio:.8g}, "
        f"Fx/Fy=({row.current_Fx_N:.8g}, {row.current_Fy_N:.8g}) N, "
        f"umax={row.maximum_displacement_m:.8g} m, "
        f"p95/p99=({row.von_mises_area_weighted_p95_Pa:.8g}, "
        f"{row.von_mises_area_weighted_p99_Pa:.8g}) Pa"
    )


def _print_summary(input_path: Path, result, artifacts: dict[str, Path]) -> None:
    counts = {state: sum(row.pass_state == state for row in result.history) for state in (
        "before_entry", "entry", "full_contact", "exit", "after_exit"
    )}
    print(f"Input file: {input_path}")
    print(f"Scan points: {len(result.history)}; state counts: {counts}")
    print(
        "Monotonic checks: "
        f"ground={result.validations.ground_length_monotonic_non_decreasing}, "
        f"unprocessed={result.validations.unprocessed_length_monotonic_non_increasing}, "
        f"entry/full/exit ratio="
        f"{result.validations.entry_contact_ratio_monotonic_non_decreasing}/"
        f"{result.validations.full_contact_ratio_is_one}/"
        f"{result.validations.exit_contact_ratio_monotonic_non_increasing}"
    )
    print(
        "Maximum force-balance residual [N]: "
        f"{result.validations.maximum_force_balance_residual_norm_N:.6e}"
    )
    print("Snapshot roles:")
    for role, index in result.snapshot_role_indices.items():
        print(f"  {role}: scan_index={index}, roles={result.history[index].snapshot_roles}")
    print("Output artifacts:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: single-pass independent quasi-static snapshots, small-deformation "
        "2D plane stress, isotropic linear elasticity, ideal circular surface, "
        "and one-way empirical force coupling."
    )
    print(
        "Nominal time is only a travel coordinate. There is no state transfer, "
        "plastic or thermal accumulation, inertia, damping, impact, chatter, true "
        "contact iteration, multiple pass/spark-out, wheel wear, Ra/Rz prediction, "
        "or industrial validation."
    )
    print(
        "Do not infer yielding, safety factor, production quality, or industrial "
        "validity directly from the scan stress values."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    final_artifacts = aggregate_artifact_paths(output_dir)
    try:
        case = PassScanCase.from_json(input_path)
        with tempfile.TemporaryDirectory(prefix="grindcae-pass-scan-") as temp_dir:
            workspace = Path(temp_dir)
            result = run_pass_scan(case, workspace / "solve", progress=_progress)
            temporary_artifacts, _ = export_pass_scan(
                result,
                workspace / "artifacts",
                output_dir,
                args.deformation_scale,
            )
            publish_pass_scan_artifacts(temporary_artifacts, final_artifacts)
    except (
        PassScanValidationError,
        PassScanPositionError,
        PassScanError,
        PassScanExportError,
        PassLoadValidationError,
        PassLoadError,
        ForceModelValidationError,
        ForceModelError,
        TrajectoryError,
        EvolvedMeshGenerationError,
        EvolvedMeshValidationRuntimeError,
        EvolvedFemWorkflowError,
        SolverError,
        PostprocessError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as exc:
        print(f"Pass-scan error: {exc}", file=sys.stderr)
        return 2
    _print_summary(input_path, result, final_artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
