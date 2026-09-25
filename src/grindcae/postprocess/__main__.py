"""Command-line entry point for phase 3 static post-processing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import sys
import tempfile

from grindcae.domain.models import SimulationCase
from grindcae.mesh import generate_mesh
from grindcae.solver import FullFieldSolution, solve_full_field_from_mesh

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
)

from .exporters import (
    ARTIFACT_FILENAMES,
    build_summary,
    deformation_scale_argument,
    write_elements_csv,
    write_nodes_csv,
    write_pngs,
    write_summary_json,
    write_vtu,
)
from .recovery import (
    StressStatistics,
    compute_stress_statistics,
    recover_fields,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a real Gmsh mesh, solve the 2D plane-stress case, "
            "recover P1 element fields, and export static results."
        )
    )
    parser.add_argument("case", help="path to a schema version 3 JSON case")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving summary.json, VTU, CSV, and PNG artifacts",
    )
    parser.add_argument(
        "--mesh-output",
        help="optional .msh path retaining the exact mesh used by this run",
    )
    parser.add_argument(
        "--deformation-scale",
        default="auto",
        type=_argparse_deformation_scale,
        help="visual-only positive scale for deformed_shape.png (default: auto)",
    )
    return parser


def _argparse_deformation_scale(value: str) -> str | float:
    try:
        return deformation_scale_argument(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _final_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        key: (output_dir / filename).resolve()
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def _print_summary(
    solution: FullFieldSolution,
    statistics: StressStatistics,
    artifacts: dict[str, Path],
    retained_mesh: Path | None,
) -> None:
    result = solution.solver_summary
    print(
        "External force [N]: "
        f"Fx={result.assembled_external_force_x:.12g}, "
        f"Fy={result.assembled_external_force_y:.12g}"
    )
    print(
        "Fixed reaction [N]: "
        f"Rx={result.fixed_reaction_x:.12g}, "
        f"Ry={result.fixed_reaction_y:.12g}"
    )
    print(
        "Balance residual [N]: "
        f"x={result.balance_residual_x:.6e}, "
        f"y={result.balance_residual_y:.6e}, "
        f"norm={result.balance_residual_norm:.6e}"
    )
    print(
        "Maximum displacement: "
        f"{result.maximum_displacement:.12g} m "
        f"({1.0e6 * result.maximum_displacement:.12g} \N{GREEK SMALL LETTER MU}m)"
    )
    print(
        "Area-weighted p95 von Mises: "
        f"{statistics.von_mises_area_weighted_p95:.12g} Pa "
        f"({1.0e-6 * statistics.von_mises_area_weighted_p95:.12g} MPa)"
    )
    print(
        "Raw element maximum von Mises (mesh-dependent): "
        f"{statistics.von_mises_raw_element_maximum_mesh_dependent:.12g} Pa "
        "- requires mesh-convergence interpretation"
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    if retained_mesh is not None:
        print(f"  {retained_mesh}")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    case = SimulationCase.from_json(args.case)
    output_dir = Path(args.output_dir).expanduser().resolve()
    final_paths = _final_artifact_paths(output_dir)

    retained_mesh: Path | None = None
    if args.mesh_output:
        retained_mesh = Path(args.mesh_output).expanduser().resolve()
        if retained_mesh.suffix.lower() != ".msh":
            raise ValueError("--mesh-output must use the .msh extension")

    with tempfile.TemporaryDirectory(prefix="grindcae-postprocess-") as temp_dir:
        workspace = Path(temp_dir)
        temporary_mesh = workspace / "solve_input.msh"
        temporary_output = workspace / "artifacts"
        temporary_output.mkdir()
        temporary_paths = {
            key: temporary_output / filename
            for key, filename in ARTIFACT_FILENAMES.items()
        }

        generate_mesh(case, temporary_mesh)
        solution = solve_full_field_from_mesh(case, temporary_mesh)
        fields = recover_fields(solution)
        statistics = compute_stress_statistics(fields)
        write_vtu(fields, temporary_paths["vtu"])
        write_nodes_csv(fields, temporary_paths["nodes_csv"])
        write_elements_csv(fields, temporary_paths["elements_csv"])
        deformation_scale = write_pngs(
            solution,
            fields,
            temporary_paths,
            args.deformation_scale,
        )

        summary_artifacts = dict(final_paths)
        if retained_mesh is not None:
            summary_artifacts["solve_input_mesh"] = retained_mesh
        summary = build_summary(
            solution,
            statistics,
            summary_artifacts,
            deformation_scale,
        )
        write_summary_json(summary, temporary_paths["summary_json"])

        output_dir.mkdir(parents=True, exist_ok=True)
        for key, source_path in temporary_paths.items():
            if key == "summary_json":
                continue
            shutil.copy2(source_path, final_paths[key])
        if retained_mesh is not None:
            retained_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporary_mesh, retained_mesh)
        shutil.copy2(
            temporary_paths["summary_json"], final_paths["summary_json"]
        )

    _print_summary(solution, statistics, final_paths, retained_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
