"""Command-line entry point for the phase 2A plane-stress solver."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import tempfile

from grindcae.domain.models import SimulationCase
from grindcae.mesh import generate_mesh

from .core import SolveResult, solve_case_from_mesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Gmsh mesh and solve the GrindCAE 2D plane-stress case."
        )
    )
    parser.add_argument("case", help="path to a schema version 3 JSON case")
    parser.add_argument(
        "--output",
        required=True,
        help="output path for the JSON solve result",
    )
    parser.add_argument(
        "--mesh-output",
        help="optional path retaining the exact Gmsh mesh used by the solve",
    )
    return parser


def _write_result(result: SolveResult, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("--output must use the .json extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def _print_summary(result: SolveResult, output_path: Path) -> None:
    print(f"Result file: {output_path}")
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
    print(f"Maximum displacement [m]: {result.maximum_displacement:.12g}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = SimulationCase.from_json(args.case)

    with tempfile.TemporaryDirectory(prefix="grindcae-solve-") as temporary_dir:
        temporary_mesh = Path(temporary_dir) / "solve_input.msh"
        generate_mesh(case, temporary_mesh)

        if args.mesh_output:
            retained_mesh = Path(args.mesh_output).expanduser().resolve()
            if retained_mesh.suffix.lower() != ".msh":
                raise ValueError("--mesh-output must use the .msh extension")
            retained_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temporary_mesh, retained_mesh)

        result = solve_case_from_mesh(case, temporary_mesh)

    output_path = _write_result(result, args.output)
    _print_summary(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
