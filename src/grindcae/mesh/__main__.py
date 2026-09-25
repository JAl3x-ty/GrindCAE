"""Command-line entry point for rectangular mesh generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from grindcae.domain.models import SimulationCase

from .generator import generate_mesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 2D Gmsh mesh for a GrindCAE input case."
    )
    parser.add_argument("case", help="path to a schema version 3 JSON case")
    parser.add_argument(
        "--output",
        required=True,
        help="output path for the Gmsh 4.1 .msh file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = SimulationCase.from_json(args.case)
    summary = generate_mesh(case, args.output)

    print(f"Output file: {summary.output_path}")
    print(f"Nodes: {summary.node_count}")
    print(f"2D triangle elements: {summary.triangle_count}")
    print(f"Physical groups: {', '.join(summary.physical_group_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
