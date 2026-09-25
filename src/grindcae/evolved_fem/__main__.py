"""Command-line entry point for phase 4C3B evolved-surface empirical FEM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

from grindcae.evolved_mesh import (
    EvolvedMeshGenerationError,
    EvolvedMeshValidationRuntimeError,
    generate_evolved_surface_mesh,
    validate_evolved_surface_mesh,
)
from grindcae.force_model import ForceModelError, ForceModelValidationError
from grindcae.pass_loading import (
    PassLoadError,
    PassLoadValidationError,
    build_pass_load,
)
from grindcae.postprocess import (
    PostprocessError,
    compute_stress_statistics,
    deformation_scale_argument,
    recover_fields,
    write_elements_csv,
    write_nodes_csv,
    write_pngs,
    write_vtu,
)
from grindcae.solver import (
    SolverError,
    assemble_facet_plane_stress_system,
    solve_facet_full_field,
)
from grindcae.trajectory import TrajectoryError

from .exporters import (
    write_active_contact_facets_csv,
    write_mesh_png,
    write_native_p1_contour_pngs,
)
from .loading import EvolvedFacetLoadError, build_evolved_facet_load
from .models import EvolvedFemCase, EvolvedFemValidationError
from .workflow import (
    ARTIFACT_FILENAMES,
    RESULT_FORMAT,
    EvolvedFemOperations,
    EvolvedFemResult,
    EvolvedFemWorkflowError,
    artifact_paths,
    run_evolved_fem_case,
)


def _deformation_scale_argument(value: str) -> str | float:
    try:
        return deformation_scale_argument(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an empirical pass load, generate and validate an "
            "evolved mesh, solve one plane-stress FEM system, and "
            "publish traceable displacement and stress results."
        )
    )
    parser.add_argument("case", help="evolved-surface FEM JSON input")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving the complete 13-file result set",
    )
    parser.add_argument(
        "--deformation-scale",
        default="auto",
        type=_deformation_scale_argument,
        help="visual-only positive scale for deformed_shape.png (default: auto)",
    )
    return parser


def _operations() -> EvolvedFemOperations:
    # Resolve from CLI globals so existing monkeypatch-based one-call audits remain valid.
    return EvolvedFemOperations(
        build_pass_load=build_pass_load,
        generate_mesh=generate_evolved_surface_mesh,
        validate_mesh=validate_evolved_surface_mesh,
        build_facet_load=build_evolved_facet_load,
        assemble_system=assemble_facet_plane_stress_system,
        solve_system=solve_facet_full_field,
        recover_fields=recover_fields,
        compute_statistics=compute_stress_statistics,
        write_vtu=write_vtu,
        write_nodes_csv=write_nodes_csv,
        write_elements_csv=write_elements_csv,
        write_active_facets_csv=write_active_contact_facets_csv,
        write_pngs=write_pngs,
        write_mesh_png=write_mesh_png,
        write_native_p1_contour_pngs=write_native_p1_contour_pngs,
        replace=os.replace,
    )


def _print_summary(
    input_path: Path,
    result: EvolvedFemResult,
    artifacts: dict[str, Path],
) -> None:
    pass_load = result.pass_load
    mesh = result.evolved_mesh
    facet_load = result.facet_load
    solver = result.solution.solver_summary
    statistics = result.stress_statistics
    print(f"Input file: {input_path}")
    print(
        f"Pass: state={pass_load.pass_state}, "
        f"feed_direction={pass_load.case.trajectory.single_pass.relative_feed_direction}, "
        f"tangential_force_direction={pass_load.case.force_mapping.tangential_force_direction}"
    )
    print(
        "Current empirical force [N]: "
        f"Fx={pass_load.current_Fx_N:.12g}, Fy={pass_load.current_Fy_N:.12g}, "
        f"contact_ratio={pass_load.contact_ratio:.12g}"
    )
    print(
        "Projection-to-edge conversion: "
        f"sum(dx)={facet_load.projected_length_sum_m:.12g} m, "
        f"sum(ds)={facet_load.actual_facet_length_sum_m:.12g} m, "
        f"active_facets={len(facet_load.facets)}"
    )
    print(
        "Assembled external force [N]: "
        f"Fx={solver.assembled_external_force_x:.12g}, "
        f"Fy={solver.assembled_external_force_y:.12g}"
    )
    print(
        "Fixed support reaction [N]: "
        f"Rx={solver.fixed_reaction_x:.12g}, Ry={solver.fixed_reaction_y:.12g}"
    )
    print(
        "Force balance residual [N]: "
        f"x={solver.balance_residual_x:.6e}, y={solver.balance_residual_y:.6e}, "
        f"norm={solver.balance_residual_norm:.6e}"
    )
    print(
        f"Mesh: nodes={mesh.node_count}, triangles={mesh.triangle_count}, "
        f"vector_P1_dofs={mesh.actual_vector_p1_degrees_of_freedom}"
    )
    print(f"Maximum displacement [m]: {solver.maximum_displacement:.12g}")
    print(
        "Von Mises stress [Pa]: "
        f"mean={statistics.von_mises_area_weighted_mean:.12g}, "
        f"p95={statistics.von_mises_area_weighted_p95:.12g}, "
        f"p99={statistics.von_mises_area_weighted_p99:.12g}, "
        "raw mesh-dependent max="
        f"{statistics.von_mises_raw_element_maximum_mesh_dependent:.12g}"
    )
    print(
        "Additional native P1 contours: stress_contour.png uses element von Mises "
        "stress [MPa]; strain_contour.png uses maximum in-plane principal "
        "infinitesimal strain [microstrain]."
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: single-pass, quasi-static, small-deformation 2D plane-stress "
        "isotropic linear elasticity with one-way empirical force coupling."
    )
    print(
        "Limit: no true wheel-workpiece contact, Hertz pressure, friction/contact "
        "iteration, grains, plasticity/failure, thermal burn, vibration/chatter, "
        "roughness, wheel wear/runout, multiple passes, spark-out, or industrial validation."
    )
    print(
        "Interpretation warning: do not infer yielding, safety factor, production "
        "quality, or industrial validity directly from the current stress result."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        case = EvolvedFemCase.from_json(input_path)
        result, artifacts, _ = run_evolved_fem_case(
            case,
            output_dir,
            args.deformation_scale,
            operations=_operations(),
        )
    except (
        EvolvedFemValidationError,
        PassLoadValidationError,
        PassLoadError,
        ForceModelValidationError,
        ForceModelError,
        TrajectoryError,
        EvolvedMeshGenerationError,
        EvolvedMeshValidationRuntimeError,
        EvolvedFacetLoadError,
        SolverError,
        PostprocessError,
        EvolvedFemWorkflowError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as exc:
        print(f"Evolved-FEM error: {exc}", file=sys.stderr)
        return 2
    _print_summary(input_path, result, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
