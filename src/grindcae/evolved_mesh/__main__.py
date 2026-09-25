"""Command-line workflow for phase 4C3A evolved-surface meshing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence
import uuid

import matplotlib.image as matplotlib_image
import meshio
from skfem.io.meshio import from_meshio

from grindcae.trajectory import TrajectoryError, build_surface_profile

from .exporters import (
    RESULT_FORMAT,
    TOP_BOUNDARY_CSV_FIELDS,
    build_summary,
    serialize_summary,
    write_mesh_png,
    write_top_boundary_facets_csv,
)
from .generator import EvolvedMeshGenerationError, generate_evolved_surface_mesh
from .models import EvolvedMeshCase, EvolvedMeshValidationError
from .validation import (
    EvolvedMeshValidationRuntimeError,
    validate_evolved_surface_mesh,
)


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "evolved_surface_msh": "evolved_surface.msh",
    "top_boundary_facets_csv": "top_boundary_facets.csv",
    "mesh_png": "mesh.png",
}


class EvolvedMeshWorkflowError(RuntimeError):
    """Raised when temporary artifacts or atomic publication are invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate a real Gmsh mesh of an evolved single-pass top surface."
        )
    )
    parser.add_argument("case", help="evolved-surface mesh JSON input")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving summary.json, evolved_surface.msh, CSV, and PNG",
    )
    return parser


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        key: (output_dir / filename).resolve()
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="\n")


def _validate_temporary_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise EvolvedMeshWorkflowError("temporary artifact set is invalid")
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise EvolvedMeshWorkflowError(
                f"temporary artifact is missing or empty: {path.name}"
            )

    serialized = artifacts["summary_json"].read_text(encoding="ascii")
    if "NaN" in serialized or "Infinity" in serialized:
        raise EvolvedMeshWorkflowError("summary contains a non-finite JSON token")
    summary = json.loads(serialized)
    if summary.get("result_format") != RESULT_FORMAT:
        raise EvolvedMeshWorkflowError("summary result_format is invalid")
    if set(summary.get("artifacts", {})) != set(ARTIFACT_FILENAMES):
        raise EvolvedMeshWorkflowError("summary artifact map is invalid")

    mesh_summary = summary.get("mesh", {})
    summary_node_count = int(mesh_summary["node_count"])
    summary_triangle_count = int(mesh_summary["triangle_count"])
    summary_dofs = int(mesh_summary["actual_vector_p1_degrees_of_freedom"])

    with artifacts["top_boundary_facets_csv"].open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows or tuple(rows[0]) != TOP_BOUNDARY_CSV_FIELDS:
        raise EvolvedMeshWorkflowError("top-boundary CSV header or rows are invalid")
    starts = [float(row["x_start_m"]) for row in rows]
    ends = [float(row["x_end_m"]) for row in rows]
    projected = [float(row["projected_length_x_m"]) for row in rows]
    actual = [float(row["actual_facet_length_m"]) for row in rows]
    if starts != sorted(starts) or any(not start < end for start, end in zip(starts, ends)):
        raise EvolvedMeshWorkflowError("top-boundary CSV is not ordered left to right")
    width = float(summary["derived_geometry"]["workpiece_length_m"])
    if starts[0] != 0.0 or ends[-1] != width:
        raise EvolvedMeshWorkflowError("top-boundary CSV does not cover exact [0, W]")
    if any(
        not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)
        for left, right in zip(ends[:-1], starts[1:])
    ):
        raise EvolvedMeshWorkflowError("top-boundary CSV facets are not contiguous")
    if not math.isclose(math.fsum(projected), width, rel_tol=1.0e-12, abs_tol=1.0e-14):
        raise EvolvedMeshWorkflowError("top-boundary CSV projections do not sum to W")
    summary_length = float(
        summary["curve_and_facet_lengths"]["discrete_complete_top_facet_length_m"]
    )
    if not math.isclose(
        math.fsum(actual), summary_length, rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise EvolvedMeshWorkflowError("top-boundary CSV length disagrees with summary")

    msh = meshio.read(artifacts["evolved_surface_msh"])
    if set(msh.field_data) != {"domain", "fixed", "contact", "free_left", "free_right"}:
        raise EvolvedMeshWorkflowError("published MSH physical groups are invalid")
    imported = from_meshio(msh)
    meshio_node_count = int(msh.points.shape[0])
    triangle_count = sum(
        int(block.data.shape[0]) for block in msh.cells if block.type == "triangle"
    )
    if not (
        summary_node_count == meshio_node_count == int(imported.nvertices)
    ):
        raise EvolvedMeshWorkflowError(
            "summary, meshio, and scikit-fem node counts disagree"
        )
    if summary_dofs != 2 * int(imported.nvertices):
        raise EvolvedMeshWorkflowError(
            "summary vector-P1 degrees of freedom do not equal 2 * nvertices"
        )
    if not (
        summary_triangle_count == triangle_count == int(imported.nelements)
    ):
        raise EvolvedMeshWorkflowError(
            "summary, meshio, and scikit-fem triangle counts disagree"
        )
    maximum_csv_node_id = max(
        max(int(row["node_start_id"]), int(row["node_end_id"])) for row in rows
    )
    if maximum_csv_node_id >= summary_node_count:
        raise EvolvedMeshWorkflowError(
            "top-boundary CSV references a node outside the final MSH"
        )
    pixels = matplotlib_image.imread(artifacts["mesh_png"])
    if pixels.size == 0 or pixels.ndim < 2:
        raise EvolvedMeshWorkflowError("mesh PNG is unreadable or empty")


def _stage_copy(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, staged)
    except OSError as exc:
        raise EvolvedMeshWorkflowError(
            f"could not stage {target.name} for publication: {exc}"
        ) from exc
    return staged


def _publish_results(
    temporary_artifacts: dict[str, Path], artifacts: dict[str, Path]
) -> None:
    if set(temporary_artifacts) != set(artifacts) or set(artifacts) != set(
        ARTIFACT_FILENAMES
    ):
        raise EvolvedMeshWorkflowError("artifact publication set is invalid")
    _validate_temporary_artifacts(temporary_artifacts)

    output_dir = artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvolvedMeshWorkflowError(
            f"output directory could not be created: {exc}"
        ) from exc

    publication_order = [
        key for key in ARTIFACT_FILENAMES if key != "summary_json"
    ] + ["summary_json"]
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in publication_order:
            staged[key] = _stage_copy(temporary_artifacts[key], artifacts[key], "new")
        for key in publication_order:
            if artifacts[key].exists():
                backups[key] = _stage_copy(artifacts[key], artifacts[key], "backup")
        for key in publication_order:
            os.replace(staged[key], artifacts[key])
            del staged[key]
            published.append(key)
    except (OSError, EvolvedMeshWorkflowError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups:
                    os.replace(backups[key], artifacts[key])
                    del backups[key]
                else:
                    artifacts[key].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{artifacts[key].name} rollback failed: {rollback_exc}"
                )
        detail = f"artifact publication failed: {exc}"
        if rollback_errors:
            detail += "; " + "; ".join(rollback_errors)
        raise EvolvedMeshWorkflowError(detail) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _print_summary(input_path: Path, result: object, artifacts: dict[str, Path]) -> None:
    print(f"Input file: {input_path}")
    print(
        f"Pass: state={result.trajectory.pass_state}, "
        f"direction={result.case.trajectory.single_pass.relative_feed_direction}"
    )
    print(
        "Arc lengths: "
        f"projected={result.trajectory.effective_contact_length_m:.12g} m, "
        f"exact_curve={result.exact_arc_curve_length_m:.12g} m, "
        f"facet_chords={result.discrete_arc_facet_length_m:.12g} m"
    )
    print(
        "Areas: "
        f"exact={result.exact_evolved_domain_area_m2:.12g} m^2, "
        f"triangles={result.discrete_triangle_area_m2:.12g} m^2, "
        f"curvature_error={result.curvature_discretization_area_error_m2:.12g} m^2"
    )
    print(
        "Mesh: "
        f"nodes={result.node_count}, "
        f"triangles={result.triangle_count}, "
        f"arc_facets={result.arc_facet_count}, "
        f"actual_vector_P1_dofs={result.actual_vector_p1_degrees_of_freedom}, "
        f"raw_gmsh_nodes={result.generation.raw_gmsh_node_count}"
    )
    print("Import validation: meshio=passed, scikit-fem=passed")
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: geometry and mesh only. Depth is user prescribed. No load, FEM, "
        "displacement, strain, stress, reaction, true contact pressure, thermal "
        "field, roughness, wheel wear, grain randomness, or multiple-pass model ran."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts = _artifact_paths(output_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="grindcae-evolved-mesh-") as temp_dir:
            workspace = Path(temp_dir)
            case = EvolvedMeshCase.from_json(input_path)
            trajectory = build_surface_profile(case.trajectory)
            temporary_artifacts = {
                key: workspace / filename
                for key, filename in ARTIFACT_FILENAMES.items()
            }
            generation = generate_evolved_surface_mesh(
                case, trajectory, temporary_artifacts["evolved_surface_msh"]
            )
            result = validate_evolved_surface_mesh(
                temporary_artifacts["evolved_surface_msh"],
                case,
                trajectory,
                generation,
            )
            write_top_boundary_facets_csv(
                result, temporary_artifacts["top_boundary_facets_csv"]
            )
            write_mesh_png(result, temporary_artifacts["mesh_png"])
            summary = build_summary(
                result, {key: str(path) for key, path in artifacts.items()}
            )
            _write_text(serialize_summary(summary), temporary_artifacts["summary_json"])
            _publish_results(temporary_artifacts, artifacts)
    except (
        EvolvedMeshValidationError,
        TrajectoryError,
        EvolvedMeshGenerationError,
        EvolvedMeshValidationRuntimeError,
        EvolvedMeshWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Evolved-mesh error: {exc}", file=sys.stderr)
        return 2

    _print_summary(input_path, result, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
