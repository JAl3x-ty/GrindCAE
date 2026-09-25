"""CLI for Phase 6A.2 fixed-mesh incremental elastoplastic FEM."""

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
from collections.abc import Sequence
import uuid

from matplotlib import image as matplotlib_image
import meshio

from grindcae.elastoplastic import MaterialPointError
from grindcae.mesh import MeshGenerationError, generate_split_top_rectangle_mesh
from grindcae.solver import SolverError, import_gmsh_mesh

from .assembly import ElastoplasticAssemblyError
from .exporters import (
    ELEMENT_CSV_FIELDS,
    INCREMENT_HISTORY_CSV_FIELDS,
    NODE_CSV_FIELDS,
    RESULT_FORMAT,
    build_summary,
    write_elements_csv,
    write_increment_history_csv,
    write_nodes_csv,
    write_pngs,
    write_summary_json,
    write_vtu,
)
from .models import ElastoplasticFemCase, ElastoplasticFemValidationError
from .recovery import ElastoplasticRecoveryError, recover_snapshot_fields
from .solver import (
    ElastoplasticConvergenceError,
    ElastoplasticSolverError,
    prepare_elastoplastic_mesh,
)
from .workflow import solve_loading_unloading


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "increment_history_csv": "increment_history.csv",
    "vtu": "results.vtu",
    "nodes_csv": "nodes.csv",
    "elements_csv": "elements.csv",
    "mesh_png": "mesh.png",
    "peak_displacement_png": "peak_displacement.png",
    "peak_von_mises_stress_png": "peak_von_mises_stress.png",
    "equivalent_plastic_strain_png": "equivalent_plastic_strain.png",
    "residual_displacement_png": "residual_displacement.png",
    "residual_von_mises_stress_png": "residual_von_mises_stress.png",
}


class ElastoplasticFemWorkflowError(RuntimeError):
    """Raised when the complete Phase 6A.2 artifact set is invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an independent fixed-mesh plane-strain J2 "
            "incremental FEM benchmark through load factor 0 -> 1 -> 0."
        )
    )
    parser.add_argument("case", help="schema version 1 JSON input")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving the complete auditable artifact set",
    )
    return parser


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {key: directory / filename for key, filename in ARTIFACT_FILENAMES.items()}


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def validate_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise ElastoplasticFemWorkflowError("artifact set does not match Phase 6A.2")
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ElastoplasticFemWorkflowError(f"artifact is missing or empty: {path.name}")
    try:
        text = artifacts["summary_json"].read_text(encoding="ascii")
        if "NaN" in text or "Infinity" in text:
            raise ValueError("summary contains a non-finite JSON token")
        summary = json.loads(text)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        if set(summary.get("artifacts", {})) != set(ARTIFACT_FILENAMES):
            raise ValueError("summary artifact map is invalid")
        node_count = int(summary["mesh"]["node_count"])
        triangle_count = int(summary["mesh"]["triangle_count"])
        history_header, history_rows = _read_csv(artifacts["increment_history_csv"])
        node_header, node_rows = _read_csv(artifacts["nodes_csv"])
        element_header, element_rows = _read_csv(artifacts["elements_csv"])
        if history_header != INCREMENT_HISTORY_CSV_FIELDS or not history_rows:
            raise ValueError("increment history CSV header or rows are invalid")
        if node_header != NODE_CSV_FIELDS or len(node_rows) != node_count:
            raise ValueError("nodes CSV header or row count is invalid")
        if element_header != ELEMENT_CSV_FIELDS or len(element_rows) != triangle_count:
            raise ValueError("elements CSV header or row count is invalid")
        if [int(row["node_id"]) for row in node_rows] != list(range(node_count)):
            raise ValueError("nodes CSV IDs are not contiguous and 0-based")
        if [int(row["element_id"]) for row in element_rows] != list(range(triangle_count)):
            raise ValueError("elements CSV IDs are not contiguous and 0-based")
        for rows, text_fields in (
            (history_rows, {"segment", "step_reduction_occurred"}),
            (
                element_rows,
                {"peak_increment_class", "residual_increment_class"},
            ),
        ):
            for row in rows:
                for field, value in row.items():
                    if field not in text_fields and not math.isfinite(float(value)):
                        raise ValueError(f"CSV field {field} contains a non-finite value")
        vtu = meshio.read(artifacts["vtu"])
        if vtu.points.shape != (node_count, 3):
            raise ValueError("VTU node count is invalid")
        if len(vtu.cells) != 1 or vtu.cells[0].type != "triangle" or len(vtu.cells[0].data) != triangle_count:
            raise ValueError("VTU must contain only the result triangles")
        for key, path in artifacts.items():
            if key.endswith("_png"):
                pixels = matplotlib_image.imread(path)
                if pixels.size == 0 or pixels.ndim < 2:
                    raise ValueError(f"PNG is unreadable: {path.name}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ElastoplasticFemWorkflowError(f"artifact validation failed: {exc}") from exc


def _stage_copy(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, staged)
    except OSError as exc:
        raise ElastoplasticFemWorkflowError(
            f"could not stage {target.name} for publication: {exc}"
        ) from exc
    return staged


def publish_artifacts(
    temporary_artifacts: dict[str, Path], final_artifacts: dict[str, Path]
) -> None:
    if set(temporary_artifacts) != set(final_artifacts) or set(final_artifacts) != set(ARTIFACT_FILENAMES):
        raise ElastoplasticFemWorkflowError("artifact publication set is invalid")
    validate_artifacts(temporary_artifacts)
    output_dir = final_artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in order:
            staged[key] = _stage_copy(temporary_artifacts[key], final_artifacts[key], "new")
        for key in order:
            if final_artifacts[key].exists():
                backups[key] = _stage_copy(final_artifacts[key], final_artifacts[key], "backup")
        for key in order:
            os.replace(staged[key], final_artifacts[key])
            del staged[key]
            published.append(key)
    except (OSError, ElastoplasticFemWorkflowError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups:
                    os.replace(backups[key], final_artifacts[key])
                    del backups[key]
                else:
                    final_artifacts[key].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{final_artifacts[key].name}: {rollback_exc}")
        message = f"artifact publication failed: {exc}"
        if rollback_errors:
            message += "; " + "; ".join(rollback_errors)
        raise ElastoplasticFemWorkflowError(message) from exc
    finally:
        for path in (*staged.values(), *backups.values()):
            if path.exists():
                path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _print_result(input_path: Path, result: object, summary: dict[str, object], artifacts: dict[str, Path]) -> None:
    peak = summary["snapshots"]["peak"]
    residual = summary["snapshots"]["residual_unloaded"]
    print(f"Input file: {input_path}")
    print(
        "Mesh: "
        f"nodes={summary['mesh']['node_count']}, triangles={summary['mesh']['triangle_count']}, "
        f"active facets={summary['mesh']['active_top_facet_count']}"
    )
    print(
        "Peak state: "
        f"load factor={peak['load_factor']:.12g}, "
        f"maximum displacement={peak['statistics']['maximum_displacement_m']:.12g} m, "
        f"maximum equivalent plastic strain={peak['statistics']['maximum_equivalent_plastic_strain']:.12g}, "
        f"plastic elements={peak['statistics']['plastic_element_count']}"
    )
    print(
        "Residual unloaded state: "
        f"load factor={residual['load_factor']:.12g}, "
        f"maximum residual displacement={residual['statistics']['maximum_displacement_m']:.12g} m, "
        f"maximum residual von Mises stress={residual['statistics']['von_mises_raw_element_maximum_mesh_dependent_Pa']:.12g} Pa"
    )
    print(
        "Newton: "
        f"maximum iterations={summary['newton_history']['maximum_iterations_used']}, "
        f"step reduction={summary['newton_history']['step_reduction_occurred']}, "
        f"maximum balance residual={summary['newton_history']['maximum_balance_residual_norm_N']:.12g} N"
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: independent fixed-mesh plane-strain elastoplastic benchmark; "
        "no grinding-load integration, no complete-pass plastic history, no GUI, and no packaging update."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    final_artifacts = artifact_paths(args.output_dir)
    try:
        case = ElastoplasticFemCase.from_json(input_path)
        with tempfile.TemporaryDirectory(prefix="grindcae-elastoplastic-fem-") as temp_dir:
            workspace = Path(temp_dir)
            temporary_artifacts = {
                key: workspace / filename for key, filename in ARTIFACT_FILENAMES.items()
            }
            msh_path = workspace / "fixed_mesh.msh"
            split_points = tuple(
                value
                for value in (case.load.x_start_m, case.load.x_end_m)
                if 0.0 < value < case.geometry.width
            )
            generate_split_top_rectangle_mesh(
                case,
                msh_path,
                top_split_x_m=tuple(sorted(set(split_points))),
            )
            imported = import_gmsh_mesh(msh_path, case)
            prepared = prepare_elastoplastic_mesh(case, imported)
            result = solve_loading_unloading(case, prepared)
            peak = recover_snapshot_fields(case, prepared, result.peak.state)
            residual = recover_snapshot_fields(case, prepared, result.residual.state)
            write_increment_history_csv(result, temporary_artifacts["increment_history_csv"])
            write_vtu(peak, residual, temporary_artifacts["vtu"])
            write_nodes_csv(peak, residual, temporary_artifacts["nodes_csv"])
            write_elements_csv(peak, residual, temporary_artifacts["elements_csv"])
            write_pngs(case, result, peak, residual, temporary_artifacts)
            summary = build_summary(case, result, peak, residual, final_artifacts)
            write_summary_json(summary, temporary_artifacts["summary_json"])
            publish_artifacts(temporary_artifacts, final_artifacts)
    except (
        ElastoplasticFemValidationError,
        MeshGenerationError,
        SolverError,
        MaterialPointError,
        ElastoplasticAssemblyError,
        ElastoplasticSolverError,
        ElastoplasticConvergenceError,
        ElastoplasticRecoveryError,
        ElastoplasticFemWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Elastoplastic FEM error: {exc}", file=sys.stderr)
        return 2
    _print_result(input_path, result, summary, final_artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
