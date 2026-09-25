"""CLI and transactional artifact publication for Phase 6A.3."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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
from grindcae.elastoplastic_fem import (
    ElastoplasticAssemblyError,
    ElastoplasticConvergenceError,
    ElastoplasticRecoveryError,
    ElastoplasticSolverError,
)
from grindcae.evolved_fem import EvolvedFacetLoadError
from grindcae.evolved_mesh import (
    EvolvedMeshGenerationError,
    EvolvedMeshValidationRuntimeError,
)
from grindcae.pass_loading import PassLoadError

from .exporters import (
    ACTIVE_CONTACT_FACET_CSV_FIELDS,
    ELEMENT_CSV_FIELDS,
    INCREMENT_HISTORY_CSV_FIELDS,
    NODE_CSV_FIELDS,
    RESULT_FORMAT,
    SURFACE_RECOVERY_CSV_FIELDS,
    build_summary,
    write_active_contact_facets_csv,
    write_elements_csv,
    write_increment_history_csv,
    write_nodes_csv,
    write_result_pngs,
    write_summary_json,
    write_surface_recovery_csv,
    write_vtu,
)
from .models import (
    SinglePositionElastoplasticCase,
    SinglePositionElastoplasticValidationError,
)
from .recovery import SurfaceRecoveryError
from .workflow import (
    SinglePositionElastoplasticResult,
    SinglePositionElastoplasticWorkflowError,
    solve_single_position_elastoplastic,
)


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "increment_history_csv": "increment_history.csv",
    "solve_input_msh": "solve_input.msh",
    "vtu": "results.vtu",
    "nodes_csv": "nodes.csv",
    "elements_csv": "elements.csv",
    "active_contact_facets_csv": "active_contact_facets.csv",
    "mesh_png": "mesh.png",
    "peak_displacement_png": "peak_displacement.png",
    "peak_von_mises_stress_png": "peak_von_mises_stress.png",
    "equivalent_plastic_strain_png": "equivalent_plastic_strain.png",
    "residual_displacement_png": "residual_displacement.png",
    "residual_von_mises_stress_png": "residual_von_mises_stress.png",
    "surface_recovery_csv": "surface_recovery.csv",
    "surface_recovery_png": "surface_recovery.png",
}


class SinglePositionElastoplasticArtifactError(RuntimeError):
    """Raised when the complete Phase 6A.3 artifact set is invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-position evolved-surface plane-strain J2 FEM "
            "through load factor 0 -> 1 -> 0."
        )
    )
    parser.add_argument("case", help="schema version 1 UTF-8 JSON input")
    parser.add_argument("--output-dir", required=True)
    return parser


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {key: directory / name for key, name in ARTIFACT_FILENAMES.items()}


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def validate_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise SinglePositionElastoplasticArtifactError("artifact set is invalid")
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise SinglePositionElastoplasticArtifactError(
                f"artifact is missing or empty: {path.name}"
            )
    try:
        serialized = artifacts["summary_json"].read_text(encoding="ascii")
        if "NaN" in serialized or "Infinity" in serialized:
            raise ValueError("summary contains a non-finite JSON token")
        summary = json.loads(serialized)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        if set(summary.get("artifacts", {})) != set(ARTIFACT_FILENAMES):
            raise ValueError("summary artifact map is invalid")
        node_count = int(summary["mesh"]["node_count"])
        element_count = int(summary["mesh"]["triangle_count"])
        expected_active = int(summary["position_and_contact"]["active_facet_count"])
        for key, header, count in (
            ("increment_history_csv", INCREMENT_HISTORY_CSV_FIELDS, None),
            ("nodes_csv", NODE_CSV_FIELDS, node_count),
            ("elements_csv", ELEMENT_CSV_FIELDS, element_count),
            ("active_contact_facets_csv", ACTIVE_CONTACT_FACET_CSV_FIELDS, expected_active),
            ("surface_recovery_csv", SURFACE_RECOVERY_CSV_FIELDS, None),
        ):
            actual_header, rows = _read_csv(artifacts[key])
            if actual_header != header or (count is not None and len(rows) != count):
                raise ValueError(f"{artifacts[key].name} header or row count is invalid")
            if key in {"increment_history_csv", "surface_recovery_csv"} and not rows:
                raise ValueError(f"{artifacts[key].name} must contain rows")
        if not artifacts["solve_input_msh"].read_bytes().startswith(b"$MeshFormat"):
            raise ValueError("solve_input.msh header is invalid")
        vtu = meshio.read(artifacts["vtu"])
        if vtu.points.shape != (node_count, 3):
            raise ValueError("VTU node count is invalid")
        if len(vtu.cells) != 1 or vtu.cells[0].type != "triangle" or len(vtu.cells[0].data) != element_count:
            raise ValueError("VTU triangle count is invalid")
        for key, path in artifacts.items():
            if key.endswith("_png"):
                pixels = matplotlib_image.imread(path)
                if pixels.size == 0 or pixels.ndim < 2:
                    raise ValueError(f"PNG is unreadable: {path.name}")
        for key in (
            "snapshots",
            "surface_definitions",
            "facet_load_conservation",
        ):
            if key not in summary:
                raise ValueError(f"summary is missing {key}")
        force = summary["facet_load_conservation"]
        for actual, target in (
            (force["integrated_Fx_N"], force["target_Fx_N"]),
            (force["integrated_Fy_N"], force["target_Fy_N"]),
        ):
            if not math.isclose(actual, target, rel_tol=1.0e-11, abs_tol=1.0e-10):
                raise ValueError("facet force conservation is invalid")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SinglePositionElastoplasticArtifactError(
            f"artifact validation failed: {exc}"
        ) from exc


def export_result(
    result: SinglePositionElastoplasticResult,
    output_dir: str | Path,
    *,
    summary_artifacts: dict[str, Path] | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    outputs = artifact_paths(output_dir)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.msh_source, outputs["solve_input_msh"])
    write_increment_history_csv(result.history, outputs["increment_history_csv"])
    write_vtu(result.peak_fields, result.residual_fields, outputs["vtu"])
    write_nodes_csv(result.peak_fields, result.residual_fields, outputs["nodes_csv"])
    write_elements_csv(result.peak_fields, result.residual_fields, outputs["elements_csv"])
    write_active_contact_facets_csv(
        result.facet_load, outputs["active_contact_facets_csv"]
    )
    write_surface_recovery_csv(
        result.surface_recovery, outputs["surface_recovery_csv"]
    )
    write_result_pngs(result, outputs)
    summary_paths = summary_artifacts or outputs
    summary = build_summary(result, summary_paths)
    write_summary_json(summary, outputs["summary_json"])
    validate_artifacts(outputs)
    return outputs, summary


def _stage_copy(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, staged)
    return staged


def publish_artifacts(
    temporary_artifacts: dict[str, Path], final_artifacts: dict[str, Path]
) -> None:
    if set(temporary_artifacts) != set(final_artifacts) or set(final_artifacts) != set(ARTIFACT_FILENAMES):
        raise SinglePositionElastoplasticArtifactError("publication set is invalid")
    validate_artifacts(temporary_artifacts)
    output_dir = final_artifacts["summary_json"].parent
    existed = output_dir.exists()
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
    except (OSError, SinglePositionElastoplasticArtifactError) as exc:
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
        raise SinglePositionElastoplasticArtifactError(message) from exc
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)
        if not existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def run_single_position_elastoplastic(
    case: SinglePositionElastoplasticCase, output_dir: str | Path
) -> tuple[SinglePositionElastoplasticResult, dict[str, Path], dict[str, object]]:
    final_artifacts = artifact_paths(output_dir)
    with tempfile.TemporaryDirectory(prefix="grindcae-single-position-elastoplastic-") as temp_dir:
        workspace = Path(temp_dir)
        result = solve_single_position_elastoplastic(case, workspace / "solve")
        temporary, summary = export_result(
            result,
            workspace / "artifacts",
            summary_artifacts=final_artifacts,
        )
        publish_artifacts(temporary, final_artifacts)
        published_mesh = replace(
            result.evolved_mesh,
            generation=replace(
                result.evolved_mesh.generation,
                output_path=final_artifacts["solve_input_msh"],
            ),
        )
        published_facet_load = replace(result.facet_load, mesh=published_mesh)
        published_imported = replace(
            result.prepared_mesh.imported_mesh,
            mesh=published_mesh.skfem_mesh,
            source_path=final_artifacts["solve_input_msh"],
        )
        published_prepared = replace(
            result.prepared_mesh,
            imported_mesh=published_imported,
        )
        published_history = replace(
            result.history,
            prepared_mesh=published_prepared,
        )
        result = replace(
            result,
            evolved_mesh=published_mesh,
            facet_load=published_facet_load,
            prepared_mesh=published_prepared,
            history=published_history,
            msh_source=final_artifacts["solve_input_msh"],
        )
    return result, final_artifacts, summary


def _print_result(
    input_path: Path,
    summary: dict[str, object],
    artifacts: dict[str, Path],
) -> None:
    contact = summary["position_and_contact"]
    force = summary["current_empirical_force"]
    peak = summary["snapshots"]["peak_loaded"]
    residual = summary["snapshots"]["residual_unloaded"]
    print(f"Input file: {input_path}")
    print(
        f"Position: xb={contact['wheel_lowest_point_x_m']:.12g} m, "
        f"state={contact['pass_state']}, contact ratio={contact['contact_ratio']:.12g}"
    )
    print(
        f"Current force: Fx={force['signed_Fx_N']:.12g} N, "
        f"Fy={force['signed_Fy_N']:.12g} N, active facets={contact['active_facet_count']}"
    )
    print(
        f"Peak: max displacement={peak['statistics']['maximum_displacement_m']:.12g} m, "
        f"max eq plastic strain={peak['statistics']['maximum_equivalent_plastic_strain']:.12g}, "
        f"plastic elements={peak['statistics']['plastic_element_count']}"
    )
    print(
        f"Residual: max displacement={residual['statistics']['maximum_displacement_m']:.12g} m, "
        f"p99 von Mises={residual['statistics']['von_mises_area_weighted_p99_Pa']:.12g} Pa"
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: one fixed wheel position, evolved mesh, empirical one-way load, "
        "plane-strain J2 loading-unloading; no multi-position plastic scan, GUI, or portable build."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    try:
        case = SinglePositionElastoplasticCase.from_json(input_path)
        _, artifacts, summary = run_single_position_elastoplastic(
            case, args.output_dir
        )
    except (
        SinglePositionElastoplasticValidationError,
        PassLoadError,
        EvolvedMeshGenerationError,
        EvolvedMeshValidationRuntimeError,
        EvolvedFacetLoadError,
        MaterialPointError,
        ElastoplasticAssemblyError,
        ElastoplasticSolverError,
        ElastoplasticConvergenceError,
        ElastoplasticRecoveryError,
        SurfaceRecoveryError,
        SinglePositionElastoplasticWorkflowError,
        SinglePositionElastoplasticArtifactError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Single-position elastoplastic error: {exc}", file=sys.stderr)
        return 2
    _print_result(input_path, summary, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
