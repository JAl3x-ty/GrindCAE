"""CLI and transactional five-artifact publication for Phase 6B.1."""

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

from grindcae.mesh import MeshGenerationError
from grindcae.pass_loading import PassLoadError
from grindcae.solver import SolverError

from .exporters import (
    HISTORY_CSV_FIELDS,
    RESULT_FORMAT,
    TOP_FACET_CSV_FIELDS,
    build_summary,
    write_history_csv,
    write_overview_png,
    write_summary_json,
    write_top_facets_csv,
)
from .loading import MovingLoadMappingError
from .models import FixedMeshMovingLoadPassCase, MovingLoadPassValidationError
from .overlap import MovingLoadOverlapError
from .positions import MovingLoadPositionError
from .workflow import (
    FixedMeshMovingLoadPassResult,
    MovingLoadPassWorkflowError,
    solve_fixed_mesh_moving_load_pass,
)


ARTIFACT_FILENAMES = {
    "summary_json": "pass_summary.json",
    "history_csv": "moving_load_history.csv",
    "overview_png": "moving_load_overview.png",
    "reference_mesh_msh": "reference_mesh.msh",
    "top_facets_csv": "top_facets.csv",
}


class MovingLoadPassArtifactError(RuntimeError):
    """Raised when Phase 6B.1 artifacts or publication are invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
        "Run a fixed-mesh complete single-pass moving-load geometry "
            "and exact partial-facet force-conservation validation."
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
        raise MovingLoadPassArtifactError("artifact set is invalid")
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise MovingLoadPassArtifactError(f"artifact is missing or empty: {path.name}")
    try:
        serialized = artifacts["summary_json"].read_text(encoding="ascii")
        if "NaN" in serialized or "Infinity" in serialized:
            raise ValueError("summary contains a non-finite JSON token")
        summary = json.loads(serialized)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        if set(summary.get("artifacts", {})) != set(ARTIFACT_FILENAMES):
            raise ValueError("summary artifact map is invalid")
        position_count = int(summary["position_sequence"]["count"])
        top_count = int(summary["mesh_reuse"]["top_facet_count"])
        history_header, history = _read_csv(artifacts["history_csv"])
        facet_header, facets = _read_csv(artifacts["top_facets_csv"])
        if history_header != HISTORY_CSV_FIELDS or len(history) != position_count:
            raise ValueError("moving_load_history.csv header or row count is invalid")
        if facet_header != TOP_FACET_CSV_FIELDS or len(facets) != top_count:
            raise ValueError("top_facets.csv header or row count is invalid")
        if float(history[0]["projected_contact_length_m"]) != 0.0 or float(
            history[-1]["projected_contact_length_m"]
        ) != 0.0:
            raise ValueError("complete-pass endpoint contact lengths are not zero")
        maximum_residual = max(float(row["force_balance_residual_N"]) for row in history)
        declared_residual = float(summary["force_conservation"]["maximum_balance_residual_N"])
        if not math.isclose(maximum_residual, declared_residual, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("history and summary force residuals disagree")
        msh = meshio.read(artifacts["reference_mesh_msh"])
        if not msh.points.size:
            raise ValueError("reference mesh is empty")
        pixels = matplotlib_image.imread(artifacts["overview_png"])
        if pixels.size == 0 or pixels.ndim < 2:
            raise ValueError("moving_load_overview.png is unreadable")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MovingLoadPassArtifactError(f"artifact validation failed: {exc}") from exc


def export_result(
    result: FixedMeshMovingLoadPassResult,
    output_dir: str | Path,
    *,
    summary_artifacts: dict[str, Path] | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    outputs = artifact_paths(output_dir)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.msh_source, outputs["reference_mesh_msh"])
    write_history_csv(result, outputs["history_csv"])
    write_top_facets_csv(result, outputs["top_facets_csv"])
    write_overview_png(result, outputs["overview_png"])
    summary = build_summary(result, summary_artifacts or outputs)
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
        raise MovingLoadPassArtifactError("publication set is invalid")
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
    except (OSError, MovingLoadPassArtifactError) as exc:
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
        raise MovingLoadPassArtifactError(message) from exc
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)
        if not existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def run_fixed_mesh_moving_load_pass(
    case: FixedMeshMovingLoadPassCase,
    output_dir: str | Path,
) -> tuple[FixedMeshMovingLoadPassResult, dict[str, Path], dict[str, object]]:
    final_artifacts = artifact_paths(output_dir)
    with tempfile.TemporaryDirectory(prefix="grindcae-moving-load-pass-") as temp_dir:
        workspace = Path(temp_dir)
        result = solve_fixed_mesh_moving_load_pass(case, workspace / "solve")
        temporary, summary = export_result(
            result,
            workspace / "artifacts",
            summary_artifacts=final_artifacts,
        )
        publish_artifacts(temporary, final_artifacts)
        published_mesh = replace(
            result.reference_mesh,
            source_path=final_artifacts["reference_mesh_msh"],
        )
        result = replace(
            result,
            reference_mesh=published_mesh,
            msh_source=final_artifacts["reference_mesh_msh"],
        )
    return result, final_artifacts, summary


def _print_result(
    input_path: Path,
    summary: dict[str, object],
    artifacts: dict[str, Path],
) -> None:
    positions = summary["position_sequence"]
    mesh = summary["mesh_reuse"]
    force = summary["force_conservation"]
    partial = summary["partial_facet_validation"]
    print(f"Input file: {input_path}")
    print(
        f"Fixed mesh: generated={mesh['generation_count']}, imported={mesh['import_count']}, "
        f"nodes={mesh['node_count']}, triangles={mesh['triangle_count']}, top facets={mesh['top_facet_count']}"
    )
    print(
        f"Positions: {positions['count']}, direction={positions['motion_direction']}, "
        f"entry={positions['entry_motion_coordinate_m']:.12g} m, "
        f"full=[{positions['full_contact_start_motion_coordinate_m']:.12g}, "
        f"{positions['full_contact_end_motion_coordinate_m']:.12g}] m, "
        f"exit={positions['exit_motion_coordinate_m']:.12g} m"
    )
    print(
        f"Maximums: contact ratio={positions['maximum_contact_ratio']:.12g}, "
        f"|Fx|={force['maximum_absolute_Fx_N']:.12g} N, "
        f"|Fy|={force['maximum_absolute_Fy_N']:.12g} N, "
        f"force residual={force['maximum_balance_residual_N']:.12g} N"
    )
    print(
        f"Partial facets: positions={partial['positions_with_partial_facets']}, "
        f"maximum per position={partial['maximum_partial_facet_count']}"
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: exact moving-load geometry and force conservation on one fixed reference mesh; "
        "no Newton solve and no plastic-state transfer between positions."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    try:
        case = FixedMeshMovingLoadPassCase.from_json(input_path)
        _, artifacts, summary = run_fixed_mesh_moving_load_pass(case, args.output_dir)
    except (
        MovingLoadPassValidationError,
        MovingLoadPositionError,
        MovingLoadOverlapError,
        MovingLoadMappingError,
        MovingLoadPassWorkflowError,
        MovingLoadPassArtifactError,
        MeshGenerationError,
        SolverError,
        PassLoadError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Moving-load-pass error: {exc}", file=sys.stderr)
        return 2
    _print_result(input_path, summary, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
