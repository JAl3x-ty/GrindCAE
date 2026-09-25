"""CLI and transactional publication for Phase 6B.2."""

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

from .exporters import (
    HISTORY_CSV_FIELDS,
    RESULT_FORMAT,
    build_summary,
    write_elements_csv,
    write_force_history_png,
    write_history_csv,
    write_nodes_csv,
    write_overview_png,
    write_plastic_history_png,
    write_residual_state_png,
    write_summary_json,
    write_surface_profile_png,
    write_vtu,
)
from .models import FixedMeshElastoplasticHistoryPassCase, HistoryPassValidationError
from .field_evolution import (
    FIELD_EVOLUTION_DIRECTORY,
    FieldEvolutionArtifactError,
    FieldEvolutionExporter,
    validate_field_evolution,
)
from .workflow import (
    FixedMeshElastoplasticHistoryPassResult,
    HistoryPassWorkflowError,
    solve_fixed_mesh_elastoplastic_history_pass,
)


ARTIFACT_FILENAMES = {
    "summary_json": "pass_summary.json",
    "history_csv": "pass_history.csv",
    "vtu": "final_results.vtu",
    "nodes_csv": "final_nodes.csv",
    "elements_csv": "final_elements.csv",
    "reference_mesh_msh": "reference_mesh.msh",
    "moving_load_overview_png": "moving_load_overview.png",
    "force_history_png": "pass_force_history.png",
    "plastic_history_png": "pass_plastic_history.png",
    "residual_state_png": "pass_residual_state.png",
    "surface_profile_png": "final_surface_profile.png",
}


class HistoryPassArtifactError(RuntimeError):
    """Raised when Phase 6B.2 artifacts or transactional publication fail."""


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {key: directory / value for key, value in ARTIFACT_FILENAMES.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fixed-mesh complete-pass elastoplastic history analysis with final unloading.")
    parser.add_argument("case", help="schema version 1 UTF-8 JSON input")
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise HistoryPassArtifactError("artifact set is invalid")
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise HistoryPassArtifactError(f"artifact is missing or empty: {path.name}")
    try:
        text = artifacts["summary_json"].read_text(encoding="ascii")
        if "NaN" in text or "Infinity" in text:
            raise ValueError("summary contains a non-finite JSON token")
        summary = json.loads(text)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        if set(summary.get("artifacts", {})) != set(ARTIFACT_FILENAMES):
            raise ValueError("summary artifact map is invalid")
        with artifacts["history_csv"].open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream); rows = list(reader)
            if tuple(reader.fieldnames or ()) != HISTORY_CSV_FIELDS:
                raise ValueError("pass_history.csv header is invalid")
        if len(rows) != int(summary["pass"]["history_row_count_including_final_unloaded"]):
            raise ValueError("pass_history.csv row count is invalid")
        if rows[-1]["pass_state"] != "final_unloaded" or rows[-1]["state_committed"] != "true":
            raise ValueError("pass_history.csv final unloaded row is invalid")
        if math.hypot(float(rows[-1]["Fx_N"]), float(rows[-1]["Fy_N"])) > 1e-10:
            raise ValueError("final unloaded target force is nonzero")
        if not meshio.read(artifacts["reference_mesh_msh"]).points.size:
            raise ValueError("reference mesh is empty")
        vtu = meshio.read(artifacts["vtu"])
        if not vtu.points.size or "residual_equivalent_plastic_strain" not in vtu.cell_data_dict:
            raise ValueError("final_results.vtu is invalid")
        for key in ("moving_load_overview_png", "force_history_png", "plastic_history_png", "residual_state_png", "surface_profile_png"):
            if matplotlib_image.imread(artifacts[key]).size == 0:
                raise ValueError(f"{artifacts[key].name} is unreadable")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HistoryPassArtifactError(f"artifact validation failed: {exc}") from exc


def export_result(
    result: FixedMeshElastoplasticHistoryPassResult,
    output_dir: str | Path,
    *,
    summary_artifacts: dict[str, Path] | None = None,
    field_evolution_index: dict[str, object] | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    outputs = artifact_paths(output_dir)
    for path in outputs.values(): path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.moving_load.msh_source, outputs["reference_mesh_msh"])
    write_history_csv(result, outputs["history_csv"])
    write_vtu(result.maximum_plastic_fields, result.final_fields, outputs["vtu"])
    write_nodes_csv(result.maximum_plastic_fields, result.final_fields, outputs["nodes_csv"])
    write_elements_csv(result.maximum_plastic_fields, result.final_fields, outputs["elements_csv"])
    write_overview_png(result.moving_load, outputs["moving_load_overview_png"])
    write_force_history_png(result, outputs["force_history_png"])
    write_plastic_history_png(result, outputs["plastic_history_png"])
    write_residual_state_png(result, outputs["residual_state_png"])
    write_surface_profile_png(result, outputs["surface_profile_png"])
    summary = build_summary(
        result,
        summary_artifacts or outputs,
        field_evolution=field_evolution_index,
    )
    write_summary_json(summary, outputs["summary_json"])
    validate_artifacts(outputs)
    return outputs, summary


def _stage_copy(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, staged); return staged


def publish_artifacts(
    temporary: dict[str, Path],
    final: dict[str, Path],
    *,
    temporary_field_evolution: Path | None = None,
    final_field_evolution: Path | None = None,
) -> None:
    if set(temporary) != set(final) or set(final) != set(ARTIFACT_FILENAMES):
        raise HistoryPassArtifactError("publication set is invalid")
    validate_artifacts(temporary)
    if (temporary_field_evolution is None) != (final_field_evolution is None):
        raise HistoryPassArtifactError("field evolution publication pair is invalid")
    if temporary_field_evolution is not None:
        try:
            validate_field_evolution(temporary_field_evolution)
        except FieldEvolutionArtifactError as exc:
            raise HistoryPassArtifactError(str(exc)) from exc
    output = final["summary_json"].parent; existed = output.exists(); output.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}; backups: dict[str, Path] = {}; published: list[str] = []
    staged_sequence: Path | None = None
    backup_sequence: Path | None = None
    sequence_published = False
    try:
        for key in order: staged[key] = _stage_copy(temporary[key], final[key], "new")
        if temporary_field_evolution is not None and final_field_evolution is not None:
            staged_sequence = final_field_evolution.with_name(
                f".{final_field_evolution.name}.new.{uuid.uuid4().hex}.tmp"
            )
            shutil.copytree(temporary_field_evolution, staged_sequence)
            validate_field_evolution(staged_sequence)
        for key in order:
            if final[key].exists(): backups[key] = _stage_copy(final[key], final[key], "backup")
        for key in order[:-1]:
            os.replace(staged[key], final[key]); del staged[key]; published.append(key)
        if staged_sequence is not None and final_field_evolution is not None:
            if final_field_evolution.exists():
                candidate_backup = final_field_evolution.with_name(
                    f".{final_field_evolution.name}.backup.{uuid.uuid4().hex}.tmp"
                )
                os.replace(final_field_evolution, candidate_backup)
                backup_sequence = candidate_backup
            try:
                os.replace(staged_sequence, final_field_evolution)
                staged_sequence = None
                sequence_published = True
            except OSError:
                if backup_sequence is not None and not final_field_evolution.exists():
                    os.replace(backup_sequence, final_field_evolution)
                    backup_sequence = None
                raise
        summary_key = order[-1]
        os.replace(staged[summary_key], final[summary_key]); del staged[summary_key]; published.append(summary_key)
    except (OSError, HistoryPassArtifactError) as exc:
        errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups: os.replace(backups[key], final[key]); del backups[key]
                else: final[key].unlink(missing_ok=True)
            except OSError as rollback_exc: errors.append(f"{final[key].name}: {rollback_exc}")
        if sequence_published and final_field_evolution is not None:
            try:
                if backup_sequence is not None:
                    shutil.rmtree(final_field_evolution)
                    os.replace(backup_sequence, final_field_evolution)
                    backup_sequence = None
                else:
                    shutil.rmtree(final_field_evolution)
            except OSError as rollback_exc:
                errors.append(f"{final_field_evolution.name}: {rollback_exc}")
        elif backup_sequence is not None and final_field_evolution is not None:
            try:
                os.replace(backup_sequence, final_field_evolution)
                backup_sequence = None
            except OSError as rollback_exc:
                errors.append(f"{final_field_evolution.name}: {rollback_exc}")
        message = f"artifact publication failed: {exc}"
        if errors: message += "; " + "; ".join(errors)
        raise HistoryPassArtifactError(message) from exc
    finally:
        for path in (*staged.values(), *backups.values()): path.unlink(missing_ok=True)
        if staged_sequence is not None and staged_sequence.exists():
            shutil.rmtree(staged_sequence, ignore_errors=True)
        if backup_sequence is not None and backup_sequence.exists():
            shutil.rmtree(backup_sequence, ignore_errors=True)
        if not existed and output.exists():
            try: output.rmdir()
            except OSError: pass


def run_fixed_mesh_elastoplastic_history_pass(case: FixedMeshElastoplasticHistoryPassCase, output_dir: str | Path, progress=None) -> tuple[FixedMeshElastoplasticHistoryPassResult, dict[str, Path], dict[str, object]]:
    final = artifact_paths(output_dir)
    final_sequence = final["summary_json"].parent / FIELD_EVOLUTION_DIRECTORY
    with tempfile.TemporaryDirectory(prefix="grindcae-history-pass-") as temporary_directory:
        workspace = Path(temporary_directory)
        temporary_sequence = workspace / "artifacts" / FIELD_EVOLUTION_DIRECTORY
        field_exporter = FieldEvolutionExporter(temporary_sequence)
        result = solve_fixed_mesh_elastoplastic_history_pass(
            case,
            workspace / "solve",
            on_target_snapshot=field_exporter.on_target_snapshot,
            progress=progress,
        )
        if progress is not None:
            total = len(result.moving_load.positions)
            progress(total, total, "位置计算完成，正在生成场演化图片与数据。")
        manifest = field_exporter.finalize()
        field_index = {
            "directory": str(final_sequence.resolve()),
            "manifest": str((final_sequence / "manifest.json").resolve()),
            "position_history_csv": str((final_sequence / "position_history.csv").resolve()),
            "pvd": str((final_sequence / "field_evolution.pvd").resolve()),
            "frame_count": manifest["frame_count"],
            "sequence_coordinate": manifest["sequence_coordinate"],
            "color_ranges": manifest["color_ranges"],
        }
        temporary, summary = export_result(
            result,
            workspace / "artifacts",
            summary_artifacts=final,
            field_evolution_index=field_index,
        )
        publish_artifacts(
            temporary,
            final,
            temporary_field_evolution=temporary_sequence,
            final_field_evolution=final_sequence,
        )
        if progress is not None:
            progress(total, total, "结果已事务发布，正在执行完整结果验证。")
    return result, final, summary


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv); input_path = Path(args.case).expanduser().resolve()
    try:
        case = FixedMeshElastoplasticHistoryPassCase.from_json(input_path)
        _, artifacts, summary = run_fixed_mesh_elastoplastic_history_pass(case, args.output_dir)
    except (HistoryPassValidationError, HistoryPassWorkflowError, HistoryPassArtifactError, OSError, RuntimeError, ValueError) as exc:
        print(f"History-pass error: {exc}", file=sys.stderr); return 2
    mesh = summary["mesh_reuse"]; history = summary["pass"]; final = summary["final_unloaded"]
    print(f"Input file: {input_path}")
    print(f"Fixed mesh: generated={mesh['generation_count']}, imported={mesh['import_count']}, nodes={mesh['node_count']}, triangles={mesh['triangle_count']}")
    print(f"History: positions={history['target_position_count']}, substeps={history['total_transition_substep_count']}, retries={history['total_retry_count']}, max Newton={history['maximum_newton_iterations']}")
    print(f"Final unloaded: external norm={final['external_force_norm_N']:.12g} N, balance={final['balance_residual_N']:.12g} N, residual displacement={final['maximum_residual_displacement_m']:.12g} m")
    print("Output files:")
    for path in artifacts.values(): print(f"  {path}")
    sequence = summary.get("field_evolution")
    if isinstance(sequence, dict):
        print(
            "Field evolution: "
            f"frames={sequence['frame_count']}, directory={sequence['directory']}"
        )
    print(
        "Manual acceptance remains pending for representative entry, full-contact, "
        "and exit field-evolution PNGs, plus retained core acceptance figures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
