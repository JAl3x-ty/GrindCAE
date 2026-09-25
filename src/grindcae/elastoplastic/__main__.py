"""Command-line workflow for phase 6A.1 material-point loading and unloading."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .exporters import HISTORY_CSV_FIELDS, write_history_csv, write_stress_strain_png
from .j2 import MaterialPointError
from .models import MaterialPointCase, MaterialPointValidationError
from .uniaxial import (
    UniaxialDriverError,
    UniaxialHistoryResult,
    run_uniaxial_loading_unloading,
)
from .workflow import RESULT_FORMAT, build_summary


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "history_csv": "history.csv",
    "stress_strain_png": "stress_strain.png",
}


class ElastoplasticWorkflowError(RuntimeError):
    """Raised when phase 6A.1 artifacts cannot be exported or published safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an independent small-strain J2 bilinear-isotropic-hardening "
            "material point through uniaxial loading and zero-stress unloading. "
            "No mesh, FEM, grinding, GUI, or packaging workflow is run."
        )
    )
    parser.add_argument(
        "case", help="path to a material-point schema version 1 JSON file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving summary.json, history.csv, and stress_strain.png",
    )
    return parser


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {
        key: directory / filename for key, filename in ARTIFACT_FILENAMES.items()
    }


def _serialize_summary(summary: dict[str, object]) -> str:
    try:
        return json.dumps(
            summary, indent=2, ensure_ascii=True, allow_nan=False
        ) + "\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise ElastoplasticWorkflowError(
            f"summary_payload [finite JSON]: serialization failed: {exc}"
        ) from exc


def _stage_copy(source: Path, destination: Path, label: str) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.{label}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary_path
    except OSError as exc:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise ElastoplasticWorkflowError(
            f"artifact [{destination.name}]: {label} staging failed: {exc}"
        ) from exc


def validate_artifacts(artifacts: dict[str, Path]) -> None:
    """Validate one complete three-file 6A.1 artifact set."""

    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise ElastoplasticWorkflowError(
            "artifact set does not match the phase 6A.1 contract"
        )
    for key, path in artifacts.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ElastoplasticWorkflowError(f"artifact [{key}] is missing or empty")
    try:
        serialized = artifacts["summary_json"].read_text(encoding="ascii")
        if "NaN" in serialized or "Infinity" in serialized:
            raise ValueError("summary contains a non-finite JSON token")
        summary = json.loads(serialized)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        response = summary["uniaxial_response"]
        step_count = int(
            response["step_counts"]["total_converged_states_including_initial"]
        )
        with artifacts["history_csv"].open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if reader.fieldnames != list(HISTORY_CSV_FIELDS):
            raise ValueError("history CSV header does not match the export contract")
        if len(rows) != step_count:
            raise ValueError("history CSV row count does not match the summary")
        if [int(row["step_id"]) for row in rows] != list(range(step_count)):
            raise ValueError("history CSV step IDs are not 0-based and contiguous")
        for row in rows:
            if row["segment"] not in {"initial", "loading", "unloading"}:
                raise ValueError("history CSV contains an invalid segment")
            if row["increment_class"] not in {"initial", "elastic", "plastic"}:
                raise ValueError("history CSV contains an invalid increment class")
            for field in HISTORY_CSV_FIELDS:
                if field in {"segment", "increment_class"}:
                    continue
                if not math.isfinite(float(row[field])):
                    raise ValueError("history CSV contains a non-finite numeric value")

        from matplotlib import image as matplotlib_image

        pixels = matplotlib_image.imread(artifacts["stress_strain_png"])
        if pixels.ndim not in (2, 3) or pixels.size == 0:
            raise ValueError("stress-strain PNG is not a decodable image")
        if not math.isfinite(float(pixels.min())) or not math.isfinite(
            float(pixels.max())
        ):
            raise ValueError("stress-strain PNG contains non-finite pixel data")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ElastoplasticWorkflowError(
            f"artifact validation failed: {exc}"
        ) from exc


def publish_artifacts(
    temporary_artifacts: dict[str, Path], final_artifacts: dict[str, Path]
) -> None:
    """Publish all three artifacts with summary last and rollback on failure."""

    if set(temporary_artifacts) != set(final_artifacts) or set(
        final_artifacts
    ) != set(ARTIFACT_FILENAMES):
        raise ElastoplasticWorkflowError("artifact publication set is invalid")
    validate_artifacts(temporary_artifacts)
    output_dir = final_artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ElastoplasticWorkflowError(
            f"output_directory [writable directory]: could not be created: {exc}"
        ) from exc

    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + [
        "summary_json"
    ]
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in order:
            staged[key] = _stage_copy(
                temporary_artifacts[key], final_artifacts[key], "new"
            )
        for key in order:
            if final_artifacts[key].exists():
                backups[key] = _stage_copy(
                    final_artifacts[key], final_artifacts[key], "backup"
                )
        for key in order:
            os.replace(staged[key], final_artifacts[key])
            del staged[key]
            published.append(key)
    except (OSError, ElastoplasticWorkflowError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups:
                    os.replace(backups[key], final_artifacts[key])
                    del backups[key]
                else:
                    final_artifacts[key].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{final_artifacts[key].name} rollback failed: {rollback_exc}"
                )
        detail = f"artifact publication failed: {exc}"
        if rollback_errors:
            detail += "; " + "; ".join(rollback_errors)
        raise ElastoplasticWorkflowError(detail) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _print_result(
    input_path: Path,
    result: UniaxialHistoryResult,
    artifacts: dict[str, Path],
) -> None:
    material = result.case.material
    print(f"Input file: {input_path}")
    print(
        "Material model: small-strain J2 bilinear isotropic hardening, "
        f"E={material.E:.12g} Pa, nu={material.nu:.12g}, "
        f"yield={material.yield_strength:.12g} Pa, "
        f"Et={material.tangent_modulus:.12g} Pa, "
        f"H={material.internal_hardening_modulus:.12g} Pa"
    )
    print(
        "Uniaxial loading: "
        f"peak strain={result.case.history.peak_axial_strain:.12g}, "
        f"loading steps={result.case.history.loading_step_count}, "
        f"unloading steps={result.case.history.unloading_step_count}"
    )
    print(f"First yielding step: {result.first_yield_step_id}")
    print(
        "Peak axial response: "
        f"strain={result.peak_step.axial_total_strain:.12g}, "
        f"stress={result.peak_step.axial_stress_Pa:.12g} Pa, "
        f"equivalent plastic strain={result.peak_step.state.equivalent_plastic_strain:.12g}"
    )
    print(
        "Final unloaded response: "
        f"residual strain={result.final_step.axial_total_strain:.12g}, "
        f"axial stress={result.final_step.axial_stress_Pa:.12g} Pa, "
        f"lateral stresses=({result.final_step.transverse_stress_y_Pa:.12g}, "
        f"{result.final_step.transverse_stress_z_Pa:.12g}) Pa"
    )
    print(
        "Maximum transverse stress: "
        f"{result.maximum_absolute_transverse_stress_Pa:.12g} Pa"
    )
    print(
        "Maximum plastic yield-function residual: "
        f"{result.maximum_plastic_yield_function_residual_Pa:.12g} Pa"
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: independent material-point loading-unloading only; no FEM, "
        "no grinding load/contact, no complete-pass state transfer, no GUI, and no packaging update."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    final_artifacts = artifact_paths(args.output_dir)
    try:
        case = MaterialPointCase.from_json(input_path)
        result = run_uniaxial_loading_unloading(case)
        with tempfile.TemporaryDirectory(prefix="grindcae-elastoplastic-") as temp_dir:
            workspace = Path(temp_dir)
            temporary_artifacts = {
                key: workspace / filename
                for key, filename in ARTIFACT_FILENAMES.items()
            }
            write_history_csv(result, temporary_artifacts["history_csv"])
            write_stress_strain_png(
                result, temporary_artifacts["stress_strain_png"]
            )
            summary = build_summary(
                result, {key: str(path) for key, path in final_artifacts.items()}
            )
            temporary_artifacts["summary_json"].write_text(
                _serialize_summary(summary), encoding="ascii", newline="\n"
            )
            publish_artifacts(temporary_artifacts, final_artifacts)
    except (
        MaterialPointValidationError,
        MaterialPointError,
        UniaxialDriverError,
        ElastoplasticWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Elastoplastic material-point error: {exc}", file=sys.stderr)
        return 2

    _print_result(input_path, result, final_artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
