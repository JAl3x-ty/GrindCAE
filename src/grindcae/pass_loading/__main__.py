"""Command-line workflow for phase 4C2 pass-load snapshots."""

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

from matplotlib import image as matplotlib_image

from .core import RESULT_FORMAT, PassLoadError, PassLoadResult, build_pass_load
from .exporters import PASS_LOAD_CSV_FIELDS, write_pass_load_csv, write_pass_load_png
from .models import PassLoadCase, PassLoadValidationError


SUMMARY_FILENAME = "summary.json"
CSV_FILENAME = "pass_load.csv"
PNG_FILENAME = "pass_load.png"
ARTIFACT_FILENAMES = {
    "summary_json": SUMMARY_FILENAME,
    "pass_load_csv": CSV_FILENAME,
    "pass_load_png": PNG_FILENAME,
}


class PassLoadWorkflowError(RuntimeError):
    """Raised when a validated phase 4C2 result cannot be exported safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine an ideal single-pass trajectory with a steady empirical "
            "steady empirical force to create a current projected-load snapshot. "
            "No meshing or FEM workflow is run."
        )
    )
    parser.add_argument("case", help="path to a pass-load JSON input")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving summary.json, pass_load.csv, and pass_load.png",
    )
    return parser


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        key: (output_dir / filename).resolve()
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def _serialize_summary(summary: dict[str, object]) -> str:
    try:
        return json.dumps(
            summary,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise PassLoadWorkflowError(
            f"summary_payload [finite JSON]: serialization failed: {exc}"
        ) from exc


def _write_temporary_summary(serialized: str, path: Path) -> None:
    try:
        path.write_text(serialized, encoding="ascii")
    except OSError as exc:
        raise PassLoadWorkflowError(
            f"temporary_summary [writable JSON]: could not be written: {exc}"
        ) from exc


def _stage_copy(source: Path, destination: Path, label: str) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.{label}-",
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
        raise PassLoadWorkflowError(
            f"{destination.name} [{label} staging]: could not be staged: {exc}"
        ) from exc


def _validate_temporary_artifacts(temporary_artifacts: dict[str, Path]) -> None:
    if set(temporary_artifacts) != set(ARTIFACT_FILENAMES):
        raise PassLoadWorkflowError(
            "temporary artifact set does not match the phase 4C2 contract"
        )
    for key, path in temporary_artifacts.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise PassLoadWorkflowError(
                f"artifact [{key}]: temporary result is missing or empty"
            )

    try:
        serialized = temporary_artifacts["summary_json"].read_text(encoding="ascii")
        if "NaN" in serialized or "Infinity" in serialized:
            raise ValueError("summary contains a non-finite JSON token")
        summary = json.loads(serialized)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        target_fx = float(
            summary["current_empirical_load_snapshot"]["signed_Fx_N"]
        )
        target_fy = float(
            summary["current_empirical_load_snapshot"]["signed_Fy_N"]
        )
        width = float(summary["normalized_input"]["trajectory"]["workpiece"]["length_m"])
        interval = summary["trajectory_contact"]["effective_projected_interval_m"]

        with temporary_artifacts["pass_load_csv"].open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        if not rows or tuple(rows[0]) != PASS_LOAD_CSV_FIELDS:
            raise ValueError("pass-load CSV header or rows are invalid")
        if [int(row["segment_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError("pass-load CSV indices are not 0-based contiguous")
        starts = [float(row["x_start_m"]) for row in rows]
        ends = [float(row["x_end_m"]) for row in rows]
        lengths = [float(row["length_m"]) for row in rows]
        if starts[0] != 0.0 or ends[-1] != width:
            raise ValueError("pass-load CSV does not cover exact [0, W]")
        if any(not start < end for start, end in zip(starts, ends)):
            raise ValueError("pass-load CSV contains a non-positive segment")
        if any(
            not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for left, right in zip(ends[:-1], starts[1:])
        ):
            raise ValueError("pass-load CSV segments are not contiguous")
        if not math.isclose(math.fsum(lengths), width, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError("pass-load CSV segment lengths do not sum to W")
        active_rows = [
            row for row in rows if row["segment_type"] == "active_projected_contact"
        ]
        if interval is None:
            if active_rows or len(rows) != 1 or rows[0]["segment_type"] != "unloaded":
                raise ValueError("zero-contact CSV partition is invalid")
        else:
            if len(active_rows) != 1:
                raise ValueError("nonzero contact requires exactly one active CSV row")
            active = active_rows[0]
            if float(active["x_start_m"]) != float(interval[0]) or float(
                active["x_end_m"]
            ) != float(interval[1]):
                raise ValueError("active CSV endpoints do not match phase 4C1")
        integrated_fx = math.fsum(float(row["integrated_Fx_N"]) for row in rows)
        integrated_fy = math.fsum(float(row["integrated_Fy_N"]) for row in rows)
        if not math.isclose(integrated_fx, target_fx, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("pass-load CSV Fx integral does not match the summary")
        if not math.isclose(integrated_fy, target_fy, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("pass-load CSV Fy integral does not match the summary")

        pixels = matplotlib_image.imread(temporary_artifacts["pass_load_png"])
        if pixels.size == 0 or pixels.ndim < 2:
            raise ValueError("pass-load PNG is unreadable or empty")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PassLoadWorkflowError(
            f"temporary artifact validation failed: {exc}"
        ) from exc


def _publish_results(
    temporary_artifacts: dict[str, Path], artifacts: dict[str, Path]
) -> None:
    if set(temporary_artifacts) != set(artifacts) or set(artifacts) != set(
        ARTIFACT_FILENAMES
    ):
        raise PassLoadWorkflowError("artifact publication set is invalid")
    _validate_temporary_artifacts(temporary_artifacts)

    output_dir = artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PassLoadWorkflowError(
            f"output_directory [writable directory]: could not be created: {exc}"
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
    except (OSError, PassLoadWorkflowError) as exc:
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
        raise PassLoadWorkflowError(detail) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _print_summary(input_path: Path, result: PassLoadResult, artifacts: dict[str, Path]) -> None:
    print(f"Input file: {input_path}")
    print(
        "Trajectory and mapping directions: "
        f"relative_feed={result.case.trajectory.single_pass.relative_feed_direction}, "
        f"tangential_force={result.case.force_mapping.tangential_force_direction}"
    )
    print(f"Pass state: {result.pass_state}")
    print(
        "Projected contact: "
        f"interval={result.effective_contact_interval_m}, "
        f"Leff={result.effective_contact_length_m:.12g} m, "
        f"contact_ratio={result.contact_ratio:.12g}"
    )
    print(
        "Length bases: "
        f"arc projected length Larc={result.exact_arc_projected_length_m:.12g} m, "
        f"steady geometric contact length lg={result.phase_4a_geometric_contact_length_m:.12g} m, "
        f"relative difference={result.length_relative_difference:.12g}"
    )
    print(
        "Steady full-contact empirical force magnitudes: "
        f"Ft={result.steady_tangential_force_magnitude_N:.12g} N, "
        f"Fn={result.steady_normal_force_magnitude_N:.12g} N, "
        f"Fr={result.steady_resultant_force_magnitude_N:.12g} N"
    )
    print(
        "Current empirical load snapshot: "
        f"Fx={result.current_Fx_N:.12g} N, Fy={result.current_Fy_N:.12g} N, "
        f"qx={result.current_qx_N_per_m:.12g} N/m, "
        f"qy={result.current_qy_N_per_m:.12g} N/m"
    )
    print(
        "Load integral residual: "
        f"Fx={result.force_integral_residual_Fx_N:.12g} N, "
        f"Fy={result.force_integral_residual_Fy_N:.12g} N"
    )
    print(f"Calibration: {result.case.force_model.calibration.calibration_id}")
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: basic contact-length-ratio empirical scaling, unvalidated by "
        "user experiments. The effective projected interval is not Hertz "
        "contact, and the uniform line load is not a true abrasive-grain "
        "pressure distribution. No Gmsh, meshio, scikit-fem, FEM, stress, "
        "temperature, impact, vibration, chatter, wheel wear, grain randomness, "
        "ploughing, plasticity, edge chipping, or multiple-pass model is run."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts = _artifact_paths(output_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="grindcae-pass-loading-") as temp_dir:
            workspace = Path(temp_dir)
            case = PassLoadCase.from_json(input_path)
            result = build_pass_load(case)
            temporary_artifacts = {
                key: workspace / filename
                for key, filename in ARTIFACT_FILENAMES.items()
            }
            write_pass_load_csv(result, temporary_artifacts["pass_load_csv"])
            write_pass_load_png(result, temporary_artifacts["pass_load_png"])
            summary = result.to_summary(
                {key: str(path) for key, path in artifacts.items()}
            )
            _write_temporary_summary(
                _serialize_summary(summary), temporary_artifacts["summary_json"]
            )
            _publish_results(temporary_artifacts, artifacts)
    except (
        PassLoadValidationError,
        PassLoadError,
        PassLoadWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Pass-load error: {exc}", file=sys.stderr)
        return 2

    _print_summary(input_path, result, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
