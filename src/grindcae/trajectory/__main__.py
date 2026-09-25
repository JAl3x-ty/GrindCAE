"""Command-line entry point for phase 4C1 ideal single-pass profiles."""

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

from .core import RESULT_FORMAT, TrajectoryError, TrajectoryResult, build_surface_profile
from .exporters import (
    SURFACE_PROFILE_CSV_FIELDS,
    write_surface_profile_csv,
    write_surface_profile_png,
)
from .models import TrajectoryCase, TrajectoryValidationError


SUMMARY_FILENAME = "summary.json"
SURFACE_PROFILE_CSV_FILENAME = "surface_profile.csv"
SURFACE_PROFILE_PNG_FILENAME = "surface_profile.png"
ARTIFACT_FILENAMES = {
    "summary_json": SUMMARY_FILENAME,
    "surface_profile_csv": SURFACE_PROFILE_CSV_FILENAME,
    "surface_profile_png": SURFACE_PROFILE_PNG_FILENAME,
}


class TrajectoryWorkflowError(RuntimeError):
    """Raised when a validated phase 4C1 result cannot be exported safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a single-pass ideal circular-wheel trajectory and mean "
            "surface profile. No Gmsh, grinding-force, or FEM workflow is run."
        )
    )
    parser.add_argument("case", help="path to a trajectory schema version 1 JSON file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving summary.json, surface_profile.csv, and surface_profile.png",
    )
    return parser


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        key: (output_dir / filename).resolve()
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def _serialize_summary(summary: dict[str, object]) -> str:
    try:
        return (
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise TrajectoryWorkflowError(
            f"summary_payload [finite JSON]: serialization failed: {exc}"
        ) from exc


def _write_temporary_summary(serialized: str, path: Path) -> None:
    try:
        path.write_text(serialized, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise TrajectoryWorkflowError(
            f"temporary_summary [writable JSON]: could not be written: {exc}"
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
        raise TrajectoryWorkflowError(
            f"artifact [{destination.name}]: staging failed: {exc}"
        ) from exc


def _validate_temporary_artifacts(temporary_artifacts: dict[str, Path]) -> None:
    if set(temporary_artifacts) != set(ARTIFACT_FILENAMES):
        raise TrajectoryWorkflowError(
            "temporary artifact set does not match the phase 4C1 contract"
        )
    for key, path in temporary_artifacts.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise TrajectoryWorkflowError(
                f"artifact [{key}]: temporary result is missing or empty"
            )
    try:
        summary = json.loads(
            temporary_artifacts["summary_json"].read_text(encoding="utf-8")
        )
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        point_count = int(summary["sampling"]["final_actual_point_count"])
        width = float(summary["normalized_input"]["workpiece"]["length_m"])
        height = float(
            summary["normalized_input"]["workpiece"]["original_surface_height_m"]
        )
        depth = float(
            summary["normalized_input"]["single_pass"]["depth_of_cut_m"]
        )
        with temporary_artifacts["surface_profile_csv"].open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if reader.fieldnames != list(SURFACE_PROFILE_CSV_FIELDS):
            raise ValueError("surface CSV header does not match the export contract")
        if len(rows) != point_count:
            raise ValueError("surface CSV row count does not match the summary")
        point_ids = [int(row["point_id"]) for row in rows]
        if point_ids != list(range(point_count)):
            raise ValueError("surface CSV point IDs are not 0-based and contiguous")
        x_values = [float(row["x_m"]) for row in rows]
        if x_values[0] != 0.0 or x_values[-1] != width:
            raise ValueError("surface CSV must begin at x=0 and end at x=W")
        if any(not first < second for first, second in zip(x_values, x_values[1:])):
            raise ValueError("surface CSV x values are not strictly increasing")
        for row in rows:
            values = (
                float(row["x_m"]),
                float(row["surface_y_m"]),
                float(row["surface_elevation_relative_to_original_m"]),
                float(row["removed_depth_m"]),
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("surface CSV contains a non-finite value")
            surface_y = values[1]
            relative = values[2]
            removed = values[3]
            if row["region"] not in {"ground", "contact_arc", "unprocessed"}:
                raise ValueError("surface CSV contains an invalid region")
            if not math.isclose(relative, surface_y - height, abs_tol=1.0e-15):
                raise ValueError("surface CSV relative elevation identity failed")
            if not math.isclose(removed, height - surface_y, abs_tol=1.0e-15):
                raise ValueError("surface CSV removed-depth identity failed")
            if not -1.0e-15 <= removed <= depth + 1.0e-15:
                raise ValueError("surface CSV removed depth is outside [0, ae]")

        from matplotlib import image as matplotlib_image

        pixels = matplotlib_image.imread(temporary_artifacts["surface_profile_png"])
        if pixels.ndim not in (2, 3) or pixels.size == 0:
            raise ValueError("surface PNG is not a decodable image")
        if not math.isfinite(float(pixels.min())) or not math.isfinite(float(pixels.max())):
            raise ValueError("surface PNG contains non-finite pixel data")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrajectoryWorkflowError(
            f"temporary artifact validation failed: {exc}"
        ) from exc


def _publish_results(
    temporary_artifacts: dict[str, Path],
    artifacts: dict[str, Path],
) -> None:
    if set(temporary_artifacts) != set(artifacts) or set(artifacts) != set(
        ARTIFACT_FILENAMES
    ):
        raise TrajectoryWorkflowError(
            "publication artifact set does not match the phase 4C1 contract"
        )
    _validate_temporary_artifacts(temporary_artifacts)

    output_dir = artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrajectoryWorkflowError(
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
    except (OSError, TrajectoryWorkflowError) as exc:
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
        raise TrajectoryWorkflowError(detail) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _print_summary(
    input_path: Path,
    result: TrajectoryResult,
    artifacts: dict[str, Path],
) -> None:
    print(f"Input file: {input_path}")
    print(
        "Wheel lowest point and relative feed: "
        f"x={result.case.single_pass.wheel_lowest_point_x_m:.12g} m, "
        f"direction={result.case.single_pass.relative_feed_direction}"
    )
    print(f"Pass state: {result.pass_state}")
    print(
        "Strict circular projected length Larc: "
        f"{result.exact_arc_projected_length_m:.12g} m"
    )
    print(
        "Shallow-cut reference Lapprox: "
        f"{result.shallow_cut_approximation_length_m:.12g} m"
    )
    print(
        "Length comparison: "
        f"absolute difference={result.length_absolute_difference_m:.12g} m, "
        f"relative difference={result.length_relative_difference:.12g}"
    )
    effective = result.effective_arc_interval_m
    print(
        "Effective projected contact: "
        f"interval={effective}, length={result.effective_contact_length_m:.12g} m, "
        f"ratio={result.contact_ratio:.12g}"
    )
    print("Surface segments:")
    for segment in result.surface_segments:
        print(
            f"  {segment.region}: [{segment.x_start_m:.12g}, "
            f"{segment.x_end_m:.12g}] m"
        )
    print(
        "Sampling: "
        f"requested={result.case.sampling.base_point_count}, "
        f"actual={len(result.sample_points)}"
    )
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Scope: ideal 2D mean surface only; depth is prescribed, with no "
        "roughness, no elastic deformation, no grinding force, no contact "
        "pressure, no FEM stress, and no temperature calculation."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts = _artifact_paths(output_dir)
    try:
        case = TrajectoryCase.from_json(input_path)
        result = build_surface_profile(case)
        with tempfile.TemporaryDirectory(prefix="grindcae-trajectory-") as temp_dir:
            workspace = Path(temp_dir)
            temporary_artifacts = {
                key: workspace / filename
                for key, filename in ARTIFACT_FILENAMES.items()
            }
            write_surface_profile_csv(
                result, temporary_artifacts["surface_profile_csv"]
            )
            write_surface_profile_png(
                result, temporary_artifacts["surface_profile_png"]
            )
            summary = result.to_summary(
                {key: str(path) for key, path in artifacts.items()}
            )
            _write_temporary_summary(
                _serialize_summary(summary), temporary_artifacts["summary_json"]
            )
            _publish_results(temporary_artifacts, artifacts)
    except (
        TrajectoryValidationError,
        TrajectoryError,
        TrajectoryWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Trajectory error: {exc}", file=sys.stderr)
        return 2

    _print_summary(input_path, result, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
