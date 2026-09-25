"""Aggregate CSV, PNG, JSON, snapshot export, validation, and publication."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_TITLES_EN

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable
import uuid

import matplotlib.image as matplotlib_image

from grindcae import __version__
from grindcae.evolved_fem import (
    ARTIFACT_FILENAMES as SNAPSHOT_ARTIFACT_FILENAMES,
    EvolvedFemOperations,
    artifact_paths as snapshot_artifact_paths,
    export_evolved_fem_result,
    validate_evolved_fem_artifacts,
)

from grindcae.plotting import plot_style, save_png

from .core import RESULT_FORMAT, PassScanResult, ScanHistoryRow


AGGREGATE_ARTIFACT_FILENAMES = {
    "scan_summary_json": "scan_summary.json",
    "scan_history_csv": "scan_history.csv",
    "scan_overview_png": "scan_overview.png",
    "snapshots": "snapshots",
}
SCAN_HISTORY_CSV_FIELDS = tuple(ScanHistoryRow.__dataclass_fields__)


class PassScanExportError(RuntimeError):
    """Raised when aggregate export or atomic publication is invalid."""


def aggregate_artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {
        key: (directory / filename).resolve()
        for key, filename in AGGREGATE_ARTIFACT_FILENAMES.items()
    }


def _snapshot_slug(roles: tuple[str, ...]) -> str:
    for role, slug in (
        ("representative_entry", "entry"),
        ("representative_full_contact", "full_contact"),
        ("representative_exit", "exit"),
    ):
        if role in roles:
            return slug
    return "maximum_response"


def snapshot_directories(
    result: PassScanResult,
    root: str | Path,
) -> dict[int, Path]:
    root_path = Path(root).expanduser().resolve()
    values: dict[int, Path] = {}
    for index in sorted(result.retained_results):
        roles = result.history[index].snapshot_roles
        values[index] = root_path / f"point_{index:04d}_{_snapshot_slug(roles)}"
    return values


def write_scan_history_csv(result: PassScanResult, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCAN_HISTORY_CSV_FIELDS)
        writer.writeheader()
        for row in result.history:
            writer.writerow(row.to_dict())
    return path


def _scan_progress_percent(rows: tuple[ScanHistoryRow, ...]) -> list[float]:
    start = rows[0].motion_coordinate_m
    end = rows[-1].motion_coordinate_m
    span = end - start
    if not math.isfinite(span) or span <= 0.0:
        raise PassScanExportError("scan motion coordinates must have a positive finite span")
    values = [100.0 * (row.motion_coordinate_m - start) / span for row in rows]
    values[0] = 0.0
    values[-1] = 100.0
    return values


def _style_scan_axis(axis, entry_end_percent: float, exit_start_percent: float) -> None:
    axis.axvspan(0.0, entry_end_percent, color="#DCEAF7", alpha=0.55, zorder=0)
    axis.axvspan(exit_start_percent, 100.0, color="#FCE5D2", alpha=0.60, zorder=0)
    axis.axvline(entry_end_percent, color="#7AA6C2", linewidth=0.9, linestyle="--")
    axis.axvline(exit_start_percent, color="#D8905B", linewidth=0.9, linestyle="--")
    axis.set_xlim(0.0, 100.0)
    axis.set_xticks((0.0, 25.0, 50.0, 75.0, 100.0))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def write_scan_overview_png(result: PassScanResult, output_path: str | Path) -> Path:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = result.history
    progress = _scan_progress_percent(rows)
    width_m = result.case.reference_case.geometry.width
    ground_percent = [100.0 * row.ground_projected_length_m / width_m for row in rows]
    arc_percent = [100.0 * row.contact_arc_projected_length_m / width_m for row in rows]
    unprocessed_percent = [
        100.0 * row.unprocessed_projected_length_m / width_m for row in rows
    ]
    contact_percent = [100.0 * row.contact_ratio for row in rows]
    tangential_force = [abs(row.current_Fx_N) for row in rows]
    normal_compression = [-row.current_Fy_N for row in rows]
    displacement_um = [1.0e6 * row.maximum_displacement_m for row in rows]
    stress_p95_MPa = [1.0e-6 * row.von_mises_area_weighted_p95_Pa for row in rows]
    stress_p99_MPa = [1.0e-6 * row.von_mises_area_weighted_p99_Pa for row in rows]

    arc_m = result.reference_pass_load.trajectory_result.exact_arc_projected_length_m
    motion_span_m = width_m + arc_m
    entry_end_percent = 100.0 * arc_m / motion_span_m
    exit_start_percent = 100.0 * width_m / motion_span_m
    direction = "+x" if rows[0].relative_feed_direction == "positive_x" else "-x"

    with plt.rc_context(plot_style()):
        fig, grid = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=True)
        axes = tuple(grid.flat)
        fig.subplots_adjust(
            left=0.075,
            right=0.975,
            bottom=0.13,
            top=0.84,
            hspace=0.34,
            wspace=0.22,
        )
        try:
            for axis in axes:
                _style_scan_axis(axis, entry_end_percent, exit_start_percent)

            axes[0].stackplot(
                progress,
                ground_percent,
                arc_percent,
                unprocessed_percent,
                colors=("#3B82B4", "#F2A541", "#DCE3E8"),
                labels=("Ground", "Contact arc", "Unprocessed"),
                edgecolor="white",
                linewidth=0.6,
            )
            axes[0].plot(
                progress,
                contact_percent,
                color="#202020",
                linewidth=2.4,
                label="Contact ratio",
                zorder=4,
            )
            axes[0].set_ylim(0.0, 108.0)
            axes[0].set_ylabel("Share of workpiece / contact [%]")
            axes[0].set_title("A  Contact and evolving surface", loc="left", fontweight="bold")
            middle_index = result.snapshot_role_indices["representative_full_contact"]
            middle_x = progress[middle_index]
            middle_ground = ground_percent[middle_index]
            middle_arc = arc_percent[middle_index]
            axes[0].text(
                middle_x,
                0.50 * middle_ground,
                "GROUND",
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
            axes[0].text(
                middle_x,
                middle_ground + middle_arc + 0.50 * unprocessed_percent[middle_index],
                "UNPROCESSED",
                color="#4D5960",
                ha="center",
                va="center",
                fontweight="bold",
            )
            axes[0].annotate(
                f"contact arc ({1.0e3 * arc_m:.2f} mm projected)",
                xy=(middle_x, middle_ground + 0.5 * middle_arc),
                xytext=(middle_x + 8.0, middle_ground + 13.0),
                color="#9A5A08",
                fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": "#9A5A08", "linewidth": 0.9},
            )
            axes[0].text(
                25.0,
                101.5,
                "100% contact plateau",
                color="#202020",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
            axes[0].text(
                0.5 * entry_end_percent,
                104.0,
                "ENTRY",
                color="#39799E",
                rotation=90,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )
            axes[0].text(
                50.0,
                104.0,
                "FULL CONTACT",
                color="#666666",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
            axes[0].text(
                0.5 * (exit_start_percent + 100.0),
                104.0,
                "EXIT",
                color="#B4672E",
                rotation=90,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )

            axes[1].plot(
                progress,
                normal_compression,
                color="#D55E00",
                linewidth=2.2,
                label="Normal compression |Fy|",
            )
            axes[1].fill_between(
                progress, normal_compression, color="#D55E00", alpha=0.10
            )
            axes[1].plot(
                progress,
                tangential_force,
                color="#2F6F9F",
                linewidth=2.2,
                label="Tangential |Fx|",
            )
            axes[1].fill_between(
                progress, tangential_force, color="#2F6F9F", alpha=0.10
            )
            maximum_force = max((*normal_compression, *tangential_force))
            axes[1].set_ylim(0.0, 1.16 * maximum_force if maximum_force > 0.0 else 1.0)
            axes[1].set_ylabel("Force magnitude [N]")
            axes[1].set_title("B  Empirical load through the pass", loc="left", fontweight="bold")
            axes[1].legend(loc="upper left", ncol=2, fontsize=8.5)
            axes[1].text(
                0.98,
                0.06,
                "Compression is plotted upward as -Fy; signed values remain in CSV/JSON.",
                transform=axes[1].transAxes,
                ha="right",
                va="bottom",
                color="#666666",
                fontsize=7.8,
            )

            displacement_index = result.snapshot_role_indices["maximum_displacement"]
            displacement_value = displacement_um[displacement_index]
            axes[2].plot(
                progress,
                displacement_um,
                color="#7E57C2",
                linewidth=2.2,
            )
            axes[2].fill_between(
                progress, displacement_um, color="#7E57C2", alpha=0.12
            )
            axes[2].scatter(
                [progress[displacement_index]],
                [displacement_value],
                color="#7E57C2",
                edgecolor="white",
                linewidth=0.8,
                s=52,
                zorder=5,
            )
            axes[2].annotate(
                f"maximum {displacement_value:.3f} um\nat {progress[displacement_index]:.1f}% pass progress",
                (progress[displacement_index], displacement_value),
                xytext=(-118, -34),
                textcoords="offset points",
                ha="left",
                va="top",
                color="#6842A0",
                fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": "#6842A0", "linewidth": 0.9},
            )
            axes[2].set_ylim(bottom=0.0)
            axes[2].set_ylabel("Maximum displacement [um]")
            axes[2].set_xlabel("Pass progress in motion order [%]")
            axes[2].set_title("C  Structural displacement response", loc="left", fontweight="bold")

            axes[3].fill_between(
                progress,
                stress_p95_MPa,
                stress_p99_MPa,
                color="#EDB477",
                alpha=0.22,
                label="p95-p99 band",
            )
            axes[3].plot(
                progress,
                stress_p99_MPa,
                color="#D55E00",
                linewidth=2.2,
                label="Area-weighted p99",
            )
            axes[3].plot(
                progress,
                stress_p95_MPa,
                color="#2F6F9F",
                linewidth=2.2,
                label="Area-weighted p95",
            )
            p95_index = result.snapshot_role_indices["maximum_von_mises_p95"]
            p99_index = result.snapshot_role_indices["maximum_von_mises_p99"]
            for index, values, color in (
                (p95_index, stress_p95_MPa, "#2F6F9F"),
                (p99_index, stress_p99_MPa, "#D55E00"),
            ):
                axes[3].scatter(
                    [progress[index]],
                    [values[index]],
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    s=46,
                    zorder=5,
                )
            if p95_index == p99_index:
                stress_label = (
                    f"maxima at {progress[p99_index]:.1f}%\n"
                    f"p95 {stress_p95_MPa[p95_index]:.2f} MPa | "
                    f"p99 {stress_p99_MPa[p99_index]:.2f} MPa"
                )
                axes[3].annotate(
                    stress_label,
                    (progress[p99_index], stress_p99_MPa[p99_index]),
                    xytext=(-164, -30),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color="#8B4A12",
                    fontsize=8.5,
                    arrowprops={"arrowstyle": "->", "color": "#8B4A12", "linewidth": 0.9},
                )
            else:
                axes[3].annotate(
                    f"max p99 {stress_p99_MPa[p99_index]:.2f} MPa",
                    (progress[p99_index], stress_p99_MPa[p99_index]),
                    xytext=(-102, -26),
                    textcoords="offset points",
                    color="#8B4A12",
                    fontsize=8.5,
                    arrowprops={"arrowstyle": "->", "color": "#8B4A12", "linewidth": 0.9},
                )
                axes[3].annotate(
                    f"max p95 {stress_p95_MPa[p95_index]:.2f} MPa",
                    (progress[p95_index], stress_p95_MPa[p95_index]),
                    xytext=(-102, 18),
                    textcoords="offset points",
                    color="#285F87",
                    fontsize=8.5,
                    arrowprops={"arrowstyle": "->", "color": "#285F87", "linewidth": 0.9},
                )
            axes[3].set_ylim(bottom=0.0)
            axes[3].set_ylabel("von Mises stress [MPa]")
            axes[3].set_xlabel("Pass progress in motion order [%]")
            axes[3].set_title("D  Area-weighted stress response", loc="left", fontweight="bold")
            axes[3].legend(loc="upper left", ncol=2, fontsize=8.2)

            fig.suptitle(
        PLOT_TITLES_EN["pass_scan"],
                y=0.965,
                fontsize=16,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.915,
                f"Read left to right: engagement -> load -> response | wheel motion: {direction} | "
                "ideal circular surface and one-way empirical coupling",
                ha="center",
                va="center",
                fontsize=10,
                color="#4D4D4D",
            )
            fig.text(
                0.5,
                0.045,
                "Independent zero-history, small-deformation 2D plane-stress snapshots. "
                "No transient dynamics, true contact, accumulated state, multiple passes, or roughness prediction.",
                ha="center",
                va="center",
                fontsize=8.2,
                color="#5A5A5A",
            )
            save_png(fig, path)
        finally:
            plt.close(fig)
    return path


def _role_summary(result: PassScanResult, final_snapshot_dirs: dict[int, Path]) -> dict[str, object]:
    values: dict[str, object] = {}
    for role, index in result.snapshot_role_indices.items():
        row = result.history[index]
        values[role] = {
            "scan_index": index,
            "motion_coordinate_m": row.motion_coordinate_m,
            "wheel_lowest_point_x_m": row.wheel_lowest_point_x_m,
            "pass_state": row.pass_state,
            "snapshot_directory": str(final_snapshot_dirs[index]),
        }
    return values


def build_scan_summary(
    result: PassScanResult,
    artifacts: dict[str, Path],
    final_snapshot_dirs: dict[int, Path],
) -> dict[str, object]:
    rows = result.history
    first = rows[0]
    last = rows[-1]
    states = {state: sum(row.pass_state == state for row in rows) for state in (
        "before_entry", "entry", "full_contact", "exit", "after_exit"
    )}
    extreme_fields = {
        "maximum_displacement": "maximum_displacement_m",
        "maximum_von_mises_p95": "von_mises_area_weighted_p95_Pa",
        "maximum_von_mises_p99": "von_mises_area_weighted_p99_Pa",
    }
    extrema = {}
    for role, field in extreme_fields.items():
        index = result.snapshot_role_indices[role]
        row = rows[index]
        extrema[role] = {
            "scan_index": index,
            "value": getattr(row, field),
            "field": field,
            "motion_coordinate_m": row.motion_coordinate_m,
            "wheel_lowest_point_x_m": row.wheel_lowest_point_x_m,
            "tie_break": "first position in motion order",
        }
    maximum_force_index = max(
        range(len(rows)),
        key=lambda index: math.hypot(rows[index].current_Fx_N, rows[index].current_Fy_N),
    )
    maximum_force_row = rows[maximum_force_index]
    reference = result.reference_pass_load
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "scan_schema_version": result.case.scan_schema_version,
        "unit_system": result.case.unit_system,
        "model_type": result.case.model_type,
        "normalized_input": result.case.to_dict(),
        "position_generation": {
            "range": result.case.scan.range,
            "coordinate": "phase 4C1 motion coordinate s",
            "s_start_m": first.motion_coordinate_m,
            "s_end_m": last.motion_coordinate_m,
            "exact_arc_projected_length_m": reference.exact_arc_projected_length_m,
            "base_position_count": result.case.scan.base_position_count,
            "actual_position_count": len(rows),
            "inserted_exact_positions": [
                "-Larc",
                "-0.5*Larc",
                "0",
                "0.5*(W-Larc)",
                "W-Larc",
                "W-0.5*Larc",
                "W",
                "reference_case position",
            ],
            "deduplicated_with_shared_phase_4c1_coordinate_tolerance": True,
            "phase_4c1_state_classification_reused": True,
            "serial_motion_order": True,
        },
        "scan_counts": {
            "total_positions": len(rows),
            "states": states,
            "unique_snapshot_directories": result.unique_snapshot_count,
        },
        "endpoints": {
            "start": {
                "motion_coordinate_m": first.motion_coordinate_m,
                "wheel_lowest_point_x_m": first.wheel_lowest_point_x_m,
                "pass_state": first.pass_state,
            },
            "end": {
                "motion_coordinate_m": last.motion_coordinate_m,
                "wheel_lowest_point_x_m": last.wheel_lowest_point_x_m,
                "pass_state": last.pass_state,
            },
        },
        "nominal_time": {
            "workpiece_feed_speed_m_per_s": result.case.reference_case.pass_load.force_model.process.workpiece_feed_speed_m_per_s,
            "total_nominal_travel_time_s": last.nominal_elapsed_time_s,
            "meaning": "auxiliary travel coordinate only; it is not transient FEM time",
        },
        "steady_full_contact_empirical_force": {
            "Fx_magnitude_N": abs(reference.steady_tangential_force_magnitude_N),
            "Fy_N": -reference.steady_normal_force_magnitude_N,
        },
        "force_extrema": {
            "maximum_current_resultant_force": {
                "scan_index": maximum_force_index,
                "motion_coordinate_m": maximum_force_row.motion_coordinate_m,
                "wheel_lowest_point_x_m": maximum_force_row.wheel_lowest_point_x_m,
                "Fx_N": maximum_force_row.current_Fx_N,
                "Fy_N": maximum_force_row.current_Fy_N,
                "resultant_magnitude_N": math.hypot(
                    maximum_force_row.current_Fx_N,
                    maximum_force_row.current_Fy_N,
                ),
                "tie_break": "first position in motion order",
            }
        },
        "response_extrema": extrema,
        "snapshot_roles": _role_summary(result, final_snapshot_dirs),
        "validations": result.validations.to_dict(),
        "visualization": {
            "scan_overview_chinese_short_note": False,
            "snapshot_chinese_short_note_on_all_pngs": False,
            "snapshot_active_loaded_facet_inset_on_fem_pngs": True,
            "snapshot_active_inset_uses_final_solve_mesh": True,
            "snapshot_active_inset_load_direction_source": "applied facet tx/ty",
            "snapshot_active_inset_arrow_scale_visual_only": True,
        },
        "peak_value_caution": {
            "raw_element_maximum_mesh_dependent": True,
            "primary_response_locations_use": [
                "maximum displacement",
                "area-weighted von Mises p95",
                "area-weighted von Mises p99",
            ],
            "raw_maximum_not_used_for_primary_location": True,
            "fixed_free_corner_singularity": "the fixed/free junction can produce a mesh-sensitive stress peak",
        },
        "physical_scope": {
            "included": [
                "single pass",
                "independent quasi-static snapshots",
                "ideal circular-arc surface evolution",
                "small-deformation 2D plane stress",
                "isotropic linear elasticity",
                "one-way empirical force coupling",
            ],
            "no_state_transfer_between_positions": True,
            "omitted": [
                "stress or strain history transfer",
                "plastic accumulation",
                "temperature accumulation",
                "mass, damping, inertia, impact, chatter, and transient integration",
                "true contact iteration",
                "multiple passes, reciprocation, and spark-out",
                "wheel wear",
                "Ra/Rz roughness prediction",
                "industrial validation",
            ],
            "interpretation_limit": "Do not infer yielding, safety factor, production quality, or industrial validity directly from scan stresses.",
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def _write_summary(summary: dict[str, object], path: Path) -> None:
    try:
        serialized = json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise PassScanExportError(f"scan summary serialization failed: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="ascii", newline="\n")


def export_pass_scan(
    result: PassScanResult,
    workspace: str | Path,
    final_output_dir: str | Path,
    deformation_scale: str | float = "auto",
    *,
    evolved_operations: EvolvedFemOperations | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Export aggregate artifacts and de-duplicated snapshots from solved results."""

    temporary = aggregate_artifact_paths(workspace)
    final = aggregate_artifact_paths(final_output_dir)
    temporary["snapshots"].mkdir(parents=True, exist_ok=True)
    temporary_snapshot_dirs = snapshot_directories(result, temporary["snapshots"])
    final_snapshot_dirs = snapshot_directories(result, final["snapshots"])
    # 中文导读：快照直接导出已保留的求解结果，不为出图再次计算。
    for index, point_result in result.retained_results.items():
        export_evolved_fem_result(
            point_result,
            temporary_snapshot_dirs[index],
            deformation_scale,
            summary_artifacts=snapshot_artifact_paths(final_snapshot_dirs[index]),
            operations=evolved_operations,
        )
    write_scan_history_csv(result, temporary["scan_history_csv"])
    write_scan_overview_png(result, temporary["scan_overview_png"])
    summary = build_scan_summary(result, final, final_snapshot_dirs)
    _write_summary(summary, temporary["scan_summary_json"])
    validate_pass_scan_artifacts(temporary)
    return temporary, summary


def validate_pass_scan_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(AGGREGATE_ARTIFACT_FILENAMES):
        raise PassScanExportError("aggregate artifact set is invalid")
    for key in ("scan_summary_json", "scan_history_csv", "scan_overview_png"):
        if not artifacts[key].is_file() or artifacts[key].stat().st_size <= 0:
            raise PassScanExportError(f"aggregate artifact is missing or empty: {artifacts[key].name}")
    if not artifacts["snapshots"].is_dir():
        raise PassScanExportError("snapshots directory is missing")
    serialized = artifacts["scan_summary_json"].read_text(encoding="ascii")
    if "NaN" in serialized or "Infinity" in serialized:
        raise PassScanExportError("scan summary contains a non-finite JSON token")
    summary = json.loads(serialized)
    if summary.get("result_format") != RESULT_FORMAT:
        raise PassScanExportError("scan summary result_format is invalid")
    expected_rows = int(summary["scan_counts"]["total_positions"])
    with artifacts["scan_history_csv"].open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != SCAN_HISTORY_CSV_FIELDS:
            raise PassScanExportError("scan history CSV header is invalid")
    if len(rows) != expected_rows:
        raise PassScanExportError("scan history CSV row count is invalid")
    if [int(row["scan_index"]) for row in rows] != list(range(expected_rows)):
        raise PassScanExportError("scan history indexing is invalid")
    numeric_fields = set(SCAN_HISTORY_CSV_FIELDS) - {
        "relative_feed_direction", "pass_state", "snapshot_roles"
    }
    for row in rows:
        if any(not math.isfinite(float(row[field])) for field in numeric_fields):
            raise PassScanExportError("scan history contains a non-finite number")
    pixels = matplotlib_image.imread(artifacts["scan_overview_png"])
    if pixels.size == 0 or pixels.ndim < 2:
        raise PassScanExportError("scan overview PNG is unreadable")
    snapshot_dirs = [path for path in artifacts["snapshots"].iterdir() if path.is_dir()]
    expected_snapshots = int(summary["scan_counts"]["unique_snapshot_directories"])
    if len(snapshot_dirs) != expected_snapshots:
        raise PassScanExportError("snapshot directory count is invalid")
    if any(path.is_file() for path in artifacts["snapshots"].iterdir()):
        raise PassScanExportError("snapshots contains an unexpected file")
    for directory in snapshot_dirs:
        snapshot_artifacts = snapshot_artifact_paths(directory)
        if set(path.name for path in directory.iterdir()) != set(SNAPSHOT_ARTIFACT_FILENAMES.values()):
            raise PassScanExportError(f"snapshot {directory.name} artifact set is invalid")
        validate_evolved_fem_artifacts(snapshot_artifacts)


def _stage_path(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    try:
        if source.is_dir():
            shutil.copytree(source, staged)
        else:
            shutil.copy2(source, staged)
    except OSError as exc:
        raise PassScanExportError(f"could not stage {target.name}: {exc}") from exc
    return staged


def _remove_staged(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish_pass_scan_artifacts(
    temporary: dict[str, Path],
    final: dict[str, Path],
    *,
    replace: Callable[..., object] = os.replace,
) -> None:
    """Publish all aggregate artifacts with snapshots managed as one directory."""

    if set(temporary) != set(final) or set(final) != set(AGGREGATE_ARTIFACT_FILENAMES):
        raise PassScanExportError("aggregate publication set is invalid")
    validate_pass_scan_artifacts(temporary)
    output_dir = final["scan_summary_json"].parent
    directory_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ["scan_history_csv", "scan_overview_png", "snapshots", "scan_summary_json"]
    # 中文导读：扫描摘要最后发布；失败时总览和快照目录也作为一组回滚。
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    transaction_succeeded = False
    try:
        for key in order:
            staged[key] = _stage_path(temporary[key], final[key], "new")
        for key in order:
            if final[key].exists():
                backup = final[key].with_name(
                    f".{final[key].name}.backup.{uuid.uuid4().hex}.tmp"
                )
                replace(final[key], backup)
                backups[key] = backup
            replace(staged[key], final[key])
            del staged[key]
            published.append(key)
        transaction_succeeded = True
    except (OSError, PassScanExportError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                _remove_staged(final[key])
                if key in backups:
                    replace(backups[key], final[key])
                    del backups[key]
            except OSError as rollback_exc:
                rollback_errors.append(f"{final[key].name}: {rollback_exc}")
        message = f"pass-scan publication failed: {exc}"
        for key in order:
            if key in backups and not final[key].exists():
                try:
                    replace(backups[key], final[key])
                    del backups[key]
                except OSError as rollback_exc:
                    rollback_errors.append(f"{final[key].name}: {rollback_exc}")
        if rollback_errors:
            message += "; rollback errors: " + "; ".join(rollback_errors)
        raise PassScanExportError(message) from exc
    finally:
        for path in staged.values():
            if path.exists():
                _remove_staged(path)
        for key, path in backups.items():
            if path.exists() and (transaction_succeeded or final[key].exists()):
                _remove_staged(path)
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass
