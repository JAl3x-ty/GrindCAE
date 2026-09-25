"""CSV and headless PNG exporters for phase 4C1 surface profiles."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_TITLES_EN

import csv
import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "grindcae-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grindcae.plotting import COLORS, add_scope_footer, plot_style, save_png, style_axis, unique_legend

from .core import TrajectoryResult


SURFACE_PROFILE_CSV_FIELDS = (
    "point_id",
    "x_m",
    "surface_y_m",
    "surface_elevation_relative_to_original_m",
    "removed_depth_m",
    "region",
)

REGION_COLORS = {
    "ground": COLORS["ground"],
    "contact_arc": COLORS["contact"],
    "unprocessed": COLORS["unprocessed_line"],
}

def write_surface_profile_csv(
    result: TrajectoryResult,
    output_path: str | Path,
) -> Path:
    """Write the 0-based sampled ideal surface using the standard csv module."""

    if not isinstance(result, TrajectoryResult):
        raise TypeError("result must be a TrajectoryResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SURFACE_PROFILE_CSV_FIELDS)
        for point_id, point in enumerate(result.sample_points):
            writer.writerow(
                (
                    point_id,
                    point.x_m,
                    point.surface_y_m,
                    point.surface_elevation_relative_to_original_m,
                    point.removed_depth_m,
                    point.region,
                )
            )
    return path


def _plot_profile_regions(axis: object, result: TrajectoryResult) -> None:
    x_mm = np.asarray([point.x_m for point in result.sample_points]) * 1.0e3
    elevation_um = (
        np.asarray(
            [
                point.surface_elevation_relative_to_original_m
                for point in result.sample_points
            ]
        )
        * 1.0e6
    )
    for region in ("ground", "contact_arc", "unprocessed"):
        mask = np.zeros(x_mm.shape, dtype=bool)
        for segment in result.surface_segments:
            if segment.region == region:
                mask |= (x_mm >= 1.0e3 * segment.x_start_m) & (
                    x_mm <= 1.0e3 * segment.x_end_m
                )
        values = np.ma.masked_where(~mask, elevation_um)
        axis.plot(
            x_mm,
            values,
            color=REGION_COLORS[region],
            linewidth=2.4,
            label=region,
        )
    axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="original surface",
    )


def _local_limits(result: TrajectoryResult) -> tuple[float, float]:
    width = result.case.workpiece.length_m
    lowest = result.case.single_pass.wheel_lowest_point_x_m
    arc = result.exact_arc_projected_length_m
    theoretical = result.theoretical_arc_interval_m
    if result.effective_arc_interval_m is not None:
        start, end = result.effective_arc_interval_m
        margin = max(1.5 * arc, 0.02 * width)
    else:
        start, end = theoretical
        margin = max(arc, 0.02 * width)
    lower = max(0.0, min(start, end, lowest) - margin)
    upper = min(width, max(start, end, lowest) + margin)
    if upper <= lower:
        return 0.0, width
    return 1.0e3 * lower, 1.0e3 * upper


def write_surface_profile_png(
    result: TrajectoryResult,
    output_path: str | Path,
) -> Path:
    """Write a readable full-workpiece and local circular-transition PNG."""

    if not isinstance(result, TrajectoryResult):
        raise TypeError("result must be a TrajectoryResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    width_mm = 1.0e3 * result.case.workpiece.length_m
    depth_um = 1.0e6 * result.case.single_pass.depth_of_cut_m
    arc_mm = 1.0e3 * result.exact_arc_projected_length_m
    lowest_mm = 1.0e3 * result.case.single_pass.wheel_lowest_point_x_m
    direction = result.case.single_pass.relative_feed_direction
    arrow_sign = 1.0 if direction == "positive_x" else -1.0

    with plt.rc_context(plot_style()):
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.78, wspace=0.24)
        try:
            for axis in axes:
                _plot_profile_regions(axis, result)
                axis.axvline(
                    lowest_mm,
                    color=COLORS["displacement"],
                    linestyle=":",
                    linewidth=1.6,
                    label="Wheel lowest point",
                )
                axis.set_ylim(-1.18 * depth_um, 0.18 * depth_um)
                axis.set_ylabel("Elevation from original surface [um]")
                style_axis(axis, grid_axis="y")

            axes[0].set_xlim(0.0, width_mm)
            axes[0].set_xlabel("Workpiece x [mm]")
            axes[0].set_title("A  Surface state across the workpiece", loc="left", fontweight="bold")
            for segment in result.surface_segments:
                midpoint = 0.5e3 * (segment.x_start_m + segment.x_end_m)
                length_mm = 1.0e3 * segment.length_m
                if length_mm < 0.03 * width_mm:
                    continue
                y = -0.68 * depth_um if segment.region == "ground" else -0.10 * depth_um
                color = "white" if segment.region == "ground" else "#555555"
                axes[0].text(
                    midpoint,
                    y,
                    segment.region.replace("_", " ").upper(),
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=8.5,
                )
            arrow_start = 0.18 * width_mm if arrow_sign > 0 else 0.82 * width_mm
            arrow_end = arrow_start + arrow_sign * 0.13 * width_mm
            axes[0].annotate(
                f"wheel motion {direction}",
                xy=(arrow_end, 0.08 * depth_um),
                xytext=(arrow_start, 0.08 * depth_um),
                arrowprops={"arrowstyle": "->", "color": COLORS["tangential"], "lw": 1.8},
                color=COLORS["tangential"],
                ha="center",
                va="center",
                fontsize=8.5,
            )

            axes[1].set_xlim(*_local_limits(result))
            axes[1].set_xlabel("Workpiece x [mm]")
            axes[1].set_title("B  Circular transition (vertical scale enlarged)", loc="left", fontweight="bold")
            if result.effective_arc_interval_m is not None:
                start, end = result.effective_arc_interval_m
                start_mm = 1.0e3 * start
                end_mm = 1.0e3 * end
                axes[1].axvspan(start_mm, end_mm, color=COLORS["contact"], alpha=0.10)
                axes[1].annotate(
                    f"effective projected contact = {result.effective_contact_length_m * 1.0e3:.3g} mm",
                    xy=(0.5 * (start_mm + end_mm), -0.45 * depth_um),
                    xytext=(0.5, 0.15),
                    textcoords="axes fraction",
                    ha="center",
                    fontsize=8.5,
                    color="#995A12",
                    arrowprops={"arrowstyle": "->", "color": "#995A12", "linewidth": 0.9},
                )
            axes[1].text(
                0.02,
                0.05,
                f"state: {result.pass_state}\n"
                f"ae = {depth_um:.4g} um | Larc = {arc_mm:.4g} mm | contact ratio = {result.contact_ratio:.3g}",
                transform=axes[1].transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
            )
            unique_legend(axes[0], loc="lower center", ncol=3, fontsize=8.0)
            figure.suptitle(
        PLOT_TITLES_EN["trajectory_surface_profile"],
                y=0.94,
                fontsize=15,
                fontweight="bold",
            )
            figure.text(
                0.5,
                0.855,
                "Read left to right: whole-workpiece surface state -> strict circular transition",
                ha="center",
                color="#4D4D4D",
                fontsize=9.5,
            )
            add_scope_footer(
                figure,
                "User-prescribed depth and ideal mean geometry only. No force, elastic deformation, roughness, wear, or transient motion.",
                y=0.055,
            )
            save_png(figure, path)
        finally:
            plt.close(figure)
    return path
