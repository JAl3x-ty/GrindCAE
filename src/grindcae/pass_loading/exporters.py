"""CSV and headless PNG exporters for phase 4C2 pass-load snapshots."""

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

from .core import PassLoadResult


PASS_LOAD_CSV_FIELDS = (
    "segment_index",
    "segment_type",
    "x_start_m",
    "x_end_m",
    "length_m",
    "qx_N_per_m",
    "qy_N_per_m",
    "integrated_Fx_N",
    "integrated_Fy_N",
)

REGION_COLORS = {
    "ground": COLORS["ground"],
    "contact_arc": COLORS["contact"],
    "unprocessed": COLORS["unprocessed_line"],
}

def write_pass_load_csv(result: PassLoadResult, output_path: str | Path) -> Path:
    """Write the complete ideal top-boundary load partition."""

    if not isinstance(result, PassLoadResult):
        raise TypeError("result must be a PassLoadResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PASS_LOAD_CSV_FIELDS)
        writer.writeheader()
        for index, segment in enumerate(result.load_segments):
            writer.writerow(segment.to_dict(index))
    return path


def _plot_surface(axis: object, result: PassLoadResult) -> None:
    trajectory = result.trajectory_result
    x_mm = np.asarray([point.x_m for point in trajectory.sample_points]) * 1.0e3
    elevation_um = np.asarray(
        [point.surface_elevation_relative_to_original_m for point in trajectory.sample_points]
    ) * 1.0e6
    for region in ("ground", "contact_arc", "unprocessed"):
        mask = np.asarray([point.region == region for point in trajectory.sample_points])
        axis.plot(
            x_mm,
            np.ma.masked_where(~mask, elevation_um),
            color=REGION_COLORS[region],
            linewidth=2.6,
            label=region.replace("_", " ").title(),
        )
    axis.axhline(0.0, color=COLORS["reference"], linestyle="--", linewidth=1.0, label="Original surface")
    if result.effective_contact_interval_m is not None:
        start, end = result.effective_contact_interval_m
        start_mm = 1.0e3 * start
        end_mm = 1.0e3 * end
        axis.axvspan(start_mm, end_mm, color=COLORS["contact"], alpha=0.12, label="Projected load interval")
        axis.annotate(
            f"Leff = {result.effective_contact_length_m * 1.0e3:.3g} mm",
            xy=(0.5 * (start_mm + end_mm), -0.52 * 1.0e6 * result.case.trajectory.single_pass.depth_of_cut_m),
            xytext=(0.5, 0.13),
            textcoords="axes fraction",
            ha="center",
            color="#995A12",
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "color": "#995A12", "linewidth": 0.9},
        )
    direction = result.case.trajectory.single_pass.relative_feed_direction
    sign = 1.0 if direction == "positive_x" else -1.0
    arrow_start = 0.18 if sign > 0.0 else 0.82
    axis.annotate(
        f"wheel motion {direction}",
        xy=(arrow_start + 0.16 * sign, 0.90),
        xytext=(arrow_start, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": COLORS["tangential"], "lw": 1.6},
        color=COLORS["tangential"],
        va="center",
        fontsize=8.5,
    )
    axis.set_xlabel("Workpiece x [mm]")
    axis.set_ylabel("Elevation from original [um]")
    axis.set_title("A  Ideal surface and projected load interval", loc="left", fontweight="bold")
    style_axis(axis, grid_axis="y")
    unique_legend(axis, loc="lower center", ncol=3, fontsize=7.8)


def _step_values(result: PassLoadResult, field: str) -> tuple[np.ndarray, np.ndarray]:
    x_values = [segment.x_start_m for segment in result.load_segments]
    x_values.append(result.load_segments[-1].x_end_m)
    q_values = [getattr(segment, field) for segment in result.load_segments]
    q_values.append(q_values[-1])
    return np.asarray(x_values) * 1.0e3, np.asarray(q_values)

def write_pass_load_png(result: PassLoadResult, output_path: str | Path) -> Path:
    """Write a readable ideal-surface and empirical-load snapshot."""

    if not isinstance(result, PassLoadResult):
        raise TypeError("result must be a PassLoadResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    x_qx, qx = _step_values(result, "qx_N_per_m")
    x_qy, qy = _step_values(result, "qy_N_per_m")
    tangential = np.abs(qx)
    compression = -qy
    if np.any(compression < -1.0e-12):
        raise ValueError("current qy must represent downward compression")

    with plt.rc_context(plot_style()):
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.77, wspace=0.25)
        try:
            surface_axis, load_axis = axes
            _plot_surface(surface_axis, result)

            load_axis.step(
                x_qy,
                compression,
                where="post",
                color=COLORS["normal"],
                linewidth=2.4,
                label="Normal compression -qy",
            )
            load_axis.fill_between(
                x_qy,
                compression,
                step="post",
                color=COLORS["normal"],
                alpha=0.10,
            )
            load_axis.step(
                x_qx,
                tangential,
                where="post",
                color=COLORS["tangential"],
                linewidth=2.4,
                label="Tangential magnitude |qx|",
            )
            load_axis.fill_between(
                x_qx,
                tangential,
                step="post",
                color=COLORS["tangential"],
                alpha=0.10,
            )
            load_axis.axhline(0.0, color=COLORS["reference"], linewidth=0.9)
            if result.effective_contact_interval_m is not None:
                start, end = result.effective_contact_interval_m
                load_axis.axvspan(
                    1.0e3 * start,
                    1.0e3 * end,
                    color=COLORS["contact"],
                    alpha=0.10,
                )
            load_axis.set_xlim(0.0, 1.0e3 * result.case.trajectory.workpiece.length_m)
            maximum = max(float(np.max(tangential)), float(np.max(compression)), 1.0)
            load_axis.set_ylim(0.0, 1.18 * maximum)
            load_axis.set_xlabel("Workpiece x [mm]")
            load_axis.set_ylabel("Line-load magnitude [N/m]")
            load_axis.set_title("B  Current empirical line load", loc="left", fontweight="bold")
            style_axis(load_axis, grid_axis="y")
            unique_legend(load_axis, loc="upper center", ncol=2, fontsize=8.0)
            load_axis.text(
                0.02,
                0.08,
                f"contact ratio = {result.contact_ratio:.3g}\n"
                f"Fx = {result.current_Fx_N:.4g} N | downward compression = {-result.current_Fy_N:.4g} N\n"
                f"tangential direction = {result.case.force_mapping.tangential_force_direction}",
                transform=load_axis.transAxes,
                va="bottom",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
            )
            load_axis.text(
                0.98,
                0.08,
                "Compression is plotted upward as -qy.\nSigned qx/qy remain unchanged in CSV/JSON.",
                transform=load_axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.8,
                color="#666666",
            )

            figure.suptitle(
        f"{PLOT_TITLES_EN['pass_load_snapshot']} | State: {result.pass_state.replace('_', ' ').title()}",
                y=0.94,
                fontsize=15,
                fontweight="bold",
            )
            figure.text(
                0.5,
                0.845,
                "Read left to right: ideal projected geometry -> current contact-ratio-scaled load",
                ha="center",
                color="#4D4D4D",
                fontsize=9.5,
            )
            add_scope_footer(
                figure,
                f"Calibration: {result.case.force_model.calibration.calibration_id}. Empirical and unvalidated; no Hertz pressure, Gmsh, FEM, stress, or transient dynamics.",
                y=0.055,
            )
            save_png(figure, path)
        finally:
            plt.close(figure)
    return path
