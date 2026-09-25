"""JSON, CSV, and headless PNG exporters for Phase 7A.2."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile

from grindcae.presentation_labels import PLOT_TITLES_EN

from .core import StatisticalGrainLoadPrediction


GROUP_FIELDS = (
    "group_id", "normalized_contact_position", "contact_position_m",
    "representative_diameter_m", "represented_physical_grain_count",
    "rubbing_weight", "ploughing_weight", "cutting_weight", "base_kernel_width_m",
)
LOAD_FIELDS = (
    "x_local_m", "normalized_x", "q_t_rubbing_N_per_m", "q_t_ploughing_N_per_m",
    "q_t_cutting_N_per_m", "q_t_total_N_per_m", "q_n_rubbing_N_per_m",
    "q_n_ploughing_N_per_m", "q_n_cutting_N_per_m", "q_n_total_N_per_m",
)
GRIT_TITLE_LABELS = {
    "GB_FEPA_F": "GB/FEPA Reference",
    "GB_W_micropowder": "W-Series Secondary Reference",
    "JIS_hash": "JIS Secondary Reference",
    "manufacturer_override": "Manufacturer Override",
}


def write_json(payload: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_groups_csv(result: StatisticalGrainLoadPrediction, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        writer.writerows(group.to_dict() for group in result.groups)
    return path


def write_distribution_csv(result: StatisticalGrainLoadPrediction, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOAD_FIELDS)
        writer.writeheader()
        writer.writerows(point.to_dict() for point in result.distribution)
    return path


def write_distribution_png(result: StatisticalGrainLoadPrediction, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dir = Path(tempfile.gettempdir()) / "grindcae-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grindcae.plotting import plot_style, style_axis

    spec = result.resolved_grit_specification
    x_mm = [point.x_local_m * 1e3 for point in result.distribution]
    with plt.rc_context(plot_style()):
        figure, axes = plt.subplots(3, 1, figsize=(10.5, 10.4), sharex=True)
        figure.subplots_adjust(left=0.11, right=0.97, bottom=0.08, top=0.89, hspace=0.34)
        try:
            axes[0].scatter(
                [group.contact_position_m * 1e3 for group in result.groups],
                [group.representative_diameter_m * 1e6 for group in result.groups],
                c=[group.cutting_weight for group in result.groups], cmap="viridis", s=36,
                edgecolor="white", linewidth=0.5,
            )
            axes[0].set_ylabel("Representative diameter [um]")
            axes[0].set_title("A | Statistical equivalent grain groups", loc="left", fontweight="bold")
            style_axis(axes[0], grid_axis="both")
            colors = {"rubbing": "#2F6F9F", "ploughing": "#E07A1F", "cutting": "#4C956C", "total": "#303030"}
            for axis, direction, title in (
                (axes[1], "t", "B | Tangential mechanism line load"),
                (axes[2], "n", "C | Normal mechanism line load"),
            ):
                for name in ("rubbing", "ploughing", "cutting", "total"):
                    axis.plot(
                        x_mm,
                        [getattr(point, f"q_{direction}_{name}_N_per_m") for point in result.distribution],
                        color=colors[name], linewidth=2.2 if name == "total" else 1.6,
                        label=name.capitalize(),
                    )
                axis.set_ylabel("Line load [N/m]")
                axis.set_title(title, loc="left", fontweight="bold")
                axis.set_ylim(bottom=0.0)
                style_axis(axis, grid_axis="both")
                axis.legend(loc="best", ncol=4, fontsize=8)
            axes[2].set_xlabel("Local contact position [mm]")
            grit_title_label = GRIT_TITLE_LABELS[spec.resolution_path]
            title = (
                f"Aluminum Oxide Grinding Wheel | {grit_title_label} | {spec.grit_designation}\n"
                f"Representative diameter d50 ≈ {spec.representative_diameter_d50_m * 1e6:.1f} µm | "
                "Provisional demonstration data"
            )
            figure.suptitle(
                f"{PLOT_TITLES_EN['statistical_grain_distribution']}\n{title}\n"
                f"Estimated N_active = {result.estimated_physical_active_grain_count:.4g} | "
                f"Equivalent groups = {result.equivalent_group_count}",
                x=0.11, ha="left", fontsize=11.2, fontweight="bold",
            )
            figure.savefig(
                path,
                dpi=300,
                facecolor="white",
                metadata={
                    "Title": (
                        f"{PLOT_TITLES_EN['statistical_grain_distribution']} | "
                        f"{title.replace(chr(10), ' | ')}"
                    )
                },
            )
        finally:
            plt.close(figure)
    return path
