"""CSV and headless PNG exporters for phase 6A.1 material-point histories."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_TITLES_EN

import csv
import os
from pathlib import Path
import tempfile

from .uniaxial import UniaxialHistoryResult


HISTORY_CSV_FIELDS = (
    "step_id",
    "segment",
    "axial_total_strain",
    "transverse_total_strain_y",
    "transverse_total_strain_z",
    "axial_stress_Pa",
    "transverse_stress_y_Pa",
    "transverse_stress_z_Pa",
    "axial_plastic_strain",
    "equivalent_plastic_strain",
    "current_yield_strength_Pa",
    "yield_function_Pa",
    "increment_class",
)


def write_history_csv(
    result: UniaxialHistoryResult, output_path: str | Path
) -> Path:
    """Write one row for every converged material-point state."""

    if not isinstance(result, UniaxialHistoryResult):
        raise TypeError("result must be an UniaxialHistoryResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_CSV_FIELDS)
        writer.writeheader()
        for step in result.steps:
            writer.writerow(step.to_dict())
    return path


def stress_strain_plot_series(
    result: UniaxialHistoryResult,
) -> dict[str, tuple[float, ...]]:
    """Return plot coordinates, including the initial zero state in loading."""

    if not isinstance(result, UniaxialHistoryResult):
        raise TypeError("result must be an UniaxialHistoryResult")
    loading = [
        step for step in result.steps if step.segment in {"initial", "loading"}
    ]
    unloading = [result.peak_step] + [
        step for step in result.steps if step.segment == "unloading"
    ]
    return {
        "loading_strain": tuple(step.axial_total_strain for step in loading),
        "loading_stress_MPa": tuple(
            step.axial_stress_Pa * 1.0e-6 for step in loading
        ),
        "unloading_strain": tuple(step.axial_total_strain for step in unloading),
        "unloading_stress_MPa": tuple(
            step.axial_stress_Pa * 1.0e-6 for step in unloading
        ),
    }


def write_stress_strain_png(
    result: UniaxialHistoryResult, output_path: str | Path
) -> Path:
    """Plot the material-point loading and unloading stress-strain curve."""

    if not isinstance(result, UniaxialHistoryResult):
        raise TypeError("result must be an UniaxialHistoryResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dir = Path(tempfile.gettempdir()) / "grindcae-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grindcae.plotting import (
        COLORS,
        add_scope_footer,
        plot_style,
        save_png,
        style_axis,
    )

    series = stress_strain_plot_series(result)
    material = result.case.material

    with plt.rc_context(plot_style()):
        figure, axis = plt.subplots(figsize=(10.8, 6.1))
        figure.subplots_adjust(left=0.105, right=0.97, bottom=0.18, top=0.84)
        try:
            axis.plot(
                series["loading_strain"],
                series["loading_stress_MPa"],
                color=COLORS["stress"],
                linewidth=2.3,
                label="Loading",
            )
            axis.plot(
                series["unloading_strain"],
                series["unloading_stress_MPa"],
                color=COLORS["displacement"],
                linewidth=2.1,
                label="Elastic unloading",
            )
            axis.scatter(
                [result.yield_strain],
                [material.yield_strength * 1.0e-6],
                color=COLORS["reference"],
                marker="o",
                s=42,
                zorder=4,
                label="Analytic yield point",
            )
            axis.scatter(
                [result.peak_step.axial_total_strain],
                [result.peak_step.axial_stress_Pa * 1.0e-6],
                color=COLORS["stress"],
                marker="D",
                s=40,
                zorder=4,
                label="Peak point",
            )
            axis.scatter(
                [result.final_step.axial_total_strain],
                [result.final_step.axial_stress_Pa * 1.0e-6],
                color=COLORS["displacement"],
                marker="s",
                s=40,
                zorder=4,
                label="Zero-stress residual strain",
            )
            axis.axhline(0.0, color="#808080", linewidth=0.8)
            axis.set_xlabel("Axial total strain [-]")
            axis.set_ylabel("Axial stress [MPa]")
            axis.set_title(
                PLOT_TITLES_EN["j2_material_point"],
                loc="left",
                fontweight="bold",
            )
            style_axis(axis, grid_axis="both")
            axis.legend(loc="best", fontsize=8.5)
            figure.text(
                0.105,
                0.88,
                "Uniaxial-stress benchmark: lateral stresses controlled to zero",
                ha="left",
                color="#4D4D4D",
                fontsize=9.5,
            )
            add_scope_footer(
                figure,
                "Independent small-strain material point only. No FEM, grinding load, contact, GUI, or cyclic plasticity.",
                y=0.065,
            )
            save_png(figure, path)
        finally:
            plt.close(figure)
    return path
