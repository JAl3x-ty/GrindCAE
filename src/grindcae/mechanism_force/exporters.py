"""CSV and headless PNG exporters for Phase 7A.1."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_TITLES_EN

import csv
import math
import os
from pathlib import Path
import tempfile

from .core import MechanismForcePrediction, interpolate_mechanism_fractions


CSV_FIELDS = (
    "eta",
    "equivalent_process_thickness_m",
    "rubbing_fraction",
    "ploughing_fraction",
    "cutting_fraction",
    "fraction_sum",
)


def eta_scan_values(current_eta: float, *, count: int = 81) -> tuple[float, ...]:
    """Return a deterministic log scan containing anchors and current eta."""

    if count < 2:
        raise ValueError("count must be at least 2")
    values = [10.0 ** (-1.0 + index * 2.0 / (count - 1)) for index in range(count)]
    values.extend((0.25, 1.0, 4.0, current_eta))
    return tuple(sorted(set(values)))


def fraction_scan_rows(result: MechanismForcePrediction) -> tuple[dict[str, float], ...]:
    rows: list[dict[str, float]] = []
    reference = result.parameters.reference_equivalent_process_thickness_m
    for eta in eta_scan_values(result.eta):
        fractions = interpolate_mechanism_fractions(eta, result.parameters)
        rows.append(
            {
                "eta": eta,
                "equivalent_process_thickness_m": eta * reference,
                "rubbing_fraction": fractions.rubbing,
                "ploughing_fraction": fractions.ploughing,
                "cutting_fraction": fractions.cutting,
                "fraction_sum": math.fsum(fractions.as_tuple()),
            }
        )
    return tuple(rows)


def write_fraction_csv(result: MechanismForcePrediction, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(fraction_scan_rows(result))
    return path


def write_fraction_png(result: MechanismForcePrediction, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dir = Path(tempfile.gettempdir()) / "grindcae-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grindcae.plotting import plot_style, save_png, style_axis

    rows = fraction_scan_rows(result)
    plot_rows = [row for row in rows if 0.1 <= row["eta"] <= 10.0]
    eta = [row["eta"] for row in plot_rows]
    with plt.rc_context(plot_style()):
        figure, axis = plt.subplots(figsize=(10.4, 6.1))
        figure.subplots_adjust(left=0.11, right=0.96, bottom=0.15, top=0.88)
        try:
            axis.plot(eta, [row["rubbing_fraction"] for row in plot_rows], linewidth=2.4, color="#2F6F9F", label="Rubbing")
            axis.plot(eta, [row["ploughing_fraction"] for row in plot_rows], linewidth=2.4, color="#E07A1F", label="Ploughing")
            axis.plot(eta, [row["cutting_fraction"] for row in plot_rows], linewidth=2.4, color="#4C956C", label="Cutting")
            current_eta = result.eta
            if 0.1 <= current_eta <= 10.0:
                axis.axvline(current_eta, color="#303030", linestyle="--", linewidth=1.2, label=f"Current eta = {result.eta:.4g}")
                axis.scatter(
                    [current_eta] * 3,
                    list(result.fractions.as_tuple()),
                    color=("#2F6F9F", "#E07A1F", "#4C956C"),
                    edgecolor="white",
                    linewidth=0.8,
                    s=48,
                    zorder=5,
                )
            axis.set_xscale("log")
            axis.set_xlim(0.1, 10.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("eta = h_eq / h_ref [-]")
            axis.set_ylabel("Mechanism fraction [-]")
            axis.set_title(PLOT_TITLES_EN["mechanism_fractions"], loc="left", fontweight="bold")
            style_axis(axis, grid_axis="both")
            axis.legend(loc="best", fontsize=9)
            save_png(figure, path)
        finally:
            plt.close(figure)
    return path
