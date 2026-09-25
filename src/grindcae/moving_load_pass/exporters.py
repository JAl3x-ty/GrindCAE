"""CSV, JSON, mesh audit, and single acceptance PNG for Phase 6B.1."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_SUBTITLES_EN, PLOT_TITLES_EN

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as RectanglePatch

from grindcae import __version__

from .workflow import FixedMeshMovingLoadPassResult


RESULT_FORMAT = "grindcae_phase_6b1_fixed_mesh_moving_load_pass"
HISTORY_CSV_FIELDS = (
    "position_id",
    "motion_coordinate_m",
    "wheel_lowest_point_x_m",
    "pass_state",
    "generation_roles",
    "contact_ratio",
    "contact_start_m",
    "contact_end_m",
    "projected_contact_length_m",
    "active_facet_count",
    "partial_facet_count",
    "Ft_N",
    "Fn_N",
    "Fx_N",
    "Fy_N",
    "qx_N_per_m",
    "qy_N_per_m",
    "assembled_Fx_N",
    "assembled_Fy_N",
    "force_balance_residual_x_N",
    "force_balance_residual_y_N",
    "force_balance_residual_N",
)
TOP_FACET_CSV_FIELDS = (
    "facet_id",
    "node_start_id",
    "node_end_id",
    "x_start_m",
    "y_start_m",
    "x_end_m",
    "y_end_m",
    "dx_projected_m",
    "ds_actual_m",
)


def _ascii_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_history_csv(result: FixedMeshMovingLoadPassResult, path: str | Path) -> None:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_CSV_FIELDS)
        writer.writeheader()
        for item in result.positions:
            interval = item.pass_load.effective_contact_interval_m
            writer.writerow(
                {
                    "position_id": item.position.position_id,
                    "motion_coordinate_m": item.position.motion_coordinate_m,
                    "wheel_lowest_point_x_m": item.position.wheel_lowest_point_x_m,
                    "pass_state": item.position.pass_state,
                    "generation_roles": "|".join(item.position.generation_roles),
                    "contact_ratio": item.pass_load.contact_ratio,
                    "contact_start_m": "" if interval is None else interval[0],
                    "contact_end_m": "" if interval is None else interval[1],
                    "projected_contact_length_m": item.pass_load.effective_contact_length_m,
                    "active_facet_count": item.load_mapping.active_facet_count,
                    "partial_facet_count": item.load_mapping.partial_facet_count,
                    "Ft_N": item.pass_load.current_tangential_force_magnitude_N,
                    "Fn_N": item.pass_load.current_normal_force_magnitude_N,
                    "Fx_N": item.pass_load.current_Fx_N,
                    "Fy_N": item.pass_load.current_Fy_N,
                    "qx_N_per_m": item.pass_load.current_qx_N_per_m,
                    "qy_N_per_m": item.pass_load.current_qy_N_per_m,
                    "assembled_Fx_N": item.load_mapping.assembled_Fx_N,
                    "assembled_Fy_N": item.load_mapping.assembled_Fy_N,
                    "force_balance_residual_x_N": item.load_mapping.force_balance_residual_x_N,
                    "force_balance_residual_y_N": item.load_mapping.force_balance_residual_y_N,
                    "force_balance_residual_N": item.load_mapping.force_balance_residual_N,
                }
            )


def write_top_facets_csv(result: FixedMeshMovingLoadPassResult, path: str | Path) -> None:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TOP_FACET_CSV_FIELDS)
        writer.writeheader()
        for item in result.top_facets:
            writer.writerow(
                {
                    "facet_id": item.facet_id,
                    "node_start_id": item.node_start_id,
                    "node_end_id": item.node_end_id,
                    "x_start_m": item.x_start_m,
                    "y_start_m": item.y_start_m,
                    "x_end_m": item.x_end_m,
                    "y_end_m": item.y_end_m,
                    "dx_projected_m": item.dx_projected_m,
                    "ds_actual_m": item.ds_actual_m,
                }
            )


def _role_item(result: FixedMeshMovingLoadPassResult, role: str):
    return next(item for item in result.positions if role in item.position.generation_roles)


def write_overview_png(result: FixedMeshMovingLoadPassResult, path: str | Path) -> None:
    width = result.case.geometry.width
    height = result.case.geometry.height
    direction = result.case.motion_direction
    representatives = (
        ("representative_entry", "Entry"),
        ("representative_full_contact", "Steady grinding"),
        ("representative_exit", "Exit"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 7.4), constrained_layout=True)
    upper, lower = axes
    vertical_gap = 0.32 * height
    for row, (role, label) in enumerate(representatives):
        item = _role_item(result, role)
        y = (2 - row) * vertical_gap
        upper.plot([0.0, width], [y, y], color="#555555", linewidth=2.0)
        interval = item.pass_load.effective_contact_interval_m
        if interval is not None:
            upper.add_patch(
                RectanglePatch(
                    (interval[0], y - 0.04 * height),
                    interval[1] - interval[0],
                    0.08 * height,
                    color="#e67e22",
                    alpha=0.8,
                )
            )
            center = 0.5 * (interval[0] + interval[1])
            arrow_dx = 0.13 * width if direction == "positive_x" else -0.13 * width
            upper.annotate(
                "",
                xy=(center + 0.5 * arrow_dx, y + 0.10 * height),
                xytext=(center - 0.5 * arrow_dx, y + 0.10 * height),
                arrowprops={"arrowstyle": "->", "color": "#2563eb", "lw": 2.0},
            )
        upper.text(
            -0.015 * width,
            y,
            f"{label}\nratio={item.pass_load.contact_ratio:.3f}",
            ha="right",
            va="center",
            fontsize=9,
        )
    upper.set_xlim(-0.17 * width, 1.02 * width)
    upper.set_ylim(-0.1 * height, 0.95 * height)
    upper.set_xlabel("Fixed reference top edge x (m)")
    upper.set_yticks([])
    upper.set_title(
        f"One fixed mesh: moving projected load windows ({direction.replace('_', ' ')})"
    )
    upper.grid(axis="x", alpha=0.2)

    motion = [item.position.motion_coordinate_m for item in result.positions]
    ratios = [item.pass_load.contact_ratio for item in result.positions]
    fx = [item.pass_load.current_Fx_N for item in result.positions]
    compression = [-item.pass_load.current_Fy_N for item in result.positions]
    lower.plot(motion, ratios, color="#e67e22", marker="o", markersize=3, label="Contact ratio")
    lower.set_xlabel("Single-Pass Motion Coordinate (m), Ordered Along Travel")
    lower.set_ylabel("Contact ratio", color="#e67e22")
    lower.tick_params(axis="y", labelcolor="#e67e22")
    lower.set_ylim(-0.05, 1.08)
    forces = lower.twinx()
    forces.plot(motion, fx, color="#2563eb", label="Fx")
    forces.plot(motion, compression, color="#dc2626", label="-Fy (compression)")
    forces.set_ylabel("Force (N)")
    critical = (
        ("entry_boundary", "entry"),
        ("full_contact_start", "full start"),
        ("full_contact_end", "full end"),
        ("exit_boundary", "exit"),
    )
    for role, label in critical:
        item = _role_item(result, role)
        lower.axvline(item.position.motion_coordinate_m, color="#888888", linestyle="--", linewidth=0.8)
        lower.text(
            item.position.motion_coordinate_m,
            1.03,
            label,
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
        )
    handles1, labels1 = lower.get_legend_handles_labels()
    handles2, labels2 = forces.get_legend_handles_labels()
    lower.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3)
    lower.grid(alpha=0.25)
    figure.suptitle(
        PLOT_TITLES_EN["moving_load_overview"] + "\n"
        + PLOT_SUBTITLES_EN["moving_load_overview"],
        fontsize=13,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_summary(
    result: FixedMeshMovingLoadPassResult,
    artifacts: dict[str, Path],
) -> dict[str, object]:
    entry = _role_item(result, "entry_boundary")
    full_start = _role_item(result, "full_contact_start")
    full_end = _role_item(result, "full_contact_end")
    exit_item = _role_item(result, "exit_boundary")
    maximum_fx = max(result.positions, key=lambda item: abs(item.pass_load.current_Fx_N))
    maximum_fy = max(result.positions, key=lambda item: abs(item.pass_load.current_Fy_N))
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "moving_load_pass_schema_version": result.case.moving_load_pass_schema_version,
        "unit_system": result.case.unit_system,
        "normalized_input": result.case.to_dict(),
        "units": {"length": "m", "force": "N", "line_load": "N/m"},
        "mesh_reuse": {
            "fixed_reference_rectangle": True,
            "generation_count": result.mesh_generation_count,
            "import_count": result.mesh_import_count,
            "node_count": result.reference_mesh.node_count,
            "triangle_count": result.reference_mesh.triangle_count,
            "top_facet_count": len(result.top_facets),
            "same_mesh_for_all_positions": True,
        },
        "position_sequence": {
            "count": len(result.positions),
            "motion_direction": result.case.motion_direction,
            "motion_coordinate_monotonic_increasing": True,
            "entry_motion_coordinate_m": entry.position.motion_coordinate_m,
            "full_contact_start_motion_coordinate_m": full_start.position.motion_coordinate_m,
            "full_contact_end_motion_coordinate_m": full_end.position.motion_coordinate_m,
            "exit_motion_coordinate_m": exit_item.position.motion_coordinate_m,
            "maximum_contact_ratio": result.maximum_contact_ratio,
        },
        "force_conservation": {
            "maximum_absolute_Fx_N": abs(maximum_fx.pass_load.current_Fx_N),
            "maximum_absolute_Fy_N": abs(maximum_fy.pass_load.current_Fy_N),
            "maximum_balance_residual_N": result.maximum_force_balance_residual_N,
            "all_positions_use_q_times_overlap_dx_equals_t_times_loaded_ds": True,
            "grinding_width_or_analysis_thickness_reapplied": False,
        },
        "partial_facet_validation": {
            "positions_with_partial_facets": sum(
                item.load_mapping.partial_facet_count > 0 for item in result.positions
            ),
            "maximum_partial_facet_count": max(
                item.load_mapping.partial_facet_count for item in result.positions
            ),
            "nearest_facet_approximation_used": False,
        },
        "scope": {
            "fem_newton_solve_run_per_position": False,
            "plastic_state_transferred_between_positions": False,
            "mesh_shape_evolved_between_positions": False,
            "phase_6b2_history_accumulation_implemented": False,
        },
        "artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
    }


def write_summary_json(summary: dict[str, object], path: str | Path) -> None:
    _ascii_json(Path(path), summary)
