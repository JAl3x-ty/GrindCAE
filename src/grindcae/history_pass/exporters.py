"""Core machine-readable and visual exporters for Phase 6B.2."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from grindcae import __version__
from grindcae.presentation_labels import PLOT_SUBTITLES_EN, PLOT_TITLES_EN
from grindcae.elastoplastic_fem.exporters import (
    write_elements_csv,
    write_nodes_csv,
    write_vtu,
)
from grindcae.moving_load_pass.exporters import write_overview_png
from grindcae.plotting import COLORS, plot_style, save_png, style_axis

from .workflow import FixedMeshElastoplasticHistoryPassResult


RESULT_FORMAT = "grindcae_phase_6b2_fixed_mesh_elastoplastic_history_pass"
HISTORY_CSV_FIELDS = (
    "position_id", "motion_coordinate_m", "pass_state", "contact_ratio",
    "Fx_N", "Fy_N", "assembled_Fx_N", "assembled_Fy_N",
    "reaction_Rx_N", "reaction_Ry_N", "balance_residual_N",
    "transition_substep_count", "retry_count", "maximum_newton_iteration_count",
    "maximum_displacement_m", "maximum_von_mises_stress_Pa",
    "maximum_equivalent_plastic_strain", "plastic_element_count",
    "plastic_element_fraction", "accumulated_plastic_dissipation_J", "state_committed",
)


def write_history_csv(result: FixedMeshElastoplasticHistoryPassResult, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_CSV_FIELDS)
        writer.writeheader()
        for row in result.history_rows:
            writer.writerow({
                "position_id": row.position_id,
                "motion_coordinate_m": row.motion_coordinate_m,
                "pass_state": row.pass_state,
                "contact_ratio": row.contact_ratio,
                "Fx_N": row.Fx_N,
                "Fy_N": row.Fy_N,
                "assembled_Fx_N": row.assembled_Fx_N,
                "assembled_Fy_N": row.assembled_Fy_N,
                "reaction_Rx_N": row.fixed_reaction_x_N,
                "reaction_Ry_N": row.fixed_reaction_y_N,
                "balance_residual_N": row.balance_residual_N,
                "transition_substep_count": row.transition_substep_count,
                "retry_count": row.retry_count,
                "maximum_newton_iteration_count": row.maximum_newton_iteration_count,
                "maximum_displacement_m": row.maximum_displacement_m,
                "maximum_von_mises_stress_Pa": row.maximum_von_mises_stress_Pa,
                "maximum_equivalent_plastic_strain": row.maximum_equivalent_plastic_strain,
                "plastic_element_count": row.plastic_element_count,
                "plastic_element_fraction": row.plastic_element_fraction,
                "accumulated_plastic_dissipation_J": row.accumulated_plastic_dissipation_J,
                "state_committed": "true" if row.state_committed else "false",
            })
    return output


def _setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)


def write_force_history_png(result: FixedMeshElastoplasticHistoryPassResult, path: str | Path) -> Path:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    output = Path(path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    rows = result.history_rows[:-1]
    s = np.asarray([row.motion_coordinate_m for row in rows]) * 1e3
    fx = np.asarray([row.Fx_N for row in rows]); fy = np.asarray([row.Fy_N for row in rows])
    ratio = np.asarray([row.contact_ratio for row in rows])
    with plt.rc_context(plot_style()):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True, constrained_layout=True)
        top.plot(s, fx, label="Fx", color=COLORS["ground"]); top.plot(s, fy, label="Fy", color=COLORS["contact"])
        top.axhline(0, color="black", linewidth=0.7); top.set_ylabel("Signed force [N]"); top.legend(); style_axis(top)
        bottom.plot(s, ratio, color=COLORS["displacement"], label="Contact ratio")
        bottom.set_xlabel("Motion coordinate [mm]"); bottom.set_ylabel("Contact ratio [-]"); bottom.set_ylim(-0.03, 1.05); style_axis(bottom)
        figure.suptitle(PLOT_TITLES_EN["history_pass_force"] + "\n" + PLOT_SUBTITLES_EN["history_pass_force"], fontweight="bold")
        save_png(figure, output); plt.close(figure)
    return output


def write_plastic_history_png(result: FixedMeshElastoplasticHistoryPassResult, path: str | Path) -> Path:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    output = Path(path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    rows = result.history_rows
    index = np.arange(len(rows)); eqp = np.asarray([row.maximum_equivalent_plastic_strain for row in rows])
    fraction = np.asarray([row.plastic_element_fraction for row in rows]); dissipation = np.asarray([row.accumulated_plastic_dissipation_J for row in rows])
    with plt.rc_context(plot_style()):
        figure, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)
        axes[0].plot(index, eqp, color=COLORS["contact"]); axes[0].set_ylabel("Max eq. plastic strain [-]")
        axes[1].plot(index, fraction, color=COLORS["ground"]); axes[1].set_ylabel("Plastic element fraction [-]")
        axes[2].plot(index, dissipation, color=COLORS["displacement"]); axes[2].set_ylabel("Plastic dissipation [J]"); axes[2].set_xlabel("Committed target-position row")
        for axis in axes: style_axis(axis)
        figure.suptitle(PLOT_TITLES_EN["history_pass_plastic"] + "\n" + PLOT_SUBTITLES_EN["history_pass_plastic"], fontweight="bold")
        save_png(figure, output); plt.close(figure)
    return output


def _configure_residual_surface_axis(
    axis: object,
    x_mm: np.ndarray,
    reference_y_mm: np.ndarray,
    residual_y_mm: np.ndarray,
) -> None:
    """Give the surface panel its own vertical display scale without changing data."""

    x_min = float(np.min(x_mm))
    x_max = float(np.max(x_mm))
    x_pad = max(0.02 * (x_max - x_min), 1.0e-9)
    y_min = float(min(np.min(reference_y_mm), np.min(residual_y_mm)))
    y_max = float(max(np.max(reference_y_mm), np.max(residual_y_mm)))
    y_pad = max(0.12 * (y_max - y_min), 1.0e-6)
    axis.set_aspect("auto")
    axis.set_xlim(x_min - x_pad, x_max + x_pad)
    axis.set_ylim(y_min - y_pad, y_max + y_pad)


def write_residual_state_png(result: FixedMeshElastoplasticHistoryPassResult, path: str | Path) -> Path:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    output = Path(path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    fields = result.final_fields; xy = fields.node_coordinates_m * 1e3
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], fields.element_connectivity)
    top_nodes = np.unique(result.prepared_mesh.imported_mesh.mesh.facets[:, result.prepared_mesh.imported_mesh.mesh.boundaries["contact"]])
    top_nodes = top_nodes[np.argsort(fields.node_coordinates_m[top_nodes, 0])]
    scale = 200.0
    top_x_mm = fields.node_coordinates_m[top_nodes, 0] * 1e3
    reference_y_mm = fields.node_coordinates_m[top_nodes, 1] * 1e3
    residual_y = (fields.node_coordinates_m[top_nodes, 1] + scale * fields.nodal_displacements_m[top_nodes, 1]) * 1e3
    with plt.rc_context(plot_style()):
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
        image0 = axes[0].tripcolor(triangulation, fields.equivalent_plastic_strain, shading="flat", cmap="magma")
        axes[0].triplot(triangulation, color="k", linewidth=0.15, alpha=0.25); figure.colorbar(image0, ax=axes[0], label="Equivalent plastic strain [-]")
        image1 = axes[1].tripcolor(triangulation, fields.von_mises_stress_Pa / 1e6, shading="flat", cmap="viridis")
        axes[1].triplot(triangulation, color="k", linewidth=0.15, alpha=0.25); figure.colorbar(image1, ax=axes[1], label="Residual von Mises [MPa]")
        axes[2].plot(top_x_mm, reference_y_mm, color=COLORS["reference"], label="Reference surface")
        axes[2].plot(top_x_mm, residual_y, color=COLORS["displacement"], label=f"Residual surface ({scale:g}x uy display)")
        axes[2].legend(fontsize=8, loc="best")
        for axis, title in zip(axes[:2], ("Final Plastic Zone", "Self-Equilibrated Residual Stress")):
            axis.set_aspect("equal", adjustable="box"); axis.set_xlabel("x [mm]"); axis.set_ylabel("y [mm]"); axis.set_title(title); style_axis(axis)
        axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("y [mm]"); axes[2].set_title("Residual Surface Profile"); style_axis(axes[2])
        _configure_residual_surface_axis(axes[2], top_x_mm, reference_y_mm, residual_y)
        figure.suptitle(PLOT_TITLES_EN["history_pass_residual_state"] + "\n" + PLOT_SUBTITLES_EN["history_pass_residual_state"], fontweight="bold")
        save_png(figure, output); plt.close(figure)
    return output


def write_surface_profile_png(result: FixedMeshElastoplasticHistoryPassResult, path: str | Path) -> Path:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    output = Path(path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    fields = result.final_fields; mesh = result.prepared_mesh.imported_mesh.mesh
    top_nodes = np.unique(mesh.facets[:, mesh.boundaries["contact"]]); top_nodes = top_nodes[np.argsort(fields.node_coordinates_m[top_nodes, 0])]
    x = fields.node_coordinates_m[top_nodes, 0] * 1e3
    reference_y = fields.node_coordinates_m[top_nodes, 1]
    residual_y = reference_y + fields.nodal_displacements_m[top_nodes, 1]
    trajectory = result.moving_load.positions[-1].pass_load.trajectory_result
    nominal_x = np.asarray([point.x_m for point in trajectory.sample_points]) * 1e3
    nominal_y = np.asarray([point.surface_y_m for point in trajectory.sample_points])
    baseline = min(float(np.min(nominal_y)), float(np.min(residual_y)))
    with plt.rc_context(plot_style()):
        figure, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
        axis.plot(x, (reference_y - baseline) * 1e6, label="reference_surface", color=COLORS["reference"])
        axis.plot(nominal_x, (nominal_y - baseline) * 1e6, label="nominal_ground_surface (4C1)", color=COLORS["ground"])
        axis.plot(x, (residual_y - baseline) * 1e6, label="final_residual_surface", color=COLORS["displacement"])
        axis.set_xlabel("x [mm]"); axis.set_ylabel("Height relative to plot baseline [um]"); style_axis(axis); axis.legend()
        axis.set_title("Fixed Reference, Ideal Nominal Ground, and Final Residual Surfaces\nResidual Surface Is Structural Displacement, Not Measured Roughness", fontweight="bold")
        save_png(figure, output); plt.close(figure)
    return output


def build_summary(
    result: FixedMeshElastoplasticHistoryPassResult,
    artifacts: dict[str, Path],
    *,
    field_evolution: dict[str, object] | None = None,
) -> dict[str, object]:
    rows = result.history_rows; final = rows[-1]
    max_displacement = max(rows, key=lambda row: row.maximum_displacement_m)
    max_plastic = max(rows, key=lambda row: row.maximum_equivalent_plastic_strain)
    summary = {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "history_pass_schema_version": result.case.history_pass_schema_version,
        "unit_system": result.case.unit_system,
        "model_type": result.case.model_type,
        "normalized_input": result.case.to_dict(),
        "mesh_reuse": {"generation_count": result.moving_load.mesh_generation_count, "import_count": result.moving_load.mesh_import_count, "node_count": result.prepared_mesh.node_count, "triangle_count": result.prepared_mesh.element_count, "same_nodes_elements_material_points_for_complete_pass": True},
        "pass": {"target_position_count": len(result.position_records), "history_row_count_including_final_unloaded": len(rows), "total_transition_substep_count": result.total_transition_substep_count, "total_retry_count": result.total_retry_count, "maximum_newton_iterations": result.maximum_newton_iterations, "maximum_balance_residual_N": result.maximum_balance_residual_N},
        "force": {"maximum_absolute_Fx_N": max(abs(row.Fx_N) for row in rows), "maximum_absolute_Fy_N": max(abs(row.Fy_N) for row in rows), "targets_reused_from_phase_6b1": True, "partial_facet_consistent_p1_assembly": True, "width_or_thickness_reapplied": False},
        "state_transfer": {"displacement_transferred_between_substeps": True, "stress_and_plastic_state_transferred_between_substeps": True, "plastic_history_transferred_between_substeps": True, "trial_state_committed_only_after_global_newton_convergence": True, "failed_target_positions_skipped": False, "equivalent_plastic_strain_nondecreasing": all(b.maximum_equivalent_plastic_strain + 1e-14 >= a.maximum_equivalent_plastic_strain for a, b in zip(rows, rows[1:]))},
        "maximum_response": {"maximum_displacement_m": max_displacement.maximum_displacement_m, "maximum_displacement_position_id": max_displacement.position_id, "maximum_equivalent_plastic_strain": max_plastic.maximum_equivalent_plastic_strain, "maximum_plastic_position_id": max_plastic.position_id, "maximum_plastic_element_count": max(row.plastic_element_count for row in rows), "accumulated_plastic_dissipation_J": result.accumulated_plastic_dissipation_J},
        "final_unloaded": {"external_force_N": [final.Fx_N, final.Fy_N], "external_force_norm_N": float(np.linalg.norm(final.target_load_vector_N)), "fixed_reaction_N": [final.fixed_reaction_x_N, final.fixed_reaction_y_N], "balance_residual_N": final.balance_residual_N, "maximum_residual_displacement_m": result.final_statistics.maximum_displacement_m, "von_mises_area_weighted_mean_Pa": result.final_statistics.von_mises_area_weighted_mean_Pa, "von_mises_area_weighted_p95_Pa": result.final_statistics.von_mises_area_weighted_p95_Pa, "von_mises_area_weighted_p99_Pa": result.final_statistics.von_mises_area_weighted_p99_Pa, "maximum_equivalent_plastic_strain": result.final_statistics.maximum_equivalent_plastic_strain, "plastic_element_count": result.final_statistics.plastic_element_count, "plastic_element_fraction": result.final_statistics.plastic_element_fraction, "accumulated_plastic_dissipation_J": result.accumulated_plastic_dissipation_J},
        "surface_definitions": {"reference_surface": "Fixed FEM reference-mesh top edge.", "nominal_ground_surface": "Ideal Phase 4C1 complete-pass surface; the FEM mesh is not reshaped to it.", "final_residual_surface": "Fixed reference top edge plus final unloaded residual displacement.", "roughness_prediction": False},
        "limitations": ["Fixed reference mesh with no material deletion or remeshing.", "Empirical grinding force with one-way coupling.", "No grain mechanism decomposition, thermal coupling, 3D analysis, GUI update, or portable rebuild."],
        "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
    }
    if field_evolution is not None:
        summary["field_evolution"] = field_evolution
    return summary


def write_summary_json(summary: dict[str, object], path: str | Path) -> Path:
    output = Path(path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="ascii", newline="\n")
    return output


__all__ = ["RESULT_FORMAT", "HISTORY_CSV_FIELDS", "build_summary", "write_history_csv", "write_force_history_png", "write_plastic_history_png", "write_residual_state_png", "write_surface_profile_png", "write_summary_json", "write_overview_png", "write_vtu", "write_nodes_csv", "write_elements_csv"]
