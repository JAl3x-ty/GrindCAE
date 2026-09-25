"""Transactional Phase 7A.3 comparison export and sole manual-acceptance plot."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import meshio
import numpy as np

from grindcae import __version__
from grindcae.presentation_labels import PLOT_SUBTITLES_EN, PLOT_TITLES_EN
from grindcae.elastoplastic_fem.exporters import write_elements_csv, write_vtu

from .workflow import MechanismHistoryPassResult


FILENAMES = {
    "summary_json": "summary.json", "comparison_history_csv": "comparison_history.csv",
    "mechanism_load_history_csv": "mechanism_load_history.csv",
    "baseline_final_results_vtu": "baseline_final_results.vtu", "mechanism_final_results_vtu": "mechanism_final_results.vtu",
    "final_nodes_csv": "final_nodes.csv", "baseline_final_elements_csv": "baseline_final_elements.csv",
    "mechanism_final_elements_csv": "mechanism_final_elements.csv", "reference_mesh_msh": "reference_mesh.msh",
    "mechanism_history_comparison_png": "mechanism_history_comparison.png",
}


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    return {key: root / name for key, name in FILENAMES.items()}


def _write_final_nodes(result: MechanismHistoryPassResult, path: Path) -> None:
    baseline = result.baseline.final_fields
    mechanism = result.mechanism.final_fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "node_id", "x_m", "y_m",
            "baseline_final_ux_m", "baseline_final_uy_m", "baseline_final_displacement_magnitude_m",
            "mechanism_final_ux_m", "mechanism_final_uy_m", "mechanism_final_displacement_magnitude_m",
        ))
        for node_id in range(baseline.node_count):
            writer.writerow((
                node_id,
                baseline.node_coordinates_m[node_id, 0], baseline.node_coordinates_m[node_id, 1],
                baseline.nodal_displacements_m[node_id, 0], baseline.nodal_displacements_m[node_id, 1], baseline.displacement_magnitude_m[node_id],
                mechanism.nodal_displacements_m[node_id, 0], mechanism.nodal_displacements_m[node_id, 1], mechanism.displacement_magnitude_m[node_id],
            ))


def _write_histories(result: MechanismHistoryPassResult, comparison: Path, mechanisms: Path) -> None:
    with comparison.open("w", encoding="utf-8", newline="") as stream:
        fields = ["position_id", "motion_coordinate_m", "pass_state", "contact_ratio", "baseline_Fx_N", "baseline_Fy_N", "mechanism_Fx_N", "mechanism_Fy_N", "nodal_load_l2_difference_N", "baseline_maximum_displacement_m", "mechanism_maximum_displacement_m", "baseline_maximum_von_mises_stress_Pa", "mechanism_maximum_von_mises_stress_Pa", "baseline_maximum_equivalent_plastic_strain", "mechanism_maximum_equivalent_plastic_strain"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for baseline, mechanism in zip(result.baseline.history_rows, result.mechanism.history_rows):
            writer.writerow({"position_id": baseline.position_id, "motion_coordinate_m": baseline.motion_coordinate_m, "pass_state": baseline.pass_state, "contact_ratio": baseline.contact_ratio, "baseline_Fx_N": baseline.Fx_N, "baseline_Fy_N": baseline.Fy_N, "mechanism_Fx_N": mechanism.Fx_N, "mechanism_Fy_N": mechanism.Fy_N, "nodal_load_l2_difference_N": float(np.linalg.norm(baseline.target_load_vector_N - mechanism.target_load_vector_N)), "baseline_maximum_displacement_m": baseline.maximum_displacement_m, "mechanism_maximum_displacement_m": mechanism.maximum_displacement_m, "baseline_maximum_von_mises_stress_Pa": baseline.maximum_von_mises_stress_Pa, "mechanism_maximum_von_mises_stress_Pa": mechanism.maximum_von_mises_stress_Pa, "baseline_maximum_equivalent_plastic_strain": baseline.maximum_equivalent_plastic_strain, "mechanism_maximum_equivalent_plastic_strain": mechanism.maximum_equivalent_plastic_strain})
    with mechanisms.open("w", encoding="utf-8", newline="") as stream:
        fields = ["position_id", "motion_coordinate_m", "pass_state", "contact_ratio", "mechanism_Fx_N", "mechanism_Fy_N", "total_force_residual_x_N", "total_force_residual_y_N", "Ft_rubbing_N", "Ft_ploughing_N", "Ft_cutting_N", "Fn_rubbing_N", "Fn_ploughing_N", "Fn_cutting_N", "Ft_rubbing_residual_N", "Ft_ploughing_residual_N", "Ft_cutting_residual_N", "Fn_rubbing_residual_N", "Fn_ploughing_residual_N", "Fn_cutting_residual_N"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for position, projection in zip(result.moving_load.positions, result.projections):
            baseline = position.pass_load
            row = {"position_id": position.position.position_id, "motion_coordinate_m": position.position.motion_coordinate_m, "pass_state": position.position.pass_state, "contact_ratio": baseline.contact_ratio, "mechanism_Fx_N": projection.total_Fx_N, "mechanism_Fy_N": projection.total_Fy_N, "total_force_residual_x_N": projection.total_Fx_N - baseline.current_Fx_N, "total_force_residual_y_N": projection.total_Fy_N - baseline.current_Fy_N}
            row.update(projection.mechanism_forces_N)
            row.update({key.replace("_N", "_residual_N"): value for key, value in projection.mechanism_residuals_N.items()})
            writer.writerow(row)


def _write_plot(result: MechanismHistoryPassResult, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction = result.prediction; full = max(result.moving_load.positions, key=lambda item: item.pass_load.contact_ratio)
    arc = full.pass_load.trajectory_result.exact_arc_projected_length_m
    u = np.asarray([point.normalized_x for point in prediction.distribution]); qn = np.asarray([point.q_n_total_N_per_m for point in prediction.distribution])
    full_normal_force = abs(full.pass_load.current_Fy_N)
    uniform = full_normal_force / arc
    mechanism_qn = (
        qn
        * prediction.contact_length_m
        / arc
        * full_normal_force
        / prediction.mechanism_prediction.phase4a_prediction.normal_force_magnitude_N
    )
    rows_b = result.baseline.history_rows[:-1]; rows_m = result.mechanism.history_rows[:-1]
    xy = result.baseline.final_fields.node_coordinates_m * 1e3; tri = mtri.Triangulation(xy[:, 0], xy[:, 1], result.baseline.final_fields.element_connectivity)
    b_eqp = result.baseline.final_fields.equivalent_plastic_strain; m_eqp = result.mechanism.final_fields.equivalent_plastic_strain; difference = m_eqp - b_eqp
    with plt.rc_context({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25}):
        figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.4), constrained_layout=True)
        axes[0, 0].plot(u, uniform * np.ones_like(u), label="Uniform baseline | integral matched")
        axes[0, 0].plot(u, mechanism_qn, label="Statistical mechanism")
        axes[0, 0].set(xlabel="Template coordinate u [-]", ylabel="Normal line load magnitude [N/m]", title="A. Full-contact line-load distribution"); axes[0, 0].legend(fontsize=7)
        coordinate = np.asarray([row.motion_coordinate_m for row in rows_b]) * 1e3
        for label, values, style in (("baseline Fx", [r.Fx_N for r in rows_b], "-"), ("baseline Fy", [r.Fy_N for r in rows_b], "-"), ("mechanism Fx", [r.Fx_N for r in rows_m], "--"), ("mechanism Fy", [r.Fy_N for r in rows_m], "--")):
            axes[0, 1].plot(coordinate, values, style, label=label)
        axes[0, 1].set(xlabel="Motion coordinate [mm]", ylabel="Signed force [N]", title="B. Identical complete-pass force history"); axes[0, 1].legend(fontsize=7, ncol=2)
        axes[0, 2].plot(coordinate, [r.maximum_equivalent_plastic_strain for r in rows_b], label="baseline")
        axes[0, 2].plot(coordinate, [r.maximum_equivalent_plastic_strain for r in rows_m], "--", label="mechanism")
        axes[0, 2].set(xlabel="Motion coordinate [mm]", ylabel="Maximum equivalent plastic strain [-]", title="C. Committed plastic history"); axes[0, 2].legend(fontsize=7)
        vmax = max(float(np.max(b_eqp)), float(np.max(m_eqp)), 1e-16)
        for axis, values, title in ((axes[1, 0], b_eqp, "D. Baseline final unloaded equivalent plastic strain"), (axes[1, 1], m_eqp, "E. Mechanism final unloaded equivalent plastic strain")):
            image = axis.tripcolor(tri, facecolors=values, shading="flat", cmap="magma", vmin=0.0, vmax=vmax); axis.triplot(tri, color="k", alpha=0.16, linewidth=0.15); axis.set_aspect("equal"); axis.set(xlabel="x [mm]", ylabel="y [mm]", title=title); figure.colorbar(image, ax=axis, label="Equivalent plastic strain [-]")
        extent = max(float(np.max(np.abs(difference))), 1e-16)
        image = axes[1, 2].tripcolor(tri, facecolors=difference, shading="flat", cmap="coolwarm", vmin=-extent, vmax=extent); axes[1, 2].triplot(tri, color="k", alpha=0.16, linewidth=0.15); axes[1, 2].set_aspect("equal"); axes[1, 2].set(xlabel="x [mm]", ylabel="y [mm]", title="F. Mechanism minus baseline final difference"); figure.colorbar(image, ax=axes[1, 2], label="Equivalent plastic strain difference [-]")
        grit = prediction.resolved_grit_specification
        figure.suptitle(
            f"{PLOT_TITLES_EN['mechanism_history_comparison']} | "
            f"{grit.resolution_path.replace('_', ' ').upper()} {grit.grit_designation} | "
            f"d50 ≈ {grit.representative_diameter_d50_m * 1e6:.1f} µm\n"
            + PLOT_SUBTITLES_EN["mechanism_history_comparison"],
            fontweight="bold",
        )
        figure.savefig(path, dpi=160); plt.close(figure)


def _summary(
    result: MechanismHistoryPassResult,
    outputs: dict[str, Path],
    field_evolution: dict[str, object] | None = None,
) -> dict[str, object]:
    baseline = result.baseline; mechanism = result.mechanism; final_b = baseline.history_rows[-1]; final_m = mechanism.history_rows[-1]
    total_identical = all(np.allclose([b.Fx_N, b.Fy_N], [m.Fx_N, m.Fy_N], rtol=1e-10, atol=1e-10) for b, m in zip(baseline.history_rows, mechanism.history_rows))
    top = result.moving_load.top_facets
    facet_length = float(np.mean([item.dx_projected_m for item in top]))
    active = [item.load_mapping.active_facet_count for item in result.moving_load.positions if item.pass_load.contact_ratio == 1.0]
    force_differences = [
        math.hypot(b.Fx_N - m.Fx_N, b.Fy_N - m.Fy_N)
        for b, m in zip(baseline.history_rows, mechanism.history_rows)
    ]
    nodal_differences = [
        float(np.linalg.norm(b.target_load_vector_N - m.target_load_vector_N))
        for b, m in zip(baseline.history_rows, mechanism.history_rows)
    ]
    component_maxima = {
        key: max(abs(projection.mechanism_residuals_N[key]) for projection in result.projections)
        for key in result.projections[0].mechanism_residuals_N
    }
    baseline_final = baseline.final_statistics.to_dict()
    mechanism_final = mechanism.final_statistics.to_dict()
    summary = {
        "result_format": "grindcae_phase_7a3_statistical_mechanism_elastoplastic_history_comparison",
        "package_version": __version__,
        "normalized_input": result.case.to_dict(),
        "phase7a2_prediction": result.prediction.to_dict(),
        "moving_load": {
            "target_position_count": len(result.moving_load.positions),
            "positions": [
                {
                    "position_id": item.position.position_id,
                    "motion_coordinate_m": item.position.motion_coordinate_m,
                    "wheel_lowest_point_x_m": item.position.wheel_lowest_point_x_m,
                    "pass_state": item.position.pass_state,
                    "contact_ratio": item.pass_load.contact_ratio,
                    "effective_contact_interval_m": item.pass_load.effective_contact_interval_m,
                }
                for item in result.moving_load.positions
            ],
        },
        "mesh_reuse": {
            "same_prepared_fixed_mesh": True,
            "same_node_coordinates": bool(np.array_equal(baseline.final_fields.node_coordinates_m, mechanism.final_fields.node_coordinates_m)),
            "same_element_topology": bool(np.array_equal(baseline.final_fields.element_connectivity, mechanism.final_fields.element_connectivity)),
            "node_count": result.prepared_mesh.node_count,
            "triangle_count": result.prepared_mesh.element_count,
        },
        "comparison": {
            "mode": result.case.comparison.mode,
            "total_force_history_identical": total_identical,
            "maximum_total_force_history_difference_N": max(force_differences, default=0.0),
            "maximum_target_nodal_load_vector_l2_difference_N": max(nodal_differences, default=0.0),
            "maximum_force_balance_residual_N": max(
                max(row.balance_residual_N for row in baseline.history_rows),
                max(row.balance_residual_N for row in mechanism.history_rows),
            ),
            "representative_nonzero_nodal_distribution_difference_present": any(value > 0.0 for value in nodal_differences),
            "independent_zero_load_initial_states": True,
            "same_shared_history_engine": True,
            "same_material_and_newton_settings": True,
        },
        "resolution": {
            "mean_top_facet_length_m": facet_length,
            "mean_active_facet_count_at_full_contact": float(np.mean(active)),
            "d50_m": result.prediction.resolved_grit_specification.representative_diameter_d50_m,
            "d50_to_mean_top_facet_length": result.prediction.resolved_grit_specification.representative_diameter_d50_m / facet_length,
            "equivalent_group_count": result.prediction.equivalent_group_count,
            "projection_method": "exact piecewise-linear template and vector-P1 edge-function Gauss-2 integration",
            "resolution_status": "macro_scale_conservative_projection_not_grain_resolved",
        },
        "baseline_final_unloaded": {
            **baseline_final,
            "external_force_N": [final_b.Fx_N, final_b.Fy_N],
            "maximum_residual_displacement_m": final_b.maximum_displacement_m,
            "maximum_residual_von_mises_stress_Pa": final_b.maximum_von_mises_stress_Pa,
            "accumulated_plastic_dissipation_J": baseline.accumulated_plastic_dissipation_J,
        },
        "mechanism_final_unloaded": {
            **mechanism_final,
            "external_force_N": [final_m.Fx_N, final_m.Fy_N],
            "maximum_residual_displacement_m": final_m.maximum_displacement_m,
            "maximum_residual_von_mises_stress_Pa": final_m.maximum_von_mises_stress_Pa,
            "accumulated_plastic_dissipation_J": mechanism.accumulated_plastic_dissipation_J,
        },
        "mechanism_projection": {
            "maximum_total_force_residual_x_N": max(abs(p.total_Fx_N - position.pass_load.current_Fx_N) for position, p in zip(result.moving_load.positions, result.projections)),
            "maximum_total_force_residual_y_N": max(abs(p.total_Fy_N - position.pass_load.current_Fy_N) for position, p in zip(result.moving_load.positions, result.projections)),
            "maximum_total_force_residual_N": max((float(np.hypot(p.total_Fx_N - position.pass_load.current_Fx_N, p.total_Fy_N - position.pass_load.current_Fy_N)) for position, p in zip(result.moving_load.positions, result.projections)), default=0.0),
            "component_maximum_absolute_residuals_N": component_maxima,
            "maximum_component_residual_N": max(component_maxima.values(), default=0.0),
        },
        "assumptions_and_cautions": [
            "The ductile_metal mechanism family and J2 structural material parameters are composed demonstration layers and are not specific-alloy experimental calibration.",
            "Statistical nonuniform loads are conservatively projected to macro-scale P1 boundary degrees of freedom.",
            "The mesh does not resolve local stress concentrations near a real 3.4 um grain; this is not explicit grain-resolved simulation.",
            "The provisional JIS-style #5000 and 3.4 um representative diameter require manufacturer or inspection verification.",
            "The effective active-grain density factor remains uncalibrated.",
            "No thermal coupling, brittle SiC fracture, GUI update, or portable rebuild is included.",
        ],
        "artifacts": {key: str(value) for key, value in outputs.items()},
    }
    if field_evolution is not None:
        summary["field_evolution"] = field_evolution
    return summary


def validate_artifacts(paths: dict[str, Path]) -> None:
    if set(paths) != set(FILENAMES):
        raise RuntimeError("Phase 7A.3 artifact set is invalid")
    for key, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"{key}: artifact is missing or empty")
    json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    for key in ("comparison_history_csv", "mechanism_load_history_csv", "final_nodes_csv", "baseline_final_elements_csv", "mechanism_final_elements_csv"):
        with paths[key].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"{key}: must contain data rows")
        for row in rows:
            for field, value in row.items():
                if field in {"pass_state", "peak_increment_class", "residual_increment_class"}:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    raise RuntimeError(f"{key}: contains non-finite numeric data")
    baseline = meshio.read(paths["baseline_final_results_vtu"])
    mechanism = meshio.read(paths["mechanism_final_results_vtu"])
    if not np.array_equal(baseline.points, mechanism.points):
        raise RuntimeError("Phase 7A.3 VTU node coordinates differ")
    if len(baseline.cells) != len(mechanism.cells) or any(
        left.type != right.type or not np.array_equal(left.data, right.data)
        for left, right in zip(baseline.cells, mechanism.cells)
    ):
        raise RuntimeError("Phase 7A.3 VTU element topology differs")
    if paths["mechanism_history_comparison_png"].read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("mechanism_history_comparison_png has an invalid signature")


def publish_artifacts(
    temporary: dict[str, Path],
    final: dict[str, Path],
    *,
    temporary_field_evolution: Path | None = None,
    final_field_evolution: Path | None = None,
) -> None:
    """Publish retained files and an optional managed sequence with summary last."""

    if set(temporary) != set(final) or "summary_json" not in temporary:
        raise RuntimeError("mechanism publication artifact sets are invalid")
    if (temporary_field_evolution is None) != (final_field_evolution is None):
        raise RuntimeError("mechanism field evolution publication paths are incomplete")
    output = final["summary_json"].parent
    existed = output.exists(); output.mkdir(parents=True, exist_ok=True)
    order = [key for key in temporary if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}; backups: dict[str, Path] = {}; published: list[str] = []
    staged_sequence: Path | None = None; backup_sequence: Path | None = None; sequence_published = False
    try:
        for key in order:
            staged[key] = output / f".{final[key].name}.{uuid.uuid4().hex}.new"
            shutil.copy2(temporary[key], staged[key])
            if final[key].exists():
                backups[key] = output / f".{final[key].name}.{uuid.uuid4().hex}.backup"
                shutil.copy2(final[key], backups[key])
        if temporary_field_evolution is not None and final_field_evolution is not None:
            staged_sequence = final_field_evolution.with_name(
                f".{final_field_evolution.name}.{uuid.uuid4().hex}.new"
            )
            shutil.copytree(temporary_field_evolution, staged_sequence)
        for key in order[:-1]:
            os.replace(staged.pop(key), final[key]); published.append(key)
        if staged_sequence is not None and final_field_evolution is not None:
            if final_field_evolution.exists():
                candidate_backup = final_field_evolution.with_name(
                    f".{final_field_evolution.name}.{uuid.uuid4().hex}.backup"
                )
                os.replace(final_field_evolution, candidate_backup)
                backup_sequence = candidate_backup
            try:
                os.replace(staged_sequence, final_field_evolution)
                staged_sequence = None; sequence_published = True
            except OSError:
                if backup_sequence is not None and not final_field_evolution.exists():
                    os.replace(backup_sequence, final_field_evolution); backup_sequence = None
                raise
        os.replace(staged.pop("summary_json"), final["summary_json"]); published.append("summary_json")
    except OSError:
        for key in reversed(published):
            if key in backups:
                os.replace(backups.pop(key), final[key])
            else:
                final[key].unlink(missing_ok=True)
        if sequence_published and final_field_evolution is not None:
            if backup_sequence is not None:
                shutil.rmtree(final_field_evolution)
                os.replace(backup_sequence, final_field_evolution); backup_sequence = None
            else:
                shutil.rmtree(final_field_evolution)
        elif backup_sequence is not None and final_field_evolution is not None:
            os.replace(backup_sequence, final_field_evolution); backup_sequence = None
        raise
    finally:
        for path in (*staged.values(), *backups.values()): path.unlink(missing_ok=True)
        if staged_sequence is not None and staged_sequence.exists():
            shutil.rmtree(staged_sequence, ignore_errors=True)
        if backup_sequence is not None and backup_sequence.exists():
            shutil.rmtree(backup_sequence, ignore_errors=True)
        if not existed:
            try: output.rmdir()
            except OSError: pass


def stage_mechanism_history_result(
    result: MechanismHistoryPassResult,
    staged: dict[str, Path],
    summary_outputs: dict[str, Path],
    *,
    field_evolution: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write and validate the retained ten artifacts without publishing them."""

    _write_histories(result, staged["comparison_history_csv"], staged["mechanism_load_history_csv"])
    write_vtu(result.baseline.maximum_plastic_fields, result.baseline.final_fields, staged["baseline_final_results_vtu"])
    write_vtu(result.mechanism.maximum_plastic_fields, result.mechanism.final_fields, staged["mechanism_final_results_vtu"])
    _write_final_nodes(result, staged["final_nodes_csv"])
    write_elements_csv(result.baseline.maximum_plastic_fields, result.baseline.final_fields, staged["baseline_final_elements_csv"])
    write_elements_csv(result.mechanism.maximum_plastic_fields, result.mechanism.final_fields, staged["mechanism_final_elements_csv"])
    shutil.copy2(result.moving_load.msh_source, staged["reference_mesh_msh"])
    _write_plot(result, staged["mechanism_history_comparison_png"])
    summary = _summary(result, summary_outputs, field_evolution)
    staged["summary_json"].write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    validate_artifacts(staged)
    return summary


def export_mechanism_history_result(result: MechanismHistoryPassResult, output_dir: str | Path) -> tuple[dict[str, Path], dict[str, object]]:
    final = artifact_paths(output_dir)
    with tempfile.TemporaryDirectory(prefix="grindcae-7a3-") as temporary:
        staged = artifact_paths(Path(temporary))
        summary = stage_mechanism_history_result(result, staged, final)
        publish_artifacts(staged, final)
        validate_artifacts(final)
    return final, summary
