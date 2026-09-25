"""Machine-readable and visual artifact exporters for Phase 6A.3."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_SUBTITLES_EN, PLOT_TITLES_EN

import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from grindcae import __version__
from grindcae.elastoplastic_fem.exporters import (
    ELEMENT_CSV_FIELDS,
    INCREMENT_HISTORY_CSV_FIELDS,
    NODE_CSV_FIELDS,
    write_elements_csv,
    write_increment_history_csv,
    write_nodes_csv,
    write_pngs as write_fem_pngs,
    write_vtu,
)
from grindcae.evolved_fem.exporters import (
    ACTIVE_CONTACT_FACET_CSV_FIELDS,
    write_active_contact_facets_csv,
    write_mesh_png,
)
from grindcae.plotting import COLORS, plot_style, save_png, style_axis

from .models import SinglePositionElastoplasticCase
from .recovery import SurfaceRecoveryResult
from .workflow import SinglePositionElastoplasticResult


RESULT_FORMAT = "grindcae_phase_6a3_single_position_elastoplastic"
SURFACE_RECOVERY_CSV_FIELDS = (
    "surface_node_id",
    "x_m",
    "nominal_y_m",
    "peak_loaded_y_m",
    "residual_unloaded_y_m",
    "peak_uy_m",
    "residual_uy_m",
    "elastic_recovery_y_m",
    "surface_region",
)


def write_surface_recovery_csv(
    recovery: SurfaceRecoveryResult, output_path: str | Path
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SURFACE_RECOVERY_CSV_FIELDS)
        writer.writeheader()
        for point in recovery.points:
            writer.writerow(
                {field: getattr(point, field) for field in SURFACE_RECOVERY_CSV_FIELDS}
            )
    return path


def write_surface_recovery_png(
    recovery: SurfaceRecoveryResult, output_path: str | Path
) -> Path:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([point.x_m for point in recovery.points]) * 1.0e3
    nominal = np.array([point.nominal_y_m for point in recovery.points])
    peak = np.array([point.peak_loaded_y_m for point in recovery.points])
    residual = np.array([point.residual_unloaded_y_m for point in recovery.points])
    baseline = float(np.min(nominal)) if nominal.size else 0.0
    nominal_relative_um = (nominal - baseline) * 1.0e6
    peak_relative_um = (peak - baseline) * 1.0e6
    residual_relative_um = (residual - baseline) * 1.0e6
    elastic_recovery_um = np.array(
        [point.elastic_recovery_y_m for point in recovery.points]
    ) * 1.0e6
    with plt.rc_context(plot_style()):
        figure, (top, bottom) = plt.subplots(
            2, 1, figsize=(10.5, 6.8), sharex=True, constrained_layout=True
        )
        try:
            top.plot(
                x,
                nominal_relative_um,
                color=COLORS["reference"],
                label="Nominal surface",
            )
            top.plot(
                x,
                peak_relative_um,
                color=COLORS["contact"],
                label="Peak-loaded surface",
            )
            top.plot(
                x,
                residual_relative_um,
                color=COLORS["ground"],
                label="Residual-unloaded surface",
            )
            top.set_ylabel("y relative to minimum nominal y [um]")
            top.set_title(
        PLOT_TITLES_EN["single_position_surface_recovery"] + "\n"
        + PLOT_SUBTITLES_EN["single_position_surface_recovery"],
                fontweight="bold",
            )
            style_axis(top)
            top.legend(loc="best")
            bottom.plot(
                x,
                elastic_recovery_um,
                color=COLORS["displacement"],
                label="Elastic recovery y",
            )
            bottom.axhline(0.0, color="black", linewidth=0.8)
            bottom.set_xlabel("x [mm]")
            bottom.set_ylabel("residual y - peak y [um]")
            style_axis(bottom)
            bottom.legend(loc="best")
            save_png(figure, path)
        finally:
            plt.close(figure)
    return path


def write_result_pngs(
    result: SinglePositionElastoplasticResult,
    paths: dict[str, Path],
) -> None:
    """Reuse field plotting, then replace the 6A.2-only mesh plot with 4C geometry."""

    write_fem_pngs(
        result.history.case,
        result.history,
        result.peak_fields,
        result.residual_fields,
        paths,
    )
    write_mesh_png(result.facet_load, paths["mesh_png"])
    write_surface_recovery_png(
        result.surface_recovery, paths["surface_recovery_png"]
    )


def _state_summary(record: object, statistics: object) -> dict[str, object]:
    return {
        "increment_id": record.increment_id,
        "load_factor": record.load_factor,
        "external_force_N": [record.external_force_x_N, record.external_force_y_N],
        "fixed_support_reaction_N": [
            record.fixed_reaction_x_N,
            record.fixed_reaction_y_N,
        ],
        "balance_residual_N": [
            record.balance_residual_x_N,
            record.balance_residual_y_N,
        ],
        "balance_residual_norm_N": record.balance_residual_norm_N,
        "statistics": statistics.to_dict(),
    }


def build_summary(
    result: SinglePositionElastoplasticResult,
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    case = result.case
    load = result.pass_load
    facet = result.facet_load
    history = result.history
    recovery_values = np.array(
        [point.elastic_recovery_y_m for point in result.surface_recovery.points],
        dtype=float,
    )
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "single_position_elastoplastic_schema_version": (
            case.single_position_elastoplastic_schema_version
        ),
        "unit_system": case.unit_system,
        "model_type": case.model_type,
        "normalized_input": case.to_dict(),
        "position_and_contact": {
            "wheel_lowest_point_x_m": case.pass_load.trajectory.single_pass.wheel_lowest_point_x_m,
            "relative_feed_direction": case.pass_load.trajectory.single_pass.relative_feed_direction,
            "tangential_force_direction": case.pass_load.force_mapping.tangential_force_direction,
            "pass_state": load.pass_state,
            "contact_ratio": load.contact_ratio,
            "effective_projected_contact_length_m": load.effective_contact_length_m,
            "actual_active_facet_length_m": facet.actual_facet_length_sum_m,
            "active_facet_count": len(facet.facets),
        },
        "current_empirical_force": {
            "tangential_force_magnitude_N": load.current_tangential_force_magnitude_N,
            "normal_force_magnitude_N": load.current_normal_force_magnitude_N,
            "resultant_force_magnitude_N": load.current_resultant_force_magnitude_N,
            "signed_Fx_N": load.current_Fx_N,
            "signed_Fy_N": load.current_Fy_N,
            "uniform_qx_N_per_m": load.current_qx_N_per_m,
            "uniform_qy_N_per_m": load.current_qy_N_per_m,
        },
        "facet_load_conservation": {
            "rule": "tx = qx * dx / ds; ty = qy * dx / ds",
            "projected_length_sum_m": facet.projected_length_sum_m,
            "actual_facet_length_sum_m": facet.actual_facet_length_sum_m,
            "integrated_Fx_N": facet.integrated_Fx_N,
            "integrated_Fy_N": facet.integrated_Fy_N,
            "target_Fx_N": load.current_Fx_N,
            "target_Fy_N": load.current_Fy_N,
        },
        "analysis": {
            "type": "plane_strain",
            "kinematics": "two-dimensional small deformation",
            "element": "fixed first-order triangle with one committed J2 state",
            "thickness_m": case.analysis.thickness,
            "thickness_usage": "internal force and tangent integration only; boundary traction is not multiplied again",
            "same_mesh_for_loading_and_unloading": True,
        },
        "mesh": {
            "node_count": result.prepared_mesh.node_count,
            "triangle_count": result.prepared_mesh.element_count,
            "total_vector_p1_degrees_of_freedom": result.prepared_mesh.total_degrees_of_freedom,
        },
        "material": {
            **case.material.to_dict(),
            "internal_hardening_modulus_Pa": case.material.internal_hardening_modulus,
        },
        "newton_history": {
            "converged_increment_count_excluding_initial": len(history.increments) - 1,
            "maximum_iterations_used": history.maximum_newton_iterations,
            "step_reduction_occurred": history.any_step_reduction,
            "step_reduction_count": sum(
                record.retry_count for record in history.increments
            ),
            "maximum_balance_residual_norm_N": history.maximum_balance_residual_norm_N,
        },
        "snapshots": {
            "peak_loaded": _state_summary(history.peak, result.peak_statistics),
            "residual_unloaded": _state_summary(
                history.residual, result.residual_statistics
            ),
        },
        "surface_definitions": {
            "nominal_surface": "Final evolved Gmsh top-boundary node coordinates without FEM displacement.",
            "peak_loaded_surface": "Nominal node coordinates plus peak-load displacement.",
            "residual_unloaded_surface": "Nominal node coordinates plus fully unloaded residual displacement.",
            "elastic_recovery_y_m": "residual_unloaded_y_m - peak_loaded_y_m",
            "surface_region_shared_endpoint_rule": "contact_arc takes priority over ground or unprocessed",
            "display_deformation_scale_in_csv": False,
            "maximum_absolute_elastic_recovery_y_m": result.surface_recovery.maximum_absolute_elastic_recovery_y_m,
            "mean_signed_elastic_recovery_y_m": result.surface_recovery.mean_elastic_recovery_y_m,
            "minimum_signed_elastic_recovery_y_m": float(np.min(recovery_values)) if recovery_values.size else 0.0,
            "maximum_signed_elastic_recovery_y_m": float(np.max(recovery_values)) if recovery_values.size else 0.0,
        },
        "field_locations": {
            "displacement": "nodes",
            "strain_stress_plastic_state": "one unsmoothed constant value per triangle",
        },
        "units": {
            "length": "m",
            "force": "N",
            "line_load": "N/m",
            "stress": "Pa",
            "strain": "dimensionless",
        },
        "assumptions_and_limitations": [
            "Phase 6A.3 is one fixed wheel position and is not a complete pass scan.",
            "The empirical grinding load is one-way coupled to a fixed small-deformation plane-strain J2 FEM mesh.",
            "There is no true contact iteration, wheel mesh, chip separation, element deletion, damage, thermal field, roughness prediction, or accumulated multi-position plastic history.",
            "Residual unloaded surface is a structural residual-displacement result and is not a prediction of roughness, burrs, pile-up, or chip formation.",
            "Area-weighted p95 and p99 are primary stress statistics; raw element maxima are mesh-dependent auxiliary values.",
            "The GUI and retained Windows portable package continue to use the earlier linear-elastic routes.",
            "Phase 6B validation, contact calibration, mesh-convergence study, and experimental validation remain pending.",
        ],
        "artifacts": {
            key: {"filename": path.name, "path": str(path.resolve())}
            for key, path in artifact_paths.items()
        },
    }


def write_summary_json(summary: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
        newline="\n",
    )
    return path


__all__ = [
    "ACTIVE_CONTACT_FACET_CSV_FIELDS",
    "ELEMENT_CSV_FIELDS",
    "INCREMENT_HISTORY_CSV_FIELDS",
    "NODE_CSV_FIELDS",
    "RESULT_FORMAT",
    "SURFACE_RECOVERY_CSV_FIELDS",
    "build_summary",
    "write_active_contact_facets_csv",
    "write_elements_csv",
    "write_increment_history_csv",
    "write_nodes_csv",
    "write_result_pngs",
    "write_summary_json",
    "write_surface_recovery_csv",
    "write_vtu",
]
