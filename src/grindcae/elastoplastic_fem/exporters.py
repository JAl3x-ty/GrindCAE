"""Auditable CSV, VTU, PNG, and summary exports for Phase 6A.2."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_SUBTITLES_EN, PLOT_TITLES_EN

import csv
import json
import math
import os
from pathlib import Path
import tempfile

import meshio
import numpy as np

from grindcae import __version__
from grindcae.plotting import COLORS, plot_style, save_png, style_axis

from .models import ElastoplasticFemCase
from .recovery import (
    ElastoplasticSnapshotFields,
    SnapshotStatistics,
    compute_snapshot_statistics,
)
from .workflow import LoadingUnloadingResult


RESULT_FORMAT = "grindcae_phase_6a2_fixed_mesh_elastoplastic_fem"

INCREMENT_HISTORY_CSV_FIELDS = (
    "increment_id",
    "segment",
    "requested_load_factor",
    "load_factor",
    "load_factor_increment",
    "retry_count",
    "step_reduction_occurred",
    "newton_iterations",
    "residual_norm_N",
    "correction_norm_m",
    "external_force_x_N",
    "external_force_y_N",
    "fixed_reaction_x_N",
    "fixed_reaction_y_N",
    "balance_residual_norm_N",
    "maximum_displacement_m",
    "maximum_equivalent_plastic_strain",
    "plastic_element_count",
    "plastic_element_fraction",
)

NODE_CSV_FIELDS = (
    "node_id",
    "x_m",
    "y_m",
    "peak_ux_m",
    "peak_uy_m",
    "peak_displacement_magnitude_m",
    "residual_ux_m",
    "residual_uy_m",
    "residual_displacement_magnitude_m",
)

_SNAPSHOT_ELEMENT_FIELDS = (
    "strain_xx",
    "strain_yy",
    "engineering_shear_strain_xy",
    "strain_zz",
    "stress_xx_Pa",
    "stress_yy_Pa",
    "stress_zz_Pa",
    "shear_stress_xy_Pa",
    "von_mises_stress_Pa",
    "plastic_strain_xx",
    "plastic_strain_yy",
    "plastic_strain_zz",
    "plastic_engineering_shear_strain_xy",
    "equivalent_plastic_strain",
    "current_yield_strength_Pa",
    "increment_class",
)

ELEMENT_CSV_FIELDS = (
    "element_id",
    "node_0",
    "node_1",
    "node_2",
    "centroid_x_m",
    "centroid_y_m",
    "area_m2",
    *(f"peak_{name}" for name in _SNAPSHOT_ELEMENT_FIELDS),
    *(f"residual_{name}" for name in _SNAPSHOT_ELEMENT_FIELDS),
)


def write_increment_history_csv(
    result: LoadingUnloadingResult, output_path: str | Path
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INCREMENT_HISTORY_CSV_FIELDS)
        writer.writeheader()
        for record in result.increments:
            state = record.state
            element_count = len(state.element_states)
            plastic_count = int(
                np.count_nonzero(record.equivalent_plastic_strain > 1.0e-15)
            )
            writer.writerow(
                {
                    "increment_id": record.increment_id,
                    "segment": record.segment,
                    "requested_load_factor": record.requested_load_factor,
                    "load_factor": record.load_factor,
                    "load_factor_increment": record.load_factor_increment,
                    "retry_count": record.retry_count,
                    "step_reduction_occurred": str(record.step_reduction_occurred).lower(),
                    "newton_iterations": getattr(state, "newton_iterations", 0),
                    "residual_norm_N": getattr(state, "residual_norm_N", 0.0),
                    "correction_norm_m": getattr(state, "correction_norm_m", 0.0),
                    "external_force_x_N": getattr(state, "external_force_x_N", 0.0),
                    "external_force_y_N": getattr(state, "external_force_y_N", 0.0),
                    "fixed_reaction_x_N": getattr(state, "fixed_reaction_x_N", 0.0),
                    "fixed_reaction_y_N": getattr(state, "fixed_reaction_y_N", 0.0),
                    "balance_residual_norm_N": getattr(state, "balance_residual_norm_N", 0.0),
                    "maximum_displacement_m": getattr(state, "maximum_displacement_m", 0.0),
                    "maximum_equivalent_plastic_strain": float(
                        np.max(record.equivalent_plastic_strain)
                    ),
                    "plastic_element_count": plastic_count,
                    "plastic_element_fraction": plastic_count / element_count,
                }
            )
    return path


def write_nodes_csv(
    peak: ElastoplasticSnapshotFields,
    residual: ElastoplasticSnapshotFields,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(NODE_CSV_FIELDS)
        for node_id in range(peak.node_count):
            writer.writerow(
                (
                    node_id,
                    peak.node_coordinates_m[node_id, 0],
                    peak.node_coordinates_m[node_id, 1],
                    peak.nodal_displacements_m[node_id, 0],
                    peak.nodal_displacements_m[node_id, 1],
                    peak.displacement_magnitude_m[node_id],
                    residual.nodal_displacements_m[node_id, 0],
                    residual.nodal_displacements_m[node_id, 1],
                    residual.displacement_magnitude_m[node_id],
                )
            )
    return path


def _element_snapshot_values(
    fields: ElastoplasticSnapshotFields, element_id: int
) -> tuple[object, ...]:
    plastic = fields.plastic_strain_tensor[element_id]
    return (
        fields.strain_xx[element_id],
        fields.strain_yy[element_id],
        fields.engineering_shear_strain_xy[element_id],
        fields.strain_zz[element_id],
        fields.stress_xx_Pa[element_id],
        fields.stress_yy_Pa[element_id],
        fields.stress_zz_Pa[element_id],
        fields.shear_stress_xy_Pa[element_id],
        fields.von_mises_stress_Pa[element_id],
        plastic[0, 0],
        plastic[1, 1],
        plastic[2, 2],
        2.0 * plastic[0, 1],
        fields.equivalent_plastic_strain[element_id],
        fields.current_yield_strength_Pa[element_id],
        fields.increment_class[element_id],
    )


def write_elements_csv(
    peak: ElastoplasticSnapshotFields,
    residual: ElastoplasticSnapshotFields,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ELEMENT_CSV_FIELDS)
        for element_id in range(peak.element_count):
            nodes = peak.element_connectivity[element_id]
            writer.writerow(
                (
                    element_id,
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    peak.element_centroids_m[element_id, 0],
                    peak.element_centroids_m[element_id, 1],
                    peak.element_areas_m2[element_id],
                    *_element_snapshot_values(peak, element_id),
                    *_element_snapshot_values(residual, element_id),
                )
            )
    return path


def _cell_fields(prefix: str, fields: ElastoplasticSnapshotFields) -> dict[str, list[np.ndarray]]:
    plastic = fields.plastic_strain_tensor
    class_code = np.array(
        [{"initial": 0, "elastic": 1, "plastic": 2}[value] for value in fields.increment_class],
        dtype=np.int32,
    )
    return {
        f"{prefix}_strain_xx": [fields.strain_xx],
        f"{prefix}_strain_yy": [fields.strain_yy],
        f"{prefix}_engineering_shear_strain_xy": [fields.engineering_shear_strain_xy],
        f"{prefix}_strain_zz": [fields.strain_zz],
        f"{prefix}_stress_xx_Pa": [fields.stress_xx_Pa],
        f"{prefix}_stress_yy_Pa": [fields.stress_yy_Pa],
        f"{prefix}_stress_zz_Pa": [fields.stress_zz_Pa],
        f"{prefix}_shear_stress_xy_Pa": [fields.shear_stress_xy_Pa],
        f"{prefix}_von_mises_stress_Pa": [fields.von_mises_stress_Pa],
        f"{prefix}_plastic_strain_xx": [plastic[:, 0, 0]],
        f"{prefix}_plastic_strain_yy": [plastic[:, 1, 1]],
        f"{prefix}_plastic_strain_zz": [plastic[:, 2, 2]],
        f"{prefix}_plastic_engineering_shear_strain_xy": [2.0 * plastic[:, 0, 1]],
        f"{prefix}_equivalent_plastic_strain": [fields.equivalent_plastic_strain],
        f"{prefix}_current_yield_strength_Pa": [fields.current_yield_strength_Pa],
        f"{prefix}_increment_class_code": [class_code],
    }


def write_vtu(
    peak: ElastoplasticSnapshotFields,
    residual: ElastoplasticSnapshotFields,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.column_stack((peak.node_coordinates_m, np.zeros(peak.node_count)))
    peak_displacement = np.column_stack((peak.nodal_displacements_m, np.zeros(peak.node_count)))
    residual_displacement = np.column_stack(
        (residual.nodal_displacements_m, np.zeros(residual.node_count))
    )
    cell_data = _cell_fields("peak", peak)
    cell_data.update(_cell_fields("residual", residual))
    meshio.write(
        path,
        meshio.Mesh(
            points=points,
            cells=[("triangle", peak.element_connectivity)],
            point_data={
                "peak_displacement_m": peak_displacement,
                "peak_displacement_magnitude_m": peak.displacement_magnitude_m,
                "residual_displacement_m": residual_displacement,
                "residual_displacement_magnitude_m": residual.displacement_magnitude_m,
            },
            cell_data=cell_data,
        ),
    )
    return path


def _configure_axis(axis: object) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    style_axis(axis)


def _triangulation(fields: ElastoplasticSnapshotFields) -> object:
    import matplotlib.tri as mtri

    xy = 1.0e3 * fields.node_coordinates_m
    return mtri.Triangulation(xy[:, 0], xy[:, 1], fields.element_connectivity)


def write_pngs(
    case: ElastoplasticFemCase,
    result: LoadingUnloadingResult,
    peak: ElastoplasticSnapshotFields,
    residual: ElastoplasticSnapshotFields,
    paths: dict[str, Path],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(plot_style()):
        triangulation = _triangulation(peak)
        xy_mm = 1.0e3 * peak.node_coordinates_m
        figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        try:
            axis.triplot(triangulation, color=COLORS["mesh"], linewidth=0.5)
            mesh = result.prepared_mesh.imported_mesh.mesh
            for name, color, label in (
                ("fixed", COLORS["fixed"], "Fixed bottom"),
                ("contact", COLORS["unprocessed_line"], "Candidate top"),
            ):
                for index, nodes in enumerate(mesh.facets[:, mesh.boundaries[name]].T):
                    axis.plot(xy_mm[nodes, 0], xy_mm[nodes, 1], color=color, linewidth=2.3, label=label if index == 0 else None)
            for index, nodes in enumerate(mesh.facets[:, result.prepared_mesh.active_facets].T):
                axis.plot(xy_mm[nodes, 0], xy_mm[nodes, 1], color=COLORS["contact"], linewidth=4.0, label="Loaded interval" if index == 0 else None)
            _configure_axis(axis)
            axis.legend(loc="best")
            axis.set_title(
                PLOT_TITLES_EN["fixed_mesh_j2_benchmark"] + "\n"
                + PLOT_SUBTITLES_EN["fixed_mesh_j2_benchmark"],
                fontweight="bold",
            )
            save_png(figure, paths["mesh_png"])
        finally:
            plt.close(figure)

        def nodal_plot(fields: ElastoplasticSnapshotFields, path: Path, title: str) -> None:
            figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
            try:
                image = axis.tripcolor(triangulation, 1.0e6 * fields.displacement_magnitude_m, shading="gouraud", cmap="magma")
                axis.triplot(triangulation, color="k", alpha=0.12, linewidth=0.2)
                _configure_axis(axis)
                axis.set_title(title, fontweight="bold")
                figure.colorbar(image, ax=axis, label="displacement magnitude [um]")
                save_png(figure, path)
            finally:
                plt.close(figure)

        def element_plot(values: np.ndarray, path: Path, title: str, label: str, cmap: str) -> None:
            figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
            try:
                image = axis.tripcolor(triangulation, facecolors=values, shading="flat", cmap=cmap)
                axis.triplot(triangulation, color="k", alpha=0.12, linewidth=0.2)
                _configure_axis(axis)
                axis.set_title(title, fontweight="bold")
                figure.colorbar(image, ax=axis, label=label)
                save_png(figure, path)
            finally:
                plt.close(figure)

        nodal_plot(peak, paths["peak_displacement_png"], PLOT_TITLES_EN["fixed_mesh_j2_displacement"])
        element_plot(1.0e-6 * peak.von_mises_stress_Pa, paths["peak_von_mises_stress_png"], PLOT_TITLES_EN["fixed_mesh_j2_von_mises"], "von Mises stress [MPa]", "inferno")
        element_plot(peak.equivalent_plastic_strain, paths["equivalent_plastic_strain_png"], PLOT_TITLES_EN["fixed_mesh_j2_plastic_strain"], "equivalent plastic strain [-]", "viridis")
        nodal_plot(residual, paths["residual_displacement_png"], "Fully Unloaded Residual Displacement")
        element_plot(1.0e-6 * residual.von_mises_stress_Pa, paths["residual_von_mises_stress_png"], "Fully Unloaded Residual von Mises Stress", "residual von Mises stress [MPa]", "inferno")


def _elastic_energy_J(
    case: ElastoplasticFemCase, fields: ElastoplasticSnapshotFields
) -> float:
    total = 0.0
    for element_id in range(fields.element_count):
        total_strain = np.array(
            [
                [fields.strain_xx[element_id], 0.5 * fields.engineering_shear_strain_xy[element_id], 0.0],
                [0.5 * fields.engineering_shear_strain_xy[element_id], fields.strain_yy[element_id], 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        elastic_strain = total_strain - fields.plastic_strain_tensor[element_id]
        stress = np.array(
            [
                [fields.stress_xx_Pa[element_id], fields.shear_stress_xy_Pa[element_id], 0.0],
                [fields.shear_stress_xy_Pa[element_id], fields.stress_yy_Pa[element_id], 0.0],
                [0.0, 0.0, fields.stress_zz_Pa[element_id]],
            ]
        )
        density = 0.5 * float(np.sum(stress * elastic_strain))
        total += density * fields.element_areas_m2[element_id] * case.analysis.thickness
    return total


def _plastic_dissipation_J(case: ElastoplasticFemCase, result: LoadingUnloadingResult) -> float:
    total = 0.0
    volumes = result.prepared_mesh.element_areas_m2 * case.analysis.thickness
    for previous, current in zip(result.increments, result.increments[1:]):
        for element_id, (old, new) in enumerate(
            zip(previous.state.element_states, current.state.element_states)
        ):
            delta = new.equivalent_plastic_strain - old.equivalent_plastic_strain
            if delta > 0.0:
                average_yield = 0.5 * (
                    old.current_yield_strength_Pa + new.current_yield_strength_Pa
                )
                total += average_yield * delta * volumes[element_id]
    return total


def build_summary(
    case: ElastoplasticFemCase,
    result: LoadingUnloadingResult,
    peak: ElastoplasticSnapshotFields,
    residual: ElastoplasticSnapshotFields,
    artifact_paths: dict[str, Path],
) -> dict[str, object]:
    peak_stats = compute_snapshot_statistics(peak)
    residual_stats = compute_snapshot_statistics(residual)
    prepared = result.prepared_mesh
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "elastoplastic_fem_schema_version": case.elastoplastic_fem_schema_version,
        "unit_system": case.unit_system,
        "model_type": case.model_type,
        "normalized_input": case.to_dict(),
        "analysis": {
            "type": "plane_strain",
            "kinematics": "two-dimensional small deformation",
            "element": "first_order_triangle_vector_P1_one_material_point",
            "thickness_m": case.analysis.thickness,
            "fixed_mesh": True,
        },
        "mesh": {
            "node_count": prepared.node_count,
            "triangle_count": prepared.element_count,
            "total_vector_p1_degrees_of_freedom": prepared.total_degrees_of_freedom,
            "active_top_facet_count": int(prepared.active_facets.size),
            "active_top_facet_length_m": prepared.active_facet_length_m,
        },
        "material": {
            **case.material.to_dict(),
            "internal_hardening_modulus_Pa": case.material.internal_hardening_modulus,
        },
        "newton_history": {
            "converged_increment_count_excluding_initial": len(result.increments) - 1,
            "maximum_iterations_used": result.maximum_newton_iterations,
            "step_reduction_occurred": result.any_step_reduction,
            "maximum_balance_residual_norm_N": result.maximum_balance_residual_norm_N,
        },
        "snapshots": {
            "peak": {
                "increment_id": result.peak.increment_id,
                "load_factor": result.peak.load_factor,
                "external_force_N": [result.peak.external_force_x_N, result.peak.external_force_y_N],
                "fixed_reaction_N": [result.peak.fixed_reaction_x_N, result.peak.fixed_reaction_y_N],
                "balance_residual_norm_N": result.peak.balance_residual_norm_N,
                "statistics": peak_stats.to_dict(),
                "elastic_strain_energy_J": _elastic_energy_J(case, peak),
            },
            "residual_unloaded": {
                "increment_id": result.residual.increment_id,
                "load_factor": result.residual.load_factor,
                "external_force_N": [result.residual.external_force_x_N, result.residual.external_force_y_N],
                "fixed_reaction_N": [result.residual.fixed_reaction_x_N, result.residual.fixed_reaction_y_N],
                "balance_residual_norm_N": result.residual.balance_residual_norm_N,
                "statistics": residual_stats.to_dict(),
                "elastic_strain_energy_J": _elastic_energy_J(case, residual),
            },
        },
        "cumulative_plastic_dissipation_J": _plastic_dissipation_J(case, result),
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
            "energy": "J",
        },
        "assumptions_and_limitations": [
            "Phase 6A.2 is an independent fixed-mesh plane-strain elastoplastic structural benchmark.",
            "The material is small-strain three-dimensional J2/von Mises associative plasticity with bilinear isotropic hardening.",
            "Each first-order triangle has one material point and one committed history state; fields are not nodally averaged or smoothed.",
            "The history is prescribed total force with load factor 0 to 1 to 0; thickness enters internal force and tangent integration only.",
            "There is no plane-stress elastoplasticity, large deformation, cyclic plasticity, kinematic hardening, damage, thermal coupling, or three-dimensional solve.",
            "Phase 6A.3 grinding-load integration remains pending; there is no 4C3B or 4C4 plastic history, GUI entry, Windows launcher change, or portable-package update.",
            "Raw element stress maxima are mesh-dependent and do not establish failure, safety factor, industrial accuracy, or mesh convergence.",
        ],
        "artifacts": {
            key: {
                "filename": path.name,
                "path": str(path.resolve()),
            }
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
