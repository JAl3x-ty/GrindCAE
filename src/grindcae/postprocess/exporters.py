"""Write phase 3 JSON, VTU, CSV, and headless PNG artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import meshio
import numpy as np
from numpy.typing import NDArray

from grindcae.solver import (
    FacetLoadFullFieldSolution,
    FullFieldSolution,
    LocalLoadFullFieldSolution,
)
from grindcae.plotting import COLORS, plot_style, save_png, style_axis, unique_legend

from .recovery import RecoveredFields, StressStatistics


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "vtu": "results.vtu",
    "nodes_csv": "nodes.csv",
    "elements_csv": "elements.csv",
    "mesh_png": "mesh.png",
    "displacement_magnitude_png": "displacement_magnitude.png",
    "von_mises_stress_png": "von_mises_stress.png",
    "principal_stress_1_in_plane_png": "principal_stress_1_in_plane.png",
    "deformed_shape_png": "deformed_shape.png",
}

NODE_CSV_FIELDS = (
    "node_id",
    "x_m",
    "y_m",
    "ux_m",
    "uy_m",
    "displacement_magnitude_m",
)

ELEMENT_CSV_FIELDS = (
    "element_id",
    "node_0",
    "node_1",
    "node_2",
    "centroid_x_m",
    "centroid_y_m",
    "area_m2",
    "strain_xx",
    "strain_yy",
    "engineering_shear_strain_xy",
    "derived_strain_zz",
    "stress_xx_Pa",
    "stress_yy_Pa",
    "shear_stress_xy_Pa",
    "von_mises_stress_Pa",
    "principal_stress_1_in_plane_Pa",
    "principal_stress_2_in_plane_Pa",
)

FullFieldSolutionLike = (
    FullFieldSolution | LocalLoadFullFieldSolution | FacetLoadFullFieldSolution
)


def _artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return {key: directory / name for key, name in ARTIFACT_FILENAMES.items()}


def write_vtu(fields: RecoveredFields, output_path: str | Path) -> Path:
    """Write triangle-only VTU data with nodal and element fields."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.column_stack(
        (fields.node_coordinates, np.zeros(fields.node_count, dtype=float))
    )
    displacement = np.column_stack(
        (fields.nodal_displacements, np.zeros(fields.node_count, dtype=float))
    )
    cell_data = {
        "strain_xx": [fields.strain_xx],
        "strain_yy": [fields.strain_yy],
        "engineering_shear_strain_xy": [
            fields.engineering_shear_strain_xy
        ],
        "derived_strain_zz": [fields.derived_strain_zz],
        "stress_xx_Pa": [fields.stress_xx],
        "stress_yy_Pa": [fields.stress_yy],
        "shear_stress_xy_Pa": [fields.shear_stress_xy],
        "von_mises_stress_Pa": [fields.von_mises_stress],
        "principal_stress_1_in_plane_Pa": [
            fields.principal_stress_1_in_plane
        ],
        "principal_stress_2_in_plane_Pa": [
            fields.principal_stress_2_in_plane
        ],
    }
    meshio.write(
        path,
        meshio.Mesh(
            points=points,
            cells=[("triangle", fields.element_connectivity)],
            point_data={
                "displacement_m": displacement,
                "displacement_magnitude_m": fields.displacement_magnitude,
            },
            cell_data=cell_data,
        ),
    )
    return path


def write_nodes_csv(fields: RecoveredFields, output_path: str | Path) -> Path:
    """Write 0-based nodal coordinates and displacements with SI units."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(NODE_CSV_FIELDS)
        for node_id in range(fields.node_count):
            writer.writerow(
                (
                    node_id,
                    fields.node_coordinates[node_id, 0],
                    fields.node_coordinates[node_id, 1],
                    fields.nodal_displacements[node_id, 0],
                    fields.nodal_displacements[node_id, 1],
                    fields.displacement_magnitude[node_id],
                )
            )
    return path


def write_elements_csv(fields: RecoveredFields, output_path: str | Path) -> Path:
    """Write 0-based triangle connectivity and native element fields."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ELEMENT_CSV_FIELDS)
        for element_id in range(fields.element_count):
            nodes = fields.element_connectivity[element_id]
            writer.writerow(
                (
                    element_id,
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    fields.element_centroids[element_id, 0],
                    fields.element_centroids[element_id, 1],
                    fields.element_areas[element_id],
                    fields.strain_xx[element_id],
                    fields.strain_yy[element_id],
                    fields.engineering_shear_strain_xy[element_id],
                    fields.derived_strain_zz[element_id],
                    fields.stress_xx[element_id],
                    fields.stress_yy[element_id],
                    fields.shear_stress_xy[element_id],
                    fields.von_mises_stress[element_id],
                    fields.principal_stress_1_in_plane[element_id],
                    fields.principal_stress_2_in_plane[element_id],
                )
            )
    return path


def _deformation_scale(
    fields: RecoveredFields,
    requested_scale: str | float,
) -> float:
    if isinstance(requested_scale, str):
        if requested_scale != "auto":
            raise ValueError("deformation scale must be 'auto' or a positive number")
        maximum_displacement = float(np.max(fields.displacement_magnitude))
        if maximum_displacement == 0.0:
            return 1.0
        extents = np.ptp(fields.node_coordinates, axis=0)
        reference_extent = float(np.max(extents))
        return max(1.0, 0.08 * reference_extent / maximum_displacement)

    scale = float(requested_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("deformation scale must be 'auto' or a positive number")
    return scale


def deformation_scale_argument(value: str) -> str | float:
    """Parse the shared CLI deformation-scale argument."""

    if value == "auto":
        return value
    try:
        scale = float(value)
    except ValueError as exc:
        raise ValueError(
            "deformation scale must be 'auto' or a positive number"
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("deformation scale must be 'auto' or a positive number")
    return scale


def _configure_axes(ax: object) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    style_axis(ax)


def _plot_mesh(
    solution: FullFieldSolutionLike,
    fields: RecoveredFields,
    output_path: Path,
    active_facets: NDArray[np.int32] | None,
    result_title: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    coordinates_mm = 1.0e3 * fields.node_coordinates
    triangulation = mtri.Triangulation(
        coordinates_mm[:, 0],
        coordinates_mm[:, 1],
        fields.element_connectivity,
    )
    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    try:
        ax.triplot(triangulation, color=COLORS["mesh"], linewidth=0.45)
        boundary_styles = {
            "fixed": (COLORS["fixed"], "Fixed support", 3.0),
            "contact": (
                COLORS["unprocessed_line"] if active_facets is not None else COLORS["contact"],
                "Candidate top boundary" if active_facets is not None else "Loaded top boundary",
                1.8 if active_facets is not None else 3.0,
            ),
            "free_left": (COLORS["free"], "Free sides", 2.2),
            "free_right": (COLORS["free"], None, 2.2),
        }
        for name, (color, label, linewidth) in boundary_styles.items():
            facets = solution.mesh.facets[:, solution.mesh.boundaries[name]]
            for facet_index, nodes in enumerate(facets.T):
                ax.plot(
                    coordinates_mm[nodes, 0],
                    coordinates_mm[nodes, 1],
                    color=color,
                    linewidth=linewidth,
                    label=label if facet_index == 0 else None,
                )
        if active_facets is not None:
            active_edges = solution.mesh.facets[:, active_facets]
            for facet_index, nodes in enumerate(active_edges.T):
                ax.plot(
                    coordinates_mm[nodes, 0],
                    coordinates_mm[nodes, 1],
                    color=COLORS["contact"],
                    linewidth=5.0,
                    solid_capstyle="round",
                    label="Active local load" if facet_index == 0 else None,
                )
        _configure_axes(ax)
        unique_legend(ax, loc="upper center", ncol=4, fontsize=8.5)
        ax.set_title(
            f"Mesh and boundary conditions\n{result_title}",
            fontweight="bold",
        )
        ax.text(
            0.99,
            0.03,
            f"{fields.node_count} nodes | {fields.element_count} triangles",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#555555",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        save_png(fig, output_path)
    finally:
        plt.close(fig)


def _plot_nodal_field(
    fields: RecoveredFields,
    values: NDArray[np.float64],
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    coordinates_mm = 1.0e3 * fields.node_coordinates
    triangulation = mtri.Triangulation(
        coordinates_mm[:, 0],
        coordinates_mm[:, 1],
        fields.element_connectivity,
    )
    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    try:
        image = ax.tripcolor(
            triangulation, values, shading="gouraud", cmap="magma"
        )
        ax.triplot(triangulation, color="k", alpha=0.13, linewidth=0.24)
        maximum_index = int(np.argmax(values))
        maximum_xy = coordinates_mm[maximum_index]
        maximum_value = float(values[maximum_index])
        annotation_dx = 18 if maximum_xy[0] <= float(np.mean(coordinates_mm[:, 0])) else -18
        annotation_ha = "left" if annotation_dx > 0 else "right"
        annotation_dy = 20 if maximum_xy[1] <= float(np.mean(coordinates_mm[:, 1])) else -20
        annotation_va = "bottom" if annotation_dy > 0 else "top"
        ax.scatter(
            [maximum_xy[0]],
            [maximum_xy[1]],
            color="white",
            edgecolor=COLORS["displacement"],
            linewidth=1.3,
            s=46,
            zorder=5,
        )
        ax.annotate(
            f"maximum {maximum_value:.3g} um",
            maximum_xy,
            xytext=(annotation_dx, annotation_dy),
            textcoords="offset points",
            ha=annotation_ha,
            va=annotation_va,
            fontsize=8.5,
            color="#5F3A99",
            arrowprops={"arrowstyle": "->", "color": "#5F3A99", "linewidth": 0.9},
        )
        _configure_axes(ax)
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, label=colorbar_label)
        save_png(fig, output_path)
    finally:
        plt.close(fig)


def _plot_element_field(
    fields: RecoveredFields,
    values: NDArray[np.float64],
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    peak_label: str,
    center_zero: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    coordinates_mm = 1.0e3 * fields.node_coordinates
    triangulation = mtri.Triangulation(
        coordinates_mm[:, 0],
        coordinates_mm[:, 1],
        fields.element_connectivity,
    )
    norm = None
    if center_zero and float(np.min(values)) < 0.0 < float(np.max(values)):
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(
            vmin=float(np.min(values)),
            vcenter=0.0,
            vmax=float(np.max(values)),
        )
    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    try:
        image = ax.tripcolor(
            triangulation,
            facecolors=values,
            shading="flat",
            cmap=cmap,
            norm=norm,
        )
        ax.triplot(triangulation, color="k", alpha=0.12, linewidth=0.2)
        peak_index = int(np.argmax(values))
        peak_xy = 1.0e3 * fields.element_centroids[peak_index]
        peak_value = float(values[peak_index])
        annotation_dx = 18 if peak_xy[0] <= float(np.mean(coordinates_mm[:, 0])) else -18
        annotation_ha = "left" if annotation_dx > 0 else "right"
        annotation_dy = 20 if peak_xy[1] <= float(np.mean(coordinates_mm[:, 1])) else -20
        annotation_va = "bottom" if annotation_dy > 0 else "top"
        ax.scatter(
            [peak_xy[0]],
            [peak_xy[1]],
            color="white",
            edgecolor=COLORS["stress"],
            linewidth=1.2,
            s=44,
            zorder=5,
        )
        ax.annotate(
            f"{peak_label}: {peak_value:.3g} MPa",
            peak_xy,
            xytext=(annotation_dx, annotation_dy),
            textcoords="offset points",
            ha=annotation_ha,
            va=annotation_va,
            fontsize=8.5,
            color="#8A3B27",
            arrowprops={"arrowstyle": "->", "color": "#8A3B27", "linewidth": 0.9},
        )
        _configure_axes(ax)
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, label=colorbar_label)
        save_png(fig, output_path)
    finally:
        plt.close(fig)


def _plot_deformed_shape(
    solution: FullFieldSolutionLike,
    fields: RecoveredFields,
    output_path: Path,
    deformation_scale: float,
    result_title: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    original_mm = 1.0e3 * fields.node_coordinates
    deformed_mm = 1.0e3 * (
        fields.node_coordinates + deformation_scale * fields.nodal_displacements
    )
    deformed = mtri.Triangulation(
        deformed_mm[:, 0], deformed_mm[:, 1], fields.element_connectivity
    )
    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    try:
        first_outline_segment = True
        for name in ("fixed", "contact", "free_left", "free_right"):
            facets = solution.mesh.facets[:, solution.mesh.boundaries[name]]
            for nodes in facets.T:
                ax.plot(
                    original_mm[nodes, 0],
                    original_mm[nodes, 1],
                    color="0.62",
                    linewidth=1.2,
                    label="Undeformed outline" if first_outline_segment else None,
                )
                first_outline_segment = False
        image = ax.tripcolor(
            deformed,
            1.0e6 * fields.displacement_magnitude,
            shading="gouraud",
            cmap="magma",
        )
        ax.triplot(deformed, color="k", alpha=0.16, linewidth=0.24)
        maximum_index = int(np.argmax(fields.displacement_magnitude))
        maximum_xy = deformed_mm[maximum_index]
        maximum_value = 1.0e6 * float(fields.displacement_magnitude[maximum_index])
        annotation_dx = 18 if maximum_xy[0] <= float(np.mean(deformed_mm[:, 0])) else -18
        annotation_ha = "left" if annotation_dx > 0 else "right"
        annotation_dy = 20 if maximum_xy[1] <= float(np.mean(deformed_mm[:, 1])) else -20
        annotation_va = "bottom" if annotation_dy > 0 else "top"
        ax.scatter(
            [maximum_xy[0]],
            [maximum_xy[1]],
            color="white",
            edgecolor=COLORS["displacement"],
            linewidth=1.3,
            s=46,
            zorder=5,
        )
        ax.annotate(
            f"maximum {maximum_value:.3g} um",
            maximum_xy,
            xytext=(annotation_dx, annotation_dy),
            textcoords="offset points",
            ha=annotation_ha,
            va=annotation_va,
            fontsize=8.5,
            color="#5F3A99",
            arrowprops={"arrowstyle": "->", "color": "#5F3A99", "linewidth": 0.9},
        )
        _configure_axes(ax)
        unique_legend(ax, loc="upper right", fontsize=8.5)
        ax.set_title(f"Deformed shape\n{result_title}", fontweight="bold")
        ax.text(
            0.02,
            0.97,
            f"Deformation scale: {deformation_scale:.6g}x (visual only)",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
        )
        fig.colorbar(image, ax=ax, label="displacement magnitude [\N{MICRO SIGN}m]")
        save_png(fig, output_path)
    finally:
        plt.close(fig)


def write_pngs(
    solution: FullFieldSolutionLike,
    fields: RecoveredFields,
    output_paths: dict[str, Path],
    deformation_scale: str | float = "auto",
    *,
    active_facets: NDArray[np.int32] | None = None,
    result_title: str = "Equivalent boundary-load FEM result",
) -> float:
    """Write the five headless static plots and return the visual-only scale."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    with matplotlib.rc_context(plot_style()):
        scale = _deformation_scale(fields, deformation_scale)
        if active_facets is None and isinstance(
            solution, (LocalLoadFullFieldSolution, FacetLoadFullFieldSolution)
        ):
            if solution.active_facets.size:
                active_facets = solution.active_facets
        if active_facets is not None:
            values = np.asarray(active_facets)
            if values.ndim != 1 or values.size == 0 or not np.issubdtype(
                values.dtype, np.integer
            ):
                raise ValueError("active_facets must be a non-empty integer array")
            active_facets = np.asarray(values, dtype=np.int32)
        _plot_mesh(
            solution,
            fields,
            output_paths["mesh_png"],
            active_facets,
            result_title,
        )
        _plot_nodal_field(
            fields,
            1.0e6 * fields.displacement_magnitude,
            output_paths["displacement_magnitude_png"],
            title=f"Displacement magnitude\n{result_title}",
            colorbar_label="displacement magnitude [\N{MICRO SIGN}m]",
        )
        _plot_element_field(
            fields,
            1.0e-6 * fields.von_mises_stress,
            output_paths["von_mises_stress_png"],
            title=f"Element-constant von Mises stress\n{result_title}",
            colorbar_label="von Mises stress [MPa]",
            cmap="inferno",
            peak_label="raw maximum (mesh-dependent)",
        )
        _plot_element_field(
            fields,
            1.0e-6 * fields.principal_stress_1_in_plane,
            output_paths["principal_stress_1_in_plane_png"],
            title=f"Element-constant first in-plane principal stress\n{result_title}",
            colorbar_label="first in-plane principal stress [MPa]",
            cmap="coolwarm",
            peak_label="maximum",
            center_zero=True,
        )
        _plot_deformed_shape(
            solution,
            fields,
            output_paths["deformed_shape_png"],
            scale,
            result_title,
        )
    return scale

def build_summary(
    solution: FullFieldSolutionLike,
    statistics: StressStatistics,
    artifacts: dict[str, Path],
    deformation_scale: float,
) -> dict[str, object]:
    """Build the serializable phase 3 summary without full field arrays."""

    return {
        "result_format": "grindcae_phase_3_postprocess",
        "solver_summary": solution.solver_summary.to_dict(),
        "indexing": {
            "node_index_base": 0,
            "element_index_base": 0,
            "connectivity_index_base": 0,
        },
        "field_definitions": {
            "displacement_m": "nodal in-plane displacement (ux, uy); no uz field",
            "strain_xx": "element-constant dux/dx",
            "strain_yy": "element-constant duy/dy",
            "engineering_shear_strain_xy": (
                "element-constant gamma_xy = dux/dy + duy/dx; "
                "tensor epsilon_xy = gamma_xy / 2"
            ),
            "derived_strain_zz": (
                "plane-stress constitutive derivation; the model has no uz "
                "displacement degree of freedom"
            ),
            "stress": "element-constant plane stress in Pa with sigma_zz = 0",
            "von_mises_stress": "plane-stress von Mises equivalent stress in Pa",
            "principal_stress_1_in_plane": "larger in-plane principal stress in Pa",
            "principal_stress_2_in_plane": "smaller in-plane principal stress in Pa",
            "third_principal_stress": "zero under the plane-stress assumption",
        },
        "field_locations": {
            "coordinates_and_displacement": "mesh nodes",
            "strain_stress_and_derived_fields": (
                "native constant values on each first-order triangle; "
                "no nodal stress smoothing"
            ),
        },
        "stress_statistics": statistics.to_dict(),
        "peak_value_caution": {
            "raw_element_maximum_mesh_dependent": True,
            "fixed_free_corner_singularity": (
                "the fixed/free boundary junction can produce a stress singularity"
            ),
            "mesh_dependence": (
                "raw element maximum stress can change as the mesh changes"
            ),
            "quantile_limit": (
                "area-weighted p95 and p99 do not replace a mesh-convergence study"
            ),
            "failure_limit": (
                "this program does not automatically determine yielding, failure, "
                "or a safety factor"
            ),
        },
        "visualization": {
            "deformation_scale": deformation_scale,
            "deformation_scale_is_visual_only": True,
            "stored_displacements_are_unscaled": True,
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def write_summary_json(summary: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def export_all(
    solution: FullFieldSolutionLike,
    fields: RecoveredFields,
    statistics: StressStatistics,
    output_dir: str | Path,
    *,
    deformation_scale: str | float = "auto",
) -> tuple[dict[str, Path], dict[str, object]]:
    """Write all phase 3 artifacts and return their paths and summary."""

    paths = _artifact_paths(output_dir)
    with tempfile.TemporaryDirectory(prefix="grindcae-export-") as temp_dir:
        temporary_paths = {
            key: Path(temp_dir) / filename
            for key, filename in ARTIFACT_FILENAMES.items()
        }
        write_vtu(fields, temporary_paths["vtu"])
        write_nodes_csv(fields, temporary_paths["nodes_csv"])
        write_elements_csv(fields, temporary_paths["elements_csv"])
        scale = write_pngs(
            solution, fields, temporary_paths, deformation_scale
        )
        summary = build_summary(solution, statistics, paths, scale)
        write_summary_json(summary, temporary_paths["summary_json"])
        for key, source_path in temporary_paths.items():
            if key != "summary_json":
                shutil.copy2(source_path, paths[key])
        shutil.copy2(temporary_paths["summary_json"], paths["summary_json"])
    return paths, summary
