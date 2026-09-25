"""Phase 4C3A CSV, PNG, and deterministic JSON exporters."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_TITLES_EN

import csv
import json
import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "grindcae-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from grindcae import __version__
from grindcae.plotting import COLORS, add_scope_footer, plot_style, save_png, style_axis, unique_legend
from grindcae.mesh_contract import PHYSICAL_GROUP_NAMES
from grindcae.trajectory import REGIONS

from .validation import EvolvedMeshValidationResult


RESULT_FORMAT = "grindcae_phase_4c3a_evolved_surface_mesh"
TOP_BOUNDARY_CSV_FIELDS = (
    "facet_id",
    "node_start_id",
    "node_end_id",
    "region",
    "x_start_m",
    "y_start_m",
    "x_end_m",
    "y_end_m",
    "projected_length_x_m",
    "actual_facet_length_m",
    "tangent_x",
    "tangent_y",
    "outward_normal_x",
    "outward_normal_y",
)

REGION_COLORS = {
    "ground": COLORS["ground"],
    "contact_arc": COLORS["contact"],
    "unprocessed": COLORS["unprocessed_line"],
}

def write_top_boundary_facets_csv(
    result: EvolvedMeshValidationResult,
    output_path: str | Path,
) -> Path:
    """Write the 0-based, left-to-right top-boundary facet audit."""

    if not isinstance(result, EvolvedMeshValidationResult):
        raise TypeError("result must be an EvolvedMeshValidationResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TOP_BOUNDARY_CSV_FIELDS)
        writer.writeheader()
        for facet in result.top_boundary_facets:
            writer.writerow(facet.to_dict())
    return path


def _plot_boundary(
    axis: object,
    points_mm: np.ndarray,
    facets: np.ndarray,
    facet_ids: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 2.0,
) -> None:
    first = True
    for facet_id in facet_ids:
        nodes = facets[:, int(facet_id)]
        axis.plot(
            points_mm[0, nodes],
            points_mm[1, nodes],
            color=color,
            linewidth=linewidth,
            label=label if first else None,
        )
        first = False


def write_mesh_png(
    result: EvolvedMeshValidationResult,
    output_path: str | Path,
) -> Path:
    """Render a readable full mesh and a true local evolved-top close-up."""

    if not isinstance(result, EvolvedMeshValidationResult):
        raise TypeError("result must be an EvolvedMeshValidationResult")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    mesh = result.skfem_mesh
    points_mm = mesh.p * 1.0e3
    triangulation = mtri.Triangulation(points_mm[0], points_mm[1], mesh.t.T)
    height = result.case.trajectory.workpiece.original_surface_height_m
    depth = result.case.trajectory.single_pass.depth_of_cut_m
    width = result.case.trajectory.workpiece.length_m
    trajectory = result.trajectory
    lowest_x = trajectory.case.single_pass.wheel_lowest_point_x_m
    arc = trajectory.exact_arc_projected_length_m
    if trajectory.effective_arc_interval_m is not None:
        local_start, local_end = trajectory.effective_arc_interval_m
    else:
        local_start, local_end = trajectory.theoretical_arc_interval_m
    local_margin = max(0.75 * arc, 0.01 * width)
    local_lower = max(0.0, min(local_start, local_end, lowest_x) - local_margin)
    local_upper = min(width, max(local_start, local_end, lowest_x) + local_margin)
    if local_upper <= local_lower:
        local_lower, local_upper = 0.0, width

    with plt.rc_context(plot_style()):
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.76, wspace=0.24)
        try:
            full_axis, local_axis = axes
            full_axis.triplot(triangulation, color=COLORS["mesh"], linewidth=0.42)
            boundary_styles = {
                "fixed": (COLORS["fixed"], "Fixed support"),
                "free_left": (COLORS["free"], "Free sides"),
                "free_right": (COLORS["free"], None),
            }
            for name, (color, label) in boundary_styles.items():
                _plot_boundary(
                    full_axis,
                    points_mm,
                    mesh.facets,
                    np.asarray(mesh.boundaries[name], dtype=np.int64),
                    color=color,
                    label=label,
                    linewidth=2.4,
                )
            for region in REGIONS:
                region_facets = [
                    facet for facet in result.top_boundary_facets if facet.region == region
                ]
                for index, facet in enumerate(region_facets):
                    full_axis.plot(
                        [1.0e3 * facet.x_start_m, 1.0e3 * facet.x_end_m],
                        [1.0e3 * facet.y_start_m, 1.0e3 * facet.y_end_m],
                        color=REGION_COLORS[region],
                        linewidth=4.0 if region == "contact_arc" else 3.0,
                        solid_capstyle="round",
                        label=region.replace("_", " ").title() if index == 0 else None,
                    )
            full_axis.set_aspect("equal", adjustable="box")
            full_axis.set_xlabel("Workpiece x [mm]")
            full_axis.set_ylabel("Workpiece y [mm]")
            full_axis.set_title("A  Real Gmsh mesh and physical boundaries", loc="left", fontweight="bold")
            style_axis(full_axis)
            unique_legend(full_axis, loc="upper center", ncol=3, fontsize=7.8)
            full_axis.text(
                0.98,
                0.06,
                f"{result.node_count} nodes | {result.triangle_count} triangles",
                transform=full_axis.transAxes,
                ha="right",
                fontsize=8.5,
                color="#555555",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

            for region in REGIONS:
                region_facets = [
                    facet for facet in result.top_boundary_facets if facet.region == region
                ]
                for index, facet in enumerate(region_facets):
                    local_axis.plot(
                        [1.0e3 * facet.x_start_m, 1.0e3 * facet.x_end_m],
                        [1.0e6 * (facet.y_start_m - height), 1.0e6 * (facet.y_end_m - height)],
                        color=REGION_COLORS[region],
                        linewidth=2.2,
                        marker="o" if region == "contact_arc" else None,
                        markersize=3.2,
                        label=f"{region.replace('_', ' ').title()} facets" if index == 0 else None,
                    )
            if trajectory.effective_arc_interval_m is not None:
                arc_points = [
                    point for point in trajectory.sample_points if point.region == "contact_arc"
                ]
                local_axis.plot(
                    [1.0e3 * point.x_m for point in arc_points],
                    [1.0e6 * (point.surface_y_m - height) for point in arc_points],
                    color="#A85C10",
                    linestyle="--",
                    linewidth=1.4,
                    label="Exact circle",
                )
            local_axis.axvline(
                1.0e3 * lowest_x,
                color=COLORS["displacement"],
                linestyle=":",
                linewidth=1.5,
                label="Wheel lowest point",
            )
            local_axis.axhline(0.0, color=COLORS["reference"], linestyle="--", linewidth=1.0)
            local_axis.set_xlim(1.0e3 * local_lower, 1.0e3 * local_upper)
            local_axis.set_ylim(-1.18 * 1.0e6 * depth, 0.18 * 1.0e6 * depth)
            local_axis.set_xlabel("Workpiece x [mm]")
            local_axis.set_ylabel("Elevation from original [um]")
            local_axis.set_title("B  Circular arc mesh close-up", loc="left", fontweight="bold")
            style_axis(local_axis, grid_axis="y")
            unique_legend(local_axis, loc="lower center", ncol=2, fontsize=7.8)
            local_axis.text(
                0.02,
                0.07,
                f"state: {trajectory.pass_state}\n"
                f"arc facets = {result.arc_facet_count} | Larc = {1.0e3 * arc:.4g} mm",
                transform=local_axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
            )

            figure.suptitle(
        PLOT_TITLES_EN["evolved_mesh"],
                y=0.94,
                fontsize=15,
                fontweight="bold",
            )
            figure.text(
                0.5,
                0.835,
                "Read left to right: complete validated mesh -> true OCC circular-arc discretization",
                ha="center",
                color="#4D4D4D",
                fontsize=9.5,
            )
            add_scope_footer(
                figure,
                "Geometry and mesh only. No applied load, FEM displacement, stress, true contact pressure, roughness, wear, or multiple passes.",
                y=0.055,
            )
            save_png(figure, path)
        finally:
            plt.close(figure)
    return path

def build_summary(
    result: EvolvedMeshValidationResult,
    artifacts: dict[str, str],
) -> dict[str, object]:
    """Build the deterministic phase 4C3A result payload."""

    trajectory = result.trajectory
    case = result.case
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "evolved_mesh_schema_version": case.evolved_mesh_schema_version,
        "trajectory_schema_version": case.trajectory.trajectory_schema_version,
        "unit_system": case.unit_system,
        "normalized_input": case.to_dict(),
        "pass_state": trajectory.pass_state,
        "relative_feed_direction": case.trajectory.single_pass.relative_feed_direction,
        "derived_geometry": {
            "workpiece_length_m": case.trajectory.workpiece.length_m,
            "original_surface_height_m": case.trajectory.workpiece.original_surface_height_m,
            "ground_surface_height_m": trajectory.ground_surface_height_m,
            "depth_of_cut_m": case.trajectory.single_pass.depth_of_cut_m,
            "wheel_radius_m": trajectory.wheel_radius_m,
            "exact_arc_projected_length_Larc_m": trajectory.exact_arc_projected_length_m,
            "wheel_lowest_point_x_m": case.trajectory.single_pass.wheel_lowest_point_x_m,
            "theoretical_arc_interval_m": list(trajectory.theoretical_arc_interval_m),
            "effective_arc_interval_m": (
                None
                if trajectory.effective_arc_interval_m is None
                else list(trajectory.effective_arc_interval_m)
            ),
            "effective_arc_projected_length_m": trajectory.effective_contact_length_m,
        },
        "surface_regions": {
            "segments": [segment.to_dict() for segment in trajectory.surface_segments],
            "ground_projected_length_m": trajectory.region_length_m("ground"),
            "contact_arc_projected_length_m": trajectory.region_length_m("contact_arc"),
            "unprocessed_projected_length_m": trajectory.region_length_m("unprocessed"),
        },
        "curve_and_facet_lengths": {
            "effective_arc_projected_length_m": trajectory.effective_contact_length_m,
            "exact_arc_curve_length_m": result.exact_arc_curve_length_m,
            "discrete_arc_facet_length_m": result.discrete_arc_facet_length_m,
            "complete_top_projected_length_m": result.complete_top_projected_length_m,
            "exact_complete_top_curve_length_m": result.exact_complete_top_curve_length_m,
            "discrete_complete_top_facet_length_m": result.discrete_complete_top_facet_length_m,
            "arc_curve_minus_facet_length_m": (
                result.exact_arc_curve_length_m - result.discrete_arc_facet_length_m
            ),
        },
        "areas": {
            "original_rectangle_area_m2": result.original_rectangle_area_m2,
            "exact_evolved_domain_area_m2": result.exact_evolved_domain_area_m2,
            "exact_removed_cross_section_area_m2": result.exact_removed_cross_section_area_m2,
            "discrete_triangle_area_m2": result.discrete_triangle_area_m2,
            "discrete_boundary_polygon_area_m2": result.discrete_boundary_polygon_area_m2,
            "curvature_discretization_area_error_m2": result.curvature_discretization_area_error_m2,
        },
        "mesh": {
            "format": "Gmsh 4.1 ASCII",
            "element_order": 1,
            "triangle_only_domain": True,
            "recombined": False,
            "node_count": result.node_count,
            "triangle_count": result.triangle_count,
            "estimated_vector_p1_degrees_of_freedom": result.generation.estimated_vector_p1_degrees_of_freedom,
            "actual_vector_p1_degrees_of_freedom": result.actual_vector_p1_degrees_of_freedom,
            "raw_gmsh_node_count": result.generation.raw_gmsh_node_count,
            "raw_gmsh_triangle_count": result.generation.raw_gmsh_triangle_count,
            "generated_element_node_count": result.generation.generated_element_node_count,
            "maximum_vector_p1_degrees_of_freedom": result.generation.maximum_vector_p1_degrees_of_freedom,
            "top_curve_count": result.generation.top_curve_count,
            "arc_curve_entity_type": result.generation.arc_curve_type,
            "requested_arc_facet_count": result.generation.requested_arc_facet_count,
            "arc_facet_count": result.arc_facet_count,
        },
        "physical_groups": {
            "names": list(PHYSICAL_GROUP_NAMES),
            "dimensions": {
                "domain": 2,
                "fixed": 1,
                "contact": 1,
                "free_left": 1,
                "free_right": 1,
            },
            "boundary_projected_lengths_m": result.boundary_projected_lengths_m,
            "boundary_actual_lengths_m": result.boundary_actual_lengths_m,
            "complete_boundary_coverage": True,
            "overlapping_boundary_facets": False,
        },
        "top_boundary_facets": {
            "count": len(result.top_boundary_facets),
            "indexing": "0-based scikit-fem facet and node IDs",
            "ordering": "physical x from left to right",
            "regions": list(REGIONS),
            "outward_normal_convention": "left-to-right tangent rotated counterclockwise",
        },
        "import_validation": {
            "meshio_read": result.meshio_read_validated,
            "scikit_fem_from_meshio": result.scikit_fem_import_validated,
            "meshio_point_count": result.node_count,
            "scikit_fem_nvertices": result.node_count,
            "meshio_triangle_count": result.triangle_count,
            "scikit_fem_nelements": result.triangle_count,
            "same_generated_msh_reused": True,
            "second_validation_mesh_generated": False,
        },
        "assumptions_and_limitations": [
            "This workflow constructs only the ideal two-dimensional geometry and a first-order Gmsh mesh.",
            "The prescribed depth represents ideal material removal and is not elastic deformation or roughness.",
            "The contact_arc boundary is a true OCC circle arc; its first-order mesh facets are straight chords.",
            "No load is assembled and no FEM displacement, strain, stress, reaction, or contact pressure is calculated.",
            "No Hertz contact, friction, thermal field, wheel wear, grain randomness, ploughing, plasticity, edge chipping, vibration, chatter, multiple passes, or industrial validation is included.",
            "The exact curved area and exact curve length are reported separately from their first-order discrete counterparts.",
        ],
        "artifacts": artifacts,
    }


def serialize_summary(summary: dict[str, object]) -> str:
    return json.dumps(
        summary,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
