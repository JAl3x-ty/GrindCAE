"""Phase 4C3B active-facet CSV and evolved-mesh visualization."""

from __future__ import annotations

from grindcae.presentation_labels import PLOT_TITLES_EN

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray

from grindcae.plotting import COLORS, add_scope_footer, plot_style, save_png, style_axis, unique_legend
from grindcae.postprocess import RecoveredFields

from .loading import EvolvedFacetLoadResult


FEM_PNG_KEYS = (
    "mesh_png",
    "displacement_magnitude_png",
    "von_mises_stress_png",
    "principal_stress_1_in_plane_png",
    "deformed_shape_png",
    "stress_contour_png",
    "strain_contour_png",
)

NATIVE_P1_CONTOUR_DEFINITIONS = {
    "stress_contour": {
        "filename": "stress_contour.png",
        "scalar": "von_mises_stress",
        "definition": (
            "sqrt(stress_xx^2 - stress_xx*stress_yy + stress_yy^2 + "
            "3*shear_stress_xy^2)"
        ),
        "source": "RecoveredFields.von_mises_stress",
        "location": "native P1 triangle element; one constant value per triangle",
        "display_unit": "MPa",
    },
    "strain_contour": {
        "filename": "strain_contour.png",
        "scalar": "maximum_in_plane_principal_infinitesimal_strain",
        "definition": (
            "(strain_xx + strain_yy)/2 + sqrt(((strain_xx - strain_yy)/2)^2 "
            "+ (engineering_shear_strain_xy/2)^2)"
        ),
        "source": "RecoveredFields infinitesimal strain tensor components",
        "location": "native P1 triangle element; one constant value per triangle",
        "display_unit": "microstrain",
    },
}

ACTIVE_CONTACT_FACET_CSV_FIELDS = (
    "facet_id",
    "node_start_id",
    "node_end_id",
    "region",
    "x_start_m",
    "y_start_m",
    "x_end_m",
    "y_end_m",
    "dx_projected_m",
    "ds_actual_m",
    "dx_over_ds",
    "qx_projected_N_per_m",
    "qy_projected_N_per_m",
    "tx_applied_per_ds_N_per_m",
    "ty_applied_per_ds_N_per_m",
    "integrated_Fx_N",
    "integrated_Fy_N",
)


@dataclass(frozen=True, slots=True)
class ActiveFacetInsetData:
    """Display-only values read from final loaded facets and solve-mesh nodes."""

    segments_m: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    node_coordinates_m: tuple[tuple[float, float], ...]
    tractions_N_per_m: tuple[tuple[float, float], ...]
    x_limits_m: tuple[float, float] | None
    y_limits_m: tuple[float, float] | None
    projected_length_sum_m: float
    actual_facet_length_sum_m: float
    integrated_Fx_N: float
    integrated_Fy_N: float


def maximum_in_plane_principal_infinitesimal_strain(
    fields: RecoveredFields,
) -> NDArray[np.float64]:
    """Derive the first in-plane principal small strain from recovered P1 fields."""

    exx = np.asarray(fields.strain_xx, dtype=float)
    eyy = np.asarray(fields.strain_yy, dtype=float)
    gamma_xy = np.asarray(fields.engineering_shear_strain_xy, dtype=float)
    expected = (fields.element_count,)
    for name, values in (("strain_xx", exx), ("strain_yy", eyy), ("engineering_shear_strain_xy", gamma_xy)):
        if values.shape != expected:
            raise ValueError(
                f"{name} shape {values.shape} does not match triangle count {fields.element_count}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")
    mean = 0.5 * (exx + eyy)
    radius = np.hypot(0.5 * (exx - eyy), 0.5 * gamma_xy)
    values = np.asarray(mean + radius, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("maximum in-plane principal infinitesimal strain is non-finite")
    return values


def _validate_native_p1_contour_data(
    fields: RecoveredFields,
    values: NDArray[np.float64],
    field_name: str,
) -> NDArray[np.float64]:
    coordinates = np.asarray(fields.node_coordinates, dtype=float)
    connectivity = np.asarray(fields.element_connectivity)
    cell_values = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] == 0:
        raise ValueError("contour node coordinates must have shape (node_count, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("contour node coordinates contain NaN or infinity")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3 or connectivity.shape[0] == 0:
        raise ValueError("contour connectivity must have shape (triangle_count, 3)")
    if not np.issubdtype(connectivity.dtype, np.integer):
        raise ValueError("contour connectivity must contain integer node IDs")
    if np.min(connectivity) < 0 or np.max(connectivity) >= coordinates.shape[0]:
        raise ValueError("contour connectivity contains an out-of-range node ID")
    if cell_values.shape != (connectivity.shape[0],):
        raise ValueError(
            f"{field_name} shape {cell_values.shape} does not match triangle count "
            f"{connectivity.shape[0]}"
        )
    if not np.all(np.isfinite(cell_values)):
        raise ValueError(f"{field_name} contains NaN or infinity")
    return cell_values


def _element_color_norm(values: NDArray[np.float64], *, center_zero: bool) -> object:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 1.0)
        return Normalize(vmin=minimum - padding, vmax=maximum + padding)
    if center_zero and minimum < 0.0 < maximum:
        return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return Normalize(vmin=minimum, vmax=maximum)


def _plot_native_p1_contour(
    fields: RecoveredFields,
    values: NDArray[np.float64],
    output_path: str | Path,
    *,
    field_name: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    center_zero: bool,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    cell_values = _validate_native_p1_contour_data(fields, values, field_name)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates_mm = 1.0e3 * np.asarray(fields.node_coordinates, dtype=float)
    triangulation = mtri.Triangulation(
        coordinates_mm[:, 0],
        coordinates_mm[:, 1],
        np.asarray(fields.element_connectivity, dtype=np.int32),
    )
    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    try:
        image = ax.tripcolor(
            triangulation,
            facecolors=cell_values,
            shading="flat",
            cmap=cmap,
            norm=_element_color_norm(cell_values, center_zero=center_zero),
        )
        ax.triplot(triangulation, color="#8A8A8A", alpha=0.55, linewidth=0.28)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Workpiece x [mm]")
        ax.set_ylabel("Workpiece y [mm]")
        ax.set_title(title, fontweight="bold")
        style_axis(ax)
        fig.colorbar(image, ax=ax, label=colorbar_label)
        add_scope_footer(
            fig,
            "Native P1 element-constant field on the original 2D small-deformation "
            "plane-stress mesh. No nodal averaging, smoothing, or true contact pressure.",
            y=0.025,
        )
        save_png(fig, path)
    finally:
        plt.close(fig)
    return path


def write_native_p1_contour_pngs(
    fields: RecoveredFields,
    output_paths: dict[str, Path],
) -> tuple[Path, Path]:
    """Write the two additional native P1 element-constant contour PNGs."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    stress_values_mpa = _validate_native_p1_contour_data(
        fields,
        1.0e-6 * np.asarray(fields.von_mises_stress, dtype=float),
        "von_mises_stress",
    )
    strain_values_microstrain = 1.0e6 * maximum_in_plane_principal_infinitesimal_strain(fields)
    with matplotlib.rc_context(plot_style()):
        stress_path = _plot_native_p1_contour(
            fields,
            stress_values_mpa,
            output_paths["stress_contour_png"],
            field_name="von_mises_stress",
            title="Native P1 element von Mises stress contour",
            colorbar_label="von Mises stress [MPa]",
            cmap="inferno",
            center_zero=False,
        )
        strain_path = _plot_native_p1_contour(
            fields,
            strain_values_microstrain,
            output_paths["strain_contour_png"],
            field_name="maximum_in_plane_principal_infinitesimal_strain",
            title="Native P1 element maximum in-plane principal infinitesimal strain contour",
            colorbar_label="maximum in-plane principal infinitesimal strain [\N{MICRO SIGN}\N{GREEK SMALL LETTER EPSILON}]",
            cmap="coolwarm",
            center_zero=True,
        )
    return stress_path, strain_path


def _active_facet_inset_data(load: EvolvedFacetLoadResult) -> ActiveFacetInsetData:
    """Collect plotting geometry without consulting theoretical surface regions."""

    if not load.facets:
        return ActiveFacetInsetData(
            segments_m=(),
            node_coordinates_m=(),
            tractions_N_per_m=(),
            x_limits_m=None,
            y_limits_m=None,
            projected_length_sum_m=0.0,
            actual_facet_length_sum_m=0.0,
            integrated_Fx_N=0.0,
            integrated_Fy_N=0.0,
        )

    final_nodes = load.mesh.skfem_mesh.p
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    nodes_by_id: dict[int, tuple[float, float]] = {}
    tractions: list[tuple[float, float]] = []
    endpoint_x: list[float] = []
    endpoint_y: list[float] = []
    local_edge_lengths: list[float] = []
    for facet in load.facets:
        start = (
            float(final_nodes[0, facet.node_start_id]),
            float(final_nodes[1, facet.node_start_id]),
        )
        end = (
            float(final_nodes[0, facet.node_end_id]),
            float(final_nodes[1, facet.node_end_id]),
        )
        nodes_by_id.setdefault(facet.node_start_id, start)
        nodes_by_id.setdefault(facet.node_end_id, end)
        segments.append((start, end))
        tractions.append(
            (
                facet.tx_applied_per_ds_N_per_m,
                facet.ty_applied_per_ds_N_per_m,
            )
        )
        endpoint_x.extend((facet.x_start_m, facet.x_end_m))
        endpoint_y.extend((facet.y_start_m, facet.y_end_m))
        local_edge_lengths.append(facet.ds_actual_m)

    representative_edge = float(np.median(local_edge_lengths))
    x_min = min(endpoint_x)
    x_max = max(endpoint_x)
    y_min = min(endpoint_y)
    y_max = max(endpoint_y)
    x_padding = 0.5 * representative_edge
    y_padding = max(0.15 * (y_max - y_min), 0.02 * representative_edge)
    return ActiveFacetInsetData(
        segments_m=tuple(segments),
        node_coordinates_m=tuple(nodes_by_id.values()),
        tractions_N_per_m=tuple(tractions),
        x_limits_m=(x_min - x_padding, x_max + x_padding),
        y_limits_m=(y_min - y_padding, y_max + y_padding),
        projected_length_sum_m=math.fsum(
            facet.dx_projected_m for facet in load.facets
        ),
        actual_facet_length_sum_m=math.fsum(facet.ds_actual_m for facet in load.facets),
        integrated_Fx_N=math.fsum(facet.integrated_Fx_N for facet in load.facets),
        integrated_Fy_N=math.fsum(facet.integrated_Fy_N for facet in load.facets),
    )


def _visual_arrow_delta(
    tx_N_per_m: float,
    ty_N_per_m: float,
    *,
    visual_length: float = 0.105,
) -> tuple[float, float]:
    """Scale an existing global traction vector for display without rotating it."""

    magnitude = math.hypot(tx_N_per_m, ty_N_per_m)
    if magnitude == 0.0:
        return 0.0, 0.0
    return (
        visual_length * tx_N_per_m / magnitude,
        visual_length * ty_N_per_m / magnitude,
    )


def _draw_active_facet_inset(
    figure: object,
    load: EvolvedFacetLoadResult,
    bounds: tuple[float, float, float, float],
) -> None:
    data = _active_facet_inset_data(load)
    inset = figure.add_axes(bounds, zorder=20)
    inset.set_facecolor((1.0, 1.0, 1.0, 0.965))
    if not data.segments_m:
        inset.set_axis_off()
        inset.text(
            0.5,
            0.84,
            "Active loaded mesh facets",
            ha="center",
            va="center",
            transform=inset.transAxes,
            fontsize=8.0,
            fontweight="bold",
        )
        inset.text(
            0.5,
            0.54,
            "No active loaded facets at this position.",
            ha="center",
            va="center",
            transform=inset.transAxes,
            fontsize=7.4,
            color="#404040",
            wrap=True,
        )
        return

    assert data.x_limits_m is not None
    assert data.y_limits_m is not None
    y_reference_m = data.y_limits_m[0]
    x_limits_mm = tuple(1.0e3 * value for value in data.x_limits_m)
    y_limits_um = tuple(1.0e6 * (value - y_reference_m) for value in data.y_limits_m)
    for start, end in data.segments_m:
        inset.plot(
            (1.0e3 * start[0], 1.0e3 * end[0]),
            (1.0e6 * (start[1] - y_reference_m), 1.0e6 * (end[1] - y_reference_m)),
            color=COLORS["contact"],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )
    node_x_mm = [1.0e3 * value[0] for value in data.node_coordinates_m]
    node_y_um = [1.0e6 * (value[1] - y_reference_m) for value in data.node_coordinates_m]
    inset.scatter(
        node_x_mm,
        node_y_um,
        s=12,
        facecolor="white",
        edgecolor=COLORS["reference"],
        linewidth=0.65,
        zorder=5,
    )
    inset.set_xlim(*x_limits_mm)
    inset.set_ylim(*y_limits_um)

    arrow_indices = np.linspace(
        0,
        len(data.segments_m) - 1,
        num=min(6, len(data.segments_m)),
        dtype=int,
    )
    x_span_mm = x_limits_mm[1] - x_limits_mm[0]
    y_span_um = y_limits_um[1] - y_limits_um[0]
    for index in np.unique(arrow_indices):
        start, end = data.segments_m[int(index)]
        midpoint_x_mm = 0.5e3 * (start[0] + end[0])
        midpoint_y_um = 0.5e6 * (start[1] + end[1] - 2.0 * y_reference_m)
        origin_x = (midpoint_x_mm - x_limits_mm[0]) / x_span_mm
        origin_y = (midpoint_y_um - y_limits_um[0]) / y_span_um
        arrow_dx, arrow_dy = _visual_arrow_delta(*data.tractions_N_per_m[int(index)])
        if arrow_dx == 0.0 and arrow_dy == 0.0:
            continue
        inset.annotate(
            "",
            xy=(origin_x + arrow_dx, origin_y + arrow_dy),
            xytext=(origin_x, origin_y),
            xycoords=inset.transAxes,
            textcoords=inset.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["reference"],
                "linewidth": 0.85,
                "mutation_scale": 7.0,
            },
            zorder=6,
        )

    inset.set_xlabel("x [mm]", fontsize=6.5, labelpad=1)
    inset.set_ylabel("local y [um]", fontsize=6.5, labelpad=1)
    inset.set_title("Active loaded mesh facets", fontsize=8.0, pad=2, fontweight="bold")
    inset.tick_params(axis="both", labelsize=6.0, length=2.0, pad=1)
    inset.grid(color="#DDDDDD", linewidth=0.45, alpha=0.65)
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.6)
    inset.text(
        0.98,
        0.98,
        f"active facets = {len(data.segments_m)}\n"
        f"sum(dx) = {1.0e3 * data.projected_length_sum_m:.4g} mm | "
        f"sum(ds) = {1.0e3 * data.actual_facet_length_sum_m:.4g} mm\n"
        f"integrated Fx = {data.integrated_Fx_N:.4g} N | "
        f"Fy = {data.integrated_Fy_N:.4g} N\n"
        "Load arrows: visual scale",
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color="#333333",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.0},
    )


def enhance_evolved_fem_pngs(
    load: EvolvedFacetLoadResult,
    output_paths: dict[str, Path],
) -> None:
    """Add a final-mesh active-facet side panel without altering the source image."""

    import matplotlib.image as matplotlib_image
    import matplotlib.pyplot as plt

    dpi = 300
    side_panel_px = 900
    for key in FEM_PNG_KEYS:
        path = Path(output_paths[key]).expanduser().resolve()
        pixels = matplotlib_image.imread(path)
        height_px, width_px = pixels.shape[:2]
        total_width_px = width_px + side_panel_px
        content_fraction = width_px / total_width_px
        side_fraction = side_panel_px / total_width_px
        staged = path.with_name(f".{path.name}.enhanced.tmp")
        with plt.rc_context(plot_style()):
            figure = plt.figure(
                figsize=(total_width_px / dpi, height_px / dpi),
                dpi=dpi,
                frameon=False,
            )
            try:
                background = figure.add_axes((0.0, 0.0, content_fraction, 1.0))
                background.imshow(pixels, interpolation="none", aspect="auto")
                background.set_axis_off()
                _draw_active_facet_inset(
                    figure,
                    load,
                    (
                        content_fraction + 0.10 * side_fraction,
                        0.22,
                        0.82 * side_fraction,
                        0.58,
                    ),
                )
                figure.savefig(staged, format="png", dpi=dpi, facecolor="white")
                os.replace(staged, path)
            finally:
                staged.unlink(missing_ok=True)
                plt.close(figure)

def write_active_contact_facets_csv(
    load: EvolvedFacetLoadResult,
    output_path: str | Path,
) -> Path:
    """Write the exact contact_arc facets and q dx to t ds audit values."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ACTIVE_CONTACT_FACET_CSV_FIELDS)
        writer.writeheader()
        for facet in load.facets:
            writer.writerow(facet.to_dict())
    return path


def write_mesh_png(
    load: EvolvedFacetLoadResult,
    output_path: str | Path,
) -> Path:
    """Plot the evolved mesh, top-surface roles, and global load direction."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = load.mesh.skfem_mesh
    coordinates_mm = 1.0e3 * mesh.p.T
    triangulation = mtri.Triangulation(
        coordinates_mm[:, 0], coordinates_mm[:, 1], mesh.t.T
    )
    with plt.rc_context(plot_style()):
        fig, ax = plt.subplots(figsize=(11.2, 6.1))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.77)
        try:
            ax.triplot(triangulation, color=COLORS["mesh"], linewidth=0.42)
            boundary_styles = {
                "fixed": (COLORS["fixed"], "Fixed support", 3.0),
                "free_left": (COLORS["free"], "Free sides", 2.2),
                "free_right": (COLORS["free"], None, 2.2),
            }
            for name, (color, label, linewidth) in boundary_styles.items():
                facet_ids = mesh.boundaries[name]
                for index, nodes in enumerate(mesh.facets[:, facet_ids].T):
                    ax.plot(
                        coordinates_mm[nodes, 0],
                        coordinates_mm[nodes, 1],
                        color=color,
                        linewidth=linewidth,
                        label=label if index == 0 else None,
                    )
            region_styles = {
                "ground": (COLORS["ground"], "Ground - unloaded", 3.2),
                "contact_arc": (COLORS["contact"], "Contact arc - active load", 5.0),
                "unprocessed": (COLORS["unprocessed_line"], "Unprocessed - unloaded", 3.2),
            }
            region_seen: set[str] = set()
            for facet in load.mesh.top_boundary_facets:
                color, label, linewidth = region_styles[facet.region]
                ax.plot(
                    [1.0e3 * facet.x_start_m, 1.0e3 * facet.x_end_m],
                    [1.0e3 * facet.y_start_m, 1.0e3 * facet.y_end_m],
                    color=color,
                    linewidth=linewidth,
                    solid_capstyle="round",
                    label=label if facet.region not in region_seen else None,
                )
                region_seen.add(facet.region)

            fx = load.pass_load.current_Fx_N
            fy = load.pass_load.current_Fy_N
            if load.facets and (fx != 0.0 or fy != 0.0):
                midpoint = load.facets[len(load.facets) // 2]
                x_mid = 0.5e3 * (midpoint.x_start_m + midpoint.x_end_m)
                y_mid = 0.5e3 * (midpoint.y_start_m + midpoint.y_end_m)
                magnitude = math.hypot(fx, fy)
                reference = 0.13 * max(
                    load.mesh.case.trajectory.workpiece.length_m,
                    load.mesh.case.trajectory.workpiece.original_surface_height_m,
                ) * 1.0e3
                x_tip = x_mid + reference * fx / magnitude
                y_tip = y_mid + reference * fy / magnitude
                ax.annotate(
                    "",
                    xy=(x_tip, y_tip),
                    xytext=(x_mid, y_mid),
                    arrowprops={"arrowstyle": "-|>", "color": COLORS["reference"], "lw": 2.2},
                )
                ax.text(
                    x_tip,
                    y_tip,
                    f"  global load\n  Fx = {fx:.4g} N\n  downward compression = {-fy:.4g} N",
                    va="top" if fy < 0.0 else "bottom",
                    fontsize=8.5,
                    bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
                )
            else:
                ax.text(
                    0.02,
                    0.94,
                    "Zero current contact load",
                    transform=ax.transAxes,
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "0.75"},
                )
            x_min = float(np.min(coordinates_mm[:, 0]))
            x_max = float(np.max(coordinates_mm[:, 0]))
            y_min = float(np.min(coordinates_mm[:, 1]))
            y_max = float(np.max(coordinates_mm[:, 1]))
            x_pad = 0.05 * max(x_max - x_min, 1.0)
            y_pad = 0.18 * max(y_max - y_min, 1.0)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Workpiece x [mm]")
            ax.set_ylabel("Workpiece y [mm]")
            style_axis(ax)
            unique_legend(ax, loc="upper center", ncol=3, fontsize=8.0)
            ax.set_title(
        PLOT_TITLES_EN["evolved_fem_mesh"],
                fontweight="bold",
                fontsize=14,
            )
            fig.text(
                0.5,
                0.845,
                f"state: {load.pass_load.pass_state} | active load exists only on runtime contact-arc facets | q dx = t ds",
                ha="center",
                color="#4D4D4D",
                fontsize=9.5,
            )
            ax.text(
                0.98,
                0.05,
                f"{load.mesh.node_count} nodes | {load.mesh.triangle_count} triangles | {len(load.facets)} active facets",
                transform=ax.transAxes,
                ha="right",
                fontsize=8.5,
                color="#555555",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            add_scope_footer(
                fig,
                "Single-pass independent quasi-static 2D plane-stress snapshot with one-way empirical coupling. No true contact pressure, state accumulation, or roughness prediction.",
                y=0.055,
            )
            save_png(fig, path)
        finally:
            plt.close(fig)
    return path
