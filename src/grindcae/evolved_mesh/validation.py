"""Independent meshio and scikit-fem validation for evolved surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import meshio
import numpy as np
from numpy.typing import NDArray
from skfem import MeshTri
from skfem.io.meshio import from_meshio

from grindcae.mesh_contract import PHYSICAL_GROUP_NAMES
from grindcae.trajectory import REGIONS, TrajectoryResult

from .generator import (
    EvolvedMeshGenerationSummary,
    MINIMUM_ARC_FACET_COUNT,
)
from .models import EvolvedMeshCase


EXPECTED_GROUP_DIMENSIONS = {
    "domain": 2,
    "fixed": 1,
    "contact": 1,
    "free_left": 1,
    "free_right": 1,
}


class EvolvedMeshValidationRuntimeError(RuntimeError):
    """Raised when a generated evolved-surface mesh fails readback checks."""


@dataclass(frozen=True, slots=True)
class TopBoundaryFacet:
    """One oriented top facet, ordered from physical left to right."""

    facet_id: int
    node_start_id: int
    node_end_id: int
    region: str
    x_start_m: float
    y_start_m: float
    x_end_m: float
    y_end_m: float
    projected_length_x_m: float
    actual_facet_length_m: float
    tangent_x: float
    tangent_y: float
    outward_normal_x: float
    outward_normal_y: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "facet_id": self.facet_id,
            "node_start_id": self.node_start_id,
            "node_end_id": self.node_end_id,
            "region": self.region,
            "x_start_m": self.x_start_m,
            "y_start_m": self.y_start_m,
            "x_end_m": self.x_end_m,
            "y_end_m": self.y_end_m,
            "projected_length_x_m": self.projected_length_x_m,
            "actual_facet_length_m": self.actual_facet_length_m,
            "tangent_x": self.tangent_x,
            "tangent_y": self.tangent_y,
            "outward_normal_x": self.outward_normal_x,
            "outward_normal_y": self.outward_normal_y,
        }


@dataclass(frozen=True, slots=True)
class EvolvedMeshValidationResult:
    """Validated imported mesh, exact geometry measures, and boundary audit."""

    case: EvolvedMeshCase
    trajectory: TrajectoryResult
    generation: EvolvedMeshGenerationSummary
    meshio_mesh: meshio.Mesh
    skfem_mesh: MeshTri
    node_count: int
    triangle_count: int
    actual_vector_p1_degrees_of_freedom: int
    top_boundary_facets: tuple[TopBoundaryFacet, ...]
    exact_arc_curve_length_m: float
    discrete_arc_facet_length_m: float
    complete_top_projected_length_m: float
    exact_complete_top_curve_length_m: float
    discrete_complete_top_facet_length_m: float
    original_rectangle_area_m2: float
    exact_evolved_domain_area_m2: float
    exact_removed_cross_section_area_m2: float
    discrete_triangle_area_m2: float
    discrete_boundary_polygon_area_m2: float
    curvature_discretization_area_error_m2: float
    boundary_projected_lengths_m: dict[str, float]
    boundary_actual_lengths_m: dict[str, float]
    arc_facet_count: int
    meshio_read_validated: bool
    scikit_fem_import_validated: bool


def _tolerance(*values: float) -> float:
    scale = max(*(abs(value) for value in values), 1.0e-300)
    return max(512.0 * math.ulp(scale), 2.0e-11 * scale, 1.0e-14)


def _exact_arc_length(trajectory: TrajectoryResult) -> float:
    interval = trajectory.effective_arc_interval_m
    if interval is None:
        return 0.0
    center_x, _ = trajectory.wheel_center_coordinates_m
    radius = trajectory.wheel_radius_m
    angles = [
        math.asin(max(-1.0, min(1.0, (x_m - center_x) / radius)))
        for x_m in interval
    ]
    return radius * abs(angles[1] - angles[0])


def _circle_integral(trajectory: TrajectoryResult, x0: float, x1: float) -> float:
    center_x, center_y = trajectory.wheel_center_coordinates_m
    radius = trajectory.wheel_radius_m

    def primitive(x_m: float) -> float:
        offset = x_m - center_x
        root = math.sqrt(max(radius - offset, 0.0)) * math.sqrt(
            max(radius + offset, 0.0)
        )
        return center_y * x_m - 0.5 * (
            offset * root + radius * radius * math.asin(offset / radius)
        )

    value = primitive(x1) - primitive(x0)
    if not math.isfinite(value):
        raise EvolvedMeshValidationRuntimeError(
            "exact circle integration produced a non-finite area"
        )
    return value


def _exact_domain_area(trajectory: TrajectoryResult) -> float:
    height = trajectory.case.workpiece.original_surface_height_m
    ground = trajectory.ground_surface_height_m
    contributions: list[float] = []
    for segment in trajectory.surface_segments:
        if segment.region == "ground":
            contributions.append(ground * segment.length_m)
        elif segment.region == "unprocessed":
            contributions.append(height * segment.length_m)
        elif segment.region == "contact_arc":
            contributions.append(
                _circle_integral(trajectory, segment.x_start_m, segment.x_end_m)
            )
        else:
            raise EvolvedMeshValidationRuntimeError("unknown phase 4C1 region")
    area = math.fsum(contributions)
    if not math.isfinite(area) or area <= 0.0:
        raise EvolvedMeshValidationRuntimeError("exact evolved area is invalid")
    return area


def _cell_blocks(
    mesh_data: meshio.Mesh,
) -> list[tuple[str, NDArray[np.int64], NDArray[np.int64]]]:
    physical = mesh_data.cell_data.get("gmsh:physical")
    if physical is None or len(physical) != len(mesh_data.cells):
        raise EvolvedMeshValidationRuntimeError(
            "MSH cell blocks do not contain complete gmsh:physical data"
        )
    blocks: list[tuple[str, NDArray[np.int64], NDArray[np.int64]]] = []
    for block, tags in zip(mesh_data.cells, physical):
        cells = np.asarray(block.data, dtype=np.int64)
        physical_tags = np.asarray(tags, dtype=np.int64)
        if cells.shape[0] != physical_tags.shape[0]:
            raise EvolvedMeshValidationRuntimeError(
                "MSH physical tags do not match their cell block"
            )
        blocks.append((block.type, cells, physical_tags))
    return blocks


def _group_tags(mesh_data: meshio.Mesh) -> dict[str, int]:
    names = set(mesh_data.field_data)
    if names != set(PHYSICAL_GROUP_NAMES):
        raise EvolvedMeshValidationRuntimeError(
            f"physical groups must be exactly {PHYSICAL_GROUP_NAMES}; got {sorted(names)}"
        )
    tags: dict[str, int] = {}
    for name, expected_dimension in EXPECTED_GROUP_DIMENSIONS.items():
        value = np.asarray(mesh_data.field_data[name], dtype=np.int64)
        if value.shape != (2,) or int(value[1]) != expected_dimension:
            raise EvolvedMeshValidationRuntimeError(
                f"physical group {name} has the wrong dimension"
            )
        tags[name] = int(value[0])
    return tags


def _boundary_edges(
    blocks: list[tuple[str, NDArray[np.int64], NDArray[np.int64]]],
    tags: dict[str, int],
) -> dict[str, list[tuple[int, int]]]:
    grouped = {name: [] for name in EXPECTED_GROUP_DIMENSIONS if name != "domain"}
    for cell_type, cells, physical_tags in blocks:
        if cell_type == "line":
            if cells.ndim != 2 or cells.shape[1] != 2:
                raise EvolvedMeshValidationRuntimeError(
                    "only first-order two-node line cells are supported"
                )
            for nodes, physical_tag in zip(cells, physical_tags):
                matches = [name for name, tag in tags.items() if tag == physical_tag]
                if len(matches) != 1 or matches[0] == "domain":
                    raise EvolvedMeshValidationRuntimeError(
                        "a boundary line has an invalid physical group"
                    )
                grouped[matches[0]].append((int(nodes[0]), int(nodes[1])))
        elif cell_type == "triangle":
            if cells.ndim != 2 or cells.shape[1] != 3:
                raise EvolvedMeshValidationRuntimeError(
                    "only first-order three-node triangle cells are supported"
                )
        else:
            raise EvolvedMeshValidationRuntimeError(
                f"unsupported MSH cell type: {cell_type}"
            )
    if any(not edges for edges in grouped.values()):
        raise EvolvedMeshValidationRuntimeError("one or more boundary groups are empty")
    edge_sets = {
        name: {tuple(sorted(edge)) for edge in edges}
        for name, edges in grouped.items()
    }
    names = list(edge_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if edge_sets[first] & edge_sets[second]:
                raise EvolvedMeshValidationRuntimeError(
                    f"boundary groups {first} and {second} overlap"
                )
    return grouped


def _triangles(
    blocks: list[tuple[str, NDArray[np.int64], NDArray[np.int64]]],
    domain_tag: int,
) -> NDArray[np.int64]:
    triangle_blocks: list[NDArray[np.int64]] = []
    for cell_type, cells, physical_tags in blocks:
        if cell_type != "triangle":
            continue
        if np.any(physical_tags != domain_tag):
            raise EvolvedMeshValidationRuntimeError(
                "domain must cover every triangle exactly once"
            )
        triangle_blocks.append(cells)
    if not triangle_blocks:
        raise EvolvedMeshValidationRuntimeError("MSH has no triangles")
    triangles = np.vstack(triangle_blocks)
    if np.unique(np.sort(triangles, axis=1), axis=0).shape[0] != triangles.shape[0]:
        raise EvolvedMeshValidationRuntimeError("domain contains duplicate triangles")
    return triangles


def _require_complete_boundary_coverage(
    triangles: NDArray[np.int64],
    edges: dict[str, list[tuple[int, int]]],
) -> None:
    counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            key = tuple(sorted((first, second)))
            counts[key] = counts.get(key, 0) + 1
    triangle_boundary = {edge for edge, count in counts.items() if count == 1}
    if any(count not in (1, 2) for count in counts.values()):
        raise EvolvedMeshValidationRuntimeError(
            "triangle topology contains a non-manifold edge"
        )
    physical_boundary = {
        tuple(sorted(edge)) for group_edges in edges.values() for edge in group_edges
    }
    if physical_boundary != triangle_boundary:
        raise EvolvedMeshValidationRuntimeError(
            "the four physical boundaries do not exactly cover the exterior facets"
        )


def _triangle_area(
    points: NDArray[np.float64], triangles: NDArray[np.int64]
) -> tuple[float, NDArray[np.float64]]:
    xy = points[:, :2]
    vertices = xy[triangles]
    signed = 0.5 * (
        (vertices[:, 1, 0] - vertices[:, 0, 0])
        * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 2, 0] - vertices[:, 0, 0])
        * (vertices[:, 1, 1] - vertices[:, 0, 1])
    )
    if np.any(signed <= 0.0) or not np.all(np.isfinite(signed)):
        raise EvolvedMeshValidationRuntimeError(
            "all imported triangles must have finite positive signed area"
        )
    return float(math.fsum(float(value) for value in signed)), signed


def _segment_for_midpoint(trajectory: TrajectoryResult, x_m: float) -> str:
    tolerance = _tolerance(trajectory.case.workpiece.length_m)
    matches = [
        segment.region
        for segment in trajectory.surface_segments
        if segment.x_start_m - tolerance <= x_m <= segment.x_end_m + tolerance
    ]
    if len(matches) != 1:
        raise EvolvedMeshValidationRuntimeError(
            "a top facet midpoint does not map to exactly one phase 4C1 region"
        )
    return matches[0]


def _expected_surface_y(
    trajectory: TrajectoryResult, region: str, x_m: float
) -> float:
    if region == "ground":
        return trajectory.ground_surface_height_m
    if region == "unprocessed":
        return trajectory.case.workpiece.original_surface_height_m
    if region != "contact_arc":
        raise EvolvedMeshValidationRuntimeError("top facet has an invalid region")
    center_x, center_y = trajectory.wheel_center_coordinates_m
    radius = trajectory.wheel_radius_m
    offset = x_m - center_x
    root = math.sqrt(max(radius - offset, 0.0)) * math.sqrt(
        max(radius + offset, 0.0)
    )
    return center_y - root


def _top_facets(
    points: NDArray[np.float64],
    contact_edges: list[tuple[int, int]],
    skfem_mesh: MeshTri,
    trajectory: TrajectoryResult,
) -> tuple[TopBoundaryFacet, ...]:
    facet_lookup = {
        tuple(sorted((int(nodes[0]), int(nodes[1])))): facet_id
        for facet_id, nodes in enumerate(skfem_mesh.facets.T)
    }
    raw: list[TopBoundaryFacet] = []
    tolerance = _tolerance(
        trajectory.case.workpiece.length_m,
        trajectory.case.workpiece.original_surface_height_m,
    )
    for first, second in contact_edges:
        start_id, end_id = sorted((first, second), key=lambda node: points[node, 0])
        x0, y0 = (float(points[start_id, 0]), float(points[start_id, 1]))
        x1, y1 = (float(points[end_id, 0]), float(points[end_id, 1]))
        dx = x1 - x0
        dy = y1 - y0
        if dx <= 0.0:
            raise EvolvedMeshValidationRuntimeError(
                "top facets must have strictly positive physical x projection"
            )
        length = math.hypot(dx, dy)
        midpoint = 0.5 * (x0 + x1)
        region = _segment_for_midpoint(trajectory, midpoint)
        for x_m, y_m in ((x0, y0), (x1, y1)):
            expected_y = _expected_surface_y(trajectory, region, x_m)
            if not math.isclose(y_m, expected_y, rel_tol=0.0, abs_tol=tolerance):
                raise EvolvedMeshValidationRuntimeError(
                    f"contact node ({x_m}, {y_m}) violates the {region} equation"
                )
        tangent_x = dx / length
        tangent_y = dy / length
        key = tuple(sorted((first, second)))
        if key not in facet_lookup:
            raise EvolvedMeshValidationRuntimeError(
                "meshio contact edge is missing from the scikit-fem facet topology"
            )
        raw.append(
            TopBoundaryFacet(
                facet_id=facet_lookup[key],
                node_start_id=start_id,
                node_end_id=end_id,
                region=region,
                x_start_m=x0,
                y_start_m=y0,
                x_end_m=x1,
                y_end_m=y1,
                projected_length_x_m=dx,
                actual_facet_length_m=length,
                tangent_x=tangent_x,
                tangent_y=tangent_y,
                outward_normal_x=-tangent_y,
                outward_normal_y=tangent_x,
            )
        )
    facets = tuple(sorted(raw, key=lambda facet: facet.x_start_m))
    width = trajectory.case.workpiece.length_m
    if not math.isclose(facets[0].x_start_m, 0.0, rel_tol=0.0, abs_tol=tolerance):
        raise EvolvedMeshValidationRuntimeError("contact does not start at x=0")
    if not math.isclose(facets[-1].x_end_m, width, rel_tol=0.0, abs_tol=tolerance):
        raise EvolvedMeshValidationRuntimeError("contact does not end at x=W")
    for left, right in zip(facets, facets[1:]):
        if left.node_end_id != right.node_start_id or not math.isclose(
            left.x_end_m, right.x_start_m, rel_tol=0.0, abs_tol=tolerance
        ):
            raise EvolvedMeshValidationRuntimeError(
                "contact facets are not continuous in physical x order"
            )
    return facets


def _edge_lengths(
    points: NDArray[np.float64], edges: list[tuple[int, int]]
) -> tuple[float, float]:
    projected = math.fsum(abs(float(points[b, 0] - points[a, 0])) for a, b in edges)
    actual = math.fsum(
        math.hypot(
            float(points[b, 0] - points[a, 0]),
            float(points[b, 1] - points[a, 1]),
        )
        for a, b in edges
    )
    return projected, actual


def validate_evolved_surface_mesh(
    mesh_path: str | Path,
    case: EvolvedMeshCase,
    trajectory: TrajectoryResult,
    generation: EvolvedMeshGenerationSummary,
) -> EvolvedMeshValidationResult:
    """Read and validate the same generated MSH through both import libraries."""

    path = Path(mesh_path).expanduser().resolve()
    if generation.output_path != path:
        raise EvolvedMeshValidationRuntimeError(
            "generation summary does not refer to the mesh being validated"
        )
    mesh_data = meshio.read(path)
    points = np.asarray(mesh_data.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise EvolvedMeshValidationRuntimeError("MSH points must be finite xyz values")
    z_tolerance = _tolerance(
        case.trajectory.workpiece.length_m,
        case.trajectory.workpiece.original_surface_height_m,
    )
    if np.any(np.abs(points[:, 2]) > z_tolerance):
        raise EvolvedMeshValidationRuntimeError("mesh is not confined to the xy plane")

    tags = _group_tags(mesh_data)
    blocks = _cell_blocks(mesh_data)
    edges = _boundary_edges(blocks, tags)
    triangles = _triangles(blocks, tags["domain"])
    _require_complete_boundary_coverage(triangles, edges)
    discrete_triangle_area, _ = _triangle_area(points, triangles)

    imported = from_meshio(mesh_data)
    if not isinstance(imported, MeshTri):
        raise EvolvedMeshValidationRuntimeError(
            "scikit-fem did not import the MSH as a triangular 2D mesh"
        )
    if imported.p.shape[1] != points.shape[0] or imported.t.shape[1] != triangles.shape[0]:
        raise EvolvedMeshValidationRuntimeError(
            "scikit-fem import changed the node or triangle count"
        )
    node_count = int(points.shape[0])
    triangle_count = int(triangles.shape[0])
    actual_dofs = 2 * int(imported.nvertices)
    if generation.raw_gmsh_node_count < node_count:
        raise EvolvedMeshValidationRuntimeError(
            "final MSH contains more nodes than the raw Gmsh model"
        )
    if generation.generated_element_node_count != node_count:
        raise EvolvedMeshValidationRuntimeError(
            "final MSH node count differs from the generated triangle-node count"
        )
    if generation.raw_gmsh_triangle_count != triangle_count:
        raise EvolvedMeshValidationRuntimeError(
            "final MSH triangle count differs from the generated Gmsh mesh"
        )
    if actual_dofs > generation.maximum_vector_p1_degrees_of_freedom:
        raise EvolvedMeshValidationRuntimeError(
            "final solvable mesh has "
            f"{actual_dofs} vector-P1 degrees of freedom, exceeding the safety "
            f"limit of {generation.maximum_vector_p1_degrees_of_freedom}"
        )
    if set(imported.boundaries) != {"fixed", "contact", "free_left", "free_right"}:
        raise EvolvedMeshValidationRuntimeError(
            "scikit-fem boundary names do not match the four physical boundaries"
        )
    if "domain" not in imported.subdomains:
        raise EvolvedMeshValidationRuntimeError("scikit-fem domain subdomain is missing")

    width = case.trajectory.workpiece.length_m
    tolerance = _tolerance(width, case.trajectory.workpiece.original_surface_height_m)
    for node_a, node_b in edges["fixed"]:
        if abs(points[node_a, 1]) > tolerance or abs(points[node_b, 1]) > tolerance:
            raise EvolvedMeshValidationRuntimeError("fixed boundary is not at y=0")
    for node_a, node_b in edges["free_left"]:
        if abs(points[node_a, 0]) > tolerance or abs(points[node_b, 0]) > tolerance:
            raise EvolvedMeshValidationRuntimeError("free_left is not at x=0")
    for node_a, node_b in edges["free_right"]:
        if (
            abs(points[node_a, 0] - width) > tolerance
            or abs(points[node_b, 0] - width) > tolerance
        ):
            raise EvolvedMeshValidationRuntimeError("free_right is not at x=W")

    top_facets = _top_facets(points, edges["contact"], imported, trajectory)
    complete_projected = math.fsum(
        facet.projected_length_x_m for facet in top_facets
    )
    if not math.isclose(complete_projected, width, rel_tol=1.0e-11, abs_tol=tolerance):
        raise EvolvedMeshValidationRuntimeError(
            "contact facet x projections do not cover exactly [0, W]"
        )

    arc_facets = tuple(facet for facet in top_facets if facet.region == "contact_arc")
    interval = trajectory.effective_arc_interval_m
    if interval is None:
        if arc_facets:
            raise EvolvedMeshValidationRuntimeError(
                "zero contact must not generate contact_arc facets"
            )
    else:
        if len(arc_facets) < MINIMUM_ARC_FACET_COUNT:
            raise EvolvedMeshValidationRuntimeError(
                "a non-empty circle arc requires at least four facets"
            )
        if not math.isclose(
            arc_facets[0].x_start_m, interval[0], rel_tol=0.0, abs_tol=tolerance
        ) or not math.isclose(
            arc_facets[-1].x_end_m, interval[1], rel_tol=0.0, abs_tol=tolerance
        ):
            raise EvolvedMeshValidationRuntimeError(
                "circle-arc facet endpoints do not match phase 4C1"
            )
        if any(
            left.node_end_id != right.node_start_id
            for left, right in zip(arc_facets, arc_facets[1:])
        ):
            raise EvolvedMeshValidationRuntimeError("circle-arc facets are not continuous")

    exact_arc_length = _exact_arc_length(trajectory)
    discrete_arc_length = math.fsum(
        facet.actual_facet_length_m for facet in arc_facets
    )
    if discrete_arc_length > exact_arc_length + tolerance:
        raise EvolvedMeshValidationRuntimeError(
            "circle-arc chord length exceeds the exact arc length"
        )
    exact_top_length = (
        trajectory.region_length_m("ground")
        + trajectory.region_length_m("unprocessed")
        + exact_arc_length
    )
    discrete_top_length = math.fsum(
        facet.actual_facet_length_m for facet in top_facets
    )

    exact_area = _exact_domain_area(trajectory)
    original_area = width * case.trajectory.workpiece.original_surface_height_m
    removed_area = original_area - exact_area
    polygon_area = math.fsum(
        0.5
        * (facet.y_start_m + facet.y_end_m)
        * facet.projected_length_x_m
        for facet in top_facets
    )
    area_tolerance = max(_tolerance(original_area), original_area * 2.0e-10)
    if not math.isclose(
        discrete_triangle_area, polygon_area, rel_tol=2.0e-10, abs_tol=area_tolerance
    ):
        raise EvolvedMeshValidationRuntimeError(
            "triangle area does not match the discrete boundary polygon area"
        )

    projected_lengths: dict[str, float] = {}
    actual_lengths: dict[str, float] = {}
    for name, group_edges in edges.items():
        projected_lengths[name], actual_lengths[name] = _edge_lengths(
            points, group_edges
        )

    return EvolvedMeshValidationResult(
        case=case,
        trajectory=trajectory,
        generation=generation,
        meshio_mesh=mesh_data,
        skfem_mesh=imported,
        node_count=node_count,
        triangle_count=triangle_count,
        actual_vector_p1_degrees_of_freedom=actual_dofs,
        top_boundary_facets=top_facets,
        exact_arc_curve_length_m=exact_arc_length,
        discrete_arc_facet_length_m=discrete_arc_length,
        complete_top_projected_length_m=complete_projected,
        exact_complete_top_curve_length_m=exact_top_length,
        discrete_complete_top_facet_length_m=discrete_top_length,
        original_rectangle_area_m2=original_area,
        exact_evolved_domain_area_m2=exact_area,
        exact_removed_cross_section_area_m2=removed_area,
        discrete_triangle_area_m2=discrete_triangle_area,
        discrete_boundary_polygon_area_m2=polygon_area,
        curvature_discretization_area_error_m2=discrete_triangle_area - exact_area,
        boundary_projected_lengths_m=projected_lengths,
        boundary_actual_lengths_m=actual_lengths,
        arc_facet_count=len(arc_facets),
        meshio_read_validated=True,
        scikit_fem_import_validated=True,
    )
