"""Build the phase 4C3A OCC surface and its first-order Gmsh mesh."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import gmsh

from grindcae.gmsh_runtime import initialize_gmsh
from grindcae.mesh_contract import (
    DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM,
    PHYSICAL_GROUP_NAMES,
)
from grindcae.trajectory import TrajectoryResult

from .models import EvolvedMeshCase


MINIMUM_ARC_FACET_COUNT = 4


class EvolvedMeshGenerationError(RuntimeError):
    """Raised when the exact evolved geometry cannot be meshed safely."""


@dataclass(frozen=True, slots=True)
class EvolvedMeshGenerationSummary:
    """Gmsh generation facts retained for independent import validation."""

    output_path: Path
    raw_gmsh_node_count: int
    raw_gmsh_triangle_count: int
    generated_element_node_count: int
    estimated_vector_p1_degrees_of_freedom: int
    maximum_vector_p1_degrees_of_freedom: int
    physical_group_names: tuple[str, ...]
    top_curve_count: int
    arc_curve_tag: int | None
    arc_curve_type: str | None
    requested_arc_facet_count: int
    generated_arc_facet_count: int


def _finite_estimate(value: float, message: str) -> int:
    if not math.isfinite(value):
        raise EvolvedMeshGenerationError(message)
    return max(1, math.ceil(value))


def estimate_evolved_mesh_degrees_of_freedom(
    case: EvolvedMeshCase,
    trajectory: TrajectoryResult,
) -> int:
    """Conservatively estimate vector-P1 size before initializing Gmsh."""

    if not isinstance(case, EvolvedMeshCase):
        raise TypeError("case must be an EvolvedMeshCase")
    if not isinstance(trajectory, TrajectoryResult):
        raise TypeError("trajectory must be a TrajectoryResult")
    if trajectory.case != case.trajectory:
        raise EvolvedMeshGenerationError("trajectory provenance does not match case")
    width = case.trajectory.workpiece.length_m
    height = case.trajectory.workpiece.original_surface_height_m
    size = case.mesh.target_size
    try:
        width_ratio = width / size
        height_ratio = height / size
        base = 4.0 * width_ratio * height_ratio + 8.0 * (
            width_ratio + height_ratio
        ) + 32.0
        arc_target = (
            0
            if trajectory.effective_arc_interval_m is None
            else max(
                MINIMUM_ARC_FACET_COUNT,
                math.ceil(_exact_arc_curve_length(trajectory) / size),
            )
        )
        estimate = base + 4.0 * arc_target
    except OverflowError as exc:
        raise EvolvedMeshGenerationError(
            "mesh.target_size produces an overflowing pre-mesh estimate"
        ) from exc
    return _finite_estimate(
        estimate, "mesh.target_size produces a non-finite pre-mesh estimate"
    )


def _exact_arc_curve_length(trajectory: TrajectoryResult) -> float:
    interval = trajectory.effective_arc_interval_m
    if interval is None:
        return 0.0
    x0, x1 = interval
    center_x, _ = trajectory.wheel_center_coordinates_m
    radius = trajectory.wheel_radius_m
    theta0 = math.asin(max(-1.0, min(1.0, (x0 - center_x) / radius)))
    theta1 = math.asin(max(-1.0, min(1.0, (x1 - center_x) / radius)))
    length = radius * abs(theta1 - theta0)
    if not math.isfinite(length) or length <= 0.0:
        raise EvolvedMeshGenerationError("effective circle arc has invalid exact length")
    return length


def _point_y(trajectory: TrajectoryResult, x_m: float) -> float:
    matches = [point.surface_y_m for point in trajectory.sample_points if point.x_m == x_m]
    if len(matches) != 1:
        raise EvolvedMeshGenerationError(
            "phase 4C1 did not retain one exact sample at a surface boundary"
        )
    return matches[0]


def _add_physical_group(dimension: int, tags: list[int], name: str) -> None:
    if not tags:
        raise EvolvedMeshGenerationError(f"physical group {name} is empty")
    physical_tag = gmsh.model.addPhysicalGroup(dimension, tags)
    gmsh.model.setPhysicalName(dimension, physical_tag, name)


def _count_first_order_elements(dimension: int, entity_tag: int | None = None) -> int:
    if entity_tag is None:
        element_types, element_blocks, _ = gmsh.model.mesh.getElements(dimension)
    else:
        element_types, element_blocks, _ = gmsh.model.mesh.getElements(
            dimension, entity_tag
        )
    count = 0
    for element_type, element_tags in zip(element_types, element_blocks):
        name, element_dimension, order, node_count = gmsh.model.mesh.getElementProperties(
            element_type
        )[:4]
        primary_count = gmsh.model.mesh.getElementProperties(element_type)[5]
        expected_nodes = 2 if dimension == 1 else 3
        expected_name = "Line" if dimension == 1 else "Triangle"
        if (
            element_dimension != dimension
            or not name.startswith(expected_name)
            or order != 1
            or node_count != expected_nodes
            or primary_count != expected_nodes
        ):
            raise EvolvedMeshGenerationError(
                "Gmsh generated elements outside the first-order line/triangle contract"
            )
        count += len(element_tags)
    return count


def _unique_first_order_triangle_node_count() -> int:
    element_types, _, node_blocks = gmsh.model.mesh.getElements(2)
    node_tags: set[int] = set()
    for element_type, element_nodes in zip(element_types, node_blocks):
        name, dimension, order, node_count = gmsh.model.mesh.getElementProperties(
            element_type
        )[:4]
        primary_count = gmsh.model.mesh.getElementProperties(element_type)[5]
        if (
            dimension != 2
            or not name.startswith("Triangle")
            or order != 1
            or node_count != 3
            or primary_count != 3
        ):
            raise EvolvedMeshGenerationError(
                "Gmsh generated elements outside the first-order triangle contract"
            )
        node_tags.update(int(tag) for tag in element_nodes)
    if not node_tags:
        raise EvolvedMeshGenerationError("Gmsh generated no triangle nodes")
    return len(node_tags)


def _validate_limit(maximum: int) -> None:
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("maximum_degrees_of_freedom must be a positive integer")


def generate_evolved_surface_mesh(
    case: EvolvedMeshCase,
    trajectory: TrajectoryResult,
    output_path: str | Path,
    *,
    maximum_degrees_of_freedom: int = DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM,
) -> EvolvedMeshGenerationSummary:
    """Generate one real OCC surface and one Gmsh 4.1 ASCII mesh."""

    if not isinstance(case, EvolvedMeshCase):
        raise TypeError("case must be an EvolvedMeshCase")
    if not isinstance(trajectory, TrajectoryResult):
        raise TypeError("trajectory must be a TrajectoryResult")
    if trajectory.case != case.trajectory:
        raise EvolvedMeshGenerationError("trajectory provenance does not match case")
    if gmsh.isInitialized():
        raise EvolvedMeshGenerationError("Gmsh is already initialized")
    _validate_limit(maximum_degrees_of_freedom)
    estimate = estimate_evolved_mesh_degrees_of_freedom(case, trajectory)
    if estimate > maximum_degrees_of_freedom:
        raise EvolvedMeshGenerationError(
            "conservative pre-mesh estimate requires approximately "
            f"{estimate} vector-P1 degrees of freedom, exceeding the configured "
            f"safety limit of {maximum_degrees_of_freedom}; increase mesh.target_size"
        )

    mesh_path = Path(output_path).expanduser().resolve()
    if mesh_path.suffix.lower() != ".msh":
        raise EvolvedMeshGenerationError("output_path must use the .msh suffix")
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    width = case.trajectory.workpiece.length_m
    size = case.mesh.target_size
    exact_arc_length = _exact_arc_curve_length(trajectory)
    requested_arc_facets = (
        0
        if exact_arc_length == 0.0
        else max(MINIMUM_ARC_FACET_COUNT, math.ceil(exact_arc_length / size))
    )
    arc_curve_tag: int | None = None
    arc_curve_type: str | None = None

    try:
        initialize_gmsh(gmsh, ["grindcae", "-nopopup"])
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        if requested_arc_facets:
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin", min(size, exact_arc_length / requested_arc_facets)
            )

        gmsh.model.add("grindcae_evolved_surface")
        boundary_x = sorted(
            {value for segment in trajectory.surface_segments for value in (segment.x_start_m, segment.x_end_m)}
        )
        top_points = {
            x_m: gmsh.model.occ.addPoint(x_m, _point_y(trajectory, x_m), 0.0)
            for x_m in boundary_x
        }
        bottom_left = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        bottom_right = gmsh.model.occ.addPoint(width, 0.0, 0.0)
        bottom = gmsh.model.occ.addLine(bottom_left, bottom_right)
        right = gmsh.model.occ.addLine(bottom_right, top_points[width])

        top_curves: list[int] = []
        for segment in trajectory.surface_segments:
            start = top_points[segment.x_start_m]
            end = top_points[segment.x_end_m]
            if segment.region == "contact_arc":
                # 中文导读：接触过渡段直接建立为 OCC 圆弧，避免用采样折线代替圆几何。
                center_x, center_y = trajectory.wheel_center_coordinates_m
                center = gmsh.model.occ.addPoint(center_x, center_y, 0.0)
                curve = gmsh.model.occ.addCircleArc(start, center, end)
                arc_curve_tag = curve
            else:
                curve = gmsh.model.occ.addLine(start, end)
            top_curves.append(curve)
        left = gmsh.model.occ.addLine(top_points[0.0], bottom_left)
        loop = gmsh.model.occ.addCurveLoop(
            [bottom, right, *(-tag for tag in reversed(top_curves)), left]
        )
        surface = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()

        # 中文导读：物理组标记完整边界；实际受载的顶部边单元稍后按 contact_arc 选择。
        _add_physical_group(2, [surface], "domain")
        _add_physical_group(1, [bottom], "fixed")
        _add_physical_group(1, top_curves, "contact")
        _add_physical_group(1, [left], "free_left")
        _add_physical_group(1, [right], "free_right")

        if arc_curve_tag is not None:
            arc_curve_type = gmsh.model.getType(1, arc_curve_tag)
            if arc_curve_type != "Circle":
                raise EvolvedMeshGenerationError(
                    "contact_arc was not constructed as a real OCC circle arc"
                )

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)
        if arc_curve_tag is not None:
            gmsh.model.mesh.setTransfiniteCurve(
                arc_curve_tag, requested_arc_facets + 1
            )
        gmsh.model.mesh.generate(2)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        triangle_count = _count_first_order_elements(2)
        if not len(node_tags) or not triangle_count:
            raise EvolvedMeshGenerationError("Gmsh generated an empty 2D mesh")
        generated_element_node_count = _unique_first_order_triangle_node_count()
        generated_solvable_dofs = 2 * generated_element_node_count
        if generated_solvable_dofs > maximum_degrees_of_freedom:
            raise EvolvedMeshGenerationError(
                "generated solvable Gmsh mesh has "
                f"{generated_solvable_dofs} vector-P1 degrees of freedom, "
                f"exceeding the safety limit of {maximum_degrees_of_freedom}"
            )
        generated_arc_facets = (
            0
            if arc_curve_tag is None
            else _count_first_order_elements(1, arc_curve_tag)
        )
        if arc_curve_tag is not None and generated_arc_facets < MINIMUM_ARC_FACET_COUNT:
            raise EvolvedMeshGenerationError(
                "a non-empty circle arc requires at least four first-order facets"
            )
        gmsh.write(str(mesh_path))
        return EvolvedMeshGenerationSummary(
            output_path=mesh_path,
            raw_gmsh_node_count=len(node_tags),
            raw_gmsh_triangle_count=triangle_count,
            generated_element_node_count=generated_element_node_count,
            estimated_vector_p1_degrees_of_freedom=estimate,
            maximum_vector_p1_degrees_of_freedom=maximum_degrees_of_freedom,
            physical_group_names=PHYSICAL_GROUP_NAMES,
            top_curve_count=len(top_curves),
            arc_curve_tag=arc_curve_tag,
            arc_curve_type=arc_curve_type,
            requested_arc_facet_count=requested_arc_facets,
            generated_arc_facet_count=generated_arc_facets,
        )
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
