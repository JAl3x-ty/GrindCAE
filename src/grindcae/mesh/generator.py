"""Generate deterministic rectangular Gmsh meshes and physical groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import gmsh

from grindcae.domain.models import SimulationCase
from grindcae.gmsh_runtime import initialize_gmsh
from grindcae.grinding import GrindingSimulationCase, ResolvedGrindingLoad
from grindcae.mesh_contract import (
    DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM,
    PHYSICAL_GROUP_NAMES,
)

from .contact import MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT


class MeshGenerationError(RuntimeError):
    """Raised when Gmsh cannot produce the required rectangular mesh."""


@dataclass(frozen=True, slots=True)
class MeshSummary:
    """Basic facts about a generated Gmsh mesh."""

    output_path: Path
    node_count: int
    triangle_count: int
    physical_group_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalContactMeshSummary:
    """Generated-mesh facts and the exact geometric interval requested."""

    output_path: Path
    node_count: int
    triangle_count: int
    physical_group_names: tuple[str, ...]
    x_start_m: float
    x_end_m: float
    requested_active_length_m: float
    top_curve_count: int
    active_geometric_curve_element_count: int
    minimum_required_active_boundary_element_count: int
    local_active_element_target: int


@dataclass(frozen=True, slots=True)
class SplitTopRectangleMeshSummary:
    """Generated rectangular mesh with exact top-boundary split coordinates."""

    output_path: Path
    node_count: int
    triangle_count: int
    physical_group_names: tuple[str, ...]
    top_split_x_m: tuple[float, ...]
    top_curve_count: int


@dataclass(frozen=True, slots=True)
class _LocalContactGeometry:
    x_start_m: float
    x_end_m: float
    active_length_m: float
    active_curve_element_target: int


def _base_mesh_estimate(width: float, height: float, target_size: float) -> int:
    # The boundary term protects long, thin rectangles where area alone is not
    # representative of the number of boundary nodes.
    try:
        width_ratio = width / target_size
        height_ratio = height / target_size
        estimate = (
            4.0 * width_ratio * height_ratio
            + 8.0 * (width_ratio + height_ratio)
            + 32.0
        )
    except OverflowError as exc:
        raise MeshGenerationError(
            "mesh.target_size produces an overflowing mesh-size estimate"
        ) from exc
    if not math.isfinite(estimate):
        raise MeshGenerationError(
            "mesh.target_size produces a non-finite mesh-size estimate"
        )
    return max(1, math.ceil(estimate))


def estimate_mesh_degrees_of_freedom(case: SimulationCase) -> int:
    """Return a conservative vector-P1 solve-size estimate before Gmsh runs."""

    if not isinstance(case, SimulationCase):
        raise TypeError("case must be a SimulationCase")
    return _base_mesh_estimate(
        case.geometry.width,
        case.geometry.height,
        case.mesh.target_size,
    )


def _active_curve_element_target(active_length: float, target_size: float) -> int:
    try:
        ratio = active_length / target_size
    except OverflowError as exc:
        raise MeshGenerationError(
            "local contact refinement produces an overflowing element-count target"
        ) from exc
    if not math.isfinite(ratio):
        raise MeshGenerationError(
            "local contact refinement produces a non-finite element-count target"
        )
    return max(MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT, math.ceil(ratio))


def _validate_local_contact_inputs(
    case: GrindingSimulationCase,
    resolved_load: ResolvedGrindingLoad,
) -> _LocalContactGeometry:
    if not isinstance(case, GrindingSimulationCase):
        raise TypeError("case must be a GrindingSimulationCase")
    if not isinstance(resolved_load, ResolvedGrindingLoad):
        raise TypeError("resolved_load must be a ResolvedGrindingLoad")
    if resolved_load.force_prediction.force_case != case.force_model:
        raise MeshGenerationError(
            "resolved_load force-model provenance does not match the grinding case"
        )
    contact_zone = case.contact_zone
    if (
        resolved_load.boundary is not contact_zone.boundary
        or resolved_load.distribution != contact_zone.distribution
        or resolved_load.tangential_direction != contact_zone.tangential_direction
        or not math.isclose(
            resolved_load.center_x_m,
            contact_zone.center_x_m,
            rel_tol=0.0,
            abs_tol=max(
                32.0 * math.ulp(case.geometry.width),
                case.geometry.width * 1.0e-12,
            ),
        )
    ):
        raise MeshGenerationError(
            "resolved_load load-mapping provenance does not match the grinding case"
        )
    x_start = resolved_load.x_start_m
    x_end = resolved_load.x_end_m
    active_length = resolved_load.active_length_m
    if not 0.0 < x_start < x_end < case.geometry.width:
        raise MeshGenerationError(
            "local contact interval must satisfy 0 < x_start_m < x_end_m < geometry.width"
        )
    if not math.isclose(
        x_end - x_start,
        active_length,
        rel_tol=1.0e-12,
        abs_tol=max(1.0e-15, 32.0 * math.ulp(active_length)),
    ):
        raise MeshGenerationError(
            "resolved_load.active_length_m must equal x_end_m - x_start_m"
        )
    return _LocalContactGeometry(
        x_start_m=x_start,
        x_end_m=x_end,
        active_length_m=active_length,
        active_curve_element_target=_active_curve_element_target(
            active_length, case.mesh.target_size
        ),
    )


def estimate_local_contact_mesh_degrees_of_freedom(
    case: GrindingSimulationCase,
    resolved_load: ResolvedGrindingLoad,
) -> int:
    """Estimate vector-P1 size while conservatively including local refinement."""

    local = _validate_local_contact_inputs(case, resolved_load)
    width = case.geometry.width
    height = case.geometry.height
    local_size = local.active_length_m / local.active_curve_element_target

    # Bound the local mesh by conservatively treating the complete rectangle
    # as if it used the finest active-curve size.  Gmsh normally transitions
    # back to target_size away from the interval, so this deliberately
    # overestimates rather than allowing refinement to bypass the guard.
    return _base_mesh_estimate(width, height, local_size)


def _coordinate_tolerance(extent: float) -> float:
    return max(extent * 1.0e-8, 32.0 * math.ulp(extent))


def _curve_endpoint_coordinates(curve_tag: int) -> tuple[tuple[float, ...], ...]:
    endpoint_entities = gmsh.model.getBoundary(
        [(1, curve_tag)],
        combined=False,
        oriented=False,
        recursive=False,
    )
    endpoint_tags = sorted(
        {tag for dimension, tag in endpoint_entities if dimension == 0}
    )
    if len(endpoint_tags) != 2:
        raise MeshGenerationError(
            f"boundary curve {curve_tag} must have exactly two distinct endpoints"
        )
    coordinates: list[tuple[float, ...]] = []
    for point_tag in endpoint_tags:
        values = tuple(float(value) for value in gmsh.model.getValue(0, point_tag, []))
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise MeshGenerationError(
                f"boundary curve {curve_tag} has invalid endpoint coordinates"
            )
        coordinates.append(values)
    return tuple(coordinates)


def _classify_boundary_curves(
    surface_tag: int, width: float, height: float
) -> dict[str, list[int]]:
    """Identify physical roles by endpoint coordinates, allowing split sides."""

    boundary_entities = gmsh.model.getBoundary(
        [(2, surface_tag)],
        combined=False,
        oriented=False,
        recursive=False,
    )
    curve_tags = [tag for dimension, tag in boundary_entities if dimension == 1]
    if len(curve_tags) < 4 or len(set(curve_tags)) != len(curve_tags):
        raise MeshGenerationError(
            "the rectangular domain boundary must contain distinct curves for all four sides"
        )

    x_tolerance = _coordinate_tolerance(width)
    y_tolerance = _coordinate_tolerance(height)
    classified: dict[str, list[int]] = {
        "fixed": [],
        "contact": [],
        "free_left": [],
        "free_right": [],
    }
    for curve_tag in curve_tags:
        endpoints = _curve_endpoint_coordinates(curve_tag)
        x_values = tuple(point[0] for point in endpoints)
        y_values = tuple(point[1] for point in endpoints)
        z_values = tuple(point[2] for point in endpoints)
        matches: list[str] = []
        if all(math.isclose(y, 0.0, rel_tol=0.0, abs_tol=y_tolerance) for y in y_values):
            matches.append("fixed")
        if all(
            math.isclose(y, height, rel_tol=0.0, abs_tol=y_tolerance)
            for y in y_values
        ):
            matches.append("contact")
        if all(math.isclose(x, 0.0, rel_tol=0.0, abs_tol=x_tolerance) for x in x_values):
            matches.append("free_left")
        if all(
            math.isclose(x, width, rel_tol=0.0, abs_tol=x_tolerance)
            for x in x_values
        ):
            matches.append("free_right")
        if len(matches) != 1 or not all(
            math.isclose(z, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            for z in z_values
        ):
            raise MeshGenerationError(
                f"could not identify physical role for boundary curve {curve_tag}"
            )
        if math.hypot(
            endpoints[1][0] - endpoints[0][0],
            endpoints[1][1] - endpoints[0][1],
        ) <= max(x_tolerance, y_tolerance):
            raise MeshGenerationError(f"boundary curve {curve_tag} is degenerate")
        classified[matches[0]].append(curve_tag)

    missing = sorted(name for name, tags in classified.items() if not tags)
    if missing:
        raise MeshGenerationError(
            f"missing boundary curves for physical groups: {missing}"
        )

    def centre_coordinate(curve_tag: int, coordinate_index: int) -> float:
        endpoints = _curve_endpoint_coordinates(curve_tag)
        return 0.5 * (
            endpoints[0][coordinate_index] + endpoints[1][coordinate_index]
        )

    classified["fixed"].sort(key=lambda tag: centre_coordinate(tag, 0))
    classified["contact"].sort(key=lambda tag: centre_coordinate(tag, 0))
    classified["free_left"].sort(key=lambda tag: centre_coordinate(tag, 1))
    classified["free_right"].sort(key=lambda tag: centre_coordinate(tag, 1))
    return classified


def _add_physical_group(dimension: int, entity_tags: list[int], name: str) -> None:
    physical_tag = gmsh.model.addPhysicalGroup(dimension, entity_tags)
    gmsh.model.setPhysicalName(dimension, physical_tag, name)


def _count_first_order_triangles() -> int:
    element_types, element_tag_blocks, _ = gmsh.model.mesh.getElements(2)
    triangle_count = 0
    for element_type, element_tags in zip(element_types, element_tag_blocks):
        (
            element_name,
            dimension,
            order,
            node_count,
            _,
            primary_node_count,
        ) = gmsh.model.mesh.getElementProperties(element_type)
        if (
            dimension != 2
            or not element_name.startswith("Triangle")
            or order != 1
            or node_count != 3
            or primary_node_count != 3
        ):
            raise MeshGenerationError(
                "the generated 2D mesh contains an element that is not a first-order triangle"
            )
        triangle_count += len(element_tags)
    return triangle_count


def _count_first_order_lines(curve_tag: int) -> int:
    element_types, element_tag_blocks, _ = gmsh.model.mesh.getElements(1, curve_tag)
    line_count = 0
    for element_type, element_tags in zip(element_types, element_tag_blocks):
        properties = gmsh.model.mesh.getElementProperties(element_type)
        name, dimension, order, node_count = properties[:4]
        primary_node_count = properties[5]
        if (
            dimension != 1
            or not name.startswith("Line")
            or order != 1
            or node_count != 2
            or primary_node_count != 2
        ):
            raise MeshGenerationError(
                "the generated active boundary contains an element that is not a first-order line"
            )
        line_count += len(element_tags)
    return line_count


def _validate_output_path(output_path: str | Path) -> Path:
    mesh_path = Path(output_path).expanduser().resolve()
    if mesh_path.suffix.lower() != ".msh":
        raise MeshGenerationError("output_path must use the .msh extension")
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    return mesh_path


def _validate_safety_limit(maximum: int) -> None:
    if type(maximum) is not int or maximum <= 0:
        raise ValueError(
            "maximum_estimated_degrees_of_freedom must be a positive integer"
        )


def _reject_unsafe_estimate(estimate: int, maximum: int) -> None:
    if estimate > maximum:
        raise MeshGenerationError(
            "conservative pre-mesh estimate requires approximately "
            f"{estimate} displacement degrees of freedom, exceeding the configured "
            f"safety limit of {maximum}; increase mesh.target_size"
        )


def _add_local_contact_rectangle(
    width: float,
    height: float,
    local: _LocalContactGeometry,
) -> tuple[int, int, tuple[int, int]]:
    bottom_left = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    bottom_right = gmsh.model.occ.addPoint(width, 0.0, 0.0)
    top_right = gmsh.model.occ.addPoint(width, height, 0.0)
    active_right = gmsh.model.occ.addPoint(local.x_end_m, height, 0.0)
    active_left = gmsh.model.occ.addPoint(local.x_start_m, height, 0.0)
    top_left = gmsh.model.occ.addPoint(0.0, height, 0.0)

    bottom = gmsh.model.occ.addLine(bottom_left, bottom_right)
    right = gmsh.model.occ.addLine(bottom_right, top_right)
    top_right_curve = gmsh.model.occ.addLine(top_right, active_right)
    active_curve = gmsh.model.occ.addLine(active_right, active_left)
    top_left_curve = gmsh.model.occ.addLine(active_left, top_left)
    left = gmsh.model.occ.addLine(top_left, bottom_left)
    loop = gmsh.model.occ.addCurveLoop(
        [bottom, right, top_right_curve, active_curve, top_left_curve, left]
    )
    surface = gmsh.model.occ.addPlaneSurface([loop])
    return surface, active_curve, (active_left, active_right)


def _add_split_top_rectangle(
    width: float,
    height: float,
    split_x_m: tuple[float, ...],
) -> int:
    bottom_left = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    bottom_right = gmsh.model.occ.addPoint(width, 0.0, 0.0)
    top_right = gmsh.model.occ.addPoint(width, height, 0.0)
    split_points = [
        gmsh.model.occ.addPoint(value, height, 0.0)
        for value in reversed(split_x_m)
    ]
    top_left = gmsh.model.occ.addPoint(0.0, height, 0.0)
    bottom = gmsh.model.occ.addLine(bottom_left, bottom_right)
    right = gmsh.model.occ.addLine(bottom_right, top_right)
    top_nodes = [top_right, *split_points, top_left]
    top_curves = [
        gmsh.model.occ.addLine(start, end)
        for start, end in zip(top_nodes, top_nodes[1:])
    ]
    left = gmsh.model.occ.addLine(top_left, bottom_left)
    loop = gmsh.model.occ.addCurveLoop([bottom, right, *top_curves, left])
    return gmsh.model.occ.addPlaneSurface([loop])


def _configure_gmsh(case: SimulationCase | GrindingSimulationCase) -> None:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", case.mesh.target_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", case.mesh.target_size)


def _finish_mesh(
    *,
    maximum_degrees_of_freedom: int,
) -> tuple[int, int]:
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    triangle_count = _count_first_order_triangles()
    if len(node_tags) == 0 or triangle_count == 0:
        raise MeshGenerationError("Gmsh generated an empty 2D mesh")
    actual_degrees_of_freedom = 2 * len(node_tags)
    if actual_degrees_of_freedom > maximum_degrees_of_freedom:
        raise MeshGenerationError(
            "generated mesh contains "
            f"{actual_degrees_of_freedom} vector-P1 displacement degrees of freedom, "
            f"exceeding the configured safety limit of {maximum_degrees_of_freedom}"
        )
    return len(node_tags), triangle_count


def generate_mesh(
    case: SimulationCase,
    output_path: str | Path,
    *,
    maximum_estimated_degrees_of_freedom: int = (
        DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM
    ),
) -> MeshSummary:
    """Generate the backward-compatible schema-3 Gmsh 4.1 triangular mesh."""

    if not isinstance(case, SimulationCase):
        raise TypeError("case must be a SimulationCase")
    if gmsh.isInitialized():
        raise MeshGenerationError("Gmsh is already initialized")
    _validate_safety_limit(maximum_estimated_degrees_of_freedom)
    _reject_unsafe_estimate(
        estimate_mesh_degrees_of_freedom(case),
        maximum_estimated_degrees_of_freedom,
    )
    mesh_path = _validate_output_path(output_path)

    try:
        initialize_gmsh(gmsh, ["grindcae", "-nopopup"])
        _configure_gmsh(case)
        gmsh.model.add("grindcae_plate_contact")
        surface_tag = gmsh.model.occ.addRectangle(
            0.0,
            0.0,
            0.0,
            case.geometry.width,
            case.geometry.height,
        )
        gmsh.model.occ.synchronize()

        boundary_curves = _classify_boundary_curves(
            surface_tag,
            case.geometry.width,
            case.geometry.height,
        )
        _add_physical_group(2, [surface_tag], "domain")
        for name in PHYSICAL_GROUP_NAMES[1:]:
            _add_physical_group(1, boundary_curves[name], name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), case.mesh.target_size)
        gmsh.model.mesh.generate(2)
        node_count, triangle_count = _finish_mesh(
            maximum_degrees_of_freedom=maximum_estimated_degrees_of_freedom,
        )
        gmsh.write(str(mesh_path))
        return MeshSummary(
            output_path=mesh_path,
            node_count=node_count,
            triangle_count=triangle_count,
            physical_group_names=PHYSICAL_GROUP_NAMES,
        )
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def generate_split_top_rectangle_mesh(
    case: object,
    output_path: str | Path,
    *,
    top_split_x_m: tuple[float, ...],
    maximum_estimated_degrees_of_freedom: int = (
        DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM
    ),
) -> SplitTopRectangleMeshSummary:
    """Generate the shared rectangle contract with exact top-edge splits."""

    geometry = getattr(case, "geometry", None)
    mesh_settings = getattr(case, "mesh", None)
    if geometry is None or mesh_settings is None:
        raise TypeError("case must provide validated geometry and mesh settings")
    width = float(geometry.width)
    height = float(geometry.height)
    target_size = float(mesh_settings.target_size)
    splits = tuple(float(value) for value in top_split_x_m)
    if tuple(sorted(set(splits))) != splits:
        raise MeshGenerationError("top_split_x_m must be sorted and unique")
    if not all(0.0 < value < width for value in splits):
        raise MeshGenerationError("top_split_x_m values must be strictly inside (0, width)")
    if gmsh.isInitialized():
        raise MeshGenerationError("Gmsh is already initialized")
    _validate_safety_limit(maximum_estimated_degrees_of_freedom)
    _reject_unsafe_estimate(
        _base_mesh_estimate(width, height, target_size),
        maximum_estimated_degrees_of_freedom,
    )
    mesh_path = _validate_output_path(output_path)
    try:
        initialize_gmsh(gmsh, ["grindcae", "-nopopup"])
        _configure_gmsh(case)
        gmsh.model.add("grindcae_split_top_rectangle")
        surface_tag = (
            _add_split_top_rectangle(width, height, splits)
            if splits
            else gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, width, height)
        )
        gmsh.model.occ.synchronize()
        boundary_curves = _classify_boundary_curves(surface_tag, width, height)
        if len(boundary_curves["contact"]) != len(splits) + 1:
            raise MeshGenerationError("the top edge does not contain every requested split")
        _add_physical_group(2, [surface_tag], "domain")
        for name in PHYSICAL_GROUP_NAMES[1:]:
            _add_physical_group(1, boundary_curves[name], name)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), target_size)
        gmsh.model.mesh.generate(2)
        node_count, triangle_count = _finish_mesh(
            maximum_degrees_of_freedom=maximum_estimated_degrees_of_freedom
        )
        gmsh.write(str(mesh_path))
        return SplitTopRectangleMeshSummary(
            output_path=mesh_path,
            node_count=node_count,
            triangle_count=triangle_count,
            physical_group_names=PHYSICAL_GROUP_NAMES,
            top_split_x_m=splits,
            top_curve_count=len(boundary_curves["contact"]),
        )
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def generate_local_contact_mesh(
    case: GrindingSimulationCase,
    resolved_load: ResolvedGrindingLoad,
    output_path: str | Path,
    *,
    maximum_estimated_degrees_of_freedom: int = (
        DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM
    ),
) -> LocalContactMeshSummary:
    """Generate a schema-4 mesh with exact top-edge splits at x0 and x1."""

    local = _validate_local_contact_inputs(case, resolved_load)
    if gmsh.isInitialized():
        raise MeshGenerationError("Gmsh is already initialized")
    _validate_safety_limit(maximum_estimated_degrees_of_freedom)
    estimate = estimate_local_contact_mesh_degrees_of_freedom(case, resolved_load)
    _reject_unsafe_estimate(estimate, maximum_estimated_degrees_of_freedom)
    mesh_path = _validate_output_path(output_path)

    try:
        initialize_gmsh(gmsh, ["grindcae", "-nopopup"])
        _configure_gmsh(case)
        local_size = local.active_length_m / local.active_curve_element_target
        gmsh.option.setNumber(
            "Mesh.MeshSizeMin", min(case.mesh.target_size, local_size)
        )

        gmsh.model.add("grindcae_local_contact")
        surface_tag, active_curve, active_point_tags = _add_local_contact_rectangle(
            case.geometry.width,
            case.geometry.height,
            local,
        )
        gmsh.model.occ.synchronize()

        boundary_curves = _classify_boundary_curves(
            surface_tag,
            case.geometry.width,
            case.geometry.height,
        )
        if active_curve not in boundary_curves["contact"]:
            raise MeshGenerationError(
                "the exact active geometric curve was not classified as contact"
            )
        if len(boundary_curves["contact"]) < 3:
            raise MeshGenerationError(
                "the local-contact top edge must contain left, active, and right curves"
            )

        _add_physical_group(2, [surface_tag], "domain")
        for name in PHYSICAL_GROUP_NAMES[1:]:
            _add_physical_group(1, boundary_curves[name], name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), case.mesh.target_size)
        gmsh.model.mesh.setSize(
            [(0, point_tag) for point_tag in active_point_tags], local_size
        )
        gmsh.model.mesh.setTransfiniteCurve(
            active_curve, local.active_curve_element_target + 1
        )
        gmsh.model.mesh.generate(2)

        active_element_count = _count_first_order_lines(active_curve)
        if active_element_count < MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT:
            raise MeshGenerationError(
                "the active geometric curve generated "
                f"{active_element_count} line element(s); at least "
                f"{MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT} are required"
            )
        node_count, triangle_count = _finish_mesh(
            maximum_degrees_of_freedom=maximum_estimated_degrees_of_freedom,
        )
        gmsh.write(str(mesh_path))
        return LocalContactMeshSummary(
            output_path=mesh_path,
            node_count=node_count,
            triangle_count=triangle_count,
            physical_group_names=PHYSICAL_GROUP_NAMES,
            x_start_m=local.x_start_m,
            x_end_m=local.x_end_m,
            requested_active_length_m=local.active_length_m,
            top_curve_count=len(boundary_curves["contact"]),
            active_geometric_curve_element_count=active_element_count,
            minimum_required_active_boundary_element_count=(
                MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT
            ),
            local_active_element_target=local.active_curve_element_target,
        )
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
