"""Preset material state, active-mesh mappings, and ordered free surfaces."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
import math

import numpy as np
from numpy.typing import NDArray

from grindcae.solver import ImportedMesh

from .models import PresetRemoval


class MaterialTopologyError(ValueError):
    """Raised when preset removal produces an unsupported active topology."""


class EventTopologyQualificationError(MaterialTopologyError):
    """A converged event cannot be released under the surface-only contract."""

    def __init__(self, candidates, removed=None, rejected=None):
        # Older trusted exception archives contain only BaseException.args;
        # pickle restores the full diagnostics dictionary after construction.
        if isinstance(candidates, str) and removed is None and rejected is None:
            super().__init__(candidates)
            return
        self.candidate_background_ids = tuple(sorted(int(v) for v in candidates))
        self.removed_background_ids = tuple(sorted(int(v) for v in removed))
        self.rejected_background_ids = tuple(sorted(int(v) for v in rejected))
        self.failure_stage = 'event_topology_qualification'
        super().__init__(
            'separation candidates must be surface-connected within the atomic event; '
            f'candidates={self.candidate_background_ids}, '
            f'removed={self.removed_background_ids}, rejected={self.rejected_background_ids}'
        )

    def __reduce__(self):
        return (type(self), (self.candidate_background_ids, self.removed_background_ids,
                             self.rejected_background_ids), self.__dict__)


class MaterialState(IntEnum):
    ACTIVE = 1
    DAMAGED = 2
    REMOVED = 3
    SEPARATING = 4


@dataclass(frozen=True, slots=True)
class OrderedFreeSurface:
    node_ids: NDArray[np.int32] = field(repr=False)
    edge_node_ids: NDArray[np.int32] = field(repr=False)
    incident_active_element_ids: NDArray[np.int32] = field(repr=False)
    reference_coordinates_m: NDArray[np.float64] = field(repr=False)
    edge_outward_normals: NDArray[np.float64] = field(repr=False)
    edge_lengths_m: NDArray[np.float64] = field(repr=False)
    tributary_lengths_m: NDArray[np.float64] = field(repr=False)
    candidate_component_dofs: NDArray[np.int32] = field(repr=False)
    candidate_keys: tuple[tuple[int, tuple[int, ...]], ...]
    edge_classifications: tuple[str, ...]
    candidate_incident_edge_ids: tuple[tuple[int, ...], ...]
    candidate_reference_normals: NDArray[np.float64] = field(repr=False)
    candidate_reference_normal_unique: NDArray[np.bool_] = field(repr=False)
    candidate_surface_component_ids: NDArray[np.int32] = field(repr=False)
    candidate_surface_classifications: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedMaterialTopology:
    element_states: NDArray[np.object_] = field(repr=False)
    active_element_ids: NDArray[np.int32] = field(repr=False)
    removed_element_ids: NDArray[np.int32] = field(repr=False)
    active_node_ids: NDArray[np.int32] = field(repr=False)
    isolated_node_ids: NDArray[np.int32] = field(repr=False)
    background_to_active_node: NDArray[np.int32] = field(repr=False)
    active_component_dofs: NDArray[np.int32] = field(repr=False)
    free_surface: OrderedFreeSurface
    removal_sources: tuple[str, ...] = ()

    @property
    def active_total_degrees_of_freedom(self) -> int:
        return int(2 * self.active_node_ids.size)


def _tolerance(coordinates: NDArray[np.float64]) -> float:
    scale = max(1.0, float(np.max(np.abs(coordinates))))
    return max(64.0 * math.ulp(scale), 1.0e-12 * scale)


def _polygon_area(vertices: NDArray[np.float64]) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross_2d(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _on_segment(point: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], tolerance: float) -> bool:
    edge = b - a
    relative = point - a
    cross = abs(float(edge[0] * relative[1] - edge[1] * relative[0]))
    if cross > tolerance * max(1.0, float(np.linalg.norm(edge))):
        return False
    dot = float(np.dot(relative, edge))
    return -tolerance <= dot <= float(np.dot(edge, edge)) + tolerance


def _point_in_polygon(point: NDArray[np.float64], vertices: NDArray[np.float64], tolerance: float) -> bool:
    inside = False
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        if _on_segment(point, start, end, tolerance):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = start[0] + (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if crossing_x >= point[0] - tolerance:
                inside = not inside
    return inside


def _validate_polygon(vertices: NDArray[np.float64], tolerance: float) -> None:
    if vertices.shape[0] < 3 or abs(_polygon_area(vertices)) <= tolerance**2:
        raise MaterialTopologyError("preset removal polygon must have positive area")
    for first in range(len(vertices)):
        a0 = vertices[first]
        a1 = vertices[(first + 1) % len(vertices)]
        if float(np.linalg.norm(a1 - a0)) <= tolerance:
            raise MaterialTopologyError("preset removal polygon contains a degenerate edge")
        for second in range(first + 1, len(vertices)):
            if second in {first, (first + 1) % len(vertices)} or (second + 1) % len(vertices) == first:
                continue
            b0 = vertices[second]
            b1 = vertices[(second + 1) % len(vertices)]
            denominator = _cross_2d(a1 - a0, b1 - b0)
            if abs(denominator) <= tolerance:
                continue
            t = _cross_2d(b0 - a0, b1 - b0) / denominator
            u = _cross_2d(b0 - a0, a1 - a0) / denominator
            if tolerance < t < 1.0 - tolerance and tolerance < u < 1.0 - tolerance:
                raise MaterialTopologyError("preset removal polygon must not self-intersect")


def _selected_elements(
    imported_mesh: ImportedMesh, preset: PresetRemoval
) -> NDArray[np.int32]:
    count = imported_mesh.triangle_count
    if preset.mode == "element_ids":
        ids = np.asarray(preset.element_ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.int32)
        if np.any(ids < 0) or np.any(ids >= count):
            raise MaterialTopologyError("preset removal element id does not exist")
        return ids.astype(np.int32)
    region = preset.region or {}
    vertices = np.asarray(region.get("vertices_m"), dtype=float)
    coordinates = np.asarray(imported_mesh.mesh.p.T, dtype=float)
    tolerance = _tolerance(coordinates)
    _validate_polygon(vertices, tolerance)
    connectivity = np.asarray(imported_mesh.mesh.t.T, dtype=np.int32)
    centroids = np.mean(coordinates[connectivity], axis=1)
    return np.asarray(
        [index for index, point in enumerate(centroids) if _point_in_polygon(point, vertices, tolerance)],
        dtype=np.int32,
    )


def _element_components(connectivity: NDArray[np.int32], active: NDArray[np.int32]) -> int:
    node_to_elements: dict[int, list[int]] = defaultdict(list)
    active_set = set(int(item) for item in active)
    for element_id in active:
        for node in connectivity[element_id]:
            node_to_elements[int(node)].append(int(element_id))
    if not active_set:
        return 0
    seen = set()
    components = 0
    for seed in active_set:
        if seed in seen:
            continue
        components += 1
        queue = deque([seed])
        seen.add(seed)
        while queue:
            current = queue.popleft()
            for node in connectivity[current]:
                for neighbor in node_to_elements[int(node)]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
    return components


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _surface(
    imported_mesh: ImportedMesh,
    active: NDArray[np.int32],
    removed: NDArray[np.int32],
    background_to_active: NDArray[np.int32],
) -> OrderedFreeSurface:
    mesh = imported_mesh.mesh
    coordinates = np.asarray(mesh.p.T, dtype=float)
    connectivity = np.asarray(mesh.t.T, dtype=np.int32)
    tolerance = _tolerance(coordinates)
    all_neighbors: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element_id, nodes in enumerate(connectivity):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            all_neighbors[_edge_key(int(a), int(b))].append(element_id)
    active_set = set(int(item) for item in active)
    removed_set = set(int(item) for item in removed)
    original_contact = {
        _edge_key(int(mesh.facets[0, facet]), int(mesh.facets[1, facet]))
        for facet in np.asarray(mesh.boundaries.get("contact", []), dtype=np.int32)
    }
    fixed = {
        _edge_key(int(mesh.facets[0, facet]), int(mesh.facets[1, facet]))
        for facet in np.asarray(mesh.boundaries.get("fixed", []), dtype=np.int32)
    }
    sides = set()
    for name in ("free_left", "free_right"):
        sides.update(
            _edge_key(int(mesh.facets[0, facet]), int(mesh.facets[1, facet]))
            for facet in np.asarray(mesh.boundaries.get(name, []), dtype=np.int32)
        )
    candidates: list[tuple[int, int, int, str]] = []
    for key, neighbors in all_neighbors.items():
        active_neighbors = [item for item in neighbors if item in active_set]
        if len(active_neighbors) != 1 or key in fixed or key in sides:
            continue
        if key in original_contact:
            classification = "contact_free_surface"
        elif any(item in removed_set for item in neighbors):
            classification = "newly_exposed_removal_surface"
        else:
            continue
        a, b = key
        centroid = np.mean(coordinates[connectivity[active_neighbors[0]]], axis=0)
        cross = _cross_2d(coordinates[b] - coordinates[a], centroid - coordinates[a])
        if cross > 0.0:
            a, b = b, a
        candidates.append((a, b, active_neighbors[0], classification))
    graph: dict[int, list[int]] = defaultdict(list)
    by_key = {}
    for item in candidates:
        graph[item[0]].append(item[1])
        graph[item[1]].append(item[0])
        by_key[_edge_key(item[0], item[1])] = item
    endpoints = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    if len(endpoints) != 2 or any(len(neighbors) not in {1, 2} for neighbors in graph.values()):
        raise MaterialTopologyError("contact free surface must be one unbranched open chain")
    start = min(endpoints, key=lambda node: (coordinates[node, 0], -coordinates[node, 1], node))
    node_order = [start]
    edge_order = []
    previous = None
    current = start
    while True:
        next_nodes = [node for node in graph[current] if node != previous]
        if not next_nodes:
            break
        if len(next_nodes) != 1:
            raise MaterialTopologyError("contact free surface traversal is not unique")
        following = next_nodes[0]
        edge_order.append(by_key[_edge_key(current, following)])
        node_order.append(following)
        previous, current = current, following
    if len(edge_order) != len(candidates):
        raise MaterialTopologyError("contact free surface contains multiple components")
    if coordinates[node_order[-1], 0] < coordinates[node_order[0], 0] - tolerance:
        node_order.reverse()
        edge_order.reverse()
    edge_nodes = np.asarray([[node_order[i], node_order[i + 1]] for i in range(len(node_order) - 1)], dtype=np.int32)
    vectors = coordinates[edge_nodes[:, 1]] - coordinates[edge_nodes[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= tolerance):
        raise MaterialTopologyError("contact free surface contains a zero-length edge")
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0])) / lengths[:, None]
    weights = np.zeros(len(node_order), dtype=float)
    weights[:-1] += 0.5 * lengths
    weights[1:] += 0.5 * lengths
    active_indices = background_to_active[np.asarray(node_order, dtype=np.int32)]
    component_dofs = np.column_stack((2 * active_indices, 2 * active_indices + 1)).astype(np.int32)
    incident: list[list[int]] = [[] for _ in node_order]
    for edge_index in range(len(edge_order)):
        incident[edge_index].append(edge_index)
        incident[edge_index + 1].append(edge_index)
    keys = tuple((int(node), tuple(items)) for node, items in zip(node_order, incident, strict=True))
    node_normals = np.zeros((len(node_order), 2), dtype=float)
    node_normal_unique = np.ones(len(node_order), dtype=bool)
    node_classifications: list[str] = []
    for node_index, edge_ids in enumerate(incident):
        weighted = np.zeros(2, dtype=float)
        classifications = []
        for edge_id in edge_ids:
            weighted += 0.5 * lengths[edge_id] * normals[edge_id]
            classifications.append(edge_order[edge_id][3])
        magnitude = float(np.linalg.norm(weighted))
        if magnitude <= tolerance:
            node_normal_unique[node_index] = False
            node_normals[node_index] = normals[edge_ids[0]]
        else:
            node_normals[node_index] = weighted / magnitude
        node_classifications.append("+".join(sorted(set(classifications))))
    return OrderedFreeSurface(
        node_ids=np.asarray(node_order, dtype=np.int32),
        edge_node_ids=edge_nodes,
        incident_active_element_ids=np.asarray([item[2] for item in edge_order], dtype=np.int32),
        reference_coordinates_m=coordinates[np.asarray(node_order, dtype=np.int32)],
        edge_outward_normals=normals,
        edge_lengths_m=lengths,
        tributary_lengths_m=weights,
        candidate_component_dofs=component_dofs,
        candidate_keys=keys,
        edge_classifications=tuple(item[3] for item in edge_order),
        candidate_incident_edge_ids=tuple(tuple(items) for items in incident),
        candidate_reference_normals=node_normals,
        candidate_reference_normal_unique=node_normal_unique,
        candidate_surface_component_ids=np.zeros(len(node_order), dtype=np.int32),
        candidate_surface_classifications=tuple(node_classifications),
    )


def _fixed_incident_elements(imported_mesh: ImportedMesh) -> set[int]:
    mesh = imported_mesh.mesh
    connectivity = np.asarray(mesh.t.T, dtype=np.int32)
    fixed_edges = {
        _edge_key(int(mesh.facets[0, facet]), int(mesh.facets[1, facet]))
        for facet in np.asarray(mesh.boundaries.get("fixed", []), dtype=np.int32)
    }
    return {
        element_id
        for element_id, nodes in enumerate(connectivity)
        if any(
            _edge_key(int(a), int(b)) in fixed_edges
            for a, b in (
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[2], nodes[0]),
            )
        )
    }


def _build_material_topology(
    imported_mesh: ImportedMesh,
    removed: NDArray[np.int32],
    removal_sources: tuple[str, ...],
) -> PreparedMaterialTopology:
    count = imported_mesh.triangle_count
    if removed.size >= count:
        raise MaterialTopologyError("preset removal must not remove all material")
    mesh = imported_mesh.mesh
    original_contact = {
        _edge_key(int(mesh.facets[0, facet]), int(mesh.facets[1, facet]))
        for facet in np.asarray(mesh.boundaries.get("contact", []), dtype=np.int32)
    }
    removed_set = set(int(item) for item in removed)
    connectivity = np.asarray(mesh.t.T, dtype=np.int32)
    touches_top = any(
        _edge_key(int(a), int(b)) in original_contact
        for element_id in removed_set
        for a, b in ((connectivity[element_id, 0], connectivity[element_id, 1]), (connectivity[element_id, 1], connectivity[element_id, 2]), (connectivity[element_id, 2], connectivity[element_id, 0]))
    )
    if removed.size and not touches_top:
        raise MaterialTopologyError("preset removal must connect to the initial top surface")
    active = np.setdiff1d(np.arange(count, dtype=np.int32), removed).astype(np.int32)
    if _element_components(connectivity, active) != 1:
        raise MaterialTopologyError("remaining effective material must be one connected body")
    active_nodes = np.unique(connectivity[active]).astype(np.int32)
    isolated = np.setdiff1d(np.arange(imported_mesh.node_count, dtype=np.int32), active_nodes).astype(np.int32)
    background_to_active = np.full(imported_mesh.node_count, -1, dtype=np.int32)
    background_to_active[active_nodes] = np.arange(active_nodes.size, dtype=np.int32)
    active_component_dofs = np.vstack((2 * np.arange(active_nodes.size), 2 * np.arange(active_nodes.size) + 1)).astype(np.int32)
    states = np.full(count, MaterialState.ACTIVE, dtype=object)
    states[removed] = MaterialState.REMOVED
    surface = _surface(imported_mesh, active, removed, background_to_active)
    return PreparedMaterialTopology(
        element_states=states,
        active_element_ids=active,
        removed_element_ids=removed,
        active_node_ids=active_nodes,
        isolated_node_ids=isolated,
        background_to_active_node=background_to_active,
        active_component_dofs=active_component_dofs,
        free_surface=surface,
        removal_sources=removal_sources,
    )


def prepare_material_topology(
    imported_mesh: ImportedMesh, preset_removal: PresetRemoval
) -> PreparedMaterialTopology:
    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")
    if not isinstance(preset_removal, PresetRemoval):
        raise TypeError("preset_removal must be a PresetRemoval")
    removed = _selected_elements(imported_mesh, preset_removal)
    sources = ["active"] * imported_mesh.triangle_count
    for element_id in removed:
        sources[int(element_id)] = "preset"
    return _build_material_topology(imported_mesh, removed, tuple(sources))


def transition_material_topology(
    imported_mesh: ImportedMesh,
    committed: PreparedMaterialTopology,
    candidate_element_ids: object,
) -> PreparedMaterialTopology:
    """Atomically rebuild topology after a surface-connected separation event."""

    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")
    if not isinstance(committed, PreparedMaterialTopology):
        raise TypeError("committed must be a PreparedMaterialTopology")
    raw = np.asarray(candidate_element_ids)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise MaterialTopologyError("separation candidates must be integer element ids")
    candidates = np.unique(raw.astype(np.int64))
    if np.any(candidates < 0) or np.any(candidates >= imported_mesh.triangle_count):
        raise MaterialTopologyError("separation candidate element id does not exist")
    old_removed = set(int(item) for item in committed.removed_element_ids)
    new_candidates = set(int(item) for item in candidates) - old_removed
    if not new_candidates:
        return committed
    if new_candidates & _fixed_incident_elements(imported_mesh):
        raise MaterialTopologyError("separation event must not remove the fixed boundary layer")

    mesh = imported_mesh.mesh
    connectivity = np.asarray(mesh.t.T, dtype=np.int32)
    edge_neighbors: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element_id, nodes in enumerate(connectivity):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edge_neighbors[_edge_key(int(a), int(b))].append(element_id)
    original_contact = {
        _edge_key(int(mesh.facets[0, facet]), int(mesh.facets[1, facet]))
        for facet in np.asarray(mesh.boundaries.get("contact", []), dtype=np.int32)
    }
    connected = set(old_removed)
    pending = set(new_candidates)
    while pending:
        accepted = {
            element_id
            for element_id in pending
            if any(
                edge in original_contact
                or any(neighbor in connected for neighbor in edge_neighbors[edge])
                for edge in (
                    _edge_key(int(connectivity[element_id, 0]), int(connectivity[element_id, 1])),
                    _edge_key(int(connectivity[element_id, 1]), int(connectivity[element_id, 2])),
                    _edge_key(int(connectivity[element_id, 2]), int(connectivity[element_id, 0])),
                )
            )
        }
        if not accepted:
            raise EventTopologyQualificationError(new_candidates, old_removed, pending)
        connected.update(accepted)
        pending.difference_update(accepted)

    removed = np.asarray(sorted(connected), dtype=np.int32)
    sources = list(committed.removal_sources)
    if len(sources) != imported_mesh.triangle_count:
        sources = [
            "preset" if index in old_removed else "active"
            for index in range(imported_mesh.triangle_count)
        ]
    for element_id in new_candidates:
        sources[element_id] = "damage_separation"
    return _build_material_topology(imported_mesh, removed, tuple(sources))


__all__ = [
    "MaterialState", "MaterialTopologyError", "OrderedFreeSurface",
    "PreparedMaterialTopology", "prepare_material_topology",
    "transition_material_topology",
]
