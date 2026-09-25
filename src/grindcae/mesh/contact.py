"""Validate the exact runtime facets for a local top-edge contact interval."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from skfem import MeshTri


MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT = 4


class MeshValidationError(RuntimeError):
    """Raised when an imported mesh does not preserve the local contact geometry."""


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeshValidationError(f"{field} [m] must be a numeric value")
    try:
        number = float(value)
    except OverflowError as exc:
        raise MeshValidationError(
            f"{field} [m] must be finite and representable"
        ) from exc
    if not math.isfinite(number):
        raise MeshValidationError(f"{field} [m] must be finite")
    return number


def _coordinate_tolerance(extent: float) -> float:
    return max(extent * 1.0e-10, 64.0 * math.ulp(extent))


def _length_tolerance(length: float) -> float:
    return max(length * 1.0e-10, 64.0 * math.ulp(length))


@dataclass(frozen=True, slots=True)
class ActiveContactFacetSelection:
    """Strictly validated imported facets spanning the requested top interval."""

    active_facets: NDArray[np.int32]
    x_start_m: float
    x_end_m: float
    requested_active_length_m: float
    actual_active_length_m: float
    contact_boundary_length_m: float
    contact_facet_count: int
    minimum_required_active_facet_count: int
    x_start_node_indices: tuple[int, ...]
    x_end_node_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        facets = np.asarray(self.active_facets, dtype=np.int32).reshape(-1).copy()
        facets.flags.writeable = False
        object.__setattr__(self, "active_facets", facets)

    @property
    def active_facet_count(self) -> int:
        """Return the number of line facets in the active interval."""

        return int(self.active_facets.size)

    @property
    def x_start_node_count(self) -> int:
        return len(self.x_start_node_indices)

    @property
    def x_end_node_count(self) -> int:
        return len(self.x_end_node_indices)


def _validated_contact_facets(mesh: MeshTri) -> NDArray[np.int32]:
    boundaries = mesh.boundaries or {}
    if "contact" not in boundaries:
        raise MeshValidationError(
            "the imported mesh is missing the physical boundary 'contact'"
        )
    raw_facets = np.asarray(boundaries["contact"])
    if raw_facets.ndim != 1 or raw_facets.size == 0:
        raise MeshValidationError(
            "physical boundary 'contact' must contain a non-empty one-dimensional facet list"
        )
    if not np.issubdtype(raw_facets.dtype, np.integer):
        raise MeshValidationError(
            "physical boundary 'contact' must contain integer facet IDs"
        )
    facets = np.asarray(raw_facets, dtype=np.int64)
    if np.any(facets < 0) or np.any(facets >= mesh.nfacets):
        raise MeshValidationError(
            "physical boundary 'contact' contains an out-of-range facet ID"
        )
    if np.unique(facets).size != facets.size:
        raise MeshValidationError(
            "physical boundary 'contact' contains duplicate facets"
        )
    boundary_facets = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    if np.setdiff1d(facets, boundary_facets).size:
        raise MeshValidationError(
            "physical boundary 'contact' contains a non-boundary facet"
        )
    return facets.astype(np.int32, copy=False)


def _facet_lengths(mesh: MeshTri, facets: NDArray[np.int32]) -> NDArray[np.float64]:
    endpoints = mesh.p[:, mesh.facets[:, facets]]
    return np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=0)


def validate_active_contact_facets(
    mesh: MeshTri,
    *,
    width_m: float,
    height_m: float,
    x_start_m: float,
    x_end_m: float,
    minimum_count: int = MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT,
) -> ActiveContactFacetSelection:
    """Select only facets exactly bounded by geometric nodes at ``x_start_m/x_end_m``.

    The interval is never rounded to nearby facets.  Both split coordinates must
    already exist as top-edge nodes in the imported Gmsh/scikit-fem mesh.
    """

    if not isinstance(mesh, MeshTri):
        raise TypeError("mesh must be a scikit-fem MeshTri")
    if type(minimum_count) is not int or minimum_count <= 0:
        raise ValueError("minimum_count must be a positive integer")

    width = _finite_number(width_m, "width_m")
    height = _finite_number(height_m, "height_m")
    x_start = _finite_number(x_start_m, "x_start_m")
    x_end = _finite_number(x_end_m, "x_end_m")
    if width <= 0.0 or height <= 0.0:
        raise MeshValidationError("width_m and height_m [m] must be greater than zero")
    if not 0.0 < x_start < x_end < width:
        raise MeshValidationError(
            "active contact interval [m] must satisfy 0 < x_start_m < x_end_m < width_m"
        )

    coordinates = np.asarray(mesh.p, dtype=float)
    if coordinates.shape != (2, mesh.nvertices) or not np.all(
        np.isfinite(coordinates)
    ):
        raise MeshValidationError(
            "the imported mesh must contain finite two-dimensional node coordinates"
        )

    x_tolerance = _coordinate_tolerance(width)
    y_tolerance = _coordinate_tolerance(height)
    top_nodes = np.isclose(
        coordinates[1], height, rtol=0.0, atol=y_tolerance
    )
    x_start_nodes = np.flatnonzero(
        top_nodes
        & np.isclose(coordinates[0], x_start, rtol=0.0, atol=x_tolerance)
    )
    x_end_nodes = np.flatnonzero(
        top_nodes
        & np.isclose(coordinates[0], x_end, rtol=0.0, atol=x_tolerance)
    )
    if x_start_nodes.size == 0:
        raise MeshValidationError(
            "x_start_m does not exist as an exact top-edge mesh node; nearest-facet approximation is forbidden"
        )
    if x_end_nodes.size == 0:
        raise MeshValidationError(
            "x_end_m does not exist as an exact top-edge mesh node; nearest-facet approximation is forbidden"
        )

    contact_facets = _validated_contact_facets(mesh)
    contact_endpoints = coordinates[:, mesh.facets[:, contact_facets]]
    if not np.allclose(
        contact_endpoints[1], height, rtol=0.0, atol=y_tolerance
    ):
        raise MeshValidationError(
            "every contact facet must lie on y = height_m"
        )
    contact_lengths = _facet_lengths(mesh, contact_facets)
    if not np.all(np.isfinite(contact_lengths)) or np.any(contact_lengths <= 0.0):
        raise MeshValidationError(
            "physical boundary 'contact' contains a non-positive facet length"
        )
    contact_length = float(math.fsum(float(value) for value in contact_lengths))
    if not math.isclose(
        contact_length,
        width,
        rel_tol=1.0e-8,
        abs_tol=32.0 * math.ulp(width),
    ):
        raise MeshValidationError(
            "physical boundary 'contact' length does not cover the complete top edge"
        )

    endpoint_x = contact_endpoints[0]
    within_interval = np.all(
        (endpoint_x >= x_start - x_tolerance)
        & (endpoint_x <= x_end + x_tolerance),
        axis=0,
    )
    active_facets = contact_facets[within_interval]
    if active_facets.size == 0:
        raise MeshValidationError("the exact active contact facet set is empty")
    if active_facets.size < minimum_count:
        raise MeshValidationError(
            "the exact active contact interval contains "
            f"{active_facets.size} facet(s); at least {minimum_count} are required"
        )
    if active_facets.size >= contact_facets.size:
        raise MeshValidationError(
            "active contact facets must be a strict subset of the complete contact boundary"
        )

    active_nodes = mesh.facets[:, active_facets]
    normalized_edges = np.sort(active_nodes, axis=0).T
    if np.unique(normalized_edges, axis=0).shape[0] != active_facets.size:
        raise MeshValidationError("active contact facets contain duplicate edges")

    active_endpoints = coordinates[:, active_nodes]
    if not np.allclose(
        active_endpoints[1], height, rtol=0.0, atol=y_tolerance
    ):
        raise MeshValidationError("every active contact facet must lie on y = height_m")
    active_left = np.min(active_endpoints[0], axis=0)
    active_right = np.max(active_endpoints[0], axis=0)
    order = np.argsort(active_left, kind="stable")
    sorted_left = active_left[order]
    sorted_right = active_right[order]
    active_facets = active_facets[order]
    if not math.isclose(
        float(sorted_left[0]), x_start, rel_tol=0.0, abs_tol=x_tolerance
    ):
        raise MeshValidationError("the left end of the active facets is not x_start_m")
    if not math.isclose(
        float(sorted_right[-1]), x_end, rel_tol=0.0, abs_tol=x_tolerance
    ):
        raise MeshValidationError("the right end of the active facets is not x_end_m")
    if sorted_left.size > 1 and not np.allclose(
        sorted_left[1:], sorted_right[:-1], rtol=0.0, atol=x_tolerance
    ):
        raise MeshValidationError(
            "active contact facets are not a continuous non-overlapping interval"
        )

    active_lengths = _facet_lengths(mesh, active_facets)
    if not np.all(np.isfinite(active_lengths)) or np.any(active_lengths <= 0.0):
        raise MeshValidationError("active contact facets contain a non-positive length")
    actual_length = float(math.fsum(float(value) for value in active_lengths))
    requested_length = x_end - x_start
    if not math.isclose(
        actual_length,
        requested_length,
        rel_tol=1.0e-10,
        abs_tol=_length_tolerance(requested_length),
    ):
        raise MeshValidationError(
            "actual active facet length does not equal x_end_m - x_start_m; load rescaling is forbidden"
        )

    return ActiveContactFacetSelection(
        active_facets=active_facets,
        x_start_m=x_start,
        x_end_m=x_end,
        requested_active_length_m=requested_length,
        actual_active_length_m=actual_length,
        contact_boundary_length_m=contact_length,
        contact_facet_count=int(contact_facets.size),
        minimum_required_active_facet_count=minimum_count,
        x_start_node_indices=tuple(int(value) for value in x_start_nodes),
        x_end_node_indices=tuple(int(value) for value in x_end_nodes),
    )
