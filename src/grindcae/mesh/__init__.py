"""Public API for deterministic 2D Gmsh mesh generation."""

from grindcae.mesh_contract import DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM

from .contact import (
    MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT,
    ActiveContactFacetSelection,
    MeshValidationError,
    validate_active_contact_facets,
)
from .generator import (
    LocalContactMeshSummary,
    MeshGenerationError,
    MeshSummary,
    SplitTopRectangleMeshSummary,
    estimate_local_contact_mesh_degrees_of_freedom,
    estimate_mesh_degrees_of_freedom,
    generate_local_contact_mesh,
    generate_mesh,
    generate_split_top_rectangle_mesh,
)

__all__ = [
    "DEFAULT_MAX_ESTIMATED_DEGREES_OF_FREEDOM",
    "MINIMUM_ACTIVE_BOUNDARY_ELEMENT_COUNT",
    "ActiveContactFacetSelection",
    "LocalContactMeshSummary",
    "MeshGenerationError",
    "MeshSummary",
    "SplitTopRectangleMeshSummary",
    "MeshValidationError",
    "estimate_local_contact_mesh_degrees_of_freedom",
    "estimate_mesh_degrees_of_freedom",
    "generate_local_contact_mesh",
    "generate_mesh",
    "generate_split_top_rectangle_mesh",
    "validate_active_contact_facets",
]
