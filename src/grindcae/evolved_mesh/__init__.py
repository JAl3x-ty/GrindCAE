"""Public phase 4C3A evolved-surface mesh API."""

from .exporters import (
    RESULT_FORMAT,
    TOP_BOUNDARY_CSV_FIELDS,
    build_summary,
    serialize_summary,
    write_mesh_png,
    write_top_boundary_facets_csv,
)
from .generator import (
    EvolvedMeshGenerationError,
    EvolvedMeshGenerationSummary,
    MINIMUM_ARC_FACET_COUNT,
    estimate_evolved_mesh_degrees_of_freedom,
    generate_evolved_surface_mesh,
)
from .models import (
    EVOLVED_MESH_SCHEMA_VERSION,
    MODEL_TYPE,
    EvolvedMeshCase,
    EvolvedMeshValidationError,
)
from .validation import (
    EvolvedMeshValidationResult,
    EvolvedMeshValidationRuntimeError,
    TopBoundaryFacet,
    validate_evolved_surface_mesh,
)

__all__ = [
    "EVOLVED_MESH_SCHEMA_VERSION",
    "MODEL_TYPE",
    "RESULT_FORMAT",
    "TOP_BOUNDARY_CSV_FIELDS",
    "EvolvedMeshCase",
    "EvolvedMeshGenerationError",
    "EvolvedMeshGenerationSummary",
    "EvolvedMeshValidationError",
    "EvolvedMeshValidationResult",
    "EvolvedMeshValidationRuntimeError",
    "MINIMUM_ARC_FACET_COUNT",
    "TopBoundaryFacet",
    "build_summary",
    "estimate_evolved_mesh_degrees_of_freedom",
    "generate_evolved_surface_mesh",
    "serialize_summary",
    "validate_evolved_surface_mesh",
    "write_mesh_png",
    "write_top_boundary_facets_csv",
]
