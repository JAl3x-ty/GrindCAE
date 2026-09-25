"""Phase 6B.1 fixed-mesh complete single-pass moving-load workflow."""

from .loading import (
    FixedTopFacetLoad,
    FixedTopLoadMapping,
    MovingLoadMappingError,
    map_pass_load_to_fixed_top_facets,
)
from .models import (
    MODEL_TYPE,
    MOVING_LOAD_PASS_SCHEMA_VERSION,
    UNIT_SYSTEM,
    FixedMeshMovingLoadPassCase,
    MovingLoadPassValidationError,
    MovingPassSettings,
)
from .overlap import (
    MovingLoadOverlapError,
    ReferenceTopFacet,
    interval_overlap,
)
from .positions import (
    MovingLoadPosition,
    MovingLoadPositionError,
    generate_moving_load_positions,
)
from .workflow import (
    FixedMeshMovingLoadPassResult,
    MovingLoadPassOperations,
    MovingLoadPassWorkflowError,
    MovingLoadPositionResult,
    default_operations,
    solve_fixed_mesh_moving_load_pass,
)


def run_fixed_mesh_moving_load_pass(*args, **kwargs):
    from .__main__ import run_fixed_mesh_moving_load_pass as run

    return run(*args, **kwargs)


__all__ = [
    "MODEL_TYPE",
    "MOVING_LOAD_PASS_SCHEMA_VERSION",
    "UNIT_SYSTEM",
    "FixedMeshMovingLoadPassCase",
    "MovingLoadPassValidationError",
    "MovingPassSettings",
    "ReferenceTopFacet",
    "MovingLoadOverlapError",
    "interval_overlap",
    "FixedTopFacetLoad",
    "FixedTopLoadMapping",
    "MovingLoadMappingError",
    "map_pass_load_to_fixed_top_facets",
    "MovingLoadPosition",
    "MovingLoadPositionError",
    "generate_moving_load_positions",
    "MovingLoadPositionResult",
    "FixedMeshMovingLoadPassResult",
    "MovingLoadPassOperations",
    "MovingLoadPassWorkflowError",
    "default_operations",
    "solve_fixed_mesh_moving_load_pass",
    "run_fixed_mesh_moving_load_pass",
]
