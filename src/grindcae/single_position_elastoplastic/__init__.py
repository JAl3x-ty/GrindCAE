"""Phase 6A.3 single-position evolved-surface elastoplastic workflow."""

from .models import (
    MODEL_TYPE,
    SINGLE_POSITION_ELASTOPLASTIC_SCHEMA_VERSION,
    UNIT_SYSTEM,
    SinglePositionElastoplasticCase,
    SinglePositionElastoplasticValidationError,
)
from .recovery import (
    SurfaceRecoveryError,
    SurfaceRecoveryPoint,
    SurfaceRecoveryResult,
    build_surface_recovery,
)
from .workflow import (
    SinglePositionElastoplasticOperations,
    SinglePositionElastoplasticResult,
    SinglePositionElastoplasticWorkflowError,
    default_operations,
    evolved_mesh_case,
    solve_single_position_elastoplastic,
)


def run_single_position_elastoplastic(*args, **kwargs):
    """Lazily enter the artifact workflow without importing the CLI at package load."""

    from .__main__ import run_single_position_elastoplastic as run

    return run(*args, **kwargs)

__all__ = [
    "MODEL_TYPE",
    "SINGLE_POSITION_ELASTOPLASTIC_SCHEMA_VERSION",
    "UNIT_SYSTEM",
    "SinglePositionElastoplasticCase",
    "SinglePositionElastoplasticValidationError",
    "SurfaceRecoveryError",
    "SurfaceRecoveryPoint",
    "SurfaceRecoveryResult",
    "build_surface_recovery",
    "SinglePositionElastoplasticOperations",
    "SinglePositionElastoplasticResult",
    "SinglePositionElastoplasticWorkflowError",
    "default_operations",
    "evolved_mesh_case",
    "solve_single_position_elastoplastic",
    "run_single_position_elastoplastic",
]
