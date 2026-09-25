"""Phase 6B.2 fixed-mesh complete-pass elastoplastic history workflow."""

from .loading import assemble_fixed_top_mapping_load_vector
from .models import (
    HISTORY_PASS_SCHEMA_VERSION,
    MODEL_TYPE,
    UNIT_SYSTEM,
    FixedMeshElastoplasticHistoryPassCase,
    HistoryPassValidationError,
    SnapshotRetentionSettings,
    TransitionSettings,
)
from .solver import (
    ExternalLoadTransitionResult,
    HistoryTransitionError,
    solve_external_load_transition,
)
from .workflow import (
    FixedMeshElastoplasticHistoryPassResult,
    HistoryPassRow,
    HistoryPassWorkflowError,
    HistoryPositionRecord,
    TargetFieldSnapshot,
    solve_fixed_mesh_elastoplastic_history_pass,
)


def run_fixed_mesh_elastoplastic_history_pass(*args, **kwargs):
    from .__main__ import run_fixed_mesh_elastoplastic_history_pass as run
    return run(*args, **kwargs)

__all__ = [
    "HISTORY_PASS_SCHEMA_VERSION",
    "MODEL_TYPE",
    "UNIT_SYSTEM",
    "FixedMeshElastoplasticHistoryPassCase",
    "HistoryPassValidationError",
    "SnapshotRetentionSettings",
    "TransitionSettings",
    "assemble_fixed_top_mapping_load_vector",
    "ExternalLoadTransitionResult",
    "HistoryTransitionError",
    "solve_external_load_transition",
    "FixedMeshElastoplasticHistoryPassResult",
    "HistoryPassRow",
    "HistoryPassWorkflowError",
    "HistoryPositionRecord",
    "TargetFieldSnapshot",
    "solve_fixed_mesh_elastoplastic_history_pass",
    "run_fixed_mesh_elastoplastic_history_pass",
]
