"""Phase 6A.2 fixed-mesh plane-strain elastoplastic FEM."""

from .assembly import (
    ElementTrialResponse,
    ElastoplasticAssemblyError,
    element_trial_response,
    triangle_B_matrix,
)
from .models import (
    ELASTOPLASTIC_FEM_SCHEMA_VERSION,
    MODEL_TYPE,
    UNIT_SYSTEM,
    ElastoplasticFemCase,
    ElastoplasticFemValidationError,
)
from .solver import (
    CommittedStructureState,
    ConvergedIncrementState,
    ElastoplasticConvergenceError,
    ElastoplasticSolverError,
    PreparedElastoplasticMesh,
    advance_external_load_increment,
    advance_load_increment,
    initial_structure_state,
    prepare_elastoplastic_mesh,
    prepare_elastoplastic_mesh_with_facet_tractions,
)
from .workflow import IncrementRecord, LoadingUnloadingResult, solve_loading_unloading
from .recovery import (
    ElastoplasticRecoveryError,
    ElastoplasticSnapshotFields,
    SnapshotStatistics,
    compute_snapshot_statistics,
    recover_snapshot_fields,
)

__all__ = [
    "ELASTOPLASTIC_FEM_SCHEMA_VERSION",
    "MODEL_TYPE",
    "UNIT_SYSTEM",
    "ElementTrialResponse",
    "ElastoplasticAssemblyError",
    "ElastoplasticFemCase",
    "ElastoplasticFemValidationError",
    "CommittedStructureState",
    "ConvergedIncrementState",
    "ElastoplasticConvergenceError",
    "ElastoplasticSolverError",
    "PreparedElastoplasticMesh",
    "advance_external_load_increment",
    "advance_load_increment",
    "element_trial_response",
    "initial_structure_state",
    "prepare_elastoplastic_mesh",
    "prepare_elastoplastic_mesh_with_facet_tractions",
    "IncrementRecord",
    "LoadingUnloadingResult",
    "solve_loading_unloading",
    "ElastoplasticRecoveryError",
    "ElastoplasticSnapshotFields",
    "SnapshotStatistics",
    "compute_snapshot_statistics",
    "recover_snapshot_fields",
    "triangle_B_matrix",
]
