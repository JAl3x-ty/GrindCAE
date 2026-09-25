"""Incremental global Newton solver for Phase 6A.2 plane-strain J2 FEM."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from skfem import Basis, ElementTriP1, ElementVector
import warnings

from grindcae.elastoplastic import (
    MaterialPointState,
    initial_material_point_state,
)
from grindcae.solver import ImportedMesh

from .assembly import element_trial_response, triangle_B_matrix
from .models import ElastoplasticFemCase


class ElastoplasticSolverError(RuntimeError):
    """Raised when the nonlinear finite-element solve cannot proceed."""


class ElastoplasticConvergenceError(ElastoplasticSolverError):
    """Raised when a load increment reaches its Newton iteration limit."""

    def __init__(
        self,
        message: str,
        *,
        attempted_load_factor: float,
        residual_norm_history_N: tuple[float, ...],
    ) -> None:
        super().__init__(message)
        self.attempted_load_factor = attempted_load_factor
        self.residual_norm_history_N = residual_norm_history_N


@dataclass(frozen=True, slots=True)
class PreparedElastoplasticMesh:
    imported_mesh: ImportedMesh
    basis: Basis
    component_dofs: NDArray[np.int32] = field(repr=False)
    fixed_dofs: NDArray[np.int32] = field(repr=False)
    fixed_x_dofs: NDArray[np.int32] = field(repr=False)
    fixed_y_dofs: NDArray[np.int32] = field(repr=False)
    free_dofs: NDArray[np.int32] = field(repr=False)
    active_facets: NDArray[np.int32] = field(repr=False)
    active_facet_length_m: float
    peak_load_vector_N: NDArray[np.float64] = field(repr=False)
    element_connectivity: NDArray[np.int32] = field(repr=False)
    element_dofs: NDArray[np.int32] = field(repr=False)
    element_B_matrices_per_m: NDArray[np.float64] = field(repr=False)
    element_areas_m2: NDArray[np.float64] = field(repr=False)

    @property
    def node_count(self) -> int:
        return self.imported_mesh.node_count

    @property
    def element_count(self) -> int:
        return self.imported_mesh.triangle_count

    @property
    def total_degrees_of_freedom(self) -> int:
        return int(self.basis.N)


@dataclass(frozen=True, slots=True)
class CommittedStructureState:
    load_factor: float
    displacement_vector_m: NDArray[np.float64] = field(repr=False)
    element_states: tuple[MaterialPointState, ...]
    external_load_vector_N: NDArray[np.float64] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ConvergedIncrementState:
    load_factor: float
    displacement_vector_m: NDArray[np.float64] = field(repr=False)
    element_states: tuple[MaterialPointState, ...]
    newton_iterations: int
    residual_norm_N: float
    correction_norm_m: float
    residual_tolerance_N: float
    correction_tolerance_m: float
    residual_norm_history_N: tuple[float, ...]
    external_force_x_N: float
    external_force_y_N: float
    fixed_reaction_x_N: float
    fixed_reaction_y_N: float
    balance_residual_x_N: float
    balance_residual_y_N: float
    balance_residual_norm_N: float
    maximum_displacement_m: float
    maximum_displacement_node_id: int
    maximum_equivalent_plastic_strain: float
    plastic_element_count: int
    internal_force_vector_N: NDArray[np.float64] = field(repr=False)
    full_residual_vector_N: NDArray[np.float64] = field(repr=False)
    element_total_strain: NDArray[np.float64] = field(repr=False)
    element_stress_Pa: NDArray[np.float64] = field(repr=False)
    element_sigma_zz_Pa: NDArray[np.float64] = field(repr=False)
    external_load_vector_N: NDArray[np.float64] = field(repr=False)

    @property
    def plastic_element_fraction(self) -> float:
        return self.plastic_element_count / len(self.element_states)


@dataclass(frozen=True, slots=True)
class _TrialAssembly:
    internal_force_N: NDArray[np.float64]
    tangent_stiffness_N_per_m: csr_matrix
    element_states: tuple[MaterialPointState, ...]
    element_total_strain: NDArray[np.float64]
    element_stress_Pa: NDArray[np.float64]
    element_sigma_zz_Pa: NDArray[np.float64]


def _coordinate_tolerance(extent: float) -> float:
    return max(extent * 1.0e-10, 64.0 * math.ulp(extent))


def _active_top_facets(
    case: ElastoplasticFemCase,
    imported_mesh: ImportedMesh,
) -> tuple[NDArray[np.int32], float]:
    mesh = imported_mesh.mesh
    try:
        contact = np.asarray(mesh.boundaries["contact"], dtype=np.int32).reshape(-1)
    except KeyError as exc:
        raise ElastoplasticSolverError(
            "imported mesh does not contain the contact physical boundary"
        ) from exc
    endpoints = mesh.p[:, mesh.facets[:, contact]]
    x0 = case.load.x_start_m
    x1 = case.load.x_end_m
    tolerance = _coordinate_tolerance(case.geometry.width)
    minimum_x = np.min(endpoints[0], axis=0)
    maximum_x = np.max(endpoints[0], axis=0)
    mask = (minimum_x >= x0 - tolerance) & (maximum_x <= x1 + tolerance)
    active = contact[mask]
    if active.size == 0:
        raise ElastoplasticSolverError(
            "load.interval has no matching facets on the imported top boundary"
        )
    active_endpoints = mesh.p[:, mesh.facets[:, active]]
    lengths = np.linalg.norm(
        active_endpoints[:, 1] - active_endpoints[:, 0], axis=0
    )
    active_length = float(math.fsum(float(value) for value in lengths))
    if not math.isclose(
        active_length,
        case.load.interval_length_m,
        rel_tol=1.0e-10,
        abs_tol=max(1.0e-14, tolerance),
    ):
        raise ElastoplasticSolverError(
            "actual active top-facet length does not equal load.interval length; "
            "the fixed mesh must contain the exact interval endpoints"
        )
    node_x = np.unique(active_endpoints[0])
    if not any(math.isclose(float(x), x0, rel_tol=0.0, abs_tol=tolerance) for x in node_x):
        raise ElastoplasticSolverError("fixed mesh is missing load.x_start_m")
    if not any(math.isclose(float(x), x1, rel_tol=0.0, abs_tol=tolerance) for x in node_x):
        raise ElastoplasticSolverError("fixed mesh is missing load.x_end_m")
    return np.asarray(active, dtype=np.int32), active_length


def _prepare_mesh_with_load(
    case: ElastoplasticFemCase,
    imported_mesh: ImportedMesh,
    active_facets: NDArray[np.int32],
    peak_load: NDArray[np.float64],
    active_length: float,
) -> PreparedElastoplasticMesh:
    if not isinstance(case, ElastoplasticFemCase):
        raise TypeError("case must be an ElastoplasticFemCase")
    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")
    mesh = imported_mesh.mesh
    coordinates = np.asarray(mesh.p.T, dtype=float)
    connectivity = np.asarray(mesh.t.T, dtype=np.int32)
    if coordinates.shape != (imported_mesh.node_count, 2):
        raise ElastoplasticSolverError("imported mesh coordinates are not two-dimensional")
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    component_dofs = np.asarray(basis.nodal_dofs, dtype=np.int32)
    if component_dofs.shape != (2, imported_mesh.node_count):
        raise ElastoplasticSolverError("unexpected vector-P1 nodal DOF mapping")
    try:
        fixed_dofs = np.asarray(basis.get_dofs("fixed").all(), dtype=np.int32)
    except KeyError as exc:
        raise ElastoplasticSolverError(
            "imported mesh does not contain the fixed physical boundary"
        ) from exc
    fixed_nodes = np.unique(mesh.facets[:, mesh.boundaries["fixed"]])
    fixed_x = component_dofs[0, fixed_nodes]
    fixed_y = component_dofs[1, fixed_nodes]
    all_dofs = np.arange(basis.N, dtype=np.int32)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs).astype(np.int32)
    peak_load = np.asarray(peak_load, dtype=float)
    if peak_load.shape != (basis.N,) or not np.all(np.isfinite(peak_load)):
        raise ElastoplasticSolverError("peak load vector is invalid")

    B_values = np.empty((connectivity.shape[0], 3, 6), dtype=float)
    areas = np.empty(connectivity.shape[0], dtype=float)
    element_dofs = np.empty((connectivity.shape[0], 6), dtype=np.int32)
    for element_id, nodes in enumerate(connectivity):
        B_values[element_id], areas[element_id] = triangle_B_matrix(coordinates[nodes])
        element_dofs[element_id] = np.array(
            [component_dofs[component, node] for node in nodes for component in (0, 1)],
            dtype=np.int32,
        )
    return PreparedElastoplasticMesh(
        imported_mesh=imported_mesh,
        basis=basis,
        component_dofs=component_dofs,
        fixed_dofs=np.sort(fixed_dofs),
        fixed_x_dofs=np.sort(fixed_x),
        fixed_y_dofs=np.sort(fixed_y),
        free_dofs=free_dofs,
        active_facets=active_facets,
        active_facet_length_m=active_length,
        peak_load_vector_N=peak_load,
        element_connectivity=connectivity,
        element_dofs=element_dofs,
        element_B_matrices_per_m=B_values,
        element_areas_m2=areas,
    )


def prepare_elastoplastic_mesh_with_facet_tractions(
    case: ElastoplasticFemCase,
    imported_mesh: ImportedMesh,
    active_facets: object,
    *,
    peak_force_x_N: object,
    peak_force_y_N: object,
    facet_line_load_x_N_per_m: object,
    facet_line_load_y_N_per_m: object,
) -> PreparedElastoplasticMesh:
    """Prepare the retained solver from explicit actual-edge facet tractions."""

    if not isinstance(case, ElastoplasticFemCase):
        raise TypeError("case must be an ElastoplasticFemCase")
    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")
    facets_raw = np.asarray(active_facets)
    if facets_raw.ndim != 1 or not np.issubdtype(facets_raw.dtype, np.integer):
        raise ElastoplasticSolverError("active_facets must be a one-dimensional integer array")
    facets = np.asarray(facets_raw, dtype=np.int32)
    mesh = imported_mesh.mesh
    if facets.size != np.unique(facets).size:
        raise ElastoplasticSolverError("active_facets must not contain duplicates")
    if np.any(facets < 0) or np.any(facets >= mesh.nfacets):
        raise ElastoplasticSolverError("active_facets contains an out-of-range facet ID")
    try:
        contact = np.asarray(mesh.boundaries["contact"], dtype=np.int32).reshape(-1)
    except KeyError as exc:
        raise ElastoplasticSolverError(
            "imported mesh does not contain the contact physical boundary"
        ) from exc
    if not np.all(np.isin(facets, contact)):
        raise ElastoplasticSolverError("all active_facets must belong to contact")
    traction_x = np.asarray(facet_line_load_x_N_per_m, dtype=float)
    traction_y = np.asarray(facet_line_load_y_N_per_m, dtype=float)
    if traction_x.ndim != 1 or traction_y.ndim != 1:
        raise ElastoplasticSolverError("facet traction arrays must be one-dimensional")
    if traction_x.shape != facets.shape or traction_y.shape != facets.shape:
        raise ElastoplasticSolverError("each active facet requires one x/y traction pair")
    try:
        force_x = float(peak_force_x_N)
        force_y = float(peak_force_y_N)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ElastoplasticSolverError("declared peak force must be numeric") from exc
    if not np.isfinite(force_x) or not np.isfinite(force_y):
        raise ElastoplasticSolverError("declared peak force must be finite")
    if not np.all(np.isfinite(traction_x)) or not np.all(np.isfinite(traction_y)):
        raise ElastoplasticSolverError("facet tractions must be finite")
    if facets.size == 0 and (force_x != 0.0 or force_y != 0.0):
        raise ElastoplasticSolverError("empty active facets require zero declared force")

    basis = Basis(mesh, ElementVector(ElementTriP1()))
    component_dofs = np.asarray(basis.nodal_dofs, dtype=np.int32)
    coordinates = np.asarray(mesh.p.T, dtype=float)
    peak_load = np.zeros(basis.N, dtype=float)
    active_length = 0.0
    for index, facet_id in enumerate(facets):
        nodes = mesh.facets[:, facet_id]
        length = float(np.linalg.norm(coordinates[nodes[1]] - coordinates[nodes[0]]))
        if not math.isfinite(length) or length <= 0.0:
            raise ElastoplasticSolverError("active facet length must be finite and positive")
        active_length += length
        for node in nodes:
            peak_load[component_dofs[0, node]] += 0.5 * traction_x[index] * length
            peak_load[component_dofs[1, node]] += 0.5 * traction_y[index] * length
    assembled_x = float(math.fsum(float(v) for v in peak_load[component_dofs[0]]))
    assembled_y = float(math.fsum(float(v) for v in peak_load[component_dofs[1]]))
    for actual, expected, name in (
        (assembled_x, force_x, "Fx"),
        (assembled_y, force_y, "Fy"),
    ):
        scale = max(abs(actual), abs(expected), 1.0)
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12 * scale):
            raise ElastoplasticSolverError(
                f"assembled peak {name} does not equal the declared total force"
            )
    return _prepare_mesh_with_load(case, imported_mesh, facets, peak_load, active_length)


def prepare_elastoplastic_mesh(
    case: ElastoplasticFemCase,
    imported_mesh: ImportedMesh,
) -> PreparedElastoplasticMesh:
    """Precompute the fixed P1 mesh, element matrices, constraints, and load."""

    active_facets, _ = _active_top_facets(case, imported_mesh)
    return prepare_elastoplastic_mesh_with_facet_tractions(
        case,
        imported_mesh,
        active_facets,
        peak_force_x_N=case.load.peak_force_x_N,
        peak_force_y_N=case.load.peak_force_y_N,
        facet_line_load_x_N_per_m=np.full(
            active_facets.size, case.load.peak_line_load_x_N_per_m
        ),
        facet_line_load_y_N_per_m=np.full(
            active_facets.size, case.load.peak_line_load_y_N_per_m
        ),
    )


def initial_structure_state(
    case: ElastoplasticFemCase,
    prepared: PreparedElastoplasticMesh,
) -> CommittedStructureState:
    """Return the prescribed zero-displacement, zero-history structure state."""

    return CommittedStructureState(
        load_factor=0.0,
        displacement_vector_m=np.zeros(prepared.total_degrees_of_freedom, dtype=float),
        element_states=tuple(
            initial_material_point_state(case.material)
            for _ in range(prepared.element_count)
        ),
        external_load_vector_N=np.zeros(prepared.total_degrees_of_freedom, dtype=float),
    )


def _assemble_trial(
    case: ElastoplasticFemCase,
    prepared: PreparedElastoplasticMesh,
    committed: CommittedStructureState,
    displacement: NDArray[np.float64],
) -> _TrialAssembly:
    total_dofs = prepared.total_degrees_of_freedom
    internal = np.zeros(total_dofs, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    states: list[MaterialPointState] = []
    strains = np.empty((prepared.element_count, 3), dtype=float)
    stresses = np.empty((prepared.element_count, 3), dtype=float)
    sigma_zz = np.empty(prepared.element_count, dtype=float)
    displacement_increment = displacement - committed.displacement_vector_m
    for element_id in range(prepared.element_count):
        dofs = prepared.element_dofs[element_id]
        response = element_trial_response(
            case.material,
            committed.element_states[element_id],
            prepared.element_B_matrices_per_m[element_id],
            prepared.element_areas_m2[element_id],
            thickness_m=case.analysis.thickness,
            displacement_increment=displacement_increment[dofs],
        )
        internal[dofs] += response.internal_force_N
        stiffness = response.tangent_stiffness_N_per_m
        rows.extend(np.repeat(dofs, 6).tolist())
        columns.extend(np.tile(dofs, 6).tolist())
        values.extend(stiffness.reshape(-1).tolist())
        states.append(response.material_update.state)
        strains[element_id] = (
            prepared.element_B_matrices_per_m[element_id] @ displacement[dofs]
        )
        stresses[element_id] = response.material_update.in_plane_stress_Pa
        sigma_zz[element_id] = response.material_update.sigma_zz_Pa
    tangent = coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(total_dofs, total_dofs),
    ).tocsr()
    if not np.all(np.isfinite(internal)) or not np.all(np.isfinite(tangent.data)):
        raise ElastoplasticSolverError("global trial assembly produced a non-finite value")
    return _TrialAssembly(
        internal_force_N=internal,
        tangent_stiffness_N_per_m=tangent,
        element_states=tuple(states),
        element_total_strain=strains,
        element_stress_Pa=stresses,
        element_sigma_zz_Pa=sigma_zz,
    )


def _norm(values: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(values))


def advance_external_load_increment(
    case: ElastoplasticFemCase,
    prepared: PreparedElastoplasticMesh,
    committed: CommittedStructureState | ConvergedIncrementState,
    *,
    target_external_load_vector_N: object,
    target_load_marker: float,
) -> ConvergedIncrementState:
    """Solve one arbitrary external-load increment and commit only on convergence."""

    target = float(target_load_marker)
    if not math.isfinite(target):
        raise ValueError("target_load_marker must be finite")
    if len(committed.element_states) != prepared.element_count:
        raise ElastoplasticSolverError("committed material-state count does not match the mesh")
    committed_displacement = np.asarray(committed.displacement_vector_m, dtype=float)
    if committed_displacement.shape != (prepared.total_degrees_of_freedom,):
        raise ElastoplasticSolverError("committed displacement vector shape is invalid")

    external = np.asarray(target_external_load_vector_N, dtype=float)
    if external.shape != (prepared.total_degrees_of_freedom,):
        raise ElastoplasticSolverError("target external load vector shape is invalid")
    if not np.all(np.isfinite(external)):
        raise ElastoplasticSolverError("target external load vector must be finite")
    external_norm = _norm(external[prepared.free_dofs])
    residual_tolerance = (
        case.newton.residual_absolute_tolerance_N
        + case.newton.residual_relative_tolerance * external_norm
    )
    displacement = committed_displacement.copy()
    correction_norm = 0.0
    residual_history: list[float] = []
    trial: _TrialAssembly | None = None
    for correction_count in range(case.newton.maximum_iterations + 1):
        trial = _assemble_trial(case, prepared, committed, displacement)
        residual = external - trial.internal_force_N
        residual_norm = _norm(residual[prepared.free_dofs])
        residual_history.append(residual_norm)
        displacement_scale = _norm(displacement[prepared.free_dofs])
        correction_tolerance = (
            case.newton.displacement_absolute_tolerance_m
            + case.newton.displacement_relative_tolerance * displacement_scale
        )
        if residual_norm <= residual_tolerance and correction_norm <= correction_tolerance:
            break
        if correction_count >= case.newton.maximum_iterations:
            raise ElastoplasticConvergenceError(
                "global Newton iteration did not converge before maximum_iterations",
                attempted_load_factor=target,
                residual_norm_history_N=tuple(residual_history),
            )
        tangent_free = trial.tangent_stiffness_N_per_m[
            prepared.free_dofs
        ][:, prepared.free_dofs]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                correction_free = np.asarray(
                    spsolve(tangent_free, residual[prepared.free_dofs]), dtype=float
                )
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                raise ElastoplasticSolverError(
                    "global Newton tangent solve failed"
                ) from exc
        if not np.all(np.isfinite(correction_free)):
            raise ElastoplasticSolverError("global Newton correction is non-finite")
        correction = np.zeros(prepared.total_degrees_of_freedom, dtype=float)
        correction[prepared.free_dofs] = correction_free
        correction_norm = _norm(correction_free)
        displacement += correction
        displacement[prepared.fixed_dofs] = 0.0
    else:  # pragma: no cover - guarded by the explicit iteration check
        raise AssertionError("unreachable Newton loop exit")

    assert trial is not None
    full_residual = trial.internal_force_N - external
    reaction_x = float(math.fsum(float(v) for v in full_residual[prepared.fixed_x_dofs]))
    reaction_y = float(math.fsum(float(v) for v in full_residual[prepared.fixed_y_dofs]))
    external_x = float(math.fsum(float(v) for v in external[prepared.component_dofs[0]]))
    external_y = float(math.fsum(float(v) for v in external[prepared.component_dofs[1]]))
    balance_x = external_x + reaction_x
    balance_y = external_y + reaction_y
    nodal = np.column_stack(
        (
            displacement[prepared.component_dofs[0]],
            displacement[prepared.component_dofs[1]],
        )
    )
    magnitudes = np.linalg.norm(nodal, axis=1)
    maximum_node = int(np.argmax(magnitudes))
    equivalent_plastic = np.array(
        [state.equivalent_plastic_strain for state in trial.element_states]
    )
    return ConvergedIncrementState(
        load_factor=target,
        displacement_vector_m=displacement.copy(),
        element_states=trial.element_states,
        newton_iterations=correction_count,
        residual_norm_N=residual_history[-1],
        correction_norm_m=correction_norm,
        residual_tolerance_N=residual_tolerance,
        correction_tolerance_m=correction_tolerance,
        residual_norm_history_N=tuple(residual_history),
        external_force_x_N=external_x,
        external_force_y_N=external_y,
        fixed_reaction_x_N=reaction_x,
        fixed_reaction_y_N=reaction_y,
        balance_residual_x_N=balance_x,
        balance_residual_y_N=balance_y,
        balance_residual_norm_N=math.hypot(balance_x, balance_y),
        maximum_displacement_m=float(magnitudes[maximum_node]),
        maximum_displacement_node_id=maximum_node,
        maximum_equivalent_plastic_strain=float(np.max(equivalent_plastic)),
        plastic_element_count=int(
            np.count_nonzero(equivalent_plastic > 1.0e-15)
        ),
        internal_force_vector_N=trial.internal_force_N,
        full_residual_vector_N=full_residual,
        element_total_strain=trial.element_total_strain,
        element_stress_Pa=trial.element_stress_Pa,
        element_sigma_zz_Pa=trial.element_sigma_zz_Pa,
        external_load_vector_N=external.copy(),
    )


def advance_load_increment(
    case: ElastoplasticFemCase,
    prepared: PreparedElastoplasticMesh,
    committed: CommittedStructureState | ConvergedIncrementState,
    *,
    target_load_factor: float,
) -> ConvergedIncrementState:
    """Compatible retained wrapper for one scalar peak-load-factor increment."""

    target = float(target_load_factor)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_load_factor must be finite and in [0, 1]")
    if math.isclose(target, committed.load_factor, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("target_load_factor must differ from the committed load factor")
    return advance_external_load_increment(
        case,
        prepared,
        committed,
        target_external_load_vector_N=target * prepared.peak_load_vector_N,
        target_load_marker=target,
    )
