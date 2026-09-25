"""Phase 6B.1 orchestration over one fixed mesh and public Phase 4C2 loads."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable

import numpy as np

from grindcae.mesh import generate_split_top_rectangle_mesh
from grindcae.pass_loading import PassLoadResult, build_pass_load
from grindcae.solver import ImportedMesh, import_gmsh_mesh

from .loading import FixedTopLoadMapping, map_pass_load_to_fixed_top_facets
from .models import FixedMeshMovingLoadPassCase
from .overlap import ReferenceTopFacet
from .positions import MovingLoadPosition, generate_moving_load_positions


class MovingLoadPassWorkflowError(RuntimeError):
    """Raised when one-mesh moving-load provenance or geometry is invalid."""


@dataclass(frozen=True, slots=True)
class MovingLoadPassOperations:
    generate_mesh: Callable[..., object]
    import_mesh: Callable[..., object]
    build_pass_load: Callable[..., object]


def default_operations() -> MovingLoadPassOperations:
    return MovingLoadPassOperations(
        generate_mesh=generate_split_top_rectangle_mesh,
        import_mesh=import_gmsh_mesh,
        build_pass_load=build_pass_load,
    )


@dataclass(frozen=True, slots=True)
class MovingLoadPositionResult:
    position: MovingLoadPosition
    pass_load: PassLoadResult
    load_mapping: FixedTopLoadMapping


@dataclass(frozen=True, slots=True)
class FixedMeshMovingLoadPassResult:
    case: FixedMeshMovingLoadPassCase
    reference_pass_load: PassLoadResult
    reference_mesh: ImportedMesh
    top_facets: tuple[ReferenceTopFacet, ...]
    positions: tuple[MovingLoadPositionResult, ...]
    msh_source: Path
    mesh_generation_count: int = 1
    mesh_import_count: int = 1

    @property
    def maximum_force_balance_residual_N(self) -> float:
        return max((item.load_mapping.force_balance_residual_N for item in self.positions), default=0.0)

    @property
    def maximum_contact_ratio(self) -> float:
        return max((item.pass_load.contact_ratio for item in self.positions), default=0.0)


def _reference_top_facets(imported: ImportedMesh, width: float, height: float) -> tuple[ReferenceTopFacet, ...]:
    mesh = imported.mesh
    try:
        facet_ids = np.asarray(mesh.boundaries["contact"], dtype=np.int32).reshape(-1)
    except KeyError as exc:
        raise MovingLoadPassWorkflowError("reference mesh has no contact boundary") from exc
    tolerance = max(1.0e-14, 1.0e-10 * max(width, height))
    facets: list[ReferenceTopFacet] = []
    for facet_id in facet_ids:
        nodes = mesh.facets[:, facet_id]
        points = mesh.p[:, nodes].T
        order = np.argsort(points[:, 0], kind="stable")
        first_node = int(nodes[order[0]])
        second_node = int(nodes[order[1]])
        first = mesh.p[:, first_node]
        second = mesh.p[:, second_node]
        if not math.isclose(float(first[1]), height, rel_tol=0.0, abs_tol=tolerance) or not math.isclose(
            float(second[1]), height, rel_tol=0.0, abs_tol=tolerance
        ):
            raise MovingLoadPassWorkflowError("contact facet is not on the fixed reference top edge")
        facets.append(
            ReferenceTopFacet(
                facet_id=int(facet_id),
                node_start_id=first_node,
                node_end_id=second_node,
                x_start_m=float(first[0]),
                y_start_m=float(first[1]),
                x_end_m=float(second[0]),
                y_end_m=float(second[1]),
            )
        )
    facets.sort(key=lambda item: (item.x_start_m, item.x_end_m, item.facet_id))
    if not facets or not math.isclose(facets[0].x_start_m, 0.0, rel_tol=0.0, abs_tol=tolerance) or not math.isclose(
        facets[-1].x_end_m, width, rel_tol=0.0, abs_tol=tolerance
    ):
        raise MovingLoadPassWorkflowError("reference top facets do not cover [0, width]")
    for first, second in zip(facets, facets[1:]):
        if not math.isclose(first.x_end_m, second.x_start_m, rel_tol=0.0, abs_tol=tolerance):
            raise MovingLoadPassWorkflowError("reference top facets are not continuous")
    return tuple(facets)


def _pass_case_at_position(case: FixedMeshMovingLoadPassCase, position: MovingLoadPosition):
    pass_case = case.reference_case.pass_load
    trajectory = pass_case.trajectory
    return replace(
        pass_case,
        trajectory=replace(
            trajectory,
            single_pass=replace(
                trajectory.single_pass,
                wheel_lowest_point_x_m=position.wheel_lowest_point_x_m,
            ),
        ),
    )


def solve_fixed_mesh_moving_load_pass(
    case: FixedMeshMovingLoadPassCase,
    workspace: str | Path,
    *,
    operations: MovingLoadPassOperations | None = None,
) -> FixedMeshMovingLoadPassResult:
    if not isinstance(case, FixedMeshMovingLoadPassCase):
        raise TypeError("case must be a FixedMeshMovingLoadPassCase")
    ops = operations or default_operations()
    workspace_path = Path(workspace).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    msh_path = (workspace_path / "reference_mesh.msh").resolve()
    reference_pass_load = ops.build_pass_load(case.reference_case.pass_load)
    positions = generate_moving_load_positions(case, reference_pass_load)
    ops.generate_mesh(case, msh_path, top_split_x_m=())
    imported = ops.import_mesh(msh_path, case)
    top_facets = _reference_top_facets(imported, case.geometry.width, case.geometry.height)
    results: list[MovingLoadPositionResult] = []
    for position in positions:
        current = ops.build_pass_load(_pass_case_at_position(case, position))
        if current.pass_state != position.pass_state:
            raise MovingLoadPassWorkflowError("position state disagrees with Phase 4C2")
        mapping = map_pass_load_to_fixed_top_facets(current, top_facets)
        results.append(
            MovingLoadPositionResult(
                position=position,
                pass_load=current,
                load_mapping=mapping,
            )
        )
    if results[0].pass_load.effective_contact_length_m != 0.0 or results[-1].pass_load.effective_contact_length_m != 0.0:
        raise MovingLoadPassWorkflowError("complete-pass endpoint loads must be zero")
    return FixedMeshMovingLoadPassResult(
        case=case,
        reference_pass_load=reference_pass_load,
        reference_mesh=imported,
        top_facets=top_facets,
        positions=tuple(results),
        msh_source=msh_path,
    )
