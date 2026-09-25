"""Map one public Phase 4C2 load snapshot to exact parts of fixed top facets."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae.pass_loading import PassLoadResult

from .overlap import ReferenceTopFacet, interval_overlap


class MovingLoadMappingError(RuntimeError):
    """Raised when partial-facet mapping does not conserve projected force."""


def _close(actual: float, expected: float) -> bool:
    scale = max(abs(actual), abs(expected), 1.0)
    return math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12 * scale)


@dataclass(frozen=True, slots=True)
class FixedTopFacetLoad:
    facet_id: int
    node_start_id: int
    node_end_id: int
    facet_x_start_m: float
    facet_x_end_m: float
    facet_dx_m: float
    facet_ds_m: float
    overlap_x_start_m: float
    overlap_x_end_m: float
    overlap_dx_m: float
    overlap_ds_m: float
    qx_projected_N_per_m: float
    qy_projected_N_per_m: float
    tx_applied_per_loaded_ds_N_per_m: float
    ty_applied_per_loaded_ds_N_per_m: float
    integrated_Fx_N: float
    integrated_Fy_N: float
    is_partial_facet: bool


@dataclass(frozen=True, slots=True)
class FixedTopLoadMapping:
    pass_load: PassLoadResult
    facets: tuple[FixedTopFacetLoad, ...]
    projected_overlap_sum_m: float
    loaded_actual_length_sum_m: float
    assembled_Fx_N: float
    assembled_Fy_N: float
    force_balance_residual_x_N: float
    force_balance_residual_y_N: float
    force_balance_residual_N: float

    @property
    def active_facet_count(self) -> int:
        return len(self.facets)

    @property
    def partial_facet_count(self) -> int:
        return sum(item.is_partial_facet for item in self.facets)


def _map_facet(facet: ReferenceTopFacet, pass_load: PassLoadResult) -> FixedTopFacetLoad | None:
    interval = pass_load.effective_contact_interval_m
    if interval is None:
        return None
    overlap = interval_overlap(facet.x_start_m, facet.x_end_m, *interval)
    if overlap is None:
        return None
    overlap_dx = overlap[1] - overlap[0]
    overlap_ds = facet.ds_actual_m * overlap_dx / facet.dx_projected_m
    if overlap_dx <= 0.0 or overlap_ds <= 0.0:
        raise MovingLoadMappingError("positive interval overlap produced invalid length")
    tx = pass_load.current_qx_N_per_m * overlap_dx / overlap_ds
    ty = pass_load.current_qy_N_per_m * overlap_dx / overlap_ds
    integrated_x = tx * overlap_ds
    integrated_y = ty * overlap_ds
    expected_x = pass_load.current_qx_N_per_m * overlap_dx
    expected_y = pass_load.current_qy_N_per_m * overlap_dx
    if not _close(integrated_x, expected_x) or not _close(integrated_y, expected_y):
        raise MovingLoadMappingError("per-facet t ds does not equal q overlap_dx")
    return FixedTopFacetLoad(
        facet_id=facet.facet_id,
        node_start_id=facet.node_start_id,
        node_end_id=facet.node_end_id,
        facet_x_start_m=facet.x_start_m,
        facet_x_end_m=facet.x_end_m,
        facet_dx_m=facet.dx_projected_m,
        facet_ds_m=facet.ds_actual_m,
        overlap_x_start_m=overlap[0],
        overlap_x_end_m=overlap[1],
        overlap_dx_m=overlap_dx,
        overlap_ds_m=overlap_ds,
        qx_projected_N_per_m=pass_load.current_qx_N_per_m,
        qy_projected_N_per_m=pass_load.current_qy_N_per_m,
        tx_applied_per_loaded_ds_N_per_m=tx,
        ty_applied_per_loaded_ds_N_per_m=ty,
        integrated_Fx_N=integrated_x,
        integrated_Fy_N=integrated_y,
        is_partial_facet=not _close(overlap_dx, facet.dx_projected_m),
    )


def map_pass_load_to_fixed_top_facets(
    pass_load: PassLoadResult,
    top_facets: tuple[ReferenceTopFacet, ...],
) -> FixedTopLoadMapping:
    if not isinstance(pass_load, PassLoadResult):
        raise TypeError("pass_load must be a PassLoadResult")
    if not all(isinstance(item, ReferenceTopFacet) for item in top_facets):
        raise TypeError("top_facets must contain ReferenceTopFacet values")
    mapped = tuple(
        item
        for facet in top_facets
        if (item := _map_facet(facet, pass_load)) is not None
    )
    projected = math.fsum(item.overlap_dx_m for item in mapped)
    loaded_actual = math.fsum(item.overlap_ds_m for item in mapped)
    assembled_x = math.fsum(item.integrated_Fx_N for item in mapped)
    assembled_y = math.fsum(item.integrated_Fy_N for item in mapped)
    residual_x = assembled_x - pass_load.current_Fx_N
    residual_y = assembled_y - pass_load.current_Fy_N
    residual = math.hypot(residual_x, residual_y)
    if not _close(projected, pass_load.effective_contact_length_m):
        raise MovingLoadMappingError(
            "sum of exact top-facet overlaps does not equal Phase 4C2 effective contact length"
        )
    if not _close(assembled_x, pass_load.current_Fx_N):
        raise MovingLoadMappingError("partial-facet loads do not reconstruct current Fx")
    if not _close(assembled_y, pass_load.current_Fy_N):
        raise MovingLoadMappingError("partial-facet loads do not reconstruct current Fy")
    return FixedTopLoadMapping(
        pass_load=pass_load,
        facets=mapped,
        projected_overlap_sum_m=projected,
        loaded_actual_length_sum_m=loaded_actual,
        assembled_Fx_N=assembled_x,
        assembled_Fy_N=assembled_y,
        force_balance_residual_x_N=residual_x,
        force_balance_residual_y_N=residual_y,
        force_balance_residual_N=residual,
    )
