"""Convert phase 4C2 projected loads to phase 4C3A facet tractions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from grindcae.evolved_mesh import EvolvedMeshValidationResult, TopBoundaryFacet
from grindcae.pass_loading import PassLoadResult


class EvolvedFacetLoadError(RuntimeError):
    """Raised when projected and actual-length load measures are inconsistent."""


def _close(actual: float, expected: float) -> bool:
    scale = max(abs(actual), abs(expected), 1.0)
    return math.isclose(actual, expected, rel_tol=1.0e-11, abs_tol=1.0e-12 * scale)


@dataclass(frozen=True, slots=True)
class EvolvedActiveFacetLoad:
    """One contact-arc facet with projection-preserving global traction."""

    facet_id: int
    node_start_id: int
    node_end_id: int
    region: str
    x_start_m: float
    y_start_m: float
    x_end_m: float
    y_end_m: float
    dx_projected_m: float
    ds_actual_m: float
    dx_over_ds: float
    qx_projected_N_per_m: float
    qy_projected_N_per_m: float
    tx_applied_per_ds_N_per_m: float
    ty_applied_per_ds_N_per_m: float
    integrated_Fx_N: float
    integrated_Fy_N: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            name: (
                getattr(self, name).item()
                if isinstance(getattr(self, name), np.generic)
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class EvolvedFacetLoadResult:
    """Validated contact_arc selection and q dx to t ds conversion."""

    pass_load: PassLoadResult
    mesh: EvolvedMeshValidationResult
    facets: tuple[EvolvedActiveFacetLoad, ...]
    projected_length_sum_m: float
    actual_facet_length_sum_m: float
    integrated_Fx_N: float
    integrated_Fy_N: float

    @property
    def facet_ids(self) -> NDArray[np.int32]:
        return np.asarray([item.facet_id for item in self.facets], dtype=np.int32)

    @property
    def tx_per_ds(self) -> NDArray[np.float64]:
        return np.asarray(
            [item.tx_applied_per_ds_N_per_m for item in self.facets], dtype=float
        )

    @property
    def ty_per_ds(self) -> NDArray[np.float64]:
        return np.asarray(
            [item.ty_applied_per_ds_N_per_m for item in self.facets], dtype=float
        )


def _convert_facet(
    facet: TopBoundaryFacet,
    pass_load: PassLoadResult,
    node_count: int,
) -> EvolvedActiveFacetLoad:
    if facet.region != "contact_arc":
        raise EvolvedFacetLoadError("only contact_arc facets may be loaded")
    if not (0 <= facet.node_start_id < node_count and 0 <= facet.node_end_id < node_count):
        raise EvolvedFacetLoadError("active facet references a node outside the final MSH")
    dx = facet.projected_length_x_m
    ds = facet.actual_facet_length_m
    if not math.isfinite(dx) or not math.isfinite(ds) or dx <= 0.0 or ds <= 0.0:
        raise EvolvedFacetLoadError("active facet dx and ds must be finite and positive")
    ratio = dx / ds
    if not (0.0 < ratio <= 1.0 + 1.0e-12):
        raise EvolvedFacetLoadError("active facet dx_over_ds must lie in (0, 1]")
    ratio = min(ratio, 1.0)
    # 中文导读：按 dx/ds 换算到实际边单元（弦长 ds），逐分量保持 q*dx = t*ds。
    tx = pass_load.current_qx_N_per_m * ratio
    ty = pass_load.current_qy_N_per_m * ratio
    integrated_x = tx * ds
    integrated_y = ty * ds
    expected_x = pass_load.current_qx_N_per_m * dx
    expected_y = pass_load.current_qy_N_per_m * dx
    if not _close(integrated_x, expected_x) or not _close(integrated_y, expected_y):
        raise EvolvedFacetLoadError("per-facet q dx and t ds force integrals disagree")
    return EvolvedActiveFacetLoad(
        facet_id=facet.facet_id,
        node_start_id=facet.node_start_id,
        node_end_id=facet.node_end_id,
        region=facet.region,
        x_start_m=facet.x_start_m,
        y_start_m=facet.y_start_m,
        x_end_m=facet.x_end_m,
        y_end_m=facet.y_end_m,
        dx_projected_m=dx,
        ds_actual_m=ds,
        dx_over_ds=ratio,
        qx_projected_N_per_m=pass_load.current_qx_N_per_m,
        qy_projected_N_per_m=pass_load.current_qy_N_per_m,
        tx_applied_per_ds_N_per_m=tx,
        ty_applied_per_ds_N_per_m=ty,
        integrated_Fx_N=integrated_x,
        integrated_Fy_N=integrated_y,
    )


def build_evolved_facet_load(
    pass_load: PassLoadResult,
    mesh: EvolvedMeshValidationResult,
) -> EvolvedFacetLoadResult:
    """Select exact contact_arc facets and preserve projected force integrals."""

    if pass_load.trajectory_result != mesh.trajectory:
        raise EvolvedFacetLoadError("pass load and evolved mesh must reuse one trajectory result")
    selected = tuple(
        _convert_facet(facet, pass_load, mesh.node_count)
        for facet in mesh.top_boundary_facets
        if facet.region == "contact_arc"
    )
    expected_length = pass_load.effective_contact_length_m
    if expected_length == 0.0 and selected:
        raise EvolvedFacetLoadError("zero contact must not select active facets")
    if expected_length > 0.0 and not selected:
        raise EvolvedFacetLoadError("nonzero contact requires contact_arc facets")
    projected = math.fsum(item.dx_projected_m for item in selected)
    actual = math.fsum(item.ds_actual_m for item in selected)
    integrated_x = math.fsum(item.integrated_Fx_N for item in selected)
    integrated_y = math.fsum(item.integrated_Fy_N for item in selected)
    if not _close(projected, expected_length):
        raise EvolvedFacetLoadError("sum(dx_projected) does not equal phase 4C2 Leff")
    if not _close(integrated_x, pass_load.current_Fx_N):
        raise EvolvedFacetLoadError("facet tractions do not reconstruct current Fx")
    if not _close(integrated_y, pass_load.current_Fy_N):
        raise EvolvedFacetLoadError("facet tractions do not reconstruct current Fy")
    return EvolvedFacetLoadResult(
        pass_load=pass_load,
        mesh=mesh,
        facets=selected,
        projected_length_sum_m=projected,
        actual_facet_length_sum_m=actual,
        integrated_Fx_N=integrated_x,
        integrated_Fy_N=integrated_y,
    )
