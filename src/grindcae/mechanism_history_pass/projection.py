"""Exact conservative projection of the 7A.2 piecewise-linear template."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from grindcae.elastoplastic_fem import PreparedElastoplasticMesh
from grindcae.moving_load_pass import MovingLoadPositionResult
from grindcae.statistical_grain_load import StatisticalGrainLoadPrediction
from grindcae.trajectory import motion_coordinate_from_physical_x


MECHANISMS = ("rubbing", "ploughing", "cutting")


@dataclass(frozen=True, slots=True)
class MechanismProjection:
    target_load_vector_N: np.ndarray
    mechanism_forces_N: dict[str, float]
    mechanism_residuals_N: dict[str, float]
    total_Fx_N: float
    total_Fy_N: float


def _linear(x: float, xp: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(x, xp, values))


def project_statistical_mechanism_load(
    prepared: PreparedElastoplasticMesh,
    position: MovingLoadPositionResult,
    prediction: StatisticalGrainLoadPrediction,
) -> MechanismProjection:
    """Integrate split template segments times P1 edge functions exactly by Gauss-2."""
    vector = np.zeros(prepared.total_degrees_of_freedom, dtype=float)
    result: dict[str, float] = {}
    residuals: dict[str, float] = {}
    interval = position.pass_load.effective_contact_interval_m
    if interval is None:
        for mechanism in MECHANISMS:
            result[f"Ft_{mechanism}_N"] = result[f"Fn_{mechanism}_N"] = 0.0
            residuals[f"Ft_{mechanism}_N"] = residuals[f"Fn_{mechanism}_N"] = 0.0
        return MechanismProjection(vector, result, residuals, 0.0, 0.0)
    trajectory = position.pass_load.trajectory_result
    width = trajectory.case.workpiece.length_m; arc = trajectory.exact_arc_projected_length_m
    direction = trajectory.case.single_pass.relative_feed_direction
    lowest_motion = position.position.motion_coordinate_m
    x0, x1 = interval
    def u_at(x: float) -> float:
        return (motion_coordinate_from_physical_x(x, width, direction) - lowest_motion) / arc
    ua, ub = sorted((u_at(x0), u_at(x1)))
    grid_u = np.asarray([point.normalized_x for point in prediction.distribution], dtype=float)
    ratio = position.pass_load.contact_ratio
    sign = 1.0 if position.pass_load.current_Fx_N >= 0.0 else -1.0
    gauss = ((0.5 - 0.5 / math.sqrt(3.0), 0.5), (0.5 + 0.5 / math.sqrt(3.0), 0.5))
    for direction_name, q_prefix, force_name, force_sign in (("t", "q_t", "tangential_force_N", sign), ("n", "q_n", "normal_force_N", -1.0)):
        for mechanism in MECHANISMS:
            template = np.asarray([getattr(point, f"{q_prefix}_{mechanism}_N_per_m") for point in prediction.distribution], dtype=float)
            target = ratio * getattr(getattr(prediction.mechanism_prediction, mechanism), force_name)
            knots = np.unique(np.concatenate(([ua, ub], grid_u[(grid_u > ua) & (grid_u < ub)])))
            integral_u = sum(0.5 * (_linear(a, grid_u, template) + _linear(b, grid_u, template)) * (b - a) for a, b in zip(knots, knots[1:]))
            scale = 0.0 if target == 0.0 else target / (arc * integral_u)
            assembled = 0.0
            for facet in position.load_mapping.facets:
                left = max(facet.overlap_x_start_m, x0); right = min(facet.overlap_x_end_m, x1)
                if right <= left: continue
                cuts = [left, right]
                for knot in knots[1:-1]:
                    # u is affine in x; solve with the two endpoint values.
                    ux0, ux1 = u_at(left), u_at(right)
                    if min(ux0, ux1) < knot < max(ux0, ux1):
                        cuts.append(left + (knot - ux0) * (right - left) / (ux1 - ux0))
                for a, b in zip(sorted(cuts), sorted(cuts)[1:]):
                    for xi, weight in gauss:
                        x = a + xi * (b - a); q = scale * _linear(u_at(x), grid_u, template)
                        local = (x - facet.facet_x_start_m) / facet.facet_dx_m
                        factor = force_sign * q * weight * (b - a)
                        vector[prepared.component_dofs[0 if direction_name == "t" else 1, facet.node_start_id]] += factor * (1.0 - local)
                        vector[prepared.component_dofs[0 if direction_name == "t" else 1, facet.node_end_id]] += factor * local
                        assembled += factor
            key = f"F{'t' if direction_name == 't' else 'n'}_{mechanism}_N"
            result[key] = target
            residuals[key] = assembled - force_sign * target
    fx = float(np.sum(vector[prepared.component_dofs[0]])); fy = float(np.sum(vector[prepared.component_dofs[1]]))
    return MechanismProjection(vector, result, residuals, fx, fy)
