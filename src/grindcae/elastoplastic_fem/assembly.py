"""Constant-strain P1 triangle operations for Phase 6A.2."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from grindcae.elastoplastic import (
    BilinearIsotropicMaterial,
    MaterialPointState,
    PlaneStrainJ2Update,
    update_plane_strain_j2,
)


class ElastoplasticAssemblyError(ValueError):
    """Raised when element geometry or trial assembly data are invalid."""


@dataclass(frozen=True, slots=True)
class ElementTrialResponse:
    total_strain_increment: NDArray[np.float64]
    material_update: PlaneStrainJ2Update
    internal_force_N: NDArray[np.float64]
    tangent_stiffness_N_per_m: NDArray[np.float64]


def triangle_B_matrix(
    node_coordinates: object,
) -> tuple[NDArray[np.float64], float]:
    """Return the engineering-strain B matrix and positive triangle area."""

    try:
        nodes = np.asarray(node_coordinates, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ElastoplasticAssemblyError(
            "triangle node coordinates must be finite numeric values"
        ) from exc
    if nodes.shape != (3, 2):
        raise ElastoplasticAssemblyError(
            f"triangle node coordinates must have shape (3, 2), got {nodes.shape}"
        )
    if not np.all(np.isfinite(nodes)):
        raise ElastoplasticAssemblyError("triangle node coordinates must be finite")
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]
    twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    scale = max(1.0, float(np.max(np.abs(nodes))))
    if abs(twice_area) <= 64.0 * math.ulp(scale * scale):
        raise ElastoplasticAssemblyError("triangle area must be strictly positive")
    area = 0.5 * abs(float(twice_area))
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float) / twice_area
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float) / twice_area
    B = np.array(
        [
            [b[0], 0.0, b[1], 0.0, b[2], 0.0],
            [0.0, c[0], 0.0, c[1], 0.0, c[2]],
            [c[0], b[0], c[1], b[1], c[2], b[2]],
        ],
        dtype=float,
    )
    return B, area


def element_trial_response(
    material: BilinearIsotropicMaterial,
    committed_state: MaterialPointState,
    B: object,
    area_m2: object,
    *,
    thickness_m: object,
    displacement_increment: object,
) -> ElementTrialResponse:
    """Evaluate one element from its last committed material state."""

    matrix = np.asarray(B, dtype=float)
    displacement = np.asarray(displacement_increment, dtype=float)
    if matrix.shape != (3, 6) or displacement.shape != (6,):
        raise ElastoplasticAssemblyError(
            "element B matrix and displacement increment must have shapes (3, 6) and (6,)"
        )
    area = float(area_m2)
    thickness = float(thickness_m)
    if not math.isfinite(area) or area <= 0.0:
        raise ElastoplasticAssemblyError("element area must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ElastoplasticAssemblyError("analysis thickness must be finite and positive")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(displacement)):
        raise ElastoplasticAssemblyError("element trial inputs must be finite")
    strain_increment = matrix @ displacement
    update = update_plane_strain_j2(material, committed_state, strain_increment)
    stress = np.asarray(update.in_plane_stress_Pa, dtype=float)
    tangent = np.asarray(update.algorithmic_tangent_Pa, dtype=float)
    weight = area * thickness
    return ElementTrialResponse(
        total_strain_increment=strain_increment,
        material_update=update,
        internal_force_N=matrix.T @ stress * weight,
        tangent_stiffness_N_per_m=matrix.T @ tangent @ matrix * weight,
    )
