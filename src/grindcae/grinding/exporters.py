"""Export traceable runtime data for schema 4 local grinding loads."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from grindcae.solver import LocalLoadFullFieldSolution


ACTIVE_CONTACT_FACET_CSV_FIELDS = (
    "facet_id",
    "node_0",
    "node_1",
    "x0_m",
    "y0_m",
    "x1_m",
    "y1_m",
    "length_m",
    "qx_N_per_m",
    "qy_N_per_m",
)


def _active_contact_rows(
    solution: LocalLoadFullFieldSolution,
) -> list[tuple[int | float, ...]]:
    if not isinstance(solution, LocalLoadFullFieldSolution):
        raise TypeError("solution must be a LocalLoadFullFieldSolution")
    solution.validate_displacement_consistency()

    mesh = solution.mesh
    facets = np.asarray(solution.active_facets, dtype=np.int32)
    if facets.ndim != 1 or facets.size == 0:
        raise ValueError("active facets must be a non-empty one-dimensional array")

    summary = solution.solver_summary
    rows: list[tuple[int | float, ...]] = []
    for facet_id in facets:
        nodes = np.asarray(mesh.facets[:, int(facet_id)], dtype=np.int32)
        coordinates = np.asarray(mesh.p[:, nodes].T, dtype=float)
        endpoint_order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
        sorted_nodes = nodes[endpoint_order]
        sorted_coordinates = coordinates[endpoint_order]
        length = float(np.linalg.norm(sorted_coordinates[1] - sorted_coordinates[0]))
        values = (
            float(sorted_coordinates[0, 0]),
            float(sorted_coordinates[0, 1]),
            float(sorted_coordinates[1, 0]),
            float(sorted_coordinates[1, 1]),
            length,
            float(summary.line_load_x),
            float(summary.line_load_y),
        )
        if not all(math.isfinite(value) for value in values) or length <= 0.0:
            raise ValueError("active contact facet rows must be finite and positive")
        rows.append(
            (
                int(facet_id),
                int(sorted_nodes[0]),
                int(sorted_nodes[1]),
                *values,
            )
        )

    rows.sort(key=lambda row: (float(row[3]), float(row[5]), int(row[0])))
    total_length = float(math.fsum(float(row[7]) for row in rows))
    integrated_x = float(
        math.fsum(float(row[7]) * float(row[8]) for row in rows)
    )
    integrated_y = float(
        math.fsum(float(row[7]) * float(row[9]) for row in rows)
    )
    if not math.isclose(
        total_length,
        summary.active_facet_length,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("active facet CSV length does not match the solved length")
    if not np.allclose(
        (integrated_x, integrated_y),
        (summary.mapped_force_x, summary.mapped_force_y),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError("active facet CSV line loads do not integrate to mapped force")
    return rows


def write_active_contact_facets_csv(
    solution: LocalLoadFullFieldSolution,
    output_path: str | Path,
) -> Path:
    """Write the exact 0-based runtime facets carrying the local line load."""

    rows = _active_contact_rows(solution)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ACTIVE_CONTACT_FACET_CSV_FIELDS)
        writer.writerows(rows)
    return path
