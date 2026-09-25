"""Exact interval overlap on the imported fixed reference top boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math


class MovingLoadOverlapError(RuntimeError):
    """Raised when an imported reference top facet is geometrically invalid."""


def interval_overlap(
    first_start_m: float,
    first_end_m: float,
    second_start_m: float,
    second_end_m: float,
) -> tuple[float, float] | None:
    start = max(float(first_start_m), float(second_start_m))
    end = min(float(first_end_m), float(second_end_m))
    return (start, end) if end > start else None


@dataclass(frozen=True, slots=True)
class ReferenceTopFacet:
    facet_id: int
    node_start_id: int
    node_end_id: int
    x_start_m: float
    y_start_m: float
    x_end_m: float
    y_end_m: float

    def __post_init__(self) -> None:
        if type(self.facet_id) is not int or self.facet_id < 0:
            raise MovingLoadOverlapError("facet_id must be a non-negative integer")
        if type(self.node_start_id) is not int or type(self.node_end_id) is not int:
            raise MovingLoadOverlapError("top-facet node IDs must be integers")
        values = (self.x_start_m, self.y_start_m, self.x_end_m, self.y_end_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise MovingLoadOverlapError("top-facet coordinates must be finite")
        if self.x_end_m <= self.x_start_m:
            raise MovingLoadOverlapError("top facet must have positive projected x length")
        if self.ds_actual_m <= 0.0:
            raise MovingLoadOverlapError("top facet must have positive actual length")

    @property
    def dx_projected_m(self) -> float:
        return self.x_end_m - self.x_start_m

    @property
    def ds_actual_m(self) -> float:
        return math.hypot(self.x_end_m - self.x_start_m, self.y_end_m - self.y_start_m)
