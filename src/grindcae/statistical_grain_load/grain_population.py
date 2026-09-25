"""Deterministic physical-grain estimates and equivalent groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .grit_resolution import ResolvedGritSpecification
from .models import GrainPopulationSettings, LoadDistributionSettings


@dataclass(frozen=True, slots=True)
class EquivalentGrainGroup:
    group_id: int
    normalized_contact_position: float
    contact_position_m: float
    representative_diameter_m: float
    represented_physical_grain_count: float
    rubbing_weight: float
    ploughing_weight: float
    cutting_weight: float
    base_kernel_width_m: float

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def equivalent_group_count(estimated_count: float, settings: GrainPopulationSettings) -> int:
    if estimated_count < 1.0:
        return 1
    raw = math.ceil(math.sqrt(max(estimated_count, 1.0)))
    return min(
        settings.maximum_equivalent_group_count,
        math.ceil(estimated_count),
        max(settings.minimum_equivalent_group_count, raw),
    )


def _diameter_quantiles(specification: ResolvedGritSpecification, count: int) -> list[float]:
    values: list[float] = []
    lower = specification.diameter_lower_m
    d50 = specification.representative_diameter_d50_m
    upper = specification.diameter_upper_m
    for index in range(count):
        quantile = (index + 0.5) / count
        if quantile <= 0.5:
            t = quantile / 0.5
            diameter = math.exp(math.log(lower) + t * (math.log(d50) - math.log(lower)))
        else:
            t = (quantile - 0.5) / 0.5
            diameter = math.exp(math.log(d50) + t * (math.log(upper) - math.log(d50)))
        values.append(diameter)
    return values


def build_equivalent_grain_groups(
    specification: ResolvedGritSpecification,
    settings: GrainPopulationSettings,
    distribution: LoadDistributionSettings,
    contact_length_m: float,
    estimated_count: float,
) -> tuple[EquivalentGrainGroup, ...]:
    count = equivalent_group_count(estimated_count, settings)
    rng = random.Random(settings.random_seed)
    positions = sorted((index + rng.random()) / count for index in range(count))
    diameters = _diameter_quantiles(specification, count)
    rng.shuffle(diameters)
    represented = estimated_count / count
    raw_weights = {
        "rubbing": [1.0 for _ in diameters],
        "ploughing": list(diameters),
        "cutting": [diameter * diameter for diameter in diameters],
    }
    normalized = {
        name: [value / math.fsum(values) for value in values]
        for name, values in raw_weights.items()
    }
    groups: list[EquivalentGrainGroup] = []
    for index, (position, diameter) in enumerate(zip(positions, diameters, strict=True), start=1):
        groups.append(
            EquivalentGrainGroup(
                group_id=index,
                normalized_contact_position=position,
                contact_position_m=position * contact_length_m,
                representative_diameter_m=diameter,
                represented_physical_grain_count=represented,
                rubbing_weight=normalized["rubbing"][index - 1],
                ploughing_weight=normalized["ploughing"][index - 1],
                cutting_weight=normalized["cutting"][index - 1],
                base_kernel_width_m=max(
                    distribution.kernel_width_factor * diameter,
                    contact_length_m / (2.0 * count),
                ),
            )
        )
    return tuple(groups)
