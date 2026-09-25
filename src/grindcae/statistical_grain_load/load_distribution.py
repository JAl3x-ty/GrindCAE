"""Conservative nonuniform continuous mechanism line loads."""

from __future__ import annotations

from dataclasses import dataclass
import math

from grindcae.mechanism_force import MechanismForcePrediction

from .grain_population import EquivalentGrainGroup
from .models import LoadDistributionSettings


@dataclass(frozen=True, slots=True)
class MechanismLoadPoint:
    x_local_m: float
    normalized_x: float
    q_t_rubbing_N_per_m: float
    q_t_ploughing_N_per_m: float
    q_t_cutting_N_per_m: float
    q_t_total_N_per_m: float
    q_n_rubbing_N_per_m: float
    q_n_ploughing_N_per_m: float
    q_n_cutting_N_per_m: float
    q_n_total_N_per_m: float

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def trapezoidal_integral(values: tuple[float, ...], x: tuple[float, ...]) -> float:
    return math.fsum(
        0.5 * (left + right) * (x_right - x_left)
        for left, right, x_left, x_right in zip(values, values[1:], x, x[1:])
    )


def _normalized_load(
    x: tuple[float, ...],
    groups: tuple[EquivalentGrainGroup, ...],
    weight_name: str,
    kernel_scale: float,
    target_force_N: float,
) -> tuple[float, ...]:
    shape = []
    for coordinate in x:
        shape.append(
            math.fsum(
                getattr(group, weight_name)
                * math.exp(-0.5 * ((coordinate - group.contact_position_m) / (kernel_scale * group.base_kernel_width_m)) ** 2)
                for group in groups
            )
        )
    shape_tuple = tuple(shape)
    integral = trapezoidal_integral(shape_tuple, x)
    if integral <= 0.0 or not math.isfinite(integral):
        raise RuntimeError("mechanism load kernel produced a non-positive finite integral")
    scale = target_force_N / integral
    return tuple(value * scale for value in shape_tuple)


def build_load_distribution(
    mechanism: MechanismForcePrediction,
    groups: tuple[EquivalentGrainGroup, ...],
    settings: LoadDistributionSettings,
    contact_length_m: float,
) -> tuple[MechanismLoadPoint, ...]:
    x = tuple(contact_length_m * index / (settings.point_count - 1) for index in range(settings.point_count))
    specifications = (
        ("rubbing", settings.rubbing_kernel_scale),
        ("ploughing", settings.ploughing_kernel_scale),
        ("cutting", settings.cutting_kernel_scale),
    )
    values: dict[str, tuple[float, ...]] = {}
    for name, kernel_scale in specifications:
        component = getattr(mechanism, name)
        values[f"q_t_{name}_N_per_m"] = _normalized_load(
            x, groups, f"{name}_weight", kernel_scale, component.tangential_force_N
        )
        values[f"q_n_{name}_N_per_m"] = _normalized_load(
            x, groups, f"{name}_weight", kernel_scale, component.normal_force_N
        )
    points: list[MechanismLoadPoint] = []
    for index, coordinate in enumerate(x):
        tangential = [values[f"q_t_{name}_N_per_m"][index] for name, _ in specifications]
        normal = [values[f"q_n_{name}_N_per_m"][index] for name, _ in specifications]
        points.append(
            MechanismLoadPoint(
                x_local_m=coordinate,
                normalized_x=coordinate / contact_length_m,
                q_t_rubbing_N_per_m=tangential[0],
                q_t_ploughing_N_per_m=tangential[1],
                q_t_cutting_N_per_m=tangential[2],
                q_t_total_N_per_m=math.fsum(tangential),
                q_n_rubbing_N_per_m=normal[0],
                q_n_ploughing_N_per_m=normal[1],
                q_n_cutting_N_per_m=normal[2],
                q_n_total_N_per_m=math.fsum(normal),
            )
        )
    return tuple(points)
