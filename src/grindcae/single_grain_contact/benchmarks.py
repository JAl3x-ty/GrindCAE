"""Hertz reference quantities and deterministic contact parameter studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from skfem import MeshTri

from grindcae.solver import ImportedMesh

from .models import SingleGrainContactCase
from .workflow import ContactTrajectoryResult, run_contact_trajectory_on_mesh


@dataclass(frozen=True, slots=True)
class HertzReference:
    force_per_thickness_N_per_m: float
    effective_modulus_Pa: float
    radius_m: float
    contact_half_width_m: float
    maximum_pressure_Pa: float


@dataclass(frozen=True, slots=True)
class HertzComparison:
    reference: HertzReference
    numerical_contact_half_width_m: float
    numerical_maximum_pressure_Pa: float
    half_width_relative_error: float
    maximum_pressure_relative_error: float
    pressure_shape_l2_relative_error: float
    symmetry_relative_error: float


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    surface_node_count: int
    normal_penalty_factor: float
    force_per_thickness_N_per_m: float
    maximum_penetration_m: float
    maximum_penetration_over_local_size: float
    newton_iterations: int
    comparison: HertzComparison


@dataclass(frozen=True, slots=True)
class FiniteDomainPoint:
    width_ratio: float
    depth_ratio: float
    width_m: float
    depth_m: float
    surface_node_count: int
    force_per_thickness_N_per_m: float
    maximum_penetration_m: float
    newton_iterations: int
    comparison: HertzComparison


def hertz_line_contact_reference(
    *, force_per_thickness_N_per_m: float, effective_modulus_Pa: float, radius_m: float
) -> HertzReference:
    if force_per_thickness_N_per_m <= 0.0 or effective_modulus_Pa <= 0.0 or radius_m <= 0.0:
        raise ValueError("Hertz inputs must be finite and positive")
    if not all(math.isfinite(v) for v in (force_per_thickness_N_per_m, effective_modulus_Pa, radius_m)):
        raise ValueError("Hertz inputs must be finite and positive")
    half_width = math.sqrt(
        4.0 * force_per_thickness_N_per_m * radius_m
        / (math.pi * effective_modulus_Pa)
    )
    pressure = 2.0 * force_per_thickness_N_per_m / (math.pi * half_width)
    return HertzReference(
        force_per_thickness_N_per_m=force_per_thickness_N_per_m,
        effective_modulus_Pa=effective_modulus_Pa,
        radius_m=radius_m,
        contact_half_width_m=half_width,
        maximum_pressure_Pa=pressure,
    )


def compare_hertz(result: ContactTrajectoryResult) -> HertzComparison:
    record = max(result.records, key=lambda item: item.normal_reaction_N)
    reference = hertz_line_contact_reference(
        force_per_thickness_N_per_m=record.normal_reaction_per_thickness_N_per_m,
        effective_modulus_Pa=result.case.effective_modulus_Pa,
        radius_m=result.case.grain.radius_m,
    )
    state = record.state
    coordinates = result.prepared_mesh.candidate_reference_coordinates_m[:, 0]
    pressure = np.asarray(state.contact_pressure_Pa, dtype=float)
    active = pressure > 0.0
    local_size = float(np.median(result.prepared_mesh.candidate_tributary_lengths_m))
    active_x = coordinates[active]
    if active_x.size == 0:
        raise ValueError("Hertz comparison requires active contact")
    active_offset = active_x - result.case.grain.initial_center_x_m
    if active_x.size >= 3:
        slope, intercept = np.polyfit(
            active_offset**2,
            pressure[active] ** 2,
            1,
        )
        fitted_squared = -intercept / slope if slope < 0.0 else -1.0
        numerical_half_width = (
            math.sqrt(fitted_squared)
            if math.isfinite(fitted_squared) and fitted_squared > 0.0
            else 0.5 * float(active_x[-1] - active_x[0]) + 0.5 * local_size
        )
    else:
        numerical_half_width = 0.5 * float(active_x[-1] - active_x[0]) + 0.5 * local_size
    center = result.case.grain.initial_center_x_m
    analytic = reference.maximum_pressure_Pa * np.sqrt(
        np.maximum(0.0, 1.0 - ((active_x - center) / reference.contact_half_width_m) ** 2)
    )
    shape_scale = max(float(np.linalg.norm(analytic)), 1.0)
    shape_error = float(np.linalg.norm(pressure[active] - analytic) / shape_scale)
    reversed_pressure = pressure[::-1]
    symmetry = float(
        np.max(np.abs(pressure - reversed_pressure))
        / max(float(np.max(pressure)), 1.0)
    )
    return HertzComparison(
        reference=reference,
        numerical_contact_half_width_m=numerical_half_width,
        numerical_maximum_pressure_Pa=float(np.max(pressure)),
        half_width_relative_error=abs(numerical_half_width - reference.contact_half_width_m) / reference.contact_half_width_m,
        maximum_pressure_relative_error=abs(float(np.max(pressure)) - reference.maximum_pressure_Pa) / reference.maximum_pressure_Pa,
        pressure_shape_l2_relative_error=shape_error,
        symmetry_relative_error=symmetry,
    )


def tensor_contact_mesh(case: SingleGrainContactCase, surface_node_count: int) -> ImportedMesh:
    if type(surface_node_count) is not int or surface_node_count < 5:
        raise ValueError("surface_node_count must be a strict integer of at least five")
    vertical_count = max(3, int(round((surface_node_count - 1) * case.geometry.height / case.geometry.width)) + 1)
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, case.geometry.width, surface_node_count),
        np.linspace(0.0, case.geometry.height, vertical_count),
    ).with_boundaries(
        {
            "fixed": lambda x: np.isclose(x[1], 0.0),
            "contact": lambda x: np.isclose(x[1], case.geometry.height),
            "free_left": lambda x: np.isclose(x[0], 0.0),
            "free_right": lambda x: np.isclose(x[0], case.geometry.width),
        }
    )
    return ImportedMesh(mesh=mesh, source_path=Path("hertz-tensor-study.msh"))


def run_hertz_sensitivity_study(
    case_factory: object,
    *,
    surface_node_counts: tuple[int, ...] = (31, 41, 61),
    normal_penalty_factors: tuple[float, ...] = (1.0, 10.0, 100.0),
) -> tuple[SensitivityPoint, ...]:
    if not callable(case_factory):
        raise TypeError("case_factory must be callable")
    points: list[SensitivityPoint] = []
    for count in surface_node_counts:
        for factor in normal_penalty_factors:
            case = case_factory(count, factor)
            result = run_contact_trajectory_on_mesh(case, tensor_contact_mesh(case, count))
            record = max(result.records, key=lambda item: item.normal_reaction_N)
            local_size = float(np.median(result.prepared_mesh.candidate_tributary_lengths_m))
            points.append(
                SensitivityPoint(
                    surface_node_count=count,
                    normal_penalty_factor=factor,
                    force_per_thickness_N_per_m=record.normal_reaction_per_thickness_N_per_m,
                    maximum_penetration_m=record.maximum_penetration_m,
                    maximum_penetration_over_local_size=record.maximum_penetration_m / local_size,
                    newton_iterations=record.newton_iterations,
                    comparison=compare_hertz(result),
                )
            )
    return tuple(points)


def run_hertz_finite_domain_study(
    case_factory: object,
    *,
    reference_half_width_m: float,
    domain_ratios: tuple[tuple[float, float], ...] = (
        (10.0, 5.0),
        (20.0, 10.0),
        (40.0, 20.0),
    ),
    surface_step_m: float,
) -> tuple[FiniteDomainPoint, ...]:
    """Run the frozen finite-domain sequence at one physical surface spacing."""

    if not callable(case_factory):
        raise TypeError("case_factory must be callable")
    if (
        not math.isfinite(reference_half_width_m)
        or reference_half_width_m <= 0.0
        or not math.isfinite(surface_step_m)
        or surface_step_m <= 0.0
    ):
        raise ValueError("finite-domain length scales must be finite and positive")
    points: list[FiniteDomainPoint] = []
    for width_ratio, depth_ratio in domain_ratios:
        if (
            not math.isfinite(width_ratio)
            or not math.isfinite(depth_ratio)
            or width_ratio <= 0.0
            or depth_ratio <= 0.0
        ):
            raise ValueError("finite-domain ratios must be finite and positive")
        width_m = width_ratio * reference_half_width_m
        depth_m = depth_ratio * reference_half_width_m
        interval_count = max(4, int(round(width_m / surface_step_m)))
        surface_node_count = interval_count + 1
        case = case_factory(width_m, depth_m, surface_node_count)
        result = run_contact_trajectory_on_mesh(
            case, tensor_contact_mesh(case, surface_node_count)
        )
        record = max(result.records, key=lambda item: item.normal_reaction_N)
        points.append(
            FiniteDomainPoint(
                width_ratio=float(width_ratio),
                depth_ratio=float(depth_ratio),
                width_m=width_m,
                depth_m=depth_m,
                surface_node_count=surface_node_count,
                force_per_thickness_N_per_m=record.normal_reaction_per_thickness_N_per_m,
                maximum_penetration_m=record.maximum_penetration_m,
                newton_iterations=record.newton_iterations,
                comparison=compare_hertz(result),
            )
        )
    return tuple(points)
