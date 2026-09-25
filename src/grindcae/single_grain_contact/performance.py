"""Reproducible Windows performance benchmark for the 3.0 contact route."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .benchmarks import tensor_contact_mesh
from .models import SingleGrainContactCase
from .workflow import run_contact_trajectory_on_mesh


@dataclass(frozen=True, slots=True)
class ContactPerformanceBenchmark:
    case: SingleGrainContactCase
    surface_node_count: int
    estimated_vector_p1_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class ContactPerformanceResult:
    surface_node_count: int
    actual_vector_p1_degrees_of_freedom: int
    trajectory_position_count: int
    elapsed_seconds: float
    peak_working_set_bytes: int
    maximum_newton_iterations: int
    maximum_balance_residual_N: float
    maximum_penetration_m: float
    final_contact_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "surface_node_count": self.surface_node_count,
            "actual_vector_p1_degrees_of_freedom": self.actual_vector_p1_degrees_of_freedom,
            "trajectory_position_count": self.trajectory_position_count,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_working_set_bytes": self.peak_working_set_bytes,
            "maximum_newton_iterations": self.maximum_newton_iterations,
            "maximum_balance_residual_N": self.maximum_balance_residual_N,
            "maximum_penetration_m": self.maximum_penetration_m,
            "final_contact_count": self.final_contact_count,
        }


def _strict_count(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a strict integer of at least {minimum}")
    return value


def build_performance_benchmark_case(
    *,
    surface_node_count: int = 221,
    trajectory_position_count: int = 101,
    indentation_depth_m: float = 0.1e-6,
    scratch_distance_m: float = 20.0e-6,
) -> ContactPerformanceBenchmark:
    """Build the fixed 50k-DOF/101-position benchmark without running it."""

    nx = _strict_count(surface_node_count, "surface_node_count", 5)
    count = _strict_count(trajectory_position_count, "trajectory_position_count", 4)
    if not math.isfinite(indentation_depth_m) or indentation_depth_m <= 0.0:
        raise ValueError("indentation_depth_m must be finite and positive")
    if not math.isfinite(scratch_distance_m) or scratch_distance_m < 0.0:
        raise ValueError("scratch_distance_m must be finite and non-negative")

    width = 400.0e-6
    height = 200.0e-6
    radius = 200.0e-6
    clearance = 1.0e-6
    center_x = 0.5 * width - 0.5 * scratch_distance_m
    initial_y = height + radius + clearance
    contact_y = height + radius - indentation_depth_m
    scratch_count = count - 3
    targets: list[dict[str, str | float]] = [
        {"segment": "initial", "center_x_m": center_x, "center_y_m": initial_y},
        {"segment": "indentation", "center_x_m": center_x, "center_y_m": contact_y},
    ]
    for index in range(1, scratch_count + 1):
        fraction = index / scratch_count if scratch_count else 1.0
        targets.append(
            {
                "segment": "scratch",
                "center_x_m": center_x + fraction * scratch_distance_m,
                "center_y_m": contact_y,
            }
        )
    targets.append(
        {
            "segment": "unloading",
            "center_x_m": center_x + scratch_distance_m,
            "center_y_m": initial_y,
        }
    )
    vertical_count = max(3, int(round((nx - 1) * height / width)) + 1)
    estimated_dofs = 2 * nx * vertical_count
    case = SingleGrainContactCase.from_mapping(
        {
            "single_grain_contact_schema_version": 1,
            "unit_system": "SI",
            "model_type": "single_grain_real_contact",
            "geometry": {"type": "rectangle", "width": width, "height": height},
            "mesh": {"target_size": width / (nx - 1)},
            "analysis": {"type": "plane_strain", "thickness": 1.0e-3},
            "fixed_boundary": {"side": "bottom", "components": ["x", "y"]},
            "material": {
                "E": 210.0e9,
                "nu": 0.3,
                "yield_strength": 1.0e20,
                "tangent_modulus": 0.0,
                "parameter_source": "builtin_numerical_performance_benchmark",
                "material_state": "elastic_limit_for_solver_performance",
                "calibration_status": "numerical_benchmark_not_calibrated",
            },
            "grain": {
                "type": "rigid_circle",
                "radius_m": radius,
                "initial_center_x_m": center_x,
                "initial_center_y_m": initial_y,
            },
            "trajectory": {"targets": targets},
            "contact": {
                "normal_algorithm": "penalty",
                "normal_penalty_factor": 10.0,
                "tangential_penalty_ratio": 0.5,
                "augmented_relaxation": 1.0,
                "augmented_maximum_iterations": 8,
                "friction_coefficient": 0.0,
                "friction_regularization_ratio": 1.0e-6,
            },
            "newton": {
                "residual_relative_tolerance": 1.0e-8,
                "residual_absolute_tolerance_N": 1.0e-6,
                "displacement_relative_tolerance": 1.0e-8,
                "displacement_absolute_tolerance_m": 1.0e-14,
                "maximum_iterations": 40,
            },
            "step_control": {
                "allow_reduction": True,
                "minimum_substep_fraction": 1.0 / 1024.0,
                "maximum_retries": 10,
            },
            "output": {"save_field_evolution": False, "snapshot_stride": 1},
            "safety": {
                "maximum_degrees_of_freedom": 50_000,
                "warn_displacement_over_local_size": 0.10,
                "stop_displacement_over_local_size": 0.50,
                "warn_displacement_over_radius": 0.05,
                "stop_displacement_over_radius": 0.20,
                "warn_strain": 0.05,
                "stop_strain": 0.15,
            },
        }
    )
    return ContactPerformanceBenchmark(case, nx, estimated_dofs)


def _peak_working_set_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if peak > 10_000_000 else peak * 1024
    except (ImportError, AttributeError, OSError):
        return 0


def run_performance_benchmark(
    benchmark: ContactPerformanceBenchmark,
) -> ContactPerformanceResult:
    if not isinstance(benchmark, ContactPerformanceBenchmark):
        raise TypeError("benchmark must be a ContactPerformanceBenchmark")
    mesh = tensor_contact_mesh(benchmark.case, benchmark.surface_node_count)
    started = time.perf_counter()
    result = run_contact_trajectory_on_mesh(benchmark.case, mesh)
    elapsed = time.perf_counter() - started
    return ContactPerformanceResult(
        surface_node_count=benchmark.surface_node_count,
        actual_vector_p1_degrees_of_freedom=result.prepared_mesh.total_degrees_of_freedom,
        trajectory_position_count=len(result.records),
        elapsed_seconds=elapsed,
        peak_working_set_bytes=_peak_working_set_bytes(),
        maximum_newton_iterations=max(item.newton_iterations for item in result.records),
        maximum_balance_residual_N=max(item.balance_residual_N for item in result.records),
        maximum_penetration_m=max(item.maximum_penetration_m for item in result.records),
        final_contact_count=result.records[-1].contact_count,
    )


def main() -> int:
    benchmark = build_performance_benchmark_case()
    result = run_performance_benchmark(benchmark)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
