"""Sequential phase 4C4 scan core built on the reusable phase 4C3B workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable

from grindcae.evolved_fem import (
    EvolvedFemOperations,
    EvolvedFemResult,
    default_operations as default_evolved_operations,
    solve_evolved_fem_case,
)
from grindcae.pass_loading import PassLoadResult

from .models import PassScanCase
from .positions import ScanPosition, generate_scan_positions


RESULT_FORMAT = "grindcae_phase_4c4_single_pass_quasi_static_scan"
SNAPSHOT_ROLE_ORDER = (
    "representative_entry",
    "representative_full_contact",
    "representative_exit",
    "maximum_displacement",
    "maximum_von_mises_p95",
    "maximum_von_mises_p99",
)


class PassScanError(RuntimeError):
    """Raised when a validated pass scan violates its physical or data checks."""


@dataclass(frozen=True, slots=True)
class ScanHistoryRow:
    """Finite aggregate fields from one independently solved scan position."""

    scan_index: int
    motion_coordinate_m: float
    wheel_lowest_point_x_m: float
    nominal_elapsed_time_s: float
    relative_feed_direction: str
    pass_state: str
    contact_ratio: float
    effective_projected_contact_length_m: float
    ground_projected_length_m: float
    contact_arc_projected_length_m: float
    unprocessed_projected_length_m: float
    current_Fx_N: float
    current_Fy_N: float
    assembled_Fx_N: float
    assembled_Fy_N: float
    fixed_reaction_Rx_N: float
    fixed_reaction_Ry_N: float
    force_balance_residual_norm_N: float
    node_count: int
    triangle_count: int
    active_facet_count: int
    maximum_displacement_m: float
    von_mises_area_weighted_mean_Pa: float
    von_mises_area_weighted_p95_Pa: float
    von_mises_area_weighted_p99_Pa: float
    von_mises_raw_element_maximum_mesh_dependent_Pa: float
    principal_stress_1_maximum_Pa: float
    principal_stress_2_minimum_Pa: float
    snapshot_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.scan_index) is not int or self.scan_index < 0:
            raise PassScanError("scan_index must be a non-negative strict integer")
        for field in (
            "motion_coordinate_m",
            "wheel_lowest_point_x_m",
            "nominal_elapsed_time_s",
            "contact_ratio",
            "effective_projected_contact_length_m",
            "ground_projected_length_m",
            "contact_arc_projected_length_m",
            "unprocessed_projected_length_m",
            "current_Fx_N",
            "current_Fy_N",
            "assembled_Fx_N",
            "assembled_Fy_N",
            "fixed_reaction_Rx_N",
            "fixed_reaction_Ry_N",
            "force_balance_residual_norm_N",
            "maximum_displacement_m",
            "von_mises_area_weighted_mean_Pa",
            "von_mises_area_weighted_p95_Pa",
            "von_mises_area_weighted_p99_Pa",
            "von_mises_raw_element_maximum_mesh_dependent_Pa",
            "principal_stress_1_maximum_Pa",
            "principal_stress_2_minimum_Pa",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PassScanError(f"{field} must be finite")
        for field in ("node_count", "triangle_count", "active_facet_count"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise PassScanError(f"{field} must be a non-negative strict integer")
        if self.node_count <= 0 or self.triangle_count <= 0:
            raise PassScanError("every scan point must retain a non-empty mesh")
        if not 0.0 <= self.contact_ratio <= 1.0:
            raise PassScanError("contact_ratio must lie in [0, 1]")

    def to_dict(self) -> dict[str, int | float | str]:
        values: dict[str, int | float | str] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = ";".join(value) if name == "snapshot_roles" else value
        return values


@dataclass(frozen=True, slots=True)
class PassScanValidations:
    ground_length_monotonic_non_decreasing: bool
    unprocessed_length_monotonic_non_increasing: bool
    entry_contact_ratio_monotonic_non_decreasing: bool
    full_contact_ratio_is_one: bool
    exit_contact_ratio_monotonic_non_increasing: bool
    all_partition_lengths_sum_to_workpiece: bool
    all_facet_integrals_pass: bool
    zero_contact_endpoints_have_zero_finite_response: bool
    full_contact_force_matches_steady_reference: bool
    mapped_assembled_reaction_balance_pass: bool
    all_force_balances_pass: bool
    maximum_force_balance_residual_norm_N: float

    @property
    def passed(self) -> bool:
        return all(
            (
                self.ground_length_monotonic_non_decreasing,
                self.unprocessed_length_monotonic_non_increasing,
                self.entry_contact_ratio_monotonic_non_decreasing,
                self.full_contact_ratio_is_one,
                self.exit_contact_ratio_monotonic_non_increasing,
                self.all_partition_lengths_sum_to_workpiece,
                self.all_facet_integrals_pass,
                self.zero_contact_endpoints_have_zero_finite_response,
                self.full_contact_force_matches_steady_reference,
                self.mapped_assembled_reaction_balance_pass,
                self.all_force_balances_pass,
            )
        )

    def to_dict(self) -> dict[str, bool | float]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        } | {"passed": self.passed}


@dataclass(frozen=True, slots=True)
class PassScanResult:
    """Completed scan history and only the unique detailed results still needed."""

    case: PassScanCase
    reference_pass_load: PassLoadResult
    positions: tuple[ScanPosition, ...]
    history: tuple[ScanHistoryRow, ...]
    snapshot_role_indices: dict[str, int]
    retained_results: dict[int, EvolvedFemResult]
    validations: PassScanValidations

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.history):
            raise PassScanError("position and history counts must match")
        if [item.scan_index for item in self.history] != list(range(len(self.history))):
            raise PassScanError("history indexing must be 0-based contiguous")
        if set(self.snapshot_role_indices) != set(SNAPSHOT_ROLE_ORDER):
            raise PassScanError("snapshot role map is incomplete")
        referenced = set(self.snapshot_role_indices.values())
        if set(self.retained_results) != referenced:
            raise PassScanError("retained result set must equal unique snapshot indices")
        if not self.validations.passed:
            raise PassScanError("scan history validation failed")

    @property
    def unique_snapshot_count(self) -> int:
        return len(self.retained_results)


ProgressCallback = Callable[[ScanHistoryRow, int], None]


def first_maximum_index(
    rows: tuple[ScanHistoryRow, ...] | list[ScanHistoryRow],
    field: str,
) -> int:
    """Return the first motion-order index attaining a requested maximum."""

    if not rows:
        raise PassScanError("cannot select a maximum from an empty scan history")
    if field not in ScanHistoryRow.__dataclass_fields__:
        raise PassScanError(f"unknown scan-history field: {field}")
    maximum = max(getattr(row, field) for row in rows)
    return next(row.scan_index for row in rows if getattr(row, field) == maximum)


def _case_at_position(reference: PassScanCase, position: ScanPosition):
    reference_case = reference.reference_case
    trajectory = reference_case.pass_load.trajectory
    single_pass = replace(
        trajectory.single_pass,
        wheel_lowest_point_x_m=position.wheel_lowest_point_x_m,
    )
    trajectory_at_position = replace(trajectory, single_pass=single_pass)
    pass_load_at_position = replace(
        reference_case.pass_load,
        trajectory=trajectory_at_position,
    )
    return replace(reference_case, pass_load=pass_load_at_position)


def _history_row(
    position: ScanPosition,
    result: EvolvedFemResult,
    start_motion_coordinate_m: float,
    feed_speed_m_per_s: float,
) -> ScanHistoryRow:
    load = result.pass_load
    trajectory = load.trajectory_result
    solver = result.solution.solver_summary
    statistics = result.stress_statistics
    return ScanHistoryRow(
        scan_index=position.scan_index,
        motion_coordinate_m=position.motion_coordinate_m,
        wheel_lowest_point_x_m=position.wheel_lowest_point_x_m,
        nominal_elapsed_time_s=(position.motion_coordinate_m - start_motion_coordinate_m) / feed_speed_m_per_s,
        relative_feed_direction=load.case.trajectory.single_pass.relative_feed_direction,
        pass_state=load.pass_state,
        contact_ratio=load.contact_ratio,
        effective_projected_contact_length_m=load.effective_contact_length_m,
        ground_projected_length_m=trajectory.region_length_m("ground"),
        contact_arc_projected_length_m=trajectory.region_length_m("contact_arc"),
        unprocessed_projected_length_m=trajectory.region_length_m("unprocessed"),
        current_Fx_N=load.current_Fx_N,
        current_Fy_N=load.current_Fy_N,
        assembled_Fx_N=solver.assembled_external_force_x,
        assembled_Fy_N=solver.assembled_external_force_y,
        fixed_reaction_Rx_N=solver.fixed_reaction_x,
        fixed_reaction_Ry_N=solver.fixed_reaction_y,
        force_balance_residual_norm_N=solver.balance_residual_norm,
        node_count=result.evolved_mesh.node_count,
        triangle_count=result.evolved_mesh.triangle_count,
        active_facet_count=len(result.facet_load.facets),
        maximum_displacement_m=solver.maximum_displacement,
        von_mises_area_weighted_mean_Pa=statistics.von_mises_area_weighted_mean,
        von_mises_area_weighted_p95_Pa=statistics.von_mises_area_weighted_p95,
        von_mises_area_weighted_p99_Pa=statistics.von_mises_area_weighted_p99,
        von_mises_raw_element_maximum_mesh_dependent_Pa=statistics.von_mises_raw_element_maximum_mesh_dependent,
        principal_stress_1_maximum_Pa=statistics.principal_stress_1_in_plane_maximum,
        principal_stress_2_minimum_Pa=statistics.principal_stress_2_in_plane_minimum,
    )


def _non_decreasing(values: list[float], tolerance: float) -> bool:
    return all(second + tolerance >= first for first, second in zip(values, values[1:]))


def _non_increasing(values: list[float], tolerance: float) -> bool:
    return all(second <= first + tolerance for first, second in zip(values, values[1:]))


def _validations(
    rows: list[ScanHistoryRow],
    width_m: float,
    facet_integrals_pass: bool,
    steady_fx_N: float,
    steady_fy_N: float,
) -> PassScanValidations:
    length_tolerance = max(1.0e-14, 1.0e-11 * width_m)
    ratio_tolerance = 1.0e-12
    ground = [row.ground_projected_length_m for row in rows]
    unprocessed = [row.unprocessed_projected_length_m for row in rows]
    first_full_index = next(
        index for index, row in enumerate(rows) if row.pass_state == "full_contact"
    )
    last_full_index = max(
        index for index, row in enumerate(rows) if row.pass_state == "full_contact"
    )
    entry = [row.contact_ratio for row in rows[: first_full_index + 1]]
    full = [row.contact_ratio for row in rows if row.pass_state == "full_contact"]
    exit_values = [row.contact_ratio for row in rows[last_full_index:]]
    maximum_residual = max(row.force_balance_residual_norm_N for row in rows)
    force_scale = max(
        1.0,
        *(abs(row.current_Fx_N) for row in rows),
        *(abs(row.current_Fy_N) for row in rows),
    )
    force_tolerance = max(1.0e-9, 1.0e-10 * force_scale)
    endpoint_fields = (
        "current_Fx_N",
        "current_Fy_N",
        "assembled_Fx_N",
        "assembled_Fy_N",
        "fixed_reaction_Rx_N",
        "fixed_reaction_Ry_N",
        "force_balance_residual_norm_N",
        "maximum_displacement_m",
        "von_mises_area_weighted_mean_Pa",
        "von_mises_area_weighted_p95_Pa",
        "von_mises_area_weighted_p99_Pa",
        "von_mises_raw_element_maximum_mesh_dependent_Pa",
        "principal_stress_1_maximum_Pa",
        "principal_stress_2_minimum_Pa",
    )
    zero_endpoints = all(
        row.contact_ratio == 0.0
        and row.active_facet_count == 0
        and all(getattr(row, field) == 0.0 for field in endpoint_fields)
        for row in (rows[0], rows[-1])
    )
    full_force_platform = all(
        math.isclose(row.current_Fx_N, steady_fx_N, rel_tol=1.0e-12, abs_tol=force_tolerance)
        and math.isclose(row.current_Fy_N, steady_fy_N, rel_tol=1.0e-12, abs_tol=force_tolerance)
        for row in rows
        if row.pass_state == "full_contact"
    )
    mapped_assembled_reaction = all(
        math.isclose(row.assembled_Fx_N, row.current_Fx_N, rel_tol=1.0e-11, abs_tol=force_tolerance)
        and math.isclose(row.assembled_Fy_N, row.current_Fy_N, rel_tol=1.0e-11, abs_tol=force_tolerance)
        and math.isclose(row.fixed_reaction_Rx_N, -row.current_Fx_N, rel_tol=1.0e-11, abs_tol=force_tolerance)
        and math.isclose(row.fixed_reaction_Ry_N, -row.current_Fy_N, rel_tol=1.0e-11, abs_tol=force_tolerance)
        for row in rows
    )
    return PassScanValidations(
        ground_length_monotonic_non_decreasing=_non_decreasing(ground, length_tolerance),
        unprocessed_length_monotonic_non_increasing=_non_increasing(unprocessed, length_tolerance),
        entry_contact_ratio_monotonic_non_decreasing=bool(entry) and _non_decreasing(entry, ratio_tolerance),
        full_contact_ratio_is_one=bool(full) and all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=ratio_tolerance) for value in full),
        exit_contact_ratio_monotonic_non_increasing=bool(exit_values) and _non_increasing(exit_values, ratio_tolerance),
        all_partition_lengths_sum_to_workpiece=all(
            math.isclose(
                row.ground_projected_length_m
                + row.contact_arc_projected_length_m
                + row.unprocessed_projected_length_m,
                width_m,
                rel_tol=1.0e-12,
                abs_tol=length_tolerance,
            )
            for row in rows
        ),
        all_facet_integrals_pass=facet_integrals_pass,
        zero_contact_endpoints_have_zero_finite_response=zero_endpoints,
        full_contact_force_matches_steady_reference=full_force_platform,
        mapped_assembled_reaction_balance_pass=mapped_assembled_reaction,
        all_force_balances_pass=maximum_residual <= force_tolerance,
        maximum_force_balance_residual_norm_N=maximum_residual,
    )


def run_pass_scan(
    case: PassScanCase,
    workspace: str | Path,
    *,
    deformation_operations: EvolvedFemOperations | None = None,
    progress: ProgressCallback | None = None,
) -> PassScanResult:
    """Solve the complete pass serially, retaining only final snapshot candidates."""

    if not isinstance(case, PassScanCase):
        raise TypeError("case must be a PassScanCase")
    operations = deformation_operations or default_evolved_operations()
    workspace_path = Path(workspace).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    reference_pass_load = operations.build_pass_load(case.reference_case.pass_load)
    positions = generate_scan_positions(case, reference_pass_load)
    feed_speed = case.reference_case.pass_load.force_model.process.workpiece_feed_speed_m_per_s
    start_motion = positions[0].motion_coordinate_m
    width = case.reference_case.geometry.width
    rows: list[ScanHistoryRow] = []
    role_indices: dict[str, int] = {}
    retained: dict[int, EvolvedFemResult] = {}
    metric_by_role = {
        "maximum_displacement": "maximum_displacement_m",
        "maximum_von_mises_p95": "von_mises_area_weighted_p95_Pa",
        "maximum_von_mises_p99": "von_mises_area_weighted_p99_Pa",
    }
    facet_integrals_pass = True

    # 中文导读：扫描按位置串行求解；每个位置都是无历史继承的独立静力快照。
    for position in positions:
        point_case = _case_at_position(case, position)
        point_workspace = workspace_path / f"point_{position.scan_index:04d}"
        precomputed = (
            reference_pass_load if "reference_case" in position.generation_roles else None
        )
        result = solve_evolved_fem_case(
            point_case,
            point_workspace,
            precomputed_pass_load=precomputed,
            operations=operations,
        )
        if result.pass_load.pass_state != position.pass_state:
            raise PassScanError("position state does not match the shared phase 4C1 classification")
        row = _history_row(position, result, start_motion, feed_speed)
        rows.append(row)

        for role in SNAPSHOT_ROLE_ORDER[:3]:
            if role in position.generation_roles:
                role_indices[role] = position.scan_index
        for role, field in metric_by_role.items():
            previous_index = role_indices.get(role)
            # 中文导读：响应极值保留首次出现的位置，保证结果可重复。
            if previous_index is None or getattr(row, field) > getattr(rows[previous_index], field):
                role_indices[role] = position.scan_index

        for facet in result.facet_load.facets:
            facet_integrals_pass = facet_integrals_pass and math.isclose(
                facet.tx_applied_per_ds_N_per_m * facet.ds_actual_m,
                facet.qx_projected_N_per_m * facet.dx_projected_m,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            ) and math.isclose(
                facet.ty_applied_per_ds_N_per_m * facet.ds_actual_m,
                facet.qy_projected_N_per_m * facet.dx_projected_m,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )

        referenced = set(role_indices.values())
        if position.scan_index in referenced:
            retained[position.scan_index] = result
        for old_index in tuple(retained):
            if old_index not in referenced:
                del retained[old_index]
        if progress is not None:
            progress(row, len(positions))

    if set(role_indices) != set(SNAPSHOT_ROLE_ORDER):
        raise PassScanError("scan did not assign all required snapshot roles")
    for role, field in metric_by_role.items():
        if role_indices[role] != first_maximum_index(rows, field):
            raise PassScanError(f"{role} did not retain the first tied maximum")
    final_rows = []
    for row in rows:
        roles = tuple(
            role for role in SNAPSHOT_ROLE_ORDER if role_indices[role] == row.scan_index
        )
        final_rows.append(replace(row, snapshot_roles=roles))
    reference_fx = (
        reference_pass_load.steady_tangential_force_magnitude_N
        if reference_pass_load.case.force_mapping.tangential_force_direction == "positive_x"
        else -reference_pass_load.steady_tangential_force_magnitude_N
    )
    validations = _validations(
        final_rows,
        width,
        facet_integrals_pass,
        reference_fx,
        -reference_pass_load.steady_normal_force_magnitude_N,
    )
    return PassScanResult(
        case=case,
        reference_pass_load=reference_pass_load,
        positions=positions,
        history=tuple(final_rows),
        snapshot_role_indices=role_indices,
        retained_results=retained,
        validations=validations,
    )
