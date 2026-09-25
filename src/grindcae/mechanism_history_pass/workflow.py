"""Phase 7A.3 orchestration over one fixed mesh and the shared 6B.2 engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from grindcae.elastoplastic_fem import prepare_elastoplastic_mesh_with_facet_tractions
from grindcae.history_pass import assemble_fixed_top_mapping_load_vector
from grindcae.history_pass.workflow import TargetFieldSnapshot, advance_prepared_history
from grindcae.moving_load_pass import solve_fixed_mesh_moving_load_pass
from grindcae.statistical_grain_load import predict_statistical_grain_load

from .models import MechanismHistoryPassCase
from .projection import MechanismProjection, project_statistical_mechanism_load


@dataclass(frozen=True, slots=True)
class MechanismHistoryPassResult:
    case: MechanismHistoryPassCase
    prediction: object
    moving_load: object
    prepared_mesh: object
    baseline: object
    mechanism: object
    projections: tuple[MechanismProjection, ...]


def solve_mechanism_elastoplastic_history_pass(
    case: MechanismHistoryPassCase,
    workspace: str | Path,
    on_baseline_snapshot: Callable[[TargetFieldSnapshot], None] | None = None,
    on_mechanism_snapshot: Callable[[TargetFieldSnapshot], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> MechanismHistoryPassResult:
    if not isinstance(case, MechanismHistoryPassCase):
        raise TypeError("case must be a MechanismHistoryPassCase")
    workspace = Path(workspace).resolve(); workspace.mkdir(parents=True, exist_ok=True)
    prediction = predict_statistical_grain_load(case.statistical_grain_load)
    moving = solve_fixed_mesh_moving_load_pass(case.baseline_history_pass.moving_load_pass, workspace / "moving")
    fem_case = case.baseline_history_pass.reference_case.to_elastoplastic_fem_case(peak_force_x_N=0.0, peak_force_y_N=0.0)
    prepared = prepare_elastoplastic_mesh_with_facet_tractions(fem_case, moving.reference_mesh, np.empty(0, dtype=np.int32), peak_force_x_N=0.0, peak_force_y_N=0.0, facet_line_load_x_N_per_m=np.empty(0), facet_line_load_y_N_per_m=np.empty(0))
    baseline_targets = tuple(assemble_fixed_top_mapping_load_vector(prepared, item.load_mapping) for item in moving.positions)
    projections = tuple(project_statistical_mechanism_load(prepared, item, prediction) for item in moving.positions)
    for baseline, mechanism in zip(baseline_targets, projections):
        if not np.allclose([np.sum(baseline[prepared.component_dofs[0]]), np.sum(baseline[prepared.component_dofs[1]])], [mechanism.total_Fx_N, mechanism.total_Fy_N], rtol=1e-10, atol=1e-10):
            raise RuntimeError("statistical mechanism projection does not match baseline total force")
    total = len(moving.positions) * 2
    baseline = advance_prepared_history(
        case.baseline_history_pass,
        moving,
        prepared,
        baseline_targets,
        on_target_snapshot=on_baseline_snapshot,
        progress=(lambda current, _route_total, _message: progress(current, total, f"基准路线已收敛位置 {current}/{len(moving.positions)}")) if progress is not None else None,
    )
    mechanism = advance_prepared_history(
        case.baseline_history_pass,
        moving,
        prepared,
        tuple(item.target_load_vector_N for item in projections),
        on_target_snapshot=on_mechanism_snapshot,
        progress=(lambda current, _route_total, _message: progress(len(moving.positions) + current, total, f"机制路线已收敛位置 {current}/{len(moving.positions)}")) if progress is not None else None,
    )
    return MechanismHistoryPassResult(case, prediction, moving, prepared, baseline, mechanism, projections)


def run_mechanism_elastoplastic_history_pass(
    case: MechanismHistoryPassCase, output_dir: str | Path, progress=None
) -> tuple[MechanismHistoryPassResult, dict[str, Path], dict[str, object]]:
    """Solve and transactionally publish retained and optional field artifacts."""
    from .exporters import (
        artifact_paths,
        publish_artifacts,
        stage_mechanism_history_result,
        validate_artifacts,
    )
    from .field_evolution import (
        FIELD_EVOLUTION_DIRECTORY,
        MechanismFieldEvolutionExporter,
        validate_mechanism_field_evolution,
    )
    import tempfile
    output = Path(output_dir).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="grindcae-7a3-solve-") as workspace:
        workspace_path = Path(workspace)
        staged_root = workspace_path / "artifacts"
        staged_root.mkdir()
        staged_sequence = staged_root / FIELD_EVOLUTION_DIRECTORY
        exporter = MechanismFieldEvolutionExporter(staged_sequence)
        result = solve_mechanism_elastoplastic_history_pass(
            case,
            workspace_path / "solve",
            on_baseline_snapshot=exporter.on_baseline_snapshot,
            on_mechanism_snapshot=exporter.on_mechanism_snapshot,
            progress=progress,
        )
        if progress is not None:
            total = len(result.moving_load.positions) * 2
            progress(total, total, "两条路线计算完成，正在生成场序列与差值图片。")
        manifest = exporter.finalize()
        validate_mechanism_field_evolution(staged_sequence)

        artifacts = artifact_paths(output)
        final_sequence = output / FIELD_EVOLUTION_DIRECTORY
        field_evolution = {
            "directory": str(final_sequence),
            "manifest": str(final_sequence / "manifest.json"),
            "position_history_csv": str(final_sequence / "position_history.csv"),
            "frame_count": manifest["frame_count"],
            "routes": {
                route: {"pvd": str(final_sequence / route / "field_evolution.pvd")}
                for route in ("baseline", "mechanism", "difference")
            },
            "sequence_coordinate": manifest["sequence_coordinate"],
            "shared_original_color_ranges": manifest["shared_original_color_ranges"],
            "difference_color_ranges": manifest["difference_color_ranges"],
        }
        staged = artifact_paths(staged_root / "retained")
        staged["summary_json"].parent.mkdir()
        summary = stage_mechanism_history_result(
            result, staged, artifacts, field_evolution=field_evolution
        )
        publish_artifacts(
            staged,
            artifacts,
            temporary_field_evolution=staged_sequence,
            final_field_evolution=final_sequence,
        )
        if progress is not None:
            progress(total, total, "结果已事务发布，正在执行完整结果验证。")
        validate_artifacts(artifacts)
        validate_mechanism_field_evolution(final_sequence)
        published_mesh = result.moving_load.__class__(
            case=result.moving_load.case,
            reference_pass_load=result.moving_load.reference_pass_load,
            reference_mesh=result.moving_load.reference_mesh,
            top_facets=result.moving_load.top_facets,
            positions=result.moving_load.positions,
            msh_source=artifacts["reference_mesh_msh"],
            mesh_generation_count=result.moving_load.mesh_generation_count,
            mesh_import_count=result.moving_load.mesh_import_count,
        )
        result = MechanismHistoryPassResult(
            case=result.case,
            prediction=result.prediction,
            moving_load=published_mesh,
            prepared_mesh=result.prepared_mesh,
            baseline=result.baseline,
            mechanism=result.mechanism,
            projections=result.projections,
        )
    return result, artifacts, summary
