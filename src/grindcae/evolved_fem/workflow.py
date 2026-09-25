"""Reusable phase 4C3B solve, export, and transactional publication workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable
import uuid

import matplotlib.image as matplotlib_image
import meshio

from grindcae import __version__
from grindcae.evolved_mesh import (
    EvolvedMeshCase,
    EvolvedMeshValidationResult,
    generate_evolved_surface_mesh,
    validate_evolved_surface_mesh,
)
from grindcae.pass_loading import PassLoadResult, build_pass_load
from grindcae.postprocess import (
    ELEMENT_CSV_FIELDS,
    NODE_CSV_FIELDS,
    RecoveredFields,
    StressStatistics,
    build_summary as build_postprocess_summary,
    compute_stress_statistics,
    recover_fields,
    write_elements_csv,
    write_nodes_csv,
    write_pngs,
    write_vtu,
)
from grindcae.solver import (
    FacetLoadFullFieldSolution,
    ImportedMesh,
    assemble_facet_plane_stress_system,
    solve_facet_full_field,
)

from .exporters import (
    ACTIVE_CONTACT_FACET_CSV_FIELDS,
    FEM_PNG_KEYS,
    NATIVE_P1_CONTOUR_DEFINITIONS,
    enhance_evolved_fem_pngs,
    write_active_contact_facets_csv,
    write_mesh_png,
    write_native_p1_contour_pngs,
)
from .loading import EvolvedFacetLoadResult, build_evolved_facet_load
from .models import MODEL_TYPE, EvolvedFemCase


RESULT_FORMAT = "grindcae_phase_4c3b_evolved_surface_empirical_load_postprocess"
RESULT_TITLE = "Evolved-surface empirical grinding-load FEM result"
ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "evolved_surface_msh": "evolved_surface.msh",
    "vtu": "results.vtu",
    "nodes_csv": "nodes.csv",
    "elements_csv": "elements.csv",
    "active_contact_facets_csv": "active_contact_facets.csv",
    "mesh_png": "mesh.png",
    "displacement_magnitude_png": "displacement_magnitude.png",
    "von_mises_stress_png": "von_mises_stress.png",
    "principal_stress_1_in_plane_png": "principal_stress_1_in_plane.png",
    "deformed_shape_png": "deformed_shape.png",
    "stress_contour_png": "stress_contour.png",
    "strain_contour_png": "strain_contour.png",
}


class EvolvedFemWorkflowError(RuntimeError):
    """Raised when a reusable solve or artifact transaction is invalid."""


@dataclass(frozen=True, slots=True)
class EvolvedFemOperations:
    """Injectable operations used to audit one-call workflow behavior."""

    build_pass_load: Callable[..., object]
    generate_mesh: Callable[..., object]
    validate_mesh: Callable[..., object]
    build_facet_load: Callable[..., object]
    assemble_system: Callable[..., object]
    solve_system: Callable[..., object]
    recover_fields: Callable[..., object]
    compute_statistics: Callable[..., object]
    write_vtu: Callable[..., object]
    write_nodes_csv: Callable[..., object]
    write_elements_csv: Callable[..., object]
    write_active_facets_csv: Callable[..., object]
    write_pngs: Callable[..., object]
    write_mesh_png: Callable[..., object]
    write_native_p1_contour_pngs: Callable[..., object]
    replace: Callable[..., object]


def default_operations() -> EvolvedFemOperations:
    """Return operations from current module globals for testable call boundaries."""

    return EvolvedFemOperations(
        build_pass_load=build_pass_load,
        generate_mesh=generate_evolved_surface_mesh,
        validate_mesh=validate_evolved_surface_mesh,
        build_facet_load=build_evolved_facet_load,
        assemble_system=assemble_facet_plane_stress_system,
        solve_system=solve_facet_full_field,
        recover_fields=recover_fields,
        compute_statistics=compute_stress_statistics,
        write_vtu=write_vtu,
        write_nodes_csv=write_nodes_csv,
        write_elements_csv=write_elements_csv,
        write_active_facets_csv=write_active_contact_facets_csv,
        write_pngs=write_pngs,
        write_mesh_png=write_mesh_png,
        write_native_p1_contour_pngs=write_native_p1_contour_pngs,
        replace=os.replace,
    )


@dataclass(frozen=True, slots=True)
class EvolvedFemResult:
    """One complete solved 4C3B snapshot retained for later export."""

    case: EvolvedFemCase
    pass_load: PassLoadResult
    evolved_mesh: EvolvedMeshValidationResult
    facet_load: EvolvedFacetLoadResult
    solution: FacetLoadFullFieldSolution
    recovered_fields: RecoveredFields
    stress_statistics: StressStatistics
    msh_source: Path

    def __post_init__(self) -> None:
        source = Path(self.msh_source).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".msh":
            raise EvolvedFemWorkflowError("result MSH source is missing or invalid")
        if self.pass_load.case != self.case.pass_load:
            raise EvolvedFemWorkflowError("pass-load result provenance is invalid")
        if self.evolved_mesh.trajectory != self.pass_load.trajectory_result:
            raise EvolvedFemWorkflowError("mesh trajectory provenance is invalid")
        if self.facet_load.mesh != self.evolved_mesh:
            raise EvolvedFemWorkflowError("facet-load mesh provenance is invalid")
        if self.solution.imported_mesh.source_path != source:
            raise EvolvedFemWorkflowError("solution MSH provenance is invalid")
        object.__setattr__(self, "msh_source", source)


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {
        key: (directory / filename).resolve()
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def evolved_mesh_case(case: EvolvedFemCase) -> EvolvedMeshCase:
    return EvolvedMeshCase(
        evolved_mesh_schema_version=1,
        unit_system=case.unit_system,
        model_type="single_pass_evolved_surface_gmsh",
        trajectory=case.pass_load.trajectory,
        mesh=case.mesh,
    )


def solve_evolved_fem_case(
    case: EvolvedFemCase,
    workspace: str | Path,
    *,
    precomputed_pass_load: PassLoadResult | None = None,
    operations: EvolvedFemOperations | None = None,
) -> EvolvedFemResult:
    """Solve one position exactly once and retain all traceable result objects."""

    if not isinstance(case, EvolvedFemCase):
        raise TypeError("case must be an EvolvedFemCase")
    ops = operations or default_operations()
    workspace_path = Path(workspace).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    msh_path = (workspace_path / ARTIFACT_FILENAMES["evolved_surface_msh"]).resolve()
    if precomputed_pass_load is None:
        pass_load = ops.build_pass_load(case.pass_load)
    else:
        if not isinstance(precomputed_pass_load, PassLoadResult):
            raise TypeError("precomputed_pass_load must be a PassLoadResult")
        if precomputed_pass_load.case != case.pass_load:
            raise EvolvedFemWorkflowError(
                "precomputed pass load does not match the evolved-FEM case"
            )
        pass_load = precomputed_pass_load
    mesh_case = evolved_mesh_case(case)
    # 中文导读：单位置主链路只走一次“载荷 -> 网格 -> 组装 -> 求解 -> 恢复”。
    generation = ops.generate_mesh(
        mesh_case,
        pass_load.trajectory_result,
        msh_path,
    )
    mesh = ops.validate_mesh(
        msh_path,
        mesh_case,
        pass_load.trajectory_result,
        generation,
    )
    facet_load = ops.build_facet_load(pass_load, mesh)
    imported = ImportedMesh(mesh=mesh.skfem_mesh, source_path=msh_path)
    system = ops.assemble_system(
        case,
        imported,
        facet_load.facet_ids,
        mapped_force_x_N=pass_load.current_Fx_N,
        mapped_force_y_N=pass_load.current_Fy_N,
        facet_line_load_x_N_per_m=facet_load.tx_per_ds,
        facet_line_load_y_N_per_m=facet_load.ty_per_ds,
    )
    solution = ops.solve_system(case, system)
    fields = ops.recover_fields(solution)
    statistics = ops.compute_statistics(fields)
    return EvolvedFemResult(
        case=case,
        pass_load=pass_load,
        evolved_mesh=mesh,
        facet_load=facet_load,
        solution=solution,
        recovered_fields=fields,
        stress_statistics=statistics,
        msh_source=msh_path,
    )


def build_evolved_fem_summary(
    result: EvolvedFemResult,
    deformation_scale: float,
    artifacts: dict[str, Path],
) -> dict[str, object]:
    case = result.case
    pass_load = result.pass_load
    mesh = result.evolved_mesh
    facet_load = result.facet_load
    solution = result.solution
    statistics = result.stress_statistics
    solver = solution.solver_summary
    postprocess = build_postprocess_summary(
        solution, statistics, artifacts, deformation_scale
    )
    active_rows = [item.to_dict() for item in facet_load.facets]
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "evolved_fem_schema_version": case.evolved_fem_schema_version,
        "unit_system": case.unit_system,
        "model_type": case.model_type,
        "normalized_input": case.to_dict(),
        "units": {
            "length": "m",
            "area": "m^2",
            "thickness": "m",
            "force": "N",
            "projected_line_load": "N/m over dx",
            "applied_line_traction": "N/m over ds",
            "displacement": "m",
            "strain": "dimensionless",
            "stress": "Pa",
        },
        "data_flow": [
            "strict EvolvedFemCase input with one embedded PassLoadCase trajectory",
            "build_pass_load reuses phase 4C1 geometry and phase 4A force prediction",
            "generate_evolved_surface_mesh and validate_evolved_surface_mesh reuse the same trajectory result and MSH",
            "runtime contact_arc facets convert projected q dx to actual-edge t ds",
            "one shared plane-stress stiffness assembly and linear solve",
            "recover_fields and compatible VTU/CSV/PNG post-processing",
        ],
        "pass_load": {
            "pass_state": pass_load.pass_state,
            "contact_ratio": pass_load.contact_ratio,
            "effective_projected_interval_m": (
                None
                if pass_load.effective_contact_interval_m is None
                else list(pass_load.effective_contact_interval_m)
            ),
            "effective_projected_contact_length_m": pass_load.effective_contact_length_m,
            "steady_full_contact_empirical_force": {
                "tangential_force_magnitude_N": pass_load.steady_tangential_force_magnitude_N,
                "normal_force_magnitude_N": pass_load.steady_normal_force_magnitude_N,
                "resultant_force_magnitude_N": pass_load.steady_resultant_force_magnitude_N,
            },
            "current_empirical_force": {
                "tangential_force_magnitude_N": pass_load.current_tangential_force_magnitude_N,
                "normal_force_magnitude_N": pass_load.current_normal_force_magnitude_N,
                "resultant_force_magnitude_N": pass_load.current_resultant_force_magnitude_N,
                "signed_Fx_N": pass_load.current_Fx_N,
                "signed_Fy_N": pass_load.current_Fy_N,
            },
            "projected_line_load": {
                "qx_N_per_m_over_dx": pass_load.current_qx_N_per_m,
                "qy_N_per_m_over_dx": pass_load.current_qy_N_per_m,
                "definition": "global x/y force per unit horizontal projection dx",
            },
            "calibration_provenance": pass_load.case.force_model.calibration.provenance_dict(),
        },
        "evolved_mesh": {
            "node_count": mesh.node_count,
            "triangle_count": mesh.triangle_count,
            "actual_vector_p1_degrees_of_freedom": mesh.actual_vector_p1_degrees_of_freedom,
            "raw_gmsh_node_count": mesh.generation.raw_gmsh_node_count,
            "meshio_point_count": mesh.node_count,
            "scikit_fem_nvertices": int(mesh.skfem_mesh.nvertices),
            "arc_facet_count": mesh.arc_facet_count,
            "exact_arc_curve_length_m": mesh.exact_arc_curve_length_m,
            "discrete_arc_facet_length_m": mesh.discrete_arc_facet_length_m,
            "exact_evolved_domain_area_m2": mesh.exact_evolved_domain_area_m2,
            "discrete_triangle_area_m2": mesh.discrete_triangle_area_m2,
            "curvature_discretization_area_error_m2": mesh.curvature_discretization_area_error_m2,
            "physical_group_names": [
                "domain",
                "fixed",
                "contact",
                "free_left",
                "free_right",
            ],
            "active_contact_is_runtime_data": True,
            "active_contact_physical_group_exists": False,
        },
        "facet_load_conversion": {
            "selection_rule": "region == contact_arc only",
            "coordinate_system": "global x/y; no local tangent/normal rotation",
            "formula": {
                "tx_per_ds": "qx_per_dx * dx_projected / ds_actual",
                "ty_per_ds": "qy_per_dx * dx_projected / ds_actual",
            },
            "active_facet_count": len(active_rows),
            "projected_length_sum_m": facet_load.projected_length_sum_m,
            "actual_facet_length_sum_m": facet_load.actual_facet_length_sum_m,
            "integrated_Fx_N": facet_load.integrated_Fx_N,
            "integrated_Fy_N": facet_load.integrated_Fy_N,
            "all_rows_q_dx_equal_t_ds": True,
            "other_top_regions_unloaded": ["ground", "unprocessed"],
            "rows": active_rows,
        },
        "solver_summary": solver.to_dict(),
        "force_tracking": {
            "current_empirical_force": {
                "Fx_N": solver.mapped_force_x,
                "Fy_N": solver.mapped_force_y,
            },
            "assembled_external_force": {
                "Fx_N": solver.assembled_external_force_x,
                "Fy_N": solver.assembled_external_force_y,
            },
            "fixed_support_reaction": {
                "Rx_N": solver.fixed_reaction_x,
                "Ry_N": solver.fixed_reaction_y,
            },
            "force_balance_residual": {
                "x_N": solver.balance_residual_x,
                "y_N": solver.balance_residual_y,
                "norm_N": solver.balance_residual_norm,
            },
        },
        "maximum_displacement": {
            "magnitude_m": solver.maximum_displacement,
            "node_id": int(solver.maximum_displacement_node),
            "coordinates_m": list(solver.maximum_displacement_coordinates),
        },
        "indexing": {
            **postprocess["indexing"],
            "facet_index_base": 0,
            "active_contact_csv_node_ids_reference_final_mesh": True,
        },
        "field_definitions": postprocess["field_definitions"],
        "field_locations": postprocess["field_locations"],
        "stress_statistics": postprocess["stress_statistics"],
        "peak_value_caution": postprocess["peak_value_caution"],
        "visualization": {
            **postprocess["visualization"],
            "native_p1_contours": NATIVE_P1_CONTOUR_DEFINITIONS,
            "native_p1_contour_shading": "flat cell coloring",
            "native_p1_contour_nodal_averaging": False,
            "native_p1_contour_smoothing": False,
            "native_p1_contour_geometry": "original final solve mesh coordinates",
            "mesh_top_regions_distinguished": True,
            "global_load_direction_shown": True,
            "chinese_short_note_on_all_pngs": False,
            "active_loaded_facet_inset_on_fem_pngs": True,
            "active_inset_uses_final_solve_mesh": True,
            "active_inset_load_direction_source": "applied facet tx/ty",
            "active_inset_arrow_scale_visual_only": True,
            "result_interpretation": RESULT_TITLE,
        },
        "mesh_and_field_provenance": {
            "mesh_generation_count": 1,
            "mesh_validation_import_count": 1,
            "fem_solve_count": 1,
            "same_msh_for_validation_and_solve": True,
            "same_imported_scikit_fem_mesh_for_solve_and_recovery": True,
            "native_vector_p1_recovery_without_smoothing": True,
        },
        "physical_scope": {
            "included": [
                "single pass",
                "quasi-static response",
                "small deformation",
                "2D plane stress",
                "isotropic linear elasticity",
                "one-way empirical force coupling",
            ],
            "omitted": [
                "true wheel-workpiece contact and contact iteration",
                "Hertz pressure",
                "friction iteration",
                "abrasive-grain random contact",
                "plasticity, material failure, and edge chipping",
                "thermal field and grinding burn",
                "machine vibration, chatter, and transient impact",
                "Ra/Rz roughness prediction",
                "wheel wear and runout",
                "multiple passes, reciprocation, and spark-out",
                "industrial validation and automatic material databases",
            ],
            "interpretation_limit": (
                "Do not infer yielding, safety factor, production quality, or "
                "industrial validity directly from the current stress result."
            ),
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def serialize_summary(summary: dict[str, object]) -> str:
    try:
        return json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise EvolvedFemWorkflowError(
            f"summary [finite JSON]: serialization failed: {exc}"
        ) from exc


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="\n")


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def validate_evolved_fem_artifacts(artifacts: dict[str, Path]) -> None:
    """Validate one complete 13-file result without re-reading its solve MSH."""

    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise EvolvedFemWorkflowError("temporary artifact set is invalid")
    for path in artifacts.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise EvolvedFemWorkflowError(
                f"temporary artifact is missing or empty: {path.name}"
            )
    serialized = artifacts["summary_json"].read_text(encoding="ascii")
    if "NaN" in serialized or "Infinity" in serialized:
        raise EvolvedFemWorkflowError("summary contains a non-finite JSON token")
    summary = json.loads(serialized)
    if summary.get("result_format") != RESULT_FORMAT:
        raise EvolvedFemWorkflowError("summary result_format is invalid")
    if summary.get("model_type") != MODEL_TYPE:
        raise EvolvedFemWorkflowError("summary model_type is invalid")
    if set(summary.get("artifacts", {})) != set(ARTIFACT_FILENAMES):
        raise EvolvedFemWorkflowError("summary artifact map is invalid")
    contours = summary.get("visualization", {}).get("native_p1_contours")
    if contours != NATIVE_P1_CONTOUR_DEFINITIONS:
        raise EvolvedFemWorkflowError("summary native P1 contour definitions are invalid")

    node_count = int(summary["evolved_mesh"]["node_count"])
    triangle_count = int(summary["evolved_mesh"]["triangle_count"])
    if not artifacts["evolved_surface_msh"].read_bytes().startswith(b"$MeshFormat"):
        raise EvolvedFemWorkflowError("evolved-surface MSH header is invalid")

    vtu = meshio.read(artifacts["vtu"])
    expected_point_fields = {"displacement_m", "displacement_magnitude_m"}
    expected_cell_fields = {
        "strain_xx",
        "strain_yy",
        "engineering_shear_strain_xy",
        "derived_strain_zz",
        "stress_xx_Pa",
        "stress_yy_Pa",
        "shear_stress_xy_Pa",
        "von_mises_stress_Pa",
        "principal_stress_1_in_plane_Pa",
        "principal_stress_2_in_plane_Pa",
    }
    if set(vtu.point_data) != expected_point_fields or set(vtu.cell_data) != expected_cell_fields:
        raise EvolvedFemWorkflowError("VTU field names are invalid")
    if vtu.points.shape != (node_count, 3):
        raise EvolvedFemWorkflowError("VTU node count is invalid")
    if (
        len(vtu.cells) != 1
        or vtu.cells[0].type != "triangle"
        or len(vtu.cells[0].data) != triangle_count
    ):
        raise EvolvedFemWorkflowError("VTU must contain only the final triangles")

    node_header, node_rows = _read_csv(artifacts["nodes_csv"])
    element_header, element_rows = _read_csv(artifacts["elements_csv"])
    active_header, active_rows = _read_csv(artifacts["active_contact_facets_csv"])
    if node_header != NODE_CSV_FIELDS or len(node_rows) != node_count:
        raise EvolvedFemWorkflowError("nodes CSV header or row count is invalid")
    if element_header != ELEMENT_CSV_FIELDS or len(element_rows) != triangle_count:
        raise EvolvedFemWorkflowError("elements CSV header or row count is invalid")
    if active_header != ACTIVE_CONTACT_FACET_CSV_FIELDS:
        raise EvolvedFemWorkflowError("active-contact CSV header is invalid")
    expected_active_count = int(summary["facet_load_conversion"]["active_facet_count"])
    if len(active_rows) != expected_active_count:
        raise EvolvedFemWorkflowError("active-contact CSV row count is invalid")
    if active_rows:
        if any(row["region"] != "contact_arc" for row in active_rows):
            raise EvolvedFemWorkflowError("active-contact CSV contains another region")
        if max(
            max(int(row["node_start_id"]), int(row["node_end_id"]))
            for row in active_rows
        ) >= node_count:
            raise EvolvedFemWorkflowError("active-contact CSV node ID is out of range")
        dx = math.fsum(float(row["dx_projected_m"]) for row in active_rows)
        fx = math.fsum(float(row["integrated_Fx_N"]) for row in active_rows)
        fy = math.fsum(float(row["integrated_Fy_N"]) for row in active_rows)
        conversion = summary["facet_load_conversion"]
        if not math.isclose(dx, conversion["projected_length_sum_m"], rel_tol=1e-12, abs_tol=1e-14):
            raise EvolvedFemWorkflowError("active-contact CSV projected length is invalid")
        if not math.isclose(fx, conversion["integrated_Fx_N"], rel_tol=1e-12, abs_tol=1e-12):
            raise EvolvedFemWorkflowError("active-contact CSV Fx integral is invalid")
        if not math.isclose(fy, conversion["integrated_Fy_N"], rel_tol=1e-12, abs_tol=1e-12):
            raise EvolvedFemWorkflowError("active-contact CSV Fy integral is invalid")
    for key in FEM_PNG_KEYS:
        pixels = matplotlib_image.imread(artifacts[key])
        if pixels.size == 0 or pixels.ndim < 2:
            raise EvolvedFemWorkflowError(f"{artifacts[key].name} is unreadable")


def export_evolved_fem_result(
    result: EvolvedFemResult,
    output_dir: str | Path,
    deformation_scale: str | float = "auto",
    *,
    summary_artifacts: dict[str, Path] | None = None,
    operations: EvolvedFemOperations | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Export one already-solved result without regenerating or re-solving it."""

    if not isinstance(result, EvolvedFemResult):
        raise TypeError("result must be an EvolvedFemResult")
    ops = operations or default_operations()
    outputs = artifact_paths(output_dir)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if result.msh_source != outputs["evolved_surface_msh"]:
        shutil.copy2(result.msh_source, outputs["evolved_surface_msh"])
    ops.write_vtu(result.recovered_fields, outputs["vtu"])
    ops.write_nodes_csv(result.recovered_fields, outputs["nodes_csv"])
    ops.write_elements_csv(result.recovered_fields, outputs["elements_csv"])
    ops.write_active_facets_csv(result.facet_load, outputs["active_contact_facets_csv"])
    scale = ops.write_pngs(
        result.solution,
        result.recovered_fields,
        outputs,
        deformation_scale,
        result_title=RESULT_TITLE,
    )
    ops.write_mesh_png(result.facet_load, outputs["mesh_png"])
    ops.write_native_p1_contour_pngs(result.recovered_fields, outputs)
    enhance_evolved_fem_pngs(result.facet_load, outputs)
    summary_paths = summary_artifacts or outputs
    if set(summary_paths) != set(ARTIFACT_FILENAMES):
        raise EvolvedFemWorkflowError("summary artifact path set is invalid")
    summary = build_evolved_fem_summary(result, scale, summary_paths)
    _write_text(serialize_summary(summary), outputs["summary_json"])
    validate_evolved_fem_artifacts(outputs)
    return outputs, summary


def _stage_copy(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, staged)
    except OSError as exc:
        raise EvolvedFemWorkflowError(
            f"could not stage {target.name} for publication: {exc}"
        ) from exc
    return staged


def publish_evolved_fem_artifacts(
    temporary_artifacts: dict[str, Path],
    artifacts: dict[str, Path],
    *,
    replace: Callable[..., object] = os.replace,
) -> None:
    """Publish one complete 13-file set with summary last and rollback."""

    if set(temporary_artifacts) != set(artifacts) or set(artifacts) != set(
        ARTIFACT_FILENAMES
    ):
        raise EvolvedFemWorkflowError("artifact publication set is invalid")
    validate_evolved_fem_artifacts(temporary_artifacts)
    output_dir = artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + [
        "summary_json"
    ]
    # 中文导读：最后发布 summary.json；中途失败会恢复旧文件，避免半套结果。
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in order:
            staged[key] = _stage_copy(temporary_artifacts[key], artifacts[key], "new")
        for key in order:
            if artifacts[key].exists():
                backups[key] = _stage_copy(artifacts[key], artifacts[key], "backup")
        for key in order:
            replace(staged[key], artifacts[key])
            del staged[key]
            published.append(key)
    except (OSError, EvolvedFemWorkflowError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups:
                    replace(backups[key], artifacts[key])
                    del backups[key]
                else:
                    artifacts[key].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{artifacts[key].name} rollback failed: {rollback_exc}"
                )
        message = f"artifact publication failed: {exc}"
        if rollback_errors:
            message += "; " + "; ".join(rollback_errors)
        raise EvolvedFemWorkflowError(message) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def run_evolved_fem_case(
    case: EvolvedFemCase,
    output_dir: str | Path,
    deformation_scale: str | float = "auto",
    *,
    operations: EvolvedFemOperations | None = None,
) -> tuple[EvolvedFemResult, dict[str, Path], dict[str, object]]:
    """Solve, export, validate, and transactionally publish one 4C3B case."""

    ops = operations or default_operations()
    final_artifacts = artifact_paths(output_dir)
    with tempfile.TemporaryDirectory(prefix="grindcae-evolved-fem-") as temp_dir:
        workspace = Path(temp_dir)
        result = solve_evolved_fem_case(
            case,
            workspace / "solve",
            operations=ops,
        )
        temporary_artifacts, summary = export_evolved_fem_result(
            result,
            workspace / "artifacts",
            deformation_scale,
            summary_artifacts=final_artifacts,
            operations=ops,
        )
        publish_evolved_fem_artifacts(
            temporary_artifacts,
            final_artifacts,
            replace=ops.replace,
        )
        published_mesh = replace(
            result.evolved_mesh,
            generation=replace(
                result.evolved_mesh.generation,
                output_path=final_artifacts["evolved_surface_msh"],
            ),
        )
        published_facet_load = replace(result.facet_load, mesh=published_mesh)
        published_solution = replace(
            result.solution,
            imported_mesh=ImportedMesh(
                mesh=published_mesh.skfem_mesh,
                source_path=final_artifacts["evolved_surface_msh"],
            ),
        )
        result = replace(
            result,
            evolved_mesh=published_mesh,
            facet_load=published_facet_load,
            solution=published_solution,
            msh_source=final_artifacts["evolved_surface_msh"],
        )
    return result, final_artifacts, summary
