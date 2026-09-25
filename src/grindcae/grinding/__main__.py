"""Command-line entry point for phase 4B2 local grinding-load results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import meshio
import numpy as np

from grindcae import __version__
from grindcae.force_model import ForceModelError, ForceModelValidationError
from grindcae.mesh import (
    ActiveContactFacetSelection,
    LocalContactMeshSummary,
    MeshGenerationError,
    MeshValidationError,
    generate_local_contact_mesh,
    validate_active_contact_facets,
)
from grindcae.postprocess import (
    ARTIFACT_FILENAMES as POSTPROCESS_ARTIFACT_FILENAMES,
    ELEMENT_CSV_FIELDS,
    NODE_CSV_FIELDS,
    PostprocessError,
    StressStatistics,
    build_summary as build_postprocess_summary,
    compute_stress_statistics,
    deformation_scale_argument,
    recover_fields,
    write_elements_csv,
    write_nodes_csv,
    write_pngs,
    write_vtu,
)
from grindcae.solver import (
    LocalLoadFullFieldSolution,
    SolverError,
    assemble_local_plane_stress_system,
    import_gmsh_mesh,
    solve_local_full_field,
)

from .exporters import (
    ACTIVE_CONTACT_FACET_CSV_FIELDS,
    write_active_contact_facets_csv,
)
from .loading import GrindingLoadError, ResolvedGrindingLoad, resolve_grinding_load
from .models import GrindingCaseValidationError, GrindingSimulationCase


RESULT_FORMAT = "grindcae_phase_4b2_local_contact_postprocess"
SUMMARY_FILENAME = "summary.json"
MESH_FILENAME = "solve_input.msh"
ACTIVE_CONTACT_FACETS_FILENAME = "active_contact_facets.csv"
RESULT_TITLE = (
    "Local empirical grinding load - Equivalent boundary-load FEM result"
)
ARTIFACT_FILENAMES = {
    "summary_json": SUMMARY_FILENAME,
    "solve_input_mesh": MESH_FILENAME,
    "vtu": POSTPROCESS_ARTIFACT_FILENAMES["vtu"],
    "nodes_csv": POSTPROCESS_ARTIFACT_FILENAMES["nodes_csv"],
    "elements_csv": POSTPROCESS_ARTIFACT_FILENAMES["elements_csv"],
    "active_contact_facets_csv": ACTIVE_CONTACT_FACETS_FILENAME,
    "mesh_png": POSTPROCESS_ARTIFACT_FILENAMES["mesh_png"],
    "displacement_magnitude_png": POSTPROCESS_ARTIFACT_FILENAMES[
        "displacement_magnitude_png"
    ],
    "von_mises_stress_png": POSTPROCESS_ARTIFACT_FILENAMES[
        "von_mises_stress_png"
    ],
    "principal_stress_1_in_plane_png": POSTPROCESS_ARTIFACT_FILENAMES[
        "principal_stress_1_in_plane_png"
    ],
    "deformed_shape_png": POSTPROCESS_ARTIFACT_FILENAMES["deformed_shape_png"],
}


class GrindingWorkflowError(RuntimeError):
    """Raised when a validated local-contact run cannot be published."""


def _deformation_scale_argument(value: str) -> str | float:
    try:
        return deformation_scale_argument(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict grinding-force magnitudes, solve their exact local top-edge "
            "load, recover plane-stress fields, and export traceable results."
        )
    )
    parser.add_argument("case", help="path to a schema version 4 JSON case")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving the complete 11-file result set",
    )
    parser.add_argument(
        "--deformation-scale",
        default="auto",
        type=_deformation_scale_argument,
        help="visual-only positive scale for deformed_shape.png (default: auto)",
    )
    return parser


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        key: (output_dir / filename).resolve()
        for key, filename in ARTIFACT_FILENAMES.items()
    }


def _build_summary(
    case: GrindingSimulationCase,
    resolved: ResolvedGrindingLoad,
    mesh_summary: LocalContactMeshSummary,
    selection: ActiveContactFacetSelection,
    solution: LocalLoadFullFieldSolution,
    statistics: StressStatistics,
    deformation_scale: float,
    artifacts: dict[str, Path],
) -> dict[str, object]:
    prediction = resolved.force_prediction
    result = solution.solver_summary
    postprocess_summary = build_postprocess_summary(
        solution,
        statistics,
        artifacts,
        deformation_scale,
    )
    peak_caution = dict(postprocess_summary["peak_value_caution"])
    peak_caution["local_load_endpoint_mesh_sensitivity"] = (
        "the two endpoints of the uniform local load can create mesh-sensitive "
        "element stresses"
    )
    cautions = [
        "The force comes from the phase 4A empirical specific-grinding-energy and force-ratio model; its parameters require experimental or reliable literature calibration for the applicable process.",
        "The geometric contact length lg = sqrt(Ds * ae) is a shallow-cut geometry approximation.",
        "The current workflow is one-way from predicted force to finite-element response; the structural response does not update the force prediction.",
        "The local boundary load is a uniform prescribed traction over validated runtime facets.",
        "The reported average surface tractions are not Hertz pressure or a true contact-pressure distribution.",
        "No friction law, contact iteration, or wheel rigid body is implemented.",
        "The two ends of the local loading interval can create mesh-sensitive element stresses.",
        "The fixed/free boundary junction can also produce a stress singularity.",
        "The raw element stress maximum can change with mesh refinement.",
        "Area-weighted p95 and p99 are descriptive statistics and do not replace a mesh-convergence study.",
        "The program does not automatically determine yielding, failure, a safety factor, or grinding burn.",
        "The current result has not been calibrated and validated against a user experimental database.",
    ]
    if resolved.calibration_id == "demo_unvalidated":
        cautions.insert(
            1,
            "Calibration ID demo_unvalidated is a demonstration parameter set, not an industrially validated material or process database.",
        )
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "schema_version": case.schema_version,
        "unit_system": case.unit_system,
        "units": {
            "length": "m",
            "area": "m^2",
            "thickness": "m",
            "force": "N",
            "line_load": "N/m",
            "surface_traction": "Pa",
            "displacement": "m",
            "strain": "dimensionless",
            "stress": "Pa",
        },
        "force_prediction": prediction.to_dict(),
        "calibration_provenance": (
            prediction.force_case.calibration.provenance_dict()
        ),
        "load_mapping": {
            "load_source": resolved.load_source,
            "boundary": resolved.boundary.value,
            "distribution": resolved.distribution,
            "tangential_direction": resolved.tangential_direction,
            "predicted_force_magnitudes": {
                "tangential_force_magnitude_N": (
                    prediction.tangential_force_magnitude_N
                ),
                "normal_force_magnitude_N": prediction.normal_force_magnitude_N,
                "resultant_force_magnitude_N": (
                    prediction.resultant_force_magnitude_N
                ),
            },
            "mapped_signed_forces": {
                "Fx_N": resolved.signed_force_x_N,
                "Fy_N": resolved.signed_force_y_N,
            },
            "line_loads": {
                "qx_N_per_m": resolved.line_load_x_N_per_m,
                "qy_N_per_m": resolved.line_load_y_N_per_m,
            },
            "average_surface_tractions": {
                "average_normal_pressure_Pa": (
                    resolved.average_normal_pressure_Pa
                ),
                "average_tangential_traction_Pa": (
                    resolved.average_tangential_traction_Pa
                ),
                "interpretation": (
                    "uniform local traction conversion; not a Hertz pressure "
                    "or a true contact-pressure distribution"
                ),
            },
        },
        "contact_zone": {
            "boundary": resolved.boundary.value,
            "distribution": resolved.distribution,
            "center_x_m": resolved.center_x_m,
            "x_start_m": resolved.x_start_m,
            "x_end_m": resolved.x_end_m,
            "requested_length_m": resolved.active_length_m,
            "actual_active_facet_length_m": selection.actual_active_length_m,
            "active_facet_count": selection.active_facet_count,
            "minimum_required_active_facet_count": (
                selection.minimum_required_active_facet_count
            ),
            "x_start_node_indices": list(selection.x_start_node_indices),
            "x_end_node_indices": list(selection.x_end_node_indices),
            "node_index_base": 0,
            "active_facet_index_base": 0,
            "active_facet_length_integral_m": result.active_facet_length,
            "active_facet_force_integral": {
                "Fx_N": result.line_load_x * result.active_facet_length,
                "Fy_N": result.line_load_y * result.active_facet_length,
            },
        },
        "mesh": {
            "node_count": result.node_count,
            "triangle_count": result.triangle_count,
            "total_degrees_of_freedom": result.total_degrees_of_freedom,
            "element_order": 1,
            "physical_group_names": list(mesh_summary.physical_group_names),
            "top_curve_count": mesh_summary.top_curve_count,
            "contact_boundary_length_m": selection.contact_boundary_length_m,
            "contact_facet_count": selection.contact_facet_count,
            "active_geometric_curve_element_count": (
                mesh_summary.active_geometric_curve_element_count
            ),
            "local_active_element_target": mesh_summary.local_active_element_target,
            "exact_split_nodes_present": True,
        },
        "solver_summary": result.to_dict(),
        "balance": {
            "mapped_signed_forces": {
                "Fx_N": result.mapped_force_x,
                "Fy_N": result.mapped_force_y,
            },
            "assembled_external_force": {
                "Fx_N": result.assembled_external_force_x,
                "Fy_N": result.assembled_external_force_y,
            },
            "fixed_boundary_reaction": {
                "Rx_N": result.fixed_reaction_x,
                "Ry_N": result.fixed_reaction_y,
            },
            "balance_residual": {
                "x_N": result.balance_residual_x,
                "y_N": result.balance_residual_y,
                "norm_N": result.balance_residual_norm,
            },
        },
        "indexing": postprocess_summary["indexing"],
        "field_definitions": postprocess_summary["field_definitions"],
        "field_locations": postprocess_summary["field_locations"],
        "stress_statistics": postprocess_summary["stress_statistics"],
        "peak_value_caution": peak_caution,
        "visualization": {
            **postprocess_summary["visualization"],
            "mesh_active_contact_overlay": True,
            "candidate_contact_and_active_load_are_distinct": True,
            "result_interpretation": RESULT_TITLE,
        },
        "assumptions_and_cautions": cautions,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }


def _serialize_summary(summary: dict[str, object]) -> str:
    try:
        return (
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise GrindingWorkflowError(
            f"summary_payload [finite JSON]: serialization failed: {exc}"
        ) from exc


def _write_temporary_summary(serialized: str, path: Path) -> None:
    try:
        path.write_text(serialized, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise GrindingWorkflowError(
            f"temporary_summary [writable JSON]: could not be written: {exc}"
        ) from exc


def _stage_copy(source: Path, destination: Path, label: str) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.{label}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary_path
    except OSError as exc:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise GrindingWorkflowError(
            f"artifact [{destination.name}]: staging failed: {exc}"
        ) from exc


def _validate_temporary_artifacts(temporary_artifacts: dict[str, Path]) -> None:
    if set(temporary_artifacts) != set(ARTIFACT_FILENAMES):
        raise GrindingWorkflowError(
            "temporary artifact set does not match the phase 4B2 contract"
        )
    for key, path in temporary_artifacts.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise GrindingWorkflowError(
                f"artifact [{key}]: temporary result is missing or empty"
            )
    try:
        summary = json.loads(
            temporary_artifacts["summary_json"].read_text(encoding="utf-8")
        )
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        node_count = int(summary["mesh"]["node_count"])
        triangle_count = int(summary["mesh"]["triangle_count"])
        active_facet_count = int(summary["contact_zone"]["active_facet_count"])

        solve_mesh = meshio.read(temporary_artifacts["solve_input_mesh"])
        if set(solve_mesh.field_data) != {
            "domain",
            "fixed",
            "contact",
            "free_left",
            "free_right",
        }:
            raise ValueError("solve mesh physical groups are invalid")
        solve_triangle_count = sum(
            len(block.data)
            for block in solve_mesh.cells
            if block.type == "triangle"
        )
        if solve_mesh.points.shape[0] != node_count:
            raise ValueError("solve mesh node count does not match the summary")
        if solve_triangle_count != triangle_count:
            raise ValueError("solve mesh triangle count does not match the summary")

        vtu = meshio.read(temporary_artifacts["vtu"])
        if {block.type for block in vtu.cells} != {"triangle"}:
            raise ValueError("VTU must contain triangle cells only")
        if set(vtu.point_data) != {
            "displacement_m",
            "displacement_magnitude_m",
        }:
            raise ValueError("VTU point-data fields are invalid")
        expected_cell_data = {
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
        if set(vtu.cell_data) != expected_cell_data:
            raise ValueError("VTU triangle cell-data fields are invalid")
        if vtu.points.shape != (node_count, 3):
            raise ValueError("VTU point count does not match the summary")
        vtu_triangle_count = sum(
            len(block.data) for block in vtu.cells if block.type == "triangle"
        )
        if vtu_triangle_count != triangle_count:
            raise ValueError("VTU triangle count does not match the summary")
        if not all(np.all(np.isfinite(values)) for values in vtu.point_data.values()):
            raise ValueError("VTU point data contains a non-finite value")
        for blocks in vtu.cell_data.values():
            if len(blocks) != 1 or not np.all(np.isfinite(blocks[0])):
                raise ValueError("VTU cell data is invalid or non-finite")

        csv_expectations = (
            ("nodes_csv", NODE_CSV_FIELDS, node_count),
            ("elements_csv", ELEMENT_CSV_FIELDS, triangle_count),
            (
                "active_contact_facets_csv",
                ACTIVE_CONTACT_FACET_CSV_FIELDS,
                active_facet_count,
            ),
        )
        csv_rows: dict[str, list[dict[str, str]]] = {}
        for key, expected_fields, expected_count in csv_expectations:
            with temporary_artifacts[key].open(
                encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            if reader.fieldnames != list(expected_fields):
                raise ValueError(f"{key} header does not match the export contract")
            if len(rows) != expected_count:
                raise ValueError(f"{key} row count does not match the summary")
            csv_rows[key] = rows

        facet_rows = csv_rows["active_contact_facets_csv"]
        facet_ids = [int(row["facet_id"]) for row in facet_rows]
        if len(set(facet_ids)) != len(facet_ids):
            raise ValueError("active contact CSV contains duplicate facet IDs")
        active_values = np.asarray(
            [
                [
                    float(row["x0_m"]),
                    float(row["y0_m"]),
                    float(row["x1_m"]),
                    float(row["y1_m"]),
                    float(row["length_m"]),
                    float(row["qx_N_per_m"]),
                    float(row["qy_N_per_m"]),
                ]
                for row in facet_rows
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(active_values)):
            raise ValueError("active contact CSV contains a non-finite value")
        if np.any(active_values[:, 4] <= 0.0):
            raise ValueError("active contact CSV contains a non-positive length")
        active_length = float(np.sum(active_values[:, 4]))
        integrated_force = np.array(
            [
                np.dot(active_values[:, 4], active_values[:, 5]),
                np.dot(active_values[:, 4], active_values[:, 6]),
            ],
            dtype=float,
        )
        expected_length = float(
            summary["contact_zone"]["actual_active_facet_length_m"]
        )
        expected_force = np.array(
            [
                summary["load_mapping"]["mapped_signed_forces"]["Fx_N"],
                summary["load_mapping"]["mapped_signed_forces"]["Fy_N"],
            ],
            dtype=float,
        )
        if not np.isclose(active_length, expected_length, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("active contact CSV length integral is invalid")
        if not np.allclose(
            integrated_force, expected_force, rtol=1.0e-10, atol=1.0e-10
        ):
            raise ValueError("active contact CSV force integral is invalid")

        from matplotlib import image as matplotlib_image

        for key in (
            "mesh_png",
            "displacement_magnitude_png",
            "von_mises_stress_png",
            "principal_stress_1_in_plane_png",
            "deformed_shape_png",
        ):
            pixels = matplotlib_image.imread(temporary_artifacts[key])
            if pixels.ndim not in (2, 3) or pixels.size == 0:
                raise ValueError(f"{key} is not a decodable image")
            if not np.all(np.isfinite(pixels)):
                raise ValueError(f"{key} contains non-finite pixel data")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GrindingWorkflowError(
            f"temporary artifact validation failed: {exc}"
        ) from exc


def _publish_results(
    temporary_artifacts: dict[str, Path],
    artifacts: dict[str, Path],
) -> None:
    if set(temporary_artifacts) != set(artifacts) or set(artifacts) != set(
        ARTIFACT_FILENAMES
    ):
        raise GrindingWorkflowError(
            "publication artifact set does not match the phase 4B2 contract"
        )
    _validate_temporary_artifacts(temporary_artifacts)

    output_dir = artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GrindingWorkflowError(
            f"output_directory [writable directory]: could not be created: {exc}"
        ) from exc

    publication_order = [
        key for key in ARTIFACT_FILENAMES if key != "summary_json"
    ] + ["summary_json"]
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in publication_order:
            staged[key] = _stage_copy(
                temporary_artifacts[key], artifacts[key], "new"
            )
        for key in publication_order:
            if artifacts[key].exists():
                backups[key] = _stage_copy(artifacts[key], artifacts[key], "backup")

        for key in publication_order:
            os.replace(staged[key], artifacts[key])
            del staged[key]
            published.append(key)
    except (OSError, GrindingWorkflowError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups:
                    os.replace(backups[key], artifacts[key])
                    del backups[key]
                else:
                    artifacts[key].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{artifacts[key].name} rollback failed: {rollback_exc}"
                )

        detail = f"artifact publication failed: {exc}"
        if rollback_errors:
            detail += "; " + "; ".join(rollback_errors)
        raise GrindingWorkflowError(detail) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _print_summary(
    resolved: ResolvedGrindingLoad,
    selection: ActiveContactFacetSelection,
    solution: LocalLoadFullFieldSolution,
    statistics: StressStatistics,
    artifacts: dict[str, Path],
) -> None:
    prediction = resolved.force_prediction
    result = solution.solver_summary
    print(
        "Material removal rate Qw: "
        f"{prediction.material_removal_rate_m3_per_s:.12g} m^3/s"
    )
    print(f"Estimated grinding power P: {prediction.estimated_grinding_power_W:.12g} W")
    print(
        "Predicted force magnitudes: "
        f"Ft={prediction.tangential_force_magnitude_N:.12g} N, "
        f"Fn={prediction.normal_force_magnitude_N:.12g} N, "
        f"Fr={prediction.resultant_force_magnitude_N:.12g} N"
    )
    print(
        "Geometric contact length lg: "
        f"{prediction.geometric_contact_length_m:.12g} m"
    )
    print(
        "Contact interval: "
        f"x0={resolved.x_start_m:.12g} m, x1={resolved.x_end_m:.12g} m"
    )
    print(f"Tangential direction: {resolved.tangential_direction}")
    print(
        "Uniform local line load: "
        f"qx={resolved.line_load_x_N_per_m:.12g} N/m, "
        f"qy={resolved.line_load_y_N_per_m:.12g} N/m"
    )
    print(
        "Average uniform surface traction: "
        f"normal pressure={resolved.average_normal_pressure_Pa:.12g} Pa, "
        "tangential traction="
        f"{resolved.average_tangential_traction_Pa:.12g} Pa"
    )
    print(
        "Active contact facets: "
        f"count={selection.active_facet_count}, "
        f"actual length={selection.actual_active_length_m:.12g} m"
    )
    print(
        "Assembled external force [N]: "
        f"Fx={result.assembled_external_force_x:.12g}, "
        f"Fy={result.assembled_external_force_y:.12g}"
    )
    print(
        "Fixed boundary reaction [N]: "
        f"Rx={result.fixed_reaction_x:.12g}, "
        f"Ry={result.fixed_reaction_y:.12g}"
    )
    print(
        "Balance residual [N]: "
        f"x={result.balance_residual_x:.6e}, "
        f"y={result.balance_residual_y:.6e}, "
        f"norm={result.balance_residual_norm:.6e}"
    )
    print(f"Maximum displacement [m]: {result.maximum_displacement:.12g}")
    print(
        "Von Mises stress [Pa]: "
        f"area-weighted mean={statistics.von_mises_area_weighted_mean:.12g}, "
        f"p95={statistics.von_mises_area_weighted_p95:.12g}, "
        f"p99={statistics.von_mises_area_weighted_p99:.12g}"
    )
    print(
        "Raw element maximum von Mises (mesh-dependent): "
        f"{statistics.von_mises_raw_element_maximum_mesh_dependent:.12g} Pa"
    )
    print(f"Calibration ID: {resolved.calibration_id}")
    print("Output files:")
    for path in artifacts.values():
        print(f"  {path}")
    print(
        "Calibration warning: us and Rnt require experiment or reliable "
        "literature calibration for the applicable grinding process."
    )
    print(
        "Contact-model warning: this is a one-way uniform local boundary load, "
        "without Hertz pressure, friction law, contact iteration, or a wheel body."
    )
    print(
        "Stress warning: local-load endpoints and fixed/free corners can be "
        "mesh-sensitive; p95 and p99 do not replace mesh convergence."
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts = _artifact_paths(output_dir)

    try:
        case = GrindingSimulationCase.from_json(input_path)
        resolved = resolve_grinding_load(case)
        with tempfile.TemporaryDirectory(prefix="grindcae-grinding-") as temp_dir:
            workspace = Path(temp_dir)
            temporary_artifacts = {
                key: workspace / filename
                for key, filename in ARTIFACT_FILENAMES.items()
            }

            mesh_summary = generate_local_contact_mesh(
                case, resolved, temporary_artifacts["solve_input_mesh"]
            )
            imported = import_gmsh_mesh(
                temporary_artifacts["solve_input_mesh"], case
            )
            selection = validate_active_contact_facets(
                imported.mesh,
                width_m=case.geometry.width,
                height_m=case.geometry.height,
                x_start_m=resolved.x_start_m,
                x_end_m=resolved.x_end_m,
            )
            system = assemble_local_plane_stress_system(
                case,
                imported,
                selection.active_facets,
                signed_force_x_N=resolved.signed_force_x_N,
                signed_force_y_N=resolved.signed_force_y_N,
                line_load_x_N_per_m=resolved.line_load_x_N_per_m,
                line_load_y_N_per_m=resolved.line_load_y_N_per_m,
            )
            solution = solve_local_full_field(case, system)
            fields = recover_fields(solution)
            statistics = compute_stress_statistics(fields)
            write_vtu(fields, temporary_artifacts["vtu"])
            write_nodes_csv(fields, temporary_artifacts["nodes_csv"])
            write_elements_csv(fields, temporary_artifacts["elements_csv"])
            write_active_contact_facets_csv(
                solution, temporary_artifacts["active_contact_facets_csv"]
            )
            deformation_scale = write_pngs(
                solution,
                fields,
                temporary_artifacts,
                args.deformation_scale,
                active_facets=solution.active_facets,
                result_title=RESULT_TITLE,
            )
            summary = _build_summary(
                case,
                resolved,
                mesh_summary,
                selection,
                solution,
                statistics,
                deformation_scale,
                artifacts,
            )
            serialized = _serialize_summary(summary)
            _write_temporary_summary(
                serialized, temporary_artifacts["summary_json"]
            )
            _publish_results(temporary_artifacts, artifacts)
    except (
        ForceModelValidationError,
        ForceModelError,
        GrindingCaseValidationError,
        GrindingLoadError,
        MeshGenerationError,
        MeshValidationError,
        SolverError,
        PostprocessError,
        GrindingWorkflowError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Grinding post-process error: {exc}", file=sys.stderr)
        return 2

    _print_summary(resolved, selection, solution, statistics, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
