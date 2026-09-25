"""Formal artifacts and transactional publication for single-grain contact."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import meshio
import numpy as np
from PIL import Image

from grindcae import __version__
from grindcae.elastoplastic_fem.recovery import recover_snapshot_fields
from grindcae.plotting import plot_style, save_png, style_axis

from .models import SingleGrainContactCase
from .workflow import ContactTrajectoryResult


RESULT_FORMAT = "grindcae_single_grain_real_contact_v1"
ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "contact_history_csv": "contact_history.csv",
    "contact_points_csv": "contact_points.csv",
    "final_vtu": "final_results.vtu",
    "final_nodes_csv": "final_nodes.csv",
    "final_elements_csv": "final_elements.csv",
    "residual_surface_profile_csv": "residual_surface_profile.csv",
    "reference_mesh_msh": "reference_mesh.msh",
    "reaction_history_png": "reaction_history.png",
    "contact_pressure_png": "contact_pressure.png",
    "contact_state_png": "contact_state.png",
    "energy_history_png": "energy_history.png",
    "final_fields_png": "final_fields.png",
    "residual_groove_png": "residual_groove.png",
}

CONTACT_HISTORY_FIELDS = (
    "index", "segment", "center_x_m", "center_y_m", "normal_reaction_N",
    "tangential_reaction_N", "normal_reaction_per_thickness_N_per_m",
    "tangential_reaction_per_thickness_N_per_m", "contact_count", "open_count",
    "stick_count", "slip_count", "maximum_pressure_Pa", "maximum_penetration_m",
    "newton_iterations", "retry_count", "substep_count", "balance_residual_N",
    "contact_work_increment_J", "friction_dissipation_increment_J",
    "plastic_dissipation_increment_J", "elastic_strain_energy_J",
    "numerical_contact_stored_energy_J", "cumulative_contact_work_J",
    "cumulative_friction_dissipation_J", "cumulative_plastic_dissipation_J",
    "energy_balance_residual_J",
)
CONTACT_POINT_FIELDS = (
    "node_id", "x_m", "y_m", "tributary_length_m", "gap_m", "pressure_Pa",
    "normal_x", "normal_y", "tangent_x", "tangent_y",
    "tangential_traction_Pa", "status",
)
FINAL_NODE_FIELDS = (
    "node_id", "x_m", "y_m", "ux_m", "uy_m", "displacement_magnitude_m",
)
FINAL_ELEMENT_FIELDS = (
    "element_id", "node_0", "node_1", "node_2", "von_mises_stress_Pa",
    "equivalent_plastic_strain",
)
RESIDUAL_SURFACE_FIELDS = (
    "node_id", "reference_x_m", "reference_y_m", "residual_x_m", "residual_y_m",
    "residual_vertical_displacement_m",
)


def artifact_paths(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    return {key: root / name for key, name in ARTIFACT_FILENAMES.items()}


def _final_fields(result: ContactTrajectoryResult):
    return recover_snapshot_fields(
        result.case.to_elastoplastic_fem_case(),
        result.prepared_mesh.structural,
        result.records[-1].state,  # compatible converged-state contract
    )


def _write_history(result: ContactTrajectoryResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONTACT_HISTORY_FIELDS); writer.writeheader()
        for record in result.records:
            writer.writerow({name: getattr(record, name) for name in CONTACT_HISTORY_FIELDS})


def _write_contact_points(result: ContactTrajectoryResult, path: Path) -> None:
    record = max(result.records, key=lambda item: item.normal_reaction_N)
    prepared = result.prepared_mesh
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CONTACT_POINT_FIELDS)
        for node, coordinate, weight, motion, pressure, state in zip(
            prepared.candidate_node_ids, prepared.candidate_reference_coordinates_m,
            prepared.candidate_tributary_lengths_m, record.state.contact_kinematics,
            record.state.contact_pressure_Pa, record.state.contact_states,
        ):
            writer.writerow((int(node), *coordinate, weight, motion.gap_m, pressure, *motion.normal, *motion.tangent, state.tangential_traction_Pa, state.status))


def _write_fields(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    fields = _final_fields(result)
    points = np.column_stack((fields.node_coordinates_m, np.zeros(fields.node_count)))
    displacement = np.column_stack((fields.nodal_displacements_m, np.zeros(fields.node_count)))
    meshio.write(
        paths["final_vtu"],
        meshio.Mesh(
            points=points,
            cells=[("triangle", fields.element_connectivity)],
            point_data={"residual_displacement_m": displacement, "residual_displacement_magnitude_m": fields.displacement_magnitude_m},
            cell_data={"von_mises_stress_Pa": [fields.von_mises_stress_Pa], "equivalent_plastic_strain": [fields.equivalent_plastic_strain]},
        ),
    )
    with paths["final_nodes_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FINAL_NODE_FIELDS)
        for i in range(fields.node_count): writer.writerow((i, *fields.node_coordinates_m[i], *fields.nodal_displacements_m[i], fields.displacement_magnitude_m[i]))
    with paths["final_elements_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FINAL_ELEMENT_FIELDS)
        for i, nodes in enumerate(fields.element_connectivity): writer.writerow((i, *nodes, fields.von_mises_stress_Pa[i], fields.equivalent_plastic_strain[i]))
    top_nodes = result.prepared_mesh.candidate_node_ids
    with paths["residual_surface_profile_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(RESIDUAL_SURFACE_FIELDS)
        for node in top_nodes:
            current = fields.node_coordinates_m[node] + fields.nodal_displacements_m[node]
            writer.writerow((int(node), *fields.node_coordinates_m[node], *current, fields.nodal_displacements_m[node, 1]))
    if result.reference_mesh_path is not None and result.reference_mesh_path.is_file():
        shutil.copy2(result.reference_mesh_path, paths["reference_mesh_msh"])
    else:
        meshio.write(paths["reference_mesh_msh"], meshio.Mesh(points=points, cells=[("triangle", fields.element_connectivity)]), file_format="gmsh22", binary=False)


def _write_plots(
    result: ContactTrajectoryResult,
    paths: dict[str, Path],
    *,
    peak_record=None,
) -> None:
    import matplotlib.pyplot as plt
    fields = _final_fields(result)
    with plt.rc_context(plot_style()):
        figure, axis = plt.subplots(figsize=(7.2, 4.2)); x = [r.index for r in result.records]
        axis.plot(x, [r.normal_reaction_N for r in result.records], label="Normal reaction")
        axis.plot(x, [r.tangential_reaction_N for r in result.records], label="Tangential reaction")
        axis.set_title("Single-grain reaction history"); axis.set_xlabel("Committed target"); axis.set_ylabel("Total force (N)"); style_axis(axis, grid_axis="y"); axis.legend(); save_png(figure, paths["reaction_history_png"]); plt.close(figure)
        peak = peak_record or max(result.records, key=lambda item: item.normal_reaction_N)
        figure, axis = plt.subplots(figsize=(7.2, 4.2)); axis.plot(result.prepared_mesh.candidate_reference_coordinates_m[:, 0] * 1e6, peak.state.contact_pressure_Pa / 1e6)
        axis.set_title("Contact pressure at peak indentation"); axis.set_xlabel("Surface x (um)"); axis.set_ylabel("Pressure (MPa)"); style_axis(axis, grid_axis="y"); save_png(figure, paths["contact_pressure_png"]); plt.close(figure)
        status_value = {"open": 0.0, "stick": 1.0, "slip": 2.0}
        figure, axis = plt.subplots(figsize=(7.2, 3.8))
        state_x = result.prepared_mesh.candidate_reference_coordinates_m[:, 0] * 1e6
        state_y = [status_value[item.status] for item in peak.state.contact_states]
        axis.scatter(state_x, state_y, c=state_y, cmap="viridis", vmin=0.0, vmax=2.0, s=28)
        axis.set_yticks((0.0, 1.0, 2.0), labels=("open", "stick", "slip"))
        axis.set_title("Contact state at peak normal reaction"); axis.set_xlabel("Surface x (um)"); axis.set_ylabel("Contact state"); style_axis(axis, grid_axis="x"); save_png(figure, paths["contact_state_png"]); plt.close(figure)
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.plot(x, [r.cumulative_contact_work_J for r in result.records], label="Signed contact work")
        axis.plot(x, [r.elastic_strain_energy_J + r.numerical_contact_stored_energy_J for r in result.records], label="Stored energy")
        axis.plot(x, [r.cumulative_friction_dissipation_J for r in result.records], label="Friction dissipation")
        axis.plot(x, [r.cumulative_plastic_dissipation_J for r in result.records], label="Plastic dissipation")
        axis.plot(x, [r.energy_balance_residual_J for r in result.records], label="Balance residual", linestyle="--")
        axis.set_title("Single-grain energy history"); axis.set_xlabel("Committed target"); axis.set_ylabel("Energy (J)"); style_axis(axis, grid_axis="y"); axis.legend(); save_png(figure, paths["energy_history_png"]); plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8)); triang = fields.element_connectivity
        p0 = axes[0].tripcolor(fields.node_coordinates_m[:, 0] * 1e6, fields.node_coordinates_m[:, 1] * 1e6, triang, facecolors=fields.von_mises_stress_Pa / 1e6); figure.colorbar(p0, ax=axes[0], label="MPa"); axes[0].set_title("Residual von Mises stress")
        p1 = axes[1].tripcolor(fields.node_coordinates_m[:, 0] * 1e6, fields.node_coordinates_m[:, 1] * 1e6, triang, facecolors=fields.equivalent_plastic_strain); figure.colorbar(p1, ax=axes[1]); axes[1].set_title("Equivalent plastic strain")
        for axis in axes: axis.set_aspect("equal"); axis.set_xlabel("x (um)"); axis.set_ylabel("y (um)")
        figure.tight_layout(); save_png(figure, paths["final_fields_png"]); plt.close(figure)
        nodes = result.prepared_mesh.candidate_node_ids; current = fields.node_coordinates_m[nodes] + fields.nodal_displacements_m[nodes]
        figure, axis = plt.subplots(figsize=(7.2, 4.2)); axis.plot(current[:, 0] * 1e6, fields.nodal_displacements_m[nodes, 1] * 1e6)
        axis.set_title("Residual fixed-mesh surface displacement"); axis.set_xlabel("Surface x (um)"); axis.set_ylabel("Residual vertical displacement (um)"); style_axis(axis, grid_axis="y"); save_png(figure, paths["residual_groove_png"]); plt.close(figure)


def _summary(result: ContactTrajectoryResult, paths: dict[str, Path]) -> dict[str, object]:
    peak = max(result.records, key=lambda item: item.normal_reaction_N); final = result.records[-1]
    return {
        "result_format": RESULT_FORMAT, "grindcae_version": __version__,
        "result_status": "completed_with_applicability_warning" if result.applicability_status == "warning" else "completed",
        "input_fingerprint": result.case.input_fingerprint(),
        "units": {"coordinate": "m", "thickness": "m", "pressure": "Pa", "line_traction": "N/m", "force_per_thickness": "N/m", "total_force": "N", "displacement": "m", "stress": "Pa", "energy": "J"},
        "workpiece_thickness_m": result.case.analysis.thickness,
        "scope": {"dimension": "2D", "kinematics": "small_strain_fixed_mesh", "physical_material_removal": False, "residual_groove_interpretation": "unloaded fixed-mesh surface displacement profile"},
        "material": result.case.material.to_dict(), "case": result.case.to_dict(),
        "contact_controls": {"normal_penalty_Pa_per_m": result.normal_penalty_Pa_per_m, "tangential_penalty_Pa_per_m": result.tangential_penalty_Pa_per_m, "normal_algorithm": result.case.contact.normal_algorithm, "friction_coefficient": result.case.contact.friction_coefficient},
        "newton_controls": {
            "reference_force_N": peak.state.reference_force_N,
            "reference_contact_size_m": peak.state.reference_contact_size_m,
            "reference_displacement_m": peak.state.reference_displacement_m,
            "effective_residual_absolute_tolerance_N": peak.state.residual_absolute_tolerance_N,
            "effective_displacement_absolute_tolerance_m": peak.state.displacement_absolute_tolerance_m,
            "residual_relative_tolerance": result.case.newton.residual_relative_tolerance,
            "displacement_relative_tolerance": result.case.newton.displacement_relative_tolerance,
            "maximum_iterations": result.case.newton.maximum_iterations,
        },
        "peak": {"normal_reaction_N": peak.normal_reaction_N, "normal_reaction_per_thickness_N_per_m": peak.normal_reaction_per_thickness_N_per_m, "tangential_reaction_N": peak.tangential_reaction_N, "tangential_reaction_per_thickness_N_per_m": peak.tangential_reaction_per_thickness_N_per_m, "maximum_pressure_Pa": peak.maximum_pressure_Pa, "maximum_penetration_m": peak.maximum_penetration_m},
        "final_state": {"segment": final.segment, "contact_count": final.contact_count, "normal_reaction_N": final.normal_reaction_N, "tangential_reaction_N": final.tangential_reaction_N},
        "energy": {
            "contact_work_convention": "signed_work_on_workpiece",
            "contact_work_J": result.contact_work_J,
            "friction_dissipation_J": result.friction_dissipation_J,
            "plastic_dissipation_J": result.plastic_dissipation_J,
            "initial_elastic_strain_energy_J": result.initial_elastic_strain_energy_J,
            "final_elastic_strain_energy_J": result.final_elastic_strain_energy_J,
            "initial_numerical_contact_stored_energy_J": result.initial_numerical_contact_stored_energy_J,
            "final_numerical_contact_stored_energy_J": result.final_numerical_contact_stored_energy_J,
            "energy_balance_residual_J": result.energy_balance_residual_J,
            "energy_balance_relative_residual": result.energy_balance_relative_residual,
        },
        "maximum_balance_residual_N": result.maximum_balance_residual_N,
        "applicability": result.applicability.to_dict(),
        "artifacts": {key: path.name for key, path in paths.items()},
    }


def write_contact_artifacts(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    if set(paths) != set(ARTIFACT_FILENAMES): raise ValueError("artifact set is invalid")
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    _write_history(result, paths["contact_history_csv"]); _write_contact_points(result, paths["contact_points_csv"]); _write_fields(result, paths); _write_plots(result, paths)
    paths["summary_json"].write_text(json.dumps(_summary(result, paths), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    read_contact_summary(paths["summary_json"])


def _validate_csv(path: Path, expected_fields: tuple[str, ...], label: str) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise ValueError(f"{label} CSV header is invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"{label} CSV cannot be read: {exc}") from exc
    if not rows:
        raise ValueError(f"{label} CSV has no data rows")
    for row_index, row in enumerate(rows):
        for name, raw in row.items():
            if name in {"segment", "status"}:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} CSV row {row_index} has invalid {name}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{label} CSV row {row_index} has non-finite {name}")
    if label == "contact_history":
        if [int(row["index"]) for row in rows] != list(range(len(rows))):
            raise ValueError("contact_history CSV indexes are not continuous")
        if rows[-1]["segment"] != "unloading" or int(rows[-1]["contact_count"]) != 0:
            raise ValueError("contact_history CSV lacks an unloaded open final state")
    if label == "contact_points":
        if any(row["status"] not in {"open", "stick", "slip"} for row in rows):
            raise ValueError("contact_points CSV contains an invalid contact state")


def _validate_vtu(path: Path) -> None:
    try:
        mesh = meshio.read(path)
    except SystemExit as exc:
        raise ValueError("VTU artifact cannot be read") from exc
    except Exception as exc:
        raise ValueError(f"VTU artifact cannot be read: {exc}") from exc
    if not any(block.type == "triangle" and len(block.data) > 0 for block in mesh.cells):
        raise ValueError("VTU artifact has no triangle cells")
    required_point = {"residual_displacement_m", "residual_displacement_magnitude_m"}
    required_cell = {"von_mises_stress_Pa", "equivalent_plastic_strain"}
    if not required_point.issubset(mesh.point_data) or not required_cell.issubset(mesh.cell_data):
        raise ValueError("VTU artifact is missing required contact result fields")
    arrays = [mesh.points, *mesh.point_data.values()]
    for values in mesh.cell_data.values():
        arrays.extend(values)
    if not all(np.all(np.isfinite(np.asarray(values, dtype=float))) for values in arrays):
        raise ValueError("VTU artifact contains non-finite values")


def _validate_reference_mesh(path: Path) -> None:
    try:
        mesh = meshio.read(path)
    except SystemExit as exc:
        raise ValueError("reference mesh cannot be read") from exc
    except Exception as exc:
        raise ValueError(f"reference mesh cannot be read: {exc}") from exc
    if not any(block.type == "triangle" and len(block.data) > 0 for block in mesh.cells):
        raise ValueError("reference mesh has no triangle cells")
    if not np.all(np.isfinite(np.asarray(mesh.points, dtype=float))):
        raise ValueError("reference mesh contains non-finite coordinates")


def _validate_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("registered image is not PNG")
            image.verify()
        with Image.open(path) as image:
            pixels = np.asarray(image)
    except Exception as exc:
        raise ValueError(f"PNG artifact cannot be decoded: {exc}") from exc
    if pixels.size == 0 or not np.all(np.isfinite(pixels.astype(float))):
        raise ValueError("PNG artifact has empty or non-finite pixels")


def _validate_registered_artifacts(source: Path, artifacts: dict[str, object]) -> None:
    resolved: dict[str, Path] = {}
    for key, name in artifacts.items():
        target = (source.parent / str(name)).resolve()
        if target.parent != source.parent or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError("registered artifact is missing or unsafe")
        resolved[key] = target
    _validate_csv(resolved["contact_history_csv"], CONTACT_HISTORY_FIELDS, "contact_history")
    _validate_csv(resolved["contact_points_csv"], CONTACT_POINT_FIELDS, "contact_points")
    _validate_csv(resolved["final_nodes_csv"], FINAL_NODE_FIELDS, "final_nodes")
    _validate_csv(resolved["final_elements_csv"], FINAL_ELEMENT_FIELDS, "final_elements")
    _validate_csv(resolved["residual_surface_profile_csv"], RESIDUAL_SURFACE_FIELDS, "residual_surface_profile")
    _validate_vtu(resolved["final_vtu"])
    _validate_reference_mesh(resolved["reference_mesh_msh"])
    for key in (
        "reaction_history_png", "contact_pressure_png", "contact_state_png",
        "energy_history_png", "final_fields_png", "residual_groove_png",
    ):
        _validate_png(resolved[key])


def read_contact_summary(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve(); text = source.read_text(encoding="utf-8")
    if "NaN" in text or "Infinity" in text: raise ValueError("summary contains nonfinite data")
    value = json.loads(text)
    if value.get("result_format") != RESULT_FORMAT: raise ValueError("wrong contact result_format")
    thickness = float(value["workpiece_thickness_m"]); peak = value["peak"]
    case = SingleGrainContactCase.from_mapping(value.get("case"))
    if value.get("input_fingerprint") != case.input_fingerprint():
        raise ValueError("input fingerprint is inconsistent with the saved case")
    applicability = value.get("applicability")
    if not isinstance(applicability, dict):
        raise ValueError("applicability diagnostics are missing")
    if applicability.get("hard_stop_reasons") not in ([], ()):
        raise ValueError("applicability diagnostics contain a hard-stop result")
    expected_status = (
        "completed_with_applicability_warning"
        if applicability.get("status") == "warning"
        else "completed"
    )
    if value.get("result_status") != expected_status:
        raise ValueError("applicability result status is inconsistent")
    final_state = value.get("final_state")
    if (
        not isinstance(final_state, dict)
        or final_state.get("segment") != "unloading"
        or int(final_state.get("contact_count", -1)) != 0
        or abs(float(final_state.get("normal_reaction_N", math.inf))) > 1.0e-8
        or abs(float(final_state.get("tangential_reaction_N", math.inf))) > 1.0e-8
    ):
        raise ValueError("result does not contain a completed unloaded open final state")
    for total, unit in (("normal_reaction_N", "normal_reaction_per_thickness_N_per_m"), ("tangential_reaction_N", "tangential_reaction_per_thickness_N_per_m")):
        if not math.isclose(float(peak[total]), thickness * float(peak[unit]), rel_tol=1e-10, abs_tol=1e-12): raise ValueError("force and thickness contract is inconsistent")
    energy = value.get("energy")
    if not isinstance(energy, dict) or energy.get("contact_work_convention") != "signed_work_on_workpiece":
        raise ValueError("energy convention is missing or invalid")
    energy_fields = (
        "contact_work_J", "friction_dissipation_J", "plastic_dissipation_J",
        "initial_elastic_strain_energy_J", "final_elastic_strain_energy_J",
        "initial_numerical_contact_stored_energy_J", "final_numerical_contact_stored_energy_J",
        "energy_balance_residual_J", "energy_balance_relative_residual",
    )
    if not all(math.isfinite(float(energy.get(name))) for name in energy_fields):
        raise ValueError("energy diagnostics contain a non-finite value")
    if any(float(energy[name]) < 0.0 for name in (
        "friction_dissipation_J", "plastic_dissipation_J",
        "initial_elastic_strain_energy_J", "final_elastic_strain_energy_J",
        "initial_numerical_contact_stored_energy_J", "final_numerical_contact_stored_energy_J",
        "energy_balance_relative_residual",
    )):
        raise ValueError("energy diagnostics contain an invalid negative value")
    newton_controls = value.get("newton_controls")
    if not isinstance(newton_controls, dict):
        raise ValueError("Newton reference controls are missing")
    for name in (
        "reference_force_N", "reference_contact_size_m", "reference_displacement_m",
        "effective_residual_absolute_tolerance_N", "effective_displacement_absolute_tolerance_m",
    ):
        number = float(newton_controls.get(name))
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("Newton reference controls are invalid")
    artifacts = value.get("artifacts", {})
    if set(artifacts) != set(ARTIFACT_FILENAMES): raise ValueError("artifact index is invalid")
    _validate_registered_artifacts(source, artifacts)
    return value


def _stage(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp"); shutil.copy2(source, staged); return staged


def publish_contact_result(temporary: dict[str, Path], final: dict[str, Path]) -> None:
    if set(temporary) != set(final) or set(final) != set(ARTIFACT_FILENAMES): raise ValueError("publication set is invalid")
    read_contact_summary(temporary["summary_json"]); root = final["summary_json"].parent; existed = root.exists(); root.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}; backups: dict[str, Path] = {}; published: list[str] = []
    try:
        for key in order: staged[key] = _stage(temporary[key], final[key], "new")
        for key in order:
            if final[key].exists(): backups[key] = _stage(final[key], final[key], "backup")
        for key in order: os.replace(staged.pop(key), final[key]); published.append(key)
        read_contact_summary(final["summary_json"])
    except Exception as exc:
        for key in reversed(published):
            if key in backups: os.replace(backups.pop(key), final[key])
            else: final[key].unlink(missing_ok=True)
        raise RuntimeError(f"contact artifact publication failed: {exc}") from exc
    finally:
        for item in (*staged.values(), *backups.values()): item.unlink(missing_ok=True)
        if not existed and root.exists():
            try: root.rmdir()
            except OSError: pass
