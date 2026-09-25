"""Schema-2 formal artifacts and strict rereading for GrindCAE 3.0.1."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import os
import shutil
import uuid

import meshio
import numpy as np
from PIL import Image

from grindcae import __version__
from grindcae.elastoplastic_fem.recovery import recover_snapshot_fields

from .exporters import (
    CONTACT_HISTORY_FIELDS,
    _summary as _v1_summary,
    _validate_png,
    _validate_reference_mesh,
    _validate_vtu,
    _write_history,
    _write_plots,
)
from .material_topology import MaterialState, PreparedMaterialTopology
from .models import SingleGrainContactCase
from .workflow import ContactTrajectoryResult
from .geometry import RigidRoundedWedgeGrain, effective_wedge_angles


RESULT_FORMAT_V2 = "grindcae_single_grain_real_contact_v2"
ARTIFACT_FILENAMES_V2 = {
    "summary_json": "summary.json",
    "contact_history_csv": "contact_history.csv",
    "contact_points_csv": "contact_points.csv",
    "material_topology_csv": "material_topology.csv",
    "topology_history_csv": "topology_history.csv",
    "free_surface_csv": "free_surface.csv",
    "final_vtu": "final_results.vtu",
    "effective_mesh_vtu": "effective_mesh.vtu",
    "final_nodes_csv": "final_nodes.csv",
    "final_elements_csv": "final_elements.csv",
    "unloaded_surface_profile_csv": "unloaded_surface_profile.csv",
    "reference_mesh_msh": "reference_mesh.msh",
    "reaction_history_png": "reaction_history.png",
    "contact_pressure_png": "contact_pressure.png",
    "contact_state_png": "contact_state.png",
    "energy_history_png": "energy_history.png",
    "final_fields_png": "final_fields.png",
    "unloaded_surface_profile_png": "unloaded_surface_profile.png",
    "grain_geometry_png": "grain_geometry.png",
    "effective_mesh_surface_png": "effective_mesh_surface.png",
}

CONTACT_POINT_FIELDS_V2 = (
    "background_node_id", "active_node_id", "reference_x_m", "reference_y_m",
    "tributary_length_m", "candidate_key", "gap_m", "pressure_Pa", "normal_x",
    "normal_y", "tangent_x", "tangent_y", "feature_id", "feature_type",
    "closest_point_unique", "at_feature_junction", "incident_edge_ids",
    "surface_component_id", "surface_classification", "reference_normal_x",
    "reference_normal_y", "reference_normal_unique", "status",
)
TOPOLOGY_FIELDS = (
    "background_element_id", "material_state", "material_state_code", "active",
)
TOPOLOGY_HISTORY_FIELDS = (
    "index", "segment", "active_count", "damaged_count", "removed_count",
    "active_area_m2", "active_node_count", "active_dof_count", "surface_edge_count",
    "surface_node_count", "surface_length_m", "component_count", "diagnostic_status",
)
FREE_SURFACE_FIELDS = (
    "index", "segment", "edge_index", "background_node_0", "background_node_1",
    "reference_x0_m", "reference_y0_m", "reference_x1_m", "reference_y1_m",
    "current_x0_m", "current_y0_m", "current_x1_m", "current_y1_m",
    "classification", "incident_active_element_id", "direction_x", "direction_y",
    "normal_x", "normal_y", "reference_length_m", "contact_eligible",
)
FINAL_NODE_FIELDS_V2 = (
    "active_node_id", "background_node_id", "x_m", "y_m", "ux_m", "uy_m",
    "displacement_magnitude_m",
)
FINAL_ELEMENT_FIELDS_V2 = (
    "active_element_id", "background_element_id", "node_0", "node_1", "node_2",
    "von_mises_stress_Pa", "equivalent_plastic_strain",
)
SURFACE_PROFILE_FIELDS_V2 = (
    "order", "background_node_id", "active_node_id", "reference_x_m", "reference_y_m",
    "current_x_m", "current_y_m", "residual_vertical_displacement_m",
)


def artifact_paths_v2(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    return {key: root / name for key, name in ARTIFACT_FILENAMES_V2.items()}


def _require_topology(result: ContactTrajectoryResult) -> PreparedMaterialTopology:
    topology = result.material_topology
    if not isinstance(topology, PreparedMaterialTopology):
        raise ValueError("schema-2 result requires prepared material topology")
    if result.case.single_grain_contact_schema_version != 2:
        raise ValueError("v2 artifacts require schema 2")
    return topology


def _fields(result: ContactTrajectoryResult):
    return recover_snapshot_fields(
        result.case.to_elastoplastic_fem_case(),
        result.prepared_mesh.structural,
        result.records[-1].state,
    )


def _peak_contact_record(result: ContactTrajectoryResult):
    return max(
        result.records,
        key=lambda record: (
            record.maximum_pressure_Pa,
            record.contact_count,
            abs(record.normal_reaction_N),
            abs(record.tangential_reaction_N),
            -record.index,
        ),
    )


def _write_contact_points(result: ContactTrajectoryResult, path: Path) -> None:
    record = _peak_contact_record(result)
    prepared = result.prepared_mesh
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(CONTACT_POINT_FIELDS_V2)
        for background_node, active_node, coordinate, weight, key, edge_ids, component_id, classification, reference_normal, reference_normal_unique, motion, pressure, state in zip(
            prepared.candidate_background_node_ids, prepared.candidate_node_ids,
            prepared.candidate_reference_coordinates_m, prepared.candidate_tributary_lengths_m,
            prepared.candidate_keys, prepared.candidate_incident_edge_ids,
            prepared.candidate_surface_component_ids, prepared.candidate_surface_classifications,
            prepared.candidate_reference_normals, prepared.candidate_reference_normal_unique,
            record.state.contact_kinematics,
            record.state.contact_pressure_Pa, record.state.contact_states, strict=True,
        ):
            writer.writerow((
                int(background_node), int(active_node), *coordinate, weight,
                json.dumps(key, separators=(",", ":")), motion.signed_gap_m, pressure,
                *motion.outward_normal, *motion.consistent_tangent, motion.feature_id,
                motion.feature_type, int(motion.closest_point_unique),
                int(motion.at_feature_junction), json.dumps(edge_ids, separators=(",", ":")),
                int(component_id), classification, *reference_normal,
                int(reference_normal_unique), state.status,
            ))


def _write_topology(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    topology = _require_topology(result)
    with paths["material_topology_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(TOPOLOGY_FIELDS)
        active = set(int(i) for i in topology.active_element_ids)
        for element_id, state in enumerate(topology.element_states):
            material_state = MaterialState(int(state))
            writer.writerow((element_id, material_state.name, int(material_state), int(element_id in active)))
    area = result.prepared_mesh.structural.element_areas_m2
    surface = topology.free_surface
    with paths["topology_history_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(TOPOLOGY_HISTORY_FIELDS)
        for record in result.records:
            writer.writerow((
                record.index, record.segment, topology.active_element_ids.size, 0,
                topology.removed_element_ids.size, float(np.sum(area)),
                topology.active_node_ids.size, topology.active_total_degrees_of_freedom,
                surface.edge_node_ids.shape[0], surface.node_ids.size,
                float(np.sum(surface.edge_lengths_m)), 1, "valid",
            ))
    with paths["free_surface_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FREE_SURFACE_FIELDS)
        for record in result.records:
            displacement = record.state.displacement_vector_m
            current = surface.reference_coordinates_m + displacement[surface.candidate_component_dofs]
            background_to_order = {int(node): index for index, node in enumerate(surface.node_ids)}
            for edge_index, (a, b) in enumerate(surface.edge_node_ids):
                first = background_to_order[int(a)]; second = background_to_order[int(b)]
                vector = surface.reference_coordinates_m[second] - surface.reference_coordinates_m[first]
                direction = vector / surface.edge_lengths_m[edge_index]
                writer.writerow((
                    record.index, record.segment, edge_index, int(a), int(b),
                    *surface.reference_coordinates_m[first], *surface.reference_coordinates_m[second],
                    *current[first], *current[second], surface.edge_classifications[edge_index],
                    int(surface.incident_active_element_ids[edge_index]), *direction,
                    *surface.edge_outward_normals[edge_index], surface.edge_lengths_m[edge_index], 1,
                ))


def _write_fields(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    topology = _require_topology(result); fields = _fields(result)
    active_to_background_nodes = result.prepared_mesh.active_to_background_node_ids
    active_to_background_elements = result.prepared_mesh.active_to_background_element_ids
    points = np.column_stack((fields.node_coordinates_m, np.zeros(fields.node_count)))
    displacement = np.column_stack((fields.nodal_displacements_m, np.zeros(fields.node_count)))
    meshio.write(paths["final_vtu"], meshio.Mesh(
        points=points, cells=[("triangle", fields.element_connectivity)],
        point_data={
            "background_node_id": active_to_background_nodes,
            "residual_displacement_m": displacement,
            "residual_displacement_magnitude_m": fields.displacement_magnitude_m,
        },
        cell_data={
            "background_element_id": [active_to_background_elements],
            "von_mises_stress_Pa": [fields.von_mises_stress_Pa],
            "equivalent_plastic_strain": [fields.equivalent_plastic_strain],
        },
    ))
    background = result.background_mesh
    if background is None:
        raise ValueError("schema-2 result lacks its background mesh")
    background_points = np.column_stack((background.mesh.p.T, np.zeros(background.node_count)))
    meshio.write(paths["effective_mesh_vtu"], meshio.Mesh(
        points=background_points, cells=[("triangle", background.mesh.t.T)],
        point_data={"background_node_id": np.arange(background.node_count, dtype=np.int32)},
        cell_data={
            "background_element_id": [np.arange(background.triangle_count, dtype=np.int32)],
            "material_state_code": [[int(state) for state in topology.element_states]],
            "active": [[int(state != MaterialState.REMOVED) for state in topology.element_states]],
        },
    ))
    with paths["final_nodes_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FINAL_NODE_FIELDS_V2)
        for active_id, background_id in enumerate(active_to_background_nodes):
            writer.writerow((active_id, int(background_id), *fields.node_coordinates_m[active_id],
                             *fields.nodal_displacements_m[active_id], fields.displacement_magnitude_m[active_id]))
    with paths["final_elements_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(FINAL_ELEMENT_FIELDS_V2)
        for active_id, (background_id, nodes) in enumerate(zip(active_to_background_elements, fields.element_connectivity, strict=True)):
            writer.writerow((active_id, int(background_id), *nodes, fields.von_mises_stress_Pa[active_id], fields.equivalent_plastic_strain[active_id]))
    surface = topology.free_surface
    current = surface.reference_coordinates_m + result.records[-1].state.displacement_vector_m[surface.candidate_component_dofs]
    with paths["unloaded_surface_profile_csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(SURFACE_PROFILE_FIELDS_V2)
        for order, (background_id, active_id, reference, point) in enumerate(zip(
            surface.node_ids, result.prepared_mesh.candidate_node_ids,
            surface.reference_coordinates_m, current, strict=True,
        )):
            writer.writerow((order, int(background_id), int(active_id), *reference, *point, point[1] - reference[1]))
    if result.reference_mesh_path is not None and result.reference_mesh_path.is_file():
        shutil.copy2(result.reference_mesh_path, paths["reference_mesh_msh"])
    else:
        meshio.write(paths["reference_mesh_msh"], meshio.Mesh(points=background_points, cells=[("triangle", background.mesh.t.T)]), file_format="gmsh22", binary=False)


def _wedge_geometry_plot_points(
    wedge: RigidRoundedWedgeGrain, *, arc_point_count: int = 121
) -> np.ndarray:
    if type(arc_point_count) is not int or arc_point_count < 2:
        raise ValueError("arc_point_count must be an integer of at least two")
    center = wedge.tip_center_local_m
    rake_angle = math.atan2(
        wedge.rake_tangent_point_local_m[1] - center[1],
        wedge.rake_tangent_point_local_m[0] - center[0],
    )
    clearance_angle = math.atan2(
        wedge.clearance_tangent_point_local_m[1] - center[1],
        wedge.clearance_tangent_point_local_m[0] - center[0],
    )
    clockwise_span = (rake_angle - clearance_angle) % (2.0 * math.pi)
    arc_angles = np.linspace(
        rake_angle, rake_angle - clockwise_span, arc_point_count
    )
    arc = center + wedge.tip_radius_m * np.column_stack(
        (np.cos(arc_angles), np.sin(arc_angles))
    )
    local = np.vstack(
        (
            wedge.rake_top_point_local_m,
            wedge.rake_tangent_point_local_m,
            arc,
            wedge.clearance_top_point_local_m,
            wedge.rake_top_point_local_m,
        )
    )
    return np.asarray([wedge.local_to_global(point) for point in local], dtype=float)


def _write_v2_plots(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    alias = dict(paths)
    alias["residual_groove_png"] = paths["unloaded_surface_profile_png"]
    peak = _peak_contact_record(result)
    _write_plots(result, alias, peak_record=peak)
    import matplotlib.pyplot as plt
    topology = _require_topology(result); grain = result.case.grain
    for key, title in (("grain_geometry_png", "Rigid grain geometry"), ("effective_mesh_surface_png", "Effective mesh and free surface")):
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        if key == "grain_geometry_png":
            if grain.type == "rounded_circle":
                theta = np.linspace(0.0, 2.0 * math.pi, 241)
                axis.plot(grain.radius_m * np.cos(theta) * 1e6, grain.radius_m * np.sin(theta) * 1e6, color="#334e68", linewidth=2.0)
                axis.scatter([0.0], [0.0], color="#c43c39", s=20, label="Reference point")
            else:
                wedge = RigidRoundedWedgeGrain(
                    tip_radius_m=grain.tip_radius_m,
                    wedge_height_m=grain.wedge_height_m,
                    intrinsic_rake_angle_rad=grain.intrinsic_rake_angle_rad,
                    intrinsic_clearance_angle_rad=grain.intrinsic_clearance_angle_rad,
                    pose_angle_rad=grain.pose_angle_rad,
                    reference_x_m=0.0,
                    reference_y_m=0.0,
                )
                displayed = _wedge_geometry_plot_points(wedge)
                axis.plot(displayed[:, 0] * 1e6, displayed[:, 1] * 1e6, color="#334e68", linewidth=2.0)
                axis.scatter([0.0], [0.0], color="#c43c39", s=20, label="Tip reference")
            axis.set_aspect("equal")
            axis.set_xlabel("Local/global x (um)")
            axis.set_ylabel("Local/global y (um)")
            axis.legend(loc="best")
        else:
            mesh = result.background_mesh.mesh
            removed = np.asarray(topology.removed_element_ids, dtype=np.int32)
            active = np.asarray(topology.active_element_ids, dtype=np.int32)
            axis.triplot(mesh.p[0] * 1e6, mesh.p[1] * 1e6, mesh.t[:, active].T, color="#738290", linewidth=0.55)
            if removed.size:
                axis.tripcolor(mesh.p[0] * 1e6, mesh.p[1] * 1e6, mesh.t[:, removed].T, facecolors=np.ones(removed.size), cmap="Reds", alpha=0.35, edgecolors="#d66")
            points = topology.free_surface.reference_coordinates_m
            axis.plot(points[:, 0] * 1e6, points[:, 1] * 1e6, color="#c43c39", linewidth=2.0)
            axis.set_aspect("equal")
        axis.set_title(title); figure.tight_layout(); figure.savefig(paths[key], dpi=160); plt.close(figure)


def _summary(result: ContactTrajectoryResult, paths: dict[str, Path]) -> dict[str, object]:
    topology = _require_topology(result)
    base = _v1_summary(result, {**paths, "residual_groove_png": paths["unloaded_surface_profile_png"]})
    peak = _peak_contact_record(result)
    base["newton_controls"] = {
        **base["newton_controls"],
        "reference_force_N": peak.state.reference_force_N,
        "reference_contact_size_m": peak.state.reference_contact_size_m,
        "reference_displacement_m": peak.state.reference_displacement_m,
        "effective_residual_absolute_tolerance_N": peak.state.residual_absolute_tolerance_N,
        "effective_displacement_absolute_tolerance_m": peak.state.displacement_absolute_tolerance_m,
    }
    base["peak"] = {
        "normal_reaction_N": peak.normal_reaction_N,
        "normal_reaction_per_thickness_N_per_m": peak.normal_reaction_per_thickness_N_per_m,
        "tangential_reaction_N": peak.tangential_reaction_N,
        "tangential_reaction_per_thickness_N_per_m": peak.tangential_reaction_per_thickness_N_per_m,
        "maximum_pressure_Pa": peak.maximum_pressure_Pa,
        "maximum_penetration_m": peak.maximum_penetration_m,
    }
    base["result_format"] = RESULT_FORMAT_V2
    base["grindcae_version"] = __version__
    base["automatic_damage_evolution"] = False
    base["physical_material_removal_prediction"] = False
    base["scope"]["physical_material_removal"] = False
    base["scope"]["preset_removed_material_state"] = True
    active_area = float(np.sum(result.prepared_mesh.structural.element_areas_m2))
    background_connectivity = np.asarray(result.background_mesh.mesh.t.T, dtype=np.int32)
    background_coordinates = np.asarray(result.background_mesh.mesh.p.T, dtype=float)
    background_triangles = background_coordinates[background_connectivity]
    first_edge = background_triangles[:, 1] - background_triangles[:, 0]
    second_edge = background_triangles[:, 2] - background_triangles[:, 0]
    background_area = 0.5 * float(
        np.sum(np.abs(first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]))
    )
    base["topology"] = {
        "active_element_count": int(topology.active_element_ids.size),
        "damaged_element_count": 0,
        "removed_element_count": int(topology.removed_element_ids.size),
        "background_element_count": int(len(topology.element_states)),
        "active_node_count": int(topology.active_node_ids.size),
        "active_degrees_of_freedom": topology.active_total_degrees_of_freedom,
        "free_surface_edge_count": int(topology.free_surface.edge_node_ids.shape[0]),
        "free_surface_node_count": int(topology.free_surface.node_ids.size),
        "free_surface_length_m": float(np.sum(topology.free_surface.edge_lengths_m)),
        "component_count": 1,
        "background_area_m2": background_area,
        "active_area_m2": active_area,
        "preset_removed_area_m2": background_area - active_area,
        "preset_removal": result.case.material_topology.preset_removal.to_dict(),
    }
    if result.case.grain.type == "rounded_circle":
        radius = float(result.case.grain.radius_m)
        geometry = {
            "type": "rounded_circle",
            "parameters": result.case.grain.to_dict(),
            "area_m2": math.pi * radius**2,
            "perimeter_m": 2.0 * math.pi * radius,
            "stable_features": [{"feature_id": 0, "feature_type": "circle"}],
        }
    else:
        first_target = result.case.trajectory.targets[0]
        wedge = RigidRoundedWedgeGrain(
            tip_radius_m=result.case.grain.tip_radius_m,
            wedge_height_m=result.case.grain.wedge_height_m,
            intrinsic_rake_angle_rad=result.case.grain.intrinsic_rake_angle_rad,
            intrinsic_clearance_angle_rad=result.case.grain.intrinsic_clearance_angle_rad,
            pose_angle_rad=result.case.grain.pose_angle_rad,
            reference_x_m=first_target.reference_x_m,
            reference_y_m=first_target.reference_y_m,
        )
        direction = np.array([1.0, 0.0]) if result.case.trajectory.nominal_scratch_direction == "positive_x" else np.array([-1.0, 0.0])
        effective = effective_wedge_angles(wedge, direction)
        committed_position_effective_angles = []
        committed_position_contact_features = []
        previous_target = None
        for record, target in zip(
            result.records, result.case.trajectory.targets, strict=True
        ):
            horizontal_increment = (
                0.0
                if previous_target is None
                else target.reference_x_m - previous_target.reference_x_m
            )
            if abs(horizontal_increment) <= wedge.geometry_tolerance_m:
                position_direction = direction
                direction_source = "nominal_scratch_direction"
            else:
                position_direction = np.array(
                    [math.copysign(1.0, horizontal_increment), 0.0], dtype=float
                )
                direction_source = "horizontal_increment"
            position_effective = effective_wedge_angles(wedge, position_direction)
            committed_position_effective_angles.append(
                {
                    "index": record.index,
                    "segment": record.segment,
                    "reference_x_m": target.reference_x_m,
                    "reference_y_m": target.reference_y_m,
                    "direction_source": direction_source,
                    "effective_rake_angle_rad": position_effective.rake_angle_rad,
                    "effective_clearance_angle_rad": position_effective.clearance_angle_rad,
                    "leading_feature_type": position_effective.leading_feature_type,
                    "trailing_feature_type": position_effective.trailing_feature_type,
                }
            )
            feature_counts = {
                name: sum(
                    motion.feature_type == name
                    for motion in record.state.contact_kinematics
                )
                for name in wedge.feature_types
            }
            committed_position_contact_features.append(
                {
                    "index": record.index,
                    "segment": record.segment,
                    "feature_counts": feature_counts,
                }
            )
            previous_target = target
        geometry = {
            "type": "rounded_wedge",
            "parameters": result.case.grain.to_dict(),
            "derived_wedge_width_m": wedge.derived_wedge_width_m,
            "area_m2": wedge.boundary_area_m2,
            "perimeter_m": wedge.boundary_perimeter_m,
            "rake_tangent_point_local_m": wedge.rake_tangent_point_local_m.tolist(),
            "clearance_tangent_point_local_m": wedge.clearance_tangent_point_local_m.tolist(),
            "rake_top_point_local_m": wedge.rake_top_point_local_m.tolist(),
            "clearance_top_point_local_m": wedge.clearance_top_point_local_m.tolist(),
            "intrinsic_rake_angle_rad": wedge.intrinsic_rake_angle_rad,
            "intrinsic_clearance_angle_rad": wedge.intrinsic_clearance_angle_rad,
            "pose_angle_rad": wedge.pose_angle_rad,
            "nominal_scratch_direction": result.case.trajectory.nominal_scratch_direction,
            "effective_rake_angle_rad": effective.rake_angle_rad,
            "effective_clearance_angle_rad": effective.clearance_angle_rad,
            "effective_angle_direction_source": "nominal_scratch_direction",
            "leading_feature_type": effective.leading_feature_type,
            "trailing_feature_type": effective.trailing_feature_type,
            "committed_position_effective_angles": committed_position_effective_angles,
            "committed_position_contact_features": committed_position_contact_features,
            "stable_features": [
                {"feature_id": index, "feature_type": name}
                for index, name in enumerate(wedge.feature_types)
            ],
        }
    base["grain_geometry"] = geometry
    base["artifacts"] = {key: path.name for key, path in paths.items()}
    return base


def write_contact_artifacts_v2(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    if set(paths) != set(ARTIFACT_FILENAMES_V2):
        raise ValueError("v2 artifact set is invalid")
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    _write_history(result, paths["contact_history_csv"])
    _write_contact_points(result, paths["contact_points_csv"])
    _write_topology(result, paths)
    _write_fields(result, paths)
    _write_v2_plots(result, paths)
    paths["summary_json"].write_text(json.dumps(_summary(result, paths), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    read_contact_summary_v2(paths["summary_json"])


def _read_csv(
    path: Path,
    fields: tuple[str, ...],
    label: str,
    *,
    text_fields: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"{label} CSV header is invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"{label} CSV cannot be read: {exc}") from exc
    if not rows:
        raise ValueError(f"{label} CSV has no rows")
    for row_index, row in enumerate(rows):
        for name, raw in row.items():
            if name in text_fields:
                if raw is None or raw == "":
                    raise ValueError(f"{label} CSV row {row_index} has empty {name}")
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} CSV row {row_index} has invalid {name}") from exc
            if not math.isfinite(number):
                raise ValueError(f"{label} CSV row {row_index} has non-finite {name}")
    return rows


def _read_vtu(path: Path, label: str):
    try:
        mesh = meshio.read(path)
    except SystemExit as exc:
        raise ValueError(f"{label} VTU cannot be read") from exc
    except Exception as exc:
        raise ValueError(f"{label} VTU cannot be read: {exc}") from exc
    arrays = [mesh.points, *mesh.point_data.values()]
    for values in mesh.cell_data.values():
        arrays.extend(values)
    if not all(np.all(np.isfinite(np.asarray(values, dtype=float))) for values in arrays):
        raise ValueError(f"{label} VTU contains non-finite values")
    return mesh


def _strict_int(raw: object, label: str) -> int:
    number = float(raw)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be a finite integer")
    return int(number)


def _triangle_cell_data(mesh: meshio.Mesh, name: str, label: str) -> np.ndarray:
    try:
        values = np.asarray(mesh.cell_data_dict[name]["triangle"])
    except KeyError as exc:
        raise ValueError(f"{label} lacks triangle {name}") from exc
    return values


def _validate_unit_vector(x: object, y: object, label: str) -> np.ndarray:
    vector = np.asarray([float(x), float(y)], dtype=float)
    if not np.all(np.isfinite(vector)) or not math.isclose(
        float(np.linalg.norm(vector)), 1.0, rel_tol=1.0e-10, abs_tol=1.0e-10
    ):
        raise ValueError(f"{label} is not a finite unit vector")
    return vector


def read_contact_summary_v2(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve(); text = source.read_text(encoding="utf-8")
    if "NaN" in text or "Infinity" in text:
        raise ValueError("v2 summary contains nonfinite data")
    value = json.loads(text)
    if value.get("result_format") != RESULT_FORMAT_V2:
        raise ValueError("wrong v2 result_format")
    if value.get("grindcae_version") != __version__:
        raise ValueError("v2 result application version is inconsistent")
    case = SingleGrainContactCase.from_mapping(value.get("case"))
    if case.single_grain_contact_schema_version != 2:
        raise ValueError("v2 result requires schema 2")
    if value.get("input_fingerprint") != case.input_fingerprint():
        raise ValueError("v2 input fingerprint is inconsistent")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_FILENAMES_V2):
        raise ValueError("v2 artifact index is invalid")
    resolved: dict[str, Path] = {}
    for key, name in artifacts.items():
        target = (source.parent / str(name)).resolve()
        if target.parent != source.parent or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError("v2 registered artifact is missing or unsafe")
        resolved[key] = target
    topology_rows = _read_csv(
        resolved["material_topology_csv"], TOPOLOGY_FIELDS, "topology",
        text_fields=frozenset({"material_state"}),
    )
    states = [row["material_state"] for row in topology_rows]
    topology_ids = [_strict_int(row["background_element_id"], "topology background element id") for row in topology_rows]
    if topology_ids != list(range(len(topology_rows))):
        raise ValueError("topology background element ids are not continuous and unique")
    for row, state_name in zip(topology_rows, states, strict=True):
        if state_name not in {item.name for item in MaterialState}:
            raise ValueError("topology contains an invalid material state")
        state = MaterialState[state_name]
        if _strict_int(row["material_state_code"], "topology material state code") != int(state):
            raise ValueError("topology material state code is inconsistent")
        expected_active = int(state != MaterialState.REMOVED)
        if _strict_int(row["active"], "topology active flag") != expected_active:
            raise ValueError("topology active flag is inconsistent")
    summary_topology = value.get("topology", {})
    counts = {name: states.count(name) for name in ("ACTIVE", "DAMAGED", "REMOVED")}
    if counts["ACTIVE"] != int(summary_topology.get("active_element_count", -1)) or counts["REMOVED"] != int(summary_topology.get("removed_element_count", -1)):
        raise ValueError("topology counts are inconsistent")
    history_rows = _read_csv(
        resolved["contact_history_csv"], CONTACT_HISTORY_FIELDS, "contact_history",
        text_fields=frozenset({"segment"}),
    )
    if [int(row["index"]) for row in history_rows] != list(range(len(history_rows))):
        raise ValueError("contact_history CSV indexes are not continuous")
    if history_rows[-1]["segment"] != "unloading" or int(history_rows[-1]["contact_count"]) != 0:
        raise ValueError("contact_history CSV lacks an unloaded open final state")
    contact_rows = _read_csv(
        resolved["contact_points_csv"], CONTACT_POINT_FIELDS_V2, "contact_points",
        text_fields=frozenset({"candidate_key", "feature_type", "incident_edge_ids", "surface_classification", "status"}),
    )
    if len({row["candidate_key"] for row in contact_rows}) != len(contact_rows):
        raise ValueError("contact candidate keys are not unique")
    final_nodes = _read_csv(resolved["final_nodes_csv"], FINAL_NODE_FIELDS_V2, "final_nodes")
    active_node_ids = [_strict_int(row["active_node_id"], "final active node id") for row in final_nodes]
    if active_node_ids != list(range(len(final_nodes))):
        raise ValueError("final active node ids are not continuous and unique")
    active_nodes = {_strict_int(row["background_node_id"], "final background node id") for row in final_nodes}
    if len(active_nodes) != len(final_nodes):
        raise ValueError("final background node ids are not unique")
    valid_features = {
        (_strict_int(item.get("feature_id"), "grain feature id"), str(item.get("feature_type")))
        for item in value.get("grain_geometry", {}).get("stable_features", ())
        if isinstance(item, dict)
    }
    if not valid_features:
        raise ValueError("v2 grain feature table is missing")
    if any(_strict_int(row["background_node_id"], "candidate background node id") not in active_nodes for row in contact_rows):
        raise ValueError("contact candidate references a removed or unknown background node")
    if any(row["status"] not in {"open", "stick", "slip"} for row in contact_rows):
        raise ValueError("contact_points CSV contains an invalid contact state")
    for row in contact_rows:
        feature = (
            _strict_int(row["feature_id"], "contact feature id"),
            row["feature_type"],
        )
        if feature not in valid_features:
            raise ValueError("contact point references an invalid grain feature")
        unique = _strict_int(row["closest_point_unique"], "closest-point uniqueness flag")
        if unique not in {0, 1}:
            raise ValueError("closest-point uniqueness flag is invalid")
        if unique == 0 and (float(row["pressure_Pa"]) > 0.0 or float(row["gap_m"]) <= 0.0):
            raise ValueError("contact point has a nonunique closest point in contact")
    topology_history = _read_csv(
        resolved["topology_history_csv"], TOPOLOGY_HISTORY_FIELDS, "topology_history",
        text_fields=frozenset({"segment", "diagnostic_status"}),
    )
    if [_strict_int(row["index"], "topology history index") for row in topology_history] != list(range(len(history_rows))):
        raise ValueError("topology history indexes are inconsistent")
    for row, contact_row in zip(topology_history, history_rows, strict=True):
        if row["segment"] != contact_row["segment"]:
            raise ValueError("topology history segment is inconsistent")
        expected_counts = {
            "active_count": counts["ACTIVE"],
            "damaged_count": counts["DAMAGED"],
            "removed_count": counts["REMOVED"],
            "active_node_count": len(final_nodes),
            "active_dof_count": 2 * len(final_nodes),
        }
        if any(_strict_int(row[name], f"topology history {name}") != expected for name, expected in expected_counts.items()):
            raise ValueError("topology history counts are inconsistent")
        if row["diagnostic_status"] != "valid":
            raise ValueError("topology history diagnostic status is invalid")
    free_surface_rows = _read_csv(
        resolved["free_surface_csv"], FREE_SURFACE_FIELDS, "free_surface",
        text_fields=frozenset({"segment", "classification"}),
    )
    final_elements = _read_csv(resolved["final_elements_csv"], FINAL_ELEMENT_FIELDS_V2, "final_elements")
    _read_csv(resolved["unloaded_surface_profile_csv"], SURFACE_PROFILE_FIELDS_V2, "unloaded_surface_profile")
    final_mesh = _read_vtu(resolved["final_vtu"], "final")
    final_triangles = final_mesh.cells_dict.get("triangle", np.empty((0, 3), dtype=int))
    if len(final_triangles) != len(final_elements) or len(final_mesh.points) != len(final_nodes):
        raise ValueError("v2 final effective mesh counts are inconsistent")
    if "background_node_id" not in final_mesh.point_data or "background_element_id" not in final_mesh.cell_data:
        raise ValueError("v2 final mesh lacks background ID mappings")
    final_csv_background_nodes = np.asarray(
        [_strict_int(row["background_node_id"], "final background node id") for row in final_nodes],
        dtype=int,
    )
    final_vtu_background_nodes = np.asarray(final_mesh.point_data["background_node_id"], dtype=int)
    if not np.array_equal(final_csv_background_nodes, final_vtu_background_nodes):
        raise ValueError("v2 final effective mesh node mappings are inconsistent")
    final_csv_active_elements = [_strict_int(row["active_element_id"], "final active element id") for row in final_elements]
    if final_csv_active_elements != list(range(len(final_elements))):
        raise ValueError("v2 final effective mesh active element ids are inconsistent")
    final_csv_background_elements = np.asarray(
        [_strict_int(row["background_element_id"], "final background element id") for row in final_elements],
        dtype=int,
    )
    final_vtu_background_elements = np.asarray(
        _triangle_cell_data(final_mesh, "background_element_id", "v2 final effective mesh"),
        dtype=int,
    )
    expected_active_elements = np.asarray(
        [index for index, state in enumerate(states) if state != MaterialState.REMOVED.name],
        dtype=int,
    )
    if not (
        np.array_equal(final_csv_background_elements, final_vtu_background_elements)
        and np.array_equal(final_csv_background_elements, expected_active_elements)
    ):
        raise ValueError("v2 final effective mesh element mappings are inconsistent")
    csv_triangles = np.asarray(
        [[_strict_int(row[name], f"final {name}") for name in ("node_0", "node_1", "node_2")] for row in final_elements],
        dtype=int,
    )
    if not np.array_equal(csv_triangles, np.asarray(final_triangles, dtype=int)):
        raise ValueError("v2 final effective mesh connectivity is inconsistent")
    effective = _read_vtu(resolved["effective_mesh_vtu"], "effective")
    if len(effective.cells_dict.get("triangle", ())) != len(topology_rows):
        raise ValueError("effective background mesh topology count is inconsistent")
    if "material_state_code" not in effective.cell_data or "von_mises_stress_Pa" in effective.cell_data:
        raise ValueError("effective background mesh fields are invalid")
    effective_background_elements = np.asarray(
        _triangle_cell_data(effective, "background_element_id", "effective background mesh"),
        dtype=int,
    )
    effective_state_codes = np.asarray(
        _triangle_cell_data(effective, "material_state_code", "effective background mesh"),
        dtype=int,
    )
    effective_active = np.asarray(
        _triangle_cell_data(effective, "active", "effective background mesh"),
        dtype=int,
    )
    topology_state_codes = np.asarray([int(MaterialState[state]) for state in states], dtype=int)
    topology_active = np.asarray([int(state != MaterialState.REMOVED.name) for state in states], dtype=int)
    if not (
        np.array_equal(effective_background_elements, np.arange(len(topology_rows)))
        and np.array_equal(effective_state_codes, topology_state_codes)
        and np.array_equal(effective_active, topology_active)
    ):
        raise ValueError("effective background mesh topology fields are inconsistent")
    reference = _read_vtu(resolved["reference_mesh_msh"], "reference mesh")
    reference_points = np.asarray(reference.points, dtype=float)
    effective_points = np.asarray(effective.points, dtype=float)
    reference_triangles = np.asarray(reference.cells_dict.get("triangle", ()), dtype=int)
    effective_triangles = np.asarray(effective.cells_dict.get("triangle", ()), dtype=int)
    if len(reference_points) != len(effective_points) or len(reference_triangles) != len(effective_triangles):
        raise ValueError("reference and effective background mesh topology are inconsistent")
    coordinate_scale = max(1.0, float(np.max(np.abs(effective_points))))
    coordinate_tolerance = max(1.0e-15, 128.0 * math.ulp(coordinate_scale))
    effective_node_lookup = {
        tuple(np.round(point / coordinate_tolerance).astype(np.int64)): index
        for index, point in enumerate(effective_points)
    }
    if len(effective_node_lookup) != len(effective_points):
        raise ValueError("effective background mesh coordinates are not unique")
    try:
        reference_to_effective = np.asarray(
            [
                effective_node_lookup[
                    tuple(np.round(point / coordinate_tolerance).astype(np.int64))
                ]
                for point in reference_points
            ],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError("reference and effective background mesh coordinates are inconsistent") from exc
    mapped_reference_triangles = {
        tuple(sorted(reference_to_effective[triangle].tolist()))
        for triangle in reference_triangles
    }
    effective_triangle_set = {
        tuple(sorted(triangle.tolist())) for triangle in effective_triangles
    }
    if mapped_reference_triangles != effective_triangle_set:
        raise ValueError("reference and effective background mesh topology are inconsistent")
    expected_surface_edges = _strict_int(summary_topology.get("free_surface_edge_count"), "summary free-surface edge count")
    expected_surface_nodes = _strict_int(summary_topology.get("free_surface_node_count"), "summary free-surface node count")
    if len(free_surface_rows) != len(history_rows) * expected_surface_edges:
        raise ValueError("free surface row count is inconsistent")
    for position_index in range(len(history_rows)):
        rows = free_surface_rows[position_index * expected_surface_edges:(position_index + 1) * expected_surface_edges]
        if [_strict_int(row["index"], "free surface position index") for row in rows] != [position_index] * expected_surface_edges:
            raise ValueError("free surface position indexes are inconsistent")
        if any(row["segment"] != history_rows[position_index]["segment"] for row in rows):
            raise ValueError("free surface segments are inconsistent")
        if [_strict_int(row["edge_index"], "free surface edge index") for row in rows] != list(range(expected_surface_edges)):
            raise ValueError("free surface edge order is inconsistent")
        surface_nodes = [_strict_int(rows[0]["background_node_0"], "free surface node")]
        for edge_index, row in enumerate(rows):
            node_0 = _strict_int(row["background_node_0"], "free surface node")
            node_1 = _strict_int(row["background_node_1"], "free surface node")
            if node_0 != surface_nodes[-1] or node_0 not in active_nodes or node_1 not in active_nodes:
                raise ValueError("free surface edge adjacency is inconsistent")
            surface_nodes.append(node_1)
            reference_vector = np.asarray(
                [float(row["reference_x1_m"]) - float(row["reference_x0_m"]), float(row["reference_y1_m"]) - float(row["reference_y0_m"])],
                dtype=float,
            )
            length = float(np.linalg.norm(reference_vector))
            if not math.isclose(length, float(row["reference_length_m"]), rel_tol=1.0e-10, abs_tol=1.0e-15):
                raise ValueError("free surface edge length is inconsistent")
            direction_vector = _validate_unit_vector(row["direction_x"], row["direction_y"], "free surface direction")
            normal_vector = _validate_unit_vector(row["normal_x"], row["normal_y"], "free surface normal")
            if not np.allclose(direction_vector, reference_vector / length, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("free surface direction is inconsistent")
            if not np.allclose(normal_vector, np.asarray([-direction_vector[1], direction_vector[0]]), rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("free surface normal is inconsistent")
            active_element = _strict_int(row["incident_active_element_id"], "free surface incident active element")
            if active_element not in set(expected_active_elements.tolist()):
                raise ValueError("free surface incident element is not active")
            background_triangle = np.asarray(effective.cells_dict["triangle"][active_element], dtype=int)
            if not {node_0, node_1}.issubset(set(background_triangle.tolist())):
                raise ValueError("free surface edge is not adjacent to its active element")
            if _strict_int(row["contact_eligible"], "free surface contact eligibility") != 1:
                raise ValueError("free surface contains an ineligible edge")
        if len(surface_nodes) != expected_surface_nodes or len(set(surface_nodes)) != expected_surface_nodes:
            raise ValueError("free surface main-chain nodes are inconsistent")
    for key in (
        "reaction_history_png", "contact_pressure_png", "contact_state_png", "energy_history_png",
        "final_fields_png", "unloaded_surface_profile_png", "grain_geometry_png", "effective_mesh_surface_png",
    ):
        _validate_png(resolved[key])
    final_state = value.get("final_state", {})
    if (
        final_state.get("segment") != "unloading"
        or int(final_state.get("contact_count", -1)) != 0
        or abs(float(final_state.get("normal_reaction_N", math.inf))) > 1.0e-8
        or abs(float(final_state.get("tangential_reaction_N", math.inf))) > 1.0e-8
    ):
        raise ValueError("v2 result lacks an unloaded open final state")
    thickness = float(value.get("workpiece_thickness_m"))
    peak = value.get("peak", {})
    for total, unit in (("normal_reaction_N", "normal_reaction_per_thickness_N_per_m"), ("tangential_reaction_N", "tangential_reaction_per_thickness_N_per_m")):
        if not math.isclose(float(peak.get(total)), thickness * float(peak.get(unit)), rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise ValueError("v2 force and thickness contract is inconsistent")
    applicability = value.get("applicability", {})
    if applicability.get("hard_stop_reasons") not in ([], ()):
        raise ValueError("v2 applicability diagnostics contain a hard-stop result")
    if value.get("automatic_damage_evolution") is not False or value.get("physical_material_removal_prediction") is not False:
        raise ValueError("v2 physical-scope flags are invalid")
    return value


def _stage(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, staged)
    return staged


def publish_contact_result_v2(temporary: dict[str, Path], final: dict[str, Path]) -> None:
    if set(temporary) != set(final) or set(final) != set(ARTIFACT_FILENAMES_V2):
        raise ValueError("v2 publication set is invalid")
    read_contact_summary_v2(temporary["summary_json"])
    root = final["summary_json"].parent; existed = root.exists(); root.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES_V2 if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}; backups: dict[str, Path] = {}; published: list[str] = []
    try:
        for key in order: staged[key] = _stage(temporary[key], final[key], "new")
        for key in order:
            if final[key].exists(): backups[key] = _stage(final[key], final[key], "backup")
        for key in order: os.replace(staged.pop(key), final[key]); published.append(key)
        read_contact_summary_v2(final["summary_json"])
    except Exception as exc:
        for key in reversed(published):
            if key in backups: os.replace(backups.pop(key), final[key])
            else: final[key].unlink(missing_ok=True)
        raise RuntimeError(f"v2 contact artifact publication failed: {exc}") from exc
    finally:
        for item in (*staged.values(), *backups.values()): item.unlink(missing_ok=True)
        if not existed and root.exists():
            try: root.rmdir()
            except OSError: pass


__all__ = [
    "ARTIFACT_FILENAMES_V2", "RESULT_FORMAT_V2", "artifact_paths_v2",
    "publish_contact_result_v2", "read_contact_summary_v2", "write_contact_artifacts_v2",
]
