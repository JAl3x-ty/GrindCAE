"""Schema-3 damage-separation artifacts and strict transactional rereading."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import uuid
from dataclasses import asdict

import matplotlib.pyplot as plt
import meshio
import numpy as np
from PIL import Image

from grindcae import __version__
from grindcae.elastoplastic_fem.recovery import recover_snapshot_fields

from .material_topology import MaterialState, PreparedMaterialTopology
from .geometry import RigidRoundedWedgeGrain
from .exporters_v2 import _wedge_geometry_plot_points
from .models import SingleGrainContactCase
from .workflow import ContactTrajectoryResult


RESULT_FORMAT_V3 = "grindcae_single_grain_damage_separation_v3"
ARTIFACT_FILENAMES_V3 = {
    "summary_json": "summary.json",
    "contact_history_csv": "contact_history.csv",
    "contact_points_csv": "contact_points.csv",
    "material_topology_csv": "material_topology.csv",
    "topology_history_csv": "topology_history.csv",
    "free_surface_csv": "free_surface.csv",
    "damage_history_csv": "damage_history.csv",
    "damage_element_history_csv": "damage_element_history.csv",
    "separation_events_csv": "separation_events.csv",
    "final_damage_state_csv": "final_damage_state.csv",
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
    "damage_evolution_png": "damage_evolution.png",
    "final_damage_field_png": "final_damage_field.png",
    "removal_area_history_png": "removal_area_history.png",
    "processed_surface_png": "processed_surface.png",
}

CONTACT_HISTORY_FIELDS = (
    "index", "segment", "reference_x_m", "reference_y_m", "normal_reaction_N",
    "tangential_reaction_N", "contact_count", "maximum_pressure_Pa",
    "maximum_penetration_m", "newton_iterations", "balance_residual_N",
    "contact_work_increment_J", "friction_dissipation_increment_J",
    "plastic_dissipation_increment_J", "damage_dissipation_increment_J",
    "separation_event_energy_increment_J", "cumulative_contact_work_J",
    "cumulative_friction_dissipation_J", "cumulative_plastic_dissipation_J",
    "cumulative_damage_dissipation_J", "cumulative_separation_event_energy_J",
    "energy_balance_residual_J",
)
CONTACT_POINT_FIELDS = (
    "background_node_id", "active_node_id", "reference_x_m", "reference_y_m",
    "gap_m", "pressure_Pa", "normal_x", "normal_y", "feature_id",
    "feature_type", "candidate_key", "status",
)
TOPOLOGY_FIELDS = (
    "background_element_id", "material_state", "material_state_code", "active",
    "removal_source",
)
TOPOLOGY_HISTORY_FIELDS = (
    "index", "segment", "active_count", "damaged_count", "removed_count",
    "active_area_m2", "active_node_count", "active_dof_count",
    "surface_edge_count", "surface_node_count", "surface_length_m",
)
FREE_SURFACE_FIELDS = (
    "index", "segment", "edge_index", "background_node_0", "background_node_1",
    "reference_x0_m", "reference_y0_m", "reference_x1_m", "reference_y1_m",
    "current_x0_m", "current_y0_m", "current_x1_m", "current_y1_m",
    "classification", "incident_active_element_id", "normal_x", "normal_y",
    "reference_length_m",
)
DAMAGE_HISTORY_FIELDS = (
    "index", "segment", "damaged_element_count", "maximum_damage", "mean_damage",
    "damage_dissipation_J", "separation_event_count", "automatic_removed_area_m2",
)
DAMAGE_ELEMENT_FIELDS = (
    "index", "segment", "background_element_id", "omega", "damage", "status",
    "equivalent_plastic_strain", "initiation_equivalent_plastic_strain",
    "initiation_equivalent_stress_Pa", "post_initiation_plastic_displacement_m",
    "damage_dissipation_density_J_per_m3",
)
SEPARATION_EVENT_FIELDS = (
    "event_index", "target_index", "trajectory_fraction", "marker",
    "background_element_ids", "removed_area_m2", "active_dof_before",
    "active_dof_after", "surface_edges_before", "surface_edges_after",
    "damage_dissipation_increment_J",
)
FINAL_DAMAGE_FIELDS = (
    "background_element_id", "omega", "damage", "material_state", "removal_source",
    "characteristic_length_m", "damage_dissipation_density_J_per_m3",
)
FINAL_NODE_FIELDS = (
    "active_node_id", "background_node_id", "x_m", "y_m", "ux_m", "uy_m",
    "displacement_magnitude_m",
)
FINAL_ELEMENT_FIELDS = (
    "active_element_id", "background_element_id", "node_0", "node_1", "node_2",
    "von_mises_stress_Pa", "equivalent_plastic_strain", "damage",
)
SURFACE_PROFILE_FIELDS = (
    "order", "background_node_id", "active_node_id", "reference_x_m", "reference_y_m",
    "current_x_m", "current_y_m", "residual_vertical_displacement_m",
)


def artifact_paths_v3(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    return {key: root / name for key, name in ARTIFACT_FILENAMES_V3.items()}


def _topology(result: ContactTrajectoryResult) -> PreparedMaterialTopology:
    if result.case.single_grain_contact_schema_version != 3 or result.case.damage is None:
        raise ValueError("v3 artifacts require schema 3 damage input")
    if not isinstance(result.material_topology, PreparedMaterialTopology):
        raise ValueError("v3 result lacks final material topology")
    return result.material_topology


def _background_areas(result: ContactTrajectoryResult) -> np.ndarray:
    mesh = result.background_mesh
    if mesh is None:
        raise ValueError("v3 result lacks its background mesh")
    coordinates = np.asarray(mesh.mesh.p.T, dtype=float)
    connectivity = np.asarray(mesh.mesh.t.T, dtype=np.int32)
    first = coordinates[connectivity[:, 1]] - coordinates[connectivity[:, 0]]
    second = coordinates[connectivity[:, 2]] - coordinates[connectivity[:, 0]]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(fields); writer.writerows(rows)


def _write_histories(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    history_rows = []
    topology_rows = []
    surface_rows = []
    damage_rows = []
    damage_element_rows = []
    areas = _background_areas(result)
    cumulative_removed: set[int] = set()
    for record in result.records:
        state = record.state
        history_rows.append((
            record.index, record.segment, record.center_x_m, record.center_y_m,
            record.normal_reaction_N, record.tangential_reaction_N,
            record.contact_count, record.maximum_pressure_Pa,
            record.maximum_penetration_m, record.newton_iterations,
            record.balance_residual_N, record.contact_work_increment_J,
            record.friction_dissipation_increment_J,
            record.plastic_dissipation_increment_J,
            record.damage_dissipation_increment_J,
            record.separation_event_energy_increment_J,
            record.cumulative_contact_work_J,
            record.cumulative_friction_dissipation_J,
            record.cumulative_plastic_dissipation_J,
            record.cumulative_damage_dissipation_J,
            record.cumulative_separation_event_energy_J,
            record.energy_balance_residual_J,
        ))
        topology = record.material_topology
        prepared = record.prepared_mesh
        if not isinstance(topology, PreparedMaterialTopology) or prepared is None:
            raise ValueError("v3 record lacks a committed topology snapshot")
        active_ids = np.asarray(state.background_element_ids, dtype=np.int32)
        damaged = sum(item.damage > 0.0 for item in state.damage_states)
        topology_rows.append((
            record.index, record.segment, len(active_ids), damaged,
            len(topology.removed_element_ids), float(np.sum(areas[active_ids])),
            len(topology.active_node_ids), topology.active_total_degrees_of_freedom,
            topology.free_surface.edge_node_ids.shape[0], topology.free_surface.node_ids.size,
            float(np.sum(topology.free_surface.edge_lengths_m)),
        ))
        surface = topology.free_surface
        current = surface.reference_coordinates_m + state.displacement_vector_m[
            surface.candidate_component_dofs
        ]
        node_order = {int(node): index for index, node in enumerate(surface.node_ids)}
        for edge_index, (a, b) in enumerate(surface.edge_node_ids):
            first = node_order[int(a)]; second = node_order[int(b)]
            surface_rows.append((
                record.index, record.segment, edge_index, int(a), int(b),
                *surface.reference_coordinates_m[first], *surface.reference_coordinates_m[second],
                *current[first], *current[second], surface.edge_classifications[edge_index],
                int(surface.incident_active_element_ids[edge_index]),
                *surface.edge_outward_normals[edge_index], surface.edge_lengths_m[edge_index],
            ))
        event_count = sum(event.target_index <= record.index for event in result.separation_events)
        cumulative_removed.update(
            element_id
            for event in result.separation_events
            if event.target_index <= record.index
            for element_id in event.background_element_ids
        )
        damage_energy = record.cumulative_damage_dissipation_J
        damage_rows.append((
            record.index, record.segment, damaged,
            max((item.damage for item in state.damage_states), default=0.0),
            math.fsum(item.damage for item in state.damage_states) / len(state.damage_states),
            damage_energy, event_count, float(np.sum(areas[list(cumulative_removed)])) if cumulative_removed else 0.0,
        ))
        for background, damage in zip(active_ids, state.damage_states, strict=True):
            if damage.omega > 0.0 or damage.damage > 0.0:
                damage_element_rows.append((
                    record.index, record.segment, int(background), damage.omega, damage.damage,
                    damage.status, damage.equivalent_plastic_strain,
                    "" if damage.initiation_equivalent_plastic_strain is None else damage.initiation_equivalent_plastic_strain,
                    "" if damage.initiation_equivalent_stress_Pa is None else damage.initiation_equivalent_stress_Pa,
                    damage.post_initiation_plastic_displacement_m,
                    damage.damage_dissipation_density_J_per_m3,
                ))
    _write_csv(paths["contact_history_csv"], CONTACT_HISTORY_FIELDS, history_rows)
    _write_csv(paths["topology_history_csv"], TOPOLOGY_HISTORY_FIELDS, topology_rows)
    _write_csv(paths["free_surface_csv"], FREE_SURFACE_FIELDS, surface_rows)
    _write_csv(paths["damage_history_csv"], DAMAGE_HISTORY_FIELDS, damage_rows)
    if not damage_element_rows:
        damage_element_rows.append((0, "initial", -1, 0.0, 0.0, "ACTIVE", 0.0, "", "", 0.0, 0.0))
    _write_csv(paths["damage_element_history_csv"], DAMAGE_ELEMENT_FIELDS, damage_element_rows)
    _write_csv(paths["separation_events_csv"], SEPARATION_EVENT_FIELDS, [
        (
            event.event_index, event.target_index, event.trajectory_fraction, event.marker,
            json.dumps(event.background_element_ids, separators=(",", ":")), event.removed_area_m2,
            event.active_degrees_of_freedom_before, event.active_degrees_of_freedom_after,
            event.free_surface_edge_count_before, event.free_surface_edge_count_after,
            event.damage_dissipation_increment_J,
        )
        for event in result.separation_events
    ] or [(0, 0, 0.0, 0.0, "[]", 0.0, 0, 0, 0, 0, 0.0)])


def _write_final(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    topology = _topology(result)
    background = result.background_mesh
    assert background is not None
    state = result.records[-1].state
    prepared = result.prepared_mesh
    fields = recover_snapshot_fields(
        result.case.to_elastoplastic_fem_case(), prepared.structural, state
    )
    active_bg = np.asarray(prepared.active_to_background_element_ids, dtype=np.int32)
    active_damage = np.asarray([item.damage for item in state.damage_states], dtype=float)
    points = np.column_stack((fields.node_coordinates_m, np.zeros(fields.node_count)))
    displacement = np.column_stack((fields.nodal_displacements_m, np.zeros(fields.node_count)))
    meshio.write(paths["final_vtu"], meshio.Mesh(
        points=points, cells=[("triangle", fields.element_connectivity)],
        point_data={"background_node_id": prepared.active_to_background_node_ids,
                    "residual_displacement_m": displacement,
                    "residual_displacement_magnitude_m": fields.displacement_magnitude_m},
        cell_data={"background_element_id": [active_bg],
                   "von_mises_stress_Pa": [fields.von_mises_stress_Pa],
                   "equivalent_plastic_strain": [fields.equivalent_plastic_strain],
                   "damage": [active_damage]},
    ))
    count = background.triangle_count
    damage_full = np.zeros(count); omega_full = np.zeros(count); diss_full = np.zeros(count)
    for background_id, damage in zip(active_bg, state.damage_states, strict=True):
        damage_full[int(background_id)] = damage.damage
        omega_full[int(background_id)] = damage.omega
        diss_full[int(background_id)] = damage.damage_dissipation_density_J_per_m3
    damage_full[topology.removed_element_ids] = 1.0
    points_bg = np.column_stack((background.mesh.p.T, np.zeros(background.node_count)))
    meshio.write(paths["effective_mesh_vtu"], meshio.Mesh(
        points=points_bg, cells=[("triangle", background.mesh.t.T)],
        point_data={"background_node_id": np.arange(background.node_count, dtype=np.int32)},
        cell_data={"background_element_id": [np.arange(count, dtype=np.int32)],
                   "material_state_code": [[int(item) for item in topology.element_states]],
                   "active": [[int(item != MaterialState.REMOVED) for item in topology.element_states]],
                   "omega": [omega_full], "damage": [damage_full],
                   "damage_dissipation_density_J_per_m3": [diss_full]},
    ))
    _write_csv(paths["final_nodes_csv"], FINAL_NODE_FIELDS, [
        (active, int(background_id), *fields.node_coordinates_m[active],
         *fields.nodal_displacements_m[active], fields.displacement_magnitude_m[active])
        for active, background_id in enumerate(prepared.active_to_background_node_ids)
    ])
    _write_csv(paths["final_elements_csv"], FINAL_ELEMENT_FIELDS, [
        (active, int(background_id), *fields.element_connectivity[active],
         fields.von_mises_stress_Pa[active], fields.equivalent_plastic_strain[active], active_damage[active])
        for active, background_id in enumerate(active_bg)
    ])
    areas = _background_areas(result)
    lengths = np.sqrt(4.0 * areas / math.pi)
    removal_sources = topology.removal_sources
    _write_csv(paths["material_topology_csv"], TOPOLOGY_FIELDS, [
        (index, MaterialState(int(material_state)).name, int(material_state),
         int(material_state != MaterialState.REMOVED), removal_sources[index])
        for index, material_state in enumerate(topology.element_states)
    ])
    active_lookup = {int(bg): index for index, bg in enumerate(active_bg)}
    _write_csv(paths["final_damage_state_csv"], FINAL_DAMAGE_FIELDS, [
        (
            index,
            (state.damage_states[active_lookup[index]].omega if index in active_lookup else 1.0),
            (state.damage_states[active_lookup[index]].damage if index in active_lookup else 1.0),
            MaterialState(int(topology.element_states[index])).name,
            removal_sources[index], lengths[index],
            (state.damage_states[active_lookup[index]].damage_dissipation_density_J_per_m3 if index in active_lookup else 0.0),
        )
        for index in range(count)
    ])
    surface = topology.free_surface
    current = surface.reference_coordinates_m + state.displacement_vector_m[surface.candidate_component_dofs]
    _write_csv(paths["unloaded_surface_profile_csv"], SURFACE_PROFILE_FIELDS, [
        (order, int(background_id), int(prepared.candidate_node_ids[order]),
         *surface.reference_coordinates_m[order], *current[order],
         current[order, 1] - surface.reference_coordinates_m[order, 1])
        for order, background_id in enumerate(surface.node_ids)
    ])
    if result.reference_mesh_path is not None and result.reference_mesh_path.is_file():
        shutil.copy2(result.reference_mesh_path, paths["reference_mesh_msh"])
    else:
        meshio.write(paths["reference_mesh_msh"], meshio.Mesh(
            points=points_bg, cells=[("triangle", background.mesh.t.T)]
        ), file_format="gmsh22", binary=False)
    final_record = result.records[-1]
    _write_csv(paths["contact_points_csv"], CONTACT_POINT_FIELDS, [
        (
            int(background_node), int(active_node), *reference, motion.signed_gap_m,
            pressure, *motion.outward_normal, motion.feature_id, motion.feature_type,
            json.dumps(key, separators=(",", ":")), contact.status,
        )
        for background_node, active_node, reference, motion, pressure, key, contact in zip(
            prepared.candidate_background_node_ids, prepared.candidate_node_ids,
            prepared.candidate_reference_coordinates_m, final_record.state.contact_kinematics,
            final_record.state.contact_pressure_Pa, prepared.candidate_keys,
            final_record.state.contact_states, strict=True,
        )
    ])


def _save_plot(path: Path, title: str, x: np.ndarray, series: list[tuple[np.ndarray, str]]) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for y, label in series:
        axis.plot(x, y, marker="o", linewidth=1.5, label=label)
    axis.set_title(title); axis.grid(True, alpha=0.25)
    if series: axis.legend(loc="best")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _write_plots(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    indexes = np.arange(len(result.records), dtype=float)
    _save_plot(paths["reaction_history_png"], "Rigid grain reaction", indexes, [
        (np.asarray([r.normal_reaction_N for r in result.records]), "Normal reaction (N)"),
        (np.asarray([r.tangential_reaction_N for r in result.records]), "Tangential reaction (N)"),
    ])
    _save_plot(paths["contact_pressure_png"], "Maximum contact pressure", indexes, [
        (np.asarray([r.maximum_pressure_Pa for r in result.records]), "Pressure (Pa)")])
    _save_plot(paths["contact_state_png"], "Contact point count", indexes, [
        (np.asarray([r.contact_count for r in result.records]), "Contact count")])
    _save_plot(paths["energy_history_png"], "Energy history", indexes, [
        (np.asarray([r.cumulative_contact_work_J for r in result.records]), "Contact work (J)"),
        (np.asarray([r.cumulative_plastic_dissipation_J for r in result.records]), "Plastic dissipation (J)"),
        (np.asarray([r.cumulative_damage_dissipation_J for r in result.records]), "Damage dissipation (J)"),
        (np.asarray([r.cumulative_separation_event_energy_J for r in result.records]), "Separation event energy (J)"),
        (np.asarray([r.energy_balance_residual_J for r in result.records]), "Balance residual (J)"),
    ])
    damage = np.asarray([item.maximum_damage for item in result.damage_history])
    _save_plot(paths["damage_evolution_png"], "Damage evolution", indexes, [(damage, "Maximum D")])
    areas = np.zeros(len(result.records))
    for event in result.separation_events:
        areas[event.target_index:] += event.removed_area_m2
    _save_plot(paths["removal_area_history_png"], "Automatic removed area", indexes, [(areas, "Area (m2)")])

    topology = _topology(result); background = result.background_mesh; assert background is not None
    state = result.records[-1].state; prepared = result.prepared_mesh
    for key, title in (
        ("effective_mesh_surface_png", "Effective mesh and free surface"),
        ("final_damage_field_png", "Final damage field"),
        ("processed_surface_png", "Model-predicted processed surface"),
        ("unloaded_surface_profile_png", "Unloaded surface profile"),
        ("final_fields_png", "Final effective mesh"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.triplot(background.mesh.p[0] * 1e6, background.mesh.p[1] * 1e6,
                     background.mesh.t[:, topology.active_element_ids].T, color="#748694", linewidth=0.6)
        if topology.removed_element_ids.size:
            axis.tripcolor(background.mesh.p[0] * 1e6, background.mesh.p[1] * 1e6,
                           background.mesh.t[:, topology.removed_element_ids].T,
                           facecolors=np.ones(topology.removed_element_ids.size), cmap="Reds", alpha=0.4)
        surface = topology.free_surface
        current = surface.reference_coordinates_m + state.displacement_vector_m[surface.candidate_component_dofs]
        axis.plot(surface.reference_coordinates_m[:, 0] * 1e6, surface.reference_coordinates_m[:, 1] * 1e6,
                  "--", color="#657786", label="Reference surface")
        axis.plot(current[:, 0] * 1e6, current[:, 1] * 1e6, color="#c43c39", linewidth=2, label="Unloaded surface")
        axis.set_aspect("equal"); axis.set_title(title); axis.legend(loc="best"); figure.tight_layout(); figure.savefig(paths[key], dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.0, 5.0))
    grain = result.case.grain
    if grain.type == "rounded_wedge":
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
        axis.plot(displayed[:, 0] * 1e6, displayed[:, 1] * 1e6)
    else:
        theta = np.linspace(0.0, 2.0 * math.pi, 241)
        radius = float(grain.radius_m)
        axis.plot(radius * np.cos(theta) * 1e6, radius * np.sin(theta) * 1e6)
    axis.set_aspect("equal"); axis.set_title("Rigid grain geometry"); figure.tight_layout(); figure.savefig(paths["grain_geometry_png"], dpi=160); plt.close(figure)


def _summary(result: ContactTrajectoryResult, paths: dict[str, Path]) -> dict[str, object]:
    topology = _topology(result); case = result.case; areas = _background_areas(result)
    removed_area = math.fsum(event.removed_area_m2 for event in result.separation_events)
    final = result.records[-1]
    interpretation = (
        "calibrated" if case.damage.calibration_status == "calibrated"
        else "model_predicted_unvalidated"
    )
    checkpoints=[]
    patch_history=[]
    if getattr(case,'circle_edge_quadrature_order',0) or case.transient is not None:
        from dataclasses import replace
        from .solver import prepare_contact_mesh
        from .state_codec import encode_committed_state
        for record in result.records:
            prepared=prepare_contact_mesh(case.to_elastoplastic_fem_case(),result.background_mesh,
                topology=record.material_topology,edge_history=True)
            prepared=replace(prepared,circle_edge_quadrature_order=case.circle_edge_quadrature_order)
            checkpoints.append(encode_committed_state(prepared,record.state))
            from .solver import _assemble_contact
            from .workflow import _grain
            grain=_grain(case,replace(case.trajectory.targets[record.index],
                reference_x_m=record.state.grain_reference_m[0],reference_y_m=record.state.grain_reference_m[1]))
            contact=_assemble_contact(case.to_elastoplastic_fem_case(),prepared,record.state,
                record.state.displacement_vector_m,grain,result.normal_penalty_Pa_per_m,
                result.tangential_penalty_Pa_per_m,case.contact.friction_coefficient,case.contact.friction_regularization_ratio)
            patch_history.append(dict(index=record.index,patches=contact.integrated_edge_patches))
    return {
        "result_format": RESULT_FORMAT_V3,
        "grindcae_version": __version__,
        "case": case.to_dict(),
        "contact_integration": ({'rule':'clipped_circle_penalty','order':case.circle_edge_quadrature_order}
            if getattr(case,'circle_edge_quadrature_order',0) else {'rule':'edge_endpoint','order':2}),
        "input_fingerprint": case.input_fingerprint(),
        "accepted_checkpoints": checkpoints,
        "integrated_contact_patch_history": patch_history,
        "contact_count_interpretation": "endpoint_samples; integrated patches reported separately",
        "artifacts": {key: path.name for key, path in paths.items()},
        "automatic_damage_evolution": True,
        "automatic_material_separation": True,
        "material_removal_interpretation": interpretation,
        "chip_formation_prediction": False,
        "damage_model": {**case.damage.to_dict(), "formula_version": (
            "grindcae_energetic_rational_v1" if case.damage.evolution=="energetic_rational_fracture"
            else "grindcae_damage_v1")},
        "kernel_state_history": [
            {"index": record.index,
             "hardening_stored_energy_J": record.hardening_stored_energy_J,
             "accepted_path_reference": (asdict(record.state.accepted_path_reference)
                 if record.state.accepted_path_reference is not None else None),
             "edge_contact_states": [
                 {"key":list(key),"normal_multiplier_Pa":state.normal_multiplier_Pa,
                  "tangential_traction_Pa":state.tangential_traction_Pa,
                  "accumulated_slip_m":state.accumulated_slip_m,"status":state.status}
                 for key,state in record.state.edge_contact_states],
             "archived_edge_contact_states": [
                 {"key":list(key),"normal_multiplier_Pa":state.normal_multiplier_Pa,
                  "tangential_traction_Pa":state.tangential_traction_Pa,
                  "accumulated_slip_m":state.accumulated_slip_m,"status":state.status}
                 for key,state in record.state.archived_edge_contact_states],
             "energetic_material_history": [
                 {"background_element_id":int(background),"damage":item.energetic_history.damage,
                  "initial_driving_energy":item.energetic_history.initial_driving_energy,
                  "fracture_energy_density":item.energetic_history.fracture_energy_density}
                 for background,item in zip(record.state.background_element_ids,record.state.damage_states,strict=True)
                 if getattr(item,"energetic_history",None) is not None]}
            for record in result.records],
        "scope": {
            "dimension": "2D longitudinal section", "kinematics": "small_strain",
            "background_mesh": "fixed", "free_chip_motion": False,
            "independent_physical_validation": interpretation == "independently_validated",
        },
        "topology": {
            "background_element_count": len(topology.element_states),
            "active_element_count": len(topology.active_element_ids),
            "removed_element_count": len(topology.removed_element_ids),
            "active_degrees_of_freedom": topology.active_total_degrees_of_freedom,
            "free_surface_edge_count": topology.free_surface.edge_node_ids.shape[0],
        },
        "separation": {
            "event_count": len(result.separation_events),
            "automatic_removed_element_ids": list(result.automatic_removed_element_ids),
            "automatic_removed_area_m2": removed_area,
            "area_identity_residual_m2": removed_area - float(np.sum(areas[list(result.automatic_removed_element_ids)])) if result.automatic_removed_element_ids else 0.0,
        },
        "accepted_target_energy_ledger": [
            dict(index=r.index,marker=r.state.marker,contact_work_J=r.cumulative_contact_work_J,
                elastic_stored_energy_J=r.elastic_strain_energy_J,
                hardening_stored_energy_J=r.hardening_stored_energy_J,
                numerical_contact_stored_energy_J=r.numerical_contact_stored_energy_J,
                kinetic_energy_J=float(getattr(r.state,'kinetic_energy_J',0.0)),
                plastic_dissipation_J=r.cumulative_plastic_dissipation_J,
                friction_dissipation_J=r.cumulative_friction_dissipation_J,
                damage_dissipation_J=r.cumulative_damage_dissipation_J,
                separation_event_attribution_J=r.cumulative_separation_event_energy_J,
                balance_residual_J=r.energy_balance_residual_J) for r in result.records],
        "energy": {
            "contact_work_J": result.contact_work_J,
            "friction_dissipation_J": result.friction_dissipation_J,
            "plastic_dissipation_J": result.plastic_dissipation_J,
            "damage_dissipation_J": result.damage_dissipation_J,
            "separation_event_energy_J": result.separation_event_energy_J,
            "balance_residual_J": result.energy_balance_residual_J,
            "balance_relative_residual": result.energy_balance_relative_residual,
            "damage_energy_thickness_identity": "J_per_m_times_thickness_equals_J",
        },
        "final_state": {
            "segment": final.segment, "contact_count": final.contact_count,
            "normal_reaction_N": final.normal_reaction_N,
            "tangential_reaction_N": final.tangential_reaction_N,
            "physical_time_s": final.state.physical_time_s,
            "kinetic_energy_J": final.state.kinetic_energy_J,
            "residual_profile_is_settled": (
                final.state.solve_mode.value=='STANDARD_NEWTON'
                and (final.state.transient_step_count==0 or final.state.transient_stable_step_count==8)),
        },
    }


def write_contact_artifacts_v3(result: ContactTrajectoryResult, paths: dict[str, Path]) -> None:
    if set(paths) != set(ARTIFACT_FILENAMES_V3):
        raise ValueError("v3 artifact set is invalid")
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    _write_histories(result, paths); _write_final(result, paths); _write_plots(result, paths)
    paths["summary_json"].write_text(
        json.dumps(_summary(result, paths), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    read_contact_summary_v3(paths["summary_json"])


def _read_csv(path: Path, fields: tuple[str, ...], label: str, text: set[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields: raise ValueError(f"{label} header is invalid")
        rows = list(reader)
    if not rows: raise ValueError(f"{label} has no rows")
    text = text or set()
    for row in rows:
        for name, raw in row.items():
            if name in text:
                if raw is None or raw == "": raise ValueError(f"{label} has empty text")
            elif raw not in (None, "") and not math.isfinite(float(raw)):
                raise ValueError(f"{label} contains non-finite data")
    return rows


def read_contact_summary_v3(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve(); raw = source.read_text(encoding="utf-8")
    if "NaN" in raw or "Infinity" in raw: raise ValueError("v3 summary contains nonfinite data")
    value = json.loads(raw)
    if value.get("result_format") != RESULT_FORMAT_V3: raise ValueError("wrong v3 result format")
    if value.get("grindcae_version") != __version__: raise ValueError("v3 version is inconsistent")
    case = SingleGrainContactCase.from_mapping(value.get("case"))
    if case.single_grain_contact_schema_version != 3: raise ValueError("v3 requires schema 3")
    expected_integration=({'rule':'clipped_circle_penalty','order':case.circle_edge_quadrature_order}
        if case.circle_edge_quadrature_order else {'rule':'edge_endpoint','order':2})
    if value.get('contact_integration',{'rule':'edge_endpoint','order':2})!=expected_integration:
        raise ValueError('v3 contact integration is inconsistent with input')
    expected_formula=("grindcae_energetic_rational_v1" if case.damage.evolution=="energetic_rational_fracture"
                      else "grindcae_damage_v1")
    if value.get('damage_model',{}).get('formula_version')!=expected_formula:
        raise ValueError("v3 material formula is inconsistent with input")
    if value.get("input_fingerprint") != case.input_fingerprint(): raise ValueError("v3 fingerprint is inconsistent")
    if not (
        value.get("automatic_damage_evolution") is True
        and value.get("automatic_material_separation") is True
        and value.get("chip_formation_prediction") is False
        and value.get("material_removal_interpretation") in {"model_predicted_unvalidated", "calibrated", "independently_validated"}
    ): raise ValueError("v3 physical scope flags are invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_FILENAMES_V3):
        raise ValueError("v3 artifact index is invalid")
    resolved = {}
    for key, name in artifacts.items():
        target = (source.parent / str(name)).resolve()
        if target.parent != source.parent or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError("v3 registered artifact is missing or unsafe")
        resolved[key] = target
    history = _read_csv(resolved["contact_history_csv"], CONTACT_HISTORY_FIELDS, "contact history", {"segment"})
    if history[-1]["segment"] != "unloading" or int(history[-1]["contact_count"]) != 0:
        raise ValueError("v3 result lacks an unloaded open final state")
    ledger=value.get('accepted_target_energy_ledger')
    if ledger is None and (case.circle_edge_quadrature_order or case.transient is not None):
        raise ValueError('v3 physical energy ledger is missing')
    if ledger is not None:
        storage_fields=('elastic_stored_energy_J','hardening_stored_energy_J','numerical_contact_stored_energy_J','kinetic_energy_J')
        dissipation_fields=('plastic_dissipation_J','friction_dissipation_J','damage_dissipation_J')
        if not isinstance(ledger,list) or len(ledger)!=len(history):
            raise ValueError('v3 energy ledger target count is invalid')
        baseline=None;previous=None
        for index,row in enumerate(ledger):
            if not isinstance(row,dict) or row.get('index')!=index:
                raise ValueError('v3 energy ledger identity is invalid')
            # Earlier static development results have no kinetic column.
            if 'kinetic_energy_J' not in row:
                row=dict(row,kinetic_energy_J=0.)
            fields=(*storage_fields,*dissipation_fields,'contact_work_J','separation_event_attribution_J','balance_residual_J','marker')
            if any(type(row.get(f)) not in (int,float) or not math.isfinite(row[f]) for f in fields):
                raise ValueError('v3 energy ledger has invalid values')
            if any(row[f]<0. for f in (*storage_fields,*dissipation_fields,'separation_event_attribution_J')):
                raise ValueError('v3 energy ledger has negative storage or dissipation')
            storage=math.fsum(row[f] for f in storage_fields)
            if baseline is None: baseline=storage
            predicted=row['contact_work_J']-(storage-baseline+math.fsum(row[f] for f in dissipation_fields))
            scale=abs(row['contact_work_J'])+storage+baseline+math.fsum(row[f] for f in dissipation_fields)
            if abs(predicted-row['balance_residual_J'])>64*np.finfo(float).eps*max(scale,1e-300):
                raise ValueError('v3 energy ledger does not reconstruct the balance')
            if previous is not None and any(row[f]<previous[f] for f in dissipation_fields):
                raise ValueError('v3 energy ledger dissipation is not irreversible')
            previous=row
        for name in ('contact_work_J',*dissipation_fields):
            if ledger[-1][name]!=value.get('energy',{}).get(name):
                raise ValueError('v3 energy ledger disagrees with summary')
    damage = _read_csv(resolved["damage_history_csv"], DAMAGE_HISTORY_FIELDS, "damage history", {"segment"})
    energies = [float(row["damage_dissipation_J"]) for row in damage]
    removed_areas = [float(row["automatic_removed_area_m2"]) for row in damage]
    if any(item < -1e-15 for item in energies) or any(b + 1e-15 < a for a, b in zip(energies, energies[1:])):
        raise ValueError("damage dissipation is negative or nonmonotonic")
    if any(b + 1e-15 < a for a, b in zip(removed_areas, removed_areas[1:])):
        raise ValueError("damage removal area is nonmonotonic")
    topology = _read_csv(resolved["material_topology_csv"], TOPOLOGY_FIELDS, "topology", {"material_state", "removal_source"})
    if [int(row["background_element_id"]) for row in topology] != list(range(len(topology))):
        raise ValueError("topology ids are invalid")
    events = _read_csv(resolved["separation_events_csv"], SEPARATION_EVENT_FIELDS, "separation events", {"background_element_ids"})
    actual_events = [row for row in events if int(row["event_index"]) > 0]
    if [int(row["event_index"]) for row in actual_events] != list(range(1, len(actual_events) + 1)):
        raise ValueError("separation event ordering is invalid")
    for row in actual_events:
        ids = json.loads(row["background_element_ids"])
        if ids != sorted(set(ids)): raise ValueError("separation event ids are invalid")
    summary_separation = value.get("separation", {})
    if int(summary_separation.get("event_count", -1)) != len(actual_events):
        raise ValueError("separation event count is inconsistent")
    if abs(float(summary_separation.get("area_identity_residual_m2", math.inf))) > max(case.damage.area_absolute_tolerance_m2, 1e-15):
        raise ValueError("separation area identity is inconsistent")
    surfaces=_read_csv(resolved["free_surface_csv"], FREE_SURFACE_FIELDS, "free surface", {"segment", "classification"})
    kernel=value.get("kernel_state_history")
    if case.damage.evolution=="energetic_rational_fracture" and kernel is None:
        raise ValueError("energetic v3 lacks kernel history")
    if kernel is not None:
        from .edge_contact_history import endpoint_keys
        expected_indices=sorted({int(row['index']) for row in surfaces})
        if not isinstance(kernel,list) or [row.get('index') for row in kernel]!=expected_indices:
            raise ValueError("kernel history indices are invalid")
        material_previous={}
        for row in kernel:
            path_reference=row.get('accepted_path_reference')
            if path_reference is not None:
                from .path_continuation import AcceptedPathReference
                try:
                    AcceptedPathReference(**{
                        **path_reference,
                        'background_node_ids':tuple(path_reference['background_node_ids']),
                        'displacement_direction_m':tuple(tuple(v) for v in path_reference['displacement_direction_m']),
                        'source_markers':tuple(path_reference['source_markers']),
                    })
                except (TypeError,KeyError,ValueError) as error:
                    raise ValueError('kernel accepted path reference is invalid') from error
            expected=set(endpoint_keys((int(face['background_node_0']),int(face['background_node_1']))
                for face in surfaces if int(face['index'])==row['index']))
            active=row.get('edge_contact_states',[])
            keys=[tuple(item['key']) for item in active]
            if len(keys)!=len(set(keys)) or set(keys)!=expected:
                raise ValueError("kernel edge identities do not match free surface")
            archive=row.get('archived_edge_contact_states',[])
            archive_keys=[tuple(item['key']) for item in archive]
            if len(archive_keys)!=len(set(archive_keys)) or set(archive_keys)&expected:
                raise ValueError('kernel archived edge overlaps an active edge or is duplicated')
            for key in keys+archive_keys:
                if (len(key)!=3 or any(type(node) is not int or node<0 for node in key)
                    or key[0]>=key[1] or key[2] not in key[:2]):
                    raise ValueError('kernel edge identity is invalid')
            storage=row.get('hardening_stored_energy_J')
            if not isinstance(storage,(float,int)) or not math.isfinite(storage) or storage<0:
                raise ValueError("kernel hardening storage is invalid")
            for item in active+row.get('archived_edge_contact_states',[]):
                if item.get('status') not in ('open','stick','slip'):
                    raise ValueError("kernel edge status is invalid")
                for field in ('normal_multiplier_Pa','tangential_traction_Pa','accumulated_slip_m'):
                    number=item.get(field)
                    if not isinstance(number,(float,int)) or not math.isfinite(number) or (field!='tangential_traction_Pa' and number<0):
                        raise ValueError("kernel edge history is invalid")
            seen=set()
            for item in row.get('energetic_material_history',[]):
                background=item.get('background_element_id')
                if type(background) is not int or not 0<=background<value['topology']['background_element_count'] or background in seen:
                    raise ValueError("kernel material identity is invalid")
                seen.add(background)
                damage=item.get('damage'); onset=item.get('initial_driving_energy'); fracture=item.get('fracture_energy_density')
                if (not all(type(number) in (float,int) and math.isfinite(number) for number in (damage,onset,fracture))
                    or not 0<=damage<=1 or onset<=0 or fracture<=0):
                    raise ValueError("kernel material energy history is invalid")
                old=material_previous.get(background)
                if old is not None and (damage<old[0] or onset!=old[1] or fracture!=old[2]):
                    raise ValueError("kernel material history is not irreversible or changes onset")
                material_previous[background]=(damage,onset,fracture)
    _read_csv(resolved["damage_element_history_csv"], DAMAGE_ELEMENT_FIELDS, "damage element history", {"segment", "status"})
    _read_csv(resolved["final_damage_state_csv"], FINAL_DAMAGE_FIELDS, "final damage", {"material_state", "removal_source"})
    _read_csv(resolved["contact_points_csv"], CONTACT_POINT_FIELDS, "contact points", {"feature_type", "candidate_key", "status"})
    _read_csv(resolved["topology_history_csv"], TOPOLOGY_HISTORY_FIELDS, "topology history", {"segment"})
    _read_csv(resolved["final_nodes_csv"], FINAL_NODE_FIELDS, "final nodes")
    _read_csv(resolved["final_elements_csv"], FINAL_ELEMENT_FIELDS, "final elements")
    _read_csv(resolved["unloaded_surface_profile_csv"], SURFACE_PROFILE_FIELDS, "surface profile")
    for key in ("final_vtu", "effective_mesh_vtu", "reference_mesh_msh"):
        mesh = meshio.read(resolved[key])
        arrays = [mesh.points, *mesh.point_data.values()]
        for item in mesh.cell_data.values(): arrays.extend(item)
        if not all(np.all(np.isfinite(np.asarray(item, dtype=float))) for item in arrays):
            raise ValueError("v3 mesh contains non-finite data")
    for key, filename in ARTIFACT_FILENAMES_V3.items():
        if filename.endswith(".png"):
            with Image.open(resolved[key]) as image: image.verify()
    final_state = value.get("final_state", {})
    if case.circle_edge_quadrature_order or case.transient is not None:
        checkpoints=value.get('accepted_checkpoints')
        if not isinstance(checkpoints,list) or len(checkpoints)!=len(history):
            raise ValueError('v3 checkpoint count mismatch')
        from dataclasses import replace
        from grindcae.solver import ImportedMesh
        from skfem.io.meshio import from_meshio
        from .material_topology import prepare_material_topology
        from .models import PresetRemoval
        from .solver import prepare_contact_mesh
        from .state_codec import decode_committed_state
        # The retained result MSH contains triangles only. Reconstruct named
        # original rectangle boundaries from the recorded physical geometry.
        mesh=from_meshio(meshio.read(resolved['reference_mesh_msh']))
        tolerance=64*np.finfo(float).eps*max(case.geometry.width,case.geometry.height)
        mesh=mesh.with_boundaries({'fixed':lambda x:np.abs(x[1])<=tolerance,
            'contact':lambda x:np.abs(x[1]-case.geometry.height)<=tolerance})
        background=ImportedMesh(mesh=mesh,source_path=resolved['reference_mesh_msh'])
        initial_topology=prepare_material_topology(background,case.material_topology.preset_removal)
        patch_history=value.get('integrated_contact_patch_history')
        if not isinstance(patch_history,list) or len(patch_history)!=len(checkpoints):
            raise ValueError('v3 integrated contact patch history is missing')
        initial_prepared=prepare_contact_mesh(case.to_elastoplastic_fem_case(),background,
            topology=initial_topology,edge_history=True)
        penalty=case.contact.normal_penalty_factor*case.effective_modulus_Pa/float(np.median(initial_prepared.candidate_tributary_lengths_m))
        removed=set(int(v) for v in initial_topology.removed_element_ids)
        expected_time = 0.
        previous_reference = None
        previous_checkpoint = None
        for index,checkpoint in enumerate(checkpoints):
            for event in actual_events:
                if int(event['target_index'])<=index:
                    removed.update(json.loads(event['background_element_ids']))
            topology_view=prepare_material_topology(background,PresetRemoval(mode='element_ids',element_ids=tuple(sorted(removed))))
            prepared=prepare_contact_mesh(case.to_elastoplastic_fem_case(),background,topology=topology_view,edge_history=True)
            prepared=replace(prepared,circle_edge_quadrature_order=case.circle_edge_quadrature_order)
            try:
                restored=decode_committed_state(prepared,checkpoint,
                    density_kg_per_m3=case.material.density_kg_per_m3,
                    thickness_m=case.analysis.thickness)
            except (ValueError,TypeError,KeyError) as exc:
                raise ValueError(f'v3 checkpoint validation failed: {exc}') from exc
            if restored.marker!=ledger[index]['marker']:
                raise ValueError('v3 checkpoint marker disagrees with energy ledger')
            if restored.kinetic_energy_J!=ledger[index].get('kinetic_energy_J',0.):
                raise ValueError('v3 checkpoint kinetic energy disagrees with target ledger')
            from .thermodynamic_energy import stored_energy
            material=case.to_elastoplastic_fem_case().material
            elastic=hardening=0.
            for point,damage_state,area in zip(restored.element_states,restored.damage_states,
                    prepared.structural.element_areas_m2,strict=True):
                energy=stored_energy(point.stress_tensor_Pa,point.equivalent_plastic_strain,
                    damage_state.damage,young_modulus=material.E,poisson_ratio=material.nu,
                    hardening_modulus=material.internal_hardening_modulus)
                volume=float(area)*case.analysis.thickness
                elastic+=volume*energy['stored_elastic']
                if (case.damage.evolution=='energetic_rational_fracture'
                        or damage_state.energetic_history is not None):
                    hardening+=volume*energy['stored_hardening']
            for field,measured in (('elastic_stored_energy_J',elastic),('hardening_stored_energy_J',hardening)):
                if not math.isclose(measured,ledger[index][field],rel_tol=1e-10,abs_tol=1e-18):
                    raise ValueError('v3 checkpoint material storage disagrees with target ledger')
            from .state_codec import validate_material_compatibility
            validate_material_compatibility(prepared,restored,material)
            if case.transient is not None:
                signed_error = abs(ledger[index]['balance_residual_J'])
                ledger_scale = max(restored.transient_maximum_energy_scale_J,
                    abs(ledger[index]['contact_work_J']),
                    math.fsum(ledger[index][field] for field in (*storage_fields,*dissipation_fields)))
                rounding = 128*np.finfo(float).eps*max(ledger_scale,1e-300)
                if signed_error > restored.transient_absolute_energy_error_J+rounding:
                    raise ValueError('v3 checkpoint absolute energy error does not cover target balance')
                target = case.trajectory.targets[index]
                reference = np.asarray((target.reference_x_m, target.reference_y_m)
                    if target.reference_x_m is not None else (target.center_x_m, target.center_y_m))
                if not np.allclose(restored.grain_reference_m, reference, rtol=0., atol=1e-15):
                    raise ValueError('v3 checkpoint does not reach its prescribed target')
                if previous_reference is not None:
                    expected_time += float(np.linalg.norm(reference-previous_reference))/case.transient.grain_speed_m_per_s
                if not math.isclose(restored.physical_time_s, expected_time, rel_tol=1e-9, abs_tol=1e-15):
                    raise ValueError('v3 checkpoint physical time disagrees with prescribed trajectory')
                if previous_checkpoint is not None and (
                    restored.transient_absolute_energy_error_J < previous_checkpoint.transient_absolute_energy_error_J
                    or restored.transient_maximum_energy_scale_J < previous_checkpoint.transient_maximum_energy_scale_J
                    or restored.transient_step_count < previous_checkpoint.transient_step_count):
                    raise ValueError('v3 checkpoint transient cumulative history decreases')
                if restored.transient_step_count > case.transient.maximum_steps:
                    raise ValueError('v3 checkpoint exceeds physical step limit')
                if index==len(checkpoints)-1:
                    settled=(restored.solve_mode.value=='STANDARD_NEWTON'
                        and (restored.transient_step_count==0 or restored.transient_stable_step_count==8))
                    if (final_state.get('residual_profile_is_settled') is not settled
                            or final_state.get('physical_time_s')!=restored.physical_time_s
                            or final_state.get('kinetic_energy_J')!=restored.kinetic_energy_J):
                        raise ValueError('v3 final settled profile and dynamic history disagree with checkpoint')
                previous_reference = reference
                previous_checkpoint = restored
            from .solver import _assemble_contact
            from .workflow import _grain
            grain=_grain(case,replace(case.trajectory.targets[index],
                reference_x_m=restored.grain_reference_m[0],reference_y_m=restored.grain_reference_m[1]))
            contact=_assemble_contact(case.to_elastoplastic_fem_case(),prepared,restored,
                restored.displacement_vector_m,grain,penalty,
                penalty*case.contact.tangential_penalty_ratio,case.contact.friction_coefficient,case.contact.friction_regularization_ratio)
            from .state_codec import validate_endpoint_momentum
            validate_endpoint_momentum(prepared,restored,case,contact.force_N)
            stored_patches=patch_history[index]
            if stored_patches.get('index')!=index or not isinstance(stored_patches.get('patches'),list):
                raise ValueError('v3 contact patch index is invalid')
            actual=contact.integrated_edge_patches
            penalty_energy=math.fsum(patch['penalty_energy_J'] for patch in actual)
            recorded_energy=ledger[index]['numerical_contact_stored_energy_J']
            if case.circle_edge_quadrature_order and abs(penalty_energy-recorded_energy)>1e-10*max(abs(penalty_energy),abs(recorded_energy),1e-300):
                raise ValueError('v3 contact patch energy disagrees with target ledger')
            reaction=-np.sum(contact.force_N[prepared.structural.basis.nodal_dofs],axis=1)
            recorded_reaction=np.array([history[index]['tangential_reaction_N'],history[index]['normal_reaction_N']],dtype=float)
            if not np.allclose(reaction,recorded_reaction,rtol=1e-10,atol=1e-8):
                raise ValueError('v3 contact patch force disagrees with target reaction')
            if len(stored_patches['patches'])!=len(actual):
                raise ValueError('v3 contact patch count disagrees with checkpoint')
            for stored,measured in zip(stored_patches['patches'],actual,strict=True):
                if tuple(stored.get('background_edge',()))!=measured['background_edge']:
                    raise ValueError('v3 contact patch background identity disagrees')
                for key in ('active_interval','reference_contact_length_m','maximum_pressure_Pa','force_N','penalty_energy_J'):
                    observed=np.asarray(stored.get(key),dtype=float); expected=np.asarray(measured[key],dtype=float)
                    if (observed.shape!=expected.shape or not np.all(np.isfinite(observed))
                        or not np.allclose(observed,expected,rtol=1e-10,atol=0.)):
                        raise ValueError('v3 contact patch measurement disagrees with checkpoint')
    if abs(float(final_state.get("normal_reaction_N", math.inf))) > 1e-8 or abs(float(final_state.get("tangential_reaction_N", math.inf))) > 1e-8:
        raise ValueError("v3 unloaded final reaction is nonzero")
    return value


def _stage(source: Path, target: Path, kind: str) -> Path:
    staged = target.with_name(f".{target.name}.{kind}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, staged); return staged


def publish_contact_result_v3(temporary: dict[str, Path], final: dict[str, Path]) -> None:
    if set(temporary) != set(final) or set(final) != set(ARTIFACT_FILENAMES_V3):
        raise ValueError("v3 publication set is invalid")
    read_contact_summary_v3(temporary["summary_json"])
    root = final["summary_json"].parent; existed = root.exists(); root.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES_V3 if key != "summary_json"] + ["summary_json"]
    staged = {}; backups = {}; published = []
    try:
        for key in order: staged[key] = _stage(temporary[key], final[key], "new")
        for key in order:
            if final[key].exists(): backups[key] = _stage(final[key], final[key], "backup")
        for key in order: os.replace(staged.pop(key), final[key]); published.append(key)
        read_contact_summary_v3(final["summary_json"])
    except Exception as exc:
        for key in reversed(published):
            if key in backups: os.replace(backups.pop(key), final[key])
            else: final[key].unlink(missing_ok=True)
        raise RuntimeError(f"v3 contact artifact publication failed: {exc}") from exc
    finally:
        for item in (*staged.values(), *backups.values()): item.unlink(missing_ok=True)
        if not existed and root.exists():
            try: root.rmdir()
            except OSError: pass


__all__ = [
    "ARTIFACT_FILENAMES_V3", "RESULT_FORMAT_V3", "artifact_paths_v3",
    "publish_contact_result_v3", "read_contact_summary_v3", "write_contact_artifacts_v3",
]
