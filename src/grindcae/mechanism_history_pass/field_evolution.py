"""Target-position field comparison export for the mechanism-history route."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import meshio
import numpy as np
from numpy.typing import NDArray

from grindcae.elastoplastic_fem import ElastoplasticSnapshotFields
from grindcae.history_pass.field_evolution import write_target_vtu
from grindcae.history_pass.workflow import TargetFieldSnapshot
from grindcae.plotting import plot_style, save_png, style_axis


FIELD_EVOLUTION_DIRECTORY = "field_evolution"
RESULT_FORMAT = "grindcae_mechanism_history_pass_target_field_evolution_comparison"
ROUTES = ("baseline", "mechanism", "difference")
POSITION_CSV_FIELDS = (
    "frame_index", "position_id", "motion_coordinate_m", "wheel_lowest_point_x_m",
    "pass_state", "contact_ratio", "is_final_unloaded_target",
    "baseline_Fx_N", "baseline_Fy_N", "mechanism_Fx_N", "mechanism_Fy_N",
    "baseline_accumulated_plastic_dissipation_J",
    "mechanism_accumulated_plastic_dissipation_J",
    "baseline_png_path", "baseline_vtu_path", "mechanism_png_path",
    "mechanism_vtu_path", "difference_png_path", "difference_vtu_path",
)


class MechanismFieldEvolutionArtifactError(RuntimeError):
    """Raised when a mechanism comparison sequence is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class MechanismFieldDifference:
    """Mechanism-minus-baseline scalar response fields on one identical mesh."""

    node_coordinates_m: NDArray[np.float64]
    element_connectivity: NDArray[np.int32]
    displacement_magnitude_difference_m: NDArray[np.float64]
    von_mises_stress_difference_Pa: NDArray[np.float64]
    equivalent_plastic_strain_difference: NDArray[np.float64]


def calculate_field_difference(
    baseline: ElastoplasticSnapshotFields,
    mechanism: ElastoplasticSnapshotFields,
) -> MechanismFieldDifference:
    """Return exact mechanism-minus-baseline scalar differences without interpolation."""

    if not np.array_equal(baseline.node_coordinates_m, mechanism.node_coordinates_m):
        raise ValueError("mechanism field node coordinates differ from baseline")
    if not np.array_equal(baseline.element_connectivity, mechanism.element_connectivity):
        raise ValueError("mechanism field element topology differs from baseline")
    if baseline.displacement_magnitude_m.shape != mechanism.displacement_magnitude_m.shape:
        raise ValueError("displacement magnitude field shapes differ")
    if baseline.von_mises_stress_Pa.shape != mechanism.von_mises_stress_Pa.shape:
        raise ValueError("von Mises stress field shapes differ")
    if baseline.equivalent_plastic_strain.shape != mechanism.equivalent_plastic_strain.shape:
        raise ValueError("equivalent plastic strain field shapes differ")
    arrays = (
        baseline.displacement_magnitude_m,
        mechanism.displacement_magnitude_m,
        baseline.von_mises_stress_Pa,
        mechanism.von_mises_stress_Pa,
        baseline.equivalent_plastic_strain,
        mechanism.equivalent_plastic_strain,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("mechanism field comparison requires finite arrays")
    return MechanismFieldDifference(
        node_coordinates_m=np.array(baseline.node_coordinates_m, copy=True),
        element_connectivity=np.array(baseline.element_connectivity, copy=True),
        displacement_magnitude_difference_m=(
            np.asarray(mechanism.displacement_magnitude_m, dtype=float)
            - np.asarray(baseline.displacement_magnitude_m, dtype=float)
        ),
        von_mises_stress_difference_Pa=(
            np.asarray(mechanism.von_mises_stress_Pa, dtype=float)
            - np.asarray(baseline.von_mises_stress_Pa, dtype=float)
        ),
        equivalent_plastic_strain_difference=(
            np.asarray(mechanism.equivalent_plastic_strain, dtype=float)
            - np.asarray(baseline.equivalent_plastic_strain, dtype=float)
        ),
    )


def _triangle_data(mesh: meshio.Mesh, name: str) -> np.ndarray:
    try:
        return np.asarray(mesh.cell_data_dict[name]["triangle"])
    except KeyError as exc:
        raise MechanismFieldEvolutionArtifactError(f"VTU field is missing: {name}") from exc


def _setup_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise MechanismFieldEvolutionArtifactError(
            "mechanism field evolution path escapes its directory"
        ) from exc
    return candidate


def _write_difference_vtu(
    baseline_path: Path, mechanism_path: Path, output_path: Path
) -> None:
    baseline = meshio.read(baseline_path)
    mechanism = meshio.read(mechanism_path)
    if not np.array_equal(baseline.points, mechanism.points):
        raise MechanismFieldEvolutionArtifactError("route VTU node coordinates differ")
    baseline_triangles = np.asarray(baseline.cells_dict["triangle"])
    mechanism_triangles = np.asarray(mechanism.cells_dict["triangle"])
    if not np.array_equal(baseline_triangles, mechanism_triangles):
        raise MechanismFieldEvolutionArtifactError("route VTU element topology differs")
    displacement = (
        np.asarray(mechanism.point_data["displacement_magnitude_m"])
        - np.asarray(baseline.point_data["displacement_magnitude_m"])
    )
    stress = _triangle_data(mechanism, "von_mises_stress_Pa") - _triangle_data(
        baseline, "von_mises_stress_Pa"
    )
    plastic = _triangle_data(mechanism, "equivalent_plastic_strain") - _triangle_data(
        baseline, "equivalent_plastic_strain"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        output_path,
        meshio.Mesh(
            points=baseline.points,
            cells=[("triangle", baseline_triangles)],
            point_data={"displacement_magnitude_difference_m": displacement},
            cell_data={
                "von_mises_stress_difference_Pa": [stress],
                "equivalent_plastic_strain_difference": [plastic],
            },
        ),
    )


class MechanismFieldEvolutionExporter:
    """Stream two committed routes and finalize synchronized comparison artifacts."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self._baseline: list[dict[str, object]] = []
        self._mechanism: list[dict[str, object]] = []

    def on_baseline_snapshot(self, snapshot: TargetFieldSnapshot) -> None:
        self._on_snapshot("baseline", self._baseline, snapshot)

    def on_mechanism_snapshot(self, snapshot: TargetFieldSnapshot) -> None:
        self._on_snapshot("mechanism", self._mechanism, snapshot)

    def _on_snapshot(
        self, route: str, records: list[dict[str, object]], snapshot: TargetFieldSnapshot
    ) -> None:
        expected = len(records)
        if snapshot.frame_index != expected:
            raise MechanismFieldEvolutionArtifactError(
                f"{route} target snapshot frame index is not continuous"
            )
        stem = f"position_{snapshot.frame_index:04d}"
        relative = f"{route}/vtu/{stem}.vtu"
        write_target_vtu(snapshot, self.directory / relative)
        records.append({
            "frame_index": snapshot.frame_index,
            "position_id": snapshot.position_id,
            "motion_coordinate_m": snapshot.motion_coordinate_m,
            "wheel_lowest_point_x_m": snapshot.wheel_lowest_point_x_m,
            "pass_state": snapshot.pass_state,
            "contact_ratio": snapshot.contact_ratio,
            "Fx_N": snapshot.Fx_N,
            "Fy_N": snapshot.Fy_N,
            "accumulated_plastic_dissipation_J": snapshot.accumulated_plastic_dissipation_J,
            "is_final_unloaded_target": snapshot.is_final_unloaded_target,
            "vtu_path": relative,
            "png_path": f"{route}/frames/{stem}.png",
        })

    def finalize(self) -> dict[str, object]:
        if not self._baseline or len(self._baseline) != len(self._mechanism):
            raise MechanismFieldEvolutionArtifactError(
                "baseline and mechanism target frame counts must match and be positive"
            )
        frames: list[dict[str, object]] = []
        original_maxima = {
            "displacement_magnitude_m": 0.0,
            "von_mises_stress_Pa": 0.0,
            "equivalent_plastic_strain": 0.0,
        }
        difference_maxima = {
            "displacement_magnitude_difference_m": 0.0,
            "von_mises_stress_difference_Pa": 0.0,
            "equivalent_plastic_strain_difference": 0.0,
        }
        for baseline, mechanism in zip(self._baseline, self._mechanism):
            shared_keys = (
                "frame_index", "position_id", "motion_coordinate_m",
                "wheel_lowest_point_x_m", "pass_state", "contact_ratio",
                "is_final_unloaded_target",
            )
            if any(baseline[key] != mechanism[key] for key in shared_keys):
                raise MechanismFieldEvolutionArtifactError(
                    "baseline and mechanism target metadata differ"
                )
            index = int(baseline["frame_index"])
            difference_vtu = f"difference/vtu/position_{index:04d}.vtu"
            difference_png = f"difference/frames/position_{index:04d}.png"
            _write_difference_vtu(
                self.directory / str(baseline["vtu_path"]),
                self.directory / str(mechanism["vtu_path"]),
                self.directory / difference_vtu,
            )
            for record in (baseline, mechanism):
                mesh = meshio.read(self.directory / str(record["vtu_path"]))
                original_maxima["displacement_magnitude_m"] = max(
                    original_maxima["displacement_magnitude_m"],
                    float(np.max(mesh.point_data["displacement_magnitude_m"])),
                )
                original_maxima["von_mises_stress_Pa"] = max(
                    original_maxima["von_mises_stress_Pa"],
                    float(np.max(_triangle_data(mesh, "von_mises_stress_Pa"))),
                )
                original_maxima["equivalent_plastic_strain"] = max(
                    original_maxima["equivalent_plastic_strain"],
                    float(np.max(_triangle_data(mesh, "equivalent_plastic_strain"))),
                )
            difference_mesh = meshio.read(self.directory / difference_vtu)
            difference_maxima["displacement_magnitude_difference_m"] = max(
                difference_maxima["displacement_magnitude_difference_m"],
                float(np.max(np.abs(difference_mesh.point_data["displacement_magnitude_difference_m"]))),
            )
            difference_maxima["von_mises_stress_difference_Pa"] = max(
                difference_maxima["von_mises_stress_difference_Pa"],
                float(np.max(np.abs(_triangle_data(difference_mesh, "von_mises_stress_difference_Pa")))),
            )
            difference_maxima["equivalent_plastic_strain_difference"] = max(
                difference_maxima["equivalent_plastic_strain_difference"],
                float(np.max(np.abs(_triangle_data(difference_mesh, "equivalent_plastic_strain_difference")))),
            )
            frames.append({
                **{key: baseline[key] for key in shared_keys},
                "baseline": {
                    "Fx_N": baseline["Fx_N"], "Fy_N": baseline["Fy_N"],
                    "accumulated_plastic_dissipation_J": baseline["accumulated_plastic_dissipation_J"],
                    "png_path": baseline["png_path"], "vtu_path": baseline["vtu_path"],
                },
                "mechanism": {
                    "Fx_N": mechanism["Fx_N"], "Fy_N": mechanism["Fy_N"],
                    "accumulated_plastic_dissipation_J": mechanism["accumulated_plastic_dissipation_J"],
                    "png_path": mechanism["png_path"], "vtu_path": mechanism["vtu_path"],
                },
                "difference": {"png_path": difference_png, "vtu_path": difference_vtu},
            })
        if not all(math.isfinite(value) and value >= 0.0 for value in (*original_maxima.values(), *difference_maxima.values())):
            raise MechanismFieldEvolutionArtifactError("mechanism field color ranges are invalid")
        for frame in frames:
            self._write_original_png(frame, "baseline", original_maxima)
            self._write_original_png(frame, "mechanism", original_maxima)
            self._write_difference_png(frame, difference_maxima)
        self._write_csv(frames)
        self._write_pvds(frames)
        manifest = {
            "result_format": RESULT_FORMAT,
            "frame_count": len(frames),
            "frame_index_start": 0,
            "frame_index_end": len(frames) - 1,
            "sequence_coordinate": {
                "field": "motion_coordinate_m", "unit": "m",
                "meaning": "wheel_motion_coordinate_not_time",
                "pvd_timestep_uses_motion_coordinate": True,
            },
            "field_locations": {
                "displacement_magnitude_m": "nodes_vector_P1",
                "von_mises_stress_Pa": "triangles_element_constant_unsmoothed",
                "equivalent_plastic_strain": "triangles_element_constant_unsmoothed",
                "displacement_magnitude_difference_m": "nodes_scalar_P1",
                "von_mises_stress_difference_Pa": "triangles_element_constant_unsmoothed",
                "equivalent_plastic_strain_difference": "triangles_element_constant_unsmoothed",
            },
            "shared_original_color_ranges": {
                name: {"minimum": 0.0, "maximum": maximum}
                for name, maximum in original_maxima.items()
            },
            "difference_color_ranges": {
                name: {"minimum": -maximum, "maximum": maximum}
                for name, maximum in difference_maxima.items()
            },
            "routes": {
                route: {"pvd_path": f"{route}/field_evolution.pvd"}
                for route in ROUTES
            },
            "frames": frames,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="ascii", newline="\n",
        )
        validate_mechanism_field_evolution(self.directory)
        return manifest

    def _write_original_png(
        self, frame: dict[str, object], route: str, maxima: dict[str, float]
    ) -> None:
        mesh = meshio.read(self.directory / str(frame[route]["vtu_path"]))
        values = (
            np.asarray(mesh.point_data["displacement_magnitude_m"]) * 1e6,
            _triangle_data(mesh, "von_mises_stress_Pa") * 1e-6,
            _triangle_data(mesh, "equivalent_plastic_strain"),
        )
        scaled = (
            maxima["displacement_magnitude_m"] * 1e6,
            maxima["von_mises_stress_Pa"] * 1e-6,
            maxima["equivalent_plastic_strain"],
        )
        definitions = (
            ("Displacement Magnitude", "Displacement [um]", "magma", "gouraud"),
            ("von Mises Stress", "von Mises [MPa]", "inferno", "flat"),
            ("Equivalent Plastic Strain", "Equivalent plastic strain [-]", "viridis", "flat"),
        )
        self._plot(mesh, values, scaled, definitions, self.directory / str(frame[route]["png_path"]), f"{route.title()} Target Field", frame, symmetric=False)

    def _write_difference_png(
        self, frame: dict[str, object], maxima: dict[str, float]
    ) -> None:
        mesh = meshio.read(self.directory / str(frame["difference"]["vtu_path"]))
        values = (
            np.asarray(mesh.point_data["displacement_magnitude_difference_m"]) * 1e6,
            _triangle_data(mesh, "von_mises_stress_difference_Pa") * 1e-6,
            _triangle_data(mesh, "equivalent_plastic_strain_difference"),
        )
        scaled = (
            maxima["displacement_magnitude_difference_m"] * 1e6,
            maxima["von_mises_stress_difference_Pa"] * 1e-6,
            maxima["equivalent_plastic_strain_difference"],
        )
        definitions = (
            ("Displacement Magnitude Difference", "Difference [um]", "coolwarm", "gouraud"),
            ("von Mises Stress Difference", "Difference [MPa]", "coolwarm", "flat"),
            ("Equivalent Plastic Strain Difference", "Difference [-]", "coolwarm", "flat"),
        )
        self._plot(mesh, values, scaled, definitions, self.directory / str(frame["difference"]["png_path"]), "Mechanism Minus Baseline", frame, symmetric=True)

    def _plot(self, mesh, values, maxima, definitions, path, title, frame, *, symmetric):
        _setup_matplotlib()
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        path.parent.mkdir(parents=True, exist_ok=True)
        xy = mesh.points[:, :2] * 1e3
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells_dict["triangle"])
        with plt.rc_context(plot_style()):
            figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
            for axis, data, maximum, definition in zip(axes, values, maxima, definitions):
                panel, label, cmap, shading = definition
                extent = maximum if maximum > 0.0 else 1.0
                kwargs = {"shading": shading, "cmap": cmap, "vmin": -extent if symmetric else 0.0, "vmax": extent}
                image = axis.tripcolor(tri, facecolors=data, **kwargs) if shading == "flat" else axis.tripcolor(tri, data, **kwargs)
                axis.triplot(tri, color="k", linewidth=0.15, alpha=0.2)
                axis.set_aspect("equal", adjustable="box"); axis.set_xlabel("x [mm]"); axis.set_ylabel("y [mm]"); axis.set_title(panel); style_axis(axis)
                colorbar = figure.colorbar(image, ax=axis, label=label)
                if maximum == 0.0: colorbar.set_ticks([0.0])
            figure.suptitle(
                f"{title} {int(frame['frame_index']):04d} | state={frame['pass_state']} | motion={float(frame['motion_coordinate_m']):.6g} m\nFixed mesh, committed target position; sequence coordinate is not transient time",
                fontweight="bold",
            )
            save_png(figure, path); plt.close(figure)

    def _write_csv(self, frames: list[dict[str, object]]) -> None:
        with (self.directory / "position_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=POSITION_CSV_FIELDS); writer.writeheader()
            for frame in frames:
                writer.writerow({
                    "frame_index": frame["frame_index"], "position_id": frame["position_id"],
                    "motion_coordinate_m": frame["motion_coordinate_m"],
                    "wheel_lowest_point_x_m": frame["wheel_lowest_point_x_m"],
                    "pass_state": frame["pass_state"], "contact_ratio": frame["contact_ratio"],
                    "is_final_unloaded_target": str(frame["is_final_unloaded_target"]).lower(),
                    "baseline_Fx_N": frame["baseline"]["Fx_N"], "baseline_Fy_N": frame["baseline"]["Fy_N"],
                    "mechanism_Fx_N": frame["mechanism"]["Fx_N"], "mechanism_Fy_N": frame["mechanism"]["Fy_N"],
                    "baseline_accumulated_plastic_dissipation_J": frame["baseline"]["accumulated_plastic_dissipation_J"],
                    "mechanism_accumulated_plastic_dissipation_J": frame["mechanism"]["accumulated_plastic_dissipation_J"],
                    "baseline_png_path": frame["baseline"]["png_path"], "baseline_vtu_path": frame["baseline"]["vtu_path"],
                    "mechanism_png_path": frame["mechanism"]["png_path"], "mechanism_vtu_path": frame["mechanism"]["vtu_path"],
                    "difference_png_path": frame["difference"]["png_path"], "difference_vtu_path": frame["difference"]["vtu_path"],
                })

    def _write_pvds(self, frames: list[dict[str, object]]) -> None:
        for route in ROUTES:
            vtk = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
            collection = ET.SubElement(vtk, "Collection")
            for frame in frames:
                relative = Path(str(frame[route]["vtu_path"])).relative_to(route).as_posix()
                ET.SubElement(collection, "DataSet", timestep=f"{float(frame['motion_coordinate_m']):.17g}", group="", part="0", file=relative)
            target = self.directory / route / "field_evolution.pvd"; target.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(vtk).write(target, encoding="utf-8", xml_declaration=True)


def validate_mechanism_field_evolution(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    try:
        text = (root / "manifest.json").read_text(encoding="ascii")
        if "NaN" in text or "Infinity" in text:
            raise ValueError("manifest contains non-finite JSON")
        manifest = json.loads(text)
        if manifest.get("result_format") != RESULT_FORMAT:
            raise ValueError("manifest result_format is invalid")
        frames = manifest["frames"]; count = manifest["frame_count"]
        if type(count) is not int or count <= 0 or len(frames) != count:
            raise ValueError("manifest frame count is invalid")
        if [frame["frame_index"] for frame in frames] != list(range(count)):
            raise ValueError("manifest frame indices are invalid")
        if frames[-1]["is_final_unloaded_target"] is not True:
            raise ValueError("last target frame is not marked unloaded")
        with (root / "position_history.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream); rows = list(reader)
            if tuple(reader.fieldnames or ()) != POSITION_CSV_FIELDS or len(rows) != count:
                raise ValueError("position history CSV is invalid")
        reference_points = None; reference_triangles = None; paths: set[Path] = set()
        for route in ROUTES:
            datasets = ET.parse(root / route / "field_evolution.pvd").getroot().findall(".//DataSet")
            if len(datasets) != count: raise ValueError(f"{route} PVD dataset count is invalid")
            for frame, dataset in zip(frames, datasets):
                vtu = _safe_relative(root, str(frame[route]["vtu_path"])); png = _safe_relative(root, str(frame[route]["png_path"]))
                if vtu in paths or png in paths: raise ValueError("route artifact paths are duplicated")
                paths.update((vtu, png))
                expected_pvd = Path(str(frame[route]["vtu_path"])).relative_to(route).as_posix()
                if dataset.attrib.get("file") != expected_pvd or float(dataset.attrib["timestep"]) != float(frame["motion_coordinate_m"]):
                    raise ValueError(f"{route} PVD metadata does not match manifest")
                if not vtu.is_file() or vtu.stat().st_size <= 0 or not png.is_file() or png.stat().st_size <= 0:
                    raise ValueError(f"{route} frame artifact is missing or empty")
                mesh = meshio.read(vtu); points = np.asarray(mesh.points); triangles = np.asarray(mesh.cells_dict["triangle"])
                if reference_points is None: reference_points, reference_triangles = points, triangles
                elif not np.array_equal(points, reference_points) or not np.array_equal(triangles, reference_triangles):
                    raise ValueError("route or frame mesh differs")
                arrays = (
                    (mesh.point_data["displacement_magnitude_difference_m"], _triangle_data(mesh, "von_mises_stress_difference_Pa"), _triangle_data(mesh, "equivalent_plastic_strain_difference"))
                    if route == "difference" else
                    (mesh.point_data["displacement_magnitude_m"], _triangle_data(mesh, "von_mises_stress_Pa"), _triangle_data(mesh, "equivalent_plastic_strain"))
                )
                if not all(np.all(np.isfinite(array)) for array in arrays): raise ValueError("route VTU contains non-finite values")
        return manifest
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ET.ParseError) as exc:
        raise MechanismFieldEvolutionArtifactError(
            f"mechanism field evolution validation failed: {exc}"
        ) from exc


__all__ = [
    "FIELD_EVOLUTION_DIRECTORY", "MechanismFieldDifference",
    "MechanismFieldEvolutionArtifactError", "MechanismFieldEvolutionExporter",
    "RESULT_FORMAT", "calculate_field_difference", "validate_mechanism_field_evolution",
]
