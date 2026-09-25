"""Target-position field sequence export for the ordinary Phase 6B.2 route."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from matplotlib import image as matplotlib_image
import meshio
import numpy as np

from grindcae.plotting import plot_style, save_png, style_axis

from .workflow import TargetFieldSnapshot


FIELD_EVOLUTION_DIRECTORY = "field_evolution"
POSITION_CSV_FIELDS = (
    "frame_index",
    "position_id",
    "motion_coordinate_m",
    "wheel_lowest_point_x_m",
    "pass_state",
    "contact_ratio",
    "Fx_N",
    "Fy_N",
    "accumulated_plastic_dissipation_J",
    "is_final_unloaded_target",
    "vtu_path",
    "png_path",
)


class FieldEvolutionArtifactError(RuntimeError):
    """Raised when a target-position field sequence is incomplete or unsafe."""


def _setup_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grindcae-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)


def _class_codes(values: tuple[str, ...]) -> np.ndarray:
    codes = {"initial": 0, "elastic": 1, "plastic": 2}
    return np.asarray([codes[value] for value in values], dtype=np.int32)


def write_target_vtu(snapshot: TargetFieldSnapshot, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = snapshot.fields
    points = np.column_stack((fields.node_coordinates_m, np.zeros(fields.node_count)))
    displacement = np.column_stack(
        (fields.nodal_displacements_m, np.zeros(fields.node_count))
    )
    plastic = fields.plastic_strain_tensor
    meshio.write(
        output,
        meshio.Mesh(
            points=points,
            cells=[("triangle", fields.element_connectivity)],
            point_data={
                "displacement_m": displacement,
                "displacement_magnitude_m": fields.displacement_magnitude_m,
            },
            cell_data={
                "strain_xx": [fields.strain_xx],
                "strain_yy": [fields.strain_yy],
                "engineering_shear_strain_xy": [fields.engineering_shear_strain_xy],
                "strain_zz": [fields.strain_zz],
                "stress_xx_Pa": [fields.stress_xx_Pa],
                "stress_yy_Pa": [fields.stress_yy_Pa],
                "stress_zz_Pa": [fields.stress_zz_Pa],
                "shear_stress_xy_Pa": [fields.shear_stress_xy_Pa],
                "von_mises_stress_Pa": [fields.von_mises_stress_Pa],
                "plastic_strain_xx": [plastic[:, 0, 0]],
                "plastic_strain_yy": [plastic[:, 1, 1]],
                "plastic_strain_zz": [plastic[:, 2, 2]],
                "plastic_engineering_shear_strain_xy": [2.0 * plastic[:, 0, 1]],
                "equivalent_plastic_strain": [fields.equivalent_plastic_strain],
                "current_yield_strength_Pa": [fields.current_yield_strength_Pa],
                "increment_class_code": [_class_codes(fields.increment_class)],
            },
        ),
    )
    return output


def _triangle_data(mesh: meshio.Mesh, name: str) -> np.ndarray:
    try:
        return np.asarray(mesh.cell_data_dict[name]["triangle"])
    except KeyError as exc:
        raise FieldEvolutionArtifactError(f"VTU field is missing: {name}") from exc


class FieldEvolutionExporter:
    """Stream target VTUs during solving, then render them with global color ranges."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.vtu_directory = self.directory / "vtu"
        self.frame_directory = self.directory / "frames"
        self._records: list[dict[str, object]] = []

    def on_target_snapshot(self, snapshot: TargetFieldSnapshot) -> None:
        expected = len(self._records)
        if snapshot.frame_index != expected:
            raise FieldEvolutionArtifactError("target snapshot frame index is not continuous")
        filename = f"position_{snapshot.frame_index:04d}"
        write_target_vtu(snapshot, self.vtu_directory / f"{filename}.vtu")
        self._records.append(
            {
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
                "vtu_path": f"vtu/{filename}.vtu",
                "png_path": f"frames/{filename}.png",
            }
        )

    def finalize(self) -> dict[str, object]:
        if not self._records:
            raise FieldEvolutionArtifactError("field evolution contains no target frames")
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        ranges = {
            "displacement_magnitude_m": 0.0,
            "von_mises_stress_Pa": 0.0,
            "equivalent_plastic_strain": 0.0,
        }
        for record in self._records:
            mesh = meshio.read(self.directory / str(record["vtu_path"]))
            ranges["displacement_magnitude_m"] = max(
                ranges["displacement_magnitude_m"],
                float(np.max(mesh.point_data["displacement_magnitude_m"])),
            )
            ranges["von_mises_stress_Pa"] = max(
                ranges["von_mises_stress_Pa"],
                float(np.max(_triangle_data(mesh, "von_mises_stress_Pa"))),
            )
            ranges["equivalent_plastic_strain"] = max(
                ranges["equivalent_plastic_strain"],
                float(np.max(_triangle_data(mesh, "equivalent_plastic_strain"))),
            )
        if not all(math.isfinite(value) and value >= 0.0 for value in ranges.values()):
            raise FieldEvolutionArtifactError("field evolution color range is invalid")
        for record in self._records:
            self._write_frame_png(record, ranges)
        self._write_position_csv()
        self._write_pvd()
        manifest = self._write_manifest(ranges)
        validate_field_evolution(self.directory)
        return manifest

    def _write_frame_png(
        self, record: dict[str, object], ranges: dict[str, float]
    ) -> None:
        _setup_matplotlib()
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        mesh = meshio.read(self.directory / str(record["vtu_path"]))
        xy = mesh.points[:, :2] * 1.0e3
        triangles = mesh.cells_dict["triangle"]
        triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
        values = (
            np.asarray(mesh.point_data["displacement_magnitude_m"]) * 1.0e6,
            _triangle_data(mesh, "von_mises_stress_Pa") * 1.0e-6,
            _triangle_data(mesh, "equivalent_plastic_strain"),
        )
        maxima = (
            ranges["displacement_magnitude_m"] * 1.0e6,
            ranges["von_mises_stress_Pa"] * 1.0e-6,
            ranges["equivalent_plastic_strain"],
        )
        definitions = (
            ("Displacement Magnitude", "Displacement [um]", "magma", "gouraud"),
            ("von Mises Stress", "von Mises [MPa]", "inferno", "flat"),
            ("Equivalent Plastic Strain", "Equivalent plastic strain [-]", "viridis", "flat"),
        )
        with plt.rc_context(plot_style()):
            figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
            for axis, data, maximum, definition in zip(axes, values, maxima, definitions):
                title, label, cmap, shading = definition
                upper = maximum if maximum > 0.0 else 1.0
                kwargs = {"shading": shading, "cmap": cmap, "vmin": 0.0, "vmax": upper}
                if shading == "flat":
                    image = axis.tripcolor(triangulation, facecolors=data, **kwargs)
                else:
                    image = axis.tripcolor(triangulation, data, **kwargs)
                axis.triplot(triangulation, color="k", linewidth=0.15, alpha=0.2)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("x [mm]")
                axis.set_ylabel("y [mm]")
                axis.set_title(title)
                style_axis(axis)
                colorbar = figure.colorbar(image, ax=axis, label=label)
                if maximum == 0.0:
                    colorbar.set_ticks([0.0])
            figure.suptitle(
                f"Single-Pass Target Field {int(record['frame_index']):04d} | "
                f"state={record['pass_state']} | motion={float(record['motion_coordinate_m']):.6g} m\n"
                "Fixed mesh, committed target position; sequence coordinate is not transient time",
                fontweight="bold",
            )
            save_png(figure, self.directory / str(record["png_path"]))
            plt.close(figure)

    def _write_position_csv(self) -> None:
        with (self.directory / "position_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=POSITION_CSV_FIELDS)
            writer.writeheader()
            for record in self._records:
                row = dict(record)
                row["is_final_unloaded_target"] = str(
                    row["is_final_unloaded_target"]
                ).lower()
                writer.writerow(row)

    def _write_pvd(self) -> None:
        vtk_file = ET.Element(
            "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
        )
        collection = ET.SubElement(vtk_file, "Collection")
        for record in self._records:
            ET.SubElement(
                collection,
                "DataSet",
                timestep=f"{float(record['motion_coordinate_m']):.17g}",
                group="",
                part="0",
                file=str(record["vtu_path"]).replace("\\", "/"),
            )
        ET.ElementTree(vtk_file).write(
            self.directory / "field_evolution.pvd",
            encoding="utf-8",
            xml_declaration=True,
        )

    def _write_manifest(self, ranges: dict[str, float]) -> dict[str, object]:
        manifest = {
            "result_format": "grindcae_history_pass_target_field_evolution",
            "frame_count": len(self._records),
            "frame_index_start": 0,
            "frame_index_end": len(self._records) - 1,
            "sequence_coordinate": {
                "field": "motion_coordinate_m",
                "unit": "m",
                "meaning": "wheel_motion_coordinate_not_time",
                "pvd_timestep_uses_motion_coordinate": True,
            },
            "field_locations": {
                "displacement_magnitude_m": "nodes_vector_P1",
                "von_mises_stress_Pa": "triangles_element_constant_unsmoothed",
                "equivalent_plastic_strain": "triangles_element_constant_unsmoothed",
            },
            "color_ranges": {
                name: {"minimum": 0.0, "maximum": maximum}
                for name, maximum in ranges.items()
            },
            "frames": list(self._records),
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="ascii",
            newline="\n",
        )
        return manifest


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise FieldEvolutionArtifactError("field evolution path escapes its directory") from exc
    return candidate


def validate_field_evolution(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    try:
        manifest_text = (root / "manifest.json").read_text(encoding="ascii")
        if "NaN" in manifest_text or "Infinity" in manifest_text:
            raise ValueError("manifest contains a non-finite JSON token")
        manifest = json.loads(manifest_text)
        records = manifest["frames"]
        count = int(manifest["frame_count"])
        if count <= 0 or len(records) != count:
            raise ValueError("manifest frame count is invalid")
        if [int(item["frame_index"]) for item in records] != list(range(count)):
            raise ValueError("manifest frame indices are invalid")
        with (root / "position_history.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if tuple(reader.fieldnames or ()) != POSITION_CSV_FIELDS:
                raise ValueError("position_history.csv header is invalid")
        if len(rows) != count:
            raise ValueError("position_history.csv row count is invalid")
        datasets = ET.parse(root / "field_evolution.pvd").getroot().findall(
            ".//DataSet"
        )
        if len(datasets) != count:
            raise ValueError("PVD dataset count is invalid")
        reference_points: np.ndarray | None = None
        reference_triangles: np.ndarray | None = None
        for record, dataset in zip(records, datasets):
            vtu = _safe_relative(root, str(record["vtu_path"]))
            png = _safe_relative(root, str(record["png_path"]))
            if dataset.attrib.get("file") != str(record["vtu_path"]).replace("\\", "/"):
                raise ValueError("PVD path does not match manifest")
            if float(dataset.attrib["timestep"]) != float(record["motion_coordinate_m"]):
                raise ValueError("PVD coordinate does not match manifest")
            mesh = meshio.read(vtu)
            points = np.asarray(mesh.points)
            triangles = np.asarray(mesh.cells_dict["triangle"])
            if reference_points is None:
                reference_points = points
                reference_triangles = triangles
            elif not np.array_equal(points, reference_points) or not np.array_equal(
                triangles, reference_triangles
            ):
                raise ValueError("field evolution mesh changed between frames")
            arrays = (
                mesh.point_data["displacement_m"],
                mesh.point_data["displacement_magnitude_m"],
                _triangle_data(mesh, "von_mises_stress_Pa"),
                _triangle_data(mesh, "equivalent_plastic_strain"),
                _triangle_data(mesh, "stress_zz_Pa"),
            )
            if not all(np.all(np.isfinite(array)) for array in arrays):
                raise ValueError("field evolution VTU contains a non-finite value")
            if np.asarray(mesh.point_data["displacement_m"]).shape != (points.shape[0], 3):
                raise ValueError("field evolution displacement vector shape is invalid")
            if np.asarray(mesh.point_data["displacement_magnitude_m"]).shape != (
                points.shape[0],
            ):
                raise ValueError("field evolution displacement magnitude shape is invalid")
            if any(np.asarray(array).shape != (triangles.shape[0],) for array in arrays[2:]):
                raise ValueError("field evolution triangle field shape is invalid")
            if np.any(np.asarray(mesh.point_data["displacement_magnitude_m"]) < 0.0):
                raise ValueError("field evolution displacement magnitude is negative")
            if np.any(_triangle_data(mesh, "von_mises_stress_Pa") < 0.0):
                raise ValueError("field evolution von Mises stress is negative")
            if np.any(_triangle_data(mesh, "equivalent_plastic_strain") < 0.0):
                raise ValueError("field evolution plastic strain is negative")
            if matplotlib_image.imread(png).size == 0:
                raise ValueError("field evolution PNG is unreadable")
        if rows[-1]["is_final_unloaded_target"] != "true":
            raise ValueError("last target frame is not marked unloaded")
        return manifest
    except (OSError, KeyError, TypeError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
        raise FieldEvolutionArtifactError(f"field evolution validation failed: {exc}") from exc


__all__ = [
    "FIELD_EVOLUTION_DIRECTORY",
    "POSITION_CSV_FIELDS",
    "FieldEvolutionArtifactError",
    "FieldEvolutionExporter",
    "validate_field_evolution",
    "write_target_vtu",
]
