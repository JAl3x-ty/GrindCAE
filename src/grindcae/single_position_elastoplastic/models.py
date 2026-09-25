"""Strict SI input contract for Phase 6A.3 single-position grinding FEM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from grindcae.domain import MeshSettings, ModelValidationError, Rectangle
from grindcae.elastoplastic import BilinearIsotropicMaterial, MaterialPointValidationError
from grindcae.elastoplastic_fem.models import (
    ElastoplasticFemCase,
    FixedBoundary,
    IncrementSettings,
    NewtonSettings,
    PlaneStrainAnalysis,
    StepControlSettings,
    TopLineLoad,
)
from grindcae.pass_loading import PassLoadCase, PassLoadValidationError


SINGLE_POSITION_ELASTOPLASTIC_SCHEMA_VERSION = 1
UNIT_SYSTEM = "SI"
MODEL_TYPE = "single_position_evolved_surface_elastoplastic_loading_unloading"


class SinglePositionElastoplasticValidationError(ValueError):
    """Raised when a Phase 6A.3 input violates its strict contract."""


def _fail(field: str, unit: str, reason: str) -> None:
    raise SinglePositionElastoplasticValidationError(f"{field} [{unit}]: {reason}")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field, "JSON object", "must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing fields {missing}")
    if unexpected:
        details.append(f"unexpected fields {unexpected}")
    if details:
        _fail(field, "exact field set", " and ".join(details))


@dataclass(frozen=True, slots=True)
class SinglePositionElastoplasticCase:
    """One validated pass-load case plus the retained Phase 6A.2 controls."""

    single_position_elastoplastic_schema_version: int
    unit_system: str
    model_type: str
    pass_load: PassLoadCase
    material: BilinearIsotropicMaterial
    mesh: MeshSettings
    analysis: PlaneStrainAnalysis
    fixed_boundary: FixedBoundary
    increments: IncrementSettings
    newton: NewtonSettings
    step_control: StepControlSettings
    geometry: Rectangle = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.single_position_elastoplastic_schema_version) is not int
            or self.single_position_elastoplastic_schema_version
            != SINGLE_POSITION_ELASTOPLASTIC_SCHEMA_VERSION
        ):
            _fail(
                "single_position_elastoplastic_schema_version",
                "integer version",
                "must be the strict integer 1",
            )
        if self.unit_system != UNIT_SYSTEM:
            _fail("unit_system", "SI", "only SI is supported")
        if self.model_type != MODEL_TYPE:
            _fail("model_type", MODEL_TYPE, "unsupported model type")
        if not isinstance(self.pass_load, PassLoadCase):
            _fail("pass_load", "PassLoadCase", "has the wrong object type")
        if not isinstance(self.material, BilinearIsotropicMaterial):
            _fail("material", "BilinearIsotropicMaterial", "has the wrong object type")
        if not isinstance(self.mesh, MeshSettings):
            _fail("mesh", "MeshSettings", "has the wrong object type")
        if not isinstance(self.analysis, PlaneStrainAnalysis):
            _fail("analysis", "PlaneStrainAnalysis", "has the wrong object type")
        if not isinstance(self.fixed_boundary, FixedBoundary):
            _fail("fixed_boundary", "FixedBoundary", "has the wrong object type")
        if not isinstance(self.increments, IncrementSettings):
            _fail("increments", "IncrementSettings", "has the wrong object type")
        if not isinstance(self.newton, NewtonSettings):
            _fail("newton", "NewtonSettings", "has the wrong object type")
        if not isinstance(self.step_control, StepControlSettings):
            _fail("step_control", "StepControlSettings", "has the wrong object type")
        grinding_width = self.pass_load.force_model.process.grinding_width_m
        if not math.isclose(
            grinding_width,
            self.analysis.thickness,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            _fail(
                "pass_load.force_model.process.grinding_width_m",
                "m",
                "must match analysis.thickness without averaging or correction",
            )
        trajectory = self.pass_load.trajectory
        object.__setattr__(
            self,
            "geometry",
            Rectangle(
                width=trajectory.workpiece.length_m,
                height=trajectory.workpiece.original_surface_height_m,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "SinglePositionElastoplasticCase":
        root = _object(value, "single_position_elastoplastic_case")
        _exact(
            root,
            {
                "single_position_elastoplastic_schema_version",
                "unit_system",
                "model_type",
                "pass_load",
                "material",
                "mesh",
                "analysis",
                "fixed_boundary",
                "increments",
                "newton",
                "step_control",
            },
            "single_position_elastoplastic_case",
        )
        mesh_data = _object(root["mesh"], "mesh")
        _exact(mesh_data, {"target_size"}, "mesh")
        try:
            return cls(
                single_position_elastoplastic_schema_version=root[
                    "single_position_elastoplastic_schema_version"
                ],
                unit_system=root["unit_system"],
                model_type=root["model_type"],
                pass_load=PassLoadCase.from_mapping(root["pass_load"]),
                material=BilinearIsotropicMaterial.from_mapping(root["material"]),
                mesh=MeshSettings(target_size=mesh_data["target_size"]),
                analysis=PlaneStrainAnalysis.from_mapping(root["analysis"]),
                fixed_boundary=FixedBoundary.from_mapping(root["fixed_boundary"]),
                increments=IncrementSettings.from_mapping(root["increments"]),
                newton=NewtonSettings.from_mapping(root["newton"]),
                step_control=StepControlSettings.from_mapping(root["step_control"]),
            )
        except SinglePositionElastoplasticValidationError:
            raise
        except (
            PassLoadValidationError,
            MaterialPointValidationError,
            ModelValidationError,
            ValueError,
        ) as exc:
            raise SinglePositionElastoplasticValidationError(str(exc)) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "SinglePositionElastoplasticCase":
        input_path = Path(path).expanduser().resolve()
        try:
            with input_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            _fail("input_file", "existing UTF-8 JSON", f"does not exist: {input_path}")
        except json.JSONDecodeError as exc:
            _fail(
                "input_file",
                "valid UTF-8 JSON",
                f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            )
        except (UnicodeDecodeError, OSError) as exc:
            _fail("input_file", "readable UTF-8 JSON", f"cannot be read: {exc}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "single_position_elastoplastic_schema_version": (
                self.single_position_elastoplastic_schema_version
            ),
            "unit_system": self.unit_system,
            "model_type": self.model_type,
            "pass_load": self.pass_load.to_dict(),
            "material": self.material.to_dict(),
            "mesh": {"target_size": self.mesh.target_size},
            "analysis": self.analysis.to_dict(),
            "fixed_boundary": self.fixed_boundary.to_dict(),
            "increments": self.increments.to_dict(),
            "newton": self.newton.to_dict(),
            "step_control": self.step_control.to_dict(),
        }

    def to_elastoplastic_fem_case(
        self,
        *,
        peak_force_x_N: float,
        peak_force_y_N: float,
    ) -> ElastoplasticFemCase:
        """Build the retained 6A.2 control object without using its interval load."""

        return ElastoplasticFemCase(
            elastoplastic_fem_schema_version=1,
            unit_system=self.unit_system,
            model_type="small_strain_plane_strain_j2_fixed_mesh",
            geometry=self.geometry,
            material=self.material,
            mesh=self.mesh,
            analysis=self.analysis,
            fixed_boundary=self.fixed_boundary,
            load=TopLineLoad(
                boundary="top",
                distribution="uniform_over_interval",
                x_start_m=0.0,
                x_end_m=self.geometry.width,
                peak_force_x_N=peak_force_x_N,
                peak_force_y_N=peak_force_y_N,
            ),
            increments=self.increments,
            newton=self.newton,
            step_control=self.step_control,
        )
