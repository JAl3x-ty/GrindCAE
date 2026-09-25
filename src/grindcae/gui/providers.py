"""Minimal provider protocols reserved for later workbench phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .project import ProjectError


EXTENSION_IDENTIFIERS = {
    "geometry_source": ("parametric_2d", "step_cad"),
    "mesh_type": ("triangle", "quadrilateral"),
    "physics": ("mechanical", "thermo_mechanical"),
    "grinding_scale": ("statistical_equivalent", "single_grain"),
    "process_mode": ("single_position", "single_pass", "multi_pass"),
    "surface_output": ("displacement", "stress", "plastic_strain", "roughness"),
}


class GeometryProvider(Protocol):
    def validate_geometry(self) -> tuple[bool, str]: ...

    def build_geometry_preview(self) -> Any: ...

    def to_geometry_schema(self) -> dict[str, Any]: ...


class MeshProvider(Protocol):
    def validate_mesh_settings(self) -> tuple[bool, str]: ...

    def generate_mesh(self, output_path: str | Path) -> Path: ...

    def mesh_summary(
        self, mesh_path: str | Path, output_directory: str | Path
    ) -> dict[str, Any]: ...

    def mesh_artifacts(self, output_directory: str | Path) -> dict[str, str]: ...


class MaterialProvider(Protocol):
    def validate_material(self) -> tuple[bool, str]: ...

    def to_material_schema(self) -> dict[str, Any]: ...

    def material_summary(self) -> dict[str, str]: ...


class WheelProvider(Protocol):
    def validate_wheel(self) -> tuple[bool, str]: ...

    def resolve_grit(self) -> dict[str, Any]: ...

    def to_wheel_schema(self) -> dict[str, Any]: ...

    def wheel_summary(self) -> dict[str, str]: ...


class ProcessProvider(Protocol):
    def validate_process(self) -> tuple[bool, str]: ...

    def to_process_schema(self) -> dict[str, Any]: ...

    def process_summary(self) -> dict[str, str]: ...


class AnalysisProvider(Protocol):
    def validate_case(self) -> tuple[bool, str]: ...

    def build_solver_input(self) -> Any: ...

    def run(self) -> Any: ...

    def read_summary(self) -> dict[str, Any]: ...

    def list_artifacts(self) -> tuple[str, ...]: ...


class ResultAdapter(Protocol):
    def summary_text(self) -> str: ...

    def plots(self) -> tuple[str, ...]: ...

    def tables(self) -> tuple[str, ...]: ...

    def warnings(self) -> tuple[str, ...]: ...


class DisabledProvider:
    """Explicit placeholder used where a future provider is not enabled yet."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def validate(self) -> tuple[bool, str]:
        return False, self.reason

    validate_geometry = validate
    validate_mesh_settings = validate
    validate_material = validate
    validate_wheel = validate
    validate_process = validate
    validate_case = validate

    def summary(self) -> dict[str, str]:
        return {"status": "disabled", "reason": self.reason}

    def summary_text(self) -> str:
        return self.reason

    def plots(self) -> tuple[str, ...]:
        return ()

    def tables(self) -> tuple[str, ...]:
        return ()

    def warnings(self) -> tuple[str, ...]:
        return (self.reason,)

    def run(self) -> None:
        raise ProjectError(f"暂未启用：{self.reason}")

    generate_mesh = run
    build_geometry_preview = run
    build_solver_input = run

    def to_geometry_schema(self) -> dict[str, Any]:
        self.run()

    def mesh_summary(self) -> dict[str, Any]:
        self.run()

    def to_material_schema(self) -> dict[str, Any]:
        self.run()

    def resolve_grit(self) -> dict[str, Any]:
        self.run()

    def to_wheel_schema(self) -> dict[str, Any]:
        self.run()

    def to_process_schema(self) -> dict[str, Any]:
        self.run()

    def read_summary(self) -> dict[str, Any]:
        self.run()

    def list_artifacts(self) -> tuple[str, ...]:
        return ()
