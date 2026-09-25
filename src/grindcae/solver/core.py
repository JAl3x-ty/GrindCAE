"""Import a Gmsh mesh and solve the phase 2A plane-stress problem."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Protocol

import meshio
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from skfem import (
    Basis,
    ElementTriP1,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTri,
    asm,
    condense,
    solve,
)
from skfem.io.meshio import from_meshio
from skfem.models.elasticity import linear_elasticity

from grindcae.domain.models import (
    AnalysisSettings,
    LinearElasticMaterial,
    Rectangle,
    SimulationCase,
)


REQUIRED_PHYSICAL_GROUPS = (
    "domain",
    "fixed",
    "contact",
    "free_left",
    "free_right",
)
DEFAULT_MAX_DEGREES_OF_FREEDOM = 200_000
MINIMUM_LOCAL_ACTIVE_FACET_COUNT = 4


class SolverError(RuntimeError):
    """Raised when a mesh cannot be imported or the solve cannot proceed."""


class _PlaneStressCase(Protocol):
    """Structural fields shared by schema 3 and schema 4 solve inputs."""

    geometry: Rectangle
    material: LinearElasticMaterial
    analysis: AnalysisSettings


@dataclass(frozen=True, slots=True)
class ImportedMesh:
    """A validated first-order triangular mesh with named physical groups."""

    mesh: MeshTri
    source_path: Path

    @property
    def node_count(self) -> int:
        return int(self.mesh.nvertices)

    @property
    def triangle_count(self) -> int:
        return int(self.mesh.nelements)


@dataclass(frozen=True, slots=True)
class AssembledSystem:
    """The full finite element system before essential boundary condensation."""

    case: SimulationCase
    imported_mesh: ImportedMesh
    basis: Basis
    stiffness_matrix: spmatrix
    load_vector: NDArray[np.float64]
    fixed_dofs: NDArray[np.int32]
    fixed_x_dofs: NDArray[np.int32]
    fixed_y_dofs: NDArray[np.int32]
    input_normal_force: float
    input_tangential_force: float
    assembled_external_force_x: float
    assembled_external_force_y: float

    def __post_init__(self) -> None:
        if not isinstance(self.case, SimulationCase):
            raise SolverError(
                "assembled system case provenance must be a SimulationCase"
            )


@dataclass(frozen=True, slots=True)
class LocalLoadAssembledSystem:
    """Full system for a directional load on runtime-selected contact facets."""

    case: object
    imported_mesh: ImportedMesh
    basis: Basis
    stiffness_matrix: spmatrix
    load_vector: NDArray[np.float64]
    fixed_dofs: NDArray[np.int32]
    fixed_x_dofs: NDArray[np.int32]
    fixed_y_dofs: NDArray[np.int32]
    active_facets: NDArray[np.int32]
    active_facet_length: float
    mapped_force_x: float
    mapped_force_y: float
    line_load_x: float
    line_load_y: float
    assembled_external_force_x: float
    assembled_external_force_y: float

    def __post_init__(self) -> None:
        _validate_plane_stress_case(self.case)


@dataclass(frozen=True, slots=True)
class FacetLoadAssembledSystem:
    """Full system for independently specified global tractions on facets."""

    case: object
    imported_mesh: ImportedMesh
    basis: Basis
    stiffness_matrix: spmatrix
    load_vector: NDArray[np.float64]
    fixed_dofs: NDArray[np.int32]
    fixed_x_dofs: NDArray[np.int32]
    fixed_y_dofs: NDArray[np.int32]
    active_facets: NDArray[np.int32]
    active_facet_lengths: NDArray[np.float64]
    facet_line_load_x: NDArray[np.float64]
    facet_line_load_y: NDArray[np.float64]
    mapped_force_x: float
    mapped_force_y: float
    assembled_external_force_x: float
    assembled_external_force_y: float

    def __post_init__(self) -> None:
        _validate_plane_stress_case(self.case)


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Plane-stress solution summary and arrays used for verification."""

    node_count: int
    triangle_count: int
    total_degrees_of_freedom: int
    analysis_type: str
    thickness: float
    input_normal_force: float
    input_tangential_force: float
    assembled_external_force_x: float
    assembled_external_force_y: float
    fixed_reaction_x: float
    fixed_reaction_y: float
    balance_residual_x: float
    balance_residual_y: float
    balance_residual_norm: float
    maximum_displacement: float
    maximum_displacement_node: int
    maximum_displacement_coordinates: tuple[float, float]
    displacements: NDArray[np.float64] = field(repr=False)
    fixed_node_indices: NDArray[np.int32] = field(repr=False)
    external_load_vector: NDArray[np.float64] = field(repr=False)
    full_residual_vector: NDArray[np.float64] = field(repr=False)

    @property
    def full_reaction_vector(self) -> NDArray[np.float64]:
        """Return the full residual under the phase 2A compatibility name."""

        return self.full_residual_vector

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready result with units and physical meaning."""

        return {
            "result_format": "grindcae_phase_2a",
            "physical_assumptions": {
                "dimension": "2D",
                "kinematics": "small_deformation",
                "material": "isotropic_linear_elastic",
                "analysis_type": self.analysis_type,
                "element": "first_order_triangle_vector_P1",
                "fixed_boundary": "bottom, zero x and y displacement",
                "loaded_boundary": (
                    "top contact physical group with uniform equivalent line load"
                ),
                "free_boundaries": "left and right",
                "contact_model": (
                    "equivalent load boundary only; no true contact algorithm"
                ),
                "force_interpretation": (
                    "supplied engineering loads and computed support reactions; "
                    "not a prediction of real grinding force"
                ),
            },
            "units": {
                "length": "m",
                "thickness": "m",
                "force": "N",
                "line_load": "N/m",
                "displacement": "m",
            },
            "sign_convention": {
                "normal_force": "non-negative compressive magnitude Fn",
                "tangential_force": "signed magnitude Ft; positive is +x",
                "assembled_force": "Fx = Ft, Fy = -Fn",
                "support_reaction": "positive components follow global +x and +y",
                "balance_residual": "external force plus fixed-boundary reaction",
            },
            "mesh": {
                "node_count": self.node_count,
                "triangle_count": self.triangle_count,
                "total_degrees_of_freedom": self.total_degrees_of_freedom,
            },
            "analysis": {
                "type": self.analysis_type,
                "thickness": self.thickness,
            },
            "equivalent_input_loads": {
                "normal_force": self.input_normal_force,
                "tangential_force": self.input_tangential_force,
            },
            "assembled_external_force": {
                "Fx": self.assembled_external_force_x,
                "Fy": self.assembled_external_force_y,
            },
            "fixed_boundary_reaction": {
                "Rx": self.fixed_reaction_x,
                "Ry": self.fixed_reaction_y,
            },
            "balance_residual": {
                "x": self.balance_residual_x,
                "y": self.balance_residual_y,
                "norm": self.balance_residual_norm,
            },
            "maximum_displacement": {
                "magnitude": self.maximum_displacement,
                "node_index": int(self.maximum_displacement_node),
                "node_index_base": 0,
                "coordinates": list(self.maximum_displacement_coordinates),
            },
        }


@dataclass(frozen=True, slots=True)
class LocalLoadSolveResult:
    """Summary for an empirical force mapped to a local boundary load."""

    node_count: int
    triangle_count: int
    total_degrees_of_freedom: int
    analysis_type: str
    thickness: float
    active_facet_count: int
    active_facet_length: float
    mapped_force_x: float
    mapped_force_y: float
    line_load_x: float
    line_load_y: float
    assembled_external_force_x: float
    assembled_external_force_y: float
    fixed_reaction_x: float
    fixed_reaction_y: float
    balance_residual_x: float
    balance_residual_y: float
    balance_residual_norm: float
    maximum_displacement: float
    maximum_displacement_node: int
    maximum_displacement_coordinates: tuple[float, float]
    displacements: NDArray[np.float64] = field(repr=False)
    fixed_node_indices: NDArray[np.int32] = field(repr=False)
    external_load_vector: NDArray[np.float64] = field(repr=False)
    full_residual_vector: NDArray[np.float64] = field(repr=False)

    @property
    def full_reaction_vector(self) -> NDArray[np.float64]:
        """Return the full residual under the historical compatibility name."""

        return self.full_residual_vector

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary with mapped and assembled forces separate."""

        return {
            "result_format": "grindcae_phase_4b1_local_load_solver",
            "physical_assumptions": {
                "dimension": "2D",
                "kinematics": "small_deformation",
                "material": "isotropic_linear_elastic",
                "analysis_type": self.analysis_type,
                "element": "first_order_triangle_vector_P1",
                "fixed_boundary": "bottom, zero x and y displacement",
                "loaded_boundary": (
                    "runtime-selected active facets on the top contact boundary"
                ),
                "load_distribution": "uniform local line load",
                "load_source": (
                    "empirical grinding-force prediction mapped to a directional "
                    "local boundary load"
                ),
                "contact_model": (
                    "prescribed local traction only; no contact iteration, "
                    "friction law, or Hertz pressure distribution"
                ),
                "force_interpretation": (
                    "mapped signed force, assembled external force, and computed "
                    "fixed-boundary reaction are distinct quantities"
                ),
            },
            "units": {
                "length": "m",
                "thickness": "m",
                "force": "N",
                "line_load": "N/m",
                "displacement": "m",
            },
            "mesh": {
                "node_count": self.node_count,
                "triangle_count": self.triangle_count,
                "total_degrees_of_freedom": self.total_degrees_of_freedom,
                "active_facet_count": self.active_facet_count,
                "active_facet_length": self.active_facet_length,
            },
            "analysis": {
                "type": self.analysis_type,
                "thickness": self.thickness,
            },
            "mapped_signed_external_force": {
                "Fx": self.mapped_force_x,
                "Fy": self.mapped_force_y,
            },
            "uniform_local_line_load": {
                "qx": self.line_load_x,
                "qy": self.line_load_y,
            },
            "assembled_external_force": {
                "Fx": self.assembled_external_force_x,
                "Fy": self.assembled_external_force_y,
            },
            "fixed_boundary_reaction": {
                "Rx": self.fixed_reaction_x,
                "Ry": self.fixed_reaction_y,
            },
            "balance_residual": {
                "x": self.balance_residual_x,
                "y": self.balance_residual_y,
                "norm": self.balance_residual_norm,
            },
            "maximum_displacement": {
                "magnitude": self.maximum_displacement,
                "node_index": int(self.maximum_displacement_node),
                "node_index_base": 0,
                "coordinates": list(self.maximum_displacement_coordinates),
            },
        }


@dataclass(frozen=True, slots=True)
class FacetLoadSolveResult:
    """Summary for global tractions that may vary between loaded facets."""

    node_count: int
    triangle_count: int
    total_degrees_of_freedom: int
    analysis_type: str
    thickness: float
    active_facet_count: int
    active_facet_length: float
    mapped_force_x: float
    mapped_force_y: float
    assembled_external_force_x: float
    assembled_external_force_y: float
    fixed_reaction_x: float
    fixed_reaction_y: float
    balance_residual_x: float
    balance_residual_y: float
    balance_residual_norm: float
    maximum_displacement: float
    maximum_displacement_node: int
    maximum_displacement_coordinates: tuple[float, float]
    displacements: NDArray[np.float64] = field(repr=False)
    fixed_node_indices: NDArray[np.int32] = field(repr=False)
    external_load_vector: NDArray[np.float64] = field(repr=False)
    full_residual_vector: NDArray[np.float64] = field(repr=False)

    @property
    def full_reaction_vector(self) -> NDArray[np.float64]:
        """Return the full residual under the historical compatibility name."""

        return self.full_residual_vector

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready variable-facet-load solver summary."""

        return {
            "result_format": "grindcae_phase_4c3b_facet_load_solver",
            "physical_assumptions": {
                "dimension": "2D",
                "kinematics": "small_deformation",
                "material": "isotropic_linear_elastic",
                "analysis_type": self.analysis_type,
                "element": "first_order_triangle_vector_P1",
                "fixed_boundary": "bottom, zero x and y displacement",
                "loaded_boundary": "runtime contact_arc facets only",
                "load_distribution": "per-facet global x/y line traction",
                "contact_model": (
                    "prescribed empirical boundary traction; no true contact "
                    "iteration, friction law, or Hertz pressure"
                ),
            },
            "units": {
                "length": "m",
                "thickness": "m",
                "force": "N",
                "displacement": "m",
            },
            "mesh": {
                "node_count": self.node_count,
                "triangle_count": self.triangle_count,
                "total_degrees_of_freedom": self.total_degrees_of_freedom,
                "active_facet_count": self.active_facet_count,
                "active_facet_length": self.active_facet_length,
            },
            "analysis": {
                "type": self.analysis_type,
                "thickness": self.thickness,
            },
            "mapped_signed_external_force": {
                "Fx": self.mapped_force_x,
                "Fy": self.mapped_force_y,
            },
            "assembled_external_force": {
                "Fx": self.assembled_external_force_x,
                "Fy": self.assembled_external_force_y,
            },
            "fixed_boundary_reaction": {
                "Rx": self.fixed_reaction_x,
                "Ry": self.fixed_reaction_y,
            },
            "balance_residual": {
                "x": self.balance_residual_x,
                "y": self.balance_residual_y,
                "norm": self.balance_residual_norm,
            },
            "maximum_displacement": {
                "magnitude": self.maximum_displacement,
                "node_index": int(self.maximum_displacement_node),
                "node_index_base": 0,
                "coordinates": list(self.maximum_displacement_coordinates),
            },
        }


@dataclass(frozen=True, slots=True)
class FullFieldSolution:
    """Internal full-system solution used by recovery and file exporters."""

    case: SimulationCase
    imported_mesh: ImportedMesh
    basis: Basis
    stiffness_matrix: spmatrix
    load_vector: NDArray[np.float64]
    fixed_dofs: NDArray[np.int32]
    displacement_vector: NDArray[np.float64]
    nodal_displacements: NDArray[np.float64]
    solver_summary: SolveResult

    def __post_init__(self) -> None:
        if not isinstance(self.case, SimulationCase):
            raise SolverError(
                "full-field solution case provenance must be a SimulationCase"
            )
        self.validate_displacement_consistency()

    def validate_displacement_consistency(self) -> None:
        """Require nodal values to exactly match the basis DOF mapping."""

        _validate_full_field_displacements(
            self.imported_mesh,
            self.basis,
            self.displacement_vector,
            self.nodal_displacements,
        )

    @property
    def mesh(self) -> MeshTri:
        """Return the actual imported finite element mesh."""

        return self.imported_mesh.mesh


@dataclass(frozen=True, slots=True)
class LocalLoadFullFieldSolution:
    """Full local-load solution kept separate from its serializable summary."""

    case: object
    imported_mesh: ImportedMesh
    basis: Basis
    stiffness_matrix: spmatrix
    load_vector: NDArray[np.float64]
    fixed_dofs: NDArray[np.int32]
    active_facets: NDArray[np.int32]
    displacement_vector: NDArray[np.float64]
    nodal_displacements: NDArray[np.float64]
    solver_summary: LocalLoadSolveResult

    def __post_init__(self) -> None:
        _validate_plane_stress_case(self.case)
        self.validate_displacement_consistency()

    def validate_displacement_consistency(self) -> None:
        """Require nodal values to exactly match the basis DOF mapping."""

        _validate_full_field_displacements(
            self.imported_mesh,
            self.basis,
            self.displacement_vector,
            self.nodal_displacements,
        )

    @property
    def mesh(self) -> MeshTri:
        """Return the actual imported finite element mesh."""

        return self.imported_mesh.mesh


@dataclass(frozen=True, slots=True)
class FacetLoadFullFieldSolution:
    """Full solution for independently specified global facet tractions."""

    case: object
    imported_mesh: ImportedMesh
    basis: Basis
    stiffness_matrix: spmatrix
    load_vector: NDArray[np.float64]
    fixed_dofs: NDArray[np.int32]
    active_facets: NDArray[np.int32]
    displacement_vector: NDArray[np.float64]
    nodal_displacements: NDArray[np.float64]
    solver_summary: FacetLoadSolveResult

    def __post_init__(self) -> None:
        _validate_plane_stress_case(self.case)
        self.validate_displacement_consistency()

    def validate_displacement_consistency(self) -> None:
        """Require nodal values to exactly match the basis DOF mapping."""

        _validate_full_field_displacements(
            self.imported_mesh,
            self.basis,
            self.displacement_vector,
            self.nodal_displacements,
        )

    @property
    def mesh(self) -> MeshTri:
        """Return the actual imported finite element mesh."""

        return self.imported_mesh.mesh


def _validate_full_field_displacements(
    imported_mesh: ImportedMesh,
    basis: Basis,
    displacement_vector: NDArray[np.float64],
    nodal_displacements: NDArray[np.float64],
) -> None:
    vector = np.asarray(displacement_vector)
    nodal = np.asarray(nodal_displacements)
    component_dofs = np.asarray(basis.nodal_dofs)
    expected_node_count = imported_mesh.node_count

    if vector.shape != (basis.N,):
        raise SolverError(
            "displacement vector shape does not match the finite element basis"
        )
    if nodal.shape != (expected_node_count, 2):
        raise SolverError("nodal displacement shape does not match the imported mesh")
    if component_dofs.shape != (2, expected_node_count):
        raise SolverError("basis nodal DOF mapping does not match the imported mesh")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(nodal)):
        raise SolverError("full-field displacement data must be finite")
    if not np.array_equal(nodal[:, 0], vector[component_dofs[0]]) or not np.array_equal(
        nodal[:, 1], vector[component_dofs[1]]
    ):
        raise SolverError(
            "nodal displacements do not match the displacement vector through "
            "the basis DOF mapping"
        )


def _validate_meshio_data(mesh_data: meshio.Mesh) -> None:
    if mesh_data.points.ndim != 2 or mesh_data.points.shape[1] < 2:
        raise SolverError("the Gmsh mesh must contain two-dimensional coordinates")
    if mesh_data.points.shape[1] > 2 and not np.allclose(
        mesh_data.points[:, 2:], 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise SolverError("the Gmsh mesh must lie in the two-dimensional xy plane")

    triangle_count = 0
    for cell_block in mesh_data.cells:
        if cell_block.type == "triangle":
            if cell_block.data.ndim != 2 or cell_block.data.shape[1] != 3:
                raise SolverError("the domain must use first-order triangles")
            triangle_count += len(cell_block.data)
        elif cell_block.type == "line":
            if cell_block.data.ndim != 2 or cell_block.data.shape[1] != 2:
                raise SolverError("the boundaries must use first-order line elements")
        else:
            raise SolverError(
                f"unsupported Gmsh cell type {cell_block.type!r}; "
                "expected first-order lines and triangles"
            )
    if triangle_count == 0:
        raise SolverError("the Gmsh mesh does not contain any triangle elements")

    expected_dimensions = {
        "domain": 2,
        "fixed": 1,
        "contact": 1,
        "free_left": 1,
        "free_right": 1,
    }
    missing = sorted(set(REQUIRED_PHYSICAL_GROUPS) - set(mesh_data.field_data))
    if missing:
        raise SolverError(f"the Gmsh mesh is missing physical groups: {missing}")
    for name, expected_dimension in expected_dimensions.items():
        field = np.asarray(mesh_data.field_data[name]).reshape(-1)
        if len(field) < 2 or int(field[1]) != expected_dimension:
            raise SolverError(
                f"physical group {name!r} must have dimension {expected_dimension}"
            )


def _coordinate_tolerance(extent: float) -> float:
    return max(extent * 1.0e-8, 32.0 * math.ulp(extent))


def _validate_named_boundary_facets(
    mesh: MeshTri,
) -> dict[str, NDArray[np.int32]]:
    names = REQUIRED_PHYSICAL_GROUPS[1:]
    validated: dict[str, NDArray[np.int32]] = {}
    for name in names:
        values = np.asarray(mesh.boundaries[name])
        if values.ndim != 1:
            raise SolverError(
                f"physical boundary {name!r} must contain a one-dimensional "
                "list of mesh facets"
            )
        if values.size == 0:
            raise SolverError(f"physical boundary {name!r} must not be empty")
        if not np.issubdtype(values.dtype, np.integer):
            raise SolverError(
                f"physical boundary {name!r} must contain integer mesh facet IDs"
            )
        facets = np.asarray(values, dtype=np.int64)
        if np.any(facets < 0) or np.any(facets >= mesh.nfacets):
            raise SolverError(
                f"physical boundary {name!r} contains a mesh facet ID outside "
                "the valid range"
            )
        if np.unique(facets).size != facets.size:
            raise SolverError(
                f"physical boundary {name!r} contains duplicate mesh facets"
            )
        validated[name] = facets.astype(np.int32, copy=False)

    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if np.intersect1d(
                validated[left_name], validated[right_name]
            ).size:
                raise SolverError(
                    f"physical boundaries {left_name!r} and {right_name!r} "
                    "overlap; rectangle boundary groups must be mutually exclusive"
                )

    named_facets = np.concatenate(tuple(validated.values()))
    actual_boundary_facets = np.asarray(
        mesh.boundary_facets(), dtype=np.int32
    ).reshape(-1)
    missing = np.setdiff1d(actual_boundary_facets, named_facets)
    unexpected = np.setdiff1d(named_facets, actual_boundary_facets)
    if missing.size or unexpected.size:
        raise SolverError(
            "named rectangle boundary groups do not exactly cover the mesh "
            f"boundary: {missing.size} missing facet(s), "
            f"{unexpected.size} unexpected non-boundary facet(s)"
        )
    return validated


def _validate_mesh_bounding_box(mesh: MeshTri, case: _PlaneStressCase) -> None:
    coordinates = np.asarray(mesh.p, dtype=float)
    if coordinates.shape != (2, mesh.nvertices):
        raise SolverError("the imported mesh must contain two-dimensional nodes")
    if not np.all(np.isfinite(coordinates)):
        raise SolverError("the imported mesh contains a non-finite node coordinate")

    width = case.geometry.width
    height = case.geometry.height
    minimum = np.min(coordinates, axis=1)
    maximum = np.max(coordinates, axis=1)
    if not (
        math.isclose(minimum[0], 0.0, rel_tol=0.0, abs_tol=_coordinate_tolerance(width))
        and math.isclose(
            maximum[0], width, rel_tol=0.0, abs_tol=_coordinate_tolerance(width)
        )
        and math.isclose(
            minimum[1], 0.0, rel_tol=0.0, abs_tol=_coordinate_tolerance(height)
        )
        and math.isclose(
            maximum[1], height, rel_tol=0.0, abs_tol=_coordinate_tolerance(height)
        )
    ):
        raise SolverError(
            "imported mesh coordinate bounding box does not match the simulation "
            "geometry rectangle from 0..width and 0..height"
        )


def _validate_boundary_positions(
    mesh: MeshTri,
    case: _PlaneStressCase,
    boundary_facets: dict[str, NDArray[np.int32]],
) -> None:
    width = case.geometry.width
    height = case.geometry.height
    expected = {
        "fixed": (1, 0.0),
        "contact": (1, height),
        "free_left": (0, 0.0),
        "free_right": (0, width),
    }
    for name, (coordinate_index, expected_value) in expected.items():
        extent = width if coordinate_index == 0 else height
        tolerance = _coordinate_tolerance(extent)
        facets = boundary_facets[name]
        coordinates = mesh.p[:, mesh.facets[:, facets]]
        if not np.allclose(
            coordinates[coordinate_index],
            expected_value,
            rtol=0.0,
            atol=tolerance,
        ):
            raise SolverError(
                f"physical boundary {name!r} is not at its expected position"
            )


def _triangle_areas(mesh: MeshTri) -> NDArray[np.float64]:
    points = mesh.p[:, mesh.t]
    twice_signed_area = (
        (points[0, 1] - points[0, 0]) * (points[1, 2] - points[1, 0])
        - (points[0, 2] - points[0, 0]) * (points[1, 1] - points[1, 0])
    )
    return 0.5 * np.abs(twice_signed_area)


def _boundary_length(
    mesh: MeshTri,
    facets: NDArray[np.int32],
) -> float:
    endpoints = mesh.p[:, mesh.facets[:, facets]]
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=0)
    return float(math.fsum(float(value) for value in lengths))


def _validate_plane_stress_case(case: object) -> _PlaneStressCase:
    """Validate the common structural model needed by either solve contract."""

    if not isinstance(getattr(case, "geometry", None), Rectangle):
        raise TypeError("case.geometry must be a Rectangle")
    if not isinstance(getattr(case, "material", None), LinearElasticMaterial):
        raise TypeError("case.material must be LinearElasticMaterial")
    if not isinstance(getattr(case, "analysis", None), AnalysisSettings):
        raise TypeError("case.analysis must be AnalysisSettings")
    return case  # type: ignore[return-value]


def _validate_mesh_case(case: object) -> _PlaneStressCase:
    """Validate geometry metadata used only by shared mesh import checks."""

    if not isinstance(getattr(case, "geometry", None), Rectangle):
        raise TypeError("case.geometry must be a Rectangle")
    material = getattr(case, "material", None)
    if not (
        isinstance(getattr(material, "E", None), (int, float))
        and isinstance(getattr(material, "nu", None), (int, float))
    ):
        raise TypeError("case.material must provide numeric E and nu")
    analysis = getattr(case, "analysis", None)
    if getattr(analysis, "type", None) not in {"plane_stress", "plane_strain"}:
        raise TypeError("case.analysis.type must be plane_stress or plane_strain")
    if not isinstance(getattr(analysis, "thickness", None), (int, float)):
        raise TypeError("case.analysis.thickness must be numeric")
    return case  # type: ignore[return-value]


def _finite_force_value(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SolverError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise SolverError(f"{field_name} must be a finite real number")
    return result


def _validate_active_facets(
    imported_mesh: ImportedMesh,
    active_facets: object,
) -> tuple[NDArray[np.int32], float]:
    values = np.asarray(active_facets)
    if values.ndim != 1:
        raise SolverError("active_facets must be a one-dimensional facet ID array")
    if values.size < MINIMUM_LOCAL_ACTIVE_FACET_COUNT:
        raise SolverError(
            "active_facets must contain at least "
            f"{MINIMUM_LOCAL_ACTIVE_FACET_COUNT} contact facets"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise SolverError("active_facets must contain integer mesh facet IDs")

    facets64 = np.asarray(values, dtype=np.int64)
    mesh = imported_mesh.mesh
    if np.any(facets64 < 0) or np.any(facets64 >= mesh.nfacets):
        raise SolverError("active_facets contains a facet ID outside the valid range")
    if np.unique(facets64).size != facets64.size:
        raise SolverError("active_facets contains duplicate mesh facets")

    contact_facets = np.asarray(mesh.boundaries["contact"], dtype=np.int64)
    if np.setdiff1d(facets64, contact_facets).size:
        raise SolverError(
            "active_facets must be a subset of the named contact boundary"
        )
    if facets64.size >= contact_facets.size:
        raise SolverError(
            "active_facets must be a strict subset of the complete contact boundary"
        )

    facets = facets64.astype(np.int32, copy=False)
    length = _boundary_length(mesh, facets)
    if not math.isfinite(length) or length <= 0.0:
        raise SolverError("active_facets must have a finite positive total length")
    return facets, length


def _validate_facet_tractions(
    imported_mesh: ImportedMesh,
    active_facets: object,
    facet_line_load_x_N_per_m: object,
    facet_line_load_y_N_per_m: object,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    values = np.asarray(active_facets)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise SolverError("active_facets must be a one-dimensional integer array")
    facets64 = np.asarray(values, dtype=np.int64)
    mesh = imported_mesh.mesh
    if np.any(facets64 < 0) or np.any(facets64 >= mesh.nfacets):
        raise SolverError("active_facets contains a facet ID outside the valid range")
    if np.unique(facets64).size != facets64.size:
        raise SolverError("active_facets contains duplicate mesh facets")
    contact_facets = np.asarray(mesh.boundaries["contact"], dtype=np.int64)
    if np.setdiff1d(facets64, contact_facets).size:
        raise SolverError("active_facets must be a subset of the contact boundary")

    line_x = np.asarray(facet_line_load_x_N_per_m, dtype=float)
    line_y = np.asarray(facet_line_load_y_N_per_m, dtype=float)
    if line_x.ndim != 1 or line_y.ndim != 1:
        raise SolverError("facet line loads must be one-dimensional arrays")
    if line_x.shape != facets64.shape or line_y.shape != facets64.shape:
        raise SolverError("each active facet must have one x and one y line load")
    if not np.all(np.isfinite(line_x)) or not np.all(np.isfinite(line_y)):
        raise SolverError("facet line loads must be finite")
    if np.any(line_y > 0.0):
        raise SolverError("facet y line loads must be compressive or zero")

    facets = facets64.astype(np.int32, copy=False)
    if facets.size:
        endpoints = mesh.p[:, mesh.facets[:, facets]]
        lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=0)
    else:
        lengths = np.empty(0, dtype=float)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise SolverError("active facet lengths must be finite and positive")
    return facets, np.asarray(lengths, dtype=float), line_x, line_y


def _make_basis_and_stiffness(
    case: _PlaneStressCase,
    imported_mesh: ImportedMesh,
    maximum_degrees_of_freedom: int,
) -> tuple[ElementVector, Basis, spmatrix]:
    if type(maximum_degrees_of_freedom) is not int or maximum_degrees_of_freedom <= 0:
        raise SolverError("maximum_degrees_of_freedom must be a positive integer")

    element = ElementVector(ElementTriP1())
    basis = Basis(imported_mesh.mesh, element)
    if basis.N > maximum_degrees_of_freedom:
        raise SolverError(
            f"solve requires {basis.N} degrees of freedom, exceeding the "
            f"configured safety limit of {maximum_degrees_of_freedom}; "
            "increase mesh.target_size"
        )

    elastic_modulus = case.material.E
    poisson_ratio = case.material.nu
    mu = elastic_modulus / (2.0 * (1.0 + poisson_ratio))
    lambda_plane_stress = (
        elastic_modulus * poisson_ratio / (1.0 - poisson_ratio**2)
    )
    stiffness_matrix = case.analysis.thickness * asm(
        linear_elasticity(Lambda=lambda_plane_stress, Mu=mu), basis
    )
    return element, basis, stiffness_matrix


def _fixed_degree_sets(
    basis: Basis,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    fixed = basis.get_dofs("fixed")
    return (
        np.asarray(fixed.all(), dtype=np.int32),
        np.asarray(fixed.nodal["u^1"], dtype=np.int32),
        np.asarray(fixed.nodal["u^2"], dtype=np.int32),
    )


def _assembled_force_components(
    basis: Basis,
    load_vector: NDArray[np.float64],
) -> tuple[float, float]:
    component_dofs = basis.nodal_dofs
    return (
        float(math.fsum(load_vector[component_dofs[0]])),
        float(math.fsum(load_vector[component_dofs[1]])),
    )


def _require_assembled_force(
    assembled_x: float,
    assembled_y: float,
    expected_x: float,
    expected_y: float,
    source_description: str,
) -> None:
    if not np.allclose(
        np.array([assembled_x, assembled_y], dtype=float),
        np.array([expected_x, expected_y], dtype=float),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise SolverError(
            "assembled external force does not match " + source_description
        )


def _validate_mesh_coverage(
    mesh: MeshTri,
    case: _PlaneStressCase,
    boundary_facets: dict[str, NDArray[np.int32]],
) -> None:
    domain = np.asarray(mesh.subdomains["domain"], dtype=np.int64).reshape(-1)
    expected_domain = np.arange(mesh.nelements, dtype=np.int64)
    if not np.array_equal(np.sort(domain), expected_domain):
        raise SolverError("physical group 'domain' must cover every triangle once")

    width = case.geometry.width
    expected_area = width * case.geometry.height
    areas = _triangle_areas(mesh)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise SolverError("the imported mesh contains a non-positive triangle area")
    total_area = float(math.fsum(float(value) for value in areas))
    if not math.isclose(total_area, expected_area, rel_tol=1.0e-8, abs_tol=0.0):
        raise SolverError(
            "imported mesh area does not match geometry.width * geometry.height"
        )

    expected_lengths = {
        "fixed": width,
        "contact": width,
        "free_left": case.geometry.height,
        "free_right": case.geometry.height,
    }
    for name, expected_length in expected_lengths.items():
        length = _boundary_length(mesh, boundary_facets[name])
        if not math.isclose(
            length,
            expected_length,
            rel_tol=1.0e-8,
            abs_tol=32.0 * math.ulp(expected_length),
        ):
            dimension = (
                "geometry.width"
                if name in ("fixed", "contact")
                else "geometry.height"
            )
            raise SolverError(
                f"physical boundary {name!r} length does not match {dimension}"
            )


def import_gmsh_mesh(
    mesh_path: str | Path,
    case: _PlaneStressCase,
) -> ImportedMesh:
    """Read a real Gmsh file through meshio and convert it to scikit-fem."""

    solve_case = _validate_mesh_case(case)
    source_path = Path(mesh_path).expanduser().resolve()
    if source_path.suffix.lower() != ".msh":
        raise SolverError("mesh_path must use the .msh extension")
    try:
        mesh_data = meshio.read(source_path)
    except Exception as exc:
        raise SolverError(f"could not read Gmsh mesh {source_path}") from exc

    _validate_meshio_data(mesh_data)
    try:
        # 中文导读：求解器复用刚读入的真实 MSH，不另造一套计算网格。
        mesh = from_meshio(mesh_data)
    except Exception as exc:
        raise SolverError(
            "meshio could not convert the Gmsh mesh to a scikit-fem mesh"
        ) from exc
    if not isinstance(mesh, MeshTri):
        raise SolverError("the imported finite element mesh must be triangular")

    boundaries = mesh.boundaries or {}
    subdomains = mesh.subdomains or {}
    missing_boundaries = sorted(
        set(REQUIRED_PHYSICAL_GROUPS[1:]) - set(boundaries)
    )
    if missing_boundaries:
        raise SolverError(
            f"the imported mesh is missing named boundaries: {missing_boundaries}"
        )
    if "domain" not in subdomains or len(subdomains["domain"]) == 0:
        raise SolverError("the imported mesh is missing the non-empty domain group")
    if mesh.nvertices == 0 or mesh.nelements == 0:
        raise SolverError("the imported finite element mesh is empty")

    boundary_facets = _validate_named_boundary_facets(mesh)
    _validate_mesh_bounding_box(mesh, solve_case)
    _validate_boundary_positions(mesh, solve_case, boundary_facets)
    _validate_mesh_coverage(mesh, solve_case, boundary_facets)
    return ImportedMesh(mesh=mesh, source_path=source_path)


def assemble_plane_stress_system(
    case: SimulationCase,
    imported_mesh: ImportedMesh,
    *,
    maximum_degrees_of_freedom: int = DEFAULT_MAX_DEGREES_OF_FREEDOM,
) -> AssembledSystem:
    """Assemble full K and f for vector P1 plane stress on the imported mesh."""

    if not isinstance(case, SimulationCase):
        raise TypeError("case must be a SimulationCase")
    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")

    # Multiplying the assembled 2D weak form by thickness supplies the
    # out-of-plane measure required by a plane-stress stiffness matrix.
    element, basis, stiffness_matrix = _make_basis_and_stiffness(
        case, imported_mesh, maximum_degrees_of_freedom
    )

    totals = case.load_totals()
    line_load_x = totals.tangential_force / case.geometry.width
    line_load_y = -totals.normal_force / case.geometry.width

    @LinearForm
    def uniform_contact_load(test_function: NDArray[np.float64], _) -> float:
        # q is already a line load in N/m; do not multiply it by thickness.
        return line_load_x * test_function[0] + line_load_y * test_function[1]

    contact_basis = FacetBasis(
        imported_mesh.mesh,
        element,
        facets=imported_mesh.mesh.boundaries["contact"],
    )
    load_vector = np.asarray(asm(uniform_contact_load, contact_basis))

    fixed_dofs, fixed_x_dofs, fixed_y_dofs = _fixed_degree_sets(basis)
    assembled_x, assembled_y = _assembled_force_components(basis, load_vector)
    _require_assembled_force(
        assembled_x,
        assembled_y,
        totals.tangential_force,
        -totals.normal_force,
        "the supplied equivalent loads",
    )

    return AssembledSystem(
        case=case,
        imported_mesh=imported_mesh,
        basis=basis,
        stiffness_matrix=stiffness_matrix,
        load_vector=load_vector,
        fixed_dofs=fixed_dofs,
        fixed_x_dofs=fixed_x_dofs,
        fixed_y_dofs=fixed_y_dofs,
        input_normal_force=totals.normal_force,
        input_tangential_force=totals.tangential_force,
        assembled_external_force_x=assembled_x,
        assembled_external_force_y=assembled_y,
    )


def assemble_local_plane_stress_system(
    case: _PlaneStressCase,
    imported_mesh: ImportedMesh,
    active_facets: object,
    *,
    signed_force_x_N: object,
    signed_force_y_N: object,
    line_load_x_N_per_m: object,
    line_load_y_N_per_m: object,
    maximum_degrees_of_freedom: int = DEFAULT_MAX_DEGREES_OF_FREEDOM,
) -> LocalLoadAssembledSystem:
    """Assemble a uniform directional line load on exact active contact facets."""

    solve_case = _validate_plane_stress_case(case)
    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")

    mapped_x = _finite_force_value(signed_force_x_N, "signed_force_x_N")
    mapped_y = _finite_force_value(signed_force_y_N, "signed_force_y_N")
    line_load_x = _finite_force_value(
        line_load_x_N_per_m, "line_load_x_N_per_m"
    )
    line_load_y = _finite_force_value(
        line_load_y_N_per_m, "line_load_y_N_per_m"
    )
    if mapped_y > 0.0 or line_load_y > 0.0:
        raise SolverError(
            "signed_force_y_N and line_load_y_N_per_m must be non-positive "
            "for a compressive load into the top boundary"
        )

    facets, active_length = _validate_active_facets(imported_mesh, active_facets)
    integrated_force = np.array(
        [line_load_x * active_length, line_load_y * active_length], dtype=float
    )
    mapped_force = np.array([mapped_x, mapped_y], dtype=float)
    if not np.all(np.isfinite(integrated_force)) or not np.allclose(
        integrated_force,
        mapped_force,
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise SolverError(
            "local line load times the actual active facet length does not "
            "match the mapped signed force"
        )

    element, basis, stiffness_matrix = _make_basis_and_stiffness(
        solve_case, imported_mesh, maximum_degrees_of_freedom
    )

    @LinearForm
    def uniform_local_contact_load(
        test_function: NDArray[np.float64], _
    ) -> float:
        # q is already a line load in N/m; thickness only belongs in stiffness.
        return (
            line_load_x * test_function[0]
            + line_load_y * test_function[1]
        )

    active_basis = FacetBasis(imported_mesh.mesh, element, facets=facets)
    load_vector = np.asarray(asm(uniform_local_contact_load, active_basis))
    if not np.all(np.isfinite(load_vector)):
        raise SolverError("local boundary-load assembly produced a non-finite value")

    fixed_dofs, fixed_x_dofs, fixed_y_dofs = _fixed_degree_sets(basis)
    assembled_x, assembled_y = _assembled_force_components(basis, load_vector)
    _require_assembled_force(
        assembled_x,
        assembled_y,
        mapped_x,
        mapped_y,
        "the mapped signed grinding force",
    )
    return LocalLoadAssembledSystem(
        case=case,
        imported_mesh=imported_mesh,
        basis=basis,
        stiffness_matrix=stiffness_matrix,
        load_vector=load_vector,
        fixed_dofs=fixed_dofs,
        fixed_x_dofs=fixed_x_dofs,
        fixed_y_dofs=fixed_y_dofs,
        active_facets=facets,
        active_facet_length=active_length,
        mapped_force_x=mapped_x,
        mapped_force_y=mapped_y,
        line_load_x=line_load_x,
        line_load_y=line_load_y,
        assembled_external_force_x=assembled_x,
        assembled_external_force_y=assembled_y,
    )


def assemble_facet_plane_stress_system(
    case: _PlaneStressCase,
    imported_mesh: ImportedMesh,
    active_facets: object,
    *,
    mapped_force_x_N: object,
    mapped_force_y_N: object,
    facet_line_load_x_N_per_m: object,
    facet_line_load_y_N_per_m: object,
    maximum_degrees_of_freedom: int = DEFAULT_MAX_DEGREES_OF_FREEDOM,
) -> FacetLoadAssembledSystem:
    """Assemble global x/y line tractions that may vary by boundary facet."""

    solve_case = _validate_plane_stress_case(case)
    if not isinstance(imported_mesh, ImportedMesh):
        raise TypeError("imported_mesh must be an ImportedMesh")
    mapped_x = _finite_force_value(mapped_force_x_N, "mapped_force_x_N")
    mapped_y = _finite_force_value(mapped_force_y_N, "mapped_force_y_N")
    if mapped_y > 0.0:
        raise SolverError("mapped_force_y_N must be compressive or zero")
    facets, lengths, line_x, line_y = _validate_facet_tractions(
        imported_mesh,
        active_facets,
        facet_line_load_x_N_per_m,
        facet_line_load_y_N_per_m,
    )
    integrated_x = float(math.fsum(float(value) for value in line_x * lengths))
    integrated_y = float(math.fsum(float(value) for value in line_y * lengths))
    # 中文导读：先核对所有活动边的线载荷积分合力与映射 Fx/Fy 一致，再组装载荷向量。
    _require_assembled_force(
        integrated_x,
        integrated_y,
        mapped_x,
        mapped_y,
        "the mapped signed force from the per-facet traction integrals",
    )

    element, basis, stiffness_matrix = _make_basis_and_stiffness(
        solve_case, imported_mesh, maximum_degrees_of_freedom
    )
    load_vector = np.zeros(basis.N, dtype=float)
    for facet_id, traction_x, traction_y in zip(facets, line_x, line_y):

        @LinearForm
        def facet_contact_load(
            test_function: NDArray[np.float64], _
        ) -> float:
            # The traction is already N/m along ds; thickness belongs in K only.
            return (
                traction_x * test_function[0]
                + traction_y * test_function[1]
            )

        facet_basis = FacetBasis(
            imported_mesh.mesh,
            element,
            facets=np.asarray([facet_id], dtype=np.int32),
        )
        load_vector += np.asarray(asm(facet_contact_load, facet_basis))
    if not np.all(np.isfinite(load_vector)):
        raise SolverError("facet boundary-load assembly produced a non-finite value")

    fixed_dofs, fixed_x_dofs, fixed_y_dofs = _fixed_degree_sets(basis)
    assembled_x, assembled_y = _assembled_force_components(basis, load_vector)
    _require_assembled_force(
        assembled_x,
        assembled_y,
        mapped_x,
        mapped_y,
        "the mapped signed facet load",
    )
    return FacetLoadAssembledSystem(
        case=case,
        imported_mesh=imported_mesh,
        basis=basis,
        stiffness_matrix=stiffness_matrix,
        load_vector=load_vector,
        fixed_dofs=fixed_dofs,
        fixed_x_dofs=fixed_x_dofs,
        fixed_y_dofs=fixed_y_dofs,
        active_facets=facets,
        active_facet_lengths=lengths,
        facet_line_load_x=line_x,
        facet_line_load_y=line_y,
        mapped_force_x=mapped_x,
        mapped_force_y=mapped_y,
        assembled_external_force_x=assembled_x,
        assembled_external_force_y=assembled_y,
    )


@dataclass(frozen=True, slots=True)
class _SolvedFieldState:
    displacement_vector: NDArray[np.float64]
    nodal_displacements: NDArray[np.float64]
    full_residual: NDArray[np.float64]
    fixed_node_indices: NDArray[np.int32]
    reaction_x: float
    reaction_y: float
    balance_x: float
    balance_y: float
    maximum_node: int
    maximum_displacement: float
    maximum_coordinates: tuple[float, float]


def _solve_system_state(
    imported_mesh: ImportedMesh,
    basis: Basis,
    stiffness_matrix: spmatrix,
    load_vector: NDArray[np.float64],
    fixed_dofs: NDArray[np.int32],
    fixed_x_dofs: NDArray[np.int32],
    fixed_y_dofs: NDArray[np.int32],
    assembled_external_force_x: float,
    assembled_external_force_y: float,
) -> _SolvedFieldState:
    condensed = condense(stiffness_matrix, load_vector, D=fixed_dofs)
    displacement_vector = np.asarray(solve(*condensed), dtype=float)
    if not np.all(np.isfinite(displacement_vector)):
        raise SolverError("the linear solve produced a non-finite displacement")

    # Reactions require the original full system, not the condensed equations.
    # 中文导读：支反力来自完整方程残差 K*u-f，并只汇总固定自由度。
    full_residual = np.asarray(
        stiffness_matrix @ displacement_vector - load_vector
    ).reshape(-1)
    reaction_x = float(math.fsum(full_residual[fixed_x_dofs]))
    reaction_y = float(math.fsum(full_residual[fixed_y_dofs]))
    balance_x = assembled_external_force_x + reaction_x
    balance_y = assembled_external_force_y + reaction_y

    component_dofs = basis.nodal_dofs
    displacements = np.column_stack(
        (
            displacement_vector[component_dofs[0]],
            displacement_vector[component_dofs[1]],
        )
    )
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    maximum_node = int(np.argmax(displacement_magnitudes))
    maximum_coordinates = tuple(
        float(value) for value in imported_mesh.mesh.p[:, maximum_node]
    )
    fixed_node_indices = np.unique(
        imported_mesh.mesh.facets[:, imported_mesh.mesh.boundaries["fixed"]]
    ).astype(np.int32)
    return _SolvedFieldState(
        displacement_vector=displacement_vector,
        nodal_displacements=displacements,
        full_residual=full_residual,
        fixed_node_indices=fixed_node_indices,
        reaction_x=reaction_x,
        reaction_y=reaction_y,
        balance_x=balance_x,
        balance_y=balance_y,
        maximum_node=maximum_node,
        maximum_displacement=float(displacement_magnitudes[maximum_node]),
        maximum_coordinates=maximum_coordinates,
    )


def solve_full_field(
    case: SimulationCase,
    system: AssembledSystem,
) -> FullFieldSolution:
    """Solve the full system and retain the mesh, basis, K, f, and displacement."""

    if not isinstance(case, SimulationCase):
        raise TypeError("case must be a SimulationCase")
    if not isinstance(system, AssembledSystem):
        raise TypeError("system must be an AssembledSystem")
    if case != system.case:
        raise SolverError(
            "simulation case does not match the case used to create the "
            "existing assembled system; assemble the requested case again"
        )
    solve_case = system.case
    state = _solve_system_state(
        system.imported_mesh,
        system.basis,
        system.stiffness_matrix,
        system.load_vector,
        system.fixed_dofs,
        system.fixed_x_dofs,
        system.fixed_y_dofs,
        system.assembled_external_force_x,
        system.assembled_external_force_y,
    )

    summary = SolveResult(
        node_count=system.imported_mesh.node_count,
        triangle_count=system.imported_mesh.triangle_count,
        total_degrees_of_freedom=int(system.basis.N),
        analysis_type=solve_case.analysis.type,
        thickness=solve_case.analysis.thickness,
        input_normal_force=system.input_normal_force,
        input_tangential_force=system.input_tangential_force,
        assembled_external_force_x=system.assembled_external_force_x,
        assembled_external_force_y=system.assembled_external_force_y,
        fixed_reaction_x=state.reaction_x,
        fixed_reaction_y=state.reaction_y,
        balance_residual_x=state.balance_x,
        balance_residual_y=state.balance_y,
        balance_residual_norm=math.hypot(state.balance_x, state.balance_y),
        maximum_displacement=state.maximum_displacement,
        maximum_displacement_node=state.maximum_node,
        maximum_displacement_coordinates=state.maximum_coordinates,
        displacements=state.nodal_displacements,
        fixed_node_indices=state.fixed_node_indices,
        external_load_vector=system.load_vector,
        full_residual_vector=state.full_residual,
    )
    return FullFieldSolution(
        case=solve_case,
        imported_mesh=system.imported_mesh,
        basis=system.basis,
        stiffness_matrix=system.stiffness_matrix,
        load_vector=system.load_vector,
        fixed_dofs=system.fixed_dofs,
        displacement_vector=state.displacement_vector,
        nodal_displacements=state.nodal_displacements,
        solver_summary=summary,
    )


def solve_assembled_system(
    case: SimulationCase,
    system: AssembledSystem,
) -> SolveResult:
    """Compatibility wrapper returning the phase 2A serializable summary."""

    return solve_full_field(case, system).solver_summary


def solve_local_full_field(
    case: _PlaneStressCase,
    system: LocalLoadAssembledSystem,
) -> LocalLoadFullFieldSolution:
    """Solve a locally loaded full system and retain its complete field state."""

    _validate_plane_stress_case(case)
    if not isinstance(system, LocalLoadAssembledSystem):
        raise TypeError("system must be a LocalLoadAssembledSystem")
    if case != system.case:
        raise SolverError(
            "simulation case does not match the case used to create the "
            "existing local-load assembled system; assemble the requested case again"
        )
    solve_case = _validate_plane_stress_case(system.case)
    state = _solve_system_state(
        system.imported_mesh,
        system.basis,
        system.stiffness_matrix,
        system.load_vector,
        system.fixed_dofs,
        system.fixed_x_dofs,
        system.fixed_y_dofs,
        system.assembled_external_force_x,
        system.assembled_external_force_y,
    )

    summary = LocalLoadSolveResult(
        node_count=system.imported_mesh.node_count,
        triangle_count=system.imported_mesh.triangle_count,
        total_degrees_of_freedom=int(system.basis.N),
        analysis_type=solve_case.analysis.type,
        thickness=solve_case.analysis.thickness,
        active_facet_count=int(system.active_facets.size),
        active_facet_length=system.active_facet_length,
        mapped_force_x=system.mapped_force_x,
        mapped_force_y=system.mapped_force_y,
        line_load_x=system.line_load_x,
        line_load_y=system.line_load_y,
        assembled_external_force_x=system.assembled_external_force_x,
        assembled_external_force_y=system.assembled_external_force_y,
        fixed_reaction_x=state.reaction_x,
        fixed_reaction_y=state.reaction_y,
        balance_residual_x=state.balance_x,
        balance_residual_y=state.balance_y,
        balance_residual_norm=math.hypot(state.balance_x, state.balance_y),
        maximum_displacement=state.maximum_displacement,
        maximum_displacement_node=state.maximum_node,
        maximum_displacement_coordinates=state.maximum_coordinates,
        displacements=state.nodal_displacements,
        fixed_node_indices=state.fixed_node_indices,
        external_load_vector=system.load_vector,
        full_residual_vector=state.full_residual,
    )
    return LocalLoadFullFieldSolution(
        case=system.case,
        imported_mesh=system.imported_mesh,
        basis=system.basis,
        stiffness_matrix=system.stiffness_matrix,
        load_vector=system.load_vector,
        fixed_dofs=system.fixed_dofs,
        active_facets=system.active_facets,
        displacement_vector=state.displacement_vector,
        nodal_displacements=state.nodal_displacements,
        solver_summary=summary,
    )


def solve_local_assembled_system(
    case: _PlaneStressCase,
    system: LocalLoadAssembledSystem,
) -> LocalLoadSolveResult:
    """Return only the serializable local-load solver summary."""

    return solve_local_full_field(case, system).solver_summary


def solve_facet_full_field(
    case: _PlaneStressCase,
    system: FacetLoadAssembledSystem,
) -> FacetLoadFullFieldSolution:
    """Solve an existing per-facet traction system through the shared solver."""

    _validate_plane_stress_case(case)
    if not isinstance(system, FacetLoadAssembledSystem):
        raise TypeError("system must be a FacetLoadAssembledSystem")
    if case != system.case:
        raise SolverError(
            "simulation case does not match the per-facet assembled system"
        )
    solve_case = _validate_plane_stress_case(system.case)
    state = _solve_system_state(
        system.imported_mesh,
        system.basis,
        system.stiffness_matrix,
        system.load_vector,
        system.fixed_dofs,
        system.fixed_x_dofs,
        system.fixed_y_dofs,
        system.assembled_external_force_x,
        system.assembled_external_force_y,
    )
    summary = FacetLoadSolveResult(
        node_count=system.imported_mesh.node_count,
        triangle_count=system.imported_mesh.triangle_count,
        total_degrees_of_freedom=int(system.basis.N),
        analysis_type=solve_case.analysis.type,
        thickness=solve_case.analysis.thickness,
        active_facet_count=int(system.active_facets.size),
        active_facet_length=float(math.fsum(system.active_facet_lengths)),
        mapped_force_x=system.mapped_force_x,
        mapped_force_y=system.mapped_force_y,
        assembled_external_force_x=system.assembled_external_force_x,
        assembled_external_force_y=system.assembled_external_force_y,
        fixed_reaction_x=state.reaction_x,
        fixed_reaction_y=state.reaction_y,
        balance_residual_x=state.balance_x,
        balance_residual_y=state.balance_y,
        balance_residual_norm=math.hypot(state.balance_x, state.balance_y),
        maximum_displacement=state.maximum_displacement,
        maximum_displacement_node=state.maximum_node,
        maximum_displacement_coordinates=state.maximum_coordinates,
        displacements=state.nodal_displacements,
        fixed_node_indices=state.fixed_node_indices,
        external_load_vector=system.load_vector,
        full_residual_vector=state.full_residual,
    )
    return FacetLoadFullFieldSolution(
        case=system.case,
        imported_mesh=system.imported_mesh,
        basis=system.basis,
        stiffness_matrix=system.stiffness_matrix,
        load_vector=system.load_vector,
        fixed_dofs=system.fixed_dofs,
        active_facets=system.active_facets,
        displacement_vector=state.displacement_vector,
        nodal_displacements=state.nodal_displacements,
        solver_summary=summary,
    )


def solve_facet_assembled_system(
    case: _PlaneStressCase,
    system: FacetLoadAssembledSystem,
) -> FacetLoadSolveResult:
    """Return only the serializable per-facet traction solver summary."""

    return solve_facet_full_field(case, system).solver_summary


def solve_case_from_mesh(
    case: SimulationCase,
    mesh_path: str | Path,
    *,
    maximum_degrees_of_freedom: int = DEFAULT_MAX_DEGREES_OF_FREEDOM,
) -> SolveResult:
    """Import a Gmsh mesh, assemble the full system, and solve it."""

    imported_mesh = import_gmsh_mesh(mesh_path, case)
    system = assemble_plane_stress_system(
        case,
        imported_mesh,
        maximum_degrees_of_freedom=maximum_degrees_of_freedom,
    )
    return solve_assembled_system(case, system)


def solve_full_field_from_mesh(
    case: SimulationCase,
    mesh_path: str | Path,
    *,
    maximum_degrees_of_freedom: int = DEFAULT_MAX_DEGREES_OF_FREEDOM,
) -> FullFieldSolution:
    """Import a Gmsh mesh and return the complete field solution."""

    imported_mesh = import_gmsh_mesh(mesh_path, case)
    system = assemble_plane_stress_system(
        case,
        imported_mesh,
        maximum_degrees_of_freedom=maximum_degrees_of_freedom,
    )
    return solve_full_field(case, system)
