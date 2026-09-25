from __future__ import annotations

import copy
import math

import pytest

from grindcae.elastoplastic_fem.models import (
    ElastoplasticFemCase,
    ElastoplasticFemValidationError,
)


def valid_payload() -> dict[str, object]:
    return {
        "elastoplastic_fem_schema_version": 1,
        "unit_system": "SI",
        "model_type": "small_strain_plane_strain_j2_fixed_mesh",
        "geometry": {"type": "rectangle", "width": 0.010, "height": 0.004},
        "material": {
            "E": 210_000_000_000.0,
            "nu": 0.3,
            "yield_strength": 250_000_000.0,
            "tangent_modulus": 2_000_000_000.0,
        },
        "mesh": {"target_size": 0.001},
        "analysis": {"type": "plane_strain", "thickness": 0.002},
        "fixed_boundary": {"side": "bottom", "components": ["x", "y"]},
        "load": {
            "boundary": "top",
            "distribution": "uniform_over_interval",
            "x_start_m": 0.003,
            "x_end_m": 0.007,
            "peak_force_x_N": 0.0,
            "peak_force_y_N": -300_000.0,
        },
        "increments": {"loading_step_count": 8, "unloading_step_count": 8},
        "newton": {
            "residual_relative_tolerance": 1.0e-8,
            "residual_absolute_tolerance_N": 1.0e-6,
            "displacement_relative_tolerance": 1.0e-8,
            "displacement_absolute_tolerance_m": 1.0e-14,
            "maximum_iterations": 25,
        },
        "step_control": {
            "allow_reduction": True,
            "minimum_load_factor_increment": 1.0e-4,
            "maximum_retries": 8,
        },
    }


def test_strict_case_accepts_only_plane_strain_and_preserves_total_force_units() -> None:
    case = ElastoplasticFemCase.from_mapping(valid_payload())

    assert case.analysis.type == "plane_strain"
    assert case.analysis.thickness == pytest.approx(0.002)
    assert case.load.peak_force_x_N == 0.0
    assert case.load.peak_force_y_N == -300_000.0
    assert case.load.interval_length_m == pytest.approx(0.004)
    assert case.load.peak_line_load_y_N_per_m == pytest.approx(-75_000_000.0)
    assert case.material.internal_hardening_modulus == pytest.approx(
        210_000_000_000.0 * 2_000_000_000.0
        / (210_000_000_000.0 - 2_000_000_000.0)
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("elastoplastic_fem_schema_version",), True, "strict integer 1"),
        (("elastoplastic_fem_schema_version",), 1.0, "strict integer 1"),
        (("unit_system",), "mm-N", "SI"),
        (("analysis", "type"), "plane_stress", "plane_strain"),
        (("analysis", "thickness"), 0.0, "greater than zero"),
        (("material", "nu"), 0.5, "less than 0.5"),
        (("material", "tangent_modulus"), 210_000_000_000.0, "less than material.E"),
        (("load", "peak_force_y_N"), 1.0, "non-positive"),
        (("increments", "loading_step_count"), 1, "between"),
        (("newton", "maximum_iterations"), 2.0, "strict integer"),
        (("newton", "residual_relative_tolerance"), math.nan, "finite"),
        (("step_control", "minimum_load_factor_increment"), 0.0, "greater than zero"),
    ],
)
def test_schema_rejects_wrong_types_ranges_and_analysis(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = copy.deepcopy(valid_payload())
    target: dict[str, object] = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value

    with pytest.raises(ElastoplasticFemValidationError, match=message):
        ElastoplasticFemCase.from_mapping(payload)


def test_schema_rejects_extra_fields_at_every_object_boundary() -> None:
    for object_path in ((), ("geometry",), ("load",), ("newton",)):
        payload = copy.deepcopy(valid_payload())
        target: dict[str, object] = payload
        for key in object_path:
            target = target[key]  # type: ignore[assignment]
        target["unexpected"] = 1

        with pytest.raises(ElastoplasticFemValidationError, match="unexpected"):
            ElastoplasticFemCase.from_mapping(payload)


@pytest.mark.parametrize(
    ("x_start", "x_end"),
    [(-1.0e-4, 0.005), (0.004, 0.004), (0.008, 0.011)],
)
def test_load_interval_must_be_positive_and_inside_top_boundary(
    x_start: float, x_end: float
) -> None:
    payload = valid_payload()
    payload["load"]["x_start_m"] = x_start  # type: ignore[index]
    payload["load"]["x_end_m"] = x_end  # type: ignore[index]

    with pytest.raises(ElastoplasticFemValidationError, match="load.*interval"):
        ElastoplasticFemCase.from_mapping(payload)


def test_fixed_boundary_contract_is_bottom_with_both_displacement_components() -> None:
    payload = valid_payload()
    payload["fixed_boundary"] = {"side": "left", "components": ["x", "y"]}
    with pytest.raises(ElastoplasticFemValidationError, match="bottom"):
        ElastoplasticFemCase.from_mapping(payload)

    payload = valid_payload()
    payload["fixed_boundary"] = {"side": "bottom", "components": ["y", "x"]}
    with pytest.raises(ElastoplasticFemValidationError, match="components"):
        ElastoplasticFemCase.from_mapping(payload)
