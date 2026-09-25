from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from grindcae.moving_load_pass import (
    FixedMeshMovingLoadPassCase,
    MovingLoadPassValidationError,
)
from grindcae.single_position_elastoplastic import SinglePositionElastoplasticCase


PROJECT_ROOT = Path(__file__).parents[2]
REFERENCE = PROJECT_ROOT / "examples" / "host_reference_single_position.json"


def valid_payload() -> dict[str, object]:
    return {
        "moving_load_pass_schema_version": 1,
        "unit_system": "SI",
        "model_type": "fixed_mesh_complete_single_pass_moving_load",
        "reference_case": json.loads(REFERENCE.read_text(encoding="utf-8")),
        "pass": {
            "range": "complete_single_pass",
            "base_position_count": 21,
            "retain_key_position_output": True,
        },
    }


def test_valid_schema_reuses_one_complete_6a3_reference_case_and_roundtrips() -> None:
    payload = valid_payload()

    case = FixedMeshMovingLoadPassCase.from_mapping(copy.deepcopy(payload))

    assert isinstance(case.reference_case, SinglePositionElastoplasticCase)
    assert case.geometry.width == pytest.approx(0.1)
    assert case.mesh.target_size == pytest.approx(0.003)
    assert case.motion_direction == "positive_x"
    assert case.to_dict() == payload
    assert FixedMeshMovingLoadPassCase.from_mapping(case.to_dict()) == case


@pytest.mark.parametrize("value", [0, 2, True, 1.0, "1"])
def test_schema_version_requires_strict_integer_one(value: object) -> None:
    payload = valid_payload()
    payload["moving_load_pass_schema_version"] = value

    with pytest.raises(MovingLoadPassValidationError, match="strict integer 1"):
        FixedMeshMovingLoadPassCase.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unit_system", "mm-N", "SI"),
        ("model_type", "other", "unsupported"),
    ],
)
def test_root_literals_are_strict(field: str, value: object, message: str) -> None:
    payload = valid_payload()
    payload[field] = value

    with pytest.raises(MovingLoadPassValidationError, match=message):
        FixedMeshMovingLoadPassCase.from_mapping(payload)


@pytest.mark.parametrize("change", ["missing", "unexpected"])
@pytest.mark.parametrize("path", [(), ("pass",)])
def test_each_new_object_boundary_has_an_exact_field_set(
    change: str, path: tuple[str, ...]
) -> None:
    payload = valid_payload()
    target = payload if not path else payload[path[0]]
    assert isinstance(target, dict)
    if change == "missing":
        target.pop(next(iter(target)))
    else:
        target["unexpected"] = 1

    with pytest.raises(MovingLoadPassValidationError, match=change):
        FixedMeshMovingLoadPassCase.from_mapping(payload)


@pytest.mark.parametrize("value", [4, 502, True, 21.0, "21"])
def test_base_position_count_has_bounded_strict_integer_contract(value: object) -> None:
    payload = valid_payload()
    payload["pass"]["base_position_count"] = value  # type: ignore[index]

    with pytest.raises(MovingLoadPassValidationError, match="base_position_count"):
        FixedMeshMovingLoadPassCase.from_mapping(payload)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_retain_key_position_output_requires_a_strict_bool(value: object) -> None:
    payload = valid_payload()
    payload["pass"]["retain_key_position_output"] = value  # type: ignore[index]

    with pytest.raises(MovingLoadPassValidationError, match="strict bool"):
        FixedMeshMovingLoadPassCase.from_mapping(payload)


def test_reference_case_keeps_6a3_width_thickness_validation() -> None:
    payload = valid_payload()
    payload["reference_case"]["analysis"]["thickness"] = 0.02  # type: ignore[index]

    with pytest.raises(MovingLoadPassValidationError, match="grinding_width_m"):
        FixedMeshMovingLoadPassCase.from_mapping(payload)
