from __future__ import annotations

import numpy as np
import pytest

from grindcae.elastoplastic import (
    BilinearIsotropicMaterial,
    initial_material_point_state,
    update_plane_strain_j2,
)


@pytest.fixture
def material() -> BilinearIsotropicMaterial:
    return BilinearIsotropicMaterial(
        E=210_000_000_000.0,
        nu=0.3,
        yield_strength=250_000_000.0,
        tangent_modulus=2_000_000_000.0,
    )


def _finite_difference_tangent(
    material: BilinearIsotropicMaterial,
    strain_increment: np.ndarray,
    committed: object | None = None,
    perturbation: float = 1.0e-8,
) -> np.ndarray:
    committed = committed or initial_material_point_state(material)
    columns: list[np.ndarray] = []
    for component in range(3):
        plus = strain_increment.copy()
        minus = strain_increment.copy()
        plus[component] += perturbation
        minus[component] -= perturbation
        stress_plus = np.asarray(
            update_plane_strain_j2(material, committed, plus).in_plane_stress_Pa
        )
        stress_minus = np.asarray(
            update_plane_strain_j2(material, committed, minus).in_plane_stress_Pa
        )
        columns.append((stress_plus - stress_minus) / (2.0 * perturbation))
    return np.column_stack(columns)


def test_elastic_plane_strain_tangent_matches_hooke_matrix_and_shear_convention(
    material: BilinearIsotropicMaterial,
) -> None:
    committed = initial_material_point_state(material)
    update = update_plane_strain_j2(
        material,
        committed,
        np.array([1.0e-5, -3.0e-6, 4.0e-6]),
    )
    shear = material.shear_modulus
    lame = material.bulk_modulus - 2.0 * shear / 3.0
    expected = np.array(
        [
            [lame + 2.0 * shear, lame, 0.0],
            [lame, lame + 2.0 * shear, 0.0],
            [0.0, 0.0, shear],
        ]
    )

    assert update.state.increment_class == "elastic"
    assert np.asarray(update.algorithmic_tangent_Pa) == pytest.approx(
        expected, rel=1.0e-13, abs=1.0e-5
    )
    assert np.asarray(update.strain_increment_tensor)[2, 2] == 0.0
    assert np.asarray(update.strain_increment_tensor)[0, 1] == pytest.approx(2.0e-6)
    assert update.in_plane_stress_Pa[2] == pytest.approx(
        shear * 4.0e-6, rel=1.0e-13, abs=1.0e-5
    )


def test_plastic_plane_strain_consistent_tangent_matches_independent_finite_difference(
    material: BilinearIsotropicMaterial,
) -> None:
    strain_increment = np.array([0.0040, -0.0010, 0.0006])
    committed = initial_material_point_state(material)

    update = update_plane_strain_j2(material, committed, strain_increment)
    numerical = _finite_difference_tangent(material, strain_increment)

    assert update.state.increment_class == "plastic"
    assert np.asarray(update.algorithmic_tangent_Pa) == pytest.approx(
        numerical,
        rel=3.0e-6,
        abs=2.0e4,
    )
    assert np.asarray(update.algorithmic_tangent_Pa) == pytest.approx(
        np.asarray(update.algorithmic_tangent_Pa).T,
        rel=1.0e-13,
        abs=1.0e-4,
    )
    assert update.sigma_zz_Pa == pytest.approx(
        np.asarray(update.state.stress_tensor_Pa)[2, 2]
    )


def test_plastic_consistent_tangent_matches_finite_difference_from_a_hardened_state(
    material: BilinearIsotropicMaterial,
) -> None:
    initial = initial_material_point_state(material)
    hardened = update_plane_strain_j2(
        material,
        initial,
        np.array([0.0035, -0.0008, 0.0004]),
    ).state
    increment = np.array([0.0010, -0.0002, -0.0003])

    update = update_plane_strain_j2(material, hardened, increment)
    numerical = _finite_difference_tangent(
        material,
        increment,
        committed=hardened,
    )

    assert update.state.increment_class == "plastic"
    assert np.asarray(update.algorithmic_tangent_Pa) == pytest.approx(
        numerical,
        rel=3.0e-6,
        abs=2.0e4,
    )


def test_zero_increment_uses_the_registered_loading_side_for_the_plastic_gradient(
    material: BilinearIsotropicMaterial,
) -> None:
    initial = initial_material_point_state(material)
    hardened = update_plane_strain_j2(
        material,
        initial,
        np.array([0.0035, -0.0008, 0.0004]),
    ).state
    loading_direction = np.array([0.0010, -0.0002, -0.0003])
    zero = update_plane_strain_j2(
        material,
        hardened,
        np.zeros(3),
        active_side_direction=loading_direction,
    )
    step = 1.0e-8
    upper = update_plane_strain_j2(
        material,
        hardened,
        step * loading_direction,
    )
    directional_finite_difference = (
        upper.state.equivalent_plastic_strain
        - hardened.equivalent_plastic_strain
    ) / step

    assert zero.state.increment_class == "elastic"
    assert np.dot(
        np.asarray(zero.equivalent_plastic_strain_gradient),
        loading_direction,
    ) == pytest.approx(directional_finite_difference, rel=3.0e-6)


@pytest.mark.parametrize(
    "increment",
    [
        np.zeros(2),
        np.zeros(4),
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, 0.0, np.inf]),
        [0.0, 0.0, "bad"],
    ],
)
def test_plane_strain_update_rejects_wrong_shape_and_nonfinite_values(
    material: BilinearIsotropicMaterial,
    increment: object,
) -> None:
    with pytest.raises(ValueError, match="plane_strain_increment"):
        update_plane_strain_j2(
            material,
            initial_material_point_state(material),
            increment,
        )


def test_reloading_direction_inside_yield_surface_remains_elastic(material):
    initial=initial_material_point_state(material)
    loaded=update_plane_strain_j2(material,initial,np.array([.0035,-.0008,.0004])).state
    unloaded=update_plane_strain_j2(material,loaded,np.array([-.0001,.00002,0.])).state
    direction=np.array([.001,-.0002,0.])
    baseline=update_plane_strain_j2(material,unloaded,np.zeros(3))
    sided=update_plane_strain_j2(material,unloaded,np.zeros(3),active_side_direction=direction)
    np.testing.assert_allclose(sided.algorithmic_tangent_Pa,baseline.algorithmic_tangent_Pa,rtol=1e-13)
    np.testing.assert_array_equal(sided.equivalent_plastic_strain_gradient,0.)
