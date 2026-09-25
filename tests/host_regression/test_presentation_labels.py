from __future__ import annotations

import importlib.util

import pytest


def test_presentation_policy_module_is_available_without_gui_dependencies() -> None:
    assert importlib.util.find_spec("grindcae.presentation_labels") is not None


def test_formal_analysis_and_load_identifiers_have_one_chinese_presentation_name() -> None:
    from grindcae.presentation_labels import (
        ANALYSIS_MODE_LABELS_ZH,
        LOAD_MODEL_LABELS_ZH,
        analysis_mode_label,
        load_model_label,
    )

    assert ANALYSIS_MODE_LABELS_ZH == {
        "literature_elastoplastic_single_pass": "文献力驱动·弹塑性单程",
        "literature_grinding_force": "文献磨削力·材料去除与塑性堆积",
        "linear_elastic_single_position": "线弹性·单位置",
        "linear_elastic_single_pass": "线弹性·单程",
        "elastoplastic_single_position": "弹塑性·单位置",
        "elastoplastic_single_pass": "弹塑性·单程",
        "mechanism_elastoplastic_single_pass": "机制化磨削·单程",
        "single_grain_high_fidelity": "单磨粒真实接触",
    }
    assert load_model_label("uniform_empirical_load") == "均匀经验载荷"
    assert load_model_label("uniform_equivalent_moving_load") == "均匀等效移动载荷"
    assert (
        load_model_label("statistical_mechanism_nonuniform_load")
        == "统计等效磨粒非均匀载荷"
    )
    assert analysis_mode_label("elastoplastic_single_pass") == "弹塑性·单程"
    assert set(LOAD_MODEL_LABELS_ZH) == {
        "literature_strip_load",
        "literature_grain_force",
        "uniform_empirical_load",
            "uniform_equivalent_moving_load",
            "statistical_mechanism_nonuniform_load",
            "prescribed_rigid_grain_contact",
        }


@pytest.mark.parametrize(
    ("mode_id", "expected"),
    (
        ("linear_elastic_single_position", ()),
        ("linear_elastic_single_pass", ("scan_base_position_count",)),
        (
            "elastoplastic_single_position",
            (
                "loading_step_count",
                "unloading_step_count",
                "newton_residual_relative_tolerance",
                "newton_maximum_iterations",
            ),
        ),
        (
            "elastoplastic_single_pass",
            (
                "scan_base_position_count",
                "loading_step_count",
                "unloading_step_count",
                "newton_residual_relative_tolerance",
                "newton_maximum_iterations",
                "history_base_substep_count",
                "history_final_unloading_substep_count",
            ),
        ),
        (
            "mechanism_elastoplastic_single_pass",
            (
                "scan_base_position_count",
                "loading_step_count",
                "unloading_step_count",
                "newton_residual_relative_tolerance",
                "newton_maximum_iterations",
                "history_base_substep_count",
                "history_final_unloading_substep_count",
                "grain_minimum_group_count",
                "grain_maximum_group_count",
                "grain_random_seed",
                "load_point_count",
            ),
        ),
    ),
)
def test_advanced_parameter_policy_matches_real_solver_consumption(
    mode_id: str, expected: tuple[str, ...]
) -> None:
    from grindcae.presentation_labels import (
        ANALYSIS_MODE_PARAMETER_KEYS,
        analysis_mode_parameter_keys,
    )

    assert ANALYSIS_MODE_PARAMETER_KEYS[mode_id] == expected
    assert analysis_mode_parameter_keys(mode_id) == expected


def test_official_plot_titles_are_english_and_do_not_expose_development_phases() -> None:
    from grindcae.presentation_labels import PLOT_TITLES_EN

    forbidden = ("Phase", "4C", "6A", "6B", "7A")
    assert PLOT_TITLES_EN["history_pass_residual_state"] == (
        "Final Fully Unloaded State | External Load = 0"
    )
    assert PLOT_TITLES_EN["mechanism_history_comparison"].startswith(
        "Statistical-Mechanism Complete-Pass Comparison"
    )
    for title in PLOT_TITLES_EN.values():
        assert all(value not in title for value in forbidden)
