from __future__ import annotations

from datetime import datetime
from pathlib import Path

from grindcae.gui.analysis import AnalysisProjectData, AnalysisSettings
from grindcae.gui.analysis_workspace import AnalysisWorkspace, default_analysis_output_directory
from grindcae.gui.project import ProjectNodeStatus

from test_gui_analysis_inputs import configured_project


def test_workspace_validates_selected_task_and_produces_chinese_input_summary() -> None:
    workspace = AnalysisWorkspace(configured_project(), "elastoplastic_single_position", AnalysisSettings.recommended())
    state = workspace.evaluate()
    assert state.can_run
    assert state.built is not None
    assert state.input_checks == (
        "几何：有效", "网格：有效", "材料：有效", "工艺：有效",
        "砂轮粒度：已解析", "分析设置：有效", "严格求解输入：已通过校验",
    )
    assert "50 mm（来自几何）" in state.mapping_summary
    assert "平面应变" in state.task_summary


def test_workspace_reports_specific_reason_when_upstream_is_stale() -> None:
    project = configured_project()
    project.set_node_status("工艺", ProjectNodeStatus.STALE, "工艺参数变化")
    state = AnalysisWorkspace(project, "linear_elastic_single_position", AnalysisSettings.recommended()).evaluate()
    assert not state.can_run
    assert state.built is None
    assert "请先完成并确认“工艺”节点" in state.error_message


def test_default_output_directory_is_unique_and_rooted_at_saved_project(tmp_path: Path) -> None:
    project_file = tmp_path / "带 空格" / "sample.gcae"
    first = default_analysis_output_directory(
        "elastoplastic_single_position", project_file=project_file,
        now=datetime(2026, 8, 23, 15, 30, 0), exists=lambda _path: False,
    )
    second = default_analysis_output_directory(
        "elastoplastic_single_position", project_file=project_file,
        now=datetime(2026, 8, 23, 15, 30, 0), exists=lambda path: path == first,
    )
    assert first == project_file.parent / "sample_results" / "20260823_153000_elastoplastic_single_position"
    assert second.name.endswith("_02")


def test_workspace_restores_saved_mode_and_advanced_settings() -> None:
    project = configured_project()
    settings = AnalysisSettings.recommended()
    project.analysis = AnalysisProjectData(
        mode="linear_elastic_single_pass", display_name="线弹性·单程",
        status="configured", output_directory="D:/results", settings=settings,
    ).to_dict()
    workspace = AnalysisWorkspace.from_project(project)
    assert workspace.mode_id == "linear_elastic_single_pass"
    assert workspace.settings == settings
    assert workspace.output_directory == "D:/results"
