from __future__ import annotations

import json
from pathlib import Path

import pytest

from grindcae.gui.project import (
    PROJECT_NODE_ORDER,
    ProjectError,
    ProjectNodeStatus,
    ResultRecord,
    create_project,
    load_project,
    save_project,
)
from grindcae.gui.providers import DisabledProvider, EXTENSION_IDENTIFIERS
from grindcae.gui.workbench import WorkbenchSession
from grindcae.gui.geometry import GeometryDisplayInput, Parametric2DGeometryProvider
from grindcae.gui.mesh_workspace import MeshDisplayInput, TriangleMeshProvider
from grindcae.gui.material_workspace import MaterialDisplayInput
from grindcae.gui.process_workspace import ProcessDisplayInput, WheelDisplayInput


def test_new_project_has_the_fixed_chinese_workflow_and_no_fake_result() -> None:
    project = create_project("刚玉砂轮单程磨削演示", "中文备注")

    assert PROJECT_NODE_ORDER == ("工程", "几何", "网格", "材料", "工艺", "分析", "结果")
    assert project.project_name == "刚玉砂轮单程磨削演示"
    assert project.description == "中文备注"
    assert project.node_status("工程") is ProjectNodeStatus.READY
    assert project.node_status("几何") is ProjectNodeStatus.NOT_CONFIGURED
    assert project.node_status("网格") is ProjectNodeStatus.DISABLED
    assert project.node_status("材料") is ProjectNodeStatus.DISABLED
    assert project.node_status("分析") is ProjectNodeStatus.NOT_CONFIGURED
    assert project.latest_result is None
    assert "完成二维几何" in project.node_reason("网格")
    assert "将在 2.6.0 实现" not in project.node_reason("网格")


def test_project_utf8_round_trip_preserves_chinese_status_and_future_fields(
    tmp_path: Path,
) -> None:
    project = create_project("刚玉砂轮单程磨削演示", "保留中文备注")
    project.set_node_status("几何", ProjectNodeStatus.FAILED, "几何占位验证失败")
    project.extra_fields["future_extension"] = {"中文键": "未来值"}
    path = save_project(project, tmp_path / "test_project.gcase.json")

    raw = path.read_bytes()
    assert "刚玉砂轮单程磨削演示".encode("utf-8") in raw
    assert b"\\u521a\\u7389" not in raw
    loaded = load_project(path)
    assert loaded.project_name == project.project_name
    assert loaded.description == project.description
    assert loaded.node_status("几何") is ProjectNodeStatus.FAILED
    assert loaded.node_reason("几何") == "几何占位验证失败"
    assert loaded.extra_fields["future_extension"] == {"中文键": "未来值"}


def test_project_save_rejects_nan_and_infinity(tmp_path: Path) -> None:
    project = create_project("有限数测试")
    project.extra_fields["bad_value"] = float("nan")
    with pytest.raises(ProjectError, match="NaN 或 Infinity"):
        save_project(project, tmp_path / "nan.gcase.json")

    project.extra_fields["bad_value"] = float("inf")
    with pytest.raises(ProjectError, match="NaN 或 Infinity"):
        save_project(project, tmp_path / "infinity.gcase.json")


def test_invalid_version_and_missing_required_fields_have_chinese_errors(
    tmp_path: Path,
) -> None:
    invalid_version = tmp_path / "invalid_version.gcase.json"
    invalid_version.write_text(
        json.dumps(
            {
                "project_schema_version": 99,
                "project_type": "grindcae_project",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectError, match="不支持的工程文件版本"):
        load_project(invalid_version)

    missing = tmp_path / "missing.gcase.json"
    missing.write_text(
        json.dumps(
            {
                "project_schema_version": 1,
                "project_type": "grindcae_project",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectError, match="缺少必需字段.*project_name"):
        load_project(missing)


def test_dependencies_block_later_work_and_disabled_nodes_never_become_completed() -> None:
    project = create_project("依赖测试")

    allowed, reason = project.can_execute("网格")
    assert allowed is False
    assert "几何" in reason
    project.set_node_status("网格", ProjectNodeStatus.DISABLED, "真实网格将在 2.6.0 实现")
    assert project.node_status("网格") is ProjectNodeStatus.DISABLED
    with pytest.raises(ProjectError, match="暂未启用"):
        project.set_node_status("网格", ProjectNodeStatus.COMPLETED)


def test_result_registry_round_trip_uses_real_paths_and_failure_preserves_history(
    tmp_path: Path,
) -> None:
    output = tmp_path / "real_output"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    image = output / "result.png"
    image.write_bytes(b"png")
    project = create_project("结果测试")
    project.register_result(
        ResultRecord(
            result_id="run-001",
            analysis_type="single_position",
            result_format="grindcae_test_result",
            output_directory=str(output),
            summary_path=str(summary),
            artifact_paths={"preview": str(image)},
            created_at="2026-08-22T12:00:00+08:00",
            status="completed",
        )
    )
    path = save_project(project, tmp_path / "result_project.gcase.json")

    loaded = load_project(path)
    assert loaded.latest_result is not None
    assert loaded.latest_result.output_directory == str(output.resolve())
    assert loaded.node_status("结果") is ProjectNodeStatus.COMPLETED

    loaded.mark_analysis_failed("计算失败：测试注入")
    assert loaded.latest_result is not None
    assert loaded.latest_result.result_id == "run-001"
    assert loaded.node_status("分析") is ProjectNodeStatus.FAILED
    assert loaded.node_status("结果") is ProjectNodeStatus.FAILED


def test_result_registry_rejects_nonexistent_output(tmp_path: Path) -> None:
    project = create_project("结果路径测试")
    with pytest.raises(ProjectError, match="真实计算输出"):
        project.register_result(
            ResultRecord(
                result_id="missing",
                analysis_type="single_pass",
                result_format="missing",
                output_directory=str(tmp_path / "missing"),
                summary_path=str(tmp_path / "missing" / "summary.json"),
                artifact_paths={},
                created_at="2026-08-22T12:00:00+08:00",
                status="completed",
            )
        )


def test_reopening_project_with_deleted_result_does_not_show_fake_completion(
    tmp_path: Path,
) -> None:
    output = tmp_path / "temporary_result"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    project = create_project("删除结果测试")
    project.register_result(
        ResultRecord(
            result_id="deleted",
            analysis_type="single_position",
            result_format="grindcae_test_result",
            output_directory=str(output),
            summary_path=str(summary),
            artifact_paths={},
            created_at="2026-08-22T12:00:00+08:00",
        )
    )
    saved = save_project(project, tmp_path / "deleted_result.gcase.json")
    summary.unlink()
    output.rmdir()

    reopened = load_project(saved)

    assert reopened.latest_result is None
    assert reopened.node_status("结果") is ProjectNodeStatus.NOT_CONFIGURED
    assert reopened.node_reason("结果") == "暂无结果。原登记结果已不存在。"


def test_disabled_provider_exposes_minimal_future_interface() -> None:
    provider = DisabledProvider("二维几何建模将在 2.5.0 实现。")

    assert provider.validate() == (False, "二维几何建模将在 2.5.0 实现。")
    assert provider.validate_geometry() == provider.validate()
    assert provider.validate_mesh_settings() == provider.validate()
    assert provider.validate_material() == provider.validate()
    assert provider.validate_case() == provider.validate()
    assert provider.summary()["status"] == "disabled"
    assert provider.summary_text() == "二维几何建模将在 2.5.0 实现。"
    assert provider.plots() == ()
    assert provider.tables() == ()
    assert provider.warnings() == ("二维几何建模将在 2.5.0 实现。",)
    with pytest.raises(ProjectError, match="暂未启用"):
        provider.run()
    with pytest.raises(ProjectError, match="暂未启用"):
        provider.generate_mesh()

    assert EXTENSION_IDENTIFIERS["geometry_source"] == ("parametric_2d", "step_cad")
    assert EXTENSION_IDENTIFIERS["mesh_type"] == ("triangle", "quadrilateral")
    assert EXTENSION_IDENTIFIERS["physics"] == ("mechanical", "thermo_mechanical")
    assert EXTENSION_IDENTIFIERS["grinding_scale"] == (
        "statistical_equivalent",
        "single_grain",
    )
    assert EXTENSION_IDENTIFIERS["process_mode"] == (
        "single_position",
        "single_pass",
        "multi_pass",
    )
    assert EXTENSION_IDENTIFIERS["surface_output"] == (
        "displacement",
        "stress",
        "plastic_strain",
        "roughness",
    )


def test_workbench_node_pages_are_chinese_and_analysis_keeps_all_legacy_modes() -> None:
    session = WorkbenchSession(create_project("工作台测试"))

    assert session.tree_rows() == (
        ("工程", "可执行"),
        ("几何", "未配置"),
        ("网格", "暂未启用"),
        ("材料", "暂未启用"),
        ("工艺", "未配置"),
        ("分析", "未配置"),
        ("结果", "未配置"),
    )

    geometry = session.select_node("几何")
    assert geometry.title == "几何"
    assert geometry.status == "未配置"
    assert "生成参数化二维轮廓和预览图" in geometry.description
    assert geometry.actions == ("生成二维模型",)

    analysis = session.select_node("分析")
    assert analysis.status == "未配置"
    assert analysis.actions == ("运行分析", "打开高级兼容计算")
    assert analysis.input_summary == (
        "当前模式：线弹性·单位置",
        "分析假设：二维平面应力",
        "计算范围：单位置",
        "材料模型：线弹性",
        "载荷模型：均匀经验载荷",
        "输入校验：未通过",
    )

    results = session.select_node("结果")
    assert results.output_summary == ("暂无结果",)


def test_workbench_success_registers_published_result_and_failure_preserves_history(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    preview = output / "preview.png"
    preview.write_bytes(b"png")
    session = WorkbenchSession(create_project("工作台结果"))

    session.register_published_result(
        mode="linear_elastic_single_position",
        result_format="grindcae_phase_4c3b",
        output_directory=output,
        summary_path=summary,
        artifact_paths={"preview": preview},
        created_at="2026-08-22T12:00:00+08:00",
    )
    result_page = session.select_node("结果")
    assert result_page.status == "已完成"
    assert f"结果目录：{output.resolve()}" in result_page.output_summary

    session.begin_analysis()
    assert session.project.latest_result is not None
    assert session.project.node_status("分析") is ProjectNodeStatus.RUNNING
    assert session.project.node_status("结果") is ProjectNodeStatus.RUNNING

    session.mark_analysis_failed("计算失败：不显示旧结果")
    failed_page = session.select_node("结果")
    assert failed_page.status == "失败"
    assert failed_page.output_summary[0] == f"结果目录：{output.resolve()}"


def test_geometry_success_persists_schema_updates_tree_without_completing_analysis(
    tmp_path: Path,
) -> None:
    project = create_project("几何工程")
    session = WorkbenchSession(project)
    provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    published = provider.publish(tmp_path / "geometry")

    session.register_geometry_result(provider, published)

    assert project.geometry["source"] == "parametric_2d"
    assert project.geometry["geometry_schema_version"] == 1
    assert project.node_status("几何") is ProjectNodeStatus.COMPLETED
    assert project.node_status("网格") is ProjectNodeStatus.READY
    assert project.node_status("材料") is ProjectNodeStatus.DISABLED
    assert project.node_status("分析") is ProjectNodeStatus.NOT_CONFIGURED
    assert project.node_status("结果") is ProjectNodeStatus.COMPLETED
    assert project.latest_geometry_result is not None
    assert project.latest_geometry_result.analysis_type == "geometry_preview"

    saved = save_project(project, tmp_path / "geometry_project.gcase.json")
    reopened = load_project(saved)
    assert reopened.geometry == project.geometry
    assert reopened.node_status("几何") is ProjectNodeStatus.COMPLETED
    assert reopened.latest_geometry_result is not None


def test_analysis_start_and_failure_preserve_valid_geometry_result(tmp_path: Path) -> None:
    project = create_project("几何保留")
    session = WorkbenchSession(project)
    provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    published = provider.publish(tmp_path / "geometry")
    session.register_geometry_result(provider, published)

    session.begin_analysis()
    session.mark_analysis_failed("分析失败")

    assert project.geometry is not None
    assert project.latest_geometry_result is not None
    assert project.node_status("几何") is ProjectNodeStatus.COMPLETED
    assert project.node_status("分析") is ProjectNodeStatus.FAILED
    assert project.node_status("结果") is ProjectNodeStatus.FAILED


def test_geometry_failure_keeps_previous_valid_geometry_and_marks_node_failed(
    tmp_path: Path,
) -> None:
    project = create_project("失败保护")
    session = WorkbenchSession(project)
    provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    published = provider.publish(tmp_path / "geometry")
    session.register_geometry_result(provider, published)
    previous_geometry = json.loads(json.dumps(project.geometry))
    previous_result = project.latest_geometry_result

    session.mark_geometry_failed("输入错误：工件长度必须大于 0。")

    assert project.geometry == previous_geometry
    assert project.latest_geometry_result == previous_result
    assert project.node_status("几何") is ProjectNodeStatus.FAILED
    assert project.node_status("结果") is ProjectNodeStatus.COMPLETED


def test_old_2_4_project_without_geometry_still_opens_as_not_configured(
    tmp_path: Path,
) -> None:
    project = create_project("旧工程")
    payload = project.to_dict()
    payload["geometry"] = None
    path = tmp_path / "old_2_4.gcase.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    reopened = load_project(path)

    assert reopened.geometry is None
    assert reopened.node_status("几何") is ProjectNodeStatus.NOT_CONFIGURED
    page = WorkbenchSession(reopened).select_node("几何")
    assert page.actions == ("生成二维模型",)
    assert page.output_summary == ("尚未生成二维几何。",)


def test_step_cad_geometry_is_preserved_and_shown_as_deferred(tmp_path: Path) -> None:
    project = create_project("CAD 预留")
    project.geometry = {
        "source": "step_cad",
        "geometry_schema_version": 1,
        "unit_system": "SI",
        "future_cad": {"filename": "future.step"},
    }
    path = save_project(project, tmp_path / "cad_reserved.gcase.json")

    reopened = load_project(path)
    page = WorkbenchSession(reopened).select_node("几何")

    assert reopened.geometry["future_cad"] == {"filename": "future.step"}
    assert page.output_summary == ("STEP/CAD 导入将在后续版本实现。",)
    assert page.status == "暂未启用"


def test_project_open_rejects_invalid_parametric_geometry_version(
    tmp_path: Path,
) -> None:
    project = create_project("非法几何版本")
    provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    project.geometry = provider.to_geometry_schema()
    project.geometry["geometry_schema_version"] = 2
    path = tmp_path / "bad_geometry_version.gcase.json"
    path.write_text(
        json.dumps(project.to_dict(), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(ProjectError, match="几何文件版本"):
        load_project(path)


def test_mesh_success_round_trip_and_new_geometry_marks_it_stale(tmp_path: Path) -> None:
    project = create_project("网格状态工程")
    session = WorkbenchSession(project)
    geometry_provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    geometry_published = geometry_provider.publish(tmp_path / "geometry")
    session.register_geometry_result(geometry_provider, geometry_published)
    mesh_provider = TriangleMeshProvider(
        geometry_provider.build_geometry(), MeshDisplayInput().build_case()
    )
    mesh_published = mesh_provider.publish(tmp_path / "mesh")

    session.register_mesh_result(mesh_provider, mesh_published)

    assert project.mesh["mesh_type"] == "triangle"
    assert project.mesh["status"] == "completed"
    assert project.node_status("网格") is ProjectNodeStatus.COMPLETED
    assert project.node_status("材料") is ProjectNodeStatus.READY
    assert project.node_status("分析") is ProjectNodeStatus.NOT_CONFIGURED
    assert project.latest_mesh_result is not None
    saved = save_project(project, tmp_path / "mesh_project.gcase.json")
    reopened = load_project(saved)
    assert reopened.mesh == project.mesh
    assert reopened.latest_mesh_result is not None

    changed_provider = Parametric2DGeometryProvider(
        GeometryDisplayInput(workpiece_length_mm="120").build_case()
    )
    changed_published = changed_provider.publish(tmp_path / "changed_geometry")
    WorkbenchSession(reopened).register_geometry_result(
        changed_provider, changed_published
    )

    assert reopened.mesh["status"] == "stale"
    assert reopened.node_status("网格") is ProjectNodeStatus.STALE
    assert reopened.latest_mesh_result is None
    assert not any(
        record.analysis_type == "mesh_preview" for record in reopened.results
    )


def test_project_without_mesh_key_opens_as_mesh_not_generated(tmp_path: Path) -> None:
    project = create_project("无 mesh 键旧工程")
    payload = project.to_dict()
    del payload["mesh"]
    path = tmp_path / "old_without_mesh.gcase.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    reopened = load_project(path)

    assert reopened.mesh is None
    assert reopened.node_status("网格") in (
        ProjectNodeStatus.DISABLED,
        ProjectNodeStatus.NOT_CONFIGURED,
    )


def test_analysis_start_and_failure_preserve_geometry_and_mesh_results(
    tmp_path: Path,
) -> None:
    project = create_project("几何网格结果保留")
    session = WorkbenchSession(project)
    geometry_provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    geometry_published = geometry_provider.publish(tmp_path / "geometry")
    session.register_geometry_result(geometry_provider, geometry_published)
    mesh_provider = TriangleMeshProvider(
        geometry_provider.build_geometry(), MeshDisplayInput().build_case()
    )
    mesh_published = mesh_provider.publish(tmp_path / "mesh")
    session.register_mesh_result(mesh_provider, mesh_published)

    session.begin_analysis()
    session.mark_analysis_failed("分析失败")

    assert project.latest_geometry_result is not None
    assert project.latest_mesh_result is not None
    assert project.node_status("网格") is ProjectNodeStatus.COMPLETED
    assert project.node_status("结果") is ProjectNodeStatus.FAILED


def test_reopening_project_with_deleted_mesh_artifacts_invalidates_mesh_status(
    tmp_path: Path,
) -> None:
    project = create_project("删除网格产物")
    session = WorkbenchSession(project)
    geometry_provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    geometry_published = geometry_provider.publish(tmp_path / "geometry")
    session.register_geometry_result(geometry_provider, geometry_published)
    mesh_provider = TriangleMeshProvider(
        geometry_provider.build_geometry(), MeshDisplayInput().build_case()
    )
    mesh_published = mesh_provider.publish(tmp_path / "mesh")
    session.register_mesh_result(mesh_provider, mesh_published)
    saved = save_project(project, tmp_path / "deleted_mesh.gcase.json")
    mesh_published.preview_path.unlink()

    reopened = load_project(saved)

    assert reopened.latest_mesh_result is None
    assert reopened.node_status("网格") is ProjectNodeStatus.STALE
    assert reopened.mesh["status"] == "stale"


def test_old_project_missing_material_wheel_and_process_still_opens(
    tmp_path: Path,
) -> None:
    payload = create_project("2.6 旧工程").to_dict()
    del payload["material"]
    del payload["wheel"]
    del payload["process"]
    path = tmp_path / "old_2_6.gcase.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    reopened = load_project(path)

    assert reopened.material is None
    assert reopened.wheel is None
    assert reopened.process is None
    assert reopened.node_status("材料") is ProjectNodeStatus.DISABLED
    assert reopened.node_status("工艺") is ProjectNodeStatus.NOT_CONFIGURED


def test_material_wheel_process_round_trip_and_pages_use_chinese_summaries(
    tmp_path: Path,
) -> None:
    project = create_project("2.7 参数工程")
    project.geometry = Parametric2DGeometryProvider(
        GeometryDisplayInput().build_case()
    ).to_geometry_schema()
    project.mesh = MeshDisplayInput().build_case().to_dict()
    project.mesh["status"] = "completed"
    project.set_node_status("几何", ProjectNodeStatus.COMPLETED)
    project.set_node_status("网格", ProjectNodeStatus.READY)
    project.set_node_status("网格", ProjectNodeStatus.COMPLETED)
    material = MaterialDisplayInput().build_case()
    wheel = WheelDisplayInput().build_case()
    process = ProcessDisplayInput().build_case(project.geometry, wheel)
    session = WorkbenchSession(project)

    session.register_material(material)
    session.register_process(wheel, process)
    saved = save_project(project, tmp_path / "parameters.gcase.json")
    reopened = load_project(saved)

    assert reopened.material == material.to_dict()
    assert reopened.wheel == wheel.to_dict()
    assert reopened.process == process.to_dict()
    assert reopened.node_status("材料") is ProjectNodeStatus.COMPLETED
    assert reopened.node_status("工艺") is ProjectNodeStatus.COMPLETED
    material_page = WorkbenchSession(reopened).select_node("材料")
    process_page = WorkbenchSession(reopened).select_node("工艺")
    assert "弹性模量：210 GPa" in material_page.input_summary
    assert "刚玉 #5000" in process_page.input_summary
    assert "砂轮直径：200 mm（来自几何，只读）" in process_page.input_summary
    assert "磨削深度：20 μm（来自几何，只读）" in process_page.input_summary


def test_geometry_change_marks_material_process_analysis_and_results_stale(
    tmp_path: Path,
) -> None:
    project = create_project("过期联动")
    session = WorkbenchSession(project)
    geometry_provider = Parametric2DGeometryProvider(GeometryDisplayInput().build_case())
    session.register_geometry_result(
        geometry_provider, geometry_provider.publish(tmp_path / "geometry")
    )
    mesh_provider = TriangleMeshProvider(
        geometry_provider.build_geometry(), MeshDisplayInput().build_case()
    )
    session.register_mesh_result(mesh_provider, mesh_provider.publish(tmp_path / "mesh"))
    material = MaterialDisplayInput().build_case()
    wheel = WheelDisplayInput().build_case()
    process = ProcessDisplayInput().build_case(project.geometry, wheel)
    session.register_material(material)
    session.register_process(wheel, process)

    analysis_output = tmp_path / "analysis"
    analysis_output.mkdir()
    summary = analysis_output / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    session.register_published_result(
        mode="linear_elastic_single_position",
        result_format="test",
        output_directory=analysis_output,
        summary_path=summary,
        artifact_paths={},
        created_at="2026-08-22T12:00:00+08:00",
    )
    analysis_record = project.latest_analysis_result

    changed = Parametric2DGeometryProvider(
        GeometryDisplayInput(workpiece_length_mm="120").build_case()
    )
    session.register_geometry_result(changed, changed.publish(tmp_path / "changed"))

    assert project.node_status("网格") is ProjectNodeStatus.STALE
    assert project.node_status("材料") is ProjectNodeStatus.STALE
    assert project.node_status("工艺") is ProjectNodeStatus.STALE
    assert project.node_status("分析") is ProjectNodeStatus.STALE
    assert project.node_status("结果") is ProjectNodeStatus.STALE
    assert project.latest_analysis_result == analysis_record
    assert project.material == material.to_dict()
    assert project.process == process.to_dict()


def test_material_or_process_change_preserves_geometry_mesh_and_old_analysis_record(
    tmp_path: Path,
) -> None:
    project = create_project("参数变更保护")
    project.geometry = Parametric2DGeometryProvider(
        GeometryDisplayInput().build_case()
    ).to_geometry_schema()
    project.mesh = MeshDisplayInput().build_case().to_dict()
    project.mesh["status"] = "completed"
    project.set_node_status("几何", ProjectNodeStatus.COMPLETED)
    project.set_node_status("网格", ProjectNodeStatus.READY)
    project.set_node_status("网格", ProjectNodeStatus.COMPLETED)
    session = WorkbenchSession(project)
    session.register_material(MaterialDisplayInput().build_case())
    wheel = WheelDisplayInput().build_case()
    session.register_process(wheel, ProcessDisplayInput().build_case(project.geometry, wheel))
    output = tmp_path / "old_analysis"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    session.register_published_result(
        mode="linear_elastic_single_position", result_format="test", output_directory=output,
        summary_path=summary, artifact_paths={},
        created_at="2026-08-22T12:00:00+08:00",
    )
    geometry_before = json.loads(json.dumps(project.geometry))
    mesh_before = json.loads(json.dumps(project.mesh))
    record_before = project.latest_analysis_result

    session.register_material(
        MaterialDisplayInput(label="用户材料", parameter_source="user_custom").build_case()
    )

    assert project.geometry == geometry_before
    assert project.mesh == mesh_before
    assert project.latest_analysis_result == record_before
    assert project.node_status("分析") is ProjectNodeStatus.STALE
    assert project.node_status("结果") is ProjectNodeStatus.STALE
