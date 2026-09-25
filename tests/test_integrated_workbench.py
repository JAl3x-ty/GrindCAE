import json
from dataclasses import replace
import time

import pytest

MODE='literature_grinding_force'


def test_registered_mode_uses_shared_runner_and_classified_results(tmp_path):
    from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS, AnalysisSettings, AnalysisInputBuilder
    assert MODE in ANALYSIS_MODE_DEFINITIONS, 'Literature mode is not integrated'
    from grindcae.gui.project import create_project, save_project, load_project
    from grindcae.gui.analysis_workspace import AnalysisWorkspace
    from grindcae.gui.analysis_runner import AnalysisRunner
    from grindcae.gui.workbench import WorkbenchSession
    from grindcae.gui.result_catalog import ResultCatalog
    project=create_project('论文集成')
    settings=AnalysisSettings.recommended()
    workspace=AnalysisWorkspace(project,MODE,settings,str(tmp_path/'run'))
    state=workspace.evaluate()
    assert state.can_run, state.error_message
    built=state.built
    assert built.case.reference_case.depth_m==pytest.approx(15e-6)
    workspace.save_configuration()
    session=WorkbenchSession(project)
    session.begin_analysis(mode=MODE,output_directory=tmp_path/'run',input_fingerprint=built.input_fingerprint,input_snapshot=built.normalized_input)
    runner=AnalysisRunner();thread=runner.start(built,tmp_path/'run');thread.join(25)
    assert not thread.is_alive()
    events=[]
    while not runner.events.empty():events.append(runner.events.get_nowait())
    assert not [e for e in events if e.kind=='error'],[(e.kind,e.message) for e in events]
    published=next(e.result for e in events if e.kind=='success')
    assert published.summary['total_n_N']==pytest.approx(65.54138202558843)
    session.register_published_result(mode=MODE,result_format=published.result_format,output_directory=published.output_directory,
        summary_path=published.summary_path,artifact_paths={'representative_image':published.representative_image},
        created_at='2026-09-25T16:00:00+08:00',input_fingerprint=built.input_fingerprint)
    catalog=ResultCatalog().read(project.latest_analysis_result)
    assert 'grains.csv' in [item.filename for item in catalog.category('data').items]
    assert 'force_components.png' in [item.filename for item in catalog.category('history').items]
    from grindcae.gui.origin_export import export_origin_data
    exported=export_origin_data(catalog,tmp_path/'origin')
    assert any(p.name=='grains_origin.csv' for p in exported.table_paths)
    path=save_project(project,tmp_path/'集成.gcae')
    restored=load_project(path)
    assert restored.analysis['mode']==MODE
    assert AnalysisInputBuilder().build(restored,MODE,settings).input_fingerprint==built.input_fingerprint
    from grindcae.gui.analysis_workspace import analysis_presentation_state
    changed=replace(settings,literature_case=replace(built.case,reference_case=replace(built.case.reference_case,depth_m=20e-6)).to_dict())
    changed_state=AnalysisWorkspace(restored,MODE,changed).evaluate()
    assert analysis_presentation_state(restored,changed_state).status=='stale'
    from grindcae.gui.project_history import import_result_directory, RecentResultStore
    entry=import_result_directory(tmp_path/'run',store=RecentResultStore(tmp_path/'recent.json'))
    assert entry.analysis_type==MODE
    from grindcae.gui.project_archive import export_project_archive, import_project_archive
    archive=export_project_archive(path,tmp_path/'paper.gcae-archive')
    imported=import_project_archive(archive,tmp_path/'restored')
    reopened=load_project(imported.project_path)
    assert ResultCatalog().read(reopened.latest_analysis_result).published.mode==MODE


def test_old_settings_roundtrip_keeps_original_field_set():
    from grindcae.gui.analysis import AnalysisSettings
    original=AnalysisSettings.recommended().to_dict()
    original.pop('literature_case',None)
    assert AnalysisSettings.from_mapping(original).to_dict()==original


def test_main_tk_mode_and_parameter_editor_save_into_project(tmp_path):
    from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS, AnalysisSettings
    assert MODE in ANALYSIS_MODE_DEFINITIONS
    import tkinter as tk
    from grindcae.gui.app import GrindCaeApp
    from grindcae.gui.literature_integration import LiteratureParameterDialog
    root=tk.Tk();root.withdraw()
    app=GrindCaeApp(root)
    try:
        app.analysis_mode_var.set(ANALYSIS_MODE_DEFINITIONS[MODE].display_name)
        app._refresh_analysis_workspace()
        assert app.analysis_run_button.instate(['!disabled'])
        assert 'Zhang 2017' in app.calibration_display_var.get()
        window=LiteratureParameterDialog(app)
        window.variables['depth_m'].set('20')
        window.apply()
        root.update()
        settings=app._analysis_settings_from_widgets()
        assert settings.literature_case['reference_case']['depth_m']==pytest.approx(20e-6)
        app._save_analysis_configuration()
        assert app.workbench.project.analysis['settings']['literature_case']['reference_case']['depth_m']==pytest.approx(20e-6)
        assert 'single_grain_high_fidelity' in ANALYSIS_MODE_DEFINITIONS
        app.analysis_mode_var.set(ANALYSIS_MODE_DEFINITIONS['linear_elastic_single_position'].display_name)
        app._refresh_analysis_workspace()
        assert 'Zhang 2017' not in app.calibration_display_var.get()
    finally:root.destroy()


def test_real_workbench_completion_updates_progress(tmp_path, monkeypatch):
    import tkinter as tk
    from grindcae.gui.app import GrindCaeApp
    from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS
    monkeypatch.setattr('grindcae.gui.app.messagebox.askyesnocancel', lambda *a, **k: False)
    root=tk.Tk();root.withdraw();app=GrindCaeApp(root)
    try:
        app.analysis_mode_var.set(ANALYSIS_MODE_DEFINITIONS[MODE].display_name)
        app.analysis_output_var.set(str(tmp_path/'result'))
        app._start_workbench_analysis()
        deadline=time.monotonic()+25
        while (app.analysis_runner.is_running or app._active_analysis_built is not None) and time.monotonic()<deadline:
            root.update();time.sleep(.02)
        assert app.workbench.project.latest_analysis_result is not None
        assert app.analysis_progress_var.get()==1
        assert '100.0%' in app.analysis_progress_detail_var.get()
        assert '准备' not in app.analysis_progress_detail_var.get()
    finally:root.destroy()


def test_explicit_process_sync_uses_engineering_sources_keeps_literature_material():
    from host_regression.test_gui_analysis_inputs import configured_project
    from grindcae.gui.literature_integration import sync_process
    from grindcae380.desktop import default_case
    case=sync_process(default_case(True),configured_project())
    assert case.reference_case.wheel_diameter_m==pytest.approx(.2)
    assert case.reference_case.depth_m==pytest.approx(20e-6)
    assert case.reference_case.wheel_speed_m_s==31
    assert case.reference_case.feed_speed_m_s==pytest.approx(.12)
    assert case.reference_case.width_m==pytest.approx(.009)
    assert case.reference_case.yield_stress_Pa==225e6
