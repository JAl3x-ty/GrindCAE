"""Real form, files, background worker and Tk flows; no mocked solver."""
import importlib.util
import json
import time
from dataclasses import replace

import pytest


def desktop():
    assert importlib.util.find_spec('grindcae380.desktop') is not None, 'Desktop workbench missing'
    from grindcae380 import desktop as d
    return d


def test_engineering_units_round_trip_and_invalid_input():
    d=desktop()
    c=d.default_case()
    form=d.case_to_form(c)
    assert form['depth_m']=='15'
    assert form['friction_coefficient']=='0.607'
    assert float(form['depth_m'])==pytest.approx(15)
    assert float(form['feed_speed_m_s'])==pytest.approx(2)
    assert float(form['yield_stress_Pa'])==pytest.approx(225)
    form['depth_m']='20'
    form['feed_speed_m_s']='3'
    out=d.form_to_case(form,c)
    assert out.reference_case.depth_m==pytest.approx(20e-6)
    assert out.reference_case.feed_speed_m_s==pytest.approx(.05)
    assert out.relief_depth_m==pytest.approx(90e-6)
    with pytest.raises(ValueError):d.form_to_case(dict(form,seed='1.2'),c)
    with pytest.raises(ValueError):d.form_to_case(dict(form,depth_m='nan'),c)


def test_display_rounding_does_not_change_saved_precision_or_large_seed():
    d=desktop()
    c=d.default_case(calibrated=True)
    c=replace(c,reference_case=replace(c.reference_case,seed=9223372036854775807))
    form=d.case_to_form(c)
    restored=d.form_to_case(form,c)
    assert restored.to_dict()==c.to_dict()


def test_case_save_does_not_overwrite_and_calibration_provenance_retained(tmp_path):
    d=desktop(); c=d.default_case(calibrated=True)
    path=tmp_path/'参数.json'
    d.save_case(c,path)
    loaded=d.load_case(path)
    assert loaded.to_dict()==c.to_dict()
    original=path.read_bytes()
    with pytest.raises(FileExistsError):d.save_case(d.default_case(),path)
    assert path.read_bytes()==original
    assert loaded.reference_case.wear_coefficient_N_m==pytest.approx(5.283606805687778)
    assert '65.53' in loaded.reference_case.wear_provenance


def test_real_tk_run_stale_result_reload_and_invalid_input(tmp_path):
    d=desktop()
    import tkinter as tk
    root=tk.Tk(); root.withdraw()
    app=d.Workbench(root,output_root=tmp_path)
    try:
        app.set_case(d.default_case(calibrated=True))
        app.start_calculation()
        assert app.busy
        first_directory=app.pending_output
        app.start_calculation() # A second click must not create a second run.
        assert app.pending_output==first_directory
        deadline=time.monotonic()+30
        while app.busy and time.monotonic()<deadline:
            root.update(); time.sleep(.01)
        assert not app.busy
        assert app.last_error is None
        assert app.result['total_n_N']==pytest.approx(65.54138202558843,rel=2e-7)
        assert len(app.grains.get_children())>0
        assert app.result_directory.joinpath('summary.json').is_file()
        assert app.result_state=='current'
        app.variables['depth_m'].set('20')
        root.update()
        assert app.result_state=='stale'
        assert app.bridge_button.instate(['disabled'])
        app.variables['depth_m'].set('bad')
        app.start_calculation()
        assert not app.busy
        assert app.last_error
        assert app.result_directory==first_directory
        app.open_result_directory(first_directory)
        deadline=time.monotonic()+30
        while app.busy and time.monotonic()<deadline:
            root.update(); time.sleep(.01)
        assert app.last_error is None
        assert app.result_state=='current'
        assert float(app.variables['depth_m'].get())==pytest.approx(15)
        app.save_to_path(tmp_path/'saved.json')
        assert d.load_case(tmp_path/'saved.json').reference_case.depth_m==pytest.approx(15e-6)
    finally:
        root.destroy()


def test_worker_failure_recovers_and_preserves_previous_result(tmp_path):
    d=desktop()
    import tkinter as tk
    root=tk.Tk();root.withdraw();app=d.Workbench(root,output_root=tmp_path)
    try:
        app.open_result_directory(tmp_path/'missing')
        deadline=time.monotonic()+15
        while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
        assert not app.busy and app.last_error
        assert app.run_button.instate(['!disabled'])
        assert app.result is None
        app.start_calculation()
        deadline=time.monotonic()+30
        while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
        assert not app.busy and app.last_error is None
        previous=app.result
        app.open_result_directory(tmp_path/'missing_again')
        deadline=time.monotonic()+15
        while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
        assert app.last_error and app.result is previous
    finally:root.destroy()


def test_compact_window_parameters_remain_scrollable(tmp_path):
    d=desktop()
    import tkinter as tk
    root=tk.Tk();app=d.Workbench(root,output_root=tmp_path)
    try:
        root.geometry('850x560');root.update()
        assert hasattr(app,'parameter_canvas'), 'Long parameter form requires scrolling'
        assert app.parameter_canvas.yview()[1]<1
        app.parameter_canvas.yview_moveto(1);root.update()
        assert app.parameter_canvas.yview()[1]==pytest.approx(1)
    finally:root.destroy()
