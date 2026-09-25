"""Real Tk regression for clipped analysis controls and misrouted wheel events."""
import tkinter as tk
from tkinter import ttk

import pytest

from grindcae.gui.app import GrindCaeApp
from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS


@pytest.fixture
def window():
    root = tk.Tk()
    application = GrindCaeApp(root)
    root.geometry('1120x720+20+20')
    application.main_notebook.select(application.workbench_tab)
    application.analysis_mode_var.set(ANALYSIS_MODE_DEFINITIONS['literature_elastoplastic_single_pass'].display_name)
    application._refresh_work_tree('分析')
    root.update()
    yield root, application
    for callback in root.tk.call('after', 'info'):
        root.tk.call('after', 'cancel', callback)
    application._session_marker_path.unlink(missing_ok=True)
    root.destroy()


def test_wheel_over_parameter_moves_current_node_both_directions(window):
    root, app = window
    canvas = app.workbench_scroll_canvas
    label = app.analysis_advanced_labels['scan_base_position_count']
    canvas.yview_moveto(0)
    label.event_generate('<MouseWheel>', delta=-120)
    root.update()
    assert canvas.yview()[0] > 0
    label.event_generate('<MouseWheel>', delta=120)
    root.update()
    assert canvas.yview()[0] == 0


@pytest.mark.parametrize('width', [1120, 1420, 1920])
@pytest.mark.parametrize('mode', ['literature_grinding_force', 'literature_elastoplastic_single_pass'])
def test_advanced_fields_and_run_button_fit_content_width(window, width, mode):
    root, app = window
    app.analysis_mode_var.set(ANALYSIS_MODE_DEFINITIONS[mode].display_name)
    app._refresh_work_tree('分析')
    root.geometry(f'{width}x720')
    root.update()
    canvas = app.workbench_scroll_canvas
    left = canvas.winfo_rootx()
    right = left + canvas.winfo_width()
    for control in [*app.analysis_advanced_labels.values(), *app.analysis_advanced_entries.values(), app.analysis_run_button]:
        assert control.winfo_rootx() >= left
        assert control.winfo_rootx() + control.winfo_width() <= right, str(control)
        parent = control.master
        assert control.winfo_rootx() + control.winfo_width() <= parent.winfo_rootx() + parent.winfo_width(), str(control)
    assert min(w.winfo_width() for w in app.analysis_advanced_entries.values()) >= 100


def test_run_button_reachable_by_wheel_and_node_change_resets_view(window):
    root, app = window
    canvas = app.workbench_scroll_canvas
    label = app.analysis_advanced_labels['scan_base_position_count']
    button = app.analysis_run_button
    canvas.yview_moveto(0)
    for _ in range(150):
        root.update()
        y = button.winfo_rooty() - canvas.winfo_rooty()
        if 0 <= y and y + button.winfo_height() <= canvas.winfo_height():
            break
        label.event_generate('<MouseWheel>', delta=-120)
    else:
        pytest.fail('Run control cannot be reached with wheel')
    app._refresh_work_tree('工艺')
    root.update()
    assert canvas.yview()[0] == 0


def test_nested_viewer_canvas_and_toplevel_do_not_scroll_workbench(window):
    root, app = window
    canvas = app.workbench_scroll_canvas
    nested = tk.Canvas(app.workbench_scroll_content, height=20)
    nested.pack()
    dialog = tk.Toplevel(root)
    label = ttk.Label(dialog, text='another window')
    label.pack()
    root.update()
    canvas.yview_moveto(.25)
    before = canvas.yview()
    nested.event_generate('<MouseWheel>', delta=-120)
    label.event_generate('<MouseWheel>', delta=-120)
    root.update()
    assert canvas.yview() == before
    dialog.destroy()


def test_wheel_does_not_move_workbench_from_other_widgets(window):
    root, app = window
    canvas = app.workbench_scroll_canvas
    canvas.yview_moveto(.25)
    before = canvas.yview()
    app.work_tree.event_generate('<MouseWheel>', delta=-120)
    app.analysis_summary_text.event_generate('<MouseWheel>', delta=-120)
    root.update()
    assert canvas.yview() == before


def test_legacy_parameter_wheel_still_scrolls(window):
    root, app = window
    app.main_notebook.select(app.legacy_tab)
    root.update()
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    canvases = [w for w in descendants(app.legacy_tab) if isinstance(w, tk.Canvas) and w.bbox('all') and w.bbox('all')[3] > w.winfo_height()]
    assert canvases
    canvas = canvases[0]
    canvas.yview_moveto(0)
    canvas.event_generate('<MouseWheel>', delta=-120)
    root.update()
    assert canvas.yview()[0] > 0
