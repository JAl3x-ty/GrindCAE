"""Exercise the new public workflow using the packaged, reproducible input."""
import json
from pathlib import Path
import sys
import tkinter as tk


def run_literature_smoke(output: Path):
    from grindcae import __version__
    from grindcae.literature_history import LiteratureHistoryCase, run_literature_history, read_result
    from grindcae.gui.app import GrindCaeApp
    from grindcae.gui.project import create_project, save_project, load_project

    root_dir = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
    case = LiteratureHistoryCase.from_mapping(json.loads((root_dir/'examples/literature_history_demo.json').read_text(encoding='utf-8')))
    output.mkdir(parents=True, exist_ok=False)
    result_dir = output/'result'
    run_literature_history(case, result_dir)
    result = read_result(result_dir)
    for route in result['routes'].values():
        assert route['final_unloaded']['external_force_norm_N'] == 0
    assert result['comparison']['maximum_component_projection_residual_N'] < 1e-8
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        app = GrindCaeApp(root)
        root.update()
        project_path = save_project(create_project('Portable 3.8.1 smoke'), output/'portable_smoke.gcae')
        app._set_workbench_project(load_project(project_path), project_path)
        app._refresh_work_tree('分析')
        root.update()
        assert not errors, errors
        app._session_marker_path.unlink(missing_ok=True)
    finally:
        for callback in root.tk.call('after', 'info'):
            root.tk.call('after', 'cancel', callback)
        root.destroy()
    report = dict(passed=True, version=__version__, frozen=bool(getattr(sys,'frozen',False)),
                  Tk_workbench_and_project_roundtrip=True, result_format=result['result_format'],
                  comparison=result['comparison'], force=result['literature_force'])
    (output/'literature_smoke_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
