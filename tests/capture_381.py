"""Real Tk runner acceptance, saved project, export and archive round trip."""
import json
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch
from PIL import ImageGrab

folder=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(folder/'src'))
from literature_history_fixture import project_and_settings
from grindcae.gui.app import GrindCaeApp
from grindcae.gui.analysis_workspace import AnalysisWorkspace
from grindcae.gui.project import save_project,load_project
from grindcae.gui.result_catalog import ResultCatalog
from grindcae.gui.origin_export import export_origin_data
from grindcae.gui.project_archive import export_project_archive,import_project_archive
from grindcae.literature_history import MODE

root=tk.Tk();app=GrindCaeApp(root)
root.geometry('1420x900+30+30');root.attributes('-topmost',True)
target=folder/'outputs'/('acceptance_381_'+str(int(time.time())))
target.mkdir()
project,settings=project_and_settings()
AnalysisWorkspace(project,MODE,settings,str(target/'result')).save_configuration()
path=save_project(project,target/'验收示例.gcae')
app._set_workbench_project(load_project(path),path)
app.main_notebook.select(app.workbench_tab);app._refresh_work_tree('分析');root.update()
errors=[]
with patch('grindcae.gui.app.messagebox.askyesnocancel',return_value=False), patch('grindcae.gui.app.messagebox.showerror',side_effect=lambda *args,**kw:errors.append(args)):
    app._start_workbench_analysis()
    deadline=time.monotonic()+180
    while (app.analysis_runner.is_running or app._active_analysis_built is not None) and time.monotonic()<deadline:
        root.update();time.sleep(.02)
if errors:raise RuntimeError(errors)
record=app.workbench.project.latest_analysis_result
if record is None or record.analysis_type!=MODE:raise RuntimeError(app.analysis_status_var.get())
save_project(app.workbench.project,path);app._accept_saved_snapshot()
reopened=load_project(path)
catalog=ResultCatalog().read(reopened.latest_analysis_result)
origin=export_origin_data(catalog,target/'origin')
archive=export_project_archive(path,target/'验收归档.gcae-archive')
imported=import_project_archive(archive,target/'archive_restored')
archive_catalog=ResultCatalog().read(load_project(imported.project_path).latest_analysis_result)
assert archive_catalog.published.mode==MODE
for node,name in [('结果','381_result_center.png'),('分析','381_analysis.png')]:
    app._refresh_work_tree(node);root.update();time.sleep(.4)
    ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(folder/'evidence'/name)
report=dict(project=str(path),output=str(target/'result'),archive=str(archive),origin_tables=len(origin.table_paths),
    restored_project=str(imported.project_path),progress=app.analysis_progress_detail_var.get(),summary=catalog.published.summary)
(folder/'evidence'/'381_gui_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='summary'},ensure_ascii=False),flush=True)
app._session_marker_path.unlink(missing_ok=True)
for callback in root.tk.call('after','info'):root.after_cancel(callback)
root.destroy()
