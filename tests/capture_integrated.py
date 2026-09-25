"""Drive the real main workbench through its run action for visual evidence."""
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch
from PIL import ImageGrab

folder=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(folder/'src'))
from grindcae.gui.app import GrindCaeApp
from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS
from grindcae.gui.project import save_project
from grindcae380.desktop import default_case

root=tk.Tk();app=GrindCaeApp(root)
root.geometry('1420x900+30+30');root.attributes('-topmost',True)
app.workbench.project.project_name='文献磨削力集成示例'
app.project_name_var.set('文献磨削力集成示例')
app.analysis_mode_var.set(ANALYSIS_MODE_DEFINITIONS['literature_grinding_force'].display_name)
app._literature_case_payload=default_case(True).to_dict()
app._save_analysis_configuration()
target=folder/'outputs'/('integrated_acceptance_'+str(int(time.time())))
target.mkdir(exist_ok=False)
app.project_path=save_project(app.workbench.project,target/'literature.gcae')
app.project_path_var.set(str(app.project_path))
app._accept_saved_snapshot()
app.analysis_output_var.set(str(target/'result'))
app.main_notebook.select(app.workbench_tab);app._refresh_work_tree('分析')
root.update()
with patch('grindcae.gui.app.messagebox.askyesnocancel',return_value=False):
    app._start_workbench_analysis()
deadline=time.monotonic()+40
while (app.analysis_runner.is_running or app.workbench.project.latest_analysis_result is None) and time.monotonic()<deadline:
    root.update();time.sleep(.03)
if app.workbench.project.latest_analysis_result is None:raise RuntimeError(app.analysis_status_var.get())
save_project(app.workbench.project,app.project_path)
app._accept_saved_snapshot()
app._refresh_work_tree('结果')
root.update();time.sleep(.4)
ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(folder/'evidence'/'integrated_result_center.png')
app._refresh_work_tree('分析');root.update();time.sleep(.4)
ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(folder/'evidence'/'integrated_analysis.png')
print(app.project_path)
(folder/'evidence'/'integrated_example_path.txt').write_text(str(app.project_path),encoding='utf-8')
app._session_marker_path.unlink(missing_ok=True)
root.destroy()
