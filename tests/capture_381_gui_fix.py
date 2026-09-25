"""Capture the repaired layout in a separate window; never run a solver."""
import json
from pathlib import Path
import sys
import tkinter as tk
import time

from PIL import ImageGrab

folder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(folder / 'src'))
from grindcae.gui.app import GrindCaeApp
from grindcae.gui.project import load_project

root = tk.Tk()
app = GrindCaeApp(root)
path = Path(json.loads((folder/'evidence/381_gui_acceptance.json').read_text(encoding='utf-8'))['project'])
app._set_workbench_project(load_project(path), path)
root.attributes('-topmost', True)
for width in (1120, 1420):
    root.geometry(f'{width}x720+15+15')
    app._refresh_work_tree('分析')
    root.update()
    canvas = app.workbench_scroll_canvas
    label = app.analysis_advanced_labels['single_grain_type']
    offset = label.winfo_rooty() - app.workbench_scroll_content.winfo_rooty() - 90
    canvas.yview_moveto(max(0, offset) / app.workbench_scroll_content.winfo_height())
    root.update()
    time.sleep(.3)
    ImageGrab.grab(bbox=(root.winfo_rootx(), root.winfo_rooty(), root.winfo_rootx()+root.winfo_width(), root.winfo_rooty()+root.winfo_height())).save(folder/f'evidence/381_gui_fixed_{width}.png')
app._session_marker_path.unlink(missing_ok=True)
for callback in root.tk.call('after', 'info'):
    root.tk.call('after', 'cancel', callback)
root.destroy()
print('Captured repaired 1120/1420 layouts; existing project only read.')
