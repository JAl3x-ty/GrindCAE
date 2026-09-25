"""Manual evidence helper: actual Tk window rendering and a real GUI run."""
from pathlib import Path
import sys
import time
import tkinter as tk
from PIL import ImageGrab

root_path=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root_path/'src'))
from grindcae380.desktop import Workbench

root=tk.Tk();app=Workbench(root,output_root=root_path/'outputs'/'desktop_acceptance')
root.geometry('1180x820+70+60');root.update();root.lift()
root.attributes('-topmost',True);root.update();time.sleep(.5)

def capture(name):
    root.update();time.sleep(.3)
    ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),
                        root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(root_path/'evidence'/name)

capture('desktop_parameters.png')
app.start_calculation()
deadline=time.monotonic()+40
while app.busy and time.monotonic()<deadline:root.update();time.sleep(.02)
if app.busy or app.last_error:raise RuntimeError(app.last_error or 'GUI calculation timed out')
capture('desktop_result.png')
print(app.result_directory)
root.destroy()
