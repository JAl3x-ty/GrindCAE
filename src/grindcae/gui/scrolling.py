"""Wheel routing scoped to one canvas, preserving nested widget behavior."""
import tkinter as tk


def bind_canvas_wheel(canvas: tk.Canvas) -> None:
    top = canvas.winfo_toplevel()

    def scroll(event):
        if not canvas.winfo_exists() or not canvas.winfo_ismapped():
            return
        widget = event.widget
        while widget is not canvas:
            # Text, lists and selectors own their wheel events. Other canvases
            # include the image viewer, whose wheel controls zoom.
            if widget is None or widget.winfo_class() in {
                'Text', 'Listbox', 'Treeview', 'TCombobox', 'Spinbox', 'TSpinbox', 'Canvas'
            }:
                return
            widget = getattr(widget, 'master', None)
        if canvas.yview() == (0.0, 1.0):
            return
        delta = getattr(event, 'delta', 0)
        number = getattr(event, 'num', None)
        units = (-max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)) if delta else (-1 if number == 4 else 1 if number == 5 else 0)
        if units:
            canvas.yview_scroll(int(units), 'units')
            return 'break'

    bindings = [(event, top.bind(event, scroll, add='+'))
                for event in ('<MouseWheel>', '<Button-4>', '<Button-5>')]

    def cleanup(event):
        if event.widget is canvas:
            for sequence, identifier in bindings:
                top.unbind(sequence, identifier)

    canvas.bind('<Destroy>', cleanup, add='+')
