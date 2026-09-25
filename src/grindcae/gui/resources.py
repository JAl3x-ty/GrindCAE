"""Locate and apply packaged GUI resources without development-machine paths."""

from __future__ import annotations

from importlib import resources
import tkinter as tk


ICON_PACKAGE = "grindcae.gui.assets"
ICON_FILENAME = "grindcae.ico"


def apply_window_icon(root: tk.Misc) -> bool:
    """Apply the packaged Windows icon, leaving GUI construction recoverable."""

    try:
        icon_resource = resources.files(ICON_PACKAGE).joinpath(ICON_FILENAME)
        with resources.as_file(icon_resource) as icon_path:
            root.iconbitmap(default=str(icon_path))
    except Exception:
        return False
    return True
