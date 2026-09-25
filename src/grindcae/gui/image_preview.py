"""Reusable aspect-preserving PNG preview for workbench and analysis pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk


def fit_image_size(
    source_width: int,
    source_height: int,
    available_width: int,
    available_height: int,
) -> tuple[int, int]:
    values = (source_width, source_height, available_width, available_height)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("图片和预览区域尺寸必须是正整数。")
    scale = min(
        1.0,
        available_width / source_width,
        available_height / source_height,
    )
    return (
        max(1, math.floor(source_width * scale)),
        max(1, math.floor(source_height * scale)),
    )


@dataclass(frozen=True)
class ImageMetadata:
    path: str
    filename: str
    width_px: int
    height_px: int
    display_mode: str
    source: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class OriginalImageWindow(tk.Toplevel):
    """Show one PNG or one registered PNG sequence without file associations."""

    def __init__(
        self,
        master: tk.Misc,
        path: str | Path,
        *,
        sequence_paths: tuple[str | Path, ...] = (),
        sequence_index: int = 0,
    ) -> None:
        super().__init__(master)
        self.path = Path(path).expanduser().resolve()
        resolved_sequence = tuple(
            Path(item).expanduser().resolve() for item in sequence_paths
        )
        if len(resolved_sequence) > 1 and self.path in resolved_sequence:
            self.sequence_paths = resolved_sequence
            self.sequence_index = resolved_sequence.index(self.path)
        else:
            self.sequence_paths = (self.path,)
            self.sequence_index = 0
        if type(sequence_index) is int and 0 <= sequence_index < len(self.sequence_paths):
            if self.sequence_paths[sequence_index] == self.path:
                self.sequence_index = sequence_index
        self._source_image = self._load_png(self.path)
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_after_id: str | None = None
        self._display_mode = "fit"
        self.zoom_scale = 1.0
        self.displayed_image_size = self._source_image.size
        self._minimum_zoom = 0.05
        self._maximum_zoom = 4.0

        self.title(f"查看原图 - {self.path.name}")
        screen_width = max(800, self.winfo_screenwidth())
        screen_height = max(600, self.winfo_screenheight())
        width = min(1200, max(720, screen_width - 120))
        height = min(820, max(520, screen_height - 160))
        self.geometry(f"{width}x{height}")
        self.minsize(640, 420)
        self.transient(master.winfo_toplevel())

        toolbar = ttk.Frame(self, padding=(6, 6, 6, 0))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="适应窗口", command=self.fit_to_window).pack(side="left")
        ttk.Button(toolbar, text="原始大小", command=self.show_original_size).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(toolbar, text="缩小", command=lambda: self.zoom_at(0.8)).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(toolbar, text="放大", command=lambda: self.zoom_at(1.25)).pack(
            side="left", padx=(4, 0)
        )
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(toolbar, textvariable=self.zoom_var, width=7, anchor="center").pack(
            side="left", padx=(6, 0)
        )
        self.image_size_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.image_size_var).pack(side="right")

        self.sequence_controls = ttk.Frame(self, padding=(6, 4, 6, 0))
        self.previous_button = ttk.Button(
            self.sequence_controls, text="上一张", command=self.previous_image
        )
        self.previous_button.pack(side="left")
        self.sequence_scale_var = tk.DoubleVar(value=float(self.sequence_index))
        self.sequence_scale = ttk.Scale(
            self.sequence_controls,
            from_=0,
            to=max(0, len(self.sequence_paths) - 1),
            variable=self.sequence_scale_var,
            command=self.select_sequence_image,
        )
        self.sequence_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.next_button = ttk.Button(
            self.sequence_controls, text="下一张", command=self.next_image
        )
        self.next_button.pack(side="left")
        self.sequence_status_var = tk.StringVar()
        ttk.Label(
            self.sequence_controls,
            textvariable=self.sequence_status_var,
            width=16,
            anchor="e",
        ).pack(side="left", padx=(8, 0))
        if len(self.sequence_paths) > 1:
            self.sequence_controls.pack(fill="x")

        canvas_host = ttk.Frame(self, padding=6)
        canvas_host.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_host,
            background="#202020",
            highlightthickness=0,
        )
        horizontal = ttk.Scrollbar(
            canvas_host, orient="horizontal", command=self.canvas.xview
        )
        vertical = ttk.Scrollbar(
            canvas_host, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(
            xscrollcommand=horizontal.set,
            yscrollcommand=vertical.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        canvas_host.rowconfigure(0, weight=1)
        canvas_host.columnconfigure(0, weight=1)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", self._schedule_fit)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self.zoom_at(1.25, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self.zoom_at(0.8, event.x, event.y))
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._update_image_metadata()
        self.after_idle(self.fit_to_window)

    @staticmethod
    def _load_png(path: Path) -> Image.Image:
        image = Image.open(path)
        try:
            image.load()
            if image.format != "PNG":
                raise ValueError("当前原图查看器仅支持 PNG 图片。")
            return image.convert("RGBA")
        finally:
            image.close()

    def _update_image_metadata(self) -> None:
        self.title(f"查看原图 - {self.path.name}")
        self.image_size_var.set(
            f"{self._source_image.width} × {self._source_image.height} px"
        )
        self.sequence_scale_var.set(float(self.sequence_index))
        self.sequence_status_var.set(
            f"位置 {self.sequence_index + 1} / {len(self.sequence_paths)}"
        )
        self.previous_button.configure(
            state="normal" if self.sequence_index > 0 else "disabled"
        )
        self.next_button.configure(
            state=(
                "normal"
                if self.sequence_index < len(self.sequence_paths) - 1
                else "disabled"
            )
        )

    def _show_sequence_index(self, index: int) -> bool:
        selected = min(len(self.sequence_paths) - 1, max(0, int(index)))
        if selected == self.sequence_index:
            self._update_image_metadata()
            return True
        path = self.sequence_paths[selected]
        try:
            source_image = self._load_png(path)
        except (OSError, ValueError):
            self.sequence_scale_var.set(float(self.sequence_index))
            return False
        self._source_image.close()
        self._source_image = source_image
        self.path = path
        self.sequence_index = selected
        self._update_image_metadata()
        if self._display_mode == "original":
            self.show_original_size()
        elif self._display_mode == "zoom":
            self._render_scale(self.zoom_scale)
        else:
            self.fit_to_window()
        return True

    def select_sequence_image(self, value: str | float) -> bool:
        return self._show_sequence_index(int(round(float(value))))

    def previous_image(self) -> bool:
        return self._show_sequence_index(self.sequence_index - 1)

    def next_image(self) -> bool:
        return self._show_sequence_index(self.sequence_index + 1)

    def _schedule_fit(self, _event=None) -> None:
        if self._display_mode != "fit":
            return
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(120, self.fit_to_window)

    def _display(self, image: Image.Image) -> None:
        self.displayed_image_size = image.size
        self._photo = ImageTk.PhotoImage(image, master=self)
        self.canvas.itemconfigure(self.image_item, image=self._photo)
        available_width = max(1, self.canvas.winfo_width())
        available_height = max(1, self.canvas.winfo_height())
        x_value = max(0, (available_width - image.width) // 2)
        y_value = max(0, (available_height - image.height) // 2)
        self.canvas.coords(self.image_item, x_value, y_value)
        self.canvas.configure(
            scrollregion=(
                0,
                0,
                max(available_width, image.width),
                max(available_height, image.height),
            )
        )
        self.zoom_var.set(f"{self.zoom_scale * 100:.0f}%")

    def _render_scale(self, scale: float) -> None:
        width = max(1, round(self._source_image.width * scale))
        height = max(1, round(self._source_image.height * scale))
        if (width, height) == self._source_image.size:
            displayed = self._source_image
        else:
            displayed = self._source_image.resize(
                (width, height), Image.Resampling.LANCZOS
            )
        self.zoom_scale = scale
        self._display(displayed)

    def fit_to_window(self) -> None:
        self._resize_after_id = None
        self._display_mode = "fit"
        available_width = max(1, self.canvas.winfo_width() - 6)
        available_height = max(1, self.canvas.winfo_height() - 6)
        width, height = fit_image_size(
            self._source_image.width,
            self._source_image.height,
            available_width,
            available_height,
        )
        self._render_scale(width / self._source_image.width)
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def show_original_size(self) -> None:
        self._display_mode = "original"
        self._render_scale(1.0)
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def zoom_at(
        self,
        factor: float,
        canvas_x: int | None = None,
        canvas_y: int | None = None,
    ) -> None:
        if not math.isfinite(factor) or factor <= 0.0:
            return
        pointer_x = self.canvas.winfo_width() // 2 if canvas_x is None else canvas_x
        pointer_y = self.canvas.winfo_height() // 2 if canvas_y is None else canvas_y
        old_scale = self.zoom_scale
        new_scale = min(
            self._maximum_zoom,
            max(self._minimum_zoom, old_scale * factor),
        )
        if math.isclose(new_scale, old_scale, rel_tol=0.0, abs_tol=1.0e-12):
            return
        image_x, image_y = self.canvas.coords(self.image_item)
        source_x = (self.canvas.canvasx(pointer_x) - image_x) / old_scale
        source_y = (self.canvas.canvasy(pointer_y) - image_y) / old_scale
        self._display_mode = "zoom"
        self._render_scale(new_scale)
        self.update_idletasks()
        new_image_x, new_image_y = self.canvas.coords(self.image_item)
        target_left = new_image_x + source_x * new_scale - pointer_x
        target_top = new_image_y + source_y * new_scale - pointer_y
        region = tuple(float(value) for value in self.canvas.cget("scrollregion").split())
        region_width = max(1.0, region[2] - region[0])
        region_height = max(1.0, region[3] - region[1])
        self.canvas.xview_moveto(max(0.0, target_left / region_width))
        self.canvas.yview_moveto(max(0.0, target_top / region_height))

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        self.zoom_at(1.25 if event.delta > 0 else 0.8, event.x, event.y)
        return "break"

    def begin_pan(self, x: int, y: int) -> None:
        self.canvas.scan_mark(x, y)
        self.canvas.configure(cursor="fleur")

    def pan_to(self, x: int, y: int) -> None:
        self.canvas.scan_dragto(x, y, gain=1)

    def end_pan(self) -> None:
        self.canvas.configure(cursor="")

    def _on_pan_start(self, event: tk.Event) -> str:
        self.begin_pan(event.x, event.y)
        return "break"

    def _on_pan_motion(self, event: tk.Event) -> str:
        self.pan_to(event.x, event.y)
        return "break"

    def _on_pan_end(self, _event: tk.Event) -> str:
        self.end_pan()
        return "break"

    def destroy(self) -> None:
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
            self._resize_after_id = None
        self._photo = None
        self._source_image.close()
        super().destroy()


def reveal_in_file_manager(path: str | Path) -> None:
    """Open Explorer and select one existing result file."""

    resolved = Path(path).expanduser().resolve()
    if os.name == "nt":
        subprocess.Popen(["explorer.exe", f"/select,{resolved}"])
        return
    os.startfile(str(resolved.parent))


class ImagePreview(ttk.Frame):
    """Display one real PNG with fit-to-window and original-file actions."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        source: str = "result",
        error_handler: Callable[[str], None] | None = None,
        open_handler: Callable[[Path], None] | None = None,
        reveal_handler: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.source = source
        self.error_handler = error_handler or (lambda _message: None)
        self.open_handler = open_handler or self._system_open
        self.reveal_handler = reveal_handler or reveal_in_file_manager
        self.filename_var = tk.StringVar(value="尚无图片")
        self.metadata_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")
        self._path: Path | None = None
        self._metadata: ImageMetadata | None = None
        self._source_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_after_id: str | None = None
        self._sequence_paths: tuple[Path, ...] = ()
        self._sequence_index = 0

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 4))
        self.fit_button = ttk.Button(
            toolbar, text="适应窗口", command=self.fit_to_window, state="disabled"
        )
        self.fit_button.pack(side="left")
        self.open_original_button = ttk.Button(
            toolbar,
            text="查看原图",
            command=self.open_original,
            state="disabled",
        )
        self.open_original_button.pack(side="left", padx=(4, 0))
        self.locate_original_button = ttk.Button(
            toolbar,
            text="在文件夹中定位",
            command=self.locate_original,
            state="disabled",
        )
        self.locate_original_button.pack(side="left", padx=(4, 0))
        ttk.Label(toolbar, textvariable=self.filename_var).pack(
            side="left", padx=(10, 0)
        )
        ttk.Label(toolbar, textvariable=self.metadata_var).pack(
            side="right", padx=(8, 0)
        )

        canvas_host = ttk.Frame(self)
        canvas_host.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_host,
            background="#F7F7F7",
            highlightthickness=1,
            highlightbackground="#C8C8C8",
        )
        horizontal = ttk.Scrollbar(
            canvas_host, orient="horizontal", command=self.canvas.xview
        )
        vertical = ttk.Scrollbar(
            canvas_host, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(
            xscrollcommand=horizontal.set, yscrollcommand=vertical.set
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        canvas_host.rowconfigure(0, weight=1)
        canvas_host.columnconfigure(0, weight=1)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.message_item = self.canvas.create_text(
            10,
            10,
            anchor="nw",
            text="尚无图片。",
            fill="#555555",
            font=("Microsoft YaHei UI", 10),
        )
        ttk.Entry(self, textvariable=self.path_var, state="readonly").pack(
            fill="x", pady=(4, 0)
        )
        self.canvas.bind("<Configure>", self._schedule_fit)

    def _system_open(self, path: Path) -> None:
        os.startfile(str(path.resolve()))

    def _error(self, message: str) -> None:
        self.error_handler(message)

    def set_image(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        sequence_paths: tuple[str | Path, ...] = (),
        sequence_index: int = 0,
    ) -> ImageMetadata:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            self.clear(f"图片文件不存在：{resolved}")
            self._error(f"图片文件不存在：{resolved}")
            raise FileNotFoundError(resolved)
        try:
            image = Image.open(resolved)
            image.load()
        except (OSError, ValueError) as exc:
            self.clear(f"无法读取图片：{resolved.name}")
            self._error(f"无法读取图片 {resolved}：{exc}")
            raise
        if image.format != "PNG":
            image.close()
            self.clear(f"当前预览仅支持真实 PNG：{resolved.name}")
            raise ValueError("当前预览仅支持 PNG 图片。")
        self._source_image = image.convert("RGBA")
        image.close()
        self._path = resolved
        resolved_sequence = tuple(
            Path(item).expanduser().resolve() for item in sequence_paths
        )
        if len(resolved_sequence) > 1 and resolved in resolved_sequence:
            self._sequence_paths = resolved_sequence
            self._sequence_index = resolved_sequence.index(resolved)
            if type(sequence_index) is int and 0 <= sequence_index < len(resolved_sequence):
                if resolved_sequence[sequence_index] == resolved:
                    self._sequence_index = sequence_index
        else:
            self._sequence_paths = ()
            self._sequence_index = 0
        self._metadata = ImageMetadata(
            path=str(resolved),
            filename=resolved.name,
            width_px=self._source_image.width,
            height_px=self._source_image.height,
            display_mode="fit",
            source=source or self.source,
        )
        self.filename_var.set(resolved.name)
        self.metadata_var.set(
            f"原始尺寸：{self._source_image.width} × {self._source_image.height} px"
        )
        self.path_var.set(str(resolved))
        self.canvas.itemconfigure(self.message_item, text="")
        self.fit_button.configure(state="normal")
        self.open_original_button.configure(state="normal")
        self.locate_original_button.configure(state="normal")
        self.fit_to_window()
        return self._metadata

    def _schedule_fit(self, _event=None) -> None:
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(120, self.fit_to_window)

    def fit_to_window(self) -> None:
        self._resize_after_id = None
        if self._source_image is None:
            return
        available_width = max(1, self.canvas.winfo_width() - 6)
        available_height = max(1, self.canvas.winfo_height() - 6)
        display_width, display_height = fit_image_size(
            self._source_image.width,
            self._source_image.height,
            available_width,
            available_height,
        )
        displayed = self._source_image.resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )
        self._photo = ImageTk.PhotoImage(displayed, master=self)
        self.canvas.itemconfigure(self.image_item, image=self._photo)
        x_value = max(0, (available_width - display_width) // 2)
        y_value = max(0, (available_height - display_height) // 2)
        self.canvas.coords(self.image_item, x_value, y_value)
        self.canvas.configure(
            scrollregion=(
                0,
                0,
                max(available_width, display_width),
                max(available_height, display_height),
            )
        )

    def open_original(self, path: str | Path | None = None) -> bool:
        target = Path(path).expanduser().resolve() if path is not None else self._path
        if target is None or not target.is_file():
            self.open_original_button.configure(state="disabled")
            self._error(f"图片文件不存在：{target or '尚未选择图片'}")
            return False
        try:
            OriginalImageWindow(
                self.winfo_toplevel(),
                target,
                sequence_paths=self._sequence_paths,
                sequence_index=self._sequence_index,
            )
        except OSError as exc:
            self._error(f"无法打开原始图片 {target}：{exc}")
            return False
        return True

    def locate_original(self) -> bool:
        target = self._path
        if target is None or not target.is_file():
            self.locate_original_button.configure(state="disabled")
            self._error(f"图片文件不存在：{target or '尚未选择图片'}")
            return False
        try:
            self.reveal_handler(target)
        except OSError as exc:
            self._error(f"无法在文件夹中定位图片 {target}：{exc}")
            return False
        return True

    def clear(self, message: str = "尚无图片。") -> None:
        if self._source_image is not None:
            self._source_image.close()
        self._path = None
        self._metadata = None
        self._source_image = None
        self._photo = None
        self._sequence_paths = ()
        self._sequence_index = 0
        self.canvas.itemconfigure(self.image_item, image="")
        self.canvas.itemconfigure(self.message_item, text=message)
        self.filename_var.set("尚无图片")
        self.metadata_var.set("")
        self.path_var.set("")
        self.fit_button.configure(state="disabled")
        self.open_original_button.configure(state="disabled")
        self.locate_original_button.configure(state="disabled")

    def image_metadata(self) -> ImageMetadata | None:
        return self._metadata
