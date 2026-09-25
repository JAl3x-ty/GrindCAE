"""Tk view for browsing classified, already-published analysis results."""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .image_preview import ImagePreview
from .origin_export import OriginExportError, OriginExportResult, export_origin_data
from .result_catalog import RESULT_CATEGORY_IDS, ResultArtifactItem, format_file_size
from .result_center import (
    FieldEvolutionBrowserState,
    MechanismFieldEvolutionBrowserState,
    ResultCenterEntry,
    ResultCenterState,
)


class ResultCenterView(ttk.Frame):
    """Independent result-center editor; it performs file browsing only."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        open_handler: Callable[[Path], None] | None = None,
        origin_export_handler: Callable[
            [object, str | Path], OriginExportResult
        ] | None = None,
        directory_dialog_handler: Callable[..., str] | None = None,
    ) -> None:
        super().__init__(master)
        self._open_handler = open_handler or self._system_open
        self._origin_export_handler = origin_export_handler or export_origin_data
        self._directory_dialog_handler = (
            directory_dialog_handler or filedialog.askdirectory
        )
        self._state = ResultCenterState((), None)
        self.selected_result_id: str | None = None
        self._selected_entry: ResultCenterEntry | None = None
        self._data_items: dict[str, ResultArtifactItem] = {}
        self._image_items: tuple[ResultArtifactItem, ...] = ()
        self._active_category_id = "summary"
        self.field_evolution_browser = FieldEvolutionBrowserState()
        self.mechanism_field_evolution_browser = MechanismFieldEvolutionBrowserState()
        self._build()

    @staticmethod
    def _system_open(path: Path) -> None:
        os.startfile(str(path.resolve()))

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        history_host = ttk.LabelFrame(self, text="正式分析历史", padding=6)
        history_host.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        history_host.rowconfigure(0, weight=1)
        history_host.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(
            history_host,
            columns=("completed", "mode", "status"),
            show="headings",
            selectmode="browse",
            height=18,
        )
        for column, title, width in (
            ("completed", "完成时间", 145), ("mode", "分析模式", 135), ("status", "状态", 82),
        ):
            self.history_tree.heading(column, text=title)
            self.history_tree.column(column, width=width, stretch=column == "mode")
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_selected)
        history_buttons = ttk.Frame(history_host)
        history_buttons.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        history_buttons.columnconfigure(0, weight=1)
        history_buttons.columnconfigure(1, weight=1)
        ttk.Button(history_buttons, text="刷新结果", command=self._request_refresh).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(history_buttons, text="打开结果目录", command=self.open_selected_result_directory).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.refresh_handler: Callable[[], None] | None = None
        self.relink_handler: Callable[[str], None] | None = None
        self.relink_result_button = ttk.Button(
            history_buttons, text="重新定位结果", command=self._request_relink
        )
        self.relink_result_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        detail = ttk.Frame(self)
        detail.grid(row=0, column=1, sticky="nsew")
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(2, weight=1)
        self.header_var = tk.StringVar(value="暂无正式分析结果")
        self.path_var = tk.StringVar(value="")
        ttk.Label(detail, textvariable=self.header_var, font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(detail, textvariable=self.path_var, wraplength=850, justify="left").grid(row=1, column=0, sticky="ew", pady=(3, 6))

        self.notebook = ttk.Notebook(detail)
        self.notebook.grid(row=2, column=0, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_category_selected)
        self.category_frames: dict[str, ttk.Frame] = {}
        for category_id, title in zip(RESULT_CATEGORY_IDS, ("结果摘要", "曲线与历程", "云图与场量", "网格与变形", "数据文件")):
            frame = ttk.Frame(self.notebook, padding=6)
            self.notebook.add(frame, text=title)
            self.category_frames[category_id] = frame

        summary_frame = self.category_frames["summary"]
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_frame, wrap="word", state="disabled", font=("Microsoft YaHei UI", 10))
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        self.image_name_var = tk.StringVar(value="")
        self.image_note_var = tk.StringVar(value="")
        self.image_choices: dict[str, ttk.Combobox] = {}
        self.image_previews: dict[str, ImagePreview] = {}
        self.image_note_labels: dict[str, ttk.Label] = {}
        for category_id in ("history", "fields", "mesh"):
            image_host = self.category_frames[category_id]
            choice = ttk.Combobox(image_host, textvariable=self.image_name_var, state="readonly")
            choice.pack(fill="x")
            choice.bind("<<ComboboxSelected>>", self._on_image_selected)
            preview = ImagePreview(image_host, source="result-center", error_handler=self._show_error, open_handler=self._open_handler)
            preview.pack(fill="both", expand=True, pady=(6, 0))
            note_label = ttk.Label(image_host, textvariable=self.image_note_var, wraplength=820, justify="left")
            note_label.pack(fill="x", pady=(5, 0))
            self.image_choices[category_id] = choice
            self.image_previews[category_id] = preview
            self.image_note_labels[category_id] = note_label
        self.image_choice = self.image_choices["history"]
        self.image_preview = self.image_previews["history"]

        fields_frame = self.category_frames["fields"]
        self.field_view_mode_var = tk.StringVar(value="单程场演化")
        self.field_view_mode_choice = ttk.Combobox(
            fields_frame,
            textvariable=self.field_view_mode_var,
            values=("单程场演化", "已有云图"),
            state="readonly",
        )
        self.field_view_mode_choice.bind("<<ComboboxSelected>>", self._on_field_view_mode_selected)
        self.field_evolution_host = ttk.Frame(fields_frame)
        controls = ttk.Frame(self.field_evolution_host)
        controls.pack(fill="x")
        self.previous_position_button = ttk.Button(
            controls, text="上一位置", command=self.previous_field_evolution_frame
        )
        self.previous_position_button.pack(side="left")
        self.field_evolution_scale_var = tk.DoubleVar(value=0.0)
        self.field_evolution_scale = ttk.Scale(
            controls,
            variable=self.field_evolution_scale_var,
            from_=0,
            to=0,
            command=self.select_field_evolution_frame,
        )
        self.field_evolution_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.next_position_button = ttk.Button(
            controls, text="下一位置", command=self.next_field_evolution_frame
        )
        self.next_position_button.pack(side="left")
        self.field_evolution_metadata_var = tk.StringVar(value="")
        ttk.Label(
            self.field_evolution_host,
            textvariable=self.field_evolution_metadata_var,
            justify="left",
            wraplength=820,
        ).pack(fill="x", pady=(6, 0))
        self.field_evolution_preview = ImagePreview(
            self.field_evolution_host,
            source="result-center-field-evolution",
            error_handler=self._show_error,
            open_handler=self._open_handler,
        )
        self.field_evolution_preview.pack(fill="both", expand=True, pady=(6, 0))
        file_buttons = ttk.Frame(self.field_evolution_host)
        file_buttons.pack(fill="x", pady=(6, 0))
        for column in range(3):
            file_buttons.columnconfigure(column, weight=1)
        self.open_current_vtu_button = ttk.Button(
            file_buttons, text="打开当前 VTU", command=self.open_current_field_vtu
        )
        self.open_current_vtu_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.open_sequence_button = ttk.Button(
            file_buttons, text="打开完整序列", command=self.open_field_evolution_pvd
        )
        self.open_sequence_button.grid(row=0, column=1, sticky="ew", padx=3)
        self.open_sequence_directory_button = ttk.Button(
            file_buttons, text="打开场演化目录", command=self.open_field_evolution_directory
        )
        self.open_sequence_directory_button.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        self.mechanism_field_view_mode_var = tk.StringVar(value="基准路线")
        self.mechanism_field_view_mode_choice = ttk.Combobox(
            fields_frame,
            textvariable=self.mechanism_field_view_mode_var,
            values=("基准路线", "机制路线", "差值对比", "基准／机制并排"),
            state="readonly",
        )
        self.mechanism_field_view_mode_choice.bind(
            "<<ComboboxSelected>>", self._on_mechanism_field_view_mode_selected
        )
        self.mechanism_existing_comparison_button = ttk.Button(
            fields_frame,
            text="查看已有对比图",
            command=self._show_existing_mechanism_comparison,
        )
        self.mechanism_field_evolution_host = ttk.Frame(fields_frame)
        mechanism_controls = ttk.Frame(self.mechanism_field_evolution_host)
        mechanism_controls.pack(fill="x")
        self.mechanism_previous_position_button = ttk.Button(
            mechanism_controls,
            text="上一位置",
            command=self.previous_mechanism_field_evolution_frame,
        )
        self.mechanism_previous_position_button.pack(side="left")
        self.mechanism_field_evolution_scale_var = tk.DoubleVar(value=0.0)
        self.mechanism_field_evolution_scale = ttk.Scale(
            mechanism_controls,
            variable=self.mechanism_field_evolution_scale_var,
            from_=0,
            to=0,
            command=self.select_mechanism_field_evolution_frame,
        )
        self.mechanism_field_evolution_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.mechanism_next_position_button = ttk.Button(
            mechanism_controls,
            text="下一位置",
            command=self.next_mechanism_field_evolution_frame,
        )
        self.mechanism_next_position_button.pack(side="left")
        self.mechanism_field_evolution_metadata_var = tk.StringVar(value="")
        ttk.Label(
            self.mechanism_field_evolution_host,
            textvariable=self.mechanism_field_evolution_metadata_var,
            justify="left",
            wraplength=820,
        ).pack(fill="x", pady=(6, 0))
        mechanism_previews = ttk.Frame(self.mechanism_field_evolution_host)
        mechanism_previews.pack(fill="both", expand=True, pady=(6, 0))
        mechanism_previews.columnconfigure(0, weight=1)
        mechanism_previews.columnconfigure(1, weight=1)
        mechanism_previews.rowconfigure(0, weight=1)
        self.mechanism_field_evolution_previews = (
            ImagePreview(mechanism_previews, source="result-center-mechanism-field", error_handler=self._show_error, open_handler=self._open_handler),
            ImagePreview(mechanism_previews, source="result-center-mechanism-field", error_handler=self._show_error, open_handler=self._open_handler),
        )
        self.mechanism_field_evolution_previews[0].grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.mechanism_field_evolution_previews[1].grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        mechanism_files = ttk.Frame(self.mechanism_field_evolution_host)
        mechanism_files.pack(fill="x", pady=(6, 0))
        for column in range(7):
            mechanism_files.columnconfigure(column, weight=1)
        self.mechanism_field_file_buttons: dict[str, ttk.Button] = {}
        for column, (key, text, command) in enumerate((
            ("baseline_vtu", "打开当前基准 VTU", lambda: self.open_current_mechanism_field_vtu("baseline")),
            ("mechanism_vtu", "打开当前机制 VTU", lambda: self.open_current_mechanism_field_vtu("mechanism")),
            ("difference_vtu", "打开当前差值 VTU", lambda: self.open_current_mechanism_field_vtu("difference")),
            ("baseline_pvd", "打开基准 PVD", lambda: self.open_mechanism_field_pvd("baseline")),
            ("mechanism_pvd", "打开机制 PVD", lambda: self.open_mechanism_field_pvd("mechanism")),
            ("difference_pvd", "打开差值 PVD", lambda: self.open_mechanism_field_pvd("difference")),
            ("directory", "打开场演化目录", self.open_mechanism_field_evolution_directory),
        )):
            button = ttk.Button(mechanism_files, text=text, command=command)
            button.grid(
                row=0, column=column, sticky="ew", padx=2
            )
            self.mechanism_field_file_buttons[key] = button

        data_frame = self.category_frames["data"]
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        self.data_tree = ttk.Treeview(data_frame, columns=("name", "filename", "type", "size", "status"), show="headings", selectmode="browse")
        for column, title, width in (
            ("name", "中文名称", 180), ("filename", "文件名", 210), ("type", "类型", 65),
            ("size", "大小", 85), ("status", "状态", 80),
        ):
            self.data_tree.heading(column, text=title)
            self.data_tree.column(column, width=width, stretch=column in ("name", "filename"))
        self.data_tree.grid(row=0, column=0, columnspan=3, sticky="nsew")
        ttk.Button(data_frame, text="打开文件", command=self._open_selected_file).grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=(6, 0))
        ttk.Button(data_frame, text="打开所在目录", command=self._open_selected_directory).grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=(6, 0))
        self.export_origin_button = ttk.Button(
            data_frame,
            text="导出 Origin 数据…",
            command=self.export_selected_origin_data,
            state="disabled",
        )
        self.export_origin_button.grid(
            row=1, column=2, sticky="ew", padx=(3, 0), pady=(6, 0)
        )

    def set_state(self, state: ResultCenterState) -> None:
        self._state = state
        self.history_tree.delete(*self.history_tree.get_children(""))
        for entry in state.entries:
            self.history_tree.insert("", "end", iid=entry.result_id, values=(entry.created_at, entry.display_name, entry.status))
        if state.selected_result_id:
            self.select_result(state.selected_result_id)
        else:
            self._show_message("暂无正式分析结果。")

    def select_result(self, result_id: str) -> None:
        entry = self._state.entry(result_id)
        self.selected_result_id = result_id
        self._selected_entry = entry
        self.export_origin_button.configure(
            state="normal" if entry.result is not None else "disabled"
        )
        self.relink_result_button.configure(
            state="normal" if entry.result is None else "disabled"
        )
        if self.history_tree.exists(result_id):
            self.history_tree.selection_set(result_id)
            self.history_tree.focus(result_id)
            self.history_tree.see(result_id)
        for preview in self.image_previews.values():
            preview.clear("请选择图片分类。")
        self.field_evolution_preview.clear("尚未选择场演化帧。")
        for preview in self.mechanism_field_evolution_previews:
            preview.clear("尚未选择机制化场演化帧。")
        self._hide_field_evolution_controls()
        self._image_items = ()
        self._data_items = {}
        if entry.result is None:
            self.field_evolution_browser.set_availability(
                self.field_evolution_browser.availability.not_applicable()
            )
            self.mechanism_field_evolution_browser.set_availability(
                self.mechanism_field_evolution_browser.availability.not_applicable()
            )
            self.header_var.set(f"{entry.display_name}｜{entry.status}")
            self.path_var.set("")
            self._show_message(f"结果文件异常：{entry.error_message}")
            return
        record = entry.result.record
        self.field_evolution_browser.set_availability(entry.result.field_evolution)
        self.mechanism_field_evolution_browser.set_availability(
            entry.result.mechanism_field_evolution
        )
        self.header_var.set(f"{entry.display_name}｜{entry.status}")
        self.path_var.set(f"输出目录：{record.output_directory}\n输入指纹：{record.input_fingerprint[:12] or '未记录'}")
        self.notebook.select(self.category_frames["summary"])
        self.show_category("summary")

    def show_category(self, category_id: str) -> None:
        self._active_category_id = category_id
        for preview in self.image_previews.values():
            preview.clear("当前分类尚未选择图片。")
        self.field_evolution_preview.clear("当前分类尚未选择场演化帧。")
        for preview in self.mechanism_field_evolution_previews:
            preview.clear("当前分类尚未选择机制化场演化帧。")
        self._hide_field_evolution_controls()
        self._image_items = ()
        self.image_name_var.set("")
        for choice in self.image_choices.values():
            choice.configure(values=())
        self.image_note_var.set("")
        self.data_tree.delete(*self.data_tree.get_children(""))
        self._data_items = {}
        entry = self._selected_entry
        if entry is None or entry.result is None:
            return
        category = entry.result.category(category_id)
        if category_id == "summary":
            record = entry.result.record
            text = "\n".join((
                f"分析模式：{entry.display_name}", f"完成时间：{record.created_at}", f"结果状态：{entry.status}",
                f"输出目录：{record.output_directory}", f"输入指纹：{record.input_fingerprint[:12] or '未记录'}", "",
                entry.result.published.summary_text,
            ))
            self._show_message(text)
            return
        if category_id == "data":
            for index, item in enumerate(category.items):
                iid = str(index)
                self._data_items[iid] = item
                self.data_tree.insert("", "end", iid=iid, values=(item.display_name, item.filename, item.file_type, format_file_size(item.size_bytes), item.status))
            if not category.items:
                self._show_message(category.empty_message)
            return
        if category_id == "fields":
            self._set_static_fields_visible(True)
        if category_id == "fields" and entry.result.record.analysis_type == "elastoplastic_single_pass":
            self.field_view_mode_choice.pack(fill="x", before=self.image_choices["fields"])
            self.field_view_mode_var.set("单程场演化")
            self._show_field_evolution_mode()
            return
        if category_id == "fields" and entry.result.record.analysis_type == "mechanism_elastoplastic_single_pass":
            self.mechanism_field_view_mode_choice.pack(
                fill="x", before=self.image_choices["fields"]
            )
            self.mechanism_existing_comparison_button.pack(
                fill="x", before=self.image_choices["fields"], pady=(4, 0)
            )
            self.mechanism_field_view_mode_var.set("基准路线")
            self._show_mechanism_field_evolution_mode()
            return
        self.image_preview = self.image_previews[category_id]
        self.image_choice = self.image_choices[category_id]
        self._image_items = tuple(item for item in category.items if item.kind == "image")
        self.image_choice.configure(values=tuple(item.display_name for item in self._image_items))
        if self._image_items:
            self.image_name_var.set(self._image_items[0].display_name)
            self._show_image(self._image_items[0])
        else:
            self.image_preview.clear(category.empty_message)

    def _hide_field_evolution_controls(self) -> None:
        self.field_view_mode_choice.pack_forget()
        self.field_evolution_host.pack_forget()
        self.mechanism_field_view_mode_choice.pack_forget()
        self.mechanism_existing_comparison_button.pack_forget()
        self.mechanism_field_evolution_host.pack_forget()

    def _set_static_fields_visible(self, visible: bool) -> None:
        choice = self.image_choices["fields"]
        preview = self.image_previews["fields"]
        note = self.image_note_labels["fields"]
        if visible:
            choice.pack(fill="x")
            preview.pack(fill="both", expand=True, pady=(6, 0))
            note.pack(fill="x", pady=(5, 0))
        else:
            choice.pack_forget()
            preview.pack_forget()
            note.pack_forget()

    def _show_field_evolution_mode(self) -> None:
        self._set_static_fields_visible(False)
        self.field_evolution_host.pack(fill="both", expand=True, pady=(6, 0))
        availability = self.field_evolution_browser.availability
        sequence = availability.sequence
        if sequence is None:
            self.field_evolution_metadata_var.set(availability.message)
            self.field_evolution_preview.clear(availability.message)
            self.field_evolution_scale.configure(from_=0, to=0, state="disabled")
            self._update_field_evolution_buttons()
            return
        self.field_evolution_scale.configure(from_=0, to=sequence.frame_count - 1, state="normal")
        self.field_evolution_scale_var.set(float(self.field_evolution_browser.frame_index))
        self._show_current_field_evolution_frame()

    def _show_existing_fields_mode(self) -> None:
        self.field_evolution_host.pack_forget()
        self._set_static_fields_visible(True)
        entry = self._selected_entry
        if entry is None or entry.result is None:
            return
        category = entry.result.category("fields")
        self.image_preview = self.image_previews["fields"]
        self.image_choice = self.image_choices["fields"]
        self._image_items = tuple(item for item in category.items if item.kind == "image")
        self.image_choice.configure(values=tuple(item.display_name for item in self._image_items))
        if self._image_items:
            self.image_name_var.set(self._image_items[0].display_name)
            self._show_image(self._image_items[0])
        else:
            self.image_preview.clear(category.empty_message)

    def _show_mechanism_field_evolution_mode(self) -> None:
        self._set_static_fields_visible(False)
        self.mechanism_field_evolution_host.pack(fill="both", expand=True, pady=(6, 0))
        availability = self.mechanism_field_evolution_browser.availability
        sequence = availability.sequence
        if sequence is None:
            self.mechanism_field_evolution_metadata_var.set(availability.message)
            self.mechanism_field_evolution_scale.configure(from_=0, to=0, state="disabled")
            for preview in self.mechanism_field_evolution_previews:
                preview.clear(availability.message)
            self._update_mechanism_field_buttons()
            return
        self.mechanism_field_evolution_scale.configure(
            from_=0, to=sequence.frame_count - 1, state="normal"
        )
        self.mechanism_field_evolution_scale_var.set(
            float(self.mechanism_field_evolution_browser.frame_index)
        )
        self._show_current_mechanism_field_frame()

    def _show_current_mechanism_field_frame(self) -> None:
        browser = self.mechanism_field_evolution_browser
        sequence = browser.availability.sequence
        self.mechanism_field_evolution_metadata_var.set(
            browser.current_frame_display_lines()
        )
        paths = browser.current_image_paths
        for index, preview in enumerate(self.mechanism_field_evolution_previews):
            if index >= len(paths):
                preview.grid_remove()
                preview.clear("")
                continue
            preview.grid()
            preview.clear("正在读取当前位置图片。")
            if browser.view_mode == "side_by_side":
                route = ("baseline", "mechanism")[index]
            else:
                route = browser.view_mode
            sequence_paths = (
                tuple(getattr(frame, route).png_path for frame in sequence.frames)
                if sequence is not None
                else ()
            )
            try:
                preview.set_image(
                    paths[index],
                    source="result-center-mechanism-field",
                    sequence_paths=sequence_paths,
                    sequence_index=browser.frame_index,
                )
            except (OSError, ValueError):
                preview.clear(f"当前机制化场演化帧读取失败：{Path(paths[index]).name}")
        self._update_mechanism_field_buttons()

    def _update_mechanism_field_buttons(self) -> None:
        browser = self.mechanism_field_evolution_browser
        self.mechanism_previous_position_button.configure(
            state="normal" if browser.can_previous else "disabled"
        )
        self.mechanism_next_position_button.configure(
            state="normal" if browser.can_next else "disabled"
        )
        sequence = browser.availability.sequence
        frame = browser.current_frame
        for route in ("baseline", "mechanism", "difference"):
            vtu_available = False
            if frame is not None:
                path = getattr(frame, route).vtu_path
                try:
                    vtu_available = path.is_file() and path.stat().st_size > 0
                except OSError:
                    pass
            self.mechanism_field_file_buttons[f"{route}_vtu"].configure(
                state="normal" if vtu_available else "disabled"
            )
            pvd_available = False
            if sequence is not None:
                path = sequence.pvd_paths[route]
                try:
                    pvd_available = path.is_file() and path.stat().st_size > 0
                except OSError:
                    pass
            self.mechanism_field_file_buttons[f"{route}_pvd"].configure(
                state="normal" if pvd_available else "disabled"
            )
        self.mechanism_field_file_buttons["directory"].configure(
            state="normal" if sequence is not None else "disabled"
        )

    def select_mechanism_field_evolution_frame(self, value: str | float) -> None:
        frame = self.mechanism_field_evolution_browser.select(float(value))
        if frame is not None:
            self.mechanism_field_evolution_scale_var.set(float(frame.frame_index))
            self._show_current_mechanism_field_frame()

    def previous_mechanism_field_evolution_frame(self) -> None:
        frame = self.mechanism_field_evolution_browser.previous()
        if frame is not None:
            self.mechanism_field_evolution_scale_var.set(float(frame.frame_index))
            self._show_current_mechanism_field_frame()

    def next_mechanism_field_evolution_frame(self) -> None:
        frame = self.mechanism_field_evolution_browser.next()
        if frame is not None:
            self.mechanism_field_evolution_scale_var.set(float(frame.frame_index))
            self._show_current_mechanism_field_frame()

    def open_current_mechanism_field_vtu(self, route: str) -> None:
        frame = self.mechanism_field_evolution_browser.current_frame
        if frame is not None:
            self._open_field_evolution_path(getattr(frame, route).vtu_path, expect_directory=False)

    def open_mechanism_field_pvd(self, route: str) -> None:
        sequence = self.mechanism_field_evolution_browser.availability.sequence
        if sequence is not None:
            self._open_field_evolution_path(sequence.pvd_paths[route], expect_directory=False)

    def open_mechanism_field_evolution_directory(self) -> None:
        sequence = self.mechanism_field_evolution_browser.availability.sequence
        if sequence is not None:
            self._open_field_evolution_path(sequence.sequence_directory, expect_directory=True)

    def _show_current_field_evolution_frame(self) -> None:
        frame = self.field_evolution_browser.current_frame
        sequence = self.field_evolution_browser.availability.sequence
        self.field_evolution_metadata_var.set(
            self.field_evolution_browser.current_frame_display_lines()
        )
        if frame is None:
            self.field_evolution_preview.clear(
                self.field_evolution_browser.availability.message
            )
        else:
            self.field_evolution_preview.clear("正在读取当前位置图片。")
            try:
                self.field_evolution_preview.set_image(
                    frame.png_path,
                    source="result-center-field-evolution",
                    sequence_paths=(
                        tuple(item.png_path for item in sequence.frames)
                        if sequence is not None
                        else ()
                    ),
                    sequence_index=self.field_evolution_browser.frame_index,
                )
            except (OSError, ValueError):
                self.field_evolution_preview.clear(
                    f"当前场演化帧读取失败：{frame.png_path.name}"
                )
        self._update_field_evolution_buttons()

    def _update_field_evolution_buttons(self) -> None:
        sequence = self.field_evolution_browser.availability.sequence
        self.previous_position_button.configure(
            state="normal" if self.field_evolution_browser.can_previous else "disabled"
        )
        self.next_position_button.configure(
            state="normal" if self.field_evolution_browser.can_next else "disabled"
        )
        file_state = "normal" if sequence is not None else "disabled"
        frame = self.field_evolution_browser.current_frame
        current_vtu_available = False
        if frame is not None:
            try:
                current_vtu_available = frame.vtu_path.is_file() and frame.vtu_path.stat().st_size > 0
            except OSError:
                current_vtu_available = False
        self.open_current_vtu_button.configure(
            state="normal" if current_vtu_available else "disabled"
        )
        self.open_sequence_button.configure(state=file_state)
        self.open_sequence_directory_button.configure(state=file_state)

    def select_field_evolution_frame(self, value: str | float) -> None:
        if self.field_evolution_browser.availability.sequence is None:
            return
        frame = self.field_evolution_browser.select(float(value))
        if frame is None:
            return
        self.field_evolution_scale_var.set(float(frame.frame_index))
        self._show_current_field_evolution_frame()

    def previous_field_evolution_frame(self) -> None:
        frame = self.field_evolution_browser.previous()
        if frame is not None:
            self.field_evolution_scale_var.set(float(frame.frame_index))
            self._show_current_field_evolution_frame()

    def next_field_evolution_frame(self) -> None:
        frame = self.field_evolution_browser.next()
        if frame is not None:
            self.field_evolution_scale_var.set(float(frame.frame_index))
            self._show_current_field_evolution_frame()

    def open_current_field_vtu(self) -> None:
        frame = self.field_evolution_browser.current_frame
        if frame is not None:
            self._open_field_evolution_path(frame.vtu_path, expect_directory=False)

    def open_field_evolution_pvd(self) -> None:
        sequence = self.field_evolution_browser.availability.sequence
        if sequence is not None:
            self._open_field_evolution_path(sequence.pvd_path, expect_directory=False)

    def open_field_evolution_directory(self) -> None:
        sequence = self.field_evolution_browser.availability.sequence
        if sequence is not None:
            self._open_field_evolution_path(sequence.sequence_directory, expect_directory=True)

    def _open_field_evolution_path(self, path: Path, *, expect_directory: bool) -> None:
        entry = self._selected_entry
        if entry is None or entry.result is None:
            return
        try:
            output = Path(entry.result.record.output_directory).expanduser().resolve()
            resolved = path.expanduser().resolve()
            resolved.relative_to(output)
            if expect_directory:
                if not resolved.is_dir():
                    raise OSError(f"目录不存在：{resolved}")
            elif not resolved.is_file() or resolved.stat().st_size <= 0:
                raise OSError(f"文件不存在或为空：{resolved}")
            self._open_handler(resolved)
        except (OSError, ValueError) as exc:
            self._show_error(f"无法打开场演化结果：{exc}")

    def _show_image(self, item: ResultArtifactItem) -> None:
        self.image_preview.clear("正在读取图片。")
        self.image_note_var.set(item.reading_note)
        try:
            self.image_preview.set_image(item.path, source="result-center")
        except (OSError, ValueError):
            pass

    def _show_message(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

    def _selected_data_item(self) -> ResultArtifactItem | None:
        selection = self.data_tree.selection()
        return self._data_items.get(selection[0]) if selection else None

    def _open_selected_file(self) -> None:
        item = self._selected_data_item()
        if item is not None:
            self._open(item.path)

    def _open_selected_directory(self) -> None:
        item = self._selected_data_item()
        if item is not None:
            self._open(item.path.parent)

    def open_selected_result_directory(self) -> None:
        entry = self._selected_entry
        if entry is not None and entry.result is not None:
            self._open(Path(entry.result.record.output_directory))

    def export_selected_origin_data(self) -> bool:
        entry = self._selected_entry
        if entry is None or entry.result is None:
            self.export_origin_button.configure(state="disabled")
            return False
        output = Path(entry.result.record.output_directory).expanduser().resolve()
        selected = self._directory_dialog_handler(
            parent=self.winfo_toplevel(),
            title="选择新的 Origin 数据导出目录",
            initialdir=str(output.parent),
            mustexist=False,
        )
        if not selected:
            return False
        destination = (
            Path(selected).expanduser().resolve()
            / f"Origin_Data_{entry.result_id}"
        )
        try:
            exported = self._origin_export_handler(entry.result, destination)
        except (OriginExportError, OSError, ValueError) as exc:
            messagebox.showerror(
                "Origin 数据导出失败", str(exc), parent=self.winfo_toplevel()
            )
            return False
        messagebox.showinfo(
            "Origin 数据导出完成",
            f"已生成 {len(exported.table_paths)} 个数据表：\n{exported.directory}",
            parent=self.winfo_toplevel(),
        )
        return True

    def _open(self, path: Path) -> None:
        try:
            if not path.exists():
                raise OSError(f"文件或目录不存在：{path}")
            self._open_handler(path.resolve())
        except OSError as exc:
            self._show_error(f"无法打开 {path}：{exc}")

    def _show_error(self, message: str) -> None:
        messagebox.showerror("结果浏览错误", message, parent=self.winfo_toplevel())

    def _request_refresh(self) -> None:
        if self.refresh_handler is not None:
            self.refresh_handler()

    def _request_relink(self) -> None:
        if self._selected_entry is not None and self.relink_handler is not None:
            self.relink_handler(self._selected_entry.result_id)

    def _on_history_selected(self, _event=None) -> None:
        selection = self.history_tree.selection()
        if selection and selection[0] != self.selected_result_id:
            self.select_result(selection[0])

    def _on_category_selected(self, _event=None) -> None:
        selected = self.notebook.select()
        for category_id, frame in self.category_frames.items():
            if str(frame) == selected:
                self.show_category(category_id)
                break

    def _on_image_selected(self, _event=None) -> None:
        name = self.image_name_var.get()
        for item in self._image_items:
            if item.display_name == name:
                self._show_image(item)
                break

    def _on_field_view_mode_selected(self, _event=None) -> None:
        if self.field_view_mode_var.get() == "已有云图":
            self._show_existing_fields_mode()
        else:
            self._show_field_evolution_mode()

    def _on_mechanism_field_view_mode_selected(self, _event=None) -> None:
        label = self.mechanism_field_view_mode_var.get()
        modes = {
            "基准路线": "baseline",
            "机制路线": "mechanism",
            "差值对比": "difference",
            "基准／机制并排": "side_by_side",
        }
        self.mechanism_field_evolution_browser.set_view_mode(modes[label])
        self._show_mechanism_field_evolution_mode()

    def _show_existing_mechanism_comparison(self) -> None:
        self.mechanism_field_evolution_host.pack_forget()
        self._show_existing_fields_mode()
        entry = self._selected_entry
        if entry is not None and entry.result is not None:
            history = entry.result.category("history")
            images = tuple(item for item in history.items if item.kind == "image")
            if images:
                self.image_preview = self.image_previews["fields"]
                self.image_choice = self.image_choices["fields"]
                self._image_items = images
                self.image_choice.configure(values=tuple(item.display_name for item in images))
                self.image_name_var.set(images[0].display_name)
                self._show_image(images[0])


__all__ = ["ResultCenterView"]
