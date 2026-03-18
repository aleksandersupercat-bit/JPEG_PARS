from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .clustering import ClusterResult, cluster_images, materialize_groups
from .features import OcrConfig, resolve_tesseract_cmd
from .template_parser import (
    OcrCandidate,
    ParsedSheet,
    TemplateRegion,
    default_region_name,
    export_results_to_excel,
    get_template_ocr_backend_info,
    load_template,
    parse_template_batch,
    save_template,
)


COLOR_PALETTE = [
    "#E76F51",
    "#2A9D8F",
    "#3A86FF",
    "#F4A261",
    "#8338EC",
    "#219EBC",
    "#FF006E",
    "#8ECAE6",
    "#6A994E",
    "#FB8500",
    "#4361EE",
    "#EF476F",
]


class JpegParsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("JPEG_PARS")
        self.geometry("1520x920")
        self.minsize(1260, 760)

        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.group_preview_image: ImageTk.PhotoImage | None = None
        self.template_preview_image: ImageTk.PhotoImage | None = None
        self.template_preview_cache_key: tuple[int, int, int] | None = None
        self.group_result: ClusterResult | None = None
        self.group_output_dir: Path | None = None
        self.group_path_index: dict[str, Path] = {}

        self.template_source_image: Image.Image | None = None
        self.template_source_path: Path | None = None
        self.template_regions: list[TemplateRegion] = []
        self.template_results: list[ParsedSheet] = []
        self.template_scale = 1.0
        self.template_zoom = 1.0
        self.template_offset_x = 0
        self.template_offset_y = 0
        self.template_display_width = 1
        self.template_display_height = 1
        self.template_pan_start: tuple[int, int] | None = None
        self.template_manual_pan = False
        self.template_draw_mode = False
        self.template_drag_start: tuple[int, int] | None = None
        self.template_drag_item: int | None = None

        self._build_ui()
        self._autofill_tesseract_path()
        self.after(120, self._poll_worker_queue)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        grouping_tab = ttk.Frame(notebook, padding=12)
        template_tab = ttk.Frame(notebook, padding=12)
        notebook.add(grouping_tab, text="Группировка")
        notebook.add(template_tab, text="Шаблон OCR")

        self._build_grouping_tab(grouping_tab)
        self._build_template_tab(template_tab)

    def _autofill_tesseract_path(self) -> None:
        resolved = resolve_tesseract_cmd()
        if not resolved:
            return
        if not self.group_tesseract_var.get():
            self.group_tesseract_var.set(resolved)
        if not self.template_tesseract_var.get():
            self.template_tesseract_var.set(resolved)

    def _build_grouping_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Параметры группировки", padding=12)
        controls.pack(fill="x")

        self.group_input_var = tk.StringVar()
        self.group_output_var = tk.StringVar()
        self.group_similarity_var = tk.IntVar(value=82)
        self.group_recursive_var = tk.BooleanVar(value=False)
        self.group_mode_var = tk.StringVar(value="copy")
        self.group_ocr_mode_var = tk.StringVar(value="auto")
        self.group_ocr_lang_var = tk.StringVar(value="eng+rus")
        self.group_tesseract_var = tk.StringVar()

        ttk.Label(controls, text="Папка с JPEG").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.group_input_var, width=90).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Выбрать", command=self._choose_group_input).grid(row=0, column=2, padx=4)

        ttk.Label(controls, text="Папка результата").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.group_output_var, width=90).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(controls, text="Выбрать", command=self._choose_group_output).grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Сходство 1..100").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(
            controls,
            from_=1,
            to=100,
            orient="horizontal",
            variable=self.group_similarity_var,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Label(controls, textvariable=self.group_similarity_var, width=6).grid(row=2, column=2, sticky="w", pady=(8, 0))

        ttk.Label(controls, text="OCR режим").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.group_ocr_mode_var,
            values=("auto", "off", "required"),
            state="readonly",
            width=20,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(controls, text="Языки OCR").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.group_ocr_lang_var, width=20).grid(row=4, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(controls, text="Путь к tesseract.exe").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.group_tesseract_var, width=90).grid(row=5, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(controls, text="Выбрать", command=self._choose_tesseract_for_grouping).grid(row=5, column=2, padx=4, pady=(8, 0))

        ttk.Checkbutton(controls, text="Рекурсивно искать JPEG", variable=self.group_recursive_var).grid(row=6, column=0, sticky="w", pady=(10, 0))
        ttk.Radiobutton(controls, text="Копировать", variable=self.group_mode_var, value="copy").grid(row=6, column=1, sticky="w", pady=(10, 0))
        ttk.Radiobutton(controls, text="Перемещать", variable=self.group_mode_var, value="move").grid(row=6, column=1, sticky="w", padx=(120, 0), pady=(10, 0))

        self.group_start_button = ttk.Button(controls, text="Запустить группировку", command=self._start_grouping)
        self.group_start_button.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        controls.columnconfigure(1, weight=1)

        result_pane = ttk.PanedWindow(parent, orient="horizontal")
        result_pane.pack(fill="both", expand=True, pady=(12, 0))

        left = ttk.Frame(result_pane)
        right = ttk.Frame(result_pane)
        result_pane.add(left, weight=3)
        result_pane.add(right, weight=2)

        tree_frame = ttk.LabelFrame(left, text="Результат", padding=8)
        tree_frame.pack(fill="both", expand=True)
        self.group_tree = ttk.Treeview(tree_frame, columns=("kind",), show="tree headings")
        self.group_tree.heading("#0", text="Группы и файлы")
        self.group_tree.heading("kind", text="Тип")
        self.group_tree.column("#0", width=560, anchor="w")
        self.group_tree.column("kind", width=100, anchor="center")
        self.group_tree.pack(fill="both", expand=True)
        self.group_tree.bind("<<TreeviewSelect>>", self._on_group_tree_select)

        preview_frame = ttk.LabelFrame(right, text="Просмотр", padding=8)
        preview_frame.pack(fill="both", expand=True)
        self.group_preview_canvas = tk.Canvas(preview_frame, bg="#F5F5F5", highlightthickness=0)
        self.group_preview_canvas.pack(fill="both", expand=True)

        self.group_status_var = tk.StringVar(value="Ожидание запуска.")
        ttk.Label(parent, textvariable=self.group_status_var).pack(fill="x", pady=(8, 0))

    def _build_template_tab(self, parent: ttk.Frame) -> None:
        top_controls = ttk.LabelFrame(parent, text="Шаблон и OCR", padding=12)
        top_controls.pack(fill="x")

        self.template_image_var = tk.StringVar()
        self.template_folder_var = tk.StringVar()
        self.template_export_var = tk.StringVar()
        self.template_ocr_lang_var = tk.StringVar(value="eng+rus")
        self.template_tesseract_var = tk.StringVar()
        self.template_recursive_var = tk.BooleanVar(value=False)

        ttk.Label(top_controls, text="Шаблон JPEG").grid(row=0, column=0, sticky="w")
        ttk.Entry(top_controls, textvariable=self.template_image_var, width=88).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top_controls, text="Создать шаблон", command=self._choose_template_image).grid(row=0, column=2)
        ttk.Button(top_controls, text="Загрузить шаблон", command=self._load_template_file).grid(row=0, column=3, padx=(8, 0))

        ttk.Label(top_controls, text="Папка с JPEG").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_controls, textvariable=self.template_folder_var, width=88).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(top_controls, text="Выбрать папку", command=self._choose_template_folder).grid(row=1, column=2, pady=(8, 0))

        ttk.Label(top_controls, text="Excel файл").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_controls, textvariable=self.template_export_var, width=88).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(top_controls, text="Экспорт в Excel", command=self._export_template_results).grid(row=2, column=2, pady=(8, 0))
        ttk.Button(top_controls, text="Сохранить шаблон", command=self._save_template_file).grid(row=2, column=3, padx=(8, 0), pady=(8, 0))

        ttk.Label(top_controls, text="Языки OCR").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_controls, textvariable=self.template_ocr_lang_var, width=20).grid(row=3, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(top_controls, text="Путь к tesseract.exe").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top_controls, textvariable=self.template_tesseract_var, width=88).grid(row=4, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(top_controls, text="Выбрать", command=self._choose_tesseract_for_template).grid(row=4, column=2, pady=(8, 0))

        button_bar = ttk.Frame(top_controls)
        button_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        self.template_add_region_button = ttk.Button(button_bar, text="Выбрать границы поиска", command=self._enable_template_draw_mode)
        self.template_add_region_button.pack(side="left")
        ttk.Button(button_bar, text="Удалить метку", command=self._delete_selected_region).pack(side="left", padx=8)
        ttk.Button(button_bar, text="Очистить метки", command=self._clear_regions).pack(side="left")
        ttk.Checkbutton(button_bar, text="Рекурсивно искать JPEG", variable=self.template_recursive_var).pack(side="left", padx=12)
        self.template_parse_button = ttk.Button(button_bar, text="Запустить парсинг", command=self._start_template_parsing)
        self.template_parse_button.pack(side="right")

        top_controls.columnconfigure(1, weight=1)

        main_pane = ttk.PanedWindow(parent, orient="horizontal")
        main_pane.pack(fill="both", expand=True, pady=(12, 0))

        left = ttk.Frame(main_pane)
        right = ttk.Frame(main_pane)
        main_pane.add(left, weight=3)
        main_pane.add(right, weight=2)

        canvas_frame = ttk.LabelFrame(left, text="Шаблон", padding=8)
        canvas_frame.pack(fill="both", expand=True)
        self.template_canvas = tk.Canvas(canvas_frame, bg="#FAFAFA", highlightthickness=0, cursor="crosshair")
        self.template_canvas.pack(fill="both", expand=True)
        self.template_canvas.bind("<ButtonPress-1>", self._on_template_mouse_down)
        self.template_canvas.bind("<B1-Motion>", self._on_template_mouse_move)
        self.template_canvas.bind("<ButtonRelease-1>", self._on_template_mouse_up)
        self.template_canvas.bind("<ButtonPress-2>", self._on_template_pan_start)
        self.template_canvas.bind("<B2-Motion>", self._on_template_pan_move)
        self.template_canvas.bind("<ButtonRelease-2>", self._on_template_pan_end)
        self.template_canvas.bind("<MouseWheel>", self._on_template_mousewheel)
        self.template_canvas.bind("<Configure>", self._on_template_canvas_resize)

        right_pane = ttk.PanedWindow(right, orient="vertical")
        right_pane.pack(fill="both", expand=True)

        regions_frame = ttk.LabelFrame(right_pane, text="Метки", padding=8)
        results_frame = ttk.LabelFrame(right_pane, text="Распознанные значения", padding=8)
        debug_frame = ttk.LabelFrame(right_pane, text="OCR Debug", padding=8)
        right_pane.add(regions_frame, weight=2)
        right_pane.add(results_frame, weight=3)
        right_pane.add(debug_frame, weight=2)

        self.region_tree = ttk.Treeview(regions_frame, columns=("color", "name", "value", "confidence"), show="headings", height=8)
        self.region_tree.heading("color", text="Цвет")
        self.region_tree.heading("name", text="Атрибут")
        self.region_tree.heading("value", text="Значение")
        self.region_tree.heading("confidence", text="Увер.")
        self.region_tree.column("color", width=90, anchor="center")
        self.region_tree.column("name", width=120, anchor="center")
        self.region_tree.column("value", width=220, anchor="w")
        self.region_tree.column("confidence", width=80, anchor="center")
        self.region_tree.pack(fill="both", expand=True)
        self.region_tree.bind("<Double-1>", self._edit_region_name)
        self.region_tree.bind("<<TreeviewSelect>>", self._on_region_tree_select)

        self.results_tree = ttk.Treeview(results_frame, show="headings")
        self.results_tree.pack(fill="both", expand=True)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_row_select)

        self.debug_tree = ttk.Treeview(
            debug_frame,
            columns=("variant", "backend", "text", "confidence", "score"),
            show="headings",
        )
        self.debug_tree.heading("variant", text="Variant")
        self.debug_tree.heading("backend", text="Backend")
        self.debug_tree.heading("text", text="Text")
        self.debug_tree.heading("confidence", text="Conf")
        self.debug_tree.heading("score", text="Score")
        self.debug_tree.column("variant", width=120, anchor="w")
        self.debug_tree.column("backend", width=80, anchor="center")
        self.debug_tree.column("text", width=220, anchor="w")
        self.debug_tree.column("confidence", width=70, anchor="e")
        self.debug_tree.column("score", width=70, anchor="e")
        self.debug_tree.pack(fill="both", expand=True)

        self.template_status_var = tk.StringVar(value="Загрузите шаблон JPEG.")
        ttk.Label(parent, textvariable=self.template_status_var).pack(fill="x", pady=(8, 0))

    def _choose_group_input(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с JPEG")
        if folder:
            self.group_input_var.set(folder)
            if not self.group_output_var.get():
                self.group_output_var.set(str(Path(folder) / "grouped_output"))

    def _choose_group_output(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку результата")
        if folder:
            self.group_output_var.set(folder)

    def _choose_tesseract_for_grouping(self) -> None:
        path = filedialog.askopenfilename(title="Выберите tesseract.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.group_tesseract_var.set(path)

    def _choose_tesseract_for_template(self) -> None:
        path = filedialog.askopenfilename(title="Выберите tesseract.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.template_tesseract_var.set(path)

    def _start_grouping(self) -> None:
        input_value = self.group_input_var.get().strip()
        output_value = self.group_output_var.get().strip()
        if not input_value:
            messagebox.showerror("Ошибка", "Укажите папку с JPEG.")
            return
        if not output_value:
            messagebox.showerror("Ошибка", "Укажите папку результата.")
            return
        input_dir = Path(input_value)
        output_dir = Path(output_value)
        if not input_dir.exists():
            messagebox.showerror("Ошибка", "Папка с JPEG не найдена.")
            return

        self.group_status_var.set("Идет группировка изображений...")
        self.group_start_button.config(state="disabled")
        worker = threading.Thread(target=self._run_grouping_worker, args=(input_dir, output_dir), daemon=True)
        worker.start()

    def _run_grouping_worker(self, input_dir: Path, output_dir: Path) -> None:
        try:
            ocr_config = OcrConfig(
                mode=self.group_ocr_mode_var.get(),
                tesseract_cmd=self.group_tesseract_var.get().strip() or None,
                languages=self.group_ocr_lang_var.get().strip() or "eng+rus",
                psm=6,
            )
            result = cluster_images(
                input_dir=input_dir,
                similarity_threshold=int(self.group_similarity_var.get()),
                recursive=self.group_recursive_var.get(),
                ocr_config=ocr_config,
            )
            materialize_groups(
                result=result,
                output_dir=output_dir,
                source_root=input_dir,
                mode=self.group_mode_var.get(),
                min_group_size=1,
                ocr_config=ocr_config,
            )
            self.worker_queue.put(("grouping_done", (result, output_dir)))
        except Exception as exc:
            self.worker_queue.put(("grouping_error", str(exc)))

    def _on_group_tree_select(self, _: object) -> None:
        selection = self.group_tree.selection()
        if not selection:
            return
        path = self.group_path_index.get(selection[0])
        if path is None:
            return
        self._render_group_preview(path)

    def _render_group_preview(self, path: Path) -> None:
        canvas = self.group_preview_canvas
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 300)
        image = Image.open(path).convert("RGB")
        image.thumbnail((width - 20, height - 20), Image.Resampling.LANCZOS)
        self.group_preview_image = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(width // 2, height // 2, image=self.group_preview_image, anchor="center")

    def _choose_template_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите шаблон JPEG",
            filetypes=[("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        self.template_source_path = Path(path)
        self.template_source_image = Image.open(path).convert("RGB")
        self.template_zoom = 1.0
        self.template_manual_pan = False
        self.template_preview_cache_key = None
        self.template_preview_image = None
        self.template_image_var.set(path)
        self.template_regions.clear()
        self.template_results.clear()
        self._refresh_region_tree()
        self._refresh_results_tree()
        self._render_template_canvas()
        self.template_status_var.set("Шаблон загружен. Можно размечать области поиска.")

    def _save_template_file(self) -> None:
        if not self.template_regions:
            messagebox.showerror("Ошибка", "Нет областей для сохранения.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить шаблон",
            defaultextension=".json",
            initialfile="template_regions.json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        save_template(self.template_regions, self.template_source_path, Path(path))
        self.template_status_var.set(f"Шаблон сохранен: {path}")

    def _load_template_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Загрузить шаблон",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        image_path, regions = load_template(Path(path))
        self.template_regions = regions
        self.template_results.clear()
        if image_path and image_path.exists():
            self.template_source_path = image_path
            self.template_source_image = Image.open(image_path).convert("RGB")
            self.template_image_var.set(str(image_path))
            self.template_zoom = 1.0
            self.template_manual_pan = False
            self.template_preview_cache_key = None
            self.template_preview_image = None
        self._refresh_region_tree()
        self._refresh_results_tree()
        self._render_template_canvas()
        self.template_status_var.set(f"Шаблон загружен: {path}")

    def _choose_template_folder(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с JPEG для парсинга")
        if folder:
            self.template_folder_var.set(folder)
            if not self.template_export_var.get():
                self.template_export_var.set(str(Path(folder) / "parsed_dimensions.xlsx"))

    def _enable_template_draw_mode(self) -> None:
        if self.template_source_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите шаблон JPEG.")
            return
        self.template_draw_mode = True
        self.template_status_var.set("Нарисуйте прямоугольник поверх изображения.")

    def _on_template_canvas_resize(self, _: object) -> None:
        self._render_template_canvas()

    def _on_template_mousewheel(self, event: tk.Event[tk.Misc]) -> None:
        if self.template_source_image is None:
            return
        factor = 1.1 if event.delta > 0 else 0.9
        self.template_zoom = min(8.0, max(0.2, self.template_zoom * factor))
        self.template_manual_pan = True
        self._render_template_canvas()

    def _on_template_pan_start(self, event: tk.Event[tk.Misc]) -> None:
        if self.template_source_image is None:
            return
        self.template_pan_start = (event.x, event.y)
        self.template_manual_pan = True

    def _on_template_pan_move(self, event: tk.Event[tk.Misc]) -> None:
        if self.template_pan_start is None:
            return
        previous_x, previous_y = self.template_pan_start
        self.template_offset_x += event.x - previous_x
        self.template_offset_y += event.y - previous_y
        self.template_pan_start = (event.x, event.y)
        self._render_template_canvas(lock_pan=True)

    def _on_template_pan_end(self, _: tk.Event[tk.Misc]) -> None:
        self.template_pan_start = None

    def _on_template_mouse_down(self, event: tk.Event[tk.Misc]) -> None:
        if not self.template_draw_mode:
            return
        self.template_drag_start = (event.x, event.y)
        self.template_drag_item = self.template_canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#4C78A8",
            fill="#4C78A8",
            width=2,
            stipple="gray25",
        )

    def _on_template_mouse_move(self, event: tk.Event[tk.Misc]) -> None:
        if not self.template_draw_mode or self.template_drag_start is None or self.template_drag_item is None:
            return
        x0, y0 = self.template_drag_start
        self.template_canvas.coords(self.template_drag_item, x0, y0, event.x, event.y)

    def _on_template_mouse_up(self, event: tk.Event[tk.Misc]) -> None:
        if not self.template_draw_mode or self.template_drag_start is None or self.template_drag_item is None:
            return
        x0, y0 = self.template_drag_start
        x1, y1 = event.x, event.y
        self.template_canvas.delete(self.template_drag_item)
        self.template_drag_item = None
        self.template_drag_start = None
        region = self._region_from_canvas_points(x0, y0, x1, y1)
        self.template_draw_mode = False
        if region is None:
            self.template_status_var.set("Область слишком мала или вышла за пределы изображения.")
            return
        self.template_regions.append(region)
        self._refresh_region_tree()
        self._render_template_canvas()
        self.template_status_var.set(f"Добавлена область {region.name}.")

    def _region_from_canvas_points(self, x0: int, y0: int, x1: int, y1: int) -> TemplateRegion | None:
        if self.template_source_image is None:
            return None
        left = max(self.template_offset_x, min(x0, x1))
        right = min(self.template_offset_x + self.template_display_width, max(x0, x1))
        top = max(self.template_offset_y, min(y0, y1))
        bottom = min(self.template_offset_y + self.template_display_height, max(y0, y1))
        if right - left < 8 or bottom - top < 8:
            return None

        nx0 = (left - self.template_offset_x) / self.template_display_width
        ny0 = (top - self.template_offset_y) / self.template_display_height
        nx1 = (right - self.template_offset_x) / self.template_display_width
        ny1 = (bottom - self.template_offset_y) / self.template_display_height
        index = len(self.template_regions)
        return TemplateRegion(
            name=default_region_name(index),
            color=COLOR_PALETTE[index % len(COLOR_PALETTE)],
            x0=max(0.0, min(1.0, nx0)),
            y0=max(0.0, min(1.0, ny0)),
            x1=max(0.0, min(1.0, nx1)),
            y1=max(0.0, min(1.0, ny1)),
        )

    def _render_template_canvas(self, lock_pan: bool = False) -> None:
        canvas = self.template_canvas
        canvas.delete("all")
        if self.template_source_image is None:
            return

        canvas_width = max(canvas.winfo_width(), 100)
        canvas_height = max(canvas.winfo_height(), 100)
        image = self.template_source_image.copy()
        base_scale = min((canvas_width - 20) / image.width, (canvas_height - 20) / image.height)
        scale = base_scale * self.template_zoom
        scale = max(scale, 0.05)
        display_width = max(1, int(image.width * scale))
        display_height = max(1, int(image.height * scale))
        cache_key = (id(self.template_source_image), display_width, display_height)
        if self.template_preview_image is None or self.template_preview_cache_key != cache_key:
            resized = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.template_preview_image = ImageTk.PhotoImage(resized)
            self.template_preview_cache_key = cache_key

        self.template_scale = scale
        self.template_display_width = display_width
        self.template_display_height = display_height
        if not self.template_manual_pan and not lock_pan:
            self.template_offset_x = (canvas_width - display_width) // 2
            self.template_offset_y = (canvas_height - display_height) // 2
        else:
            min_x = canvas_width - display_width - 20
            min_y = canvas_height - display_height - 20
            self.template_offset_x = min(20, max(min_x, self.template_offset_x))
            self.template_offset_y = min(20, max(min_y, self.template_offset_y))

        canvas.create_image(self.template_offset_x, self.template_offset_y, image=self.template_preview_image, anchor="nw")
        for region in self.template_regions:
            x0 = self.template_offset_x + region.x0 * display_width
            y0 = self.template_offset_y + region.y0 * display_height
            x1 = self.template_offset_x + region.x1 * display_width
            y1 = self.template_offset_y + region.y1 * display_height
            canvas.create_rectangle(x0, y0, x1, y1, outline=region.color, fill=region.color, width=2, stipple="gray25")
            canvas.create_text(x0 + 8, y0 + 8, text=region.name, fill=region.color, anchor="nw", font=("Segoe UI", 10, "bold"))

    def _refresh_region_tree(self) -> None:
        self.region_tree.delete(*self.region_tree.get_children())
        selected_values: dict[str, str] = {}
        selected_confidences: dict[str, float] = {}
        selected_result = self._get_selected_template_result()
        if selected_result is not None:
            selected_values = selected_result.values
            selected_confidences = selected_result.confidences
        for region in self.template_regions:
            self.region_tree.insert(
                "",
                "end",
                values=(
                    region.color,
                    region.name,
                    selected_values.get(region.name, ""),
                    f"{selected_confidences.get(region.name, 0.0):.1f}",
                ),
            )
        self._refresh_debug_tree()

    def _edit_region_name(self, _: object) -> None:
        selected = self.region_tree.selection()
        if not selected:
            return
        index = self.region_tree.index(selected[0])
        region = self.template_regions[index]
        new_name = self._prompt_text("Имя атрибута", f"Введите имя для области {region.name}", region.name)
        if not new_name:
            return
        self.template_regions[index] = TemplateRegion(
            name=new_name.strip(),
            color=region.color,
            x0=region.x0,
            y0=region.y0,
            x1=region.x1,
            y1=region.y1,
        )
        self._refresh_region_tree()
        self._refresh_results_tree()
        self._render_template_canvas()

    def _delete_selected_region(self) -> None:
        selected = self.region_tree.selection()
        if not selected:
            return
        index = self.region_tree.index(selected[0])
        self.template_regions.pop(index)
        self._refresh_region_tree()
        self._refresh_results_tree()
        self._render_template_canvas()

    def _clear_regions(self) -> None:
        self.template_regions.clear()
        self.template_results.clear()
        self._refresh_region_tree()
        self._refresh_results_tree()
        self._render_template_canvas()
        self._refresh_debug_tree()

    def _start_template_parsing(self) -> None:
        folder = Path(self.template_folder_var.get().strip())
        if self.template_source_image is None:
            messagebox.showerror("Ошибка", "Сначала выберите шаблон JPEG.")
            return
        if not folder.exists():
            messagebox.showerror("Ошибка", "Папка с JPEG для парсинга не найдена.")
            return
        if not self.template_regions:
            messagebox.showerror("Ошибка", "Нужно создать хотя бы одну область поиска.")
            return

        self.template_status_var.set("Идет OCR-парсинг по шаблону... Подготовка OCR backend.")
        self.template_parse_button.config(state="disabled")
        worker = threading.Thread(target=self._run_template_worker, args=(folder,), daemon=True)
        worker.start()

    def _run_template_worker(self, folder: Path) -> None:
        try:
            ocr_config = OcrConfig(
                mode="required",
                tesseract_cmd=self.template_tesseract_var.get().strip() or None,
                languages=self.template_ocr_lang_var.get().strip() or "eng+rus",
                psm=6,
            )
            backend, backend_message = get_template_ocr_backend_info(ocr_config)
            if backend == "none":
                self.worker_queue.put(("template_error", backend_message))
                return
            self.worker_queue.put(("template_status", f"Идет OCR-парсинг по шаблону... Backend: {backend_message}."))
            rows = parse_template_batch(
                image_folder=folder,
                regions=self.template_regions,
                ocr_config=ocr_config,
                recursive=self.template_recursive_var.get(),
            )
            self.worker_queue.put(("template_done", rows))
        except Exception as exc:
            self.worker_queue.put(("template_error", str(exc)))

    def _refresh_results_tree(self) -> None:
        columns = ["file_name"]
        for region in self.template_regions:
            columns.append(region.name)
            columns.append(f"{region.name}_conf")
        self.results_tree.delete(*self.results_tree.get_children())
        self.results_tree["columns"] = columns
        for column in columns:
            self.results_tree.heading(column, text=column)
            width = 180 if column == "file_name" else 110
            if column.endswith("_conf"):
                width = 80
            self.results_tree.column(column, width=width, anchor="w")

        for row in self.template_results:
            values = [row.file_name]
            for region in self.template_regions:
                values.append(row.values.get(region.name, ""))
                values.append(f"{row.confidences.get(region.name, 0.0):.1f}")
            self.results_tree.insert("", "end", values=values)
        self._refresh_region_tree()
        self._refresh_debug_tree()

    def _on_result_row_select(self, _: object) -> None:
        self._refresh_region_tree()
        self._refresh_debug_tree()

    def _get_selected_template_result(self) -> ParsedSheet | None:
        selected = self.results_tree.selection()
        if not selected:
            return self.template_results[0] if self.template_results else None
        index = self.results_tree.index(selected[0])
        if 0 <= index < len(self.template_results):
            return self.template_results[index]
        return None

    def _get_selected_region_name(self) -> str | None:
        selected = self.region_tree.selection()
        if selected:
            values = self.region_tree.item(selected[0], "values")
            if len(values) >= 2:
                return str(values[1])
        if self.template_regions:
            return self.template_regions[0].name
        return None

    def _on_region_tree_select(self, _: object) -> None:
        self._refresh_debug_tree()

    def _refresh_debug_tree(self) -> None:
        self.debug_tree.delete(*self.debug_tree.get_children())
        selected_result = self._get_selected_template_result()
        region_name = self._get_selected_region_name()
        if selected_result is None or not region_name:
            return
        candidates = selected_result.debug_candidates.get(region_name, [])
        for candidate in candidates:
            self.debug_tree.insert(
                "",
                "end",
                values=(
                    candidate.variant_name,
                    candidate.backend,
                    candidate.text,
                    f"{candidate.confidence:.1f}",
                    f"{candidate.score:.1f}",
                ),
            )

    def _export_template_results(self) -> None:
        if not self.template_results:
            messagebox.showerror("Ошибка", "Нет распознанных данных для экспорта.")
            return
        default_path = self.template_export_var.get().strip() or str(Path.cwd() / "parsed_dimensions.xlsx")
        path = filedialog.asksaveasfilename(
            title="Сохранить Excel",
            defaultextension=".xlsx",
            initialfile=Path(default_path).name,
            initialdir=str(Path(default_path).parent),
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        export_results_to_excel(self.template_results, self.template_regions, Path(path))
        self.template_export_var.set(path)
        self.template_status_var.set(f"Excel сохранен: {path}")

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "grouping_done":
                    result, output_dir = payload
                    self.group_result = result
                    self.group_output_dir = output_dir
                    self._fill_group_tree(result)
                    self.group_status_var.set(f"Готово: {len(result.groups)} групп. OCR active: {'yes' if result.ocr_enabled else 'no'}.")
                    self.group_start_button.config(state="normal")
                elif event == "grouping_error":
                    self.group_status_var.set(f"Ошибка: {payload}")
                    self.group_start_button.config(state="normal")
                    messagebox.showerror("Ошибка группировки", str(payload))
                elif event == "template_done":
                    self.template_results = payload
                    self._refresh_results_tree()
                    self.template_status_var.set(f"Парсинг завершен. Обработано файлов: {len(self.template_results)}.")
                    self.template_parse_button.config(state="normal")
                elif event == "template_status":
                    self.template_status_var.set(str(payload))
                elif event == "template_error":
                    self.template_status_var.set(f"Ошибка: {payload}")
                    self.template_parse_button.config(state="normal")
                    messagebox.showerror("Ошибка OCR", str(payload))
        except queue.Empty:
            pass
        self.after(120, self._poll_worker_queue)

    def _fill_group_tree(self, result: ClusterResult) -> None:
        self.group_tree.delete(*self.group_tree.get_children())
        self.group_path_index.clear()
        for group_index, group in enumerate(result.groups, start=1):
            group_id = self.group_tree.insert("", "end", text=f"Group {group_index:03d} ({len(group)})", values=("group",))
            for item in group:
                item_id = self.group_tree.insert(group_id, "end", text=item.name, values=("file",))
                self.group_path_index[item_id] = item
        first_group = self.group_tree.get_children()
        if first_group:
            self.group_tree.item(first_group[0], open=True)

    def _prompt_text(self, title: str, label: str, initial_value: str) -> str | None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        value = tk.StringVar(value=initial_value)
        ttk.Label(dialog, text=label).pack(fill="x", padx=12, pady=(12, 4))
        entry = ttk.Entry(dialog, textvariable=value, width=36)
        entry.pack(fill="x", padx=12, pady=(0, 12))
        entry.focus_set()

        result: dict[str, str | None] = {"value": None}

        def submit() -> None:
            result["value"] = value.get()
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(buttons, text="OK", command=submit).pack(side="right")
        ttk.Button(buttons, text="Отмена", command=cancel).pack(side="right", padx=(0, 8))
        dialog.wait_window()
        return result["value"]


def main() -> None:
    app = JpegParsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
