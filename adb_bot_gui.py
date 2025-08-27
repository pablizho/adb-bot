import yaml
import tempfile
import uuid
import shutil
import threading
import subprocess
import sys as _sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Импорт API и функций из ядра
import adb_bot as core


class _DualWriter:
    """Дублирует stdout/stderr в Text виджет логов (потокобезопасно)."""
    def __init__(self, widget, orig):
        self.widget = widget
        self.orig = orig

    def write(self, s):
        try:
            self.orig.write(s)
        except Exception:
            pass
        if self.widget:
            self.widget.after(0, lambda: (self.widget.insert("end", s),
                                          self.widget.see("end")))

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ADB Bot — автотесты Android")
        self.geometry("980x620")
        self.minsize(900, 560)

        self._player = None
        self._player_serial = None
        self._ap_running = False

        self._ap_once_thread = None
        self._ap_once_stop = None




        # --------- переменные состояния ---------
        self.serial_var = tk.StringVar(value="(авто)")
        self.live_var = tk.BooleanVar(value=False)

        self.scenario_path_var = tk.StringVar(value="")
        self.grid_path_var = tk.StringVar(value=os.path.join("templates", "grid.yaml"))

        self.rows_var = tk.IntVar(value=5)
        self.cols_var = tk.IntVar(value=7)
        self.mode_var = tk.StringVar(value="swipe")      # swipe | tap
        self.max_moves_var = tk.IntVar(value=200)
        self.delay_var = tk.DoubleVar(value=0.30)

        self._build_ui()
        self._wire_logs()

        # начальный список устройств
        self.refresh_devices()

    # ================= UI =================
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        # --- строка 1: выбор устройства и глобальные действия ---
        row1 = ttk.Frame(root)
        row1.pack(fill="x", pady=(0, 8))

        ttk.Label(row1, text="Устройство:").pack(side="left")
        self.device_combo = ttk.Combobox(
            row1, textvariable=self.serial_var, state="readonly", width=28
        )
        self.device_combo.pack(side="left", padx=(6, 8))
        ttk.Button(row1, text="Обновить список", command=self.refresh_devices).pack(side="left")

        ttk.Separator(row1, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Checkbutton(row1, text="LIVE при записи (повторять клики на устройстве)",
                        variable=self.live_var).pack(side="left")

        # --- строка 2: пути файлов (всегда на виду) ---
        files = ttk.LabelFrame(root, text="Файлы", padding=8)
        files.pack(fill="x", pady=(0, 8))

        # сценарий
        ttk.Label(files, text="Сценарий YAML:").grid(row=0, column=0, sticky="e")
        ttk.Entry(files, textvariable=self.scenario_path_var).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(files, text="Обзор…", command=self._browse_scenario).grid(row=0, column=2, padx=(0, 8))

        # grid
        ttk.Label(files, text="Grid (сеткa) YAML:").grid(row=1, column=0, sticky="e", pady=(6, 0))
        ttk.Entry(files, textvariable=self.grid_path_var).grid(row=1, column=1, sticky="we", padx=6, pady=(6, 0))
        ttk.Button(files, text="Обзор…", command=self._browse_grid).grid(row=1, column=2, padx=(0, 8), pady=(6, 0))

        files.columnconfigure(1, weight=1)

        # --- строка 3: сценарий (кнопки всегда видны) ---
        scen = ttk.LabelFrame(root, text="Сценарий", padding=8)
        scen.pack(fill="x", pady=(0, 8))

        ttk.Button(scen, text="Записать сценарий…", command=self.on_record, width=20).pack(side="left")
        ttk.Button(scen, text="Воспроизвести сценарий…", command=self.on_play, width=22).pack(side="left", padx=8)
        ttk.Button(scen, text="Редактировать сценарий…", command=self.on_edit, width=18).pack(side="left")


        # --- строка 3.5: подсказки по горячим клавишам ---
        cheats = ttk.LabelFrame(root, text="Подсказки по горячим клавишам (в окне записи)", padding=8)
        cheats.pack(fill="x", pady=(0, 8))

        help_text = (
            "ЛКМ — тап; перетаскивание — свайп\n"
            "1..9 — ждать N секунд\n"
            "i — выделить область и добавить шаг wait_image\n"
            "m — один ход Match-3\n"
            "a — Match-3 автоплей (ограниченное число ходов)\n"
            "f — старт фонового автоплея; g — стоп фонового автоплея\n"
            "u — автоплей до появления/исчезновения картинки (после выделения)\n"
            "c — отменить последний шаг\n"
            "s — сохранить сценарий и выйти из записи\n"
            "q — выйти без сохранения\n"
            "h — показать/скрыть подсказку в окне записи"
        )
        lbl = ttk.Label(cheats, text=help_text, justify="left")
        lbl.pack(anchor="w")

        # --- строка 4: панель Match-3 (все действия под рукой) ---
        m3 = ttk.LabelFrame(root, text="Match-3 (поле/ходы/автоплей)", padding=8)
        m3.pack(fill="x")

        ttk.Label(m3, text="Строк:").grid(row=0, column=0, sticky="e")
        ttk.Entry(m3, textvariable=self.rows_var, width=6).grid(row=0, column=1, sticky="w", padx=(4, 10))

        ttk.Label(m3, text="Столбцов:").grid(row=0, column=2, sticky="e")
        ttk.Entry(m3, textvariable=self.cols_var, width=6).grid(row=0, column=3, sticky="w", padx=(4, 10))

        ttk.Label(m3, text="Режим:").grid(row=0, column=4, sticky="e")
        ttk.Combobox(m3, values=["swipe", "tap"], textvariable=self.mode_var,
                     width=8, state="readonly").grid(row=0, column=5, sticky="w", padx=(4, 10))

        ttk.Label(m3, text="Макс. ходов:").grid(row=0, column=6, sticky="e")
        ttk.Entry(m3, textvariable=self.max_moves_var, width=8).grid(row=0, column=7, sticky="w", padx=(4, 10))

        ttk.Label(m3, text="Пауза, c:").grid(row=0, column=8, sticky="e")
        ttk.Entry(m3, textvariable=self.delay_var, width=8).grid(row=0, column=9, sticky="w", padx=(4, 10))

        # Кнопки действий
        ttk.Button(m3, text="Калибровать сетку…", command=self.on_calibrate).grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="we")
        ttk.Button(m3, text="Сделать 1 ход", command=self.on_match3).grid(row=1, column=2, columnspan=2, pady=(8, 0), sticky="we", padx=(8, 0))
        ttk.Button(m3, text="Автоплей (огр. ходов)…", command=self.on_autoplay).grid(row=1, column=4, columnspan=3, pady=(8, 0), sticky="we", padx=(8, 0))
        self.ap_once_stop_btn = ttk.Button(m3, text="Прервать автоплей (огр.)", command=self.on_autoplay_stop)
        self.ap_once_stop_btn.grid(row=1, column=7, columnspan=2, pady=(8, 0), sticky="we", padx=(8, 0))
        self.ap_once_stop_btn.state(["disabled"])

        # НОВОЕ: фоновые старт/стоп
        self.ap_start_btn = ttk.Button(m3, text="Старт автоплея (фон)", command=self.on_ap_start)
        self.ap_start_btn.grid(row=2, column=0, columnspan=4, pady=(8, 0), sticky="we")

        self.ap_stop_btn = ttk.Button(m3, text="Стоп автоплея", command=self.on_ap_stop)
        self.ap_stop_btn.grid(row=2, column=4, columnspan=3, pady=(8, 0), sticky="we", padx=(8, 0))
        self.ap_stop_btn.state(["disabled"])  # по умолчанию выключена

        for col in range(10):
            m3.columnconfigure(col, weight=1 if col in (1,3,5,7,9) else 0)

        # --- строка 5: лог + утилиты ---
        log_box = ttk.LabelFrame(root, text="Лог выполнения", padding=8)
        log_box.pack(fill="both", expand=True, pady=(8, 0))

        tools = ttk.Frame(log_box)
        tools.pack(fill="x", pady=(0, 6))
        ttk.Button(tools, text="Очистить лог", command=self._clear_log).pack(side="left")
        ttk.Button(tools, text="Скопировать в буфер", command=self._copy_log).pack(side="left", padx=8)

        self.log = tk.Text(log_box, wrap="word", height=16)
        self.log.pack(fill="both", expand=True)
        self.log.insert("end",
                        "Готово. Выберите устройство, задайте пути к файлам и используйте кнопки сверху.\n"
                        "Во время записи откроется окно предпросмотра со всеми горячими клавишами.\n")

    def _wire_logs(self):
        # Дублируем stdout/stderr в окно лога
        _sys.stdout = _DualWriter(self.log, _sys.__stdout__)
        _sys.stderr = _DualWriter(self.log, _sys.__stderr__)

    # ================= helpers =================
    def _selected_serial(self):
        val = (self.serial_var.get() or "").strip()
        return None if val in {"", "(авто)"} else val

    def _run_async(self, fn, *args, **kwargs):
        t = threading.Thread(target=self._wrap_run, args=(fn,) + args, kwargs=kwargs, daemon=True)
        t.start()

    def _wrap_run(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("ADB", f"Ошибка ADB: {e}")
            print("ADB error:", e)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print("Error:", e)

    def _ensure_path_dir(self, path: str):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # ================= actions =================
    def refresh_devices(self):
        try:
            out = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True).stdout
            devices = []
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[-1] == "device" and not parts[0].startswith("List"):
                    devices.append(parts[0])
            values = ["(авто)"] + devices
            self.device_combo["values"] = values
            if self.serial_var.get() not in values:
                self.serial_var.set(values[0])
            print(f"Устройства: {devices}\n")
        except Exception as e:
            messagebox.showerror("ADB", f"Не удалось получить список устройств: {e}")

    def _browse_scenario(self):
        path = filedialog.askopenfilename(
            title="Выбрать сценарий YAML",
            filetypes=[("YAML", "*.yaml"), ("Все файлы", "*.*")]
        )
        if path:
            self.scenario_path_var.set(path)

    def _browse_grid(self):
        path = filedialog.askopenfilename(
            title="Выбрать grid.yaml",
            filetypes=[("YAML", "*.yaml"), ("Все файлы", "*.*")]
        )
        if path:
            self.grid_path_var.set(path)

    def on_record(self):
        # всегда спрашиваем файл для записи
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml")],
            title="Сохранить сценарий как…",
            initialfile=(self.scenario_path_var.get().strip() or "scenario.yaml"),
        )
        if not path:
            return
        self.scenario_path_var.set(path)
        self._ensure_path_dir(path)

        serial = self._selected_serial()
        live = self.live_var.get()

        # Запускаем отдельный процесс: python adb_bot.py record ...
        adb_bot_py = os.path.join(os.path.dirname(__file__), "adb_bot.py")
        cmd = [_sys.executable, "-u", adb_bot_py, "record", "--out", path]
        if serial:
            cmd += ["--serial", serial]
        if live:
            cmd += ["--live"]

        print("Запуск записи в отдельном процессе:\n ", " ".join(cmd), "\n")

        def task():
            # Читаем stdout подпроцесса и транслируем в лог GUI
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,           # = universal_newlines=True
                bufsize=1            # построчный вывод
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    # отправляем в лог GUI (через _DualWriter это попадёт в Text)
                    print(line, end="")
            finally:
                rc = proc.wait()
                print(f"Запись завершена (код выхода {rc})\n")

        self._run_async(task)



    def on_play(self):
        path = self.scenario_path_var.get().strip()
        if not path or not os.path.exists(path):
            path = filedialog.askopenfilename(
                title="Открыть сценарий YAML",
                filetypes=[("YAML", "*.yaml"), ("Все файлы", "*.*")]
            )
            if not path:
                return
            self.scenario_path_var.set(path)

        serial = self._selected_serial()
        print(f"Воспроизведение {path} (устройство={serial or 'авто'})\n")

        def task():
            adb = core.ADB(serial=serial)
            core.Player(adb).play(path)
            print("Воспроизведение завершено\n")

        self._run_async(task)

    def on_calibrate(self):
        grid_path = self.grid_path_var.get().strip()
        if not grid_path:
            grid_path = filedialog.asksaveasfilename(
                defaultextension=".yaml",
                filetypes=[("YAML", "*.yaml")],
                title="Сохранить grid.yaml как…"
            )
            if not grid_path:
                return
            self.grid_path_var.set(grid_path)

        self._ensure_path_dir(grid_path)
        rows, cols = self.rows_var.get(), self.cols_var.get()
        serial = self._selected_serial()
        print(f"Калибровка сетки {rows}×{cols} → {grid_path} (устройство={serial or 'авто'})\n")

        def task():
            adb = core.ADB(serial=serial)
            core.calibrate_grid(adb, rows=rows, cols=cols, out_path=grid_path)
            print("Калибровка завершена\n")

        self._run_async(task)

    def on_match3(self):
        grid = self.grid_path_var.get().strip()
        if not grid or not os.path.exists(grid):
            messagebox.showwarning("Match-3", "Укажите корректный путь к grid.yaml.")
            return
        serial = self._selected_serial()
        mode = self.mode_var.get()
        print(f"Match-3: один ход (grid={grid}, режим={mode}, устройство={serial or 'авто'})\n")

        def task():
            adb = core.ADB(serial=serial)
            ok = core.match3_once(adb, grid_yaml=grid, action_mode=mode)
            print("Ход выполнен\n" if ok else "Ход не найден\n")

        self._run_async(task)

    def on_autoplay(self):
        grid = self.grid_path_var.get().strip()
        if not grid or not os.path.exists(grid):
            messagebox.showwarning("Автоплей", "Укажите корректный путь к grid.yaml.")
            return
        serial = self._selected_serial()
        mode = self.mode_var.get()
        max_moves = self.max_moves_var.get()
        delay = self.delay_var.get()

        if self._ap_once_thread and self._ap_once_thread.is_alive():
            messagebox.showinfo("Автоплей", "Уже запущен автоплей с ограничением ходов.")
            return

        self._ap_once_stop = threading.Event()
        self.ap_once_stop_btn.state(["!disabled"])  # включаем кнопку «Прервать»

        print(f"Автоплей (ограниченный): grid={grid}, режим={mode}, макс. ходов={max_moves}, пауза={delay} c "
            f"(устройство={serial or 'авто'})\n")

        def run():
            try:
                adb = core.ADB(serial=serial)
                n = core.match3_autoplay(adb, grid_yaml=grid, mode=mode,
                                        max_moves=max_moves, delay=delay,
                                        stop_event=self._ap_once_stop)
                print(f"Автоплей завершён, сделано ходов: {n}\n")
            finally:
                # по завершении (или прерывании) выключаем кнопку «Прервать»
                self.ap_once_stop_btn.state(["disabled"])
                self._ap_once_thread = None
                self._ap_once_stop = None

        self._ap_once_thread = threading.Thread(target=run, daemon=True)
        self._ap_once_thread.start()

    def on_autoplay_stop(self):
        if self._ap_once_stop:
            self._ap_once_stop.set()
            print("Запрошена остановка автоплея (ограниченного)\n")



    def on_edit(self):
        path = self.scenario_path_var.get().strip()
        if not path or not os.path.exists(path):
            path = filedialog.askopenfilename(
                title="Открыть сценарий YAML",
                filetypes=[("YAML", "*.yaml"), ("Все файлы", "*.*")])
            if not path:
                return
            self.scenario_path_var.set(path)

        ScenarioEditor(self, path)

    def _player_for_serial(self):
        """Ленивая инициализация core.Player под выбранное устройство."""
        serial = self._selected_serial()
        if self._player is None or self._player_serial != serial:
            self._player_serial = serial
            self._player = core.Player(core.ADB(serial=serial))
        return self._player

    def _toggle_ap_buttons(self, running: bool):
        if running:
            self.ap_start_btn.state(["disabled"])
            self.ap_stop_btn.state(["!disabled"])
        else:
            self.ap_start_btn.state(["!disabled"])
            self.ap_stop_btn.state(["disabled"])

    def on_ap_start(self):
        grid = self.grid_path_var.get().strip()
        if not grid or not os.path.exists(grid):
            messagebox.showwarning("Автоплей", "Укажите корректный путь к grid.yaml.")
            return
        mode = self.mode_var.get()
        delay = self.delay_var.get()
        p = self._player_for_serial()
        ok = p._start_autoplay(grid, mode=mode, delay=delay)
        if ok:
            self._ap_running = True
            self._toggle_ap_buttons(True)
            print(f"Фоновый автоплей запущен (grid={grid}, режим={mode}, пауза={delay} c)\n")
        else:
            print("Автоплей уже запущен\n")

    def on_ap_stop(self):
        if not self._player:
            return
        self._player._stop_autoplay()
        self._ap_running = False
        self._toggle_ap_buttons(False)
        print("Фоновый автоплей остановлен\n")



    # --------- лог ---------
    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _copy_log(self):
        txt = self.log.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.update()  # чтобы буфер точно сохранился
        messagebox.showinfo("Буфер обмена", "Лог скопирован.")


class ScenarioEditor(tk.Toplevel):
    """
    Простой редактор: список шагов + кнопки:
    Вверх/Вниз, Дублировать, Удалить, Редактировать (YAML), Вставить фрагмент (запись), Сохранить.
    """
    def __init__(self, master, scenario_path: str):
        super().__init__(master)
        self.title(f"Редактор сценария — {os.path.basename(scenario_path)}")
        self.geometry("880x560")
        self.minsize(800, 500)

        self.scenario_path = scenario_path
        self.steps = self._load_steps(scenario_path)

        # UI
        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)

        # список
        self.listbox = tk.Listbox(top, activestyle="dotbox")
        self.listbox.pack(side="left", fill="both", expand=True)
        self._refresh_list()

        # правые кнопки
        btns = ttk.Frame(top)
        btns.pack(side="left", fill="y", padx=(8,0))

        ttk.Button(btns, text="Вверх ▲", command=self._move_up).pack(fill="x")
        ttk.Button(btns, text="Вниз ▼", command=self._move_down).pack(fill="x", pady=(6,0))
        ttk.Separator(btns, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(btns, text="Дублировать", command=self._duplicate).pack(fill="x")
        ttk.Button(btns, text="Удалить", command=self._delete).pack(fill="x", pady=(6,0))
        ttk.Separator(btns, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(btns, text="Редактировать YAML…", command=self._edit_yaml).pack(fill="x")
        ttk.Button(btns, text="Вставить фрагмент (запись)…", command=self._insert_fragment_record).pack(fill="x", pady=(6,0))
        ttk.Separator(btns, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(btns, text="Сохранить", command=self._save).pack(fill="x")
        ttk.Button(btns, text="Закрыть", command=self.destroy).pack(fill="x", pady=(6,0))

        # двойной клик = редактировать YAML
        self.listbox.bind("<Double-1>", lambda e: self._edit_yaml())

    # ---------- данные ----------
    def _load_steps(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                steps = yaml.safe_load(f) or []
            if not isinstance(steps, list):
                raise ValueError("Файл сценария должен содержать список шагов.")
            return steps
        except Exception as e:
            messagebox.showerror("Открытие сценария", f"Не удалось прочитать YAML:\n{e}")
            return []

    def _save(self):
        try:
            with open(self.scenario_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.steps, f, allow_unicode=True, sort_keys=False)
            messagebox.showinfo("Сценарий", "Сохранено.")
        except Exception as e:
            messagebox.showerror("Сохранение", str(e))

    # ---------- list ----------
    def _refresh_list(self):
        self.listbox.delete(0, "end")
        for i, st in enumerate(self.steps):
            self.listbox.insert("end", f"{i+1:02d}. {self._summarize(st)}")

    def _selected_index(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def _summarize(self, step: dict) -> str:
        t = step.get("type", "?")
        if t == "tap":
            return f"tap at {step.get('at')}"
        if t == "swipe":
            return f"swipe {step.get('from')} → {step.get('to')} (ms={step.get('ms',180)})"
        if t == "wait":
            return f"wait {step.get('seconds',1)}s"
        if t in ("wait_image","wait_disappear"):
            return f"{t} thr={step.get('threshold',0.9)} tpl={os.path.basename(step.get('template',''))}"
        if t.startswith("match3"):
            return f"{t}"
        return yaml.safe_dump(step, allow_unicode=True).strip().replace("\n"," ")

    # ---------- операции ----------
    def _move_up(self):
        i = self._selected_index()
        if i is None or i == 0: return
        self.steps[i-1], self.steps[i] = self.steps[i], self.steps[i-1]
        self._refresh_list()
        self.listbox.selection_set(i-1)

    def _move_down(self):
        i = self._selected_index()
        if i is None or i >= len(self.steps)-1: return
        self.steps[i+1], self.steps[i] = self.steps[i], self.steps[i+1]
        self._refresh_list()
        self.listbox.selection_set(i+1)

    def _duplicate(self):
        i = self._selected_index()
        if i is None: return
        self.steps.insert(i+1, yaml.safe_load(yaml.safe_dump(self.steps[i], allow_unicode=True)))
        self._refresh_list()
        self.listbox.selection_set(i+1)

    def _delete(self):
        i = self._selected_index()
        if i is None: return
        if messagebox.askyesno("Удалить шаг", "Удалить выбранный шаг?"):
            self.steps.pop(i)
            self._refresh_list()

    def _edit_yaml(self):
        i = self._selected_index()
        if i is None: return
        StepEditor(self, self.steps, i, on_saved=self._refresh_list)

    def _insert_fragment_record(self):
        """
        Запускает Recorder в подпроцессе, пишет во временный YAML,
        по завершении читает его и вставляет шаги в список.
        """
        # куда вставлять — после выбранного
        i = self._selected_index()
        insert_at = (i + 1) if i is not None else len(self.steps)

        # временный файл
        tmp_dir = tempfile.mkdtemp(prefix="adb_scn_")
        tmp_yaml = os.path.join(tmp_dir, f"frag_{uuid.uuid4().hex}.yaml")

        # собираем команду как в on_record
        adb_bot_py = os.path.join(os.path.dirname(__file__), "adb_bot.py")
        cmd = [_sys.executable, "-u", adb_bot_py, "record", "--out", tmp_yaml]
        # используем то же устройство и LIVE-настройку, что в главном окне
        app = self.master
        serial = app._selected_serial()
        if serial:
            cmd += ["--serial", serial]
        if app.live_var.get():
            cmd += ["--live"]

        # запускаем синхронно (блокируем редактор) —
        # хотим дождаться, пока пользователь завершит запись
        try:
            self._run_recorder_blocking(cmd)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            messagebox.showerror("Запись фрагмента", str(e))
            return

        # читаем результат и вставляем
        try:
            with open(tmp_yaml, "r", encoding="utf-8") as f:
                frag = yaml.safe_load(f) or []
            if not isinstance(frag, list):
                raise ValueError("Фрагмент должен быть списком шагов.")
            for off, st in enumerate(frag):
                self.steps.insert(insert_at + off, st)
            self._refresh_list()
            self.listbox.selection_set(insert_at)
        except Exception as e:
            messagebox.showerror("Вставка фрагмента", f"Не удалось прочитать фрагмент:\n{e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_recorder_blocking(self, cmd):
        """
        Запускает подпроцесс Recorder и стримит его stdout в маленькое окно-лог.
        Возвращается, когда запись завершена.
        """
        logwin = tk.Toplevel(self)
        logwin.title("Запись фрагмента — лог")
        logwin.geometry("720x360")
        txt = tk.Text(logwin, wrap="word")
        txt.pack(fill="both", expand=True)

        txt.insert("end", "Запущен рекордер:\n" + " ".join(cmd) + "\n\n")
        txt.see("end")
        self.update()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                txt.insert("end", line)
                txt.see("end")
                logwin.update()
        finally:
            rc = proc.wait()
            txt.insert("end", f"\nЗапись завершена (код {rc}).\n")
            txt.see("end")
            # не закрываем окно автоматически — пусть пользователь прочитает лог

class StepEditor(tk.Toplevel):
    """Редактирование одного шага как YAML-словаря."""
    def __init__(self, master, steps: list, index: int, on_saved=None):
        super().__init__(master)
        self.steps = steps
        self.index = index
        self.on_saved = on_saved

        self.title(f"Шаг {index+1}: редактирование")
        self.geometry("700x500")

        frm = ttk.Frame(self, padding=8)
        frm.pack(fill="both", expand=True)

        self.text = tk.Text(frm, wrap="none")
        self.text.pack(fill="both", expand=True)

        # текущий шаг -> YAML
        data = yaml.safe_dump(steps[index], allow_unicode=True, sort_keys=False)
        self.text.insert("1.0", data)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8,0))
        ttk.Button(btns, text="Проверить", command=self._validate).pack(side="left")
        ttk.Button(btns, text="OK", command=self._ok).pack(side="right")
        ttk.Button(btns, text="Отмена", command=self.destroy).pack(side="right", padx=6)

    def _validate(self):
        try:
            d = yaml.safe_load(self.text.get("1.0", "end"))
            if not isinstance(d, dict) or "type" not in d:
                raise ValueError("Должен быть словарь с ключом 'type'.")
            messagebox.showinfo("Проверка", "OK")
        except Exception as e:
            messagebox.showerror("Проверка", str(e))

    def _ok(self):
        try:
            d = yaml.safe_load(self.text.get("1.0", "end"))
            if not isinstance(d, dict) or "type" not in d:
                raise ValueError("Должен быть словарь с ключом 'type'.")
            self.steps[self.index] = d
            if self.on_saved:
                self.on_saved()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Сохранение шага", str(e))



if __name__ == "__main__":
    app = App()
    app.mainloop()
