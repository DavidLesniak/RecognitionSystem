"""Wybór, edycja i klasyfikacja zdjęć."""

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import hashlib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from classifier_adapter import ClassifierAdapter, V3_MODEL
from image_processing import EditSettings, apply_adjustments, experiment_metadata, save_experiment
from modul_weryfikacji.v3.ocena_predykcji_v3 import evaluate_prediction


DECISION_LABELS = {
    "ACCEPT": "wynik spełnia progi akceptacji",
    "WARN": "wynik wymaga ostrożności",
    "REJECT": "niska wiarygodność",
}
DECISION_COLORS = {
    "ACCEPT": ("#e7f4eb", "#205b39"),
    "WARN": ("#fff3d4", "#704b00"),
    "REJECT": ("#fae8e6", "#8c2f28"),
}
NO_PREDICTION = "Kliknij „Klasyfikuj zdjęcie”, aby zobaczyć wynik."
SIGN_LABELS_PATH = V3_MODEL.with_name("etykiety_znaki.csv")
WORKSPACE = "#edf2f7"
SURFACE = "#ffffff"
INK = "#17283d"
MUTED = "#43546a"
EDGE = "#d4deea"
ACCENT = "#2059bd"


def read_image(path):
    """Odczytaj obraz, także ze ścieżki zawierającej polskie znaki."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if not data.size:
            raise ValueError("Plik jest pusty.")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        raise ValueError("Nie można otworzyć lub odczytać pliku.") from error
    if image is None:
        raise ValueError("Plik jest uszkodzony lub ma nieobsługiwany format obrazu.")
    return image


def make_preview(image, max_width=850, max_height=480):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1:
        return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    return image


def load_sign_labels(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if not {"Znak", "Etykieta"}.issubset(reader.fieldnames or ()):
            raise ValueError("Plik etykiet musi zawierać kolumny Znak i Etykieta.")
        return {row["Znak"].strip(): row["Etykieta"].strip()
                for row in reader if row["Znak"] and row["Etykieta"]}


def configure_styles(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=WORKSPACE)
    style.configure("TFrame", background=SURFACE)
    style.configure("TLabel", background=SURFACE, foreground=INK)
    style.configure("Workspace.TFrame", background=WORKSPACE)
    style.configure("Workspace.TLabel", background=WORKSPACE, foreground=MUTED)
    style.configure("Title.TLabel", background=WORKSPACE, foreground=INK, font=("Segoe UI", 19, "bold"))
    style.configure("TLabelframe", background=SURFACE, bordercolor=EDGE,
                    lightcolor=SURFACE, darkcolor=SURFACE, relief="solid")
    style.configure("TLabelframe.Label", background=SURFACE, foreground=INK, font=("Segoe UI", 11, "bold"))
    style.configure("Preview.TLabel", background="#f4f7fb", foreground=MUTED, font=("Segoe UI", 11))
    style.configure("FilterValue.TLabel", background="#f2f5f8", foreground=INK,
                    padding=(4, 2), font=("Segoe UI", 9, "bold"))
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("TCheckbutton", background=SURFACE, foreground=INK)
    style.map("TCheckbutton", background=[("active", SURFACE), ("disabled", SURFACE)])
    style.configure("TCombobox", padding=(4, 2))
    style.configure("TButton", padding=(9, 5), foreground=INK, background=SURFACE,
                    bordercolor=EDGE, lightcolor=SURFACE, darkcolor=SURFACE, focuscolor=ACCENT)
    style.map("TButton", background=[("disabled", "#f1f5f9"), ("pressed", "#dbe7f8"),
                                     ("active", "#e5eefb")],
              foreground=[("disabled", "#6e7f95")])
    style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), foreground=SURFACE,
                    background=ACCENT, bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                    focuscolor=SURFACE)
    style.map("Primary.TButton",
              background=[("disabled", "#dce6f5"), ("pressed", "#133c88"), ("active", "#184da7")],
              foreground=[("disabled", "#536a89")],
              bordercolor=[("disabled", "#dce6f5"), ("pressed", "#133c88"), ("active", "#184da7")],
              lightcolor=[("disabled", "#dce6f5"), ("pressed", "#133c88"), ("active", "#184da7")],
              darkcolor=[("disabled", "#dce6f5"), ("pressed", "#133c88"), ("active", "#184da7")])


class DecisionPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Decyzja modelu", padding=12)
        self.message = ttk.Label(self, justify="left", anchor="nw", style="Muted.TLabel")
        self.body = ttk.Frame(self)
        self.body.pack(fill="x")
        # Stałe miejsce na nazwę i drugi wynik także przy dłuższych etykietach.
        self.sign = tk.Label(self.body, font=("Segoe UI", 12, "bold"), height=3,
                             background=SURFACE, foreground=INK, anchor="nw",
                             justify="left", borderwidth=0, padx=0, pady=0)
        self.sign.pack(fill="x", pady=(0, 6))

        verdict = ttk.Frame(self.body)
        verdict.pack(fill="x")
        self.badge = tk.Label(verdict, font=("Segoe UI", 9, "bold"), padx=8, pady=3)
        self.badge.pack(side="left")
        self.explanation = ttk.Label(verdict)
        self.explanation.pack(side="left", padx=(8, 0))

        metrics = ttk.Frame(self.body)
        metrics.pack(fill="x", pady=10)
        self.values = {}
        for column, title in enumerate(("Pewność", "Margines", "Entropia")):
            metrics.columnconfigure(column, weight=1, uniform="metric")
            tile = tk.Frame(metrics, background="#f2f5f8", highlightbackground="#dbe2e9",
                            highlightthickness=1, padx=7, pady=5)
            tile.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column < 2 else 0)
            tk.Label(tile, text=title, background="#f2f5f8", foreground="#43546a").pack(anchor="w")
            value = tk.Label(tile, background="#f2f5f8", foreground="#17283d",
                             font=("Segoe UI", 12, "bold"))
            value.pack(anchor="w")
            self.values[title] = value

        self.alternative = tk.Label(self.body, font="TkDefaultFont", height=2,
                                    background=SURFACE, foreground=INK, anchor="nw",
                                    justify="left", borderwidth=0, padx=0, pady=0)
        self.alternative.pack(fill="x")
        self.bind("<Configure>", self.wrap_text)
        self.show_message(NO_PREDICTION)

    def wrap_text(self, event):
        if event.width > 32:
            for label in (self.message, self.sign, self.alternative):
                label.configure(wraplength=event.width - 30)

    def show_message(self, message):
        self.message.configure(text=message)
        self.message.place(relwidth=1, relheight=1)
        self.message.lift()

    def show_result(self, prediction, assessment, class_names, sign_labels):
        def sign_text(code):
            name = sign_labels.get(code)
            return f"{code} — {name}" if name else code

        decision = assessment["decision"]
        background, foreground = DECISION_COLORS[decision]
        self.sign.configure(text=sign_text(prediction.class_name))
        self.badge.configure(text=decision, background=background, foreground=foreground)
        self.explanation.configure(text=DECISION_LABELS[decision])
        self.values["Pewność"].configure(text=f'{assessment["confidence"]:.1%}'.replace(".", ","))
        self.values["Margines"].configure(text=f'{assessment["margin"] * 100:.1f}'.replace(".", ",") + " p.p.")
        self.values["Entropia"].configure(text=f'{assessment["entropy"]:.3f}'.replace(".", ","))

        if decision in ("WARN", "REJECT"):
            probabilities = prediction.probabilities
            runner_up = max((index for index in range(len(probabilities))
                             if index != prediction.class_index), key=probabilities.__getitem__)
            probability = f"{probabilities[runner_up]:.1%}".replace(".", ",")
            self.alternative.configure(text=f"Drugi wynik: {sign_text(class_names[runner_up])} ({probability})")
        else:
            self.alternative.configure(text="")

        self.message.place_forget()


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.image = self.result = self.source_path = self.applied_settings = None
        self.source_sha256 = None
        self.photo = None
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.classifier = None
        self.sign_labels = None
        self.classification_executor = ThreadPoolExecutor(max_workers=1)
        self.classification_future = None
        self.classification_job = None
        self.resize_job = None
        self.edit_job = None
        self.resetting = False
        self.last_observed_settings = None
        root.title("RecognitionSystem — rozpoznawanie znaków drogowych")
        root.geometry("1150x800")
        root.minsize(900, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)
        configure_styles(root)

        content = ttk.Frame(root, padding=16, style="Workspace.TFrame")
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Rozpoznawanie znaków drogowych", style="Title.TLabel").pack(anchor="w")
        review = ttk.Frame(content, style="Workspace.TFrame")
        review.pack(fill="both", expand=True, pady=(12, 0))
        review.columnconfigure(0, weight=6, uniform="review")
        review.columnconfigure(1, weight=5, minsize=385, uniform="review")
        review.rowconfigure(0, weight=1)
        panel = ttk.LabelFrame(review, text="Podgląd zdjęcia", padding=10)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        self.preview = ttk.Label(panel, text="Nie wybrano zdjęcia", anchor="center", style="Preview.TLabel")
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.bind("<Configure>", self.schedule_preview)

        sidebar = ttk.Frame(review, style="Workspace.TFrame")
        sidebar.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.open_button = ttk.Button(sidebar, text="Wybierz zdjęcie", command=self.choose_image)
        self.open_button.pack(anchor="w")
        self.info = tk.StringVar(value="Wybierz zdjęcie, aby rozpocząć.")
        info_label = ttk.Label(sidebar, textvariable=self.info, wraplength=370, style="Workspace.TLabel")
        info_label.pack(fill="x", pady=(5, 10))
        self.decision_panel = DecisionPanel(sidebar)
        self.decision_panel.pack(fill="x", pady=(0, 10))

        controls = ttk.LabelFrame(sidebar, text="Filtry", padding=12)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        self.variables, self.widgets = {}, []
        defaults = asdict(EditSettings())
        specs = [("brightness", "Jasność", -100, 100, 1),
                 ("contrast", "Kontrast", 0.5, 2, 0.05),
                 ("saturation", "Nasycenie", 0, 2, 0.05),
                 ("gamma", "Gamma", 0.3, 3, 0.05),
                 ("sharpen", "Wyostrzanie", 0, 2, 0.05)]
        for row, (key, title, low, high, step) in enumerate(specs):
            ttk.Label(controls, text=title).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=defaults[key])
            self.variables[key] = var
            slider = tk.Scale(controls, from_=low, to=high, resolution=step,
                              orient="horizontal", variable=var, showvalue=False,
                              width=11, sliderlength=18, borderwidth=1,
                              relief="flat", sliderrelief="raised", highlightthickness=0,
                              background=SURFACE, troughcolor="#dbe5f2", activebackground="#acc9f1")
            slider.grid(row=row, column=1, sticky="ew", padx=10, pady=2)
            value = tk.StringVar(value=f"{var.get():.2f}")
            ttk.Label(controls, textvariable=value, width=6, anchor="e",
                      style="FilterValue.TLabel").grid(row=row, column=2, sticky="e")
            var.trace_add("write", lambda *args, v=var, text=value: self.changed(v, text))
            self.widgets.append(slider)
        ttk.Label(controls, text="Odszumianie").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self.denoise = tk.StringVar(value="Wyłączone")
        self.combo = ttk.Combobox(controls, textvariable=self.denoise,
                                 values=["Wyłączone", "Słabe", "Średnie", "Mocne"],
                                 state="readonly", width=15)
        self.combo.grid(row=5, column=1, sticky="w", padx=10, pady=(4, 0))
        self.denoise.trace_add("write", lambda *args: self.mark_dirty())
        self.clahe = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(controls, text="Lokalna poprawa kontrastu (CLAHE)",
                                variable=self.clahe, command=self.mark_dirty)
        check.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.widgets.append(check)
        ttk.Label(controls, text="Podgląd na żywo. Gamma poniżej 1 rozjaśnia.", style="Muted.TLabel").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
        actions = ttk.Frame(sidebar, style="Workspace.TFrame")
        actions.pack(fill="x", pady=(10, 6))
        self.classify_button = ttk.Button(actions, text="Klasyfikuj zdjęcie",
                                         command=self.start_classification, style="Primary.TButton")
        self.classify_button.pack(side="left", padx=(0, 8))
        self.reset_button = ttk.Button(actions, text="Przywróć oryginał", command=self.reset)
        self.reset_button.pack(side="left", padx=(0, 8))
        self.save_button = ttk.Button(actions, text="Zapisz wynik", command=self.save)
        self.save_button.pack(side="left")
        self.status = tk.StringVar(value="Gotowe.")
        status_label = ttk.Label(sidebar, textvariable=self.status, wraplength=370, style="Workspace.TLabel")
        status_label.pack(fill="x")
        def wrap_sidebar_text(event):
            info_label.configure(wraplength=event.width)
            status_label.configure(wraplength=event.width)

        sidebar.bind("<Configure>", wrap_sidebar_text)
        self.set_busy(False)

    def settings(self):
        values = {key: round(var.get(), 2) for key, var in self.variables.items()}
        values["brightness"] = int(values["brightness"])
        return EditSettings(**values, denoise={"Wyłączone": 0, "Słabe": 3, "Średnie": 7, "Mocne": 10}[
            self.denoise.get()], clahe=self.clahe.get())

    def changed(self, var, text):
        text.set(f"{var.get():.2f}")
        self.mark_dirty()

    def mark_dirty(self):
        if self.image is not None and not self.resetting:
            current = self.settings()
            # Tk może zapisać tę samą wartość suwaka po zmianie jego stanu.
            # Taki sygnał nie oznacza edycji i nie powinien uruchamiać obliczeń.
            if current == self.last_observed_settings:
                return
            self.last_observed_settings = current
            self.clear_classification()
            self.status.set("Aktualizowanie podglądu…")
            self.save_button.configure(state="disabled")
            if self.edit_job is not None:
                self.root.after_cancel(self.edit_job)
            self.edit_job = self.root.after(180, self.apply)
            self.update_classify_button()

    def set_busy(self, busy):
        self.open_button.configure(state="disabled" if busy else "normal")
        enabled = self.image is not None
        for widget in self.widgets:
            widget.configure(state="normal" if enabled else "disabled")
        self.combo.configure(state="readonly" if enabled else "disabled")
        self.reset_button.configure(state="normal" if enabled and not busy else "disabled")
        valid = enabled and not busy and self.edit_job is None and self.result is not None and self.settings() == self.applied_settings
        self.save_button.configure(state="normal" if valid else "disabled")
        self.update_classify_button()

    def update_classify_button(self):
        available = self.can_classify()
        self.classify_button.configure(state="normal" if available else "disabled")

    def can_classify(self):
        return (self.image is not None and self.result is not None
                and self.future is None and self.edit_job is None
                and self.settings() == self.applied_settings
                and self.classification_future is None)

    def clear_classification(self):
        self.decision_panel.show_message(NO_PREDICTION)

    def classify_image(self, image):
        if self.classifier is None:
            self.sign_labels = load_sign_labels(SIGN_LABELS_PATH)
            self.classifier = ClassifierAdapter.from_v3()
        prediction = self.classifier.predict(image)
        assessment = evaluate_prediction(prediction.probabilities)
        return prediction, assessment

    def start_classification(self):
        if not self.can_classify():
            return
        image = self.result
        self.decision_panel.show_message("Klasyfikowanie…")
        self.classification_future = self.classification_executor.submit(self.classify_image, image)
        self.update_classify_button()
        self.classification_job = self.root.after(50, self.poll_classification, image)

    def poll_classification(self, image):
        if not self.classification_future.done():
            self.classification_job = self.root.after(50, self.poll_classification, image)
            return
        future, self.classification_future = self.classification_future, None
        self.classification_job = None
        self.update_classify_button()
        if image is not self.result or not self.can_classify():
            return
        try:
            prediction, assessment = future.result()
        except Exception as error:
            self.decision_panel.show_message(f"Błąd klasyfikacji: {error}")
            return
        self.decision_panel.show_result(prediction, assessment, self.classifier.class_names, self.sign_labels)

    def choose_image(self):
        path = filedialog.askopenfilename(parent=self.root, title="Wybierz zdjęcie", filetypes=[
            ("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.ppm"),
            ("Wszystkie pliki", "*.*")])
        if not path:
            return
        try:
            data = Path(path).read_bytes()
            if not data:
                raise ValueError("Plik jest pusty.")
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Plik jest uszkodzony lub ma nieobsługiwany format obrazu.")
        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror("Błąd odczytu", str(error), parent=self.root)
            return
        self.image, self.source_path = image, Path(path)
        self.source_sha256 = hashlib.sha256(data).hexdigest()
        height, width = image.shape[:2]
        self.info.set(f"Plik: {self.source_path.name}  |  Wymiary: {width} × {height} px")
        self.reset()

    def reset(self):
        if self.image is None or self.future is not None:
            return
        self.clear_classification()
        defaults = asdict(EditSettings())
        if self.edit_job is not None:
            self.root.after_cancel(self.edit_job)
            self.edit_job = None
        self.resetting = True
        for key, var in self.variables.items():
            var.set(defaults[key])
        self.denoise.set("Wyłączone")
        self.clahe.set(False)
        self.resetting = False
        self.result = self.image.copy()
        self.applied_settings = EditSettings()
        self.last_observed_settings = self.applied_settings
        self.render_preview()
        self.status.set("Przywrócono oryginał i ustawienia początkowe.")
        self.set_busy(False)

    def apply(self):
        if self.edit_job is not None:
            self.root.after_cancel(self.edit_job)
            self.edit_job = None
        if self.image is None or self.future is not None:
            return
        self.pending_settings = self.settings()
        self.status.set("Przetwarzanie…")
        self.set_busy(True)
        self.future = self.executor.submit(apply_adjustments, self.image, self.pending_settings)
        self.root.after(50, self.poll_result)

    def poll_result(self):
        if not self.future.done():
            self.root.after(50, self.poll_result)
            return
        future, self.future = self.future, None
        # Pomiń stary wynik, jeśli w trakcie obliczeń przesunięto suwaki.
        if self.settings() != self.pending_settings:
            try:
                future.result()
            except Exception:
                pass
            self.apply()
            return
        try:
            result = future.result()
            self.result = result
            self.applied_settings = self.pending_settings
            self.render_preview()
            self.status.set("Zmiany zastosowane. Możesz zapisać wynik wraz z parametrami.")
        except Exception as error:
            self.status.set("Przetwarzanie nie powiodło się.")
            messagebox.showerror("Błąd przetwarzania", str(error), parent=self.root)
        self.set_busy(False)

    def schedule_preview(self, event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self.render_preview)

    def render_preview(self):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
            self.resize_job = None
        image = self.result
        if image is None:
            return
        preview = make_preview(image, max(1, self.preview.winfo_width() - 12),
                               max(1, self.preview.winfo_height() - 12))
        success, encoded = cv2.imencode(".png", preview)
        if not success:
            raise ValueError("Nie można przygotować podglądu.")
        self.photo = tk.PhotoImage(master=self.root, data=encoded.tobytes(), format="png")
        self.preview.configure(image=self.photo, text="")

    def save(self):
        if self.result is None or self.future is not None or self.settings() != self.applied_settings:
            return
        selected = filedialog.asksaveasfilename(parent=self.root, title="Zapisz wynik i parametry",
            defaultextension=".png", initialfile=f"{self.source_path.stem}_edited.png",
            filetypes=[("Obraz PNG", "*.png")], confirmoverwrite=False)
        if not selected:
            return
        path = Path(selected).with_suffix(".png")
        metadata_path = path.with_suffix(".json")
        if self.source_path.resolve() in (path.resolve(), metadata_path.resolve()):
            messagebox.showerror("Błąd zapisu",
                                "Wybierz inną nazwę — oryginał musi pozostać bez zmian.",
                                parent=self.root)
            return
        if (path.exists() or metadata_path.exists()) and not messagebox.askyesno(
                "Potwierdzenie zapisu", "Plik PNG lub JSON już istnieje. Nadpisać istniejące pliki tej pary?",
                parent=self.root):
            return
        metadata = experiment_metadata(self.source_path.name, self.source_sha256,
                                       self.result, self.applied_settings)
        try:
            save_experiment(path, self.result, metadata)
        except (OSError, ValueError, cv2.error) as error:
            self.status.set("Zapis nie został ukończony.")
            messagebox.showerror("Błąd zapisu", str(error), parent=self.root)
            return
        self.status.set(f"Zapisano: {path.name} i {metadata_path.name}")

    def close(self):
        for job in (self.edit_job, self.resize_job, self.classification_job):
            if job is not None:
                self.root.after_cancel(job)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.classification_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    ImageViewer(window)
    window.mainloop()
