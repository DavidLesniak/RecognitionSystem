"""Wybór, porównanie i edycja zdjęć."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from image_processing import EditSettings, apply_adjustments, experiment_metadata, save_experiment


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


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.image = self.result = self.source_path = self.applied_settings = None
        self.source_sha256 = None
        self.photos = []
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.resize_job = None
        self.edit_job = None
        self.resetting = False
        self.last_observed_settings = None
        root.title("RecognitionSystem — edycja zdjęć")
        root.geometry("1150x800")
        root.minsize(900, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)

        content = ttk.Frame(root, padding=16)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Edycja zdjęć", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        self.open_button = ttk.Button(content, text="Wybierz zdjęcie", command=self.choose_image)
        self.open_button.pack(anchor="w", pady=8)
        self.info = tk.StringVar(value="Wybierz zdjęcie, aby rozpocząć.")
        ttk.Label(content, textvariable=self.info, wraplength=850).pack(anchor="w")
        comparison = ttk.Frame(content)
        comparison.pack(fill="both", expand=True, pady=10)
        comparison.columnconfigure((0, 1), weight=1, uniform="preview")
        comparison.rowconfigure(0, weight=1)
        self.previews = []
        for column, title in enumerate(("Oryginał", "Po edycji")):
            panel = ttk.LabelFrame(comparison, text=title, padding=6)
            panel.grid(row=0, column=column, sticky="nsew", padx=4)
            panel.grid_propagate(False)
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(0, weight=1)
            label = ttk.Label(panel, text="Nie wybrano zdjęcia", anchor="center")
            label.grid(row=0, column=0, sticky="nsew")
            self.previews.append(label)
        comparison.bind("<Configure>", self.schedule_preview)

        controls = ttk.LabelFrame(content, text="Ustawienia", padding=10)
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
            ttk.Label(controls, text=title, width=16).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=defaults[key])
            self.variables[key] = var
            slider = tk.Scale(controls, from_=low, to=high, resolution=step,
                              orient="horizontal", variable=var, showvalue=False,
                              highlightthickness=0)
            slider.grid(row=row, column=1, sticky="ew", padx=10)
            value = tk.StringVar(value=f"{var.get():.2f}")
            ttk.Label(controls, textvariable=value, width=6).grid(row=row, column=2)
            var.trace_add("write", lambda *args, v=var, text=value: self.changed(v, text))
            self.widgets.append(slider)
        ttk.Label(controls, text="Odszumianie").grid(row=5, column=0, sticky="w")
        self.denoise = tk.StringVar(value="Wyłączone")
        self.combo = ttk.Combobox(controls, textvariable=self.denoise,
                                 values=["Wyłączone", "Słabe", "Średnie", "Mocne"],
                                 state="readonly", width=15)
        self.combo.grid(row=5, column=1, sticky="w", padx=10)
        self.denoise.trace_add("write", lambda *args: self.mark_dirty())
        self.clahe = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(controls, text="Lokalna poprawa kontrastu (CLAHE)",
                                variable=self.clahe, command=self.mark_dirty)
        check.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.widgets.append(check)
        ttk.Label(controls, text="Podgląd odświeża się automatycznie. Gamma poniżej 1 rozjaśnia.").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
        actions = ttk.Frame(content)
        actions.pack(fill="x", pady=10)
        self.reset_button = ttk.Button(actions, text="Przywróć oryginał", command=self.reset)
        self.reset_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(actions, text="Zapisz wynik", command=self.save)
        self.save_button.pack(side="left")
        self.status = tk.StringVar(value="Gotowe.")
        ttk.Label(content, textvariable=self.status, wraplength=1050).pack(anchor="w")
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
            self.status.set("Aktualizowanie podglądu…")
            self.save_button.configure(state="disabled")
            if self.edit_job is not None:
                self.root.after_cancel(self.edit_job)
            self.edit_job = self.root.after(180, self.apply)

    def set_busy(self, busy):
        self.open_button.configure(state="disabled" if busy else "normal")
        enabled = self.image is not None
        for widget in self.widgets:
            widget.configure(state="normal" if enabled else "disabled")
        self.combo.configure(state="readonly" if enabled else "disabled")
        self.reset_button.configure(state="normal" if enabled and not busy else "disabled")
        valid = enabled and not busy and self.edit_job is None and self.result is not None and self.settings() == self.applied_settings
        self.save_button.configure(state="normal" if valid else "disabled")

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
        self.render_previews()
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
            self.render_previews()
            self.status.set("Zmiany zastosowane. Możesz zapisać wynik wraz z parametrami.")
        except Exception as error:
            self.status.set("Przetwarzanie nie powiodło się.")
            messagebox.showerror("Błąd przetwarzania", str(error), parent=self.root)
        self.set_busy(False)

    def schedule_preview(self, event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self.render_previews)

    def render_previews(self):
        self.resize_job = None
        if self.image is None:
            return
        photos = []
        for label, image in zip(self.previews, (self.image, self.result)):
            preview = make_preview(image, max(1, label.winfo_width() - 12),
                                   max(1, label.winfo_height() - 12))
            success, encoded = cv2.imencode(".png", preview)
            if not success:
                raise ValueError("Nie można przygotować podglądu.")
            photo = tk.PhotoImage(master=self.root, data=encoded.tobytes(), format="png")
            label.configure(image=photo, text="")
            photos.append(photo)
        self.photos = photos

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
        for job in (self.edit_job, self.resize_job):
            if job is not None:
                self.root.after_cancel(job)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    ImageViewer(window)
    window.mainloop()
