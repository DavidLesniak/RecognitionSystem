"""Prototyp wyboru i podglądu obrazów."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


def read_image(path):
    """Odczytaj obraz, także ze ścieżki zawierającej polskie znaki."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise ValueError("Plik jest pusty.")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        raise ValueError("Nie można otworzyć lub odczytać pliku.") from error

    if image is None:
        raise ValueError("Plik jest uszkodzony lub ma nieobsługiwany format obrazu.")
    return image


def make_preview(image, max_width=850, max_height=480):
    """Dopasuj podgląd do okna, zachowując proporcje i oryginalny obraz."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1:
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.photo = None
        root.title("RecognitionSystem — podgląd zdjęć")
        root.geometry("920x650")
        root.minsize(920, 650)

        content = ttk.Frame(root, padding=20)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Podgląd zdjęć", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Button(content, text="Wybierz zdjęcie", command=self.choose_image).pack(anchor="w", pady=12)
        self.info = tk.StringVar(value="Wybierz zdjęcie, aby wyświetlić podgląd i wymiary.")
        ttk.Label(content, textvariable=self.info, wraplength=850).pack(anchor="w")
        self.preview = ttk.Label(content, text="Nie wybrano zdjęcia", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=(12, 0))

    def choose_image(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Wybierz zdjęcie",
            filetypes=[
                ("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.ppm"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not path:
            return

        try:
            image = read_image(path)
            preview = make_preview(image)
            success, encoded = cv2.imencode(".png", preview)
            if not success:
                raise ValueError("Nie można przygotować podglądu obrazu.")
            photo = tk.PhotoImage(master=self.root, data=encoded.tobytes(), format="png")
        except (ValueError, cv2.error, tk.TclError) as error:
            self.image = None
            self.photo = None
            self.preview.configure(image="", text="Nie udało się wczytać zdjęcia")
            self.info.set("Nie wczytano obrazu. Wybierz inny plik.")
            messagebox.showerror("Błąd odczytu", str(error), parent=self.root)
            return

        self.image = image
        self.photo = photo  # Referencja zapobiega usunięciu obrazu przez Tkinter.
        height, width = image.shape[:2]
        self.info.set(f"Plik: {Path(path).name}\nWymiary: {width} × {height} px")
        self.preview.configure(image=self.photo, text="")


if __name__ == "__main__":
    window = tk.Tk()
    ImageViewer(window)
    window.mainloop()
