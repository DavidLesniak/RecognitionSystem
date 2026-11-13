import csv
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageTk

from reguly_weryfikacji_v3 import verify_prediction


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "PTSD_znaki_zbalansowane"
EXAMPLES = [
    ("A-1", "2019_0719_142601_003 0157_0.jpg"),
    ("C-9", "2019_0809_155001_003 1218_0.jpg"),
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def predict(model, image_path):
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    # RGB, 128 × 128 i interpolacja jak w pierwszej predykcji V3.
    # Model sam skaluje piksele.
    image = tf.image.resize(image, (128, 128), method="bilinear")
    probabilities = model(image[None, ...], training=False).numpy()[0]

    top_two = np.argsort(probabilities)[-2:][::-1]
    index = int(top_two[0])
    confidence = float(probabilities[index])
    margin = confidence - float(probabilities[top_two[1]])
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return index, confidence, margin, entropy


def show_image(label, path):
    with Image.open(path) as image:
        preview = ImageOps.contain(
            image.convert("RGB"), (300, 240), Image.Resampling.NEAREST
        )
    label.preview = ImageTk.PhotoImage(preview)
    label.configure(image=label.preview)


def main():
    train_dir = DATASET_DIR / "train"
    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    with (DATASET_DIR / "etykiety_znaki.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        sign_names = {
            row["Znak"]: row["Etykieta"]
            for row in csv.DictReader(file, delimiter=";")
        }

    model = tf.keras.models.load_model(
        PROJECT_DIR / "best_cnn_znaki_v3.keras", compile=False
    )
    if len(class_names) != model.output_shape[-1] or set(class_names) != set(sign_names):
        raise ValueError("Kody w folderze train, CSV i wyjścia modelu muszą się zgadzać.")

    window = tk.Tk()
    window.title("Weryfikacja predykcji CNN V3")
    content = ttk.Frame(window, padding=18)
    content.grid()
    content.columnconfigure(0, minsize=310)
    content.columnconfigure(1, minsize=310)

    ttk.Label(content, text="Zdjęcie testowe").grid(row=1, column=0, pady=(14, 6))
    ttk.Label(content, text="Przykład klasy wskazanej przez CNN").grid(
        row=1, column=1, pady=(14, 6)
    )
    input_photo = ttk.Label(content)
    reference_photo = ttk.Label(content)
    input_photo.grid(row=2, column=0)
    reference_photo.grid(row=2, column=1)

    input_caption = ttk.Label(content, wraplength=290, justify="center")
    reference_caption = ttk.Label(content, wraplength=290, justify="center")
    input_caption.grid(row=3, column=0, pady=8, sticky="n")
    reference_caption.grid(row=3, column=1, pady=8, sticky="n")
    result = ttk.Label(content, justify="center")
    result.grid(row=4, column=0, columnspan=2, pady=(8, 0))

    def show_example(true_class, filename):
        image_path = DATASET_DIR / "test" / true_class / filename
        index, confidence, margin, entropy = predict(model, image_path)
        predicted_class = class_names[index]
        decision = verify_prediction(confidence, margin, entropy)

        # Obraz poglądowy pochodzi z katalogu treningowego przewidzianej klasy.
        reference_path = min(
            path for path in (train_dir / predicted_class).rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        show_image(input_photo, image_path)
        show_image(reference_photo, reference_path)
        input_caption.configure(
            text=f"Etykieta: {true_class} – {sign_names[true_class]}\n{filename}"
        )
        reference_caption.configure(
            text=f"Predykcja: {predicted_class} – {sign_names[predicted_class]}"
        )
        result.configure(
            text=f"Decyzja modułu: {decision}\n"
                 f"Confidence: {confidence:.6f}    "
                 f"Margin: {margin:.6f}    Entropy: {entropy:.6f}"
        )

    for column, (code, filename) in enumerate(EXAMPLES):
        ttk.Button(
            content, text=f"Pokaż {code}",
            command=lambda code=code, filename=filename: show_example(code, filename),
        ).grid(row=0, column=column)

    show_example(*EXAMPLES[0])
    window.mainloop()


if __name__ == "__main__":
    main()
