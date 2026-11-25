import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageTk

from reguly_weryfikacji_v3 import verify_prediction


PROJECT_PATH = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_PATH / "PTSD_znaki_zbalansowane"
MODEL_PATH = PROJECT_PATH / "best_cnn_znaki_v3.keras"
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

EXAMPLES = [
    ("A-1", "2019_0719_142601_003 0157_0.jpg"),
    ("C-10", "2019_0811_183255_004 1455_mirrored_0.jpg"),
    ("C-9", "2019_0809_155001_003 1218_0.jpg"),
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DECISION_MESSAGES = {
    "ACCEPT": "Wynik spełnia progi akceptacji.",
    "WARN": "Wynik niepewny. Sprawdź znak ręcznie.",
    "REJECT": "Niska wiarygodność. Wymagana ręczna weryfikacja.",
}


def predict(model, image_path, conv_layer_index):
    # Przygotowanie zdjęcia tak jak w analizie zbioru testowego
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image_array = tf.image.resize(image, (128, 128), method="bilinear").numpy()

    # Model sam skaluje piksele w warstwie Rescaling
    image_batch = np.expand_dims(image_array, axis=0)

    # Aktywacje ostatniej warstwy splotowej modelu Sequential V3
    features = tf.convert_to_tensor(image_batch)
    for layer in model.layers[:conv_layer_index + 1]:
        features = layer(features, training=False)

    with tf.GradientTape() as tape:
        tape.watch(features)
        hidden = features
        for layer in model.layers[conv_layer_index + 1:-1]:
            hidden = layer(hidden, training=False)

        # Wynik klasy przed Softmax pozwala obliczyć mapę także przy confidence bliskim 1.
        output_layer = model.layers[-1]
        hidden = tf.cast(hidden, output_layer.kernel.dtype)
        scores = tf.matmul(hidden, output_layer.kernel) + output_layer.bias
        probabilities = tf.nn.softmax(scores)
        class_index = tf.argmax(probabilities[0])
        class_score = tf.gather(scores[0], class_index)

    gradients = tape.gradient(class_score, features)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()
    predictions = probabilities[0].numpy()

    top_two = np.argsort(predictions)[-2:][::-1]
    predicted_index = int(class_index.numpy())
    confidence = float(predictions[predicted_index])

    # Obliczanie margin
    margin = confidence - float(predictions[top_two[1]])

    # Obliczanie entropy
    nonzero_probabilities = predictions[predictions > 0]
    entropy = float(
        -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))
    )

    return predicted_index, confidence, margin, entropy, heatmap


def show_image(label, path, heatmap=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if heatmap is not None:
            mask = Image.fromarray((heatmap * 180).astype("uint8"))
            mask = mask.resize(image.size, Image.Resampling.BILINEAR)
            image = Image.composite(Image.new("RGB", image.size, "red"), image, mask)
        preview = ImageOps.contain(
            image, (300, 240), Image.Resampling.NEAREST
        )
    label.preview = ImageTk.PhotoImage(preview)
    label.configure(image=label.preview)


def main():
    # Wczytanie wytrenowanego modelu
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    conv_layer_index = max(
        index for index, layer in enumerate(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    )

    # Odczytanie klas w takiej samej kolejności jak podczas treningu
    class_names = sorted(
        folder.name for folder in TRAIN_PATH.iterdir()
        if folder.is_dir()
    )

    # Opisy znaków odczytujemy według kodu znaku
    with (DATASET_PATH / "etykiety_znaki.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        sign_names = {
            row["Znak"]: row["Etykieta"]
            for row in csv.DictReader(file, delimiter=";")
        }

    if len(class_names) != model.output_shape[-1] or set(class_names) != set(sign_names):
        raise ValueError("Kody w folderze train, CSV i wyjścia modelu muszą się zgadzać.")

    window = tk.Tk()
    window.title("Weryfikacja predykcji CNN V3")
    content = ttk.Frame(window, padding=18)
    content.grid()
    content.columnconfigure(0, minsize=310)
    content.columnconfigure(1, minsize=310)

    ttk.Label(content, text="Zdjęcie wejściowe").grid(row=1, column=0, pady=(14, 6))
    ttk.Label(content, text="Przykład klasy wskazanej przez CNN").grid(
        row=1, column=1, pady=(14, 6)
    )
    ttk.Label(content, text="Grad-CAM dla wskazanej klasy").grid(row=1, column=2, pady=(14, 6))
    heatmap_photo = ttk.Label(content)
    heatmap_photo.grid(row=2, column=2)
    heatmap_caption = ttk.Label(content, wraplength=290, justify="center")
    heatmap_caption.grid(row=3, column=2, pady=8, sticky="n")
    input_photo = ttk.Label(content)
    reference_photo = ttk.Label(content)
    input_photo.grid(row=2, column=0)
    reference_photo.grid(row=2, column=1)

    input_caption = ttk.Label(content, wraplength=290, justify="center")
    reference_caption = ttk.Label(content, wraplength=290, justify="center")
    input_caption.grid(row=3, column=0, pady=8, sticky="n")
    reference_caption.grid(row=3, column=1, pady=8, sticky="n")
    result = ttk.Label(content, justify="center")
    result.grid(row=4, column=0, columnspan=3, pady=(8, 0))

    def show_prediction(image_path, true_class=None):
        predicted_index, confidence, margin, entropy, heatmap = predict(
            model, image_path, conv_layer_index
        )
        predicted_class = class_names[predicted_index]
        decision = verify_prediction(confidence, margin, entropy)

        # Obraz poglądowy pochodzi z katalogu treningowego przewidzianej klasy.
        reference_path = min(
            path for path in (TRAIN_PATH / predicted_class).rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        show_image(heatmap_photo, image_path, heatmap)
        if heatmap.any():
            description = "Czerwień wskazuje obszary wspierające wynik tej klasy."
        else:
            description = "Brak dodatnich wskazań na mapie."
        heatmap_caption.configure(text=f"{predicted_class} – {description}")
        show_image(input_photo, image_path)
        show_image(reference_photo, reference_path)
        caption = image_path.name
        if true_class:
            caption = f"Etykieta: {true_class} – {sign_names[true_class]}\n{caption}"
        input_caption.configure(text=caption)
        reference_caption.configure(
            text=f"Najwyższy wynik CNN: {predicted_class} – {sign_names[predicted_class]}"
        )
        result.configure(
            text=(
                f"Decyzja modułu: {decision}\n"
                f"{DECISION_MESSAGES[decision]}\n"
                f"Confidence: {confidence:.6f}    "
                f"Margin: {margin:.6f}    Entropy: {entropy:.6f}"
            )
        )

    def open_photo():
        photo_path = PROJECT_PATH / "zdjecia_spoza_ptsd"
        if not photo_path.is_dir():
            photo_path = PROJECT_PATH
        filename = filedialog.askopenfilename(
            parent=window,
            title="Wybierz zdjęcie znaku",
            initialdir=photo_path,
            filetypes=[("Zdjęcia", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not filename:
            return

        try:
            show_prediction(Path(filename))
        except (OSError, tf.errors.OpError):
            messagebox.showerror(
                "Błąd obrazu", "Nie udało się odczytać zdjęcia.", parent=window
            )

    buttons = ttk.Frame(content)
    buttons.grid(row=0, column=0, columnspan=3)
    for column, (code, filename) in enumerate(EXAMPLES):
        image_path = TEST_PATH / code / filename
        ttk.Button(
            buttons, text=f"Pokaż {code}",
            command=lambda image_path=image_path, code=code: show_prediction(image_path, code),
        ).grid(row=0, column=column, padx=4)
    ttk.Button(buttons, text="Otwórz zdjęcie…", command=open_photo).grid(
        row=0, column=len(EXAMPLES), padx=4
    )

    code, filename = EXAMPLES[0]
    show_prediction(TEST_PATH / code / filename, code)
    window.mainloop()


if __name__ == "__main__":
    main()
