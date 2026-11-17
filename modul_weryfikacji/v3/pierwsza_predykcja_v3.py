import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

project_path = Path(__file__).resolve().parent

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model(
    project_path / "best_cnn_znaki_v3.keras",
    compile=False
)

dataset_path = project_path / "PTSD_znaki_zbalansowane"

# Odczytanie klas w takiej samej kolejności jak podczas treningu
train_path = dataset_path / "train"
class_names = sorted(
    folder.name for folder in train_path.iterdir()
    if folder.is_dir()
)

# Opisy znaków odczytujemy według kodu znaku
with (dataset_path / "etykiety_znaki.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    sign_names = {
        row["Znak"]: row["Etykieta"]
        for row in csv.DictReader(file, delimiter=";")
    }

if len(class_names) != model.output_shape[-1] or set(class_names) != set(sign_names):
    raise ValueError("Klasy w folderze train, modelu i CSV nie zgadzają się.")

# Wybranie pierwszego dostępnego zdjęcia ze zbioru testowego
test_path = dataset_path / "test"
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    path for path in test_path.glob("*/*")
    if path.is_file() and path.suffix.lower() in extensions
)

if not images:
    raise RuntimeError(f"Nie znaleziono zdjęć w {test_path}")

image_path = images[0]
true_class = image_path.parent.name

# Przygotowanie zdjęcia dla modelu CNN
image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128),
    color_mode="rgb",
    interpolation="bilinear"
)

image_array = tf.keras.utils.img_to_array(image)
image_batch = np.expand_dims(image_array, axis=0)

# Wykonanie predykcji
predictions = model.predict(image_batch, verbose=0)[0]

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
confidence = float(predictions[predicted_index])

# Obliczanie margin
sorted_probabilities = np.sort(predictions)[::-1]
margin = float(sorted_probabilities[0] - sorted_probabilities[1])

# Obliczanie entropy
nonzero_probabilities = predictions[predictions > 0]
entropy = float(
    -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))
)

# Wyświetlenie wyniku
print("\n--- PIERWSZA PREDYKCJA V3 ---")
print("Zdjęcie:", image_path.relative_to(project_path))
print("Prawdziwa klasa:", true_class, "–", sign_names[true_class])
print("Klasa przewidziana:", predicted_class, "–", sign_names[predicted_class])
print(f"Confidence: {confidence:.8f}")
print(f"Margin:     {margin:.8f}")
print(f"Entropy:    {entropy:.8f}")

if true_class == predicted_class:
    print("Wynik: POPRAWNA PREDYKCJA")
else:
    print("Wynik: BŁĘDNA PREDYKCJA")

print("\nTrzy najwyższe wyniki Softmax:")

top_3 = np.argsort(predictions)[::-1][:3]

for index in top_3:
    print(f"{class_names[index]}: {predictions[index]:.4f}")
