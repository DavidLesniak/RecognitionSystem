import csv
from pathlib import Path

import numpy as np
import tensorflow as tf


project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / "PTSD_znaki_zbalansowane"
train_dir = dataset_dir / "train"
test_dir = dataset_dir / "test"
model_path = project_dir / "best_cnn_znaki_v3.keras"

# Keras ustala indeksy na podstawie posortowanych nazw katalogów train.
class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
with (dataset_dir / "etykiety_znaki.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    sign_names = {
        row["Znak"]: row["Etykieta"]
        for row in csv.DictReader(file, delimiter=";")
    }

if set(class_names) != set(sign_names):
    raise ValueError("Kody klas w folderze train i w CSV różnią się.")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
images = sorted(
    path for path in test_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
)
if not images:
    raise ValueError(f"Nie znaleziono zdjęć w {test_dir}.")

image_path = images[0]
true_class = image_path.relative_to(test_dir).parts[0]
if true_class not in sign_names:
    raise ValueError(f"Nieznana klasa zdjęcia testowego: {true_class}")

model = tf.keras.models.load_model(model_path, compile=False)
if len(class_names) != model.output_shape[-1]:
    raise ValueError("Liczba klas w folderze train nie zgadza się z modelem.")

image_bytes = tf.io.read_file(str(image_path))
image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
image = tf.image.resize(image, (128, 128), method="bilinear")
probabilities = model(image[None, ...], training=False).numpy()[0]

top_3 = np.argsort(probabilities)[::-1][:3]
predicted_class = class_names[top_3[0]]
confidence = float(probabilities[top_3[0]])
margin = confidence - float(probabilities[top_3[1]])
positive = probabilities[probabilities > 0]
entropy = float(-np.sum(positive * np.log(positive)))

print("Zdjęcie:", image_path.relative_to(project_dir))
print(f"Etykieta w zbiorze: {true_class} – {sign_names[true_class]}")
print(f"Predykcja: {predicted_class} – {sign_names[predicted_class]}")
print(f"Confidence: {confidence:.8f}")
print(f"Margin:     {margin:.8f}")
print(f"Entropy:    {entropy:.8f}")

print("\nTrzy najwyższe wyniki Softmax")
for index in top_3:
    code = class_names[index]
    print(f"{code} – {sign_names[code]}: {probabilities[index]:.4f}")
