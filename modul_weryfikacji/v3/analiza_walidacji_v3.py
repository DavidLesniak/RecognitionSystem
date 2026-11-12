import csv
from pathlib import Path

import numpy as np
import tensorflow as tf


project_dir = Path(__file__).resolve().parent
train_dir = project_dir / "PTSD_znaki_zbalansowane" / "train"
model_path = project_dir / "best_cnn_znaki_v3.keras"
output_path = project_dir / "wyniki_walidacji_v3.csv"

# Podział jak w treningu V3, z mniejszą partią do pracy na CPU.
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=13,
    image_size=(128, 128),
    batch_size=32,
)
class_names = validation_dataset.class_names

model = tf.keras.models.load_model(model_path, compile=False)
if len(class_names) != model.output_shape[-1]:
    raise ValueError("Liczba klas w folderze train nie zgadza się z modelem.")

results = []
correct_count = 0

for images, labels in validation_dataset:
    # Model sam skaluje piksele przez warstwę Rescaling.
    predictions = model(images, training=False).numpy()

    for probabilities, true_index in zip(predictions, labels.numpy()):
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        sorted_probabilities = np.sort(probabilities)
        margin = confidence - float(sorted_probabilities[-2])
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive)))

        correct = int(true_index == predicted_index)
        correct_count += correct
        results.append([
            class_names[true_index],
            class_names[predicted_index],
            correct,
            confidence,
            margin,
            entropy,
        ])

    print(f"Przeanalizowano zdjęć: {len(results)}", end="\r", flush=True)

if not results:
    raise ValueError("Zbiór walidacyjny jest pusty.")

with output_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow([
        "prawdziwa_klasa",
        "przewidziana_klasa",
        "poprawna",
        "confidence",
        "margin",
        "entropy",
    ])
    writer.writerows(results)

total = len(results)
print("\n\n--- ANALIZA WALIDACYJNA V3 ---")
print(f"Zdjęcia: {total}")
print(f"Zgodne z etykietami: {correct_count}")
print(f"Niezgodne z etykietami: {total - correct_count}")
print(f"Zgodność z etykietami: {100 * correct_count / total:.2f}%")
print(f"Wyniki zapisano do: {output_path}")
