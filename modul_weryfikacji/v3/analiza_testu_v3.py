import csv
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from reguly_weryfikacji_v3 import verify_prediction


project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / "PTSD_znaki_zbalansowane"
train_dir = dataset_dir / "train"
test_dir = dataset_dir / "test"
model_path = project_dir / "best_cnn_znaki_v3.keras"
output_path = project_dir / "wyniki_testu_v3.csv"

# Bez losowania ścieżki zdjęć mają tę samą kolejność co predykcje.
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False,
)
class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
if class_names != test_dataset.class_names:
    raise ValueError("Klasy w katalogach train i test różnią się.")

model = tf.keras.models.load_model(model_path, compile=False)
if len(class_names) != model.output_shape[-1]:
    raise ValueError("Liczba klas w folderze train nie zgadza się z modelem.")

results = []
counts = Counter()
image_paths = iter(test_dataset.file_paths)

for images, labels in test_dataset:
    # Skalowanie pikseli wykonuje warstwa Rescaling w modelu.
    predictions = model(images, training=False).numpy()

    for probabilities, true_index in zip(predictions, labels.numpy(), strict=True):
        image_path = Path(next(image_paths))
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        sorted_probabilities = np.sort(probabilities)
        margin = confidence - float(sorted_probabilities[-2])
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive)))

        decision = verify_prediction(confidence, margin, entropy)
        correct = int(true_index == predicted_index)
        counts[decision, correct] += 1
        results.append([
            image_path.relative_to(project_dir),
            class_names[true_index],
            class_names[predicted_index],
            correct,
            confidence,
            margin,
            entropy,
            decision,
        ])

    print(f"Przeanalizowano zdjęć: {len(results)}", end="\r", flush=True)

if not results:
    raise ValueError("Zbiór testowy jest pusty.")
if len(results) != len(test_dataset.file_paths):
    raise ValueError("Liczba predykcji różni się od liczby zdjęć testowych.")

with output_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow([
        "zdjecie",
        "prawdziwa_klasa",
        "przewidziana_klasa",
        "poprawna",
        "confidence",
        "margin",
        "entropy",
        "decyzja",
    ])
    writer.writerows(results)

total = len(results)
accepted = counts["ACCEPT", 1] + counts["ACCEPT", 0]
flagged_disagreements = counts["WARN", 0] + counts["REJECT", 0]
all_disagreements = flagged_disagreements + counts["ACCEPT", 0]

print("\n\n--- OCENA NA ZBIORZE TESTOWYM V3 ---")
print(f"Zdjęcia: {total}")
print(f"Zgodność CNN z etykietami: {100 * (total - all_disagreements) / total:.2f}%")
for decision in ("ACCEPT", "WARN", "REJECT"):
    print(
        f"{decision}: {counts[decision, 1] + counts[decision, 0]} "
        f"(zgodne: {counts[decision, 1]}, niezgodne: {counts[decision, 0]})"
    )
print(f"Pokrycie ACCEPT: {100 * accepted / total:.2f}%")
print(f"Niezgodności w WARN lub REJECT: {flagged_disagreements}/{all_disagreements}")
print(f"Niezgodności w ACCEPT: {counts['ACCEPT', 0]}")
print(f"Zgodne predykcje w WARN lub REJECT: {counts['WARN', 1] + counts['REJECT', 1]}")
print(f"Wyniki zapisano do: {output_path}")
