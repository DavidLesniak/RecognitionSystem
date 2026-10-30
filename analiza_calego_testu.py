
from pathlib import Path
import csv

import numpy as np
import tensorflow as tf


MODEL_PATH = "traffic_classifier_v2.keras"
TRAIN_PATH = Path("PTSD/train")
TEST_PATH = "PTSD/test"
OUTPUT_PATH = Path("wyniki_calego_testu.csv")

BATCH_SIZE = 32

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Odczytanie kolejności klas ze zbioru treningowego
train_classes = sorted(
    folder.name
    for folder in TRAIN_PATH.iterdir()
    if folder.is_dir()
)

# Wczytanie zdjęć testowych partiami, bez losowania
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(64, 64),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names
file_paths = test_dataset.file_paths

if class_names != train_classes:
    raise ValueError(
        "Kolejność klas w zbiorze testowym nie zgadza się "
        "z kolejnością klas treningowych."
    )

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Liczba klas nie zgadza się z modelem.")

total = 0
correct_count = 0
incorrect_count = 0
high_confidence_errors = 0

file_index = 0

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file, delimiter=";")

    writer.writerow([
        "zdjecie",
        "prawdziwa_klasa",
        "przewidziana_klasa",
        "poprawna",
        "confidence",
        "margin",
        "entropy"
    ])

    for images, labels in test_dataset:

        probabilities_batch = model(
            images, training=False
        ).numpy()

        for probabilities, true_index in zip(
            probabilities_batch, labels.numpy()
        ):

            image_path = file_paths[file_index]
            file_index += 1

            predicted_index = int(np.argmax(probabilities))
            true_index = int(true_index)

            true_class = class_names[true_index]
            predicted_class = class_names[predicted_index]

            confidence = float(np.max(probabilities))

            sorted_probs = np.sort(probabilities)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])

            positive_probs = probabilities[
                probabilities > 0
            ]
            entropy = float(
                -np.sum(positive_probs * np.log(positive_probs))
            )

            correct = true_index == predicted_index

            total += 1

            if correct:
                correct_count += 1
            else:
                incorrect_count += 1

                if confidence >= 0.90:
                    high_confidence_errors += 1

            writer.writerow([
                image_path,
                true_class,
                predicted_class,
                int(correct),
                confidence,
                margin,
                entropy
            ])

        print(f"Przeanalizowano: {total} zdjęć", end="\r")

accuracy = correct_count / total * 100 if total else 0

print("\n\n--- WYNIKI CAŁEGO ZBIORU TESTOWEGO ---")
print(f"Liczba zdjęć: {total}")
print(f"Poprawne predykcje: {correct_count}")
print(f"Błędne predykcje: {incorrect_count}")
print(f"Skuteczność względem etykiet: {accuracy:.2f}%")
print(f"Błędy z confidence >= 0.90: {high_confidence_errors}")
print(f"Wyniki zapisano do: {OUTPUT_PATH}")
