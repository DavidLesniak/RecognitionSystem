
from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

MODEL_PATH = "traffic_classifier_v2.keras"
TRAIN_PATH = "PTSD/train"
OUTPUT_PATH = Path("wyniki_walidacji.csv")

# Ustawienia zgodne z kodem treningowym V2
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 13

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Odtworzenie podziału walidacyjnego
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = validation_dataset.class_names

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Liczba klas nie zgadza się z modelem.")

total = 0
correct_count = 0

with OUTPUT_PATH.open(
    "w", newline="", encoding="utf-8-sig"
) as file:

    writer = csv.writer(file, delimiter=";")

    writer.writerow([
        "prawdziwa_klasa",
        "przewidziana_klasa",
        "poprawna",
        "confidence",
        "margin",
        "entropy"
    ])

    for images, labels in validation_dataset:

        probabilities_batch = model(
            images, training=False
        ).numpy()

        for probabilities, true_index in zip(
            probabilities_batch, labels.numpy()
        ):
            true_index = int(true_index)
            predicted_index = int(np.argmax(probabilities))

            confidence = float(np.max(probabilities))

            sorted_probs = np.sort(probabilities)[::-1]
            margin = float(
                sorted_probs[0] - sorted_probs[1]
            )

            positive_probs = probabilities[
                probabilities > 0
            ]
            entropy = float(
                -np.sum(
                    positive_probs * np.log(positive_probs)
                )
            )

            correct = true_index == predicted_index

            total += 1
            correct_count += int(correct)

            writer.writerow([
                class_names[true_index],
                class_names[predicted_index],
                int(correct),
                confidence,
                margin,
                entropy
            ])

accuracy = 100 * correct_count / total if total else 0

print("\n--- ANALIZA WALIDACYJNA ---")
print(f"Zdjęcia: {total}")
print(f"Zgodne predykcje: {correct_count}")
print(f"Niezgodne predykcje: {total - correct_count}")
print(f"Zgodność z etykietami: {accuracy:.2f}%")
print(f"Wyniki zapisano do: {OUTPUT_PATH}")
