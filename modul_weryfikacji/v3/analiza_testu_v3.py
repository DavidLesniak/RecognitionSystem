from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

from reguly_weryfikacji_v3 import verify_prediction


PROJECT_PATH = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_PATH / "PTSD_znaki_zbalansowane"
MODEL_PATH = PROJECT_PATH / "best_cnn_znaki_v3.keras"
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
OUTPUT_PATH = PROJECT_PATH / "wyniki_testu_v3.csv"

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
    image_size=(128, 128),
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

if not file_paths:
    raise ValueError("Zbiór testowy jest pusty.")

counters = {
    "ACCEPT": {"poprawne": 0, "bledne": 0},
    "WARN": {"poprawne": 0, "bledne": 0},
    "REJECT": {"poprawne": 0, "bledne": 0},
}

total = 0
correct_count = 0

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
        "entropy",
        "decyzja"
    ])

    for images, labels in test_dataset:

        # Warstwa Rescaling w modelu skaluje piksele.
        probabilities_batch = model(
            images, training=False
        ).numpy()

        for probabilities, true_index in zip(
            probabilities_batch, labels.numpy(), strict=True
        ):

            image_path = Path(file_paths[total]).relative_to(PROJECT_PATH)

            predicted_index = int(np.argmax(probabilities))
            true_index = int(true_index)

            true_class = class_names[true_index]
            predicted_class = class_names[predicted_index]

            confidence = float(np.max(probabilities))

            sorted_probs = np.sort(probabilities)[::-1]
            margin = confidence - float(sorted_probs[1])

            positive_probs = probabilities[
                probabilities > 0
            ]
            entropy = float(
                -np.sum(positive_probs * np.log(positive_probs))
            )

            decision = verify_prediction(confidence, margin, entropy)
            correct = true_index == predicted_index

            total += 1

            if correct:
                correct_count += 1
                counters[decision]["poprawne"] += 1
            else:
                counters[decision]["bledne"] += 1

            writer.writerow([
                image_path,
                true_class,
                predicted_class,
                int(correct),
                confidence,
                margin,
                entropy,
                decision
            ])

        print(f"Przeanalizowano zdjęć: {total}", end="\r", flush=True)

if total != len(file_paths):
    raise ValueError("Liczba predykcji różni się od liczby zdjęć testowych.")

accuracy = correct_count / total * 100
accepted = counters["ACCEPT"]
accepted_total = accepted["poprawne"] + accepted["bledne"]
flagged_disagreements = counters["WARN"]["bledne"] + counters["REJECT"]["bledne"]
flagged_correct = counters["WARN"]["poprawne"] + counters["REJECT"]["poprawne"]

print("\n\n--- OCENA NA ZBIORZE TESTOWYM V3 ---")
print(f"Zdjęcia: {total}")
print(f"Zgodność CNN z etykietami: {accuracy:.2f}%")

for decision, counts in counters.items():
    decision_total = counts["poprawne"] + counts["bledne"]
    print(
        f"{decision}: {decision_total} "
        f"(zgodne: {counts['poprawne']}, niezgodne: {counts['bledne']})"
    )

print(f"Pokrycie ACCEPT: {100 * accepted_total / total:.2f}%")
print(f"Niezgodności w WARN lub REJECT: {flagged_disagreements}/{total - correct_count}")
print(f"Niezgodności w ACCEPT: {accepted['bledne']}")
print(f"Zgodne predykcje w WARN lub REJECT: {flagged_correct}")
print(f"Wyniki zapisano do: {OUTPUT_PATH}")
