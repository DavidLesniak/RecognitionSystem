
from pathlib import Path
import csv
import random

import numpy as np
import tensorflow as tf


# Ustawienia
MODEL_PATH = "traffic_classifier_v2.keras"
TRAIN_PATH = Path("PTSD/train")
TEST_PATH = Path("PTSD/test")
RESULTS_PATH = Path("wyniki_predykcji.csv")

IMAGE_SIZE = (64, 64)
NUMBER_OF_IMAGES = 100
SEED = 13


# Wczytanie modelu
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Nazwy klas w kolejności używanej podczas treningu
class_names = sorted(
    folder.name
    for folder in TRAIN_PATH.iterdir()
    if folder.is_dir()
)

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Liczba klas nie zgadza się z modelem.")

# Lista zdjęć testowych
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted(
    path for path in TEST_PATH.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
)

if not image_paths:
    raise RuntimeError("Nie znaleziono zdjęć testowych.")

# Losowanie 100 zdjęć; stały seed zapewnia powtarzalność
rng = random.Random(SEED)
selected_images = rng.sample(
    image_paths,
    min(NUMBER_OF_IMAGES, len(image_paths))
)

# Przygotowanie obrazów
image_arrays = []

for image_path in selected_images:
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        interpolation="bilinear"
    )
    image_arrays.append(
        tf.keras.utils.img_to_array(image)
    )

image_batch = np.stack(image_arrays)

# Predykcje dla wszystkich wybranych zdjęć
predictions_batch = model.predict(
    image_batch,
    batch_size=32,
    verbose=0
)

# Obliczanie miar i porównanie z prawdziwymi klasami
results = []

for image_path, predictions in zip(
    selected_images, predictions_batch
):
    true_class = image_path.parent.name

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]

    confidence = float(np.max(predictions))

    sorted_probabilities = np.sort(predictions)[::-1]
    margin = float(
        sorted_probabilities[0] - sorted_probabilities[1]
    )

    nonzero_probabilities = predictions[predictions > 0]
    entropy = float(
        -np.sum(
            nonzero_probabilities * np.log(nonzero_probabilities)
        )
    )

    correct = true_class == predicted_class

    results.append({
        "zdjecie": str(image_path),
        "prawdziwa_klasa": true_class,
        "przewidziana_klasa": predicted_class,
        "poprawna": int(correct),
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy
    })

# Zapis wyników do CSV
with RESULTS_PATH.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys(),
        delimiter=";"
    )
    writer.writeheader()
    writer.writerows(results)

# Podsumowanie analizy
correct_results = [
    result for result in results
    if result["poprawna"] == 1
]

incorrect_results = [
    result for result in results
    if result["poprawna"] == 0
]

accuracy = len(correct_results) / len(results) * 100

print("\n--- ANALIZA PREDYKCJI ---")
print(f"Przetestowane zdjęcia: {len(results)}")
print(f"Poprawne predykcje: {len(correct_results)}")
print(f"Błędne predykcje: {len(incorrect_results)}")
print(f"Skuteczność w próbce: {accuracy:.2f}%")
print(f"Wyniki zapisano do: {RESULTS_PATH}")

print("\n--- PRZYKŁADY BŁĘDNYCH PREDYKCJI ---")

if not incorrect_results:
    print("Nie znaleziono błędnych predykcji w próbce.")
else:
    for result in incorrect_results[:5]:
        print("\nZdjęcie:", result["zdjecie"])
        print("Prawdziwa klasa:", result["prawdziwa_klasa"])
        print("Przewidziana klasa:", result["przewidziana_klasa"])
        print(f"Confidence: {result['confidence']:.8f}")
        print(f"Margin: {result['margin']:.8f}")
        print(f"Entropy: {result['entropy']:.8f}")
