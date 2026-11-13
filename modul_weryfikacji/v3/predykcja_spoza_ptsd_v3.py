import argparse
import csv
from pathlib import Path

import tensorflow as tf

from demo_weryfikacji_v3 import DATASET_DIR, PROJECT_DIR, predict
from reguly_weryfikacji_v3 import verify_prediction


def main():
    parser = argparse.ArgumentParser(description="Predykcja V3 na zdjęciu spoza PTSD.")
    parser.add_argument("zdjecie", type=Path, help="Ścieżka do pliku ze zdjęciem znaku.")
    parser.add_argument("kod", help="Ręcznie ustalony kod znaku, np. A-1.")
    args = parser.parse_args()
    if not args.zdjecie.is_file():
        parser.error(f"Nie znaleziono zdjęcia: {args.zdjecie}")

    class_names = sorted(
        path.name for path in (DATASET_DIR / "train").iterdir() if path.is_dir()
    )
    with (DATASET_DIR / "etykiety_znaki.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        sign_names = {
            row["Znak"]: row["Etykieta"]
            for row in csv.DictReader(file, delimiter=";")
        }

    expected = args.kod.strip()
    if expected not in class_names:
        parser.error(f"Model V3 nie ma klasy {expected}.")

    model = tf.keras.models.load_model(
        PROJECT_DIR / "best_cnn_znaki_v3.keras", compile=False
    )
    if len(class_names) != model.output_shape[-1] or set(class_names) != set(sign_names):
        raise ValueError("Kody w folderze train, CSV i wyjścia modelu muszą się zgadzać.")

    index, confidence, margin, entropy = predict(model, args.zdjecie)
    predicted = class_names[index]
    decision = verify_prediction(confidence, margin, entropy)

    print(f"Zdjęcie: {args.zdjecie}")
    print(f"Etykieta ręczna: {expected} – {sign_names[expected]}")
    print(f"Predykcja CNN: {predicted} – {sign_names[predicted]}")
    print(f"Zgodność z etykietą: {'tak' if predicted == expected else 'nie'}")
    print(f"Confidence: {confidence:.6f}")
    print(f"Margin:     {margin:.6f}")
    print(f"Entropy:    {entropy:.6f}")
    print(f"Decyzja modułu: {decision}")


if __name__ == "__main__":
    main()
