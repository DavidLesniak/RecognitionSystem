import csv
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_PATH / "wyniki_testu_v3.csv"
BIN_COUNT = 10

# Każdy przedział przechowuje wyniki zdjęć o podobnym confidence.
groups = [[] for _ in range(BIN_COUNT)]
with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file, delimiter=";"):
        confidence = float(row["confidence"])
        if not 0 <= confidence <= 1 or row["poprawna"] not in ("0", "1"):
            raise ValueError(f"Nieprawidłowy wynik w wierszu: {row}")

        # Confidence równe 1 należy do ostatniego przedziału.
        index = min(int(confidence * BIN_COUNT), BIN_COUNT - 1)
        groups[index].append((confidence, int(row["poprawna"])))

total_count = sum(len(group) for group in groups)
if not total_count:
    raise RuntimeError("Plik z wynikami testu jest pusty.")

print("--- KALIBRACJA CNN V3 NA ZBIORZE TESTOWYM ---")
print(f"Zdjęcia: {total_count}")
print("Confidence     Zdjęcia    Śr. pewność    Trafność")

ece = 0.0
for index, group in enumerate(groups):
    lower = index / BIN_COUNT
    upper = (index + 1) / BIN_COUNT
    end = "]" if index == BIN_COUNT - 1 else ")"
    interval = f"[{lower:.1f}, {upper:.1f}{end}"
    count = len(group)
    if count == 0:
        print(f"{interval:<14} {count:6}              -           -")
        continue

    mean_confidence = sum(confidence for confidence, correct in group) / count
    accuracy = sum(correct for confidence, correct in group) / count
    ece += count / total_count * abs(mean_confidence - accuracy)

    print(f"{interval:<14} {count:6} {mean_confidence:14.2%} {accuracy:11.2%}")

print(f"\nECE: {ece:.5f} ({100 * ece:.2f} p.p.)")
