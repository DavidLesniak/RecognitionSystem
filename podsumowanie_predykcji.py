
import csv
from statistics import mean, median

# Odczyt wyników poprzedniego testu
with open(
    "wyniki_predykcji.csv",
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:
    reader = csv.DictReader(file, delimiter=";")
    results = list(reader)

# Podział wyników na poprawne i błędne
correct = [
    row for row in results
    if row["poprawna"] == "1"
]

incorrect = [
    row for row in results
    if row["poprawna"] == "0"
]


def show_summary(name, rows):
    print(f"\n--- {name} ---")
    print(f"Liczba predykcji: {len(rows)}")

    if not rows:
        print("Brak wyników w tej grupie.")
        return

    for metric in ["confidence", "margin", "entropy"]:
        values = [float(row[metric]) for row in rows]

        print(
            f"{metric}: "
            f"średnia = {mean(values):.6f}, "
            f"mediana = {median(values):.6f}"
        )


show_summary("POPRAWNE PREDYKCJE", correct)
show_summary("BŁĘDNE PREDYKCJE", incorrect)

# Sprawdzenie błędów o bardzo wysokim confidence
high_confidence_errors = [
    row for row in incorrect
    if float(row["confidence"]) >= 0.90
]

print("\n--- BŁĘDY Z WYSOKIM CONFIDENCE ---")
print(
    "Liczba błędnych predykcji z confidence >= 0.90:",
    len(high_confidence_errors)
)
