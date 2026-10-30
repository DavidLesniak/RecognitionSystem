
import csv
from statistics import mean, median

INPUT_FILE = "wyniki_calego_testu.csv"
OUTPUT_FILE = "bledne_predykcje.csv"

# Odczyt wyników z całego zbioru testowego
with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file, delimiter=";")
    results = list(reader)

# Podział na predykcje zgodne i niezgodne z etykietami
correct = [row for row in results if row["poprawna"] == "1"]
incorrect = [row for row in results if row["poprawna"] == "0"]


def show_metrics(name, rows):
    print(f"\n--- {name} ---")
    print(f"Liczba przypadków: {len(rows)}")

    if not rows:
        return

    for metric in ["confidence", "margin", "entropy"]:
        values = [float(row[metric]) for row in rows]

        print(
            f"{metric}: "
            f"średnia = {mean(values):.6f}, "
            f"mediana = {median(values):.6f}"
        )


show_metrics("ZGODNE Z ETYKIETAMI", correct)
show_metrics("NIEZGODNE Z ETYKIETAMI", incorrect)

# Sortowanie niezgodnych predykcji:
# najwyższe confidence na początku
incorrect.sort(
    key=lambda row: float(row["confidence"]),
    reverse=True
)

# Zapis listy do osobnego CSV
if incorrect:
    with open(
        OUTPUT_FILE, "w",
        encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=incorrect[0].keys(),
            delimiter=";"
        )
        writer.writeheader()
        writer.writerows(incorrect)

print(f"\nZapisano {len(incorrect)} przypadków do {OUTPUT_FILE}")

# Przykłady niezgodnych predykcji z wysokim confidence
print("\n--- 10 PRZYPADKÓW Z NAJWYŻSZYM CONFIDENCE ---")

for row in incorrect[:10]:
    print(
        f"{row['prawdziwa_klasa']} -> "
        f"{row['przewidziana_klasa']} | "
        f"confidence = {float(row['confidence']):.8f} | "
        f"margin = {float(row['margin']):.8f} | "
        f"entropy = {float(row['entropy']):.8f}"
    )
