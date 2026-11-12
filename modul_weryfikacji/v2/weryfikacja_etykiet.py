
import csv
from pathlib import Path

INPUT_FILE = Path("bledne_predykcje.csv")
OUTPUT_FILE = Path("weryfikacja_etykiet.csv")

# Zabezpieczenie przed nadpisaniem ręcznych ocen
if OUTPUT_FILE.exists():
    raise FileExistsError(
        "Plik weryfikacja_etykiet.csv już istnieje. "
        "Nie nadpisujemy wykonanej pracy."
    )

# Wczytanie predykcji niezgodnych z etykietami
with INPUT_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    results = list(csv.DictReader(file, delimiter=";"))

# Dodanie kolumn do ręcznej weryfikacji
for row in results:
    row["ocena_reczna"] = "do_sprawdzenia"
    row["zweryfikowana_klasa"] = ""
    row["uwagi"] = ""

fieldnames = list(results[0].keys())

# Zapis tabeli
with OUTPUT_FILE.open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file, fieldnames=fieldnames, delimiter=";"
    )
    writer.writeheader()
    writer.writerows(results)

print(f"Przygotowano {len(results)} przypadków do weryfikacji.")
print(f"Plik: {OUTPUT_FILE}")
