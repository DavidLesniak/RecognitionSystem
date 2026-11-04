
import csv
from pathlib import Path
from reguly_weryfikacji import verify_prediction

INPUT_FILE = Path("wyniki_walidacji.csv")
OUTPUT_FILE = Path("decyzje_walidacji.csv")

with INPUT_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    results = list(csv.DictReader(file, delimiter=";"))

if not results:
    raise RuntimeError("Brak wyników walidacyjnych.")

counters = {
    "ACCEPT": {"poprawne": 0, "bledne": 0},
    "WARN": {"poprawne": 0, "bledne": 0},
    "REJECT": {"poprawne": 0, "bledne": 0},
}

for row in results:
    confidence = float(row["confidence"])
    margin = float(row["margin"])
    entropy = float(row["entropy"])

    decision = verify_prediction(confidence, margin, entropy)
    row["decyzja"] = decision

    if row["poprawna"] == "1":
        counters[decision]["poprawne"] += 1
    else:
        counters[decision]["bledne"] += 1

with OUTPUT_FILE.open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file, fieldnames=results[0].keys(), delimiter=";"
    )
    writer.writeheader()
    writer.writerows(results)


print("\n--- WERYFIKACJA PREDYKCJI ---")

for decision, counts in counters.items():
    total = counts["poprawne"] + counts["bledne"]

    print(f"\n{decision}: {total}")
    print(f"  Zgodne z etykietami: {counts['poprawne']}")
    print(f"  Niezgodne z etykietami: {counts['bledne']}")

accepted = counters["ACCEPT"]
accepted_total = accepted["poprawne"] + accepted["bledne"]

coverage = accepted_total / len(results) * 100
accepted_error_rate = (
    accepted["bledne"] / accepted_total * 100
    if accepted_total else 0
)

print("\n--- PODSUMOWANIE ---")
print(f"Przeanalizowane predykcje: {len(results)}")
print(f"Pokrycie ACCEPT: {coverage:.2f}%")
print(
    "Odsetek niezgodności w ACCEPT: "
    f"{accepted_error_rate:.2f}%"
)
print(f"Wyniki zapisano do: {OUTPUT_FILE}")
