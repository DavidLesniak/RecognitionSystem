import csv
from pathlib import Path

from reguly_weryfikacji_v3 import verify_prediction


project_dir = Path(__file__).resolve().parent
input_file = project_dir / "wyniki_walidacji_v3.csv"
output_file = project_dir / "decyzje_walidacji_v3.csv"

with input_file.open(encoding="utf-8-sig", newline="") as file:
    results = list(csv.DictReader(file, delimiter=";"))

if not results:
    raise ValueError("Plik z wynikami walidacji jest pusty.")

counters = {
    "ACCEPT": {"1": 0, "0": 0},
    "WARN": {"1": 0, "0": 0},
    "REJECT": {"1": 0, "0": 0},
}

for row in results:
    decision = verify_prediction(
        float(row["confidence"]),
        float(row["margin"]),
        float(row["entropy"]),
    )
    row["decyzja"] = decision
    counters[decision][row["poprawna"]] += 1

with output_file.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys(), delimiter=";")
    writer.writeheader()
    writer.writerows(results)

print("\n--- DECYZJE WALIDACYJNE V3 ---")
for decision, counts in counters.items():
    print(
        f"{decision}: {sum(counts.values())} "
        f"(zgodne: {counts['1']}, niezgodne: {counts['0']})"
    )

accepted_count = sum(counters["ACCEPT"].values())
flagged_disagreements = counters["WARN"]["0"] + counters["REJECT"]["0"]
all_disagreements = counters["ACCEPT"]["0"] + flagged_disagreements

print(f"Pokrycie ACCEPT: {100 * accepted_count / len(results):.2f}%")
print(f"Niezgodności w WARN lub REJECT: {flagged_disagreements}/{all_disagreements}")
print(f"Wyniki zapisano do: {output_file}")
