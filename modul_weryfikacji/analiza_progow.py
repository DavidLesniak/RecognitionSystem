
import csv

INPUT_FILE = "wyniki_walidacji.csv"

# Progi do porównania
CONFIDENCE_THRESHOLDS = [0.90, 0.95, 0.99, 0.999]
MARGIN_THRESHOLDS = [0.50, 0.70, 0.90, 0.99]
ENTROPY_THRESHOLDS = [0.01, 0.05, 0.10, 0.50]

# Wczytanie wyników walidacji
with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
    results = list(csv.DictReader(file, delimiter=";"))

if not results:
    raise RuntimeError("Plik walidacyjny jest pusty.")


def analyze_thresholds(metric, thresholds, higher_is_better):
    print(f"\n--- ANALIZA PROGÓW: {metric.upper()} ---")

    for threshold in thresholds:
        if higher_is_better:
            accepted = [
                row for row in results
                if float(row[metric]) >= threshold
            ]
        else:
            accepted = [
                row for row in results
                if float(row[metric]) <= threshold
            ]

        not_accepted = [
            row for row in results
            if row not in accepted
        ]

        accepted_errors = sum(
            row["poprawna"] == "0"
            for row in accepted
        )

        not_accepted_correct = sum(
            row["poprawna"] == "1"
            for row in not_accepted
        )

        coverage = 100 * len(accepted) / len(results)

        error_rate = (
            100 * accepted_errors / len(accepted)
            if accepted else 0
        )

        print(f"\nPróg: {threshold}")
        print(f"Zaakceptowane: {len(accepted)}")
        print(f"Pokrycie: {coverage:.2f}%")
        print(f"Błędy w zaakceptowanych: {accepted_errors}")
        print(f"Odłożone poprawne: {not_accepted_correct}")
        print(f"Odsetek błędów w zaakceptowanych: {error_rate:.2f}%")


analyze_thresholds(
    "confidence",
    CONFIDENCE_THRESHOLDS,
    higher_is_better=True
)

analyze_thresholds(
    "margin",
    MARGIN_THRESHOLDS,
    higher_is_better=True
)

analyze_thresholds(
    "entropy",
    ENTROPY_THRESHOLDS,
    higher_is_better=False
)
