"""Find and verify examples where a filter corrected the predicted sign."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark_filters import read_predictions, write_csv


BASE = "Bez filtra"


def shortlist(repairs, count):
    selected = []
    remaining = list(repairs)
    used_class = Counter()
    used_filter = Counter()
    used_images = set()
    while remaining and len(selected) < count:
        chosen = min(remaining, key=lambda r: (
            r["after_status"] != "ACCEPT",
            r["before_status"] == "ACCEPT",
            used_class[r["true_class"]],
            used_filter[r["filter"]],
            "mirrored" in r["source"],
            -r["after_confidence_pct"],
            r["source"],
        ))
        selected.append(chosen)
        used_class[chosen["true_class"]] += 1
        used_filter[chosen["filter"]] += 1
        used_images.add(chosen["source"])
        remaining = [row for row in remaining if row["source"] not in used_images]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    if args.site_packages:
        sys.path.insert(0, str(args.site_packages.resolve()))

    import cv2
    import numpy as np

    from classifier_adapter import ClassifierAdapter
    from image_processing import EditSettings, apply_adjustments
    from modul_weryfikacji.v3.ocena_predykcji_v3 import evaluate_prediction

    rows = read_predictions(args.folder)
    params = json.loads((args.folder / "parametry.json").read_text(encoding="utf-8"))
    assert params["full_test"]
    original = {row["image"]: row for row in rows if row["filter"] == BASE}
    assert len(original) == params["test_archive_size"]
    assert len(rows) == len(original) * len(params["filters"])
    assert len({(row["image"], row["filter"]) for row in rows}) == len(rows)
    repairs = []
    counts = defaultdict(Counter)
    for row in rows:
        before = original[row["image"]]
        valid_before = before["correct"] == "1"
        valid_after = row["correct"] == "1"
        status = row["decision"]
        stat = counts[row["filter"]]
        stat.update({
            "correct": valid_after,
            "wrong_to_correct": not valid_before and valid_after,
            "correct_to_wrong": valid_before and not valid_after,
            "wrong_to_correct_accept": not valid_before and valid_after and status == "ACCEPT",
            "wrong_accept": not valid_after and status == "ACCEPT",
        })
        if not valid_before and valid_after:
            repairs.append({
                "source": row["image"],
                "true_class": row["expected"],
                "filter": row["filter"],
                "before_class": before["predicted"],
                "before_confidence_pct": 100 * float(before["confidence"]),
                "before_status": before["decision"],
                "after_class": row["predicted"],
                "after_confidence_pct": 100 * float(row["confidence"]),
                "after_status": row["decision"],
            })

    assert repairs
    selected = shortlist(repairs, args.examples)
    model = ClassifierAdapter.from_v3()
    with ZipFile(args.repo / "PTSD_znaki.zip") as archive, ZipFile(
        args.folder / "zdjecia-przykladowe.zip", "w", compression=ZIP_DEFLATED
    ) as examples:
        instructions = [
            "Otwórz wskazany plik JPG w aplikacji i sklasyfikuj go bez korekty.",
            "Następnie zmień tylko podany filtr i ponów klasyfikację.",
            "Odszumianie 7 odpowiada opcji Średnie.",
            "",
        ]
        for index, candidate in enumerate(selected, start=1):
            data = archive.read(candidate["source"])
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert image is not None
            for version, settings in (("before", BASE), ("after", candidate["filter"])):
                edited = apply_adjustments(image, EditSettings(**params["filters"][settings]))
                prediction = model.predict(edited)
                decision = evaluate_prediction(prediction.probabilities)["decision"]
                assert prediction.class_name == candidate[f"{version}_class"]
                assert decision == candidate[f"{version}_status"]
                assert abs(100 * prediction.confidence - candidate[f"{version}_confidence_pct"]) < 0.001
            suffix = PurePosixPath(candidate["source"]).suffix
            filename = f"{index:02d}-{candidate['true_class']}{suffix}"
            candidate["file_in_zip"] = filename
            examples.writestr(filename, data)
            instructions.append(f"{filename}  |  {candidate['filter']}")
        write_csv(args.folder / "wybrane-przypadki.csv", selected)
        examples.write(args.folder / "wybrane-przypadki.csv", "wybrane-przypadki.csv")
        examples.writestr("instrukcja.txt", "\n".join(instructions) + "\n")

    write_csv(args.folder / "wszystkie-poprawione-predykcje.csv", [
        {key: value for key, value in case.items() if key != "file_in_zip"} for case in repairs
    ])
    summary = [{"filter": name, "n": len(original), **counter} for name, counter in counts.items()]
    write_csv(args.folder / "podsumowanie.csv", summary)
    print("Original photos:", len(original))
    print("Repairs:", len(repairs), "on", len({r["source"] for r in repairs}), "distinct photos")
    accepted = [r for r in repairs if r["after_status"] == "ACCEPT"]
    print("Repairs with ACCEPT:", len(accepted), "on", len({r["source"] for r in accepted}), "photos")
    strict = [r for r in accepted if r["before_status"] != "ACCEPT"]
    print("Repairs with status upgrade:", len(strict), "on", len({r["source"] for r in strict}), "photos")
    for item in summary:
        print(item)
    for case in selected:
        print(case["file_in_zip"], case["filter"],
              case["before_class"], f"{case['before_confidence_pct']:.1f}", case["before_status"], "->",
              case["after_class"], f"{case['after_confidence_pct']:.1f}", case["after_status"], case["source"])


if __name__ == "__main__":
    main()
