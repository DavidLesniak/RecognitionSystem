"""Paired evaluation of the editor's filters with the saved CNN v3 and verifier."""

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(folder):
    raw = folder / "predykcje.csv"
    packed = folder / "predykcje.csv.gz"
    if raw.is_file():
        with raw.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    with gzip.open(packed, mode="rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--full-test", action="store_true", help="Evaluate all original test crops without simulated changes.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    if args.site_packages:
        sys.path.insert(0, str(args.site_packages.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    import tensorflow as tf

    from classifier_adapter import ClassifierAdapter, V3_MODEL
    from image_processing import EditSettings, apply_adjustments
    from modul_weryfikacji.v3.ocena_predykcji_v3 import evaluate_prediction

    seed = 181126
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    adapter = ClassifierAdapter.from_v3()
    filters = {
        "Bez filtra": EditSettings(),
        "Jasność +15": EditSettings(brightness=15),
        "Gamma 0,75": EditSettings(gamma=0.75),
        "Kontrast ×1,25": EditSettings(contrast=1.25),
        "CLAHE": EditSettings(clahe=True),
        "Nasycenie ×1,25": EditSettings(saturation=1.25),
        "Wyostrzanie 0,50": EditSettings(sharpen=0.50),
        "Odszumianie 7": EditSettings(denoise=7),
    }
    test_by_class = defaultdict(list)
    images = []
    with ZipFile(repo / "PTSD_znaki.zip") as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) >= 4 and parts[-3] == "test" and name.lower().endswith((".jpg", ".jpeg", ".png")):
                test_by_class[parts[-2]].append(name)
        if set(test_by_class) != set(adapter.class_names):
            raise ValueError("The test data and model do not share exactly the same class labels.")
        for label in sorted(test_by_class):
            names = sorted(test_by_class[label])
            if not args.full_test:
                names = rng.sample(names, 4)
            for idx, name in enumerate(names):
                data = np.frombuffer(archive.read(name), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot decode {name}")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                p10, p90 = np.percentile(gray, [10, 90])
                stress = not args.full_test and idx < 2
                images.append((name, label, image, float(gray.mean()), float(p90 - p10), stress))
        refs = (
            "PTSD_znaki/test/A-1/2019_0719_142601_003 0157_0.jpg",
            "PTSD_znaki/test/C-9/2019_0809_155001_003 1218_0.jpg",
        )
        for filename in refs:
            image = cv2.imdecode(np.frombuffer(archive.read(filename), dtype=np.uint8), cv2.IMREAD_COLOR)
            for filter_name in ("Bez filtra", "Jasność +15"):
                frame = apply_adjustments(image, filters[filter_name])
                prediction = adapter.predict(frame)
                result = evaluate_prediction(prediction.probabilities)
                print("reference:", filename.split("/")[-2], filter_name, prediction.class_name,
                      round(prediction.confidence * 100, 1), result["decision"], flush=True)

    @tf.function(reduce_retracing=True)
    def infer(batch):
        return adapter.model(batch, training=False)

    def variants(image, stress):
        yield "real", image
        if stress:
            work = image.astype(np.float32)
            yield "dark", np.rint(work * 0.60).astype(np.uint8)
            yield "flat", np.rint(128 + (work - 128) * 0.45).astype(np.uint8)
            noise = noise_rng.normal(0, 16, size=work.shape)
            yield "noise", np.clip(np.rint(work + noise), 0, 255).astype(np.uint8)

    pending = []
    prepared = []
    results = []

    def flush():
        if not prepared:
            return
        predictions = np.asarray(infer(tf.concat(prepared, axis=0)))
        for row, prediction in zip(pending, predictions, strict=True):
            outcome = evaluate_prediction(prediction.astype(np.float64))
            label = adapter.class_names[outcome["class_index"]]
            results.append({
                **row, "predicted": label, "correct": int(label == row["expected"]),
                "decision": outcome["decision"],
                "confidence": outcome["confidence"], "margin": outcome["margin"], "entropy": outcome["entropy"],
            })
        pending.clear()
        prepared.clear()

    for idx, (name, label, image, brightness, contrast, stress) in enumerate(images, start=1):
        for condition, variant in variants(image, stress):
            for filter_name, settings in filters.items():
                frame = apply_adjustments(variant, settings)
                prepared.append(adapter.prepare(frame, adapter.image_size))
                pending.append({
                    "image": name, "expected": label, "condition": condition, "filter": filter_name,
                    "original_luma_mean": brightness, "original_luma_p90_p10": contrast,
                })
                if len(prepared) == 32:
                    flush()
        if idx % (500 if args.full_test else 47) == 0:
            print(f"{idx}/{len(images)} photos; {len(results)} inferences complete", flush=True)
    flush()
    write_csv(args.out / "predykcje.csv", results)
    meta = {
        "tensorflow": tf.__version__,
        "opencv": cv2.__version__,
        "seed": seed,
        "full_test": args.full_test,
        "model_sha256": hashlib.sha256(V3_MODEL.read_bytes()).hexdigest(),
        "test_archive_size": sum(map(len, test_by_class.values())),
        "filters": {key: vars(value) for key, value in filters.items()},
    }
    (args.out / "parametry.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} inferences.", flush=True)


if __name__ == "__main__":
    main()
