"""Build paired summaries and presentation-ready charts from benchmark predictions."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmark_filters import read_predictions


BASE = "Bez filtra"
GROUPS = (
    ("real", "Wszystkie", "Oryginalne"),
    ("real_dark", "Najciemniejsza ¼", "Oryginalne"),
    ("real_flat", "Najniższy kontrast ¼", "Oryginalne"),
    ("dark", "Przyciemnienie", "Symulacja · ×0,60"),
    ("flat", "Spadek kontrastu", "Symulacja · ×0,45"),
    ("noise", "Szum", "Symulacja · σ = 16"),
)


def chart_font(size, bold=False):
    candidates = (
        Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def percent(num, digits=1, signed=False):
    if abs(num) < 0.5 * 10**-digits:
        num = 0.0
    spec = f"{'+' if signed and num else ''}.{digits}f"
    return format(num, spec).replace(".", ",").replace("-", "−")


def summarize(rows):
    by_condition = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_condition[row["condition"]][row["filter"]][row["image"]] = row
    real = by_condition["real"][BASE]
    n_quarter = len(real) // 4
    dark_members = sorted(real, key=lambda x: (float(real[x]["original_luma_mean"]), x))[:n_quarter]
    flat_members = sorted(real, key=lambda x: (float(real[x]["original_luma_p90_p10"]), x))[:n_quarter]
    for name, members in (("real_dark", dark_members), ("real_flat", flat_members)):
        by_condition[name] = {
            filt: {image: candidates[image] for image in members}
            for filt, candidates in by_condition["real"].items()
        }

    summary = []
    for condition, title, context in GROUPS:
        original = by_condition[condition][BASE]
        n = len(original)
        for filter_name, edited in by_condition[condition].items():
            if original.keys() != edited.keys():
                raise ValueError("The benchmark groups must be paired by image.")
            pairs = [(baseline, edited[image]) for image, baseline in original.items()]
            improved = sum(a["correct"] == "0" and b["correct"] == "1" for a, b in pairs)
            worsened = sum(a["correct"] == "1" and b["correct"] == "0" for a, b in pairs)
            accepted_wrong_baseline = sum(a["correct"] == "0" and a["decision"] == "ACCEPT" for a, _ in pairs)
            accepted_wrong = sum(b["correct"] == "0" and b["decision"] == "ACCEPT" for _, b in pairs)
            base_accuracy = 100 * sum(a["correct"] == "1" for a, _ in pairs) / n
            current_accuracy = 100 * sum(b["correct"] == "1" for _, b in pairs) / n
            summary.append({
                "condition": condition, "group": title, "context": context, "filter": filter_name, "n": n,
                "base_accuracy_pct": base_accuracy, "accuracy_pct": current_accuracy,
                "delta_accuracy_pp": 100 * (improved - worsened) / n,
                "incorrect_to_correct": improved, "correct_to_incorrect": worsened,
                "changed_label_pct": 100 * sum(a["predicted"] != b["predicted"] for a, b in pairs) / n,
                "changed_decision_pct": 100 * sum(a["decision"] != b["decision"] for a, b in pairs) / n,
                "wrong_accept_count": accepted_wrong, "wrong_accept_base_count": accepted_wrong_baseline,
                "wrong_accept_delta_pp": 100 * (accepted_wrong - accepted_wrong_baseline) / n,
                "accept_count": sum(b["decision"] == "ACCEPT" for _, b in pairs),
            })
    return summary


def create_chart(path, summary, metric, title, subtitle, footnote, baseline_field, positive_good):
    font = chart_font

    img = Image.new("RGB", (1680, 1000), "#f6f8fb")
    draw = ImageDraw.Draw(img)
    ink = "#142940"
    muted = "#455d74"
    w, pad, start, top, xgap, row_h = 203, 44, 360, 318, 9, 62
    index = {(r["filter"], r["condition"]): r for r in summary}
    filters = list(dict.fromkeys(r["filter"] for r in summary if r["filter"] != BASE))

    draw.text((pad, 42), "RECOGNITIONSYSTEM  /  TEST FILTRÓW", font=font(21, True), fill="#22597c")
    draw.text((pad, 82), title, font=font(45, True), fill=ink)
    draw.text((pad, 155), subtitle, font=font(25), fill=muted)
    for idx, label in ((0, "ZDJĘCIA TESTOWE BEZ SYMULOWANYCH USZKODZEŃ"), (3, "SYMULACJE ZMIAN NA ZDJĘCIACH TESTOWYCH")):
        x = start + idx * (w + xgap)
        draw.rounded_rectangle((x, 218, x + 3*w + 2*xgap, 253), 7, fill="#e7eef6")
        draw.text((x + (3*w + 2*xgap) / 2, 234), label, font=font(16, True), fill="#234f6f", anchor="mm")
    draw.text((pad, 280), "Filtr w aplikacji", font=font(22, True), fill=ink)
    for idx, (condition, header, _) in enumerate(GROUPS):
        x = start + idx * (w + xgap) + w / 2
        row = index[(BASE, condition)]
        draw.text((x, 275), header, font=font(18, True), anchor="mm", fill=ink)
        stress = {"dark": "×0,60  ·  ", "flat": "×0,45  ·  ", "noise": "σ = 16  ·  "}.get(condition, "")
        draw.text((x, 300), f"{stress}n = {row['n']}", font=font(17), anchor="mm", fill=muted)
    scale = max(5, math.ceil(max(abs(row[metric]) for row in summary) / 5) * 5)

    for line, filter_name in enumerate(filters):
        y = top + line * row_h
        draw.text((pad, y + 22), filter_name, font=font(24, True), fill=ink, anchor="lm")
        for idx, (condition, _, _) in enumerate(GROUPS):
            x = start + idx * (w + xgap)
            val = index[(filter_name, condition)][metric]
            positive = val > 0 if positive_good else val < 0
            strength = min(1, abs(val) / scale)
            tone = (24, 126, 110) if positive else (194, 66, 49)
            neutral = (236, 241, 247)
            tint = tuple(round(c * (1.0 - strength) + b * strength) for c, b in zip(neutral, tone))
            color = "#ffffff" if strength > 0.76 else "#15283c"
            draw.rounded_rectangle((x, y, x + w, y + row_h - 7), radius=8, fill=tint)
            draw.text((x + w / 2, y + (row_h - 7) / 2 - 2), percent(val, signed=True), font=font(25, True), fill=color, anchor="mm")

    base_y = top + len(filters) * row_h + 19
    draw.line((pad, base_y - 14, 1623, base_y - 14), fill="#ced9e6", width=2)
    draw.text((pad, base_y + 8), "Bez filtra", font=font(22, True), fill=ink, anchor="lm")
    for idx, (condition, _, _) in enumerate(GROUPS):
        row = index[(BASE, condition)]
        raw = row[baseline_field]
        if baseline_field == "wrong_accept_base_count":
            raw = raw / row["n"] * 100
        x = start + idx * (w + xgap) + w / 2
        draw.text((x, base_y + 7), percent(raw) + "%", font=font(22, True), fill=ink, anchor="mm")
    draw.text((pad, 840), footnote, font=font(21), fill=ink)
    draw.text((pad, 895), "CNN v3 znaki · PTSD (podzbiór testowy) · 4 zdjęcia na klasę · jedno ustawienie na filtr", font=font(20), fill=muted)
    draw.text((pad, 929), "Ćwiartki są wyznaczone na podstawie jasności pikseli i kontrastu kadru.", font=font(19), fill=muted)
    img.save(path, dpi=(150, 150))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    rows = read_predictions(args.folder)

    summary = summarize(rows)
    with (args.folder / "podsumowanie.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    create_chart(
        args.folder / "wplyw-filtrow-trafnosc.png", summary, "delta_accuracy_pp",
        "Jak filtry zmieniają trafność klasyfikacji?",
        "Zmiana odsetka poprawnie rozpoznanych znaków względem tego samego wariantu bez filtra [p.p.]",
        "Zielony = więcej poprawnych klas. Czerwony = mniej poprawnych klas. Wiersz „Bez filtra” pokazuje poziom wyjściowy.",
        "base_accuracy_pct", True,
    )
    create_chart(
        args.folder / "wplyw-filtrow-weryfikacja.png", summary, "wrong_accept_delta_pp",
        "Ile błędnych predykcji otrzymało ACCEPT?",
        "Zmiana udziału błędnych i zaakceptowanych predykcji wśród wszystkich zdjęć danej grupy [p.p.]",
        "Zielony = mniej zaakceptowanych błędów. Czerwony = więcej zaakceptowanych błędów. Wiersz „Bez filtra” to poziom wyjściowy.",
        "wrong_accept_base_count", False,
    )

if __name__ == "__main__":
    main()
