"""Rank the filters by how often the predicted class changes."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from plot_filter_benchmark import BASE, chart_font, percent


def create_plot(folder):
    with (folder / "podsumowanie.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = [r for r in csv.DictReader(handle) if r["condition"] == "real"]
    reference = next(row for row in source if row["filter"] == BASE)
    metric = "changed_label_pct"
    ranked = sorted(
        (row for row in source if row["filter"] != BASE),
        key=lambda row: (-float(row[metric]), -float(row["changed_decision_pct"])),
    )

    font = chart_font

    surface = Image.new("RGB", (1600, 900), "#f6f8fb")
    draw = ImageDraw.Draw(surface)
    ink, muted = "#142940", "#455d74"
    left, chart_width = 326, 348
    decision = 762
    right = 1387

    draw.text((62, 41), "RECOGNITIONSYSTEM  /  CNN V3", font=font(22, True), fill="#245879")
    draw.text((62, 85), "Które filtry najbardziej zmieniają wynik?", font=font(48, True), fill=ink)
    count = reference["n"]
    accuracy = percent(float(reference["base_accuracy_pct"]))
    draw.text(
        (62, 157), f"{count} zdjęć testowych PTSD z 94 klas  ·  trafność bez filtrów {accuracy}%",
        font=font(24), fill=muted,
    )

    for x, label in ((62, "Filtr"), (left, "Zmiana rozpoznanej klasy"), (decision, "Zmiana decyzji weryfikatora")):
        draw.text((x, 229), label, font=font(22, True), fill=ink)
    draw.text((right, 229), "Zmiana trafności", font=font(22, True), fill=ink, anchor="mt")
    draw.text((right, 265), "w punktach procentowych", font=font(17), fill=muted, anchor="mt")

    for x in (left, decision):
        for tick in (0, 5, 10, 15, 20, 25):
            xpos = x + chart_width * tick / 25
            draw.text((xpos, 274), f"{tick}%", font=font(16), fill=muted, anchor="mt")
            draw.line((xpos, 306, xpos, 731), fill="#dce5ee", width=1)

    for index, row in enumerate(ranked):
        y = 327 + 63 * index
        draw.text((62, y), row["filter"], font=font(25, True), fill=ink, anchor="lm")
        for x, name, col in (
            (left, metric, "#246990"),
            (decision, "changed_decision_pct", "#6a5aa3"),
        ):
            value = float(row[name])
            end = x + chart_width * value / 25
            draw.rounded_rectangle((x, y - 11, end, y + 11), radius=5, fill=col)
            draw.text((end + 13, y), percent(value) + "%", font=font(21, True), fill=ink, anchor="lm")
        delta = float(row["delta_accuracy_pp"])
        fill, txt = ("#fbe9e7", "#953f33") if delta < 0 else ("#e2f1e9", "#176751")
        draw.rounded_rectangle((right - 79, y - 20, right + 79, y + 20), radius=10, fill=fill)
        draw.text((right, y), percent(delta, signed=True) + " p.p.", font=font(22, True), fill=txt, anchor="mm")

    draw.line((62, 778, 1538, 778), fill="#cddae6", width=2)
    draw.text((62, 798), "Słupki pokazują odsetek zdjęć, dla których wynik różnił się od wyniku bez filtra.", font=font(22), fill=ink)
    draw.text((62, 834), "Decyzja weryfikatora to ACCEPT, WARN lub REJECT. Kolejność według zmiany klasy. Jedno ustawienie na filtr.", font=font(19), fill=muted)
    surface.save(folder / "ranking-wplywu-filtrow.png", dpi=(150, 150))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    create_plot(parser.parse_args().folder)
