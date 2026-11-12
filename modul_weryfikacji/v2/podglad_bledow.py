
import csv
from pathlib import Path
from PIL import Image, ImageDraw

CSV_PATH = "bledne_predykcje.csv"
OUTPUT_PATH = "podglad_bledow.jpg"

NUMBER_OF_IMAGES = 10
COLUMNS = 2
ROWS = 5

CELL_WIDTH = 440
CELL_HEIGHT = 220

# Odczyt niezgodnych predykcji
with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
    results = list(csv.DictReader(file, delimiter=";"))

selected = results[:NUMBER_OF_IMAGES]

# Przygotowanie planszy
board = Image.new(
    "RGB",
    (COLUMNS * CELL_WIDTH, ROWS * CELL_HEIGHT),
    color="white"
)

draw = ImageDraw.Draw(board)

for index, row in enumerate(selected):
    column = index % COLUMNS
    row_number = index // COLUMNS

    x = column * CELL_WIDTH
    y = row_number * CELL_HEIGHT

    image_path = Path(row["zdjecie"])

    # Wczytanie i powiększenie zdjęcia
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((180, 150))

        scale = min(180 / image.width, 150 / image.height)
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale))
        )

        image = image.resize(new_size)

    board.paste(image, (x + 15, y + 15))

    # Opis zdjęcia
    text_x = x + 205

    draw.text(
        (text_x, y + 20),
        f"Przypadek {index + 1}",
        fill="black"
    )

    draw.text(
        (text_x, y + 55),
        f"Etykieta: {row['prawdziwa_klasa']}",
        fill="black"
    )

    draw.text(
        (text_x, y + 80),
        f"Predykcja: {row['przewidziana_klasa']}",
        fill="black"
    )

    draw.text(
        (text_x, y + 105),
        f"Confidence: {float(row['confidence']):.6f}",
        fill="black"
    )

    draw.text(
        (x + 15, y + 185),
        image_path.name[:55],
        fill="black"
    )

    draw.rectangle(
        (x, y, x + CELL_WIDTH - 1, y + CELL_HEIGHT - 1),
        outline="gray"
    )

board.save(OUTPUT_PATH, quality=95)

print(f"Zapisano plansze: {OUTPUT_PATH}")
print(f"Liczba zdjec: {len(selected)}")
