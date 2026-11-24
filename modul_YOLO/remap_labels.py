import os
from pathlib import Path

DATASET_DIR = Path(os.path.join(os.getcwd(), '..', 'TROBP12k_ujecia')).resolve()
SIGN_CLASSES = {1, 6, 8, 10}

for txt_file in DATASET_DIR.rglob("*.txt"):
    if txt_file.name in ("classes.txt", "README.txt"):
        continue
        
    lines = txt_file.read_text(encoding="utf-8").splitlines()
    new_lines = [
        f"0 {' '.join(line.split()[1:])}"
        for line in lines
        if line.strip() and int(line.split()[0]) in SIGN_CLASSES
    ]
    txt_file.write_text("\n".join(new_lines), encoding="utf-8")

print(f"Gotowe! Etykiety w {DATASET_DIR} zostały zaktualizowane do 1 klasy.")
