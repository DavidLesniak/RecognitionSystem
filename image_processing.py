"""Powtarzalne przekształcenia zdjęć i zapis eksperymentu."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np


PIPELINE_VERSION = 1
OPERATIONS = ["denoise", "brightness_contrast", "gamma", "clahe", "saturation", "sharpen"]


@dataclass(frozen=True)
class EditSettings:
    brightness: int = 0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    sharpen: float = 0.0
    denoise: int = 0
    clahe: bool = False


def apply_adjustments(image, settings):
    """Przyjmij BGR uint8 i zwróć nowy obraz w oryginalnej rozdzielczości."""
    result = image.copy()
    if settings.denoise:
        result = cv2.fastNlMeansDenoisingColored(
            result, None, settings.denoise, settings.denoise, 7, 21
        )
    if settings.brightness != 0 or settings.contrast != 1:
        result = np.clip(
            (result.astype(np.float32) - 127.5) * settings.contrast
            + 127.5 + settings.brightness, 0, 255
        ).astype(np.uint8)
    if settings.gamma != 1:
        lut = np.round(255 * (np.arange(256) / 255.0) ** settings.gamma).astype(np.uint8)
        result = cv2.LUT(result, lut)
    if settings.clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if settings.saturation != 1:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * settings.saturation, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if settings.sharpen:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
        result = cv2.addWeighted(result, 1 + settings.sharpen, blurred, -settings.sharpen, 0)
    return result


def experiment_metadata(source_name, source_sha256, image, settings):
    height, width = image.shape[:2]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "source_name": source_name,
        "source_sha256": source_sha256,
        "width": width,
        "height": height,
        "settings": asdict(settings),
        "operations": OPERATIONS,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }


def save_experiment(path, image, metadata):
    """Zapisz PNG i JSON; zgłoś niepełny zapis, jeśli druga operacja zawiedzie."""
    path = Path(path)
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Nie można zakodować obrazu PNG.")
    data.tofile(path)
    try:
        path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as error:
        raise OSError(f"Obraz zapisano, ale zapis parametrów JSON nie powiódł się: {error}") from error
