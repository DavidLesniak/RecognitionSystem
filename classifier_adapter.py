"""Wspólne wejście i wyjście klasyfikatora dla obrazów z edytora."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


DEFAULT_MODEL = Path(__file__).resolve().parent / 'traffic classifier archive/v2/traffic_classifier_v2.keras'


def inspect_model_input(path):
    """Odczytaj kształt wejścia bez importu TensorFlow i bez ładowania wag."""
    with ZipFile(path) as archive:
        config = json.loads(archive.read('config.json'))
    inputs = [layer for layer in config['config']['layers'] if layer['class_name'] == 'InputLayer']
    if len(inputs) != 1:
        raise ValueError('Obsługiwany jest model z jednym wejściem obrazu.')
    shape = inputs[0]['config'].get('batch_shape') or inputs[0]['config'].get('batch_input_shape')
    return validate_input_shape(shape)


def validate_input_shape(shape):
    if not shape or len(shape) != 4 or shape[0] not in (None, 1) or shape[3] != 3:
        raise ValueError('Model musi przyjmować jeden obraz RGB: (batch, wysokość, szerokość, 3).')
    if not all(isinstance(dimension, int) and dimension > 0 for dimension in shape[1:3]):
        raise ValueError('Model musi określać stałą wysokość i szerokość wejścia.')
    return tuple(shape[1:3])


def prepare_image(image_bgr, image_size=(64, 64)):
    """BGR uint8 → RGB, bilinear Pillow, float32 0–255, batch=1.

    Przygotowanie zgodne ze skryptem pierwsza_predykcja.py dla modelu v2.
    Nie dzielimy ponownie przez 255. image_size ma kolejność (wysokość, szerokość).
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise ValueError('Oczekiwany jest obraz NumPy uint8 z OpenCV.')
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError('Oczekiwany jest niepusty obraz z trzema kanałami BGR.')
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError('Wymiary wejściowe muszą być dodatnie.')
    rgb = Image.fromarray(image_bgr[:, :, ::-1])
    resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, ...]


def load_class_names(path):
    """Lista JSON musi pochodzić z class_names zbioru treningowego danego modelu."""
    names = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Plik klas musi zawierać niepustą listę unikalnych nazw.')
    return tuple(names)


@dataclass(frozen=True)
class Prediction:
    class_index: int
    class_name: str
    confidence: float
    probabilities: tuple


class ClassifierAdapter:
    def __init__(self, model, class_names):
        self.image_size = validate_input_shape(model.input_shape)
        self.class_names = tuple(class_names)
        if (not self.class_names or len(set(self.class_names)) != len(self.class_names)
                or not all(isinstance(name, str) and name.strip() for name in self.class_names)):
            raise ValueError('Wymagana jest uporządkowana lista unikalnych nazw klas z treningu.')
        if (len(model.output_shape) != 2
                or model.output_shape[-1] != len(self.class_names)):
            raise ValueError('Liczba klas nie odpowiada wyjściu modelu.')
        self.model = model

    @classmethod
    def from_files(cls, model_path, classes_path):
        names = load_class_names(classes_path)
        try:
            from tensorflow.keras.models import load_model
        except ImportError as error:
            raise RuntimeError('Uruchomienie klasyfikacji wymaga TensorFlow z requirements.txt.') from error
        return cls(load_model(model_path, compile=False), names)

    def predict(self, image_bgr):
        batch = prepare_image(image_bgr, self.image_size)
        output = np.asarray(self.model.predict(batch, verbose=0))
        if output.shape != (1, len(self.class_names)):
            raise ValueError('Nieprawidłowy kształt predykcji modelu.')
        probabilities = output[0]
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1, atol=1e-4)):
            raise ValueError('Model nie zwrócił prawidłowych prawdopodobieństw softmax.')
        index = int(np.argmax(probabilities))
        return Prediction(index, self.class_names[index], float(probabilities[index]),
                          tuple(float(value) for value in probabilities))

    def compare(self, original_bgr, edited_bgr):
        """Jednakowe przygotowanie obu wariantów; bez oceny poprawności wyniku."""
        return {'original': self.predict(original_bgr), 'edited': self.predict(edited_bgr)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sprawdzenie wejścia modelu bez wykonywania klasyfikacji.')
    parser.add_argument('image', type=Path, help='Ścieżka do przykładowego zdjęcia')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    from app import read_image
    size = inspect_model_input(args.model)
    batch = prepare_image(read_image(args.image), size)
    print(f'Model: {args.model.name}')
    print(f'Wejście: {batch.shape}, typ: {batch.dtype}, kanały: RGB')
    print(f'Zakres pikseli: {batch.min():.1f}–{batch.max():.1f} (skala 0–255)')
    print('Przygotowanie wejścia: OK. Predykcja nie była uruchamiana.')
