from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from classifier_adapter import (DEFAULT_MODEL, ClassifierAdapter, inspect_model_input,
                                load_class_names, prepare_image)


class ExampleModel:
    input_shape = (None, 64, 64, 3)
    output_shape = (None, 2)

    def __init__(self):
        self.inputs = []

    def predict(self, batch, verbose=0):
        self.inputs.append(batch.copy())
        return np.array([[0.2, 0.8]], dtype=np.float32)


class AdapterTests(unittest.TestCase):
    def test_saved_model_shape(self):
        self.assertEqual(inspect_model_input(DEFAULT_MODEL), (64, 64))

    def test_rgb_batch_and_original_unchanged(self):
        image = np.full((20, 30, 3), (10, 40, 255), dtype=np.uint8)
        before = image.copy()
        batch = prepare_image(image)
        self.assertEqual(batch.shape, (1, 64, 64, 3))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_array_equal(batch[0, 0, 0], [255, 40, 10])
        np.testing.assert_array_equal(image, before)
        self.assertEqual(prepare_image(image, (32, 48)).shape, (1, 32, 48, 3))

    def test_pair_and_full_probabilities(self):
        model = ExampleModel()
        adapter = ClassifierAdapter(model, ['A-1', 'B-2'])
        image = np.full((20, 30, 3), 80, dtype=np.uint8)
        pair = adapter.compare(image, image + 10)
        self.assertEqual(pair['original'].class_index, 1)
        self.assertEqual(pair['original'].class_name, 'B-2')
        self.assertAlmostEqual(pair['edited'].confidence, 0.8)
        self.assertEqual(len(pair['edited'].probabilities), 2)
        self.assertEqual(len(model.inputs), 2)
        np.testing.assert_array_equal(model.inputs[1] - model.inputs[0], 10)

    def test_invalid_contract(self):
        with self.assertRaises(ValueError):
            ClassifierAdapter(ExampleModel(), ['A-1'])
        with self.assertRaises(ValueError):
            ClassifierAdapter(ExampleModel(), ['A-1', 'A-1'])
        for image in (np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32)):
            with self.assertRaises(ValueError):
                prepare_image(image)
        model = ExampleModel()
        model.predict = lambda batch, verbose=0: np.array([[float('nan'), 0]])
        with self.assertRaises(ValueError):
            ClassifierAdapter(model, ['A-1', 'B-2']).predict(np.zeros((3, 3, 3), np.uint8))

    def test_class_file_preserves_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'classes.json'
            path.write_text('["B-2", "A-1"]', encoding='utf-8')
            self.assertEqual(load_class_names(path), ('B-2', 'A-1'))


if __name__ == '__main__':
    unittest.main()
