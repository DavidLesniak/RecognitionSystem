"""Sprawdzenia przekształceń, zapisu oraz przepływu edytora."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import threading
import tkinter as tk
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import ImageViewer, read_image
from image_processing import EditSettings, apply_adjustments, experiment_metadata, save_experiment


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)

    def test_neutral_and_repeatability(self):
        original = self.image.copy()
        neutral = apply_adjustments(self.image, EditSettings())
        np.testing.assert_array_equal(neutral, original)
        self.assertFalse(np.shares_memory(neutral, self.image))
        settings = EditSettings(brightness=10, contrast=1.2, gamma=0.8,
                                saturation=1.3, sharpen=1, denoise=3, clahe=True)
        result = apply_adjustments(self.image, settings)
        np.testing.assert_array_equal(result, apply_adjustments(self.image, settings))
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_direction_and_saturation(self):
        image = np.full((8, 8, 3), 100, np.uint8)
        self.assertTrue((apply_adjustments(image, EditSettings(brightness=20)) == 120).all())
        self.assertTrue((apply_adjustments(image, EditSettings(gamma=0.5)) > image).all())
        self.assertTrue((apply_adjustments(image, EditSettings(gamma=2)) < image).all())
        gray = apply_adjustments(self.image, EditSettings(saturation=0))
        np.testing.assert_array_equal(gray[:, :, 0], gray[:, :, 1])
        np.testing.assert_array_equal(gray[:, :, 1], gray[:, :, 2])
        self.assertTrue((apply_adjustments(np.full_like(image, 255), EditSettings(brightness=100)) == 255).all())
        self.assertTrue((apply_adjustments(np.zeros_like(image), EditSettings(brightness=-100)) == 0).all())

    def test_small_and_gray_images(self):
        for image in (np.zeros((1, 1, 3), np.uint8), np.full((20, 30, 3), 120, np.uint8)):
            for settings in (EditSettings(clahe=True), EditSettings(denoise=10),
                             EditSettings(sharpen=2), EditSettings(contrast=0.5)):
                self.assertEqual(apply_adjustments(image, settings).shape, image.shape)

    def test_export_and_reproduction(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'zdjęcie żółć.png'
            settings = EditSettings(brightness=12, clahe=True)
            result = apply_adjustments(self.image, settings)
            metadata = experiment_metadata('źródło.jpg', 'abc', result, settings)
            save_experiment(target, result, metadata)
            np.testing.assert_array_equal(read_image(target), result)
            saved = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
            np.testing.assert_array_equal(
                apply_adjustments(self.image, EditSettings(**saved['settings'])), result)
            with patch.object(Path, 'write_text', side_effect=OSError('brak dostępu')):
                with self.assertRaisesRegex(OSError, 'Obraz zapisano'):
                    save_experiment(target, result, metadata)

    def test_large_image(self):
        image = np.random.default_rng(7).integers(0, 256, (800, 1200, 3), dtype=np.uint8)
        result = apply_adjustments(image, EditSettings(
            brightness=-15, contrast=1.3, gamma=0.7, saturation=1.2,
            sharpen=0.5, denoise=3, clahe=True))
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)


class EditorTests(unittest.TestCase):
    def test_visible_window_settles_and_allows_save(self):
        root = tk.Tk()
        viewer = ImageViewer(root)
        calls = []

        def process(image, settings):
            calls.append(settings)
            return apply_adjustments(image, settings)

        try:
            root.update()
            viewer.image = np.full((467, 700, 3), 100, np.uint8)
            viewer.reset()
            with patch('app.apply_adjustments', side_effect=process):
                for name, value in dict(brightness=51, contrast=1.9, saturation=1.85,
                                        gamma=0.55, sharpen=2).items():
                    viewer.variables[name].set(value)
                viewer.denoise.set('Mocne')
                viewer.clahe.set(True)
                viewer.mark_dirty()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                    if viewer.future is None and viewer.edit_job is None:
                        break
                # Po zakończeniu poczekaj na dodatkowe zdarzenia widocznych suwaków.
                until = time.monotonic() + 0.7
                while time.monotonic() < until:
                    root.update()
                    time.sleep(0.01)
                self.assertEqual(len(calls), 1)
                self.assertIsNone(viewer.future)
                self.assertIsNone(viewer.edit_job)
                for button in (viewer.save_button, viewer.open_button, viewer.reset_button):
                    self.assertEqual(str(button['state']), 'normal')
                with TemporaryDirectory() as directory:
                    viewer.source_path = Path(directory) / 'source.jpg'
                    target = Path(directory) / 'wynik.png'
                    with patch('app.filedialog.asksaveasfilename', return_value=str(target)):
                        viewer.save_button.invoke()
                    np.testing.assert_array_equal(read_image(target), viewer.result)
                    saved = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
                    self.assertEqual(saved['settings']['brightness'], 51)
        finally:
            viewer.close()

    def test_live_updates_keep_latest_settings(self):
        root = tk.Tk()
        root.withdraw()
        viewer = ImageViewer(root)
        started, release = threading.Event(), threading.Event()
        calls = []

        def slow_adjust(image, settings):
            calls.append(settings.brightness)
            started.set()
            release.wait(5)
            return apply_adjustments(image, settings)

        def pump_until(condition):
            deadline = time.monotonic() + 8
            while not condition() and time.monotonic() < deadline:
                root.update()
                time.sleep(0.01)
            self.assertTrue(condition())

        try:
            viewer.image = np.full((20, 30, 3), 100, np.uint8)
            viewer.reset()
            with patch('app.apply_adjustments', side_effect=slow_adjust):
                for value in (5, 10, 15):
                    viewer.variables['brightness'].set(value)
                pump_until(started.is_set)
                self.assertEqual(calls, [15])
                self.assertEqual(str(viewer.widgets[0]['state']), 'normal')
                viewer.variables['brightness'].set(40)
                self.assertEqual(str(viewer.save_button['state']), 'disabled')
                release.set()
                pump_until(lambda: viewer.future is None and viewer.edit_job is None)
                self.assertEqual(calls, [15, 40])
                self.assertTrue((viewer.result == 140).all())
                self.assertEqual(str(viewer.save_button['state']), 'normal')
                viewer.variables['brightness'].set(70)
                viewer.reset()
                self.assertIsNone(viewer.edit_job)
                np.testing.assert_array_equal(viewer.result, viewer.image)
        finally:
            release.set()
            viewer.close()

    def test_workflow(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'oryginał.png'
            image = np.full((200, 300, 3), (0, 80, 200), np.uint8)
            cv2.imencode('.png', image)[1].tofile(source)
            bad = Path(directory) / 'błąd.png'
            bad.write_bytes(b'not an image')
            root = tk.Tk()
            root.withdraw()
            viewer = ImageViewer(root)
            try:
                self.assertEqual(str(viewer.save_button['state']), 'disabled')
                with patch('app.filedialog.askopenfilename', return_value=str(source)):
                    viewer.choose_image()
                np.testing.assert_array_equal(viewer.result, image)
                viewer.variables['brightness'].set(25)
                self.assertEqual(str(viewer.save_button['state']), 'disabled')
                viewer.apply()
                self.assertEqual(str(viewer.open_button['state']), 'disabled')
                deadline = time.monotonic() + 10
                while viewer.future is not None and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                self.assertIsNone(viewer.future)
                self.assertEqual(str(viewer.save_button['state']), 'normal')
                self.assertTrue((viewer.result[:, :, 0] == 25).all())
                np.testing.assert_array_equal(viewer.image, image)
                target = Path(directory) / 'wynik.png'
                with patch('app.filedialog.asksaveasfilename', return_value=str(target)):
                    viewer.save()
                self.assertTrue(target.exists())
                with patch('app.filedialog.asksaveasfilename', return_value=str(source)), patch('app.messagebox.showerror') as error:
                    viewer.save()
                    error.assert_called_once()
                np.testing.assert_array_equal(read_image(source), image)
                with patch('app.filedialog.asksaveasfilename', return_value=str(target)), patch('app.messagebox.askyesno', return_value=False) as confirm:
                    viewer.save()
                    confirm.assert_called_once()
                with patch('app.filedialog.askopenfilename', return_value=''):
                    viewer.choose_image()
                self.assertEqual(viewer.source_path, source)
                with patch('app.filedialog.askopenfilename', return_value=str(bad)), patch('app.messagebox.showerror') as error:
                    viewer.choose_image()
                    error.assert_called_once()
                self.assertEqual(viewer.source_path, source)
                viewer.reset()
                np.testing.assert_array_equal(viewer.result, image)
                self.assertEqual(viewer.settings(), EditSettings())
                with patch('app.apply_adjustments', side_effect=ValueError('próba błędu')), patch('app.messagebox.showerror') as error:
                    viewer.variables['gamma'].set(0.8)
                    viewer.apply()
                    deadline = time.monotonic() + 10
                    while viewer.future is not None and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    error.assert_called_once()
                    self.assertEqual(str(viewer.save_button['state']), 'disabled')
                    self.assertEqual(str(viewer.open_button['state']), 'normal')
                with patch('app.filedialog.askopenfilename', return_value=str(source)):
                    viewer.choose_image()
                self.assertEqual(viewer.settings(), EditSettings())
            finally:
                viewer.close()


if __name__ == '__main__':
    unittest.main()
