import unittest

from reguly_weryfikacji_v3 import verify_prediction


class TestRegulyWeryfikacjiV3(unittest.TestCase):
    def test_accept_na_granicy(self):
        self.assertEqual(verify_prediction(0.97, 0.95, 0.12), "ACCEPT")

    def test_accept_wymaga_wszystkich_warunkow(self):
        cases = [
            (0.969, 0.95, 0.12),
            (0.97, 0.949, 0.12),
            (0.97, 0.95, 0.121),
        ]
        for metrics in cases:
            with self.subTest(metrics=metrics):
                self.assertEqual(verify_prediction(*metrics), "WARN")

    def test_warn_na_granicy_odrzucenia(self):
        self.assertEqual(verify_prediction(0.90, 0.70, 0.50), "WARN")

    def test_reject_wystarczy_jeden_warunek(self):
        cases = [
            (0.899, 0.70, 0.50),
            (0.90, 0.699, 0.50),
            (0.90, 0.70, 0.501),
        ]
        for metrics in cases:
            with self.subTest(metrics=metrics):
                self.assertEqual(verify_prediction(*metrics), "REJECT")

    def test_accept_dla_przykladu_a1(self):
        metrics = (0.99992716, 0.99985431, 0.00076688)
        self.assertEqual(verify_prediction(*metrics), "ACCEPT")


if __name__ == "__main__":
    unittest.main()
