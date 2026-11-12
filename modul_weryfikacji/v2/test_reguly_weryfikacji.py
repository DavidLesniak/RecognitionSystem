
import unittest

from reguly_weryfikacji import verify_prediction


class TestRegulyWeryfikacji(unittest.TestCase):

    def test_accept(self):
        self.assertEqual(
            verify_prediction(0.999, 0.999, 0.001),
            "ACCEPT"
        )

    def test_warn(self):
        self.assertEqual(
            verify_prediction(0.95, 0.90, 0.20),
            "WARN"
        )

    def test_reject(self):
        self.assertEqual(
            verify_prediction(0.60, 0.20, 0.80),
            "REJECT"
        )

    def test_accept_na_granicy(self):
        self.assertEqual(
            verify_prediction(0.99, 0.99, 0.05),
            "ACCEPT"
        )

    def test_reject_niskie_confidence(self):
        self.assertEqual(
            verify_prediction(0.89, 0.90, 0.20),
            "REJECT"
        )


if __name__ == "__main__":
    unittest.main()
