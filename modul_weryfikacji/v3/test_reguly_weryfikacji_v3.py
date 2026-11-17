import unittest

from reguly_weryfikacji_v3 import verify_prediction


class TestRegulyWeryfikacjiV3(unittest.TestCase):

    def test_accept(self):
        self.assertEqual(
            verify_prediction(0.99992716, 0.99985431, 0.00076688),
            "ACCEPT"
        )

    def test_accept_na_granicy(self):
        self.assertEqual(
            verify_prediction(0.97, 0.95, 0.12),
            "ACCEPT"
        )

    def test_warn_confidence_ponizej_progu_accept(self):
        self.assertEqual(
            verify_prediction(0.969, 0.95, 0.12),
            "WARN"
        )

    def test_warn_margin_ponizej_progu_accept(self):
        self.assertEqual(
            verify_prediction(0.97, 0.949, 0.12),
            "WARN"
        )

    def test_warn_entropy_powyzej_progu_accept(self):
        self.assertEqual(
            verify_prediction(0.97, 0.95, 0.121),
            "WARN"
        )

    def test_warn_na_granicy_odrzucenia(self):
        self.assertEqual(
            verify_prediction(0.90, 0.70, 0.50),
            "WARN"
        )

    def test_reject_niskie_confidence(self):
        self.assertEqual(
            verify_prediction(0.899, 0.70, 0.50),
            "REJECT"
        )

    def test_reject_niski_margin(self):
        self.assertEqual(
            verify_prediction(0.90, 0.699, 0.50),
            "REJECT"
        )

    def test_reject_wysoka_entropy(self):
        self.assertEqual(
            verify_prediction(0.90, 0.70, 0.501),
            "REJECT"
        )


if __name__ == "__main__":
    unittest.main()
