def verify_prediction(confidence, margin, entropy):
    """
    Zwraca decyzję dotyczącą wiarygodności predykcji.

    ACCEPT - predykcja spełnia rygorystyczne kryteria.
    WARN   - predykcja wymaga ostrożności.
    REJECT - predykcja wykazuje dużą niepewność.

    Progi dobrano na zbiorze walidacyjnym CNN V3.
    """

    if (
        confidence >= 0.97
        and margin >= 0.95
        and entropy <= 0.12
    ):
        return "ACCEPT"

    if (
        confidence < 0.90
        or margin < 0.70
        or entropy > 0.50
    ):
        return "REJECT"

    return "WARN"
