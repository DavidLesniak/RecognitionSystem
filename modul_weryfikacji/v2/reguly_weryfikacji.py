
def verify_prediction(confidence, margin, entropy):
    """
    Zwraca decyzję dotyczącą wiarygodności predykcji.

    ACCEPT - predykcja spełnia rygorystyczne kryteria.
    WARN   - predykcja wymaga ostrożności.
    REJECT - predykcja wykazuje dużą niepewność.

    Progi stanowią wstępną konfigurację modułu.
    """

    if (
        confidence >= 0.99
        and margin >= 0.99
        and entropy <= 0.05
    ):
        return "ACCEPT"

    if (
        confidence < 0.90
        or margin < 0.70
        or entropy > 0.50
    ):
        return "REJECT"

    return "WARN"
