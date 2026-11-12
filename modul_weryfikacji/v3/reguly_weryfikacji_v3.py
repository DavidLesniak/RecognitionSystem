def verify_prediction(confidence, margin, entropy):
    """Zwraca decyzję według progów dobranych na walidacji CNN V3."""
    if confidence >= 0.97 and margin >= 0.95 and entropy <= 0.12:
        return "ACCEPT"

    if confidence < 0.90 or margin < 0.70 or entropy > 0.50:
        return "REJECT"

    return "WARN"
