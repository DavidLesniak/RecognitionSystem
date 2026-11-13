import numpy as np

from reguly_weryfikacji_v3 import verify_prediction


def evaluate_prediction(probabilities):
    """Ocenia jeden wektor Softmax za pomocą progów dla modelu V3 znaki."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("Podaj wektor Softmax dla jednego zdjęcia.")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-5)
    ):
        raise ValueError("Wyniki Softmax muszą być nieujemne i sumować się do 1.")

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    sorted_probabilities = np.sort(probabilities)
    margin = confidence - float(sorted_probabilities[-2])
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))

    return {
        "class_index": class_index,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "decision": verify_prediction(confidence, margin, entropy),
    }
