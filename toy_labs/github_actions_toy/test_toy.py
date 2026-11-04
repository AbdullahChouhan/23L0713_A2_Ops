"""Task 6.6 Toy CI Test File (23L0713_A2_Ops)."""


def normalize_confidence(prob_ai: float, threshold: float = 0.50) -> tuple[str, float]:
    if prob_ai >= threshold:
        return "AI-Generated", round(prob_ai, 4)
    return "Real", round(1.0 - prob_ai, 4)


def test_ai_prediction():
    label, conf = normalize_confidence(0.85, threshold=0.50)
    assert label == "AI-Generated"
    assert conf == 0.85


def test_real_prediction():
    label, conf = normalize_confidence(0.20, threshold=0.50)
    assert label == "Real"
    assert conf == 0.80
