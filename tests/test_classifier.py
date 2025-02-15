from ai_vs_human_text.deps import get_classifier


def test_predict_human_leaning() -> None:
    clf = get_classifier()
    out = clf.predict("lol ngl this is kinda rough tbh")
    assert out["label"] in ("human", "ai")
    assert out.get("probabilities") is not None


def test_predict_ai_leaning() -> None:
    clf = get_classifier()
    out = clf.predict(
        "Furthermore, the methodology adheres to established best practices and limitations."
    )
    assert out["label"] in ("human", "ai")
