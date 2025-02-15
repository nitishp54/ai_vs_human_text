from ai_vs_human_text.ml.classifier import TextClassifierService

_classifier: TextClassifierService | None = None


def get_classifier() -> TextClassifierService:
    global _classifier
    if _classifier is None:
        _classifier = TextClassifierService()
    return _classifier


def reset_classifier_for_tests() -> None:
    global _classifier
    _classifier = None
