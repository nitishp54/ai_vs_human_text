from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import structlog

from ai_vs_human_text.config import Settings, get_settings
from ai_vs_human_text.exceptions import (
    InputValidationError,
    ModelNotLoadedError,
    PredictionError,
)

logger = structlog.get_logger(__name__)


class TextClassifierService:
    """Thread-safe lazy-loaded sklearn pipeline (CountVectorizer + Tfidf + MultinomialNB)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._pipeline: Any | None = None
        self._metrics: dict[str, Any] | None = None

    def _load_metrics(self) -> dict[str, Any]:
        path: Path = self._settings.metrics_path
        if not path.is_file():
            return {
                "note": "metrics file missing; run scripts/train_model.py",
                "source": "ecc_coursework_reference",
            }
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            path = self._settings.model_path
            if not path.is_file():
                raise ModelNotLoadedError(
                    f"Model artifact not found at {path}. Run: python scripts/train_model.py",
                    code="model_artifact_missing",
                    details={"path": str(path)},
                )
            self._pipeline = joblib.load(path)
            self._metrics = self._load_metrics()
            logger.info("model_loaded", path=str(path))

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def model_info(self) -> dict[str, Any]:
        self.load()
        assert self._metrics is not None
        return {
            "model_path": str(self._settings.model_path),
            "metrics": self._metrics,
        }

    def _validate_text(self, text: str) -> str:
        t = text.strip()
        if len(t) < self._settings.min_input_chars:
            raise InputValidationError(
                "Text is empty or too short after stripping whitespace.",
                code="text_too_short",
                details={"min_chars": self._settings.min_input_chars},
            )
        if len(text) > self._settings.max_input_chars:
            raise InputValidationError(
                f"Text exceeds maximum length ({self._settings.max_input_chars} characters).",
                code="text_too_long",
                details={"max_chars": self._settings.max_input_chars},
            )
        return t

    def predict(self, text: str) -> dict[str, Any]:
        self.load()
        assert self._pipeline is not None
        cleaned = self._validate_text(text)
        try:
            label = self._pipeline.predict([cleaned])[0]
            proba = None
            if hasattr(self._pipeline, "predict_proba"):
                proba_arr = self._pipeline.predict_proba([cleaned])[0]
                classes = list(self._pipeline.classes_)
                proba = {str(c): float(p) for c, p in zip(classes, proba_arr, strict=True)}
        except Exception as e:
            logger.exception("prediction_failed")
            raise PredictionError(
                "Classifier failed on input; check preprocessing and model health.",
                code="prediction_failed",
                details={"error": str(e)},
            ) from e

        label_str = str(label)
        return {
            "label": label_str,
            "confidence": (max(proba.values()) if proba else None),
            "probabilities": proba,
            "input_char_count": len(cleaned),
        }
