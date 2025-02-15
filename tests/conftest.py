from pathlib import Path

import pytest

# Ensure package import
import ai_vs_human_text.ml.train as train_mod
from ai_vs_human_text.ml.train import train_and_save


@pytest.fixture(scope="session", autouse=True)
def _ensure_model() -> None:
    root = Path(train_mod.__file__).resolve().parents[3]
    artifacts = root / "artifacts"
    model_path = artifacts / "model.joblib"
    if not model_path.is_file():
        train_and_save(artifacts)
