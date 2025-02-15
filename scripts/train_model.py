#!/usr/bin/env python3
"""Train the sklearn pipeline and write artifacts (model.joblib + metrics.json)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

try:
    from ai_vs_human_text.ml.train import train_and_save
except ImportError:
    sys.path.insert(0, str(_ROOT / "src"))
    from ai_vs_human_text.ml.train import train_and_save  # noqa: E402


def main() -> None:
    path = train_and_save()
    print(f"Wrote model to {path}")


if __name__ == "__main__":
    main()
