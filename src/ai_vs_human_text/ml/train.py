"""Train and persist the sklearn pipeline + metrics (used by scripts and tests)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Demo corpus: clearly separated styles. For publication-grade numbers, train on the Kaggle
# dataset cited in README (same pipeline as IU ECC coursework).
_HUMAN_SAMPLES = [
    "idk lol kinda tired today ngl",
    "brb grabbing food",
    "u coming to the thing tonight??",
    "haha yeah that tracks",
    "omg no way really",
    "im so done with this assignment lol",
    "gonna sleep early for once maybe",
    "tbh i didnt read the whole thing",
    "wait what that doesnt make sense",
    "yo can u send that link again",
    "nah im good thanks tho",
    "this coffee is hitting different",
    "skipped lecture again whoops",
    "lol same here honestly",
    "cant believe its already thursday",
    "my brain is fried rn",
    "lets just wing it",
    "that's rough buddy",
    "i swear the bus was late again",
    "pizza > salad any day",
    "gonna nap then figure it out",
    "why is everything due the same week",
    "lowkey stressed but whatever",
    "k cool sounds good",
    "i forgot my charger again",
    "random but ok",
    "not my proudest moment",
    "we should hang sometime soon",
    "yeah no i get what u mean",
    "alright bet see u there",
]

_AI_SAMPLES = [
    "Furthermore, it is important to note that the methodology adheres to established best practices.",
    "In conclusion, the analysis demonstrates a comprehensive evaluation of the proposed framework.",
    "This section outlines the key considerations relevant to scalability and operational efficiency.",
    "The implementation leverages industry-standard techniques to ensure robustness and maintainability.",
    "It should be emphasized that the results are subject to the limitations described herein.",
    "Moreover, the approach facilitates seamless integration with existing enterprise systems.",
    "The following paragraphs summarize the principal findings and their implications.",
    "To address this challenge, a multi-layered strategy was adopted to mitigate potential risks.",
    "The dataset was preprocessed to remove extraneous noise and normalize textual representations.",
    "Consequently, the model exhibits strong performance across the evaluated benchmark tasks.",
    "It is worth noting that additional validation may be required in production environments.",
    "The proposed solution balances accuracy with computational efficiency in a principled manner.",
    "Several stakeholders contributed feedback that informed the final design decisions.",
    "The architecture is modular, enabling incremental enhancements without service disruption.",
    "Empirical results indicate a statistically significant improvement over baseline approaches.",
    "Security considerations were incorporated at each stage of the development lifecycle.",
    "The system provides observability hooks for monitoring latency and error budgets.",
    "Documentation has been structured to support onboarding and operational runbooks.",
    "Future work may explore multilingual extensions and adversarial robustness.",
    "The evaluation protocol aligns with commonly accepted metrics in the literature.",
    "Configuration parameters can be tuned to accommodate diverse deployment contexts.",
    "Error handling paths were designed to degrade gracefully under partial failures.",
    "The pipeline supports reproducible training through versioned artifacts and manifests.",
    "Input validation ensures that malformed requests are rejected with actionable errors.",
    "Load testing suggests the service can sustain expected peak traffic profiles.",
    "Caching strategies were considered to reduce redundant computation where appropriate.",
    "The design prioritizes clear separation between domain logic and transport adapters.",
    "Structured logging facilitates correlation across distributed request traces.",
    "Compliance requirements were reviewed with respect to data retention policies.",
    "The following table summarizes comparative performance across representative scenarios.",
]


def build_demo_pipeline() -> tuple[Pipeline, dict]:
    texts = _HUMAN_SAMPLES + _AI_SAMPLES
    labels = ["human"] * len(_HUMAN_SAMPLES) + ["ai"] * len(_AI_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipe = Pipeline(
        [
            ("vect", CountVectorizer(stop_words="english", max_features=20_000, ngram_range=(1, 2))),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {
        "artifact": "synthetic_demo",
        "note": (
            "Trained on bundled stylistic demo corpus for CI and local dev. "
            "For ECC-comparable metrics, train on the Kaggle AI-vs-human dataset (see README)."
        ),
        "accuracy": acc,
        "classification_report": report,
        "coursework_reference_metrics": {
            "source": "IU ENGR-E516 ECC final report (April 2024)",
            "overall_accuracy": 0.95,
            "human": {"precision": 0.94, "recall": 0.99, "f1": 0.96},
            "ai": {"precision": 0.98, "recall": 0.89, "f1": 0.94},
        },
    }
    return pipe, metrics


def train_and_save(artifacts_dir: Path | None = None) -> Path:
    # ml/train.py -> parents[3] is repository root (contains /artifacts)
    root = Path(__file__).resolve().parents[3]
    out_dir = artifacts_dir or (root / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    pipe, metrics = build_demo_pipeline()
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model_path
