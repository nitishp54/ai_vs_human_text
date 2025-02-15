"""
FastMCP server: stdio transport for Claude Desktop / Cursor / custom MCP clients.

Tools:
  - classify_text: TF-IDF + MultinomialNB prediction
  - model_info: artifact path + metrics
  - explain_classification: optional Groq narrative (requires GROQ_API_KEY)
"""

from __future__ import annotations

import json

import structlog
from fastmcp import FastMCP

from ai_vs_human_text.deps import get_classifier
from ai_vs_human_text.exceptions import AppError, LLMUnavailableError
from ai_vs_human_text.services.llm import GroqCompletionService

logger = structlog.get_logger(__name__)

mcp = FastMCP(
    "AI vs Human Text",
    instructions=(
        "Use classify_text for statistical human vs AI labels. "
        "Call model_info when the user asks about accuracy, metrics, or the model. "
        "Use explain_classification only when an LLM-style rationale is explicitly requested "
        "and GROQ_API_KEY is configured in the server environment."
    ),
)


@mcp.tool()
def classify_text(text: str) -> dict:
    """Classify text as human-written or AI-generated (sklearn pipeline)."""
    try:
        return get_classifier().predict(text)
    except AppError:
        raise
    except Exception as e:
        logger.exception("mcp_classify_failed")
        raise RuntimeError(f"classification_failed: {e}") from e


@mcp.tool()
def model_info() -> dict:
    """Return model path and evaluation metrics JSON."""
    try:
        return get_classifier().model_info()
    except AppError:
        raise
    except Exception as e:
        logger.exception("mcp_model_info_failed")
        raise RuntimeError(f"model_info_failed: {e}") from e


@mcp.tool()
async def explain_classification(text: str, label: str, probabilities_json: str = "{}") -> str:
    """
    Generate a short LLM explanation for a given label and optional probability map.
    probabilities_json: JSON object string, e.g. '{"human":0.2,"ai":0.8}'.
    """
    llm = GroqCompletionService()
    if not llm.available:
        raise LLMUnavailableError(
            "GROQ_API_KEY is not set on the MCP server process.",
            code="groq_not_configured",
        )
    try:
        probs = json.loads(probabilities_json) if probabilities_json.strip() else {}
        if not isinstance(probs, dict):
            probs = {}
        probs_f = {str(k): float(v) for k, v in probs.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        probs_f = {}
    return await llm.explain_classification(
        text_excerpt=text,
        label=label,
        probabilities=probs_f or None,
    )
