"""Optional Groq chat completions with model fallback (rate-limit resilience)."""

from __future__ import annotations

import structlog
from groq import APIStatusError, AsyncGroq, GroqError, RateLimitError

from ai_vs_human_text.config import get_groq_settings
from ai_vs_human_text.exceptions import LLMProviderError, LLMUnavailableError

logger = structlog.get_logger(__name__)


class GroqCompletionService:
    def __init__(self) -> None:
        g = get_groq_settings()
        self._api_key = g.api_key
        self._models = [g.model_primary, g.model_fallback]
        self._client: AsyncGroq | None = (
            AsyncGroq(api_key=self._api_key) if self._api_key else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self._client:
            raise LLMUnavailableError(
                "GROQ_API_KEY is not set; LLM explanations are disabled.",
                code="groq_not_configured",
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for model in self._models:
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = resp.choices[0].message.content
                if not choice:
                    raise LLMProviderError("Empty completion from Groq", code="groq_empty")
                logger.info("groq_completion", model=model)
                return choice.strip()
            except RateLimitError as e:
                last_err = e
                logger.warning("groq_rate_limit", model=model, error=str(e))
                continue
            except APIStatusError as e:
                if getattr(e, "status_code", None) == 429:
                    last_err = e
                    logger.warning("groq_429", model=model)
                    continue
                raise LLMProviderError(
                    f"Groq API error: {e}",
                    code="groq_api_error",
                    details={"status": getattr(e, "status_code", None)},
                ) from e
            except GroqError as e:
                raise LLMProviderError(str(e), code="groq_error") from e
        raise LLMProviderError(
            "All Groq models exhausted (rate limits or errors).",
            code="groq_models_exhausted",
            details={"last_error": str(last_err) if last_err else None},
        ) from last_err

    async def explain_classification(
        self,
        *,
        text_excerpt: str,
        label: str,
        probabilities: dict[str, float] | None,
    ) -> str:
        excerpt = text_excerpt[:4000] if len(text_excerpt) > 4000 else text_excerpt
        prob_str = str(probabilities) if probabilities else "n/a"
        user = (
            f"Classifier label: {label}\n"
            f"Class probabilities: {prob_str}\n"
            f"Text (excerpt):\n{excerpt}\n\n"
            "Give a short, factual explanation of why this might be classified this way "
            "(stylistic cues, repetition, formality). Do not claim certainty; the model is statistical."
        )
        system = (
            "You assist analysts reviewing AI vs human text detection. "
            "Be concise (3-6 sentences). No markdown title."
        )
        return await self.complete(system=system, user=user)
