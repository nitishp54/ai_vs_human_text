"""Single-turn agent: Groq tool-calling + Pydantic tool-arg repair + classifier execution."""

from __future__ import annotations

import json
from typing import Any

import structlog
from groq import AsyncGroq

from ai_vs_human_text.agent.schemas import ClassifyToolArgs, ModelInfoToolArgs
from ai_vs_human_text.agent.tool_repair import repair_and_validate
from ai_vs_human_text.config import get_groq_settings
from ai_vs_human_text.deps import get_classifier
from ai_vs_human_text.exceptions import AppError, LLMUnavailableError

logger = structlog.get_logger(__name__)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify_text",
            "description": (
                "Classify whether the given text is more likely human-written or AI-generated "
                "using the production TF-IDF + Naive Bayes pipeline."
            ),
            "parameters": ClassifyToolArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "model_info",
            "description": "Return model artifact path and offline evaluation metrics.",
            "parameters": ModelInfoToolArgs.model_json_schema(),
        },
    },
]


async def _execute_tool(name: str, arguments: str) -> dict[str, Any]:
    raw = json.loads(arguments) if arguments.strip() else {}
    clf = get_classifier()
    if name == "classify_text":
        args = repair_and_validate(ClassifyToolArgs, raw if isinstance(raw, dict) else {})
        return clf.predict(args.text)
    if name == "model_info":
        repair_and_validate(ModelInfoToolArgs, raw if isinstance(raw, dict) else {})
        return clf.model_info()
    return {"error": f"unknown_tool:{name}"}


async def run_agent_turn(user_message: str) -> dict[str, Any]:
    g = get_groq_settings()
    if not g.api_key:
        raise LLMUnavailableError(
            "Agent requires GROQ_API_KEY.",
            code="agent_requires_groq",
        )
    client = AsyncGroq(api_key=g.api_key)
    models = [g.model_primary, g.model_fallback]
    system = (
        "You are a careful assistant for AI-vs-human text analysis. "
        "When the user asks to classify text, call classify_text with the full excerpt. "
        "If they ask about accuracy or metrics, call model_info first, then answer. "
        "After tools return, summarize results clearly for a technical reader."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    last_err: Exception | None = None
    for model in models:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=2048,
            )
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            tool_results: list[dict[str, Any]] = []
            for tc in tool_calls:
                fn = tc.function
                name = fn.name
                args_raw = fn.arguments or "{}"
                try:
                    result = await _execute_tool(name, args_raw)
                    tool_results.append({"name": name, "ok": True, "result": result})
                except json.JSONDecodeError as e:
                    logger.warning("tool_args_json_invalid", tool=name, error=str(e))
                    tool_results.append({"name": name, "ok": False, "error": "invalid_json"})
                except AppError as e:
                    tool_results.append(
                        {
                            "name": name,
                            "ok": False,
                            "error": e.message,
                            "code": e.code,
                            "details": e.details,
                        }
                    )
                except Exception as e:
                    logger.exception("tool_execution_failed", tool=name)
                    tool_results.append({"name": name, "ok": False, "error": str(e)})

            if tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(assistant_msg)
                for tc, tr in zip(tool_calls, tool_results, strict=True):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tr),
                        }
                    )
                final = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                )
                summary = final.choices[0].message.content or ""
            else:
                summary = msg.content or ""

            return {
                "model": model,
                "summary": summary,
                "tool_results": tool_results if tool_calls else [],
            }
        except Exception as e:
            last_err = e
            logger.warning("agent_turn_model_failed", model=model, error=str(e))
            continue
    raise AppError(
        f"Agent failed on all models: {last_err}",
        code="agent_models_exhausted",
        details={"last_error": str(last_err) if last_err else None},
    ) from last_err
