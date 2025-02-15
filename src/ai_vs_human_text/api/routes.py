from __future__ import annotations

from fastapi import APIRouter, Depends

from ai_vs_human_text import __version__
from ai_vs_human_text.agent.orchestrator import run_agent_turn
from ai_vs_human_text.deps import get_classifier
from ai_vs_human_text.exceptions import LLMUnavailableError
from ai_vs_human_text.ml.classifier import TextClassifierService
from ai_vs_human_text.schemas import (
    AgentRequest,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    ModelInfoResponse,
)
from ai_vs_human_text.services.llm import GroqCompletionService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(clf: TextClassifierService = Depends(get_classifier)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=clf.is_loaded,
        version=__version__,
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify(
    body: ClassifyRequest,
    clf: TextClassifierService = Depends(get_classifier),
) -> ClassifyResponse:
    out = clf.predict(body.text)
    return ClassifyResponse(**out)


@router.get("/model", response_model=ModelInfoResponse)
async def model_info(clf: TextClassifierService = Depends(get_classifier)) -> ModelInfoResponse:
    info = clf.model_info()
    return ModelInfoResponse(model_path=info["model_path"], metrics=info["metrics"])


@router.post("/explain")
async def explain(
    body: ClassifyRequest,
    clf: TextClassifierService = Depends(get_classifier),
) -> dict:
    pred = clf.predict(body.text)
    llm = GroqCompletionService()
    if not llm.available:
        raise LLMUnavailableError(
            "Set GROQ_API_KEY to enable explanations.",
            code="groq_required",
        )
    text = await llm.explain_classification(
        text_excerpt=body.text,
        label=pred["label"],
        probabilities=pred.get("probabilities"),
    )
    return {"explanation": text, "classification": pred}


@router.post("/agent")
async def agent_endpoint(body: AgentRequest) -> dict:
    return await run_agent_turn(body.message.strip())
