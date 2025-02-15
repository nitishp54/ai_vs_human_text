from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ai_vs_human_text import __version__
from ai_vs_human_text.api.routes import router
from ai_vs_human_text.config import get_settings
from ai_vs_human_text.deps import get_classifier
from ai_vs_human_text.exceptions import AppError
from ai_vs_human_text.logging_config import configure_logging
from ai_vs_human_text.schemas import ErrorResponse

logger = structlog.get_logger(__name__)


def _serializable_validation_errors(exc: RequestValidationError) -> list[dict]:
    out: list[dict] = []
    for err in exc.errors():
        item = dict(err)
        ctx = item.get("ctx")
        if isinstance(ctx, dict):
            item["ctx"] = {k: str(v) for k, v in ctx.items()}
        out.append(item)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        get_classifier().load()
        logger.info("startup_model_warmed", path=str(settings.model_path))
    except AppError as e:
        logger.warning("startup_model_warm_failed", error=e.message, code=e.code)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI vs Human Text API",
        version=__version__,
        lifespan=lifespan,
        description="Production REST API for human vs AI text classification (ECC-derived pipeline).",
    )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        body = ErrorResponse(error=exc.message, code=exc.code, details=exc.details or None)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(exclude_none=True))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Request validation failed",
                code="request_validation_error",
                details={"errors": _serializable_validation_errors(exc)},
            ).model_dump(exclude_none=True),
        )

    app.include_router(router, prefix="/v1")
    return app
