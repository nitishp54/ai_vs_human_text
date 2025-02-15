"""Shared async HTTP helpers with retries (pattern aligned with resilient tool handlers)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
import structlog

from ai_vs_human_text.config import get_settings
from ai_vs_human_text.exceptions import ExternalServiceError

logger = structlog.get_logger(__name__)
T = TypeVar("T")


async def resilient_request(
    factory: Callable[[], Awaitable[httpx.Response]],
    *,
    operation: str,
    max_retries: int | None = None,
) -> httpx.Response:
    settings = get_settings()
    retries = max_retries if max_retries is not None else settings.http_max_retries
    last_exc: Exception | None = None
    delay = 0.5
    for attempt in range(retries + 1):
        try:
            resp = await factory()
            if resp.status_code >= 500 and attempt < retries:
                logger.warning(
                    "http_retry",
                    operation=operation,
                    attempt=attempt,
                    status=resp.status_code,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            return resp
        except httpx.TimeoutException as e:
            last_exc = e
            logger.warning("http_timeout", operation=operation, attempt=attempt)
        except httpx.RequestError as e:
            last_exc = e
            logger.warning("http_request_error", operation=operation, attempt=attempt, error=str(e))
        if attempt < retries:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise ExternalServiceError(
        f"{operation} failed after {retries + 1} attempts",
        code="http_exhausted_retries",
        details={"last_error": str(last_exc) if last_exc else None},
    ) from last_exc
