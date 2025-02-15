"""Console entrypoints: REST API (Uvicorn) and MCP (stdio)."""

from __future__ import annotations


def run_api() -> None:
    import uvicorn

    from ai_vs_human_text.config import get_settings
    from ai_vs_human_text.logging_config import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "ai_vs_human_text.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


def run_mcp() -> None:
    from ai_vs_human_text.logging_config import configure_logging
    from ai_vs_human_text.mcp_server import mcp

    configure_logging("INFO")
    mcp.run()
