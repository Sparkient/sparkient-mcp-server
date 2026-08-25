"""Package entrypoint: ``python -m sparkient_mcp``.

This avoids the double-import problem that occurs when running
``python -m sparkient_mcp.server`` directly.  That command makes
``server.py`` run as ``__main__``, but tool modules import it as
``sparkient_mcp.server``, creating *two* separate FastMCP instances.
The tools register on the imported one while ``__main__`` uses its
own empty copy.

By running through ``__main__.py``, the server module is always
imported as ``sparkient_mcp.server`` — one instance, one set of tools.
"""

from __future__ import annotations

import anyio
import structlog
import uvicorn

from sparkient_mcp.middleware import AuthMiddleware, UnknownMethodGuard
from sparkient_mcp.server import mcp

log = structlog.get_logger()

_port = mcp.settings.port


def _build_app():
    """Build the ASGI app with public discovery outside authenticated calls.

    Starlette applies middleware in last-added, first-run order.  The
    discovery guard therefore has to be added after the auth middleware so
    that it can answer unauthenticated schema-only requests before auth runs.
    All other MCP requests continue through to ``AuthMiddleware``.
    """
    starlette_app = mcp.streamable_http_app()
    starlette_app.add_middleware(AuthMiddleware)
    starlette_app.add_middleware(UnknownMethodGuard, fastmcp=mcp)
    return starlette_app


async def _serve() -> None:
    starlette_app = _build_app()
    # Wrap with middleware for directory scanner compatibility:
    # 1. Returns -32601 for non-standard methods (ai.smithery/*)
    # 2. Serves tool metadata directly when scanners skip initialize

    log.info(
        "sparkient_mcp_starting",
        transport="streamable-http",
        port=_port,
    )

    config = uvicorn.Config(
        starlette_app,
        host="0.0.0.0",
        port=_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    anyio.run(_serve)
