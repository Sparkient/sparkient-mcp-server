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
import uvicorn

from sparkient_mcp.server import mcp
from sparkient_mcp.middleware import UnknownMethodGuard

import structlog

log = structlog.get_logger()

_port = mcp.settings.port


async def _serve() -> None:
    starlette_app = mcp.streamable_http_app()
    # Wrap with middleware for directory scanner compatibility:
    # 1. Returns -32601 for non-standard methods (ai.smithery/*)
    # 2. Serves tool metadata directly when scanners skip initialize
    starlette_app.add_middleware(UnknownMethodGuard, fastmcp=mcp)

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


anyio.run(_serve)
