"""ASGI middleware for MCP directory scanner compatibility.

Handles two issues that prevent directories like Smithery from
discovering server capabilities:

1. **Non-standard methods** — Smithery sends proprietary JSON-RPC
   methods (e.g. ``ai.smithery/events/list``).  FastMCP's Pydantic
   validation rejects these.  This middleware returns the correct
   ``-32601 Method Not Found`` error so the scanner can continue.

2. **Out-of-order requests** — Smithery sends ``tools/list`` without
   a prior ``initialize``, which causes FastMCP to return empty tool
   lists in stateless mode.  This middleware detects the pattern and
   returns tool metadata directly from the FastMCP instance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.requests import Request
from starlette.responses import JSONResponse

import structlog

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

log = structlog.get_logger()

# Standard MCP 2024-11-05 methods that FastMCP handles.
_KNOWN_MCP_METHODS = frozenset(
    {
        "initialize",
        "ping",
        "notifications/initialized",
        "notifications/cancelled",
        "notifications/roots/list_changed",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
        "resources/templates/list",
        "resources/subscribe",
        "resources/unsubscribe",
        "prompts/list",
        "prompts/get",
        "logging/setLevel",
        "completion/complete",
        "sampling/createMessage",
        "tasks/cancel",
    }
)

# Methods that scanners often send without initialize
_DISCOVERY_METHODS = frozenset(
    {"tools/list", "resources/list", "prompts/list", "resources/templates/list"}
)


class UnknownMethodGuard:
    """ASGI middleware for MCP directory scanner compatibility.

    Sits in front of FastMCP's Streamable HTTP handler and:

    1. Returns proper ``-32601`` errors for non-standard methods.
    2. Serves tool/resource/prompt metadata directly when discovery
       methods arrive without a prior ``initialize`` (common with
       directory scanners like Smithery).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        mcp_path: str = "/mcp",
        fastmcp: FastMCP | None = None,
    ) -> None:
        self._app = app
        self._mcp_path = mcp_path
        self._fastmcp = fastmcp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)

        if request.method != "POST" or request.url.path != self._mcp_path:
            await self._app(scope, receive, send)
            return

        # Read the body and check the JSON-RPC method
        body = await request.body()
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not valid JSON — let FastMCP handle the error
            await self._forward_with_body(scope, receive, send, body)
            return

        # Handle single requests only (batches pass through)
        if isinstance(data, list):
            await self._forward_with_body(scope, receive, send, body)
            return

        method = data.get("method", "")
        msg_id = data.get("id")

        # ── Non-standard method → -32601 ──────────────────────────
        if method not in _KNOWN_MCP_METHODS:
            log.debug(
                "unknown_mcp_method_rejected",
                method=method,
                id=msg_id,
            )
            error_response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
            response = JSONResponse(error_response)
            await response(scope, receive, send)
            return

        # ── Discovery method without session → serve directly ─────
        # Directory scanners (Smithery) send tools/list without
        # initialize.  FastMCP returns empty in that case.  If we
        # have a FastMCP reference, serve the metadata directly.
        if (
            self._fastmcp is not None
            and method in _DISCOVERY_METHODS
            and not request.headers.get("mcp-session-id")
        ):
            result = await self._handle_discovery(method, msg_id)
            if result is not None:
                response = JSONResponse(result)
                await response(scope, receive, send)
                return

        # ── Standard MCP method → pass through ────────────────────
        await self._forward_with_body(scope, receive, send, body)

    async def _handle_discovery(
        self, method: str, msg_id: int | str | None
    ) -> dict | None:
        """Return tool/resource/prompt metadata directly from FastMCP."""
        assert self._fastmcp is not None
        try:
            if method == "tools/list":
                tools = await self._fastmcp.list_tools()
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [t.model_dump(mode="json", exclude_none=True) for t in tools]
                    },
                }
            elif method == "resources/list":
                resources = await self._fastmcp.list_resources()
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "resources": [
                            r.model_dump(mode="json", exclude_none=True) for r in resources
                        ]
                    },
                }
            elif method == "prompts/list":
                prompts = await self._fastmcp.list_prompts()
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "prompts": [
                            p.model_dump(mode="json", exclude_none=True) for p in prompts
                        ]
                    },
                }
            elif method == "resources/templates/list":
                templates = await self._fastmcp.list_resource_templates()
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "resourceTemplates": [
                            t.model_dump(mode="json", exclude_none=True) for t in templates
                        ]
                    },
                }
        except Exception:
            log.exception("discovery_handler_error", method=method)
        return None

    async def _forward_with_body(
        self, scope: Scope, receive: Receive, send: Send, body: bytes
    ) -> None:
        """Forward the request to the downstream app with the body intact.

        Since we already consumed ``receive``, we replay the body via a
        synthetic ASGI receive callable.
        """
        body_sent = False

        async def replay_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # After body is sent, wait for disconnect
            return await receive()

        await self._app(scope, replay_receive, send)
