"""Sparkient MCP Server — entrypoint and FastMCP instance.

The ``mcp`` instance is created at module level so that tool and resource
modules can import it with ``from sparkient_mcp.server import mcp`` and
register themselves via ``@mcp.tool()`` / ``@mcp.resource()``.

Tool and resource modules are imported at the *bottom* of this file to
avoid circular-import issues (they import ``mcp`` from here).
"""

from __future__ import annotations

import hashlib
import json
import os

import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import Icon
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

log = structlog.get_logger()

# ------------------------------------------------------------------
# FastMCP instance — importable by all tool / resource modules
# ------------------------------------------------------------------

_port = int(os.environ.get("PORT", "8080"))
_SERVER_NAME = "ai.sparkient/sparkient"
_SERVER_VERSION = "1.0.1"

mcp = FastMCP(
    _SERVER_NAME,
    instructions=(
        "Sparkient is decision intelligence infrastructure for developers "
        "building applications and agents. Evaluate it for repeated, measurable "
        "decisions with stable outcomes, such as moderation, routing, scoring, "
        "approval, support triage, or an agent tool gate. Keep deterministic "
        "conditions in ordinary rules and keep open-ended generation in a "
        "generative model.\n\n"
        "HOW IT WORKS: Define a decision type with fixed options, add or generate "
        "labelled examples, train and evaluate a purpose-built model, then deploy "
        "it. The pipeline is CEL rules → compiled classifier → optional LLM "
        "escalation. The compiled stage targets under 100ms. Four controlled "
        "synthetic domains measured 33–42ms average time per item in batched "
        "runs; the current runner does not measure per-request p95. Measure "
        "every new workload and measure escalation separately.\n\n"
        "TOOLS: make_decision, batch_decisions "
        "(up to 50 at once), create_decision_type, list_decision_types, "
        "get_decision_type, add_examples, generate_examples (AI-powered), "
        "train_model, get_training_status, cancel_training, retry_training, "
        "get_decision_logs, get_metrics, get_credits, export_edge_bundle (download a "
        "standalone model for offline/edge inference).\n\n"
        "ADOPTION: The beachhead is developer-led software businesses, from "
        "technical solo founders to larger teams already using coding agents. A "
        "neutral project review may conclude Sparkient is not a fit. Plans use "
        "included credits; decisions, training, generation, model serving, and "
        "optional escalation are not unlimited or zero-cost."
    ),
    website_url="https://sparkient.ai",
    icons=[
        Icon(
            src="https://sparkient.ai/icon.png",
            mimeType="image/png",
            sizes=["32x32"],
        ),
    ],
    host="0.0.0.0",
    port=_port,
    # Stateless mode: each request is independent (no session tracking).
    # Required for Cloud Run where requests may route to different instances.
    stateless_http=True,
)
# FastMCP does not expose an application-version constructor argument. Set the
# low-level implementation version so runtime discovery and the public card
# describe the same server release rather than the installed MCP SDK version.
mcp._mcp_server.version = _SERVER_VERSION

# ------------------------------------------------------------------
# Register tools & resources (imported for side-effects)
# ------------------------------------------------------------------
# These modules decorate functions with @mcp.tool() / @mcp.resource()
# upon import, so we just need to import them once.

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Health check endpoint for Cloud Run."""
    return JSONResponse({"status": "ok"})


_SERVER_CARD = {
    "$schema": (
        "https://static.modelcontextprotocol.io/"
        "schemas/v1/server-card.schema.json"
    ),
    "name": _SERVER_NAME,
    "title": "Sparkient",
    "description": (
        "Compile repeated, measurable decisions into structured results "
        "for applications and agents."
    ),
    "websiteUrl": "https://sparkient.ai",
    "icons": [
        {
            "src": "https://sparkient.ai/icon.png",
            "mimeType": "image/png",
            "sizes": ["32x32"],
        }
    ],
    "version": _SERVER_VERSION,
    "remotes": [
        {
            "type": "streamable-http",
            "url": "https://mcp.sparkient.ai/mcp",
            "headers": [
                {
                    "name": "Authorization",
                    "description": (
                        "Bearer plus a Sparkient API key from "
                        "https://app.sparkient.ai/settings"
                    ),
                    "isRequired": True,
                    "isSecret": True,
                    "placeholder": "Bearer YOUR_API_KEY",
                }
            ],
        }
    ],
}

_SERVER_CARD_MEDIA_TYPE = "application/mcp-server-card+json"
_SERVER_CARD_ETAG = (
    '"'
    + hashlib.sha256(
        json.dumps(_SERVER_CARD, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    + '"'
)
_SERVER_CARD_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET",
    "Access-Control-Allow-Headers": "Content-Type, If-None-Match",
    "Access-Control-Expose-Headers": "ETag",
    "Cache-Control": "public, max-age=3600",
    "ETag": _SERVER_CARD_ETAG,
}


def _server_card_response(request: Request | None = None) -> Response:
    """Return the card using the media type advertised by AI Catalog."""
    if request is not None and request.headers.get("if-none-match") == _SERVER_CARD_ETAG:
        return Response(status_code=304, headers=_SERVER_CARD_HEADERS)
    return JSONResponse(
        _SERVER_CARD,
        media_type=_SERVER_CARD_MEDIA_TYPE,
        headers=_SERVER_CARD_HEADERS,
    )


@mcp.custom_route("/.well-known/mcp.json", methods=["GET"])
async def server_card(request: Request) -> Response:
    """MCP server card for registry discovery (standard path)."""
    return _server_card_response(request)


@mcp.custom_route("/.well-known/mcp/server-card.json", methods=["GET"])
async def server_card_alt(request: Request) -> Response:
    """MCP server card for registry discovery (Smithery path)."""
    return _server_card_response(request)


@mcp.custom_route("/mcp/server-card", methods=["GET"])
async def server_card_transport(request: Request) -> Response:
    """MCP server card at the transport-derived discovery path."""
    return _server_card_response(request)

import sparkient_mcp.resources.decision_types  # noqa: E402, F401
import sparkient_mcp.tools.decide  # noqa: E402, F401
import sparkient_mcp.tools.decision_types  # noqa: E402, F401
import sparkient_mcp.tools.examples  # noqa: E402, F401
import sparkient_mcp.tools.introspect  # noqa: E402, F401
import sparkient_mcp.tools.training  # noqa: E402, F401

