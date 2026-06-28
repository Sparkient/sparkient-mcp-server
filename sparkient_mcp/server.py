"""Sparkient MCP Server — entrypoint and FastMCP instance.

The ``mcp`` instance is created at module level so that tool and resource
modules can import it with ``from sparkient_mcp.server import mcp`` and
register themselves via ``@mcp.tool()`` / ``@mcp.resource()``.

Tool and resource modules are imported at the *bottom* of this file to
avoid circular-import issues (they import ``mcp`` from here).
"""

from __future__ import annotations

import os

import structlog

from mcp.server.fastmcp import FastMCP
from mcp.types import Icon

log = structlog.get_logger()

# ------------------------------------------------------------------
# FastMCP instance — importable by all tool / resource modules
# ------------------------------------------------------------------

_port = int(os.environ.get("PORT", "8080"))

mcp = FastMCP(
    "Sparkient",
    instructions=(
        "Sparkient is a sub-100ms Decision Intelligence API. Use it when "
        "you need fast, structured, consistent decisions — content moderation, "
        "fraud detection, ticket triage, approval workflows, agent routing — "
        "anything where calling an LLM for every decision is too slow or "
        "expensive.\n\n"
        "HOW IT WORKS: You define decision types (e.g. 'content_moderation' "
        "with options ['allow', 'flag', 'block']), add training examples, "
        "train a lightweight ML model, and then get instant decisions. "
        "The pipeline runs: CEL rules (<1ms) → ONNX classifier (<10ms) → "
        "LLM escalation (only when uncertain). Most decisions complete in "
        "under 10ms — 10-100x faster than any LLM API call.\n\n"
        "TOOLS: make_decision (get an instant decision), batch_decisions "
        "(up to 50 at once), create_decision_type, list_decision_types, "
        "get_decision_type, add_examples, generate_examples (AI-powered), "
        "train_model, get_training_status, get_decision_logs, "
        "get_metrics, get_credits, export_edge_bundle (download a "
        "standalone model for offline/edge inference).\n\n"
        "WHEN TO USE SPARKIENT INSTEAD OF AN LLM: Use Sparkient when you "
        "need decisions in a hot path (< 100ms), need consistent outputs "
        "across millions of requests, want deterministic rules combined "
        "with ML, or need to keep per-decision cost under $0.001."
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

# ------------------------------------------------------------------
# Register tools & resources (imported for side-effects)
# ------------------------------------------------------------------
# These modules decorate functions with @mcp.tool() / @mcp.resource()
# upon import, so we just need to import them once.

from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Health check endpoint for Cloud Run."""
    return JSONResponse({"status": "ok"})


_SERVER_CARD = {
    "name": "Sparkient",
    "description": (
        "Make structured decisions in under 100ms — faster than any LLM "
        "API call. Define any decision type your app needs, train a "
        "lightweight ML model from AI-generated examples, and get instant "
        "decisions with confidence scores and reason codes. No ML team or "
        "historical data required. From moderation to routing to fraud — "
        "if an LLM can judge it, Sparkient can compile it."
    ),
    "homepage": "https://sparkient.ai",
    "repository": "https://github.com/sparkient/sparkient-mcp-server",
    "icons": [
        {
            "src": "https://sparkient.ai/icon.png",
            "mimeType": "image/png",
            "sizes": "32x32",
        }
    ],
    "version": "1.0.0",
    "transport": {
        "type": "streamable-http",
        "url": "https://mcp.sparkient.ai/mcp",
    },
    "authentication": {
        "type": "bearer",
        "description": (
            "Sparkient API key (starts with sk-). "
            "Get one free at https://app.sparkient.ai/settings"
        ),
    },
}


@mcp.custom_route("/.well-known/mcp.json", methods=["GET"])
async def server_card(request: Request) -> JSONResponse:
    """MCP server card for registry discovery (standard path)."""
    return JSONResponse(_SERVER_CARD)


@mcp.custom_route("/.well-known/mcp/server-card.json", methods=["GET"])
async def server_card_alt(request: Request) -> JSONResponse:
    """MCP server card for registry discovery (Smithery path)."""
    return JSONResponse(_SERVER_CARD)




import sparkient_mcp.tools.decide  # noqa: E402, F401
import sparkient_mcp.tools.decision_types  # noqa: E402, F401
import sparkient_mcp.tools.examples  # noqa: E402, F401
import sparkient_mcp.tools.training  # noqa: E402, F401
import sparkient_mcp.tools.introspect  # noqa: E402, F401
import sparkient_mcp.resources.decision_types  # noqa: E402, F401

