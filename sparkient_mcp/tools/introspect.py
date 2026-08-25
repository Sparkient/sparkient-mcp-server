"""Introspection tools — logs, metrics, credits, and edge export guidance."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sparkient_mcp.client import get_client
from sparkient_mcp.server import mcp


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_decision_logs(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
    page: Annotated[
        int,
        Field(ge=1, description="Page number (1-indexed)."),
    ] = 1,
    page_size: Annotated[
        int,
        Field(ge=1, le=100, description="Results per page (max 100)."),
    ] = 20,
) -> dict[str, Any]:
    """Query past decision logs for a decision type.

    Returns a paginated list of decisions including input data,
    decision, confidence, latency, and timestamp.

    Useful for auditing, debugging, or finding examples for retraining.
    """
    client = get_client()
    return await client.get_decision_logs(decision_type_id, page, page_size)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_metrics() -> dict[str, Any]:
    """Get organisation-level aggregate metrics for the last 24 hours.

    Returns total decisions, compiled and escalated average latency,
    ``success_rate``, escalation rate, average confidence, decision
    distribution, active decision-type count, and the five most recent
    decisions. It does not include per-decision-type breakdowns.
    """
    client = get_client()
    return await client.get_metrics()


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_credits() -> dict[str, Any]:
    """Check your organisation's current credit balance.

    Returns how many credits remain, the current allocation, percentage used,
    plan tier, and the API's ``resets_at`` value.

    Use this before running expensive operations (batch decisions,
    training, example generation) to ensure you have enough credits.
    """
    client = get_client()
    return await client.get_credits()


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def get_edge_export_instructions(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type whose bundle you need."),
    ],
) -> dict[str, Any]:
    """Get authenticated download instructions for an edge bundle.

    This tool does not download, base64-encode, or transfer the ZIP because a
    normal text-model bundle can be hundreds of megabytes. It returns the
    protected REST endpoint, required Bearer authentication, and the real
    dashboard decision-type page. The REST endpoint streams the bundle after
    authenticating the caller.

    Export requires a Growth or Scale plan and an active deployed policy. Once
    downloaded, the bundle can run without Sparkient cloud calls through the
    ``sparkient-edge`` SDK; compatible local runtime dependencies still apply.
    """
    client = get_client()
    return await client.get_edge_export_instructions(decision_type_id)
