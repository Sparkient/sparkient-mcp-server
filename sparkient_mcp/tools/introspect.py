"""Introspection tools — logs, metrics, credits, and edge export."""

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
        Field(description="Page number (1-indexed)."),
    ] = 1,
    page_size: Annotated[
        int,
        Field(description="Results per page (max 100)."),
    ] = 20,
) -> dict[str, Any]:
    """Query past decision logs for a decision type.

    Returns a paginated list of decisions including input data,
    decision, confidence, stage, latency, and timestamp.

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
    """Get org-level aggregate metrics.

    Returns summary statistics for your organisation including
    total decisions, decisions today, average latency, and
    per-decision-type breakdowns.
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

    Returns how many credits remain this billing period,
    the total allocation, percentage used, plan tier,
    and when credits reset.

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
        openWorldHint=True,
    ),
)
async def export_edge_bundle(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type to export."),
    ],
) -> dict[str, Any]:
    """Export a trained model as a standalone edge bundle (ZIP).

    Downloads the active deployed model, feature config, expression rules,
    and metadata as a self-contained ZIP file. After download, the bundle
    can run without calling Sparkient's cloud API using the sparkient-edge SDK;
    compatible local package and runtime dependencies still apply:

        from sparkient_edge import EdgePredictor
        predictor = EdgePredictor.from_bundle("bundle.zip")
        result = predictor.predict({"text": "hello"})

    Requires a deployed model (call train_model first).
    Requires Growth plan or above.
    """
    client = get_client()
    return await client.export_edge_bundle(decision_type_id)
