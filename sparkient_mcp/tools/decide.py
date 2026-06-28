"""Decision tools — the core value of the Sparkient MCP server."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sparkient_mcp.server import mcp
from sparkient_mcp.client import get_client


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def make_decision(
    decision_type: Annotated[
        str,
        Field(description="Name of the decision type (e.g. 'content_moderation')."),
    ],
    input_data: Annotated[
        dict[str, Any],
        Field(description="Input payload matching the decision type's expected schema."),
    ],
    request_id: Annotated[
        str | None,
        Field(description="Optional idempotency key for deduplication."),
    ] = None,
) -> dict[str, Any]:
    """Make a structured decision in under 100ms.

    Requires a trained and deployed model for the given decision type.
    If no model is deployed, the API returns 428 (model_not_deployed).
    Use train_model to train and deploy a model first.

    The decision goes through a multi-stage pipeline:
      1. CEL rules — deterministic, <1ms
      2. ML classifier — ONNX model, <10ms
      3. LLM escalation — Gemini fallback, only if low confidence

    Every decision returns: decision, confidence, reason_codes,
    latency_ms, stage, and whether it was escalated.
    """
    client = get_client()
    return await client.decide(decision_type, input_data, request_id)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def batch_decisions(
    decisions: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "List of decision requests. Each item should have: "
                "decision_type (str), input_data (dict), and optionally request_id (str)."
            )
        ),
    ],
) -> dict[str, Any]:
    """Make up to 50 decisions in a single batch call.

    Each item in the list should have:
      - decision_type: str — name of the decision type
      - input_data: dict — input payload
      - request_id: str (optional) — idempotency key
    """
    client = get_client()
    return await client.batch_decide(decisions)
