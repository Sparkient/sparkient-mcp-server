"""Decision tools — the core value of the Sparkient MCP server."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sparkient_mcp.client import get_client
from sparkient_mcp.server import mcp


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
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
        Field(description="Optional correlation ID sent as X-Request-ID."),
    ] = None,
) -> dict[str, Any]:
    """Make a structured decision through the Sparkient pipeline.

    Requires a trained and deployed model for the given decision type.
    If no model is deployed, the API returns 428 (model_not_deployed).
    Use train_model to train and deploy a model first.

    The decision goes through a multi-stage pipeline:
      1. CEL rules — deterministic and usually sub-millisecond
      2. Compiled classifier — under-100ms target; four public synthetic
         domains measured 33–42ms average time per item in batched runs
      3. Optional LLM escalation — only for configured low-confidence cases

    Every decision returns ``stage`` plus three distinct status flags:
    ``escalate`` means the result requires human review; ``llm_escalated``
    means the optional live-LLM stage produced the decision; and
    ``fallback_used`` means the configured non-LLM fallback produced it.

    This call consumes credits and writes a decision log, so it is not
    idempotent even when the same input is submitted again.
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
                "decision_type (str), input (dict), and optionally latency_budget_ms. "
                "The input_data alias is also accepted for compatibility. "
                "Results preserve this ordering; a failed position is null and "
                "its safe details appear in the errors list."
            ),
            min_length=1,
            max_length=50,
        ),
    ],
) -> dict[str, Any]:
    """Make up to 50 decisions in a single batch call.

    Each item in the list should have:
      - decision_type: str — name of the decision type
      - input: dict — input payload (input_data is accepted as an alias)
      - latency_budget_ms: float (optional) — skip slower stages when needed

    The response keeps one ``results`` position per request.  A null result is
    not a decision: inspect the matching ``errors`` entry by its zero-based
    index.  ``succeeded`` and ``failed`` provide an at-a-glance summary.
    """
    client = get_client()
    return await client.batch_decide(decisions)
