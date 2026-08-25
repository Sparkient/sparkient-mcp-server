"""Decision type management tools."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from sparkient_mcp.client import get_client
from sparkient_mcp.server import mcp


class RuleDefinition(BaseModel):
    """Structured deterministic rule accepted by the Sparkient REST API."""

    name: str = Field(description="Unique rule name.")
    condition: str = Field(
        description="Boolean CEL expression evaluated against the input under `ctx`."
    )
    then: str = Field(description="Configured decision option to return when true.")
    reason_code: str = Field(description="Configured reason code attached when the rule fires.")
    priority: int = Field(
        default=10,
        ge=0,
        description="Evaluation priority; lower numbers run first.",
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def list_decision_types(
    page: Annotated[
        int,
        Field(description="Page number (1-indexed)."),
    ] = 1,
    page_size: Annotated[
        int,
        Field(description="Results per page (max 100)."),
    ] = 20,
    search: Annotated[
        str | None,
        Field(description="Optional text to filter by name or description."),
    ] = None,
) -> dict[str, Any]:
    """List all decision types in your organisation.

    Returns a paginated list of decision types and their active configuration.
    ``model_deployed`` reports whether a compatible trained policy is currently
    deployed. Search is case-insensitive across name and description.
    """
    client = get_client()
    return await client.list_decision_types(page, page_size, search)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_decision_type(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
) -> dict[str, Any]:
    """Get the full configuration of a specific decision type.

    Returns the decision type metadata, its active immutable configuration
    version (options, reason codes, rules, input schema, thresholds and optional
    LLM-escalation policy), and ``model_deployed``. It does not return live
    training progress; use the policy ID returned by ``train_model`` with
    ``get_training_status`` for that.
    """
    client = get_client()
    return await client.get_decision_type(decision_type_id)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def create_decision_type(
    name: Annotated[
        str,
        Field(description="Unique name (lowercase, underscores). E.g. 'content_moderation'."),
    ],
    description: Annotated[
        str,
        Field(description="What this decision type is for."),
    ],
    options: Annotated[
        list[str],
        Field(description="The possible decision outcomes. E.g. ['approve', 'reject']."),
    ],
    reason_codes: Annotated[
        list[str] | None,
        Field(description="Optional labels for why a decision was made."),
    ] = None,
    rules: Annotated[
        list[RuleDefinition] | None,
        Field(
            description=(
                "Optional deterministic CEL rules. Each rule requires name, a boolean "
                "condition using the `ctx` input namespace, a configured `then` option, "
                "a configured reason_code, and optional non-negative priority."
            )
        ),
    ] = None,
    escalation_enabled: Annotated[
        bool,
        Field(
            description=(
                "Allow metered live-LLM escalation to produce low-confidence decisions. "
                "Defaults to false for classifier-only operation."
            )
        ),
    ] = False,
    escalate_below: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Global confidence threshold. Below it, Sparkient invokes the live LLM "
                "when escalation_enabled is true; otherwise it returns the classifier "
                "result with escalate=true for human review."
            ),
        ),
    ] = 0.7,
    per_option_thresholds: Annotated[
        dict[str, float] | None,
        Field(description=("Optional confidence thresholds keyed by configured decision option.")),
    ] = None,
    input_schema: Annotated[
        dict[str, Any] | None,
        Field(description="Optional JSON Schema describing the accepted decision input."),
    ] = None,
) -> dict[str, Any]:
    """Create a new decision type.

    A decision type defines a repeated measurable decision and its output
    contract. Creation stores configuration only: add at least 38 labelled
    examples per option, then call ``train_model`` before ``make_decision``.
    """
    client = get_client()
    return await client.create_decision_type(
        name,
        description,
        options,
        reason_codes,
        [rule.model_dump() for rule in rules] if rules is not None else None,
        escalation_enabled,
        escalate_below,
        per_option_thresholds,
        input_schema,
    )
