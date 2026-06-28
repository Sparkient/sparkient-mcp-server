"""Decision type management tools."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from sparkient_mcp.server import mcp
from sparkient_mcp.client import get_client


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

    Returns a paginated list of decision types with their name,
    description, status, and option labels.
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

    Returns the complete decision type including name, description,
    options, reason codes, rules, input schema, training status,
    active model version, and confidence thresholds.
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
        list[dict[str, Any]] | None,
        Field(description="Optional CEL rules for deterministic decisions."),
    ] = None,
) -> dict[str, Any]:
    """Create a new decision type.

    A decision type defines a category of decisions your system makes.
    For example, 'content_moderation' with options ['approve', 'reject', 'escalate'].
    """
    client = get_client()
    return await client.create_decision_type(
        name, description, options, reason_codes, rules
    )
