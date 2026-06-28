"""Training example tools."""

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
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def add_examples(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
    examples: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "List of labelled examples. Each should have: "
                "input_payload (dict) and expected_decision (str)."
            )
        ),
    ],
) -> dict[str, Any]:
    """Add labelled training examples to a decision type.

    More examples improve model accuracy. Aim for at least 50 examples
    per option, with balanced class distribution.
    """
    client = get_client()
    return await client.add_examples(decision_type_id, examples)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def generate_examples(
    decision_type_id: Annotated[
        str,
        Field(description="UUID of the decision type."),
    ],
    count: Annotated[
        int,
        Field(description="Number of examples to generate (1-50)."),
    ] = 10,
) -> dict[str, Any]:
    """Generate synthetic training examples using AI.

    Uses the decision type's description and options to generate
    realistic labelled examples. Good for bootstrapping a new
    decision type before you have real data.
    """
    client = get_client()
    return await client.generate_examples(decision_type_id, count)
