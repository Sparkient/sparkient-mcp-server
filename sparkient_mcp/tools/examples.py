"""Training example tools."""

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
) -> list[dict[str, Any]] | dict[str, Any]:
    """Add labelled training examples to a decision type.

    More representative examples can improve model accuracy. Training requires
    at least 38 labelled examples per option, with balanced class distribution.
    A decision type can store
    up to 5,000 examples; capacity errors report the exact remaining space.
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
) -> list[dict[str, Any]] | dict[str, Any]:
    """Generate synthetic training examples using AI.

    Uses the decision type's description and options to generate
    realistic labelled examples. Good for bootstrapping a new
    decision type before you have real data. Generation is all-or-nothing:
    Sparkient returns the remaining capacity instead of silently creating
    fewer examples than requested.
    """
    client = get_client()
    return await client.generate_examples(decision_type_id, count)
