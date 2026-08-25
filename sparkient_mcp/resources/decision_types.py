"""MCP resources for decision type discovery."""

from __future__ import annotations

import json
from typing import Any

from sparkient_mcp.client import get_client
from sparkient_mcp.server import mcp


@mcp.resource("sparkient://decision-types")
async def list_all_decision_types() -> str:
    """List all decision types in the organisation.

    Returns a JSON array of decision type summaries with name,
    description, status, and option labels.  Agents can use this
    to discover what decisions are available.
    """
    client = get_client()
    data: dict[str, Any] = await client.list_decision_types(page_size=100)
    if "error" in data:
        return json.dumps(data, indent=2)

    # Extract just the items for a cleaner resource
    items = data.get("items", data.get("data", []))
    summaries = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "description": item.get("description"),
            "status": item.get("status"),
            "options": item.get("options", []),
        }
        for item in items
    ]
    return json.dumps(summaries, indent=2)


@mcp.resource("sparkient://decision-types/{decision_type_id}")
async def get_decision_type_by_id(decision_type_id: str) -> str:
    """Get the full configuration of a specific decision type by UUID.

    Returns the complete schema including options, reason codes,
    rules, input schema, and training status.

    Args:
        decision_type_id: UUID returned by the list resource or list tool.
    """
    client = get_client()
    data = await client.get_decision_type(decision_type_id)
    return json.dumps(data, indent=2)
