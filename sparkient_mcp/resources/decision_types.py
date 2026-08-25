"""MCP resources for decision type discovery."""

from __future__ import annotations

import json
from typing import Any

from sparkient_mcp.client import get_client
from sparkient_mcp.server import mcp


@mcp.resource("sparkient://decision-types")
async def list_all_decision_types() -> str:
    """List all decision types in the organisation.

    Returns a JSON array of decision type summaries with name, description,
    deployment state, and active-version option labels. Agents can use this
    to discover what decisions are available.
    """
    client = get_client()
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        data: dict[str, Any] = await client.list_decision_types(
            page=page,
            page_size=100,
        )
        if "error" in data:
            return json.dumps(data, indent=2)

        page_items = data.get("items", data.get("data", []))
        if not isinstance(page_items, list):
            page_items = []
        items.extend(item for item in page_items if isinstance(item, dict))

        pages = data.get("pages")
        total = data.get("total")
        if isinstance(pages, int):
            has_next_page = page < pages
        else:
            has_next_page = (
                isinstance(total, int)
                and len(items) < total
                and bool(page_items)
            )
        if not has_next_page:
            break
        page += 1

    summaries = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "description": item.get("description"),
            "model_deployed": item.get("model_deployed", False),
            "options": (item.get("active_version") or {}).get("options", []),
        }
        for item in items
    ]
    return json.dumps(summaries, indent=2)


@mcp.resource("sparkient://decision-types/{decision_type_id}")
async def get_decision_type_by_id(decision_type_id: str) -> str:
    """Get the full configuration of a specific decision type by UUID.

    Returns the configuration including options, reason codes, rules, input
    schema, and whether a compatible model is deployed. It does not include
    live training progress.

    Args:
        decision_type_id: UUID returned by the list resource or list tool.
    """
    client = get_client()
    data = await client.get_decision_type(decision_type_id)
    return json.dumps(data, indent=2)
