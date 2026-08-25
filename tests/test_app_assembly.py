"""Integration tests for the assembled MCP ASGI middleware stack."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from sparkient_mcp.__main__ import _build_app


def _rpc(method: str, *, params: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }


@pytest.fixture(scope="module")
def assembled_client():
    """Run the singleton FastMCP session manager once for this test module."""
    with TestClient(_build_app()) as client:
        yield client


def test_public_tools_discovery_bypasses_auth(assembled_client: TestClient) -> None:
    response = assembled_client.post("/mcp", json=_rpc("tools/list"))

    assert response.status_code == 200
    tool_names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert "make_decision" in tool_names
    assert "export_edge_bundle" in tool_names


def test_public_resource_discovery_bypasses_auth(
    assembled_client: TestClient,
) -> None:
    response = assembled_client.post("/mcp", json=_rpc("resources/list"))

    assert response.status_code == 200
    resource_uris = {
        resource["uri"] for resource in response.json()["result"]["resources"]
    }
    assert "sparkient://decision-types" in resource_uris


def test_unauthenticated_tool_call_is_still_rejected(
    assembled_client: TestClient,
) -> None:
    params = {
        "name": "make_decision",
        "arguments": {
            "decision_type": "content_moderation",
            "input_data": {"text": "example"},
        },
    }
    response = assembled_client.post("/mcp", json=_rpc("tools/call", params=params))

    assert response.status_code == 401
    assert response.json()["error"]["code"] == -32000


def test_unauthenticated_normal_mcp_initialization_is_still_rejected(
    assembled_client: TestClient,
) -> None:
    params = {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0.0"},
    }
    response = assembled_client.post("/mcp", json=_rpc("initialize", params=params))

    assert response.status_code == 401
    assert response.json()["error"]["code"] == -32000
