"""Tests for the Sparkient MCP server initialisation and tool registration."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from sparkient_mcp.server import (
    _SERVER_CARD_ETAG,
    _SERVER_NAME,
    _SERVER_VERSION,
    _server_card_response,
    mcp,
    server_card,
    server_card_alt,
    server_card_transport,
)


EXPECTED_TOOLS = {
    "make_decision",
    "batch_decisions",
    "list_decision_types",
    "get_decision_type",
    "create_decision_type",
    "add_examples",
    "generate_examples",
    "train_model",
    "get_training_status",
    "cancel_training",
    "retry_training",
    "get_decision_logs",
    "get_metrics",
    "get_credits",
    "export_edge_bundle",
}


class TestServerInstance:
    """Verify the FastMCP instance is set up correctly."""

    def test_server_name(self) -> None:
        assert mcp.name == _SERVER_NAME

    def test_runtime_identity_matches_server_card(self) -> None:
        options = mcp._mcp_server.create_initialization_options()

        assert options.server_name == _SERVER_NAME
        assert options.server_version == _SERVER_VERSION

    def test_runtime_version_matches_package_metadata(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

        assert _SERVER_VERSION == project["version"]

    def test_tools_registered(self) -> None:
        """The documented 15-tool inventory must match registration exactly."""
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert tool_names == EXPECTED_TOOLS

    def test_minimum_tool_count(self) -> None:
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [server_card, server_card_alt, server_card_transport],
)
async def test_server_card_routes_return_advertised_media_type(handler) -> None:
    response = await handler(None)

    assert response.media_type == "application/mcp-server-card+json"
    body = json.loads(response.body)
    assert body["$schema"].endswith("/schemas/v1/server-card.schema.json")
    assert body["name"] == "ai.sparkient/sparkient"
    assert body["title"] == "Sparkient"
    assert body["remotes"][0]["url"] == "https://mcp.sparkient.ai/mcp"
    assert body["remotes"][0]["headers"][0]["name"] == "Authorization"
    assert body["remotes"][0]["headers"][0]["isSecret"] is True
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.headers["etag"] == _SERVER_CARD_ETAG
    assert "repository" not in body


def test_server_card_honours_matching_etag() -> None:
    request = type(
        "CardRequest",
        (),
        {"headers": {"if-none-match": _SERVER_CARD_ETAG}},
    )()

    response = _server_card_response(request)

    assert response.status_code == 304
    assert response.body == b""
    assert response.headers["etag"] == _SERVER_CARD_ETAG
