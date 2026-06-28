"""Tests for the Sparkient MCP server initialisation and tool registration."""

from __future__ import annotations

from sparkient_mcp.server import mcp


class TestServerInstance:
    """Verify the FastMCP instance is set up correctly."""

    def test_server_name(self) -> None:
        assert mcp.name == "Sparkient"

    def test_tools_registered(self) -> None:
        """All 13 tools should be registered."""
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        expected = {
            "make_decision",
            "batch_decisions",
            "list_decision_types",
            "get_decision_type",
            "create_decision_type",
            "add_examples",
            "generate_examples",
            "train_model",
            "get_training_status",
            "get_decision_logs",
            "get_metrics",
            "get_credits",
            "export_edge_bundle",
        }
        assert expected.issubset(tool_names), (
            f"Missing tools: {expected - tool_names}"
        )

    def test_minimum_tool_count(self) -> None:
        tools = mcp._tool_manager.list_tools()
        assert len(tools) >= 13
