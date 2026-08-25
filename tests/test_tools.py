"""Tests for individual MCP tool functions."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

import sparkient_mcp.client as client_mod
from sparkient_mcp.client import _request_client, set_request_client
from sparkient_mcp.server import mcp
from sparkient_mcp.tools.decide import batch_decisions, make_decision
from sparkient_mcp.tools.decision_types import (
    RuleDefinition,
    create_decision_type,
    get_decision_type,
    list_decision_types,
)
from sparkient_mcp.tools.examples import add_examples, generate_examples
from sparkient_mcp.tools.introspect import (
    get_decision_logs,
    get_edge_export_instructions,
    get_metrics,
)
from sparkient_mcp.tools.training import cancel_training, retry_training, train_model


@pytest.fixture(autouse=True)
def _mock_client():
    """Inject a mocked SparkientClient for every test.

    Sets the per-request context variable (simulating AuthMiddleware)
    and also patches the module singleton for backward compatibility.
    """
    mock = AsyncMock()
    mock.decide.return_value = {
        "decision": "approve",
        "confidence": 0.95,
        "stage": "classifier",
        "latency_ms": 8,
    }
    mock.batch_decide.return_value = {"results": []}
    mock.list_decision_types.return_value = {"items": [], "total": 0}
    mock.get_decision_type.return_value = {"id": "abc", "name": "test"}
    mock.create_decision_type.return_value = {"id": "new-id", "name": "new"}
    mock.add_examples.return_value = [{"id": "example-1", "expected_decision": "yes"}]
    mock.generate_examples.return_value = [{"id": "generated-1", "expected_decision": "yes"}]
    mock.train_model.return_value = {"status": "accepted", "job_id": "j-1"}
    mock.cancel_training.return_value = {"status": "cancelled", "id": "policy-1"}
    mock.retry_training.return_value = {
        "status": "training",
        "policy_id": "policy-1",
        "dataset": {"manifest_id": "sha256:test"},
    }
    mock.get_decision_logs.return_value = {"items": [], "total": 0}
    mock.get_metrics.return_value = {"total_decisions": 100}
    mock.get_edge_export_instructions.return_value = {
        "decision_type_id": "abc",
        "transfers_bundle": False,
        "download": {
            "method": "GET",
            "url": "https://api.sparkient.ai/api/v1/decision-types/abc/export",
        },
    }

    # Set via context variable (how stateless HTTP mode works)
    set_request_client(mock)
    with patch.object(client_mod, "_client", mock):
        yield mock
    # Reset the context variable after each test
    _request_client.set(None)


# ── Decide tools ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_make_decision_calls_client(_mock_client: AsyncMock) -> None:
    result = await make_decision("moderation", {"text": "hello"})
    _mock_client.decide.assert_awaited_once_with("moderation", {"text": "hello"}, None)
    assert result["decision"] == "approve"


@pytest.mark.asyncio
async def test_make_decision_with_request_id(_mock_client: AsyncMock) -> None:
    await make_decision("moderation", {"text": "hi"}, request_id="req-1")
    _mock_client.decide.assert_awaited_once_with("moderation", {"text": "hi"}, "req-1")


@pytest.mark.asyncio
async def test_batch_decisions(_mock_client: AsyncMock) -> None:
    items = [{"decision_type": "mod", "input_data": {"text": "a"}}]
    result = await batch_decisions(items)
    _mock_client.batch_decide.assert_awaited_once_with(items)
    assert "results" in result


@pytest.mark.asyncio
async def test_batch_decisions_surfaces_null_with_matching_error(
    _mock_client: AsyncMock,
) -> None:
    """The tool must expose an item failure, not present null as a decision."""
    partial = {
        "results": [{"decision": "allow"}, None],
        "errors": [
            {
                "index": 1,
                "decision_type": "mod",
                "code": "internal_error",
                "message": "An internal error occurred while processing this item.",
                "status_code": 500,
                "retryable": True,
                "request_id": "item-2",
            }
        ],
        "succeeded": 1,
        "failed": 1,
    }
    _mock_client.batch_decide.return_value = partial
    items = [
        {"decision_type": "mod", "input": {"text": "a"}},
        {"decision_type": "mod", "input": {"text": "b"}},
    ]

    result = await batch_decisions(items)

    assert result == partial
    assert result["results"][1] is None
    assert result["errors"][0]["index"] == 1


# ── Decision type tools ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_decision_types(_mock_client: AsyncMock) -> None:
    result = await list_decision_types(page=2, page_size=10)
    _mock_client.list_decision_types.assert_awaited_once_with(2, 10, None)
    assert "items" in result


@pytest.mark.asyncio
async def test_list_decision_types_forwards_search(_mock_client: AsyncMock) -> None:
    await list_decision_types(search="risk routing")

    _mock_client.list_decision_types.assert_awaited_once_with(
        1,
        20,
        "risk routing",
    )


@pytest.mark.asyncio
async def test_get_decision_type(_mock_client: AsyncMock) -> None:
    result = await get_decision_type("abc")
    _mock_client.get_decision_type.assert_awaited_once_with("abc")
    assert result["id"] == "abc"


@pytest.mark.asyncio
async def test_create_decision_type(_mock_client: AsyncMock) -> None:
    result = await create_decision_type("test", "A test type", ["yes", "no"])
    _mock_client.create_decision_type.assert_awaited_once_with(
        "test", "A test type", ["yes", "no"], None, None, False, 0.7, None, None
    )
    assert result["id"] == "new-id"


@pytest.mark.asyncio
async def test_create_decision_type_can_enable_escalation(
    _mock_client: AsyncMock,
) -> None:
    await create_decision_type(
        "test",
        "A test type",
        ["yes", "no"],
        escalation_enabled=True,
        escalate_below=0.4,
        per_option_thresholds={"no": 0.9},
    )

    _mock_client.create_decision_type.assert_awaited_once_with(
        "test",
        "A test type",
        ["yes", "no"],
        None,
        None,
        True,
        0.4,
        {"no": 0.9},
        None,
    )


@pytest.mark.asyncio
async def test_create_decision_type_sends_structured_rules_and_input_schema(
    _mock_client: AsyncMock,
) -> None:
    rule = RuleDefinition(
        name="block_large",
        condition="ctx.amount > 1000",
        then="no",
        reason_code="large_amount",
        priority=2,
    )
    input_schema = {
        "type": "object",
        "properties": {"amount": {"type": "number"}},
        "required": ["amount"],
    }

    await create_decision_type(
        "test",
        "A test type",
        ["yes", "no"],
        reason_codes=["large_amount"],
        rules=[rule],
        input_schema=input_schema,
    )

    _mock_client.create_decision_type.assert_awaited_once_with(
        "test",
        "A test type",
        ["yes", "no"],
        ["large_amount"],
        [rule.model_dump()],
        False,
        0.7,
        None,
        input_schema,
    )


@pytest.mark.asyncio
async def test_fastmcp_validates_and_converts_rule_objects(
    _mock_client: AsyncMock,
) -> None:
    rule = {
        "name": "block_large",
        "condition": "ctx.amount > 1000",
        "then": "no",
        "reason_code": "large_amount",
        "priority": 2,
    }

    await mcp._tool_manager.call_tool(
        "create_decision_type",
        {
            "name": "test",
            "description": "A test type",
            "options": ["yes", "no"],
            "reason_codes": ["large_amount"],
            "rules": [rule],
        },
    )

    _mock_client.create_decision_type.assert_awaited_once_with(
        "test",
        "A test type",
        ["yes", "no"],
        ["large_amount"],
        [rule],
        False,
        0.7,
        None,
        None,
    )


# ── Example tools ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_examples(_mock_client: AsyncMock) -> None:
    examples = [{"input_payload": {"x": 1}, "expected_decision": "yes"}]
    result = await add_examples("abc", examples)
    _mock_client.add_examples.assert_awaited_once_with("abc", examples)
    assert result == [{"id": "example-1", "expected_decision": "yes"}]


@pytest.mark.asyncio
async def test_generate_examples(_mock_client: AsyncMock) -> None:
    result = await generate_examples("abc", count=20)
    _mock_client.generate_examples.assert_awaited_once_with("abc", 20)
    assert result == [{"id": "generated-1", "expected_decision": "yes"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    [
        (
            "add_examples",
            {
                "decision_type_id": "abc",
                "examples": [
                    {
                        "input_payload": {"x": 1},
                        "expected_decision": "yes",
                    }
                ],
            },
            [{"id": "example-1", "expected_decision": "yes"}],
        ),
        (
            "generate_examples",
            {"decision_type_id": "abc", "count": 10},
            [{"id": "generated-1", "expected_decision": "yes"}],
        ),
    ],
)
async def test_fastmcp_validates_rest_shaped_example_arrays(
    _mock_client: AsyncMock,
    tool_name: str,
    arguments: dict,
    expected: list[dict],
) -> None:
    """Real REST arrays must survive FastMCP structured-output validation."""
    _content, structured = await mcp._tool_manager.call_tool(
        tool_name,
        arguments,
        convert_result=True,
    )

    assert structured == {"result": expected}


def test_make_decision_metadata_reports_side_effects_and_distinct_flags() -> None:
    tool = mcp._tool_manager.get_tool("make_decision")

    assert tool is not None
    assert tool.annotations.idempotentHint is False
    assert "human review" in tool.description
    assert "llm_escalated" in tool.description
    assert "fallback_used" in tool.description


def test_create_decision_type_schema_exposes_structured_rule_contract() -> None:
    tool = mcp._tool_manager.get_tool("create_decision_type")

    assert tool is not None
    rule_schema = tool.parameters["$defs"]["RuleDefinition"]
    assert set(rule_schema["required"]) == {
        "name",
        "condition",
        "then",
        "reason_code",
    }
    assert rule_schema["properties"]["priority"]["minimum"] == 0


# ── Training tool ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_train_model(_mock_client: AsyncMock) -> None:
    result = await train_model("abc", auto_deploy=False)
    _mock_client.train_model.assert_awaited_once_with("abc", False)
    assert result["status"] == "accepted"


def test_train_model_does_not_expose_training_presets() -> None:
    assert "preset" not in inspect.signature(train_model).parameters


@pytest.mark.asyncio
async def test_cancel_training_targets_exact_policy(
    _mock_client: AsyncMock,
) -> None:
    result = await cancel_training("dt-1", "policy-1")

    _mock_client.cancel_training.assert_awaited_once_with("dt-1", "policy-1")
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_retry_training_preserves_policy_and_snapshot(
    _mock_client: AsyncMock,
) -> None:
    result = await retry_training("dt-1", "policy-1")

    _mock_client.retry_training.assert_awaited_once_with("dt-1", "policy-1")
    assert result["policy_id"] == "policy-1"
    assert result["dataset"]["manifest_id"] == "sha256:test"


# ── Introspection tools ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_decision_logs(_mock_client: AsyncMock) -> None:
    result = await get_decision_logs("abc", page=3, page_size=5)
    _mock_client.get_decision_logs.assert_awaited_once_with("abc", 3, 5)
    assert "items" in result


@pytest.mark.asyncio
async def test_get_metrics(_mock_client: AsyncMock) -> None:
    result = await get_metrics()
    _mock_client.get_metrics.assert_awaited_once()
    assert result["total_decisions"] == 100


@pytest.mark.asyncio
async def test_get_edge_export_instructions_does_not_claim_transfer(
    _mock_client: AsyncMock,
) -> None:
    result = await get_edge_export_instructions("abc")

    _mock_client.get_edge_export_instructions.assert_awaited_once_with("abc")
    assert result["transfers_bundle"] is False
    assert result["download"]["method"] == "GET"
    assert "bundle_base64" not in result
