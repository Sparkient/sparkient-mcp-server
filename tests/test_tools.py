"""Tests for individual MCP tool functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import sparkient_mcp.client as client_mod
from sparkient_mcp.tools.decide import make_decision, batch_decisions
from sparkient_mcp.tools.decision_types import (
    list_decision_types,
    get_decision_type,
    create_decision_type,
)
from sparkient_mcp.tools.examples import add_examples, generate_examples
from sparkient_mcp.tools.training import train_model
from sparkient_mcp.tools.introspect import get_decision_logs, get_metrics


@pytest.fixture(autouse=True)
def _mock_client():
    """Inject a mocked SparkientClient for every test."""
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
    mock.add_examples.return_value = {"added": 5}
    mock.generate_examples.return_value = {"generated": 10}
    mock.train_model.return_value = {"status": "accepted", "job_id": "j-1"}
    mock.get_decision_logs.return_value = {"items": [], "total": 0}
    mock.get_metrics.return_value = {"total_decisions": 100}

    with patch.object(client_mod, "_client", mock):
        yield mock


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


# ── Decision type tools ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_decision_types(_mock_client: AsyncMock) -> None:
    result = await list_decision_types(page=2, page_size=10)
    _mock_client.list_decision_types.assert_awaited_once_with(2, 10, None)
    assert "items" in result


@pytest.mark.asyncio
async def test_get_decision_type(_mock_client: AsyncMock) -> None:
    result = await get_decision_type("abc")
    _mock_client.get_decision_type.assert_awaited_once_with("abc")
    assert result["id"] == "abc"


@pytest.mark.asyncio
async def test_create_decision_type(_mock_client: AsyncMock) -> None:
    result = await create_decision_type("test", "A test type", ["yes", "no"])
    _mock_client.create_decision_type.assert_awaited_once_with(
        "test", "A test type", ["yes", "no"], None, None
    )
    assert result["id"] == "new-id"


# ── Example tools ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_examples(_mock_client: AsyncMock) -> None:
    examples = [{"input_payload": {"x": 1}, "expected_decision": "yes"}]
    result = await add_examples("abc", examples)
    _mock_client.add_examples.assert_awaited_once_with("abc", examples)
    assert result["added"] == 5


@pytest.mark.asyncio
async def test_generate_examples(_mock_client: AsyncMock) -> None:
    result = await generate_examples("abc", count=20)
    _mock_client.generate_examples.assert_awaited_once_with("abc", 20)
    assert result["generated"] == 10


# ── Training tool ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_train_model(_mock_client: AsyncMock) -> None:
    result = await train_model("abc", preset="fast", auto_deploy=False)
    _mock_client.train_model.assert_awaited_once_with("abc", "fast", False)
    assert result["status"] == "accepted"


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
