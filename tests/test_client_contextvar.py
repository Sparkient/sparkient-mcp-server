"""Tests for per-request client initialisation via contextvars.

Verifies that the stateless HTTP mode (Cloud Run) correctly creates
and caches clients per API key, and that stdio mode still works via
the module-level singleton fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

import sparkient_mcp.client as client_mod
from sparkient_mcp.client import (
    SparkientClient,
    _request_client,
    get_client,
    get_or_create_client,
    init_client,
    set_request_client,
)


@pytest.fixture(autouse=True)
def _clean_client_state():
    """Reset all client state between tests."""
    _request_client.set(None)
    original_client = client_mod._client
    original_cache = client_mod._client_cache.copy()
    client_mod._client = None
    client_mod._client_cache.clear()
    yield
    _request_client.set(None)
    client_mod._client = original_client
    client_mod._client_cache = original_cache


@pytest.mark.asyncio
async def test_http_errors_preserve_sparkient_code_and_message() -> None:
    """MCP callers receive actionable errors, not a stringified JSON blob."""
    client = SparkientClient("https://api.example.com", "test-key")
    request = httpx.Request("POST", "https://api.example.com/api/v1/decide")
    response = httpx.Response(
        428,
        request=request,
        json={
            "error": {
                "code": "model_not_deployed",
                "message": "Train and deploy a model before making decisions.",
            }
        },
    )
    client._http.request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await client.decide("new_type", {"text": "hello"})

    assert result == {
        "error": "Train and deploy a model before making decisions.",
        "code": "model_not_deployed",
        "message": "Train and deploy a model before making decisions.",
        "status": 428,
    }
    await client.close()


@pytest.mark.asyncio
async def test_cancel_training_posts_to_policy_scoped_endpoint() -> None:
    client = SparkientClient("https://api.example.com", "test-key")
    client._request = AsyncMock(  # type: ignore[method-assign]
        return_value={"id": "policy-1", "status": "cancelled"}
    )

    result = await client.cancel_training("dt-1", "policy-1")

    client._request.assert_awaited_once_with(
        "POST",
        "/decision-types/dt-1/policies/policy-1/cancel",
    )
    assert result["status"] == "cancelled"
    await client.close()


@pytest.mark.asyncio
async def test_structured_training_error_preserves_retry_guidance() -> None:
    client = SparkientClient("https://api.example.com", "test-key")
    request = httpx.Request("POST", "https://api.example.com/api/v1/train")
    response = httpx.Response(
        413,
        request=request,
        json={
            "detail": {
                "code": "training_dataset_example_limit_exceeded",
                "message": "The selected snapshot is too large.",
                "retryable": False,
                "recommended_action": "Select fewer examples.",
                "dataset": {"example_count": 1200, "max_examples": 1000},
            }
        },
    )
    client._http.request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await client.train_model("dt-1")

    assert result["code"] == "training_dataset_example_limit_exceeded"
    assert result["retryable"] is False
    assert result["recommended_action"] == "Select fewer examples."
    assert result["dataset"]["max_examples"] == 1000
    await client.close()


@pytest.mark.asyncio
async def test_example_capacity_error_preserves_exact_counts() -> None:
    client = SparkientClient("https://api.example.com", "test-key")
    request = httpx.Request("POST", "https://api.example.com/api/v1/examples")
    response = httpx.Response(
        409,
        request=request,
        json={
            "error": {
                "code": "example_storage_limit_exceeded",
                "message": "The requested examples exceed storage capacity.",
                "details": {
                    "current_examples": 4990,
                    "requested_examples": 20,
                    "max_examples": 5000,
                    "remaining_examples": 10,
                    "retryable": False,
                    "recommended_action": "Add no more than 10 examples.",
                },
            }
        },
    )
    client._http.request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await client.add_examples("dt-1", [{"input": {"value": 1}}])

    assert result["code"] == "example_storage_limit_exceeded"
    assert result["current_examples"] == 4990
    assert result["requested_examples"] == 20
    assert result["max_examples"] == 5000
    assert result["remaining_examples"] == 10
    assert result["retryable"] is False
    assert result["recommended_action"] == "Add no more than 10 examples."
    await client.close()


@pytest.mark.asyncio
async def test_retry_training_posts_to_same_policy() -> None:
    client = SparkientClient("https://api.example.com", "test-key")
    client._request = AsyncMock(return_value={"policy_id": "policy-1", "status": "training"})  # type: ignore[method-assign]

    result = await client.retry_training("dt-1", "policy-1")

    client._request.assert_awaited_once_with(
        "POST",
        "/decision-types/dt-1/policies/policy-1/retry",
    )
    assert result["policy_id"] == "policy-1"
    await client.close()


class TestGetClientResolution:
    """get_client() should check context var first, then module singleton."""

    def test_raises_when_neither_set(self) -> None:
        """No context var, no singleton → RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialised"):
            get_client()

    def test_returns_module_singleton(self) -> None:
        """When only the module singleton is set, get_client() returns it."""
        mock = AsyncMock(spec=SparkientClient)
        client_mod._client = mock
        assert get_client() is mock

    def test_returns_context_var_over_singleton(self) -> None:
        """Context var takes priority over the module singleton."""
        singleton = AsyncMock(spec=SparkientClient)
        ctx_client = AsyncMock(spec=SparkientClient)
        client_mod._client = singleton
        set_request_client(ctx_client)
        assert get_client() is ctx_client

    def test_context_var_alone(self) -> None:
        """Context var works without a module singleton."""
        ctx_client = AsyncMock(spec=SparkientClient)
        set_request_client(ctx_client)
        assert get_client() is ctx_client


class TestGetOrCreateClient:
    """get_or_create_client() should cache by API key and set context var."""

    def test_creates_new_client(self) -> None:
        key = "a" * 64
        client = get_or_create_client(key)
        assert isinstance(client, SparkientClient)
        assert key in client_mod._client_cache
        assert _request_client.get(None) is client

    def test_reuses_cached_client(self) -> None:
        key = "b" * 64
        first = get_or_create_client(key)
        second = get_or_create_client(key)
        assert first is second

    def test_different_keys_different_clients(self) -> None:
        key1 = "c" * 64
        key2 = "d" * 64
        client1 = get_or_create_client(key1)
        client2 = get_or_create_client(key2)
        assert client1 is not client2

    def test_sets_context_var(self) -> None:
        key = "e" * 64
        client = get_or_create_client(key)
        assert _request_client.get(None) is client

    def test_uses_sparkient_api_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPARKIENT_API_URL", "https://custom.api.example.com")
        key = "f" * 64
        client = get_or_create_client(key)
        assert client._base_url == "https://custom.api.example.com"


class TestSetRequestClient:
    """set_request_client() should set the context var directly."""

    def test_sets_client(self) -> None:
        mock = AsyncMock(spec=SparkientClient)
        set_request_client(mock)
        assert _request_client.get(None) is mock


class TestInitClientStillWorks:
    """init_client() for stdio mode should still work as before."""

    def test_init_client_sets_singleton(self) -> None:
        key = "1" * 64
        client = init_client(key)
        assert isinstance(client, SparkientClient)
        assert client_mod._client is client

    def test_init_client_get_client_roundtrip(self) -> None:
        """init_client + get_client works when no context var is set."""
        key = "2" * 64
        created = init_client(key)
        retrieved = get_client()
        assert retrieved is created


class TestDecisionPayloads:
    """The MCP client translates tool inputs to the REST API schema."""

    @pytest.mark.asyncio
    async def test_single_decision_uses_input_and_correlation_header(self) -> None:
        client = SparkientClient("https://api.example.com", "a" * 64)
        client._request = AsyncMock(return_value={"decision": "allow"})  # type: ignore[method-assign]

        await client.decide(
            "moderation",
            {"text": "hello"},
            request_id="req-123",
        )

        client._request.assert_awaited_once_with(  # type: ignore[attr-defined]
            "POST",
            "/decide",
            json={
                "decision_type": "moderation",
                "input": {"text": "hello"},
            },
            headers={"X-Request-ID": "req-123"},
        )
        await client._http.aclose()

    @pytest.mark.asyncio
    async def test_create_type_defaults_to_classifier_only_policy(self) -> None:
        client = SparkientClient("https://api.example.com", "c" * 64)
        client._request = AsyncMock(return_value={"id": "new"})  # type: ignore[method-assign]

        await client.create_decision_type(
            "moderation",
            "Moderate content",
            ["allow", "block"],
        )

        client._request.assert_awaited_once_with(  # type: ignore[attr-defined]
            "POST",
            "/decision-types",
            json={
                "name": "moderation",
                "description": "Moderate content",
                "options": ["allow", "block"],
                "confidence_thresholds": {
                    "escalate_below": 0.7,
                    "per_option": {},
                },
                "escalation_policy": {"enabled": False},
            },
        )
        await client._http.aclose()

    @pytest.mark.asyncio
    async def test_batch_translates_input_data_alias(self) -> None:
        client = SparkientClient("https://api.example.com", "b" * 64)
        client._request = AsyncMock(  # type: ignore[method-assign]
            return_value={"results": [{"decision": "allow", "confidence": 0.9}]}
        )

        result = await client.batch_decide(
            [
                {
                    "decision_type": "moderation",
                    "input_data": {"text": "hello"},
                    "latency_budget_ms": 40,
                }
            ]
        )

        client._request.assert_awaited_once_with(  # type: ignore[attr-defined]
            "POST",
            "/decide/batch",
            json=[
                {
                    "decision_type": "moderation",
                    "input": {"text": "hello"},
                    "latency_budget_ms": 40,
                }
            ],
        )
        assert result == {
            "results": [{"decision": "allow", "confidence": 0.9}],
            "errors": [],
            "succeeded": 1,
            "failed": 0,
        }
        await client._http.aclose()

    @pytest.mark.asyncio
    async def test_batch_surfaces_partial_errors_by_position(self) -> None:
        client = SparkientClient("https://api.example.com", "c" * 64)
        client._request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "results": [
                    {"decision": "allow", "confidence": 0.9},
                    None,
                ],
                "errors": [
                    {
                        "index": 1,
                        "decision_type": "moderation",
                        "code": "internal_error",
                        "message": "An internal error occurred while processing this item.",
                        "status_code": 500,
                        "retryable": True,
                        "request_id": "item-2",
                        "internal_detail": "must not be exposed",
                    }
                ],
            }
        )

        result = await client.batch_decide(
            [
                {"decision_type": "moderation", "input": {"text": "one"}},
                {"decision_type": "moderation", "input": {"text": "two"}},
            ]
        )

        assert result["results"] == [
            {"decision": "allow", "confidence": 0.9},
            None,
        ]
        assert result["succeeded"] == 1
        assert result["failed"] == 1
        assert result["errors"] == [
            {
                "index": 1,
                "decision_type": "moderation",
                "code": "internal_error",
                "message": "An internal error occurred while processing this item.",
                "status_code": 500,
                "retryable": True,
                "request_id": "item-2",
            }
        ]
        assert "internal_detail" not in result["errors"][0]
        await client._http.aclose()

    @pytest.mark.asyncio
    async def test_batch_converts_legacy_processing_error_to_null(self) -> None:
        client = SparkientClient("https://api.example.com", "d" * 64)
        client._request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "results": [
                    {
                        "decision": "allow",
                        "confidence": 0.0,
                        "reason_codes": ["processing_error"],
                        "fallback_used": True,
                        "request_id": "legacy-1",
                    }
                ]
            }
        )

        result = await client.batch_decide(
            [
                {"decision_type": "moderation", "input": {"text": "hello"}},
            ]
        )

        assert result["results"] == [None]
        assert result["succeeded"] == 0
        assert result["failed"] == 1
        assert result["errors"][0]["index"] == 0
        assert result["errors"][0]["code"] == "processing_error"
        assert result["errors"][0]["request_id"] == "legacy-1"
        await client._http.aclose()
