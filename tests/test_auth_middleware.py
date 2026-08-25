"""Tests for AuthMiddleware — API key extraction and per-request client setup.

Verifies that AuthMiddleware correctly:
- Extracts API keys from Authorization headers
- Creates per-request clients via contextvars
- Rejects unauthenticated MCP requests with -32000
- Allows non-MCP paths (health, server card) through without auth
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import sparkient_mcp.client as client_mod
from sparkient_mcp.client import _request_client, get_client
from sparkient_mcp.middleware import AuthMiddleware


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset client state between tests."""
    _request_client.set(None)
    original = client_mod._client
    original_cache = client_mod._client_cache.copy()
    client_mod._client = None
    client_mod._client_cache.clear()
    yield
    _request_client.set(None)
    client_mod._client = original
    client_mod._client_cache = original_cache


def _make_app() -> Starlette:
    """Create a minimal Starlette app with AuthMiddleware for testing."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def mcp_endpoint(request: Request) -> JSONResponse:
        """Simulates an MCP endpoint that uses get_client()."""
        try:
            client = get_client()
            return JSONResponse({"client_type": type(client).__name__})
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/mcp", mcp_endpoint, methods=["POST", "GET"]),
        ],
    )
    app.add_middleware(AuthMiddleware)
    return app


_VALID_KEY = "a" * 64


class TestAuthMiddlewareRejectsUnauthed:
    """POST /mcp without a valid API key should be rejected."""

    def test_no_auth_header(self) -> None:
        client = TestClient(_make_app())
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        resp = client.post("/mcp", json=body)
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == -32000
        assert "Authentication required" in data["error"]["message"]

    def test_invalid_auth_header(self) -> None:
        client = TestClient(_make_app())
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        resp = client.post(
            "/mcp",
            json=body,
            headers={"Authorization": "Bearer not-valid"},
        )
        assert resp.status_code == 401

    def test_wrong_scheme(self) -> None:
        client = TestClient(_make_app())
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        resp = client.post(
            "/mcp",
            json=body,
            headers={"Authorization": f"Basic {_VALID_KEY}"},
        )
        assert resp.status_code == 401


class TestAuthMiddlewareAllowsAuthed:
    """POST /mcp with a valid API key should pass through."""

    def test_valid_bearer_passes_through(self) -> None:
        client = TestClient(_make_app())
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        resp = client.post(
            "/mcp",
            json=body,
            headers={"Authorization": f"Bearer {_VALID_KEY}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["client_type"] == "SparkientClient"


class TestAuthMiddlewarePassesNonMcp:
    """Non-MCP paths should pass through without auth."""

    def test_health_no_auth(self) -> None:
        client = TestClient(_make_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_get_mcp_no_auth(self) -> None:
        """GET /mcp (SSE stream) should not be rejected by auth middleware."""
        client = TestClient(_make_app(), raise_server_exceptions=False)
        resp = client.get("/mcp")
        # Auth middleware only enforces on POST /mcp — GET passes through.
        # The endpoint may error for other reasons (no client set), but
        # it should NOT be a 401 auth rejection.
        assert resp.status_code != 401


class TestAuthMiddlewareCachesClients:
    """Same API key should reuse the cached client."""

    def test_same_key_reuses_client(self) -> None:
        app = _make_app()
        client = TestClient(app)
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        headers = {"Authorization": f"Bearer {_VALID_KEY}"}

        client.post("/mcp", json=body, headers=headers)
        assert _VALID_KEY in client_mod._client_cache

        client.post("/mcp", json=body, headers=headers)
        # Still only one client in the cache
        assert len(client_mod._client_cache) == 1

    def test_different_keys_different_clients(self) -> None:
        key2 = "b" * 64
        app = _make_app()
        client = TestClient(app)
        body = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        client.post("/mcp", json=body, headers={"Authorization": f"Bearer {_VALID_KEY}"})
        client.post("/mcp", json=body, headers={"Authorization": f"Bearer {key2}"})
        assert len(client_mod._client_cache) == 2
