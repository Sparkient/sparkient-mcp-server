"""Async HTTP client wrapping the Sparkient REST API.

All methods return plain dicts — either the API response payload or a
structured error dict ``{"error": "...", "status": <int>}``.  This keeps
the MCP tool layer free of exception handling.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class SparkientClient:
    """Thin async wrapper around the Sparkient Decision Intelligence API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=f"{self._base_url}/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[dict[str, Any]] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a request and return parsed JSON or a structured error."""
        try:
            response = await self._http.request(
                method, path, json=json, params=params,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            try:
                detail = exc.response.json().get("detail", body)
            except Exception:
                detail = body
            log.warning(
                "sparkient_api_error",
                status=exc.response.status_code,
                detail=detail,
                path=path,
            )
            return {"error": str(detail), "status": exc.response.status_code}
        except httpx.RequestError as exc:
            log.error("sparkient_api_request_error", error=str(exc), path=path)
            return {"error": f"Connection error: {exc}", "status": 502}

    # ------------------------------------------------------------------
    # Decide
    # ------------------------------------------------------------------

    async def decide(
        self,
        decision_type: str,
        input_data: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Make a single decision via the 3-stage pipeline."""
        payload: dict[str, Any] = {
            "decision_type": decision_type,
            "input_data": input_data,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        return await self._request("POST", "/decide", json=payload)

    async def batch_decide(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        """Make up to 50 decisions in a single batch call."""
        return await self._request("POST", "/decide/batch", json=decisions)

    # ------------------------------------------------------------------
    # Decision types
    # ------------------------------------------------------------------

    async def list_decision_types(
        self,
        page: int = 1,
        page_size: int = 20,
        search: str | None = None,
    ) -> dict[str, Any]:
        """List decision types with optional search and pagination."""
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if search is not None:
            params["search"] = search
        return await self._request("GET", "/decision-types", params=params)

    async def get_decision_type(self, decision_type_id: str) -> dict[str, Any]:
        """Get full configuration of a specific decision type."""
        return await self._request("GET", f"/decision-types/{decision_type_id}")

    async def create_decision_type(
        self,
        name: str,
        description: str,
        options: list[str],
        reason_codes: list[str] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a new decision type."""
        payload: dict[str, Any] = {
            "name": name,
            "description": description,
            "options": options,
        }
        if reason_codes is not None:
            payload["reason_codes"] = reason_codes
        if rules is not None:
            payload["rules"] = rules
        return await self._request("POST", "/decision-types", json=payload)

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------

    async def add_examples(
        self,
        decision_type_id: str,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add labelled training examples to a decision type."""
        return await self._request(
            "POST",
            f"/decision-types/{decision_type_id}/examples",
            json={"examples": examples},
        )

    async def generate_examples(
        self,
        decision_type_id: str,
        count: int = 10,
    ) -> dict[str, Any]:
        """Generate synthetic examples using Gemini."""
        return await self._request(
            "POST",
            f"/decision-types/{decision_type_id}/examples/generate",
            json={"count": count},
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def train_model(
        self,
        decision_type_id: str,
        preset: str = "balanced",
        auto_deploy: bool = True,
    ) -> dict[str, Any]:
        """Trigger async model training for a decision type."""
        return await self._request(
            "POST",
            f"/decision-types/{decision_type_id}/train",
            json={"preset": preset, "auto_deploy": auto_deploy},
        )

    async def get_training_progress(
        self,
        decision_type_id: str,
        policy_id: str,
    ) -> dict[str, Any]:
        """Get real-time training progress for a policy."""
        return await self._request(
            "GET",
            f"/decision-types/{decision_type_id}/policies/{policy_id}/progress",
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def get_decision_logs(
        self,
        decision_type_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Query decision logs for a decision type."""
        return await self._request(
            "GET",
            f"/decision-types/{decision_type_id}/logs",
            params={"page": page, "page_size": page_size},
        )

    async def get_metrics(self) -> dict[str, Any]:
        """Get org-level aggregate metrics."""
        return await self._request("GET", "/metrics")

    async def get_credits(self) -> dict[str, Any]:
        """Get the org's current credit balance."""
        return await self._request("GET", "/credits")

    async def export_edge_bundle(
        self,
        decision_type_id: str,
    ) -> dict[str, Any]:
        """Export the active model as a standalone edge bundle (ZIP).

        Returns base64-encoded ZIP bytes and a suggested filename,
        or a structured error dict.
        """
        import base64

        try:
            response = await self._http.request(
                "GET",
                f"/decision-types/{decision_type_id}/export",
            )
            response.raise_for_status()
            bundle_b64 = base64.b64encode(response.content).decode("ascii")
            # Extract filename from Content-Disposition header
            cd = response.headers.get("content-disposition", "")
            filename = "edge_bundle.zip"
            if 'filename="' in cd:
                filename = cd.split('filename="')[1].rstrip('"')
            return {
                "filename": filename,
                "size_bytes": len(response.content),
                "bundle_base64": bundle_b64,
            }
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            try:
                detail = exc.response.json().get("detail", body)
            except Exception:
                detail = body
            log.warning(
                "sparkient_api_error",
                status=exc.response.status_code,
                detail=detail,
                path=f"/decision-types/{decision_type_id}/export",
            )
            return {"error": str(detail), "status": exc.response.status_code}
        except httpx.RequestError as exc:
            log.error(
                "sparkient_api_request_error",
                error=str(exc),
                path=f"/decision-types/{decision_type_id}/export",
            )
            return {"error": f"Connection error: {exc}", "status": 502}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()


# ======================================================================
# Module-level singleton
# ======================================================================

_client: SparkientClient | None = None


def init_client(api_key: str) -> SparkientClient:
    """Create (or replace) the module-level SparkientClient singleton."""
    global _client  # noqa: PLW0603
    base_url = os.environ.get("SPARKIENT_API_URL", "https://api.sparkient.ai")
    _client = SparkientClient(base_url=base_url, api_key=api_key)
    log.info("sparkient_client_initialized", base_url=base_url)
    return _client


def get_client() -> SparkientClient:
    """Return the active SparkientClient, raising if not yet initialised."""
    if _client is None:
        raise RuntimeError(
            "SparkientClient not initialised — call init_client(api_key) first."
        )
    return _client
