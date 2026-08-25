"""Async HTTP client wrapping the Sparkient REST API.

Methods return the REST API's JSON shape. Most endpoints return objects;
example-creation endpoints return arrays. Failures are normalized to a
structured error object ``{"error": "...", "status": <int>}``. This keeps
the MCP tool layer free of exception handling without misrepresenting arrays.
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _is_legacy_processing_error(result: object) -> bool:
    """Recognise the old REST API's fabricated batch-error placeholder."""
    if not isinstance(result, dict):
        return False
    reason_codes = result.get("reason_codes")
    return (
        isinstance(reason_codes, list)
        and "processing_error" in reason_codes
        and result.get("confidence") == 0.0
    )


def _normalise_batch_error(
    raw_error: object,
    *,
    index: int,
    decision_type: str,
    default_code: str = "batch_item_failed",
    default_message: str = "This batch item did not produce a decision.",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Keep only the documented, safe fields of a per-item API error."""
    error = raw_error if isinstance(raw_error, dict) else {}

    raw_status = error.get("status_code")
    status_code = (
        raw_status
        if isinstance(raw_status, int)
        and not isinstance(raw_status, bool)
        and 400 <= raw_status <= 599
        else 500
    )
    raw_retryable = error.get("retryable")
    retryable = (
        raw_retryable
        if isinstance(raw_retryable, bool)
        else status_code == 429 or status_code >= 500
    )
    raw_code = error.get("code")
    code = raw_code if isinstance(raw_code, str) and raw_code else default_code
    raw_message = error.get("message")
    message = raw_message if isinstance(raw_message, str) and raw_message else default_message
    raw_decision_type = error.get("decision_type")
    safe_decision_type = (
        raw_decision_type
        if isinstance(raw_decision_type, str) and raw_decision_type
        else decision_type
    )
    raw_request_id = error.get("request_id", request_id)

    return {
        "index": index,
        "decision_type": safe_decision_type,
        "code": code,
        "message": message,
        "status_code": status_code,
        "retryable": retryable,
        "request_id": raw_request_id if isinstance(raw_request_id, str) else None,
    }


def _normalise_batch_response(
    response: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose partial failures explicitly and reject decision placeholders.

    Whole-request error responses are returned unchanged.  A response with a
    ``results`` member is normalized to one positional entry per request;
    every null or invalid entry has a matching error.
    """
    if "results" not in response:
        return response

    raw_results = response.get("results")
    expected_count = len(decisions)
    if not isinstance(raw_results, list):
        raw_results = []

    results: list[dict[str, Any] | None | object] = list(raw_results[:expected_count])
    if len(results) < expected_count:
        results.extend([None] * (expected_count - len(results)))

    errors_by_index: dict[int, dict[str, Any]] = {}
    raw_errors = response.get("errors", [])
    if isinstance(raw_errors, list):
        for raw_error in raw_errors:
            if not isinstance(raw_error, dict):
                continue
            raw_index = raw_error.get("index")
            if (
                not isinstance(raw_index, int)
                or isinstance(raw_index, bool)
                or not 0 <= raw_index < expected_count
            ):
                continue
            raw_decision_type = decisions[raw_index].get("decision_type")
            decision_type = raw_decision_type if isinstance(raw_decision_type, str) else ""
            errors_by_index[raw_index] = _normalise_batch_error(
                raw_error,
                index=raw_index,
                decision_type=decision_type,
            )
            # The error is authoritative.  Never expose a result at the same
            # position as though it were a usable business decision.
            results[raw_index] = None

    for index, result in enumerate(results):
        if isinstance(result, dict) and not _is_legacy_processing_error(result):
            continue

        results[index] = None
        if index in errors_by_index:
            continue

        raw_decision_type = decisions[index].get("decision_type")
        decision_type = raw_decision_type if isinstance(raw_decision_type, str) else ""
        request_id = result.get("request_id") if isinstance(result, dict) else None
        if _is_legacy_processing_error(result):
            code = "processing_error"
            message = "This batch item failed during processing."
        elif result is None:
            code = "batch_item_failed"
            message = "This batch item did not produce a decision."
        else:
            code = "upstream_response_invalid"
            message = "Sparkient returned an invalid result for this batch item."
        errors_by_index[index] = _normalise_batch_error(
            None,
            index=index,
            decision_type=decision_type,
            default_code=code,
            default_message=message,
            request_id=request_id if isinstance(request_id, str) else None,
        )

    normalized = dict(response)
    normalized["results"] = results
    normalized["errors"] = [errors_by_index[index] for index in sorted(errors_by_index)]
    normalized["succeeded"] = sum(result is not None for result in results)
    normalized["failed"] = len(errors_by_index)
    return normalized


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

    @staticmethod
    def _error_response(response: httpx.Response) -> dict[str, Any]:
        """Normalize Sparkient and FastAPI error envelopes for MCP callers."""
        status = response.status_code
        code = "http_error"
        message = response.text or f"Sparkient API returned HTTP {status}"
        try:
            payload = response.json()
        except (TypeError, ValueError):
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                raw_code = error.get("code")
                raw_message = error.get("message")
                if isinstance(raw_code, str) and raw_code:
                    code = raw_code
                if isinstance(raw_message, str) and raw_message:
                    message = raw_message
            elif isinstance(payload.get("detail"), str):
                message = payload["detail"]
            elif isinstance(payload.get("detail"), dict):
                detail = payload["detail"]
                raw_code = detail.get("code")
                raw_message = detail.get("message")
                if isinstance(raw_code, str) and raw_code:
                    code = raw_code
                if isinstance(raw_message, str) and raw_message:
                    message = raw_message
            elif payload.get("detail") is not None:
                message = str(payload["detail"])

        result = {
            "error": message,
            "code": code,
            "message": message,
            "status": status,
        }
        detail = None
        if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
            detail = payload["detail"]
        elif isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error_details = payload["error"].get("details")
            if isinstance(error_details, dict):
                detail = error_details
        if detail is not None:
            for key in (
                "retryable",
                "recommended_action",
                "policy_id",
                "policy",
                "dataset",
                "unavailable_example_ids",
                "current_examples",
                "requested_examples",
                "max_examples",
                "remaining_examples",
            ):
                if key in detail:
                    result[key] = detail[key]
        return result

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
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Issue a request and return parsed JSON or a structured error."""
        try:
            response = await self._http.request(
                method,
                path,
                json=json,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            error = self._error_response(exc.response)
            log.warning(
                "sparkient_api_error",
                status=error["status"],
                code=error["code"],
                detail=error["message"],
                path=path,
            )
            return error
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
            "input": input_data,
        }
        headers = {"X-Request-ID": request_id} if request_id is not None else None
        return await self._request("POST", "/decide", json=payload, headers=headers)

    async def batch_decide(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        """Make up to 50 decisions, preserving positions and item errors."""
        canonical: list[dict[str, Any]] = []
        for item in decisions:
            input_payload = item.get("input", item.get("input_data"))
            request: dict[str, Any] = {
                "decision_type": item.get("decision_type"),
                "input": input_payload,
            }
            if "latency_budget_ms" in item:
                request["latency_budget_ms"] = item["latency_budget_ms"]
            canonical.append(request)
        response = await self._request("POST", "/decide/batch", json=canonical)
        return _normalise_batch_response(response, canonical)

    # ------------------------------------------------------------------
    # Decision types
    # ------------------------------------------------------------------

    async def list_decision_types(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """List decision types with pagination."""
        params: dict[str, Any] = {"page": page, "page_size": page_size}
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
        escalation_enabled: bool = False,
        escalate_below: float = 0.7,
        per_option_thresholds: dict[str, float] | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new decision type."""
        payload: dict[str, Any] = {
            "name": name,
            "description": description,
            "options": options,
            "confidence_thresholds": {
                "escalate_below": escalate_below,
                "per_option": per_option_thresholds or {},
            },
            "escalation_policy": {"enabled": escalation_enabled},
        }
        if reason_codes is not None:
            payload["reason_codes"] = reason_codes
        if rules is not None:
            payload["rules"] = rules
        if input_schema is not None:
            payload["input_schema"] = input_schema
        return await self._request("POST", "/decision-types", json=payload)

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------

    async def add_examples(
        self,
        decision_type_id: str,
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | dict[str, Any]:
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
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Generate synthetic examples using the LLM teacher."""
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
        auto_deploy: bool = True,
    ) -> dict[str, Any]:
        """Trigger async model training for a decision type."""
        return await self._request(
            "POST",
            f"/decision-types/{decision_type_id}/train",
            json={"auto_deploy": auto_deploy},
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

    async def cancel_training(
        self,
        decision_type_id: str,
        policy_id: str,
    ) -> dict[str, Any]:
        """Request safe cancellation of one policy's active training run."""
        return await self._request(
            "POST",
            f"/decision-types/{decision_type_id}/policies/{policy_id}/cancel",
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

    async def get_edge_export_instructions(
        self,
        decision_type_id: str,
    ) -> dict[str, Any]:
        """Describe the authenticated ways to download an edge bundle.

        Edge bundles can contain hundreds of megabytes of model assets, so they
        are deliberately not transferred inside an MCP JSON response.
        """
        return {
            "decision_type_id": decision_type_id,
            "transfers_bundle": False,
            "download": {
                "method": "GET",
                "url": (f"{self._base_url}/api/v1/decision-types/{decision_type_id}/export"),
                "authentication": (
                    "Send an Authorization header with Bearer followed by a "
                    "Sparkient API key. The endpoint streams the ZIP response."
                ),
            },
            "dashboard": {
                "url": (f"https://app.sparkient.ai/decision-types/{decision_type_id}"),
                "action": "Open the decision type and choose Export.",
            },
            "eligibility": ("Growth or Scale plan and an active deployed policy."),
            "note": (
                "This MCP tool provides download instructions only; it does not "
                "download, encode, or transfer the bundle."
            ),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()


# ======================================================================
# Per-request client via contextvars (stateless HTTP mode)
# ======================================================================

# In stateless HTTP mode (Cloud Run), each request is independent and
# there is no init_client() call.  Instead, AuthMiddleware extracts the
# API key from the Authorization header and creates (or reuses) a
# SparkientClient for that key, storing it in this context variable.
_request_client: contextvars.ContextVar[SparkientClient | None] = contextvars.ContextVar(
    "sparkient_request_client",
    default=None,
)

# Cache of clients keyed by API key.  Avoids creating a new httpx
# connection pool for every single request when the same key is reused.
_client_cache: dict[str, SparkientClient] = {}


def get_or_create_client(api_key: str) -> SparkientClient:
    """Return a cached SparkientClient for *api_key*, creating one if needed.

    The returned client is also set on the per-request context variable so
    that ``get_client()`` can find it during tool dispatch.
    """
    client = _client_cache.get(api_key)
    if client is None:
        base_url = os.environ.get("SPARKIENT_API_URL", "https://api.sparkient.ai")
        client = SparkientClient(base_url=base_url, api_key=api_key)
        _client_cache[api_key] = client
        log.info("sparkient_client_created_for_key", base_url=base_url)
    _request_client.set(client)
    return client


def set_request_client(client: SparkientClient) -> None:
    """Explicitly set the per-request client (used by AuthMiddleware)."""
    _request_client.set(client)


# ======================================================================
# Module-level singleton (stdio / local edge mode)
# ======================================================================

_client: SparkientClient | None = None


def init_client(api_key: str) -> SparkientClient:
    """Create (or replace) the module-level SparkientClient singleton."""
    global _client
    base_url = os.environ.get("SPARKIENT_API_URL", "https://api.sparkient.ai")
    _client = SparkientClient(base_url=base_url, api_key=api_key)
    log.info("sparkient_client_initialized", base_url=base_url)
    return _client


def get_client() -> SparkientClient:
    """Return the active SparkientClient.

    Resolution order:
    1. Per-request context variable (set by AuthMiddleware in stateless HTTP mode)
    2. Module-level singleton (set by init_client() in stdio mode)

    Raises ``RuntimeError`` if neither is available.
    """
    # 1. Check per-request context (stateless HTTP mode)
    ctx_client = _request_client.get(None)
    if ctx_client is not None:
        return ctx_client

    # 2. Fall back to module singleton (stdio / local edge mode)
    if _client is not None:
        return _client

    raise RuntimeError(
        "SparkientClient not initialised — either pass an Authorization header "
        "(stateless HTTP mode) or call init_client(api_key) first (stdio mode)."
    )
