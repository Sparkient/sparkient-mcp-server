"""API key extraction and validation helpers.

The MCP Streamable HTTP transport runs on Starlette/ASGI.  API keys arrive
via the standard ``Authorization: Bearer YOUR_API_KEY`` header.  These helpers
keep auth logic out of the tool layer.
"""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

# Sparkient API keys are 64-character hexadecimal tokens.
_API_KEY_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


def extract_api_key(headers: dict[str, str]) -> str | None:
    """Extract the API key from an ``Authorization: Bearer YOUR_API_KEY`` header.

    Returns ``None`` if the header is missing, malformed, or does not
    contain a valid-looking key.
    """
    auth_value = headers.get("authorization") or headers.get("Authorization")
    if not auth_value:
        return None

    match = _BEARER_RE.match(auth_value)
    if not match:
        log.debug("auth_header_not_bearer", value=auth_value[:20])
        return None

    token = match.group(1).strip()
    if not validate_api_key_format(token):
        log.debug("auth_key_invalid_format", prefix=token[:5])
        return None

    return token


def validate_api_key_format(key: str) -> bool:
    """Return ``True`` if *key* looks like a valid Sparkient API key.

    Checks that the token contains exactly 64 hexadecimal characters, matching
    the format produced by the Sparkient API.
    """
    return _API_KEY_RE.fullmatch(key) is not None
