"""API key extraction and validation helpers.

The MCP Streamable HTTP transport runs on Starlette/ASGI.  API keys arrive
via the standard ``Authorization: Bearer sk-...`` header.  These helpers
keep auth logic out of the tool layer.
"""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

# Sparkient API keys are prefixed ``sk-`` followed by ≥32 hex/alphanum chars.
_MIN_KEY_LENGTH = 36  # "sk-" (3) + 32 random chars + 1 margin
_PREFIX = "sk-"
_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


def extract_api_key(headers: dict[str, str]) -> str | None:
    """Extract the API key from an ``Authorization: Bearer sk-...`` header.

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

    Checks:
    - starts with ``sk-``
    - minimum length of 36 characters
    """
    return key.startswith(_PREFIX) and len(key) >= _MIN_KEY_LENGTH
