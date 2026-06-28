"""Tests for API key auth helpers."""

from __future__ import annotations

from sparkient_mcp.auth import extract_api_key, validate_api_key_format


class TestExtractApiKey:
    """Test header parsing."""

    def test_valid_bearer_header(self) -> None:
        key = "sk-" + "a" * 40
        headers = {"Authorization": f"Bearer {key}"}
        assert extract_api_key(headers) == key

    def test_lowercase_header_name(self) -> None:
        key = "sk-" + "b" * 40
        headers = {"authorization": f"Bearer {key}"}
        assert extract_api_key(headers) == key

    def test_missing_header(self) -> None:
        assert extract_api_key({}) is None

    def test_empty_header(self) -> None:
        assert extract_api_key({"Authorization": ""}) is None

    def test_non_bearer_scheme(self) -> None:
        assert extract_api_key({"Authorization": "Basic abc123"}) is None

    def test_invalid_key_format(self) -> None:
        """Bearer token present but key doesn't start with sk-."""
        assert extract_api_key({"Authorization": "Bearer not-a-valid-key"}) is None

    def test_key_too_short(self) -> None:
        assert extract_api_key({"Authorization": "Bearer sk-short"}) is None


class TestValidateApiKeyFormat:
    """Test key format validation."""

    def test_valid_key(self) -> None:
        assert validate_api_key_format("sk-" + "x" * 40) is True

    def test_minimum_length(self) -> None:
        """sk- (3) + 33 chars = 36 minimum."""
        assert validate_api_key_format("sk-" + "a" * 33) is True

    def test_too_short(self) -> None:
        assert validate_api_key_format("sk-abc") is False

    def test_wrong_prefix(self) -> None:
        assert validate_api_key_format("pk-" + "a" * 40) is False

    def test_empty_string(self) -> None:
        assert validate_api_key_format("") is False
