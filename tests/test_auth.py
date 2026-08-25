"""Tests for API key auth helpers."""

from __future__ import annotations

from sparkient_mcp.auth import extract_api_key, validate_api_key_format


class TestExtractApiKey:
    """Test header parsing."""

    def test_valid_bearer_header(self) -> None:
        key = "a" * 64
        headers = {"Authorization": f"Bearer {key}"}
        assert extract_api_key(headers) == key

    def test_lowercase_header_name(self) -> None:
        key = "b" * 64
        headers = {"authorization": f"Bearer {key}"}
        assert extract_api_key(headers) == key

    def test_missing_header(self) -> None:
        assert extract_api_key({}) is None

    def test_empty_header(self) -> None:
        assert extract_api_key({"Authorization": ""}) is None

    def test_non_bearer_scheme(self) -> None:
        assert extract_api_key({"Authorization": "Basic abc123"}) is None

    def test_invalid_key_format(self) -> None:
        """Bearer token present but key is not 64-character hexadecimal."""
        assert extract_api_key({"Authorization": "Bearer not-a-valid-key"}) is None

    def test_key_too_short(self) -> None:
        assert extract_api_key({"Authorization": "Bearer abc123"}) is None


class TestValidateApiKeyFormat:
    """Test key format validation."""

    def test_valid_key(self) -> None:
        assert validate_api_key_format("0123456789abcdef" * 4) is True

    def test_uppercase_hex_is_accepted(self) -> None:
        assert validate_api_key_format("ABCDEF0123456789" * 4) is True

    def test_too_short(self) -> None:
        assert validate_api_key_format("a" * 63) is False

    def test_non_hex_characters(self) -> None:
        assert validate_api_key_format("g" * 64) is False

    def test_too_long(self) -> None:
        assert validate_api_key_format("a" * 65) is False

    def test_empty_string(self) -> None:
        assert validate_api_key_format("") is False
