"""Mesh API gateway tests: configuration contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import MESH_DEFAULT_BASE_URL, Settings


def test_mesh_base_url_defaults_to_mesh_gateway(make_settings):
    settings = make_settings()
    assert settings.mesh_base_url == MESH_DEFAULT_BASE_URL == "https://api.meshapi.ai/v1"
    assert settings.mesh_configured is False


def test_mesh_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("MESH_API_KEY", "rsk_test_key")
    monkeypatch.setenv("MESH_BASE_URL", "https://mesh.example.test/v1/")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.mesh_configured
    assert settings.mesh_api_key is not None
    assert settings.mesh_api_key.get_secret_value() == "rsk_test_key"
    assert settings.mesh_base_url == "https://mesh.example.test/v1"


def test_secrets_never_leak_into_public_summary(make_settings):
    settings = make_settings(mesh_api_key="rsk_super_secret", tavily_api_key="tvly-secret")
    summary = repr(settings.public_summary()) + repr(settings)
    assert "rsk_super_secret" not in summary
    assert "tvly-secret" not in summary
    assert settings.public_summary()["tavily_configured"] is True


def test_blank_keys_are_treated_as_missing(make_settings):
    settings = make_settings(mesh_api_key="   ", tavily_api_key="")
    assert settings.mesh_configured is False
    assert settings.tavily_configured is False


def test_per_agent_model_override(make_settings):
    settings = make_settings(verifier_model="openai/gpt-4o", planner_model="")
    assert settings.model_for("verifier") == "openai/gpt-4o"
    assert settings.model_for("planner") == settings.mesh_default_model


def test_chunk_overlap_must_be_smaller_than_chunk(make_settings):
    with pytest.raises(ValidationError):
        make_settings(chunk_size=300, chunk_overlap=300)
