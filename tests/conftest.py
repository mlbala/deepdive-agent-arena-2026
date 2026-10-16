"""Shared pytest fixtures. The suite is hermetic: no network, no `.env` leakage."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.core.config import Settings


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip real credentials from the environment so tests never hit live endpoints."""
    for var in ("MESH_API_KEY", "TAVILY_API_KEY", "MESH_BASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Build a Settings object that ignores `.env` and applies explicit overrides."""

    def _factory(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "environment": "test",
            "database_url": "sqlite:///:memory:",
            "search_mode": "offline",
            "vector_backend": "tfidf",
        }
        base.update(overrides)
        return Settings(_env_file=None, **base)  # type: ignore[call-arg]

    return _factory
