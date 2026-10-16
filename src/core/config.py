"""Typed runtime configuration loaded from environment variables / `.env`.

All secrets are wrapped in `SecretStr` so they never leak into logs, reprs, or
API responses. Every LLM and embedding call in DeepDive is routed through the
Mesh API gateway configured here (`MESH_BASE_URL` + `MESH_API_KEY`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MESH_DEFAULT_BASE_URL = "https://api.meshapi.ai/v1"

AgentName = Literal["guardian", "planner", "researcher", "drafter", "verifier", "synthesizer", "followup"]


class Settings(BaseSettings):
    """Application settings. Field names map 1:1 to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -------------------------------------------------------
    app_name: str = "DeepDive"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # --- Mesh API gateway ----------------------------------------------------
    mesh_api_key: SecretStr | None = None
    mesh_base_url: str = MESH_DEFAULT_BASE_URL
    mesh_default_model: str = "openai/gpt-4o-mini"
    mesh_embedding_model: str = "openai/text-embedding-3-small"
    planner_model: str | None = None
    approver_model: str | None = None
    researcher_model: str | None = None
    drafter_model: str | None = None
    verifier_model: str | None = None
    synthesizer_model: str | None = None
    followup_model: str | None = None
    guardian_model: str | None = None

    mesh_timeout_seconds: float = Field(default=60.0, gt=0)
    mesh_max_retries: int = Field(default=4, ge=0, le=10)
    mesh_backoff_base_seconds: float = Field(default=1.0, ge=0)
    mesh_backoff_max_seconds: float = Field(default=20.0, ge=0)
    mesh_json_repair_attempts: int = Field(default=2, ge=0, le=5)
    mesh_json_mode: bool = True
    mesh_temperature: float = Field(default=0.1, ge=0, le=2)

    # --- Web retrieval -------------------------------------------------------
    tavily_api_key: SecretStr | None = None
    search_mode: Literal["auto", "tavily", "ddg", "offline"] = "auto"
    search_max_results: int = Field(default=5, ge=1, le=10)
    fetch_timeout_seconds: float = Field(default=12.0, gt=0)
    max_document_chars: int = Field(default=12_000, ge=1_000)  # per web page; evidence sits near the top

    # --- Evidence index ------------------------------------------------------
    vector_backend: Literal["hybrid", "tfidf"] = "hybrid"
    chroma_persist_dir: str | None = None
    chunk_size: int = Field(default=900, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    evidence_per_subquestion: int = Field(default=6, ge=1, le=20)

    # --- Self-verification loop ---------------------------------------------
    max_iterations: int = Field(default=3, ge=1, le=6)
    coverage_threshold: float = Field(default=0.70, ge=0, le=1)
    grounding_threshold: float = Field(default=0.75, ge=0, le=1)
    min_source_authority: float = Field(default=0.45, ge=0, le=1)  # evidence admissibility floor

    # --- Plan review gate (Planner <-> Approver) --------------------------------
    plan_approval_threshold: float = Field(default=0.80, ge=0, le=1)
    max_plan_revisions: int = Field(default=2, ge=0, le=4)

    # --- Storage & concurrency ----------------------------------------------
    database_url: str = "sqlite:///./data/deepdive.db"
    agent_max_workers: int = Field(default=4, ge=1, le=16)

    # --- Guardrails ---------------------------------------------------------
    guardrails_enabled: bool = True
    guardrail_llm_screen: bool = True  # Mesh-backed Guardian classifier after the deterministic shield
    block_prompt_injection: bool = True
    redact_pii_in_queries: bool = True
    redact_pii_in_documents: bool = True
    redact_pii_in_output: bool = True
    blocked_domains: str = ""  # comma-separated, never fetched or cited
    fetch_max_bytes: int = Field(default=4 * 1024 * 1024, ge=64 * 1024)

    # --- Governance ---------------------------------------------------------
    policy_version: str = "deepdive-governance-2026.10"
    allowed_models: str = ""  # comma-separated; empty = only the models configured above
    max_llm_calls_per_run: int = Field(default=150, ge=1)
    max_tokens_per_run: int = Field(default=2_000_000, ge=1_000)
    budget_soft_fraction: float = Field(default=0.75, gt=0, le=1)
    data_retention_days: int = Field(default=0, ge=0)  # 0 = keep forever

    # --- API security -------------------------------------------------------
    api_keys: SecretStr | None = None  # comma-separated; when set every /api route needs a key
    admin_api_key: SecretStr | None = None
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    rate_limit_research_per_minute: int = Field(default=10, ge=1)
    rate_limit_followup_per_minute: int = Field(default=30, ge=1)
    rate_limit_upload_per_minute: int = Field(default=20, ge=1)
    max_concurrent_runs: int = Field(default=4, ge=1)
    max_request_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)

    @field_validator(
        "mesh_api_key",
        "tavily_api_key",
        "api_keys",
        "admin_api_key",
        mode="before",
    )
    @classmethod
    def _blank_secret_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "planner_model",
        "researcher_model",
        "drafter_model",
        "verifier_model",
        "synthesizer_model",
        "followup_model",
        "guardian_model",
        "chroma_persist_dir",
        mode="before",
    )
    @classmethod
    def _blank_optional_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("mesh_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/") or MESH_DEFAULT_BASE_URL

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # --- Derived helpers ----------------------------------------------------
    @property
    def mesh_configured(self) -> bool:
        return self.mesh_api_key is not None

    @property
    def tavily_configured(self) -> bool:
        return self.tavily_api_key is not None

    @staticmethod
    def _csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def allowed_model_set(self) -> frozenset[str]:
        """Model allowlist: explicit ALLOWED_MODELS, else exactly the configured models."""
        explicit = self._csv(self.allowed_models)
        if explicit:
            return frozenset(explicit)
        agents = ("planner", "approver", "researcher", "drafter", "verifier", "synthesizer", "followup", "guardian")
        return frozenset({self.mesh_default_model, self.mesh_embedding_model, *(self.model_for(a) for a in agents)})

    @property
    def blocked_domain_set(self) -> frozenset[str]:
        return frozenset(d.lower().removeprefix("www.") for d in self._csv(self.blocked_domains))

    @property
    def cors_origin_list(self) -> list[str]:
        return self._csv(self.cors_origins)

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(self._csv(self.api_keys.get_secret_value())) if self.api_keys else frozenset()

    def model_for(self, agent: str) -> str:
        """Resolve the Mesh model id for an agent (per-agent override or default)."""
        override = getattr(self, f"{agent}_model", None)
        return override or self.mesh_default_model

    def public_summary(self) -> dict[str, object]:
        """Non-secret configuration snapshot safe to expose via the API."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "mesh_base_url": self.mesh_base_url,
            "mesh_configured": self.mesh_configured,
            "default_model": self.mesh_default_model,
            "embedding_model": self.mesh_embedding_model,
            "tavily_configured": self.tavily_configured,
            "search_mode": self.search_mode,
            "vector_backend": self.vector_backend,
            "max_iterations": self.max_iterations,
            "coverage_threshold": self.coverage_threshold,
            "grounding_threshold": self.grounding_threshold,
            "min_source_authority": self.min_source_authority,
            "plan_approval_threshold": self.plan_approval_threshold,
            "max_plan_revisions": self.max_plan_revisions,
            "policy_version": self.policy_version,
            "auth_required": bool(self.api_key_set),
        }

    def governance_policy(self) -> dict[str, object]:
        """The active guardrail/governance policy, safe to publish (no secrets)."""
        return {
            "policy_version": self.policy_version,
            "guardrails": {
                "enabled": self.guardrails_enabled,
                "llm_screen": self.guardrail_llm_screen,
                "block_prompt_injection": self.block_prompt_injection,
                "redact_pii_in_queries": self.redact_pii_in_queries,
                "redact_pii_in_documents": self.redact_pii_in_documents,
                "redact_pii_in_output": self.redact_pii_in_output,
                "blocked_domains": sorted(self.blocked_domain_set),
                "fetch_max_bytes": self.fetch_max_bytes,
            },
            "evidence": {
                "coverage_threshold": self.coverage_threshold,
                "grounding_threshold": self.grounding_threshold,
                "min_source_authority": self.min_source_authority,
                "max_iterations": self.max_iterations,
            },
            "quality_gates": {
                "plan_approval_threshold": self.plan_approval_threshold,
                "max_plan_revisions": self.max_plan_revisions,
                "coverage_threshold": self.coverage_threshold,
                "grounding_threshold": self.grounding_threshold,
            },
            "budgets": {
                "max_llm_calls_per_run": self.max_llm_calls_per_run,
                "max_tokens_per_run": self.max_tokens_per_run,
                "budget_soft_fraction": self.budget_soft_fraction,
            },
            "models": {"allowed": sorted(self.allowed_model_set), "gateway": self.mesh_base_url},
            "data": {"retention_days": self.data_retention_days or None, "storage": "sqlite (append-only audit tables)"},
            "api": {
                "auth_required": bool(self.api_key_set),
                "rate_limits_per_minute": {
                    "research": self.rate_limit_research_per_minute,
                    "followup": self.rate_limit_followup_per_minute,
                    "upload": self.rate_limit_upload_per_minute,
                },
                "max_concurrent_runs": self.max_concurrent_runs,
                "max_request_bytes": self.max_request_bytes,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()
