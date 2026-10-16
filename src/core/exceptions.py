"""Typed domain exceptions.

Each exception carries a stable machine-readable `code` so the API layer can
map failures to HTTP responses and the UI can render actionable messages.
"""

from __future__ import annotations


class DeepDiveError(Exception):
    """Root of the DeepDive exception hierarchy."""

    code: str = "deepdive_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": self.details}


# --- Configuration ------------------------------------------------------------
class ConfigurationError(DeepDiveError):
    code = "configuration_error"


# --- Mesh API gateway -----------------------------------------------------------
class MeshGatewayError(DeepDiveError):
    """Any failure talking to the Mesh API gateway."""

    code = "mesh_gateway_error"


class MeshNotConfiguredError(MeshGatewayError, ConfigurationError):
    code = "mesh_not_configured"


class MeshTransientError(MeshGatewayError):
    """Retryable failure (rate limit, timeout, 5xx) that exhausted all retries."""

    code = "mesh_transient_error"


class MeshStructuredOutputError(MeshGatewayError):
    """The model output could not be parsed into the required schema after repairs."""

    code = "mesh_structured_output_error"


# --- Retrieval ------------------------------------------------------------------
class RetrievalError(DeepDiveError):
    code = "retrieval_error"


class SearchProviderError(RetrievalError):
    code = "search_provider_error"

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}", details={"provider": provider})
        self.provider = provider


# --- Documents & vector index ---------------------------------------------------
class DocumentIngestionError(DeepDiveError):
    code = "document_ingestion_error"


class UnsupportedDocumentTypeError(DocumentIngestionError):
    code = "unsupported_document_type"


class VectorStoreError(DeepDiveError):
    code = "vector_store_error"


# --- Orchestration & storage ----------------------------------------------------
class OrchestrationError(DeepDiveError):
    code = "orchestration_error"


class RunNotFoundError(DeepDiveError):
    code = "run_not_found"


class ImmutableRecordError(DeepDiveError):
    """Raised when code attempts to mutate an append-only audit record."""

    code = "immutable_record"


class ReportExportError(DeepDiveError):
    code = "report_export_error"
